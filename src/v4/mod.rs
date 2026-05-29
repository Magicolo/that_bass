pub mod error;
pub mod insert;
pub mod meta;
pub mod module;
pub mod query;
pub mod remove;
pub mod slice;
pub mod state;
pub mod table;
pub mod utility;
pub mod vector;

use crate::v4::{
    module::{Access, Dependency, IntoModule, Module, Resource},
    utility::Push,
};
use core::{
    iter::once,
    marker::PhantomData,
    num::NonZeroU32,
    sync::atomic::{AtomicU32, Ordering},
};
pub use error::Error;
pub use insert::insert;
pub use meta::Meta;
pub use query::query;
pub use remove::remove;
use std::collections::{HashMap, hash_map::Entry};
pub use table::{Index, Rows, Table};
pub use vector::Vector;

pub struct Store {
    identifier: u32,
    version: NonZeroU32,
    tables: Vec<Table>,
}

pub struct One;
pub struct More;
pub struct State<M, C = One> {
    identifier: u32,
    version: u32,
    module: M,
    _marker: PhantomData<C>,
}

impl Store {
    pub fn new() -> Self {
        static COUNT: AtomicU32 = AtomicU32::new(0);
        Self {
            identifier: COUNT.fetch_add(1, Ordering::Relaxed),
            version: NonZeroU32::MIN,
            tables: Vec::new(),
        }
    }

    pub fn state<M: IntoModule>(&mut self, module: M) -> Result<State<M::Module>, Error> {
        Ok(State {
            identifier: self.identifier,
            version: self.version.get(),
            module: module.into_module(self)?,
            _marker: PhantomData,
        })
    }

    pub fn get<'a, M: Module, C>(
        &'a mut self,
        state: &'a mut State<M, C>,
    ) -> Result<M::Item<'a>, Error> {
        self.ensure(state.identifier)?;
        if self.update(state)? {
            analyze(&mut HashMap::new(), state.module.declare(self)).map_or(Ok(()), Err)?;
        }
        todo!()
    }

    pub fn resolve<M: Module>(&mut self, state: &mut State<M>) -> Result<(), Error> {
        self.ensure(state.identifier)?;
        state.module.resolve(self)
    }

    fn find_table(&self, metas: &[Meta]) -> Option<u32> {
        self.tables
            .iter()
            .position(|table| {
                table
                    .columns()
                    .iter()
                    .map(|column| column.meta().identifier)
                    .eq(metas.iter().map(|meta| meta.identifier))
            })?
            .try_into()
            .ok()
    }

    fn find_or_insert_table(
        &mut self,
        metas: impl IntoIterator<Item = Meta>,
    ) -> Result<u32, Error> {
        let metas = sort(metas)?;
        Ok(match self.find_table(&metas) {
            Some(index) => index,
            None => {
                let index = self
                    .tables
                    .len()
                    .try_into()
                    .map_err(Error::TablesOverflow)?;
                self.tables.push(Table::new(index, metas)?);
                self.increment_version()?;
                index
            }
        })
    }

    fn increment_version(&mut self) -> Result<(), Error> {
        self.version = self.version.checked_add(1).ok_or(Error::VersionOverflow)?;
        Ok(())
    }

    fn ensure(&self, identifier: u32) -> Result<(), Error> {
        if self.identifier == identifier {
            Ok(())
        } else {
            Err(Error::StoreMismatch)
        }
    }

    fn update<M: Module, C>(&mut self, state: &mut State<M, C>) -> Result<bool, Error> {
        let mut did = false;
        while (state.version < self.version.get()) | state.module.update(self)? {
            state.version = self.version.get();
            did = true;
        }
        Ok(did)
    }
}

impl<M, C> State<M, C> {
    pub const fn as_mut(&mut self) -> State<&mut M, C> {
        State {
            identifier: self.identifier,
            version: self.version,
            module: &mut self.module,
            _marker: PhantomData,
        }
    }
}

impl<M> State<M, One> {
    pub fn and<N>(self, state: State<N>) -> State<(M, (N, ())), More> {
        State {
            identifier: self.identifier,
            version: 0,
            module: (self.module, (state.module, ())),
            _marker: PhantomData,
        }
    }
}

impl<M> State<M, More> {
    pub fn and<N>(self, state: State<N>) -> State<M::Out, More>
    where
        M: Push<N>,
    {
        State {
            identifier: self.identifier,
            version: 0,
            module: self.module.push(state.module),
            _marker: PhantomData,
        }
    }
}

fn sort(metas: impl IntoIterator<Item = Meta>) -> Result<Vec<Meta>, Error> {
    let mut metas = metas.into_iter().collect::<Vec<_>>();
    metas.sort_unstable_by_key(|meta| meta.identifier);
    for [left, right] in metas.array_windows::<2>() {
        if left.identifier == right.identifier {
            return Err(Error::DuplicateMeta);
        }
    }
    Ok(metas)
}

fn analyze(
    map: &mut HashMap<Resource, Access>,
    dependencies: impl IntoIterator<Item = Dependency>,
) -> Option<Error> {
    let errors = dependencies
        .into_iter()
        .flat_map(|Dependency { access, resource }| {
            resource
                .ancestors()
                .map(|resource| (resource, Access::Read))
                .chain(once((resource, access)))
        })
        .filter_map(|(resource, access)| conflict(map, resource, access));
    Error::all(errors)
}

fn conflict(
    map: &mut HashMap<Resource, Access>,
    resource: Resource,
    access: Access,
) -> Option<Error> {
    let entry = map.entry(resource);
    match (entry, access) {
        (Entry::Occupied(entry), Access::Read) => match entry.get() {
            Access::Read => None,
            Access::Write => Some(Error::ReadWriteConflict(resource, *entry.key())),
        },
        (Entry::Occupied(entry), Access::Write) => match entry.get() {
            Access::Read => Some(Error::ReadWriteConflict(*entry.key(), resource)),
            Access::Write => Some(Error::WriteWriteConflict(*entry.key(), resource)),
        },
        (Entry::Vacant(entry), access) => {
            entry.insert(access);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v4::utility::{IntoFlat, IteratorExtension};

    #[test]
    fn access() -> Result<(), Error> {
        let mut store = Store::new();
        let mut query1 = store.state(query().read::<char>().try_write::<String>())?;
        let mut query2 = store.state(query().read::<char>().not::<String>())?;
        let mut insert = store.state(insert().key().column::<char>())?;
        let mut remove = store.state(remove())?;
        {
            let item0 = store.get(&mut query1)?;
            let item1 = store.get(&mut query2)?;
            let mut query3 = query1.as_mut().and(query2.as_mut());
            let (item0, item1) = store.get(&mut query3)?;
        }
        {
            let mut state = query1.as_mut().and(insert.as_mut()).and(remove.as_mut());
            let (mut item0, item1, item2) = store.get(&mut state)?.into_flat();
            for (a, b) in item0.columns().into_flat() {}
        }

        // let mut state = store.state(
        //     State::build()
        //         .push((query().read::<char>().write::<String>(),))
        //         .push((query().read::<isize>(), Remove::build()))
        //         .push((query().read::<[u32; 100]>(),))
        //         .push((Insert::build().key().column::<u8>(),))
        //         .push((query().read::<usize>(),))
        //         .push((query().read::<char>(),))
        //         .push((query().read::<i32>(),)),
        // )?;
        // let guard = state.guard();
        // let guard = guard.next()?;
        // let guard = guard.next()?;
        // let mut guard = guard.next()?;
        // let (mut insert,) = guard.get()?;
        // insert.one(((), 1u8));
        // let guard = guard.next()?;
        // let guard = guard.next()?;
        // let guard = guard.next()?;
        // let _guard = guard.next()?;
        Ok(())
    }

    // #[test]
    // fn read_write_conflict() -> Result<(), Error> {
    //     let mut store = Store::new();
    //     let mut state = store.state(
    //         State::build()
    //             .push((Insert::build().column::<u8>(),))
    //             .push((query().read::<u8>().write::<u8>(),)),
    //     )?;
    //     let guard = state.guard();
    //     let mut guard = guard.next()?;
    //     assert!(matches!(guard.get(), Err(Error::ReadWriteConflict(_, _))));
    //     Ok(())
    // }
}
