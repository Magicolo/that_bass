pub mod error;
pub mod insert;
pub mod meta;
pub mod module;
pub mod query;
pub mod remove;
pub mod state;
pub mod table;
pub mod utility;
pub mod vector;

use crate::v4::module::Module;
use core::{
    num::NonZeroU32,
    sync::atomic::{AtomicU32, Ordering},
};
pub use error::Error;
pub use meta::Meta;
pub use query::query;
pub use table::{Index, Rows, Table};
pub use vector::Vector;

pub struct Store {
    identifier: u32,
    version: NonZeroU32,
    tables: Vec<Table>,
}

pub struct State<M: Module> {
    identifier: u32,
    version: u32,
    module: M,
    state: M::State,
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

mod staty {
    use super::*;
    use crate::v4::module::{Access, Dependency, Resource};
    use core::iter::once;
    use std::collections::{HashMap, hash_map::Entry};

    impl Store {
        pub fn query<Q: query::Query, F: query::Filter>(
            &mut self,
            query: query::Build<Q, F>,
        ) -> Result<State<query::Build<Q, F>>, Error> {
            self.staty(query)
        }

        pub fn get<'a, M: Module>(
            &'a mut self,
            state: &'a mut State<M>,
        ) -> Result<M::Item<'a>, Error> {
            self.ensure(state.identifier)?;
            if self.update(state)? {
                analyze(
                    &mut HashMap::new(),
                    state.module.declare(&state.state, self),
                )
                .map_or(Ok(()), Err)?;
            }
            todo!()
        }

        pub fn resolve<M: Module>(&mut self, state: &mut State<M>) -> Result<(), Error> {
            self.ensure(state.identifier)?;
            state.module.resolve(&mut state.state, self)
        }

        fn ensure(&self, identifier: u32) -> Result<(), Error> {
            if self.identifier == identifier {
                Ok(())
            } else {
                Err(Error::StoreMismatch)
            }
        }

        fn update<M: Module>(&mut self, state: &mut State<M>) -> Result<bool, Error> {
            let mut did = false;
            while (state.version < self.version.get())
                | state.module.update(&mut state.state, self)?
            {
                state.version = self.version.get();
                did = true;
            }
            Ok(did)
        }

        fn staty<M: Module>(&mut self, module: M) -> Result<State<M>, Error> {
            Ok(State {
                identifier: self.identifier,
                version: self.version.get(),
                state: module.initialize(self)?,
                module,
            })
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v4::{insert::Insert, remove::Remove, state::State};
    #[test]
    fn access() -> Result<(), Error> {
        let mut store = Store::new();
        let mut state = store.state(
            State::build()
                .push((query().read::<char>().write::<String>(),))
                .push((query().read::<isize>(), Remove::build()))
                .push((query().read::<[u32; 100]>(),))
                .push((Insert::build().key().column::<u8>(),))
                .push((query().read::<usize>(),))
                .push((query().read::<char>(),))
                .push((query().read::<i32>(),)),
        )?;
        let guard = state.guard();
        let guard = guard.next()?;
        let guard = guard.next()?;
        let mut guard = guard.next()?;
        let (mut insert,) = guard.get()?;
        insert.one(((), 1u8));
        let guard = guard.next()?;
        let guard = guard.next()?;
        let guard = guard.next()?;
        let _guard = guard.next()?;
        Ok(())
    }

    #[test]
    fn read_write_conflict() -> Result<(), Error> {
        let mut store = Store::new();
        let mut state = store.state(
            State::build()
                .push((Insert::build().column::<u8>(),))
                .push((query().read::<u8>().write::<u8>(),)),
        )?;
        let guard = state.guard();
        let mut guard = guard.next()?;
        assert!(matches!(guard.get(), Err(Error::ReadWriteConflict(_, _))));
        Ok(())
    }
}
