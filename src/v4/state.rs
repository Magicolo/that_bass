use crate::v4::{
    Error, Store,
    module::{self, Access, Dependency, Resource},
    utility::Push,
};
use core::iter::empty;
use ref_cast::RefCast;
use std::collections::{HashMap, hash_map::Entry};

#[derive(RefCast)]
#[repr(transparent)]
pub struct Module<M = ()>(M);

pub struct State<'a, M: module::Module = ()> {
    store: &'a mut Store,
    dependencies: HashMap<Resource, Access>,
    module: M,
    state: M::State,
}

pub struct Guard<'a, M: module::Module = ()> {
    module: &'a M,
    dependencies: &'a mut HashMap<Resource, Access>,
    state: &'a mut M::State,
    store: &'a mut Store,
}

impl State<'_> {
    pub const fn build() -> Module {
        Module::new()
    }
}

impl<M: module::Module> State<'_, M> {
    pub fn guard(&mut self) -> Guard<'_, M> {
        Guard {
            module: &self.module,
            dependencies: &mut self.dependencies,
            state: &mut self.state,
            store: &mut *self.store,
        }
    }
}

impl Store {
    pub fn state<M: module::Module>(&mut self, module: Module<M>) -> Result<State<'_, M>, Error> {
        Ok(State {
            state: module.0.initialize(self)?,
            dependencies: HashMap::new(),
            store: self,
            module: module.0,
        })
    }
}

impl<'a, H: module::Module, T: module::Module> Guard<'a, (H, T)> {
    pub fn get(&mut self) -> Result<H::Item<'_>, Error> {
        if self.update()? {
            self.analyze()?;
        }
        Ok(self.module.0.get(&mut self.state.0, self.store))
    }

    pub fn next(self) -> Result<Guard<'a, T>, Error> {
        let Self {
            module,
            dependencies,
            state,
            store,
        } = self;
        module.0.resolve(&mut state.0, store)?;
        Ok(Guard {
            module: &module.1,
            state: &mut state.1,
            dependencies,
            store,
        })
    }

    pub fn with<F: FnOnce(H::Item<'_>)>(mut self, with: F) -> Result<Guard<'a, T>, Error> {
        with(self.get()?);
        self.next()
    }

    fn analyze(&mut self) -> Result<(), Error> {
        self.dependencies.clear();

        let dependencies = self.module.0.declare(&self.state.0, self.store);
        let errors = dependencies.filter_map(|dependency| {
            let entry = self.dependencies.entry(dependency.resource());
            match (entry, dependency.access()) {
                (Entry::Occupied(entry), Access::Read) => match entry.get() {
                    Access::Read => None,
                    Access::Write => Some(Error::ReadWriteConflict(
                        dependency.resource(),
                        *entry.key(),
                    )),
                },
                (Entry::Occupied(entry), Access::Write) => match dependency.access() {
                    Access::Read => Some(Error::ReadWriteConflict(
                        *entry.key(),
                        dependency.resource(),
                    )),
                    Access::Write => Some(Error::WriteWriteConflict(
                        *entry.key(),
                        dependency.resource(),
                    )),
                },
                (Entry::Vacant(entry), access) => {
                    entry.insert(access);
                    None
                }
            }
        });
        Error::all(errors).map_or(Ok(()), Err)
    }

    fn update(&mut self) -> Result<bool, Error> {
        let mut did = false;
        while self.module.0.update(&mut self.state.0, self.store)? {
            did = true;
        }
        Ok(did)
    }
}

impl Module {
    pub(crate) const fn new() -> Self {
        Self(())
    }
}

impl<M: module::Module> Module<M> {
    pub fn push<N: module::Module>(self, module: N) -> Module<M::Out>
    where
        M: Push<N, Out: module::Module>,
    {
        Module(self.0.push(module))
    }
}

impl<H, T> Module<(H, T)> {
    fn split_own(self) -> (Module<H>, Module<T>) {
        let Self((head, tail)) = self;
        (Module(head), Module(tail))
    }

    fn split_ref(&self) -> (&Module<H>, &Module<T>) {
        let Self((head, tail)) = self;
        (Module::ref_cast(head), Module::ref_cast(tail))
    }

    #[allow(dead_code)]
    fn split_mut(&mut self) -> (&mut Module<H>, &mut Module<T>) {
        let Self((head, tail)) = self;
        (Module::ref_cast_mut(head), Module::ref_cast_mut(tail))
    }
}

impl module::Module for Module<()> {
    type Item<'a>
        = <() as module::Module>::Item<'a>
    where
        Self: 'a;
    type State = <() as module::Module>::State;

    fn declare(&self, _: &Self::State, _: &Store) -> impl Iterator<Item = Dependency> {
        empty()
    }

    fn initialize(&self, store: &mut Store) -> Result<Self::State, Error> {
        self.0.initialize(store)
    }

    fn update(&self, state: &mut Self::State, store: &mut Store) -> Result<bool, Error> {
        self.0.update(state, store)
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, store: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        self.0.get(state, store)
    }

    fn resolve(&self, state: &mut Self::State, store: &mut Store) -> Result<(), Error> {
        self.0.resolve(state, store)
    }
}

impl<H: module::Module, T: module::Module> module::Module for Module<(H, T)>
where
    Module<T>: module::Module,
{
    type Item<'a>
        = (H::Item<'a>, <Module<T> as module::Module>::Item<'a>)
    where
        Self: 'a;
    type State = (H::State, <Module<T> as module::Module>::State);

    fn declare(&self, state: &Self::State, store: &Store) -> impl Iterator<Item = Dependency> {
        self.0.0.declare(&state.0, store)
    }

    fn initialize(&self, store: &mut Store) -> Result<Self::State, Error> {
        let (head, tail) = self.split_ref();
        Ok((head.0.initialize(store)?, tail.initialize(store)?))
    }

    fn update(&self, state: &mut Self::State, store: &mut Store) -> Result<bool, Error> {
        let (head, tail) = self.split_ref();
        Ok(head.0.update(&mut state.0, store)? | tail.update(&mut state.1, store)?)
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, store: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        let (head, tail) = self.split_ref();
        (
            head.0.get(&mut state.0, store),
            tail.get(&mut state.1, store),
        )
    }

    fn resolve(&self, state: &mut Self::State, store: &mut Store) -> Result<(), Error> {
        let (head, tail) = self.split_ref();
        head.0.resolve(&mut state.0, store)?;
        tail.resolve(&mut state.1, store)
    }
}
