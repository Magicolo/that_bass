use crate::v4::{Error, Store};
use core::iter::{empty, from_fn};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Dependency {
    pub access: Access,
    pub resource: Resource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum Access {
    Read,
    Write,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum Resource {
    Store,
    Tables,
    Table { index: u32 },
    Columns { table: u32 },
    Column { table: u32, index: u32 },
}

pub trait Module {
    type Item<'a>
    where
        Self: 'a;
    type State;

    fn declare(&self, state: &Self::State, store: &Store) -> impl Iterator<Item = Dependency>;
    fn initialize(&self, store: &mut Store) -> Result<Self::State, Error>;
    fn update(&self, state: &mut Self::State, store: &mut Store) -> Result<bool, Error>;
    fn get<'a>(&'a self, state: &'a mut Self::State, store: &'a Store) -> Self::Item<'a>
    where
        Self: 'a;
    fn resolve(&self, _: &mut Self::State, _: &mut Store) -> Result<(), Error> {
        Ok(())
    }
}

impl<M: Module> Module for &mut M {
    type Item<'a>
        = M::Item<'a>
    where
        Self: 'a;
    type State = M::State;

    fn declare(&self, state: &Self::State, store: &Store) -> impl Iterator<Item = Dependency> {
        M::declare(self, state, store)
    }

    fn initialize(&self, store: &mut Store) -> Result<Self::State, Error> {
        M::initialize(self, store)
    }

    fn update(&self, state: &mut Self::State, store: &mut Store) -> Result<bool, Error> {
        M::update(self, state, store)
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, store: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        M::get(self, state, store)
    }

    fn resolve(&self, state: &mut Self::State, store: &mut Store) -> Result<(), Error> {
        M::resolve(self, state, store)
    }
}

impl<M: Module> Module for &M {
    type Item<'a>
        = M::Item<'a>
    where
        Self: 'a;
    type State = M::State;

    fn declare(&self, state: &Self::State, store: &Store) -> impl Iterator<Item = Dependency> {
        M::declare(self, state, store)
    }

    fn initialize(&self, store: &mut Store) -> Result<Self::State, Error> {
        M::initialize(self, store)
    }

    fn update(&self, state: &mut Self::State, store: &mut Store) -> Result<bool, Error> {
        M::update(self, state, store)
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, store: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        M::get(self, state, store)
    }

    fn resolve(&self, state: &mut Self::State, store: &mut Store) -> Result<(), Error> {
        M::resolve(self, state, store)
    }
}

impl Module for () {
    type Item<'a>
        = ()
    where
        Self: 'a;
    type State = ();

    fn declare(&self, _: &Self::State, _: &Store) -> impl Iterator<Item = Dependency> {
        empty()
    }

    fn initialize(&self, _: &mut Store) -> Result<Self::State, Error> {
        Ok(())
    }

    fn update(&self, _: &mut Self::State, _: &mut Store) -> Result<bool, Error> {
        Ok(false)
    }

    fn get<'a>(&'a self, _: &'a mut Self::State, _: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
    }
}

impl<M0: Module, M1: Module> Module for (M0, M1) {
    type Item<'a>
        = (M0::Item<'a>, M1::Item<'a>)
    where
        Self: 'a;
    type State = (M0::State, M1::State);

    fn declare(&self, state: &Self::State, store: &Store) -> impl Iterator<Item = Dependency> {
        self.0
            .declare(&state.0, store)
            .chain(self.1.declare(&state.1, store))
    }

    fn initialize(&self, store: &mut Store) -> Result<Self::State, Error> {
        Ok((self.0.initialize(store)?, self.1.initialize(store)?))
    }

    fn update(&self, state: &mut Self::State, store: &mut Store) -> Result<bool, Error> {
        Ok(self.0.update(&mut state.0, store)? | self.1.update(&mut state.1, store)?)
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, store: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        (
            self.0.get(&mut state.0, store),
            self.1.get(&mut state.1, store),
        )
    }

    fn resolve(&self, state: &mut Self::State, store: &mut Store) -> Result<(), Error> {
        self.0.resolve(&mut state.0, store)?;
        self.1.resolve(&mut state.1, store)?;
        Ok(())
    }
}

impl Resource {
    pub const fn parent(self) -> Option<Self> {
        match self {
            Self::Store => None,
            Self::Tables => Some(Self::Store),
            Self::Table { .. } => Some(Self::Tables),
            Self::Columns { table } => Some(Self::Table { index: table }),
            Self::Column { table, .. } => Some(Self::Columns { table }),
        }
    }

    pub fn ancestors(self) -> impl Iterator<Item = Self> {
        let mut child = self;
        from_fn(move || {
            child = child.parent()?;
            Some(child)
        })
    }
}
