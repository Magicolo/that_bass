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
    Store { identifier: u32 },
    Tables { store: u32 },
    Table { store: u32, index: u32 },
    Columns { store: u32, table: u32 },
    Column { store: u32, table: u32, index: u32 },
}

pub trait IntoModule {
    type Module: Module;
    fn into_module(self, store: &mut Store) -> Result<Self::Module, Error>;
}

pub trait Module {
    type Item<'a>
    where
        Self: 'a;

    fn declare(&self, store: &Store) -> impl Iterator<Item = Dependency>;
    fn update(&mut self, store: &mut Store) -> Result<bool, Error>;
    fn resolve(&mut self, store: &mut Store) -> Result<(), Error>;
    fn get<'a>(&'a mut self, store: &'a Store) -> Self::Item<'a>
    where
        Self: 'a;
}

impl<M: Module> IntoModule for M {
    type Module = Self;

    fn into_module(self, _: &mut Store) -> Result<Self::Module, Error> {
        Ok(self)
    }
}

impl<M: Module> Module for &mut M {
    type Item<'a>
        = M::Item<'a>
    where
        Self: 'a;

    fn declare(&self, store: &Store) -> impl Iterator<Item = Dependency> {
        M::declare(self, store)
    }

    fn update(&mut self, store: &mut Store) -> Result<bool, Error> {
        M::update(self, store)
    }

    fn get<'a>(&'a mut self, store: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        M::get(self, store)
    }

    fn resolve(&mut self, store: &mut Store) -> Result<(), Error> {
        M::resolve(self, store)
    }
}

impl Module for () {
    type Item<'a>
        = ()
    where
        Self: 'a;

    fn declare(&self, _: &Store) -> impl Iterator<Item = Dependency> {
        empty()
    }

    fn update(&mut self, _: &mut Store) -> Result<bool, Error> {
        Ok(false)
    }

    fn resolve(&mut self, _: &mut Store) -> Result<(), Error> {
        Ok(())
    }

    fn get<'a>(&'a mut self, _: &'a Store) -> Self::Item<'a>
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

    fn declare(&self, store: &Store) -> impl Iterator<Item = Dependency> {
        self.0.declare(store).chain(self.1.declare(store))
    }

    fn update(&mut self, store: &mut Store) -> Result<bool, Error> {
        Ok(self.0.update(store)? | self.1.update(store)?)
    }

    fn resolve(&mut self, store: &mut Store) -> Result<(), Error> {
        self.0.resolve(store)?;
        self.1.resolve(store)?;
        Ok(())
    }

    fn get<'a>(&'a mut self, store: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        (self.0.get(store), self.1.get(store))
    }
}

impl Resource {
    pub const fn parent(self) -> Option<Self> {
        match self {
            Self::Store { .. } => None,
            Self::Tables { store, .. } => Some(Self::Store { identifier: store }),
            Self::Table { store, .. } => Some(Self::Tables { store }),
            Self::Columns { store, table, .. } => Some(Self::Table {
                store,
                index: table,
            }),
            Self::Column { store, table, .. } => Some(Self::Columns { store, table }),
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
