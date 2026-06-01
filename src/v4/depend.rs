use crate::v4::Error;
use core::{
    any::TypeId,
    iter::{empty, from_fn, once},
};
use std::collections::{HashMap, hash_map::Entry};

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
    Table,
    Column(TypeId),
}

pub unsafe trait Depend {
    fn depend(&self) -> impl Iterator<Item = Dependency>;
    fn analyze(&self) -> Result<(), Error> {
        analyze(&mut HashMap::new(), self.depend()).map_or(Ok(()), Err)
    }
}

unsafe impl<D: Depend> Depend for &D {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        D::depend(self)
    }
}

unsafe impl<D: Depend> Depend for &mut D {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        D::depend(self)
    }
}

unsafe impl Depend for () {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        empty()
    }
}

unsafe impl<D0: Depend, D1: Depend> Depend for (D0, D1) {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        self.0.depend().chain(self.1.depend())
    }
}

impl Resource {
    pub const fn parent(self) -> Option<Self> {
        match self {
            Self::Store => None,
            Self::Tables => Some(Self::Store),
            Self::Table => Some(Self::Tables),
            Self::Column(_) => Some(Self::Table),
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
