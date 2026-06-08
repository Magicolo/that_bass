use crate::v4::{Error, utility::Push};
use core::{
    any::TypeId,
    iter::{empty, from_fn, once},
};
use itertools::Itertools;
use std::{rc::Rc, sync::Arc};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Dependency {
    pub resource: Resource,
    pub access: Access,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum Access {
    Write,
    Read,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum Resource {
    Column(TypeId),
    Table,
    Tables,
    Store,
}

pub struct Analysis<D>(D);

impl Analysis<()> {
    pub const fn new() -> Self {
        Analysis(())
    }
}

impl<D: Depend> Analysis<D> {
    pub fn add<E: Depend>(self, depend: E) -> Analysis<D::Out>
    where
        D: Push<E>,
    {
        Analysis(self.0.push(depend))
    }

    pub fn analyze(&self) -> Result<(), Error> {
        self.0.analyze()
    }
}

pub unsafe trait Depend {
    fn depend(&self) -> impl Iterator<Item = Dependency>;
    fn analyze(&self) -> Result<(), Error> {
        analyze(self.depend()).map_or(Ok(()), Err)
    }
}

unsafe impl Depend for Dependency {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(*self)
    }

    fn analyze(&self) -> Result<(), Error> {
        Ok(())
    }
}

unsafe impl<D: Depend + ?Sized> Depend for &D {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        D::depend(self)
    }

    fn analyze(&self) -> Result<(), Error> {
        D::analyze(self)
    }
}

unsafe impl<D: Depend + ?Sized> Depend for &mut D {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        D::depend(self)
    }

    fn analyze(&self) -> Result<(), Error> {
        D::analyze(self)
    }
}

unsafe impl Depend for () {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        empty()
    }

    fn analyze(&self) -> Result<(), Error> {
        Ok(())
    }
}

unsafe impl<D0: Depend, D1: Depend> Depend for (D0, D1) {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        self.0.depend().merge(self.1.depend())
    }
}

unsafe impl<D: Depend + ?Sized> Depend for Box<D> {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        D::depend(self)
    }

    fn analyze(&self) -> Result<(), Error> {
        D::analyze(self)
    }
}

unsafe impl<D: Depend + ?Sized> Depend for Rc<D> {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        D::depend(self)
    }

    fn analyze(&self) -> Result<(), Error> {
        D::analyze(self)
    }
}

unsafe impl<D: Depend + ?Sized> Depend for Arc<D> {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        D::depend(self)
    }

    fn analyze(&self) -> Result<(), Error> {
        D::analyze(self)
    }
}

unsafe impl<D: Depend + ?Sized> Depend for triomphe::Arc<D> {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        D::depend(self)
    }

    fn analyze(&self) -> Result<(), Error> {
        D::analyze(self)
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

fn analyze(dependencies: impl IntoIterator<Item = Dependency>) -> Option<Error> {
    let mut last = None::<Dependency>;
    let errors = dependencies
        .into_iter()
        .flat_map(|dependency| {
            once(dependency).chain(dependency.resource.ancestors().map(|resource| Dependency {
                resource,
                access: Access::Read,
            }))
        })
        .filter_map(|dependency| match last {
            Some(last) if last.resource == dependency.resource => {
                match (last.access, dependency.access) {
                    (Access::Read, Access::Write) | (Access::Write, Access::Read) => {
                        Some(Error::ReadWriteConflict(dependency.resource, last.resource))
                    }
                    (Access::Write, Access::Write) => Some(Error::WriteWriteConflict(
                        dependency.resource,
                        last.resource,
                    )),
                    (Access::Read, Access::Read) => None,
                }
            }
            Some(_) | None => {
                last = Some(dependency);
                None
            }
        });
    Error::all(errors)
}
