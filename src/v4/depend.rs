use crate::v4::{Error, Meta, utility::Push};
use core::iter::{empty, once};
use itertools::Itertools;
use std::{rc::Rc, sync::Arc};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Dependency {
    meta: Meta,
    access: Access,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum Access {
    Write,
    Read,
}

pub struct Analysis<D>(D);

impl Dependency {
    pub const fn read(meta: Meta) -> Self {
        Self {
            meta,
            access: Access::Read,
        }
    }

    pub const fn write(meta: Meta) -> Self {
        Self {
            meta,
            access: Access::Write,
        }
    }

    pub const fn meta(self) -> Meta {
        self.meta
    }

    pub const fn access(self) -> Access {
        self.access
    }
}

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

fn analyze(dependencies: impl IntoIterator<Item = Dependency>) -> Option<Error> {
    let mut old = None::<Dependency>;
    let errors = dependencies.into_iter().filter_map(|new| {
        if new.meta.is_key() && new.access == Access::Write {
            Some(Error::KeyWriteConflict)
        } else if new.meta.is_table() && new.access == Access::Write {
            Some(Error::TableWriteConflict)
        } else {
            match old {
                Some(old) if old.meta == new.meta => match (old.access, new.access) {
                    (Access::Read, Access::Write) | (Access::Write, Access::Read) => {
                        Some(Error::ReadWriteConflict(new.meta, old.meta))
                    }
                    (Access::Write, Access::Write) => {
                        Some(Error::WriteWriteConflict(new.meta, old.meta))
                    }
                    (Access::Read, Access::Read) => None,
                },
                Some(_) | None => {
                    old = Some(new);
                    None
                }
            }
        }
    });
    Error::all(errors)
}
