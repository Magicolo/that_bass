use crate::v4::{Error, Store, depend::Depend};
use core::iter::{empty, from_fn};

pub trait Module: Depend {
    fn update(&mut self) -> Result<bool, Error>;
    fn resolve(&mut self) -> Result<(), Error>;
}

impl<M: Module> Module for &mut M {
    fn update(&mut self) -> Result<bool, Error> {
        M::update(self)
    }

    fn resolve(&mut self) -> Result<(), Error> {
        M::resolve(self)
    }
}

impl Module for () {
    fn update(&mut self) -> Result<bool, Error> {
        Ok(false)
    }

    fn resolve(&mut self) -> Result<(), Error> {
        Ok(())
    }
}

impl<M0: Module, M1: Module> Module for (M0, M1) {
    fn update(&mut self) -> Result<bool, Error> {
        Ok(self.0.update()? | self.1.update()?)
    }

    fn resolve(&mut self) -> Result<(), Error> {
        self.0.resolve()?;
        self.1.resolve()?;
        Ok(())
    }
}
