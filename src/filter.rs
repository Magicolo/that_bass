use crate::{table::Table, Datum};
use std::marker::PhantomData;

pub trait Filter: Sized {
    fn filter(&self, table: &Table) -> bool;
    fn not(self) -> Not<Self> {
        Not(self)
    }
    fn and<F: Filter>(self, filter: F) -> (Self, F) {
        (self, filter)
    }
    fn and_has<D: Datum>(self) -> (Self, Has<D>) {
        self.and(Has(PhantomData))
    }
}

#[derive(Debug, Default)]
pub struct Not<F>(F);
#[derive(Debug)]
pub struct Has<D: Datum>(PhantomData<D>);

impl<D: Datum> Has<D> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<D: Datum> Default for Has<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Filter> Filter for Not<F> {
    fn filter(&self, table: &Table) -> bool {
        !self.0.filter(table)
    }
}

impl<D: Datum> Filter for Has<D> {
    fn filter(&self, table: &Table) -> bool {
        table.has::<D>()
    }
}

impl Filter for bool {
    fn filter(&self, _: &Table) -> bool {
        *self
    }
}

impl<F: Fn(&Table) -> bool> Filter for F {
    fn filter(&self, table: &Table) -> bool {
        self(table)
    }
}

impl Filter for () {
    fn filter(&self, _: &Table) -> bool {
        true
    }
}

impl<F1: Filter> Filter for (F1,) {
    fn filter(&self, table: &Table) -> bool {
        self.0.filter(table)
    }
}

impl<F1: Filter, F2: Filter> Filter for (F1, F2) {
    fn filter(&self, table: &Table) -> bool {
        self.0.filter(table) && self.1.filter(table)
    }
}
