use crate::v4::{Meta, Table};
use core::{any::TypeId, marker::PhantomData};

pub trait Filter {
    fn filter(&self, table: &Table) -> bool;
}

#[derive(Debug)]
pub struct Has<T: ?Sized>(pub(crate) PhantomData<T>);
#[derive(Debug, Clone, Copy)]
pub struct HasWith(pub(crate) Meta);
#[derive(Debug, Clone, Copy)]
pub struct Not<F: ?Sized>(pub(crate) F);

impl<F: Filter> Filter for &F {
    fn filter(&self, table: &Table) -> bool {
        F::filter(self, table)
    }
}

impl<F: Filter> Filter for &mut F {
    fn filter(&self, table: &Table) -> bool {
        F::filter(self, table)
    }
}

impl Filter for () {
    fn filter(&self, _: &Table) -> bool {
        true
    }
}

impl<F0: Filter, F1: Filter> Filter for (F0, F1) {
    fn filter(&self, table: &Table) -> bool {
        self.0.filter(table) && self.1.filter(table)
    }
}

impl<T: ?Sized> Clone for Has<T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl<T: ?Sized> Copy for Has<T> {}

impl<T: 'static> Filter for Has<T> {
    fn filter(&self, table: &Table) -> bool {
        table.column(TypeId::of::<T>()).is_some()
    }
}

impl Filter for HasWith {
    fn filter(&self, table: &Table) -> bool {
        table.column(self.0.identifier()).is_some()
    }
}

impl<F: Filter> Filter for Not<F> {
    fn filter(&self, table: &Table) -> bool {
        !self.0.filter(table)
    }
}
