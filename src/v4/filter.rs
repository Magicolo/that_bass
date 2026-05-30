use crate::v4::{Meta, Table};
use core::{any::TypeId, marker::PhantomData};

pub trait Filter {
    fn filter(&self, table: &Table) -> bool;
}

pub struct Has<T: ?Sized>(pub(crate) PhantomData<T>);
pub struct HasWith(pub(crate) Meta);
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
