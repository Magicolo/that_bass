use crate::{core::tuples, create, table::Table, template::Template, Database};
use std::{any::TypeId, marker::PhantomData};

pub trait Filter: Sized {
    fn filter(&self, table: &Table, database: &Database) -> bool;

    fn not(self) -> Not<Self> {
        Not(self)
    }
    fn and<F: Filter>(self, filter: F) -> (Self, F) {
        (self, filter)
    }
    fn or<F: Filter>(self, filter: F) -> Any<(Self, F)> {
        Any((self, filter))
    }
}

pub type None<F> = Not<Any<F>>;
pub struct Not<F>(F);
pub struct Any<F>(F);
pub struct Has<T>(PhantomData<T>);
pub struct HasWith(Box<[TypeId]>);
pub struct Is<T>(PhantomData<T>);
pub struct IsWith(Box<[TypeId]>);

#[inline]
pub const fn has<T: Template>() -> Has<T> {
    Has(PhantomData)
}

pub fn has_with<I: IntoIterator<Item = TypeId>>(types: I) -> HasWith {
    let mut types: Vec<_> = types.into_iter().collect();
    types.sort_unstable();
    types.dedup();
    HasWith(types.into_boxed_slice())
}

#[inline]
pub const fn is<T: Template>() -> Is<T> {
    Is(PhantomData)
}

pub fn is_with<I: IntoIterator<Item = TypeId>>(types: I) -> IsWith {
    let mut types: Vec<_> = types.into_iter().collect();
    types.sort_unstable();
    types.dedup();
    IsWith(types.into_boxed_slice())
}

impl Filter for bool {
    fn filter(&self, _: &Table, _: &Database) -> bool {
        *self
    }
}

impl<T: Template> Filter for Has<T> {
    fn filter(&self, table: &Table, database: &Database) -> bool {
        create::has::<T>(table, database)
    }
}

impl Filter for HasWith {
    fn filter(&self, table: &Table, _: &Database) -> bool {
        table.has_all(self.0.iter().copied())
    }
}

impl<T: Template> Filter for Is<T> {
    fn filter(&self, table: &Table, database: &Database) -> bool {
        create::is::<T>(table, database)
    }
}

impl Filter for IsWith {
    fn filter(&self, table: &Table, _: &Database) -> bool {
        table.is_all(self.0.iter().copied())
    }
}

impl<F: Filter> Filter for Not<F> {
    fn filter(&self, table: &Table, database: &Database) -> bool {
        !self.0.filter(table, database)
    }
}

impl<F: Fn(&Table, &Database) -> bool> Filter for F {
    fn filter(&self, table: &Table, database: &Database) -> bool {
        self(table, database)
    }
}

macro_rules! tuple {
    ($n:ident, $c:expr $(, $p:ident, $t:ident, $i:tt)*) => {
        impl<$($t: Filter,)*> Filter for ($($t,)*) {
            fn filter(&self, _table: &Table, _database: &Database) -> bool {
                true $(&& self.$i.filter(_table, _database))*
            }
        }

        impl<$($t: Filter,)*> Filter for Any<($($t,)*)> {
            fn filter(&self,_table: &Table, _database: &Database) -> bool {
                false $(|| self.0.$i.filter(_table, _database))*
            }
        }
    };
}
tuples!(tuple);
