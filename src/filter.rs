use crate::{core::tuples, create, table::Table, template::Template, Database};
use std::marker::PhantomData;

pub trait Filter: Sized {
    fn filter(table: &Table, database: &Database) -> bool;
}

pub type True = ();
pub type False = Not<True>;
pub type None<F> = Not<Any<F>>;
pub struct Not<F>(PhantomData<F>);
pub struct Any<F>(PhantomData<F>);
pub struct Is<T>(PhantomData<T>);
pub struct Has<T>(PhantomData<T>);

impl<T: Template> Filter for Is<T> {
    fn filter(table: &Table, database: &Database) -> bool {
        create::is::<T>(table, database)
    }
}

impl<T: Template> Filter for Has<T> {
    fn filter(table: &Table, database: &Database) -> bool {
        create::has::<T>(table, database)
    }
}

impl<F: Filter> Filter for Not<F> {
    fn filter(table: &Table, database: &Database) -> bool {
        !F::filter(table, database)
    }
}

macro_rules! tuple {
    ($n:ident, $c:expr $(, $p:ident, $t:ident, $i:tt)*) => {
        impl<$($t: Filter,)*> Filter for ($($t,)*) {
            fn filter(_table: &Table, _database: &Database) -> bool {
                true $(&& $t::filter(_table, _database))*
            }
        }

        impl<$($t: Filter,)*> Filter for Any<($($t,)*)> {
            fn filter(_table: &Table, _database: &Database) -> bool {
                false $(|| $t::filter(_table, _database))*
            }
        }
    };
}
tuples!(tuple);
