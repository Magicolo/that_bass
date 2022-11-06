use crate::{create, table::Table, template::Template, Database};
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

impl Filter for () {
    fn filter(_: &Table, _: &Database) -> bool {
        true
    }
}

impl<F1: Filter> Filter for (F1,) {
    fn filter(table: &Table, database: &Database) -> bool {
        F1::filter(table, database)
    }
}

impl<F1: Filter, F2: Filter> Filter for (F1, F2) {
    fn filter(table: &Table, database: &Database) -> bool {
        F1::filter(table, database) && F2::filter(table, database)
    }
}

impl Filter for Any<()> {
    fn filter(_: &Table, _: &Database) -> bool {
        false
    }
}

impl<F1: Filter> Filter for Any<(F1,)> {
    fn filter(table: &Table, database: &Database) -> bool {
        F1::filter(table, database)
    }
}

impl<F1: Filter, F2: Filter> Filter for Any<(F1, F2)> {
    fn filter(table: &Table, database: &Database) -> bool {
        F1::filter(table, database) || F2::filter(table, database)
    }
}
