use crate::{
    core::{tuple::tuples, utility},
    create,
    table::Table,
    template::Template,
    Database,
};
use std::{any::TypeId, marker::PhantomData};

pub trait Filter {
    fn filter(&self, table: &Table, database: &Database) -> bool;

    fn not(self) -> Not<Self>
    where
        Self: Sized,
    {
        Not(self)
    }
    fn and<F: Filter>(self, filter: F) -> (Self, F)
    where
        Self: Sized,
    {
        (self, filter)
    }
    fn or<F: Filter>(self, filter: F) -> Any<(Self, F)>
    where
        Self: Sized,
    {
        Any((self, filter))
    }
}

pub type None<F> = Not<Any<F>>;
pub type Difer<F> = Not<Same<F>>;
#[derive(Default)]
pub struct Not<F>(F);
#[derive(Default)]
pub struct Any<F>(F);
#[derive(Default)]
pub struct Same<F>(F);
pub struct Has<T>(PhantomData<T>);
pub struct Is<T>(PhantomData<T>);
pub struct HasWith(Box<[TypeId]>);
pub struct IsWith(Box<[TypeId]>);
pub struct With<F>(F);

impl<T: Template> Default for Has<T> {
    fn default() -> Self {
        has::<T>()
    }
}

impl<T: Template> Default for Is<T> {
    fn default() -> Self {
        is::<T>()
    }
}

pub const fn has<T: Template>() -> Has<T> {
    Has(PhantomData)
}

pub const fn not<F: Filter>(filter: F) -> Not<F>
where
    Not<F>: Filter,
{
    Not(filter)
}

pub const fn any<F: Filter>(filter: F) -> Any<F>
where
    Any<F>: Filter,
{
    Any(filter)
}

pub const fn same<F: Filter>(filter: F) -> Same<F>
where
    Same<F>: Filter,
{
    Same(filter)
}

pub fn has_with<I: IntoIterator<Item = TypeId>>(types: I) -> HasWith {
    let mut types: Vec<_> = types.into_iter().collect();
    types.sort_unstable();
    types.dedup();
    HasWith(types.into_boxed_slice())
}

pub const fn is<T: Template>() -> Is<T> {
    Is(PhantomData)
}

pub fn is_with<I: IntoIterator<Item = TypeId>>(types: I) -> IsWith {
    let mut types: Vec<_> = types.into_iter().collect();
    types.sort_unstable();
    types.dedup();
    IsWith(types.into_boxed_slice())
}

pub const fn with<F: Fn(&Table, &Database)>(filter: F) -> With<F> {
    With(filter)
}

impl<F> Any<F> {
    pub const fn inner(&self) -> &F {
        &self.0
    }
}

impl<F> Same<F> {
    pub const fn inner(&self) -> &F {
        &self.0
    }
}

impl<F> Not<F> {
    pub const fn inner(&self) -> &F {
        &self.0
    }
}

impl<F: Filter> Filter for &F {
    fn filter(&self, table: &Table, database: &Database) -> bool {
        F::filter(self, table, database)
    }
}

impl<F: Filter> Filter for &mut F {
    fn filter(&self, table: &Table, database: &Database) -> bool {
        F::filter(self, table, database)
    }
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

impl<F: Fn(&Table, &Database) -> bool> Filter for With<F> {
    fn filter(&self, table: &Table, database: &Database) -> bool {
        self.0(table, database)
    }
}

impl<T> Filter for PhantomData<T> {
    fn filter(&self, _: &Table, _: &Database) -> bool {
        true
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

        impl<$($t: Filter,)*> Filter for Same<($($t,)*)> {
            fn filter(&self,_table: &Table, _database: &Database) -> bool {
                utility::same::<[bool; $c]>([$(self.0.$i.filter(_table, _database),)*]).unwrap_or(true)
            }
        }
    };
}
tuples!(tuple);
