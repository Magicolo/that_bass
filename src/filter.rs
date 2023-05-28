use crate::{
    core::{tuple::tuples, utility},
    table::Table,
    template::{ShareMeta, Template},
    Database,
};
use std::{any::TypeId, marker::PhantomData};

pub trait Filter {
    fn filter(&self, table: &Table, database: &Database) -> bool;
    fn dynamic(&self, database: &Database) -> Dynamic;

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

#[derive(Debug, Clone)]
pub struct Dynamic(Inner);
#[derive(Debug, Clone)]
enum Inner {
    Is(Vec<TypeId>),
    Has(Vec<TypeId>),
    Not(Box<Dynamic>),
    Any(Vec<Dynamic>),
    All(Vec<Dynamic>),
    Same(Vec<Dynamic>),
}

pub type None<F> = Not<Any<F>>;
pub type Differ<F> = Not<Same<F>>;
#[derive(Default)]
pub struct Not<F>(F);
#[derive(Default)]
pub struct Any<F>(F);
#[derive(Default)]
pub struct Same<F>(F);
pub struct Has<T>(PhantomData<T>);
pub struct Is<T>(PhantomData<T>);

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

pub const fn is<T: Template>() -> Is<T> {
    Is(PhantomData)
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

impl Dynamic {
    pub fn has<I: IntoIterator<Item = TypeId>>(types: I) -> Self {
        let mut types: Vec<_> = types.into_iter().collect();
        types.sort_unstable();
        types.dedup();
        Self(Inner::Has(types))
    }

    pub fn is<I: IntoIterator<Item = TypeId>>(types: I) -> Self {
        let mut types: Vec<_> = types.into_iter().collect();
        types.sort_unstable();
        types.dedup();
        Self(Inner::Is(types))
    }
}

impl<F: Filter> Filter for &F {
    fn filter(&self, table: &Table, database: &Database) -> bool {
        F::filter(self, table, database)
    }

    fn dynamic(&self, database: &Database) -> Dynamic {
        F::dynamic(self, database)
    }
}

impl<F: Filter> Filter for &mut F {
    fn filter(&self, table: &Table, database: &Database) -> bool {
        F::filter(self, table, database)
    }

    fn dynamic(&self, database: &Database) -> Dynamic {
        F::dynamic(self, database)
    }
}

impl Filter for bool {
    fn filter(&self, _: &Table, _: &Database) -> bool {
        *self
    }

    fn dynamic(&self, _: &Database) -> Dynamic {
        if *self {
            Dynamic(Inner::All(vec![]))
        } else {
            Dynamic(Inner::Any(vec![]))
        }
    }
}

impl<T: Template> Filter for Has<T> {
    fn filter(&self, table: &Table, database: &Database) -> bool {
        match ShareMeta::<T>::from(database) {
            Ok(metas) => table.has_all(metas.iter().map(|meta| meta.identifier())),
            Err(_) => false,
        }
    }

    fn dynamic(&self, database: &Database) -> Dynamic {
        match ShareMeta::<T>::from(database) {
            Ok(metas) => Dynamic(Inner::Has(
                metas.iter().map(|meta| meta.identifier()).collect(),
            )),
            Err(_) => false.dynamic(database),
        }
    }
}

impl<T: Template> Filter for Is<T> {
    fn filter(&self, table: &Table, database: &Database) -> bool {
        match ShareMeta::<T>::from(database) {
            Ok(metas) => table.is_all(metas.iter().map(|meta| meta.identifier())),
            Err(_) => false,
        }
    }

    fn dynamic(&self, database: &Database) -> Dynamic {
        match ShareMeta::<T>::from(database) {
            Ok(metas) => Dynamic(Inner::Is(
                metas.iter().map(|meta| meta.identifier()).collect(),
            )),
            Err(_) => false.dynamic(database),
        }
    }
}

impl<F: Filter> Filter for Not<F> {
    fn filter(&self, table: &Table, database: &Database) -> bool {
        !self.0.filter(table, database)
    }

    fn dynamic(&self, database: &Database) -> Dynamic {
        Dynamic(Inner::Not(self.0.dynamic(database).into()))
    }
}

impl Filter for Dynamic {
    fn filter(&self, table: &Table, _database: &Database) -> bool {
        match &self.0 {
            Inner::Is(types) => table.is_all(types.iter().copied()),
            Inner::Has(types) => table.has_all(types.iter().copied()),
            Inner::Not(filter) => !filter.filter(table, _database),
            Inner::Any(filters) => filters.iter().any(|filter| filter.filter(table, _database)),
            Inner::All(filters) => filters.iter().all(|filter| filter.filter(table, _database)),
            Inner::Same(filters) => {
                utility::same(filters.iter().map(|filter| filter.filter(table, _database)))
                    .unwrap_or(true)
            }
        }
    }

    fn dynamic(&self, _: &Database) -> Dynamic {
        self.clone()
    }
}

impl<T> Filter for PhantomData<T> {
    fn filter(&self, _: &Table, _: &Database) -> bool {
        true
    }

    fn dynamic(&self, database: &Database) -> Dynamic {
        true.dynamic(database)
    }
}

macro_rules! tuple {
    ($n:ident, $c:expr $(, $p:ident, $t:ident, $i:tt)*) => {
        impl<$($t: Filter,)*> Filter for ($($t,)*) {
            fn filter(&self, _table: &Table, _database: &Database) -> bool {
                true $(&& self.$i.filter(_table, _database))*
            }

            fn dynamic(&self, _database: &Database) -> Dynamic {
                Dynamic(Inner::All(vec![$(self.$i.dynamic(_database),)*]))
            }
        }

        impl<$($t: Filter,)*> Filter for Any<($($t,)*)> {
            fn filter(&self,_table: &Table, _database: &Database) -> bool {
                false $(|| self.0.$i.filter(_table, _database))*
            }

            fn dynamic(&self, _database: &Database) -> Dynamic {
                Dynamic(Inner::Any(vec![$(self.0.$i.dynamic(_database),)*]))
            }
        }

        impl<$($t: Filter,)*> Filter for Same<($($t,)*)> {
            fn filter(&self,_table: &Table, _database: &Database) -> bool {
                utility::same::<[bool; $c]>([$(self.0.$i.filter(_table, _database),)*]).unwrap_or(true)
            }

            fn dynamic(&self, _database: &Database) -> Dynamic {
                Dynamic(Inner::Same(vec![$(self.0.$i.dynamic(_database),)*]))
            }
        }
    };
}
tuples!(tuple);
