use crate::{
    core::tuples,
    create,
    resources::Resources,
    table::{Table, Tables},
    template::Template,
    Database,
};
use std::{any::TypeId, marker::PhantomData};

#[derive(Clone)]
pub struct Context<'a> {
    tables: &'a Tables,
    resources: &'a Resources,
}

pub trait Filter {
    fn filter(&self, table: &Table, context: Context) -> bool;

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
#[derive(Default)]
pub struct Not<F>(F);
#[derive(Default)]
pub struct Any<F>(F);
pub struct Has<T>(PhantomData<T>);
pub struct Is<T>(PhantomData<T>);
pub struct HasWith(Box<[TypeId]>);
pub struct IsWith(Box<[TypeId]>);
pub struct With<F>(F);

impl<'a, L> From<&'a Database<L>> for Context<'a> {
    fn from(database: &'a Database<L>) -> Self {
        Self {
            tables: database.tables(),
            resources: database.resources(),
        }
    }
}

impl<'a> From<&'a crate::Inner> for Context<'a> {
    fn from(database: &'a crate::Inner) -> Self {
        Self {
            tables: &database.tables,
            resources: &database.resources,
        }
    }
}

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

pub const fn not<F: Filter>(filter: F) -> Not<F> {
    Not(filter)
}

pub const fn any<F: Filter>(filter: F) -> Any<F> {
    Any(filter)
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

pub const fn with<F: Fn(&Table, Context)>(filter: F) -> With<F> {
    With(filter)
}

impl<F> Any<F> {
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
    fn filter(&self, table: &Table, context: Context) -> bool {
        F::filter(self, table, context)
    }
}

impl<F: Filter> Filter for &mut F {
    fn filter(&self, table: &Table, context: Context) -> bool {
        F::filter(self, table, context)
    }
}

impl Filter for bool {
    fn filter(&self, _: &Table, _: Context) -> bool {
        *self
    }
}

impl<T: Template> Filter for Has<T> {
    fn filter(&self, table: &Table, context: Context) -> bool {
        create::has::<T>(table, context.tables, context.resources)
    }
}

impl Filter for HasWith {
    fn filter(&self, table: &Table, _: Context) -> bool {
        table.has_all(self.0.iter().copied())
    }
}

impl<T: Template> Filter for Is<T> {
    fn filter(&self, table: &Table, context: Context) -> bool {
        create::is::<T>(table, context.tables, context.resources)
    }
}

impl Filter for IsWith {
    fn filter(&self, table: &Table, _: Context) -> bool {
        table.is_all(self.0.iter().copied())
    }
}

impl<F: Filter> Filter for Not<F> {
    fn filter(&self, table: &Table, context: Context) -> bool {
        !self.0.filter(table, context)
    }
}

impl<F: Fn(&Table, Context) -> bool> Filter for With<F> {
    fn filter(&self, table: &Table, context: Context) -> bool {
        self.0(table, context)
    }
}

impl<T> Filter for PhantomData<T> {
    fn filter(&self, _: &Table, _: Context) -> bool {
        true
    }
}

macro_rules! tuple {
    ($n:ident, $c:expr $(, $p:ident, $t:ident, $i:tt)*) => {
        impl<$($t: Filter,)*> Filter for ($($t,)*) {
            fn filter(&self, _table: &Table, _context: Context) -> bool {
                true $(&& self.$i.filter(_table, _context.clone()))*
            }
        }

        impl<$($t: Filter,)*> Filter for Any<($($t,)*)> {
            fn filter(&self,_table: &Table, _context: Context) -> bool {
                false $(|| self.0.$i.filter(_table, _context.clone()))*
            }
        }
    };
}
tuples!(tuple);
