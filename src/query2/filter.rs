use std::marker::PhantomData;

use super::{Context, Guard, Query};
use crate::{key::Key, table::Table, Datum};

pub struct Filter<Q, F>(Q, F);

pub trait FilterCondition {
    fn filter(&self, table: &Table) -> bool;
}

pub struct Not<F: FilterCondition>(F);
pub struct Has<D: Datum>(PhantomData<D>);

impl<F: FilterCondition> FilterCondition for Not<F> {
    fn filter(&self, table: &Table) -> bool {
        !self.0.filter(table)
    }
}

impl<D: Datum> FilterCondition for Has<D> {
    fn filter(&self, table: &Table) -> bool {
        table.has::<D>()
    }
}

impl<F: Fn(&Table) -> bool> FilterCondition for F {
    fn filter(&self, table: &Table) -> bool {
        self(table)
    }
}

impl<'a, Q: Query<'a>, F: FilterCondition> Filter<Q, F> {
    pub fn new(query: Q, filter: F) -> Self {
        Self(query, filter)
    }
}

impl<'a, Q: Query<'a>, F: FnMut(&Table) -> bool> Query<'a> for Filter<Q, F> {
    type Item = Q::Item;
    type Items<'b> = Q::Items<'b> where Self: 'b;
    type Guard = Q::Guard;
    type Read = Filter<Q::Read, F>;

    #[inline]
    fn item<'b>(
        &'b mut self,
        key: Key,
        context: Context<'a>,
    ) -> Result<Guard<Self::Item, Self::Guard>, crate::Error> {
        self.0.item(key, context)
    }

    #[inline]
    fn items<'b>(&'b mut self, context: Context<'a>) -> Self::Items<'b> {
        self.0.items(context)
    }

    fn read(self) -> Self::Read {
        Filter(self.0.read(), self.1)
    }

    fn add(&mut self, table: &'a Table) -> bool {
        self.1(table) && self.0.add(table)
    }
}
