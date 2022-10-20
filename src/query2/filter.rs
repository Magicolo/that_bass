use std::marker::PhantomData;

use super::{Context, Query};
use crate::{key::Key, table::Table, Datum, Error};

pub struct Filter<Q, C>(Q, C);

pub trait Condition {
    fn filter(&self, table: &Table) -> bool;
}

pub struct Not<C: Condition>(C);
pub struct Has<D: Datum>(PhantomData<D>);

impl<C: Condition> Condition for Not<C> {
    fn filter(&self, table: &Table) -> bool {
        !self.0.filter(table)
    }
}

impl<D: Datum> Condition for Has<D> {
    fn filter(&self, table: &Table) -> bool {
        table.has::<D>()
    }
}

impl<F: Fn(&Table) -> bool> Condition for F {
    fn filter(&self, table: &Table) -> bool {
        self(table)
    }
}

impl<'a, Q: Query<'a>, C: Condition> Filter<Q, C> {
    pub fn new(query: Q, filter: C) -> Self {
        Self(query, filter)
    }
}

impl<'d, Q: Query<'d>, C: Condition> Query<'d> for Filter<Q, C> {
    type Item<'a> = Q::Item<'a>;
    type Read = Filter<Q::Read, C>;

    fn initialize(&mut self, table: &'d Table) -> Result<(), Error> {
        if self.1.filter(table) {
            self.0.initialize(table)
        } else {
            Ok(())
        }
    }

    fn read(self) -> Self::Read {
        Filter(self.0.read(), self.1)
    }

    #[inline]
    fn try_find<T, F: FnOnce(Result<Self::Item<'_>, Error>) -> T>(
        &mut self,
        key: Key,
        context: super::Context<'d>,
        find: F,
    ) -> T {
        self.0.try_find(key, context, find)
    }

    #[inline]
    fn find<T, F: FnOnce(Self::Item<'_>) -> T>(
        &mut self,
        key: Key,
        context: Context<'d>,
        find: F,
    ) -> Result<T, Error> {
        self.0.find(key, context, find)
    }

    #[inline]
    fn try_fold<S, F: FnMut(S, Self::Item<'_>) -> Result<S, S>>(
        &mut self,
        context: Context<'d>,
        state: S,
        fold: F,
    ) -> S {
        self.0.try_fold(context, state, fold)
    }

    #[inline]
    fn fold<S, F: FnMut(S, Self::Item<'_>) -> S>(
        &mut self,
        context: Context<'d>,
        state: S,
        fold: F,
    ) -> S {
        self.0.fold(context, state, fold)
    }

    #[inline]
    fn try_each<F: FnMut(Self::Item<'_>) -> bool>(&mut self, context: Context<'d>, each: F) {
        self.0.try_each(context, each)
    }

    #[inline]
    fn each<F: FnMut(Self::Item<'_>)>(&mut self, context: Context<'d>, each: F) {
        self.0.each(context, each)
    }
}
