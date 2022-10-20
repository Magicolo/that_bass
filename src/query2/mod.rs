use self::{
    filter::{Condition, Filter},
    join::{Join, JoinKey},
    row::{Row, Rows},
};
use crate::{database::Database, key::Key, table::Table, Error};

pub mod filter;
pub mod join;
pub mod row;

/*
    TODO: Try to implement a `FullJoin` that returns both `L::Item` and `R::Item`.
        - This join would force coherence between `L` and `R`, meaning that the join key retrieved from `L::Item` would still be
        the same when both items are yielded.
        - A `FullJoin` will require very careful synchronization between locks to prevent the `symmetric join` problem
        (i.e. `FullJoin<A, B>` running concurrently with `FullJoin<B, A>` must not deadlock).
    TODO: Implement `Filter`.
        - Static filtering using types (or a function pointer?).
        - Should be possible to filter at any depth of a query since it adds nothing to the items.
    TODO: Implement `Permute`.
        - Returns all permutations (with repetitions) of two queries.
        - Order of items matters, so (A, B) is considered different than (B, A), thus both will be returned.
        - May be unified with `Combine`.
    TODO: Implement `Combine`.
        - Returns all combinations (no repetitions) of two queries.
        - Order of items does not matter, so (A, B) is the same as (B, A), thus only one of those will be returned.
        - Use a filter similar to `A.key() < B.key()` to eliminate duplicates.
    TODO: `Query::Item` should implement `Item`?
    TODO: Implement compile-time checking of `Columns`, if possible.
    TODO: Fix the lifetime problem with `Query::Item`.
        - `trait Items {
            type Item<'a>;
            fn next(&mut self) -> Option<Self::Item<'_>>;
        }`
        - `Query::Item<'b>: Item<'b>;`
        - `Query::Items<'b>: for<'c> Items<Item<'c> = Self::Item<'c>>`
*/

// pub type Item<'d, 'a, 'b, Q> = <<Q as Query<'d>>::Items<'a> as Iterate>::Item<'b>;

pub trait Query<'d>: Sized {
    type Item<'a>;
    type Read: Query<'d>;

    fn initialize(&mut self, table: &'d Table) -> Result<(), Error>;
    fn read(self) -> Self::Read;

    fn try_find<T, F: FnOnce(Result<Self::Item<'_>, Error>) -> T>(
        &mut self,
        key: Key,
        context: Context<'d>,
        find: F,
    ) -> T;

    #[inline]
    fn find<T, F: FnOnce(Self::Item<'_>) -> T>(
        &mut self,
        key: Key,
        context: Context<'d>,
        find: F,
    ) -> Result<T, Error> {
        self.try_find(key, context, |item| item.map(find))
    }

    fn try_fold<S, F: FnMut(S, Self::Item<'_>) -> Result<S, S>>(
        &mut self,
        context: Context<'d>,
        state: S,
        fold: F,
    ) -> S;

    #[inline]
    fn fold<S, F: FnMut(S, Self::Item<'_>) -> S>(
        &mut self,
        context: Context<'d>,
        state: S,
        mut fold: F,
    ) -> S {
        self.try_fold(context, state, |state, item| Ok(fold(state, item)))
    }

    #[inline]
    fn try_each<F: FnMut(Self::Item<'_>) -> bool>(&mut self, context: Context<'d>, mut each: F) {
        self.try_fold(context, (), |_, item| each(item).then_some(()).ok_or(()))
    }

    #[inline]
    fn each<F: FnMut(Self::Item<'_>)>(&mut self, context: Context<'d>, mut each: F) {
        self.fold(context, (), |_, item| each(item))
    }

    #[inline]
    fn join<Q: Query<'d>, B: FnMut(Self::Item<'_>) -> K, K: JoinKey>(
        self,
        query: Q,
        by: B,
    ) -> Join<'d, Self, Q, K, B> {
        Join::new(self, query, by)
    }

    #[inline]
    fn filter<F: Condition>(self, filter: F) -> Filter<Self, F> {
        Filter::new(self, filter)
    }
}

pub struct Root<'d, Q> {
    index: usize,
    database: &'d Database,
    query: Q,
}

pub struct Context<'d> {
    database: &'d Database,
}

impl Database {
    pub fn query2<R: Row>(&self) -> Result<Root<Rows<R>>, Error> {
        Ok(Root {
            index: 0,
            database: self,
            query: Rows::new()?,
        })
    }
}

impl<'d, Q: Query<'d>> Root<'d, Q> {
    #[inline]
    pub fn try_find<T, F: FnOnce(Result<Q::Item<'_>, Error>) -> T>(
        &mut self,
        key: Key,
        find: F,
    ) -> T {
        self.update();
        self.query.try_find(key, Context::new(self.database), find)
    }

    #[inline]
    pub fn find<T, F: FnOnce(Q::Item<'_>) -> T>(&mut self, key: Key, find: F) -> Result<T, Error> {
        self.update();
        self.query.find(key, Context::new(self.database), find)
    }

    #[inline]
    pub fn try_fold<S, F: FnMut(S, Q::Item<'_>) -> Result<S, S>>(
        &mut self,
        state: S,
        fold: F,
    ) -> S {
        self.update();
        self.query
            .try_fold(Context::new(self.database), state, fold)
    }

    #[inline]
    pub fn fold<S, F: FnMut(S, Q::Item<'_>) -> S>(&mut self, state: S, fold: F) -> S {
        self.update();
        self.query.fold(Context::new(self.database), state, fold)
    }

    #[inline]
    pub fn try_each<F: FnMut(Q::Item<'_>) -> bool>(&mut self, each: F) {
        self.update();
        self.query.try_each(Context::new(self.database), each)
    }

    #[inline]
    pub fn each<F: FnMut(Q::Item<'_>)>(&mut self, each: F) {
        self.update();
        self.query.each(Context::new(self.database), each)
    }

    pub fn read(self) -> Root<'d, Q::Read> {
        Root {
            index: self.index,
            database: self.database,
            query: self.query.read(),
        }
    }

    #[inline]
    pub fn join<R: Row, K: JoinKey, B: FnMut(<Q::Read as Query<'d>>::Item<'_>) -> K>(
        self,
        by: B,
    ) -> Result<Root<'d, Join<'d, Q::Read, Rows<'d, R>, K, B>>, Error> {
        Ok(Root {
            index: self.index,
            database: self.database,
            query: self.query.read().join(Rows::new()?, by),
        })
    }

    #[inline]
    fn update(&mut self) {
        while let Some(table) = self.database.tables().get(self.index) {
            self.index += 1;
            let _ = self.query.initialize(table);
        }
    }
}

impl<'d> Context<'d> {
    pub fn new(database: &'d Database) -> Self {
        Self { database }
    }

    pub fn own(&mut self) -> Context<'d> {
        Context {
            database: self.database,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{database::Database, key::Key, Datum, Error};

    #[test]
    fn join() -> Result<(), Error> {
        struct Position([f64; 3]);
        struct Target(Key, &'static str);
        impl Datum for Target {}
        impl Datum for Position {}

        let database = Database::new();
        let key1 = database.create()?.one(Position([1.; 3]));
        let key2 = database.create()?.one(Target(key1, "boba"));
        let key3 = database
            .create()?
            .one((Target(key2, "fett"), Position([2.; 3])));
        let mut query = database
            .query2::<(Key, &Target)>()?
            .join::<&mut Position, _, _>(|(key, target)| (target.0, key, target.1))?;

        query
            .find(key2, |(by, item)| {
                assert_eq!(by.0, key1);
                assert_eq!(by.1, key2);
                assert_eq!(by.2, "boba");
                assert_eq!(item.unwrap().0, [1.; 3]);
            })
            .unwrap();

        query
            .find(key3, |(by, item)| {
                assert_eq!(by.0, key2);
                assert_eq!(by.1, key3);
                assert_eq!(by.2, "fett");
                assert_eq!(item.err(), Some(Error::KeyNotInQuery(key2)));
            })
            .unwrap();

        assert_eq!(query.fold(0, |count, _| count + 1), 2);

        Ok(())
    }
}
