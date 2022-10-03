use self::{
    filter::{Filter, FilterCondition},
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

pub trait Query<'d>: Sized {
    type Item;
    type Items<'a>: Iterator<Item = Self::Item>
    where
        Self: 'a;
    type Guard;
    type Read: Query<'d>;

    fn item<'a>(
        &'a mut self,
        key: Key,
        context: Context<'d>,
    ) -> Result<Guard<Self::Item, Self::Guard>, Error>;
    fn items<'b>(&'b mut self, context: Context<'d>) -> Self::Items<'b>;
    fn read(self) -> Self::Read;
    fn add(&mut self, table: &'d Table) -> bool;

    #[inline]
    fn join<Q: Query<'d>, B: FnMut(Self::Item) -> K, K: JoinKey>(
        self,
        query: Q,
        by: B,
    ) -> Join<'d, Self, Q, K, B> {
        Join::new(self, query, by)
    }

    #[inline]
    fn filter<F: FilterCondition>(self, filter: F) -> Filter<Self, F> {
        Filter::new(self, filter)
    }
}

pub struct Guard<T, G>(T, G);

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
    pub fn item(&mut self, key: Key) -> Result<Guard<Q::Item, Q::Guard>, Error> {
        self.update();
        self.query.item(
            key,
            Context {
                database: self.database,
            },
        )
    }

    #[inline]
    pub fn item_with<T>(&mut self, key: Key, with: impl FnOnce(Q::Item) -> T) -> Result<T, Error> {
        let Guard(item, guard) = self.item(key)?;
        let value = with(item);
        drop(guard);
        Ok(value)
    }

    #[inline]
    pub fn items(&mut self) -> Q::Items<'_> {
        self.update();
        self.query.items(Context {
            database: self.database,
        })
    }

    pub fn read(self) -> Root<'d, Q::Read> {
        Root {
            index: self.index,
            database: self.database,
            query: self.query.read(),
        }
    }

    #[inline]
    pub fn join<R: Row, K: JoinKey, B: FnMut(<Q::Read as Query<'d>>::Item) -> K>(
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
            self.query.add(table);
            self.index += 1;
        }
    }
}

impl<'d> Context<'d> {
    pub fn own(&mut self) -> Context<'d> {
        Context {
            database: self.database,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Datum;

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

        match &*query.item(key2)? {
            (by, Some(Position(position))) => {
                assert_eq!(by.0, key1);
                assert_eq!(by.1, key2);
                assert_eq!(by.2, "boba");
                assert_eq!(*position, [1.; 3]);
            }
            _ => assert!(false),
        }

        match &*query.item(key3)? {
            (by, None) => {
                assert_eq!(by.0, key2);
                assert_eq!(by.1, key3);
                assert_eq!(by.2, "fett");
            }
            _ => assert!(false),
        }

        assert_eq!(query.items().count(), 2);

        let mut a = None;
        for (_, item) in query.items() {
            a = item;
        }
        // TODO: Fix this access outside the iterator...
        a.unwrap().0 = [3.; 3];

        Ok(())
    }
}
