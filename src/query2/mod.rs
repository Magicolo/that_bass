use self::{
    join::{Join, JoinKey},
    row::{Row, Rows},
};
use crate::{database::Database, key::Key, Error};

pub mod join;
pub mod row;

/*
    TODO: Try to implement a `FullJoin` that returns both `L::Item` and `R::Item`.
        - This join would force coherence between `L` and `R`, meaning that the join key retrieved from `L::Item` would still be
        the same when both items are yielded.
        - A `FullJoin` will require very careful synchronization between locks to prevent the `symmetric join` problem
        (i.e. `FullJoin<A, B>` running concurrently with `FullJoin<B, A>` must not deadlock).

    TODO: Share some of the state in `Rows`?
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
*/

pub trait Query<'a>: Sized {
    type Item;
    type Items<'b>: Iterator<Item = Self::Item>
    where
        Self: 'b;
    type Guard;
    type Read: Query<'a>;

    fn item<'b>(
        &'b mut self,
        key: Key,
        context: ItemContext<'a, 'b>,
    ) -> Result<Guard<Self::Item, Self::Guard>, Error>;
    fn items<'b>(&'b mut self, context: ItemContext<'a, 'b>) -> Self::Items<'b>;
    fn read(self) -> Self::Read;

    #[inline]
    fn join<Q: Query<'a>, B: FnMut(Self::Item) -> K, K: JoinKey>(
        self,
        query: Q,
        by: B,
    ) -> Join<'a, Self, Q, K, B> {
        Join::new(self, query, by)
    }
}

pub struct Guard<T, G>(T, G);

pub struct Root<'a, Q> {
    database: &'a Database,
    query: Q,
}

pub struct ItemContext<'a, 'b> {
    database: &'a Database,
    table_locks: &'b mut [()],
}

impl Database {
    pub fn query2<R: Row>(&self) -> Result<Root<Rows<R>>, Error> {
        Ok(Root {
            database: self,
            query: Rows::new()?,
        })
    }
}

impl<'a, Q: Query<'a>> Root<'a, Q> {
    #[inline]
    pub fn item(&mut self, key: Key) -> Result<Guard<Q::Item, Q::Guard>, Error> {
        self.query.item(
            key,
            ItemContext {
                database: self.database,
                table_locks: &mut [],
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
        self.query.items(ItemContext {
            database: self.database,
            table_locks: &mut [],
        })
    }

    pub fn read(self) -> Root<'a, Q::Read> {
        Root {
            database: self.database,
            query: self.query.read(),
        }
    }

    #[inline]
    pub fn join<R: Row, K: JoinKey, B: FnMut(<Q::Read as Query<'a>>::Item) -> K>(
        self,
        by: B,
    ) -> Result<Root<'a, Join<'a, Q::Read, Rows<'a, R>, K, B>>, Error> {
        Ok(Root {
            database: self.database,
            query: self.query.read().join(Rows::new()?, by),
        })
    }
}

impl<'a> ItemContext<'a, '_> {
    pub fn own(&mut self) -> ItemContext<'a, '_> {
        ItemContext {
            database: self.database,
            table_locks: self.table_locks,
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

        Ok(())
    }
}
