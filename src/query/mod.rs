use self::{
    join::{By, Join},
    row::{Filter, Row, Rows},
};
use crate::{database::Database, key::Key, Error};

pub mod join;
pub mod row;

/*
    TODO: Implement `Chunk`:

    TODO: Implement `Nest`:
        - Forces the outer query to take additionnal locks in case the inner query needs to produce the same item as the outer query:
            - An outer read that conflicts with an inner write becomes an upgradable read.
            - An inner read that is missing in outer becomes a read if its index is lower than the last outer store.
            - An inner write that is missing in outer becomes an upgradable read if its index is lower than the last outer store.
        - Always skip the current outer key in the inner query.
        - When inner takes store locks from a table:
            - GREATER than outer => hard lock the stores directly as usual
            - EQUAL to outer => add/upgrade the missing locks
            - LESS than outer => first try to lock the stores and on failure, drop the stores
                locks (while keeping the table locks) from the outer query and hard lock all the store locks in order.
            - FIX: this strategy may allow an outer immutable reference to be modified elsewhere while holding on to it...

        database
            .query::<(&mut A, &CopyFrom)>()
            .nest::<&A>()
            .each(|((a1, copy), nest)| nest.find(copy.0, |a2| a1.0 = a2.0));
        database
            .query::<&A>()
            .nest::<&mut A>()
            .each(|(a1, nest)| nest.each(|a2| a2.0 = a1.0));

    TODO: Try to implement a `FullJoin` that returns both `L::Item` and `R::Item`.
        1. For each left table 'T', lock it stores that are the strictest intersection of the left and right queries.
            - A variation on this might take only upgradable locks on a `left read/ right write` conflict.
        2. For each item `I` in `T`, run the `by` function to retrieve the join key/value.
            - If the join key is the same as the left item key, skip.
            - If the join key is in table `T`, fold right away since the locks are already taken (upgrade locks if required).
            - If the join key is in another table, enqueue (left key, (right key/value)) sorted*.
                - The sorting must allow to lock tables in `index` order.
                - Some fancy merging of left and right table locks may be accomplished here.
        3. Complete the iterations.
        4. Resolve the queued keys. Note that at this point, the left and right keys will not be in the same table since
        those are already resolved.
            - Since the keys are sorted in such a way that tables can be locked in `index` order, no deadlock can occur at
            the table level and as long as stores are also locked in order, no deadlock can occur at the store level.
            -

        - A `FullJoin` will require very careful synchronization between locks to prevent the `symmetric join` problem
        (i.e. `FullJoin<A, B>` running concurrently with `FullJoin<B, A>` must not deadlock).
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

pub struct Query<'d, Q> {
    index: usize,
    database: &'d Database,
    query: Q,
}

pub trait TableQuery<'d>: Sized {
    type Item<'a>;

    fn count(&mut self, context: Context<'d>) -> usize;

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
}

pub trait ReadQuery<'d>: TableQuery<'d> {
    type Read: TableQuery<'d>;
    fn read(self) -> Self::Read;
}

pub trait ChunkQuery<'d>: TableQuery<'d> {
    type Chunk: TableQuery<'d>;
    fn chunk(self) -> Self::Chunk;
}

pub trait FilterQuery<'d>: TableQuery<'d> {
    type Filter<F: Filter>: TableQuery<'d>;
    fn filter<F: Filter>(self, filter: F) -> Self::Filter<F>;
}

pub trait KeyQuery<'d>: TableQuery<'d> {
    #[inline]
    fn has(&mut self, key: Key, context: Context<'d>) -> bool {
        self.try_find(key, context, |result| result.is_ok())
    }

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

    #[inline]
    fn join<Q: KeyQuery<'d>, V, B: FnMut(By<V>, Self::Item<'_>)>(
        self,
        query: Q,
        by: B,
    ) -> Join<'d, Self, Q, V, B> {
        Join::new(self, query, by)
    }
}

pub struct Context<'d> {
    database: &'d Database,
}

impl Database {
    pub fn query<R: Row>(&self) -> Result<Query<Rows<R, ()>>, Error> {
        Ok(Query {
            index: 0,
            database: self,
            query: Rows::new(())?,
        })
    }
}

impl<'d, Q: TableQuery<'d>> Query<'d, Q> {
    #[inline]
    pub fn count(&mut self) -> usize {
        self.query.count(Context::new(self.database))
    }

    #[inline]
    pub fn try_fold<S, F: FnMut(S, Q::Item<'_>) -> Result<S, S>>(
        &mut self,
        state: S,
        fold: F,
    ) -> S {
        self.query
            .try_fold(Context::new(self.database), state, fold)
    }

    #[inline]
    pub fn fold<S, F: FnMut(S, Q::Item<'_>) -> S>(&mut self, state: S, fold: F) -> S {
        self.query.fold(Context::new(self.database), state, fold)
    }

    #[inline]
    pub fn try_each<F: FnMut(Q::Item<'_>) -> bool>(&mut self, each: F) {
        self.query.try_each(Context::new(self.database), each)
    }

    #[inline]
    pub fn each<F: FnMut(Q::Item<'_>)>(&mut self, each: F) {
        self.query.each(Context::new(self.database), each)
    }
}

impl<'d, Q: ReadQuery<'d>> Query<'d, Q> {
    pub fn read(self) -> Query<'d, Q::Read> {
        Query {
            index: self.index,
            database: self.database,
            query: self.query.read(),
        }
    }
}

impl<'d, Q: ChunkQuery<'d>> Query<'d, Q> {
    pub fn chunk(self) -> Query<'d, Q::Chunk> {
        Query {
            index: self.index,
            database: self.database,
            query: self.query.chunk(),
        }
    }
}

impl<'d, Q: FilterQuery<'d>> Query<'d, Q> {
    pub fn filter<F: Filter>(self, filter: F) -> Query<'d, Q::Filter<F>> {
        Query {
            index: self.index,
            database: self.database,
            query: self.query.filter(filter),
        }
    }
}

impl<'d, Q: KeyQuery<'d>> Query<'d, Q> {
    #[inline]
    pub fn has(&mut self, key: Key) -> bool {
        self.query.has(key, Context::new(self.database))
    }

    #[inline]
    pub fn try_find<T, F: FnOnce(Result<Q::Item<'_>, Error>) -> T>(
        &mut self,
        key: Key,
        find: F,
    ) -> T {
        self.query.try_find(key, Context::new(self.database), find)
    }

    #[inline]
    pub fn find<T, F: FnOnce(Q::Item<'_>) -> T>(&mut self, key: Key, find: F) -> Result<T, Error> {
        self.query.find(key, Context::new(self.database), find)
    }

    pub fn join<R: Row, V, B: FnMut(By<V>, Q::Item<'_>)>(
        self,
        by: B,
    ) -> Result<Query<'d, Join<'d, Q, Rows<'d, R, ()>, V, B>>, Error> {
        Ok(Query {
            index: self.index,
            database: self.database,
            query: self.query.join(Rows::new(())?, by),
        })
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
            .query::<(Key, &Target)>()?
            .join::<&mut Position, _, _>(|by, (key, target)| {
                by.pair(target.0, (target.0, key, target.1))
            })?;

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
