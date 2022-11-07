#![feature(type_alias_impl_trait)]
#![feature(auto_traits)]
#![feature(negative_impls)]
#![feature(portable_simd)]
#![feature(slice_swap_unchecked)]
#![feature(nonzero_min_max)]

pub mod add;
pub mod core;
pub mod create;
pub mod defer;
pub mod destroy;
pub mod filter;
pub mod key;
pub mod query;
pub mod remove;
pub mod resources;
pub mod row;
pub mod table;
pub mod template;

/*
    TODO: No need to use `columns: &[NonNull<()>]`.
        - Everywhere where this is used, wrap `&[Column]` and simply bypass the lock.
    TODO: Implement `Defer`:
        - Will order the resolution of deferred operations such that coherence is maintained.
    TODO: Implement `Remove<T: Template>`
    TODO: Implement `Keep<T: Template>/Trim<T: Template>`:
        - Removes all datum from a key except to ones provided by `T`.
    TODO: Implement `Change<T: Template>/Convert<T: Template>/Conform<T: Template>`:
        - Converts a key such that it corresponds to the `T: Template` (adds missing datum, removes exceeding datum).
    TODO: Implement `Template for Set<D: Datum>`:
        - Sets the datum `D` for a key only if the key already has the datum.
    TODO: Implement `Row` for `Get<D: Datum + Copy>`:
        - At the beginning of iteration, `Get` makes a copy of the whole column to a temporary buffer, then the column lock can be
        released immediately.
    TODO: Implement traits for many tuple.
    TODO: The `Table::commit` mechanism allows for an incoherent state where a create operation has been resolved, but the
    created keys are reported to not be present in a corresponding query.
        - This happens because `Table::commit` can technically fail for an unlimited amount of time...
        - Would require to force a successful commit at key moments.

    TODO: Implement `Permute`.
        - Returns all permutations (with repetitions) of two queries.
        - Order of items matters, so (A, B) is considered different than (B, A), thus both will be returned.
        - May be unified with `Combine`.
    TODO: Implement `Combine`.
        - Returns all combinations (no repetitions) of two queries.
        - Order of items does not matter, so (A, B) is the same as (B, A), thus only one of those will be returned.
        - Use a filter similar to `A.key() < B.key()` to eliminate duplicates.
    TODO: Implement compile-time checking of `Columns`, if possible.
    TODO: Prevent using a query within a query using an auto trait `Nest` that is negatively implemented for `Query`.

    TODO (POSTPONED): Implement `Row` for `Add<T: Template>`?
        - The added constraints on `Query`, the added locks, the added complexity and the added confusion in the API tend
        to outweigh the potential benefits of being able to make more assumptions about queued keys...

        - Has only one method `Add::add(self, T)` (consuming `self` ensures that there are no duplicate keys).
        - Since a table lock will be held while iterating the query up to the resolving of `Add` (inclusively), keys can be assumed
        to be valid and to not have been moved.
        - Keys can also be assumed to be ordered in the table!
        - `Add` would be resolved right after a table is done being iterated.
        - The query will need to take upgradable locks on its tables rather than read locks.
        - Upgradable locks will also need to be taken for source tables that have a target table with a lower index (such that table
        locks can be taken in index order).
        - To prevent visiting keys twice, some ordering must apply on the query tables:
            - Group tables by overlap, order them from most to least columns, groups can be reordered freely within themselves.
                - The overlap is the size of the intersection of all of the combined `Add<T>` (or similar operator) metas.
            - With `Add<A>`, all tables with an `A` must be visited first (in any order within themselves).
            - With `Add<A>, Add<B>`: `A & B`, `A | B`, others.
            - With `Add<A>, Add<B>, Add<C>`: `A & B & C`, (`A & B`) | (`A & C`) | (`B & C`), A | B | C, others.
        - Because of this ordering, `Add<T>` will not be allowed with `Remove<U>` in the same query.
    TODO (POSTPONED): Implement `Row` for `Remove<T: Template>`:
        - Same benefits and inconvenients as `Add<T>`.
        - Requires the reverse ordering as `Add<T>`.
        - Do not allow `Add`-like and `Remove`-like operators in the same query.
    TODO (POSTPONED): Implement nest operations for query:
        - Forces the outer query to take additionnal locks in case the inner query needs to produce the same item as the outer query:
            - An outer read that conflicts with an inner write becomes an upgradable read.
            - An inner read that is missing in outer becomes a read if its index is lower than the last outer column.
            - An inner write that is missing in outer becomes an upgradable read if its index is lower than the last outer column.
        - Always skip the current outer key in the inner query.
        - When inner takes column locks from a table:
            - GREATER than outer => hard lock the columns directly as usual
            - EQUAL to outer => add/upgrade the missing locks
            - LESS than outer => first try to lock the columns and on failure, drop the columns
                locks (while keeping the table locks) from the outer query and hard lock all the column locks in order.
            - FIX: this strategy may allow an outer immutable reference to be modified elsewhere while holding on to it...
        database
            .query::<(&mut A, &CopyFrom)>()
            .nest::<&A>()
            .each(|((a1, copy), nest)| nest.find(copy.0, |a2| a1.0 = a2.0));
        database
            .query::<&A>()
            .nest::<&mut A>()
            .each(|(a1, nest)| nest.each(|a2| a2.0 = a1.0));
*/

/*
TODO: There is no need to take a `Table` lock when querying as long as the column locks are always taken from left to right.
    - By declaring used metas in `Item` it should be possible to pass the `Column`.
    - The `Lock` might need specialized methods
        `fn read_lock() -> Self::ReadGuard`
        `fn read_chunk() -> Self::ReadChunk`
        `fn read_item() -> Self::ReadItem`
        `fn write_lock() -> Self::WriteGuard`
        `fn write_chunk() -> Self::WriteChunk`
        `fn write_item() -> Self::WriteItem`
TODO: Allow creating with a struct with a `#[derive(Template)]`. All members will need to implement `Template`.
TODO: Allow querying with a struct or enum with a `#[derive(Row)]`. All members will need to implement `Item`.
TODO: Allow filtering with a struct or enum with a `#[derive(Filter)]`. All members will need to implement `Filter`.


Scheduler library:
    - Can be completely independent from this library.
    - A `Run<I, O>` holds an `fn(I) -> O` where `I: Depend`.
    - With depend, the scheduler will be able to parallelize efficiently and maintain a chosen level of coherence.

    let database = Database::new();
    // Scheduler::new(): Scheduler<()>
    // Scheduler::with(&database): Scheduler<&Database>
    database
        .scheduler()
        .add(a_system) // impl FnOnce(T) -> Run<impl Depend>
        .schedule()?;

    fn a_system(database: &Database) -> Run<impl Depend> {
        Run::new(
            (database.create(), database.query()), // impl Depend
            |(create, query)| {
                let key = create.one(());
                query.find(key, |item| {});
            }) // impl FnMut(&mut S) -> O + 'static
    }
*/

use key::{Key, Keys};
use resources::Resources;
use std::{
    any::{type_name, TypeId},
    mem::{needs_drop, size_of, ManuallyDrop},
    num::NonZeroUsize,
    ptr::{copy, drop_in_place, slice_from_raw_parts_mut, NonNull},
};
use table::Tables;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Error {
    DuplicateMeta,
    ReadWriteConflict,
    WriteWriteConflict,
    AddRemoveConflict,
    WouldBlock,
    WouldDeadlock,
    InvalidKey,
    InvalidType,
    InvalidGuard,
    InsufficientGuard,
    WrongGeneration,
    WrongGenerationOrTable,
    WrongRow,
    FailedToLockTable,
    FailedToLockColumns,
    KeyNotInQuery(Key),
    KeysMustDiffer(Key),
    QueryConflict,
    Invalid,
    MissingColumn,
    MissingIndex,
    MissingJoinKey,
    InvalidTable,
    MissingTable,
}

pub struct Database {
    keys: Keys,
    tables: Tables,
    resources: Resources,
}

pub struct Meta {
    identifier: fn() -> TypeId,
    name: fn() -> &'static str,
    size: usize,
    new: unsafe fn(usize) -> NonNull<()>,
    free: unsafe fn(NonNull<()>, usize, usize),
    copy: unsafe fn((NonNull<()>, usize), (NonNull<()>, usize), NonZeroUsize),
    drop: unsafe fn(NonNull<()>, usize, NonZeroUsize),
}

impl PartialEq for Meta {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.identifier() == other.identifier()
    }
}
impl Eq for Meta {}

pub trait Datum: Sized + 'static {
    #[inline]
    fn meta() -> &'static Meta {
        &Meta {
            identifier: TypeId::of::<Self>,
            name: type_name::<Self>,
            size: size_of::<Self>(),
            new: |capacity| unsafe {
                let mut items = ManuallyDrop::new(Vec::<Self>::with_capacity(capacity));
                debug_assert!(items.capacity() == capacity || items.capacity() == usize::MAX);
                NonNull::new_unchecked(items.as_mut_ptr()).cast::<()>()
            },
            free: |data, count, capacity| unsafe {
                Vec::from_raw_parts(data.as_ptr().cast::<Self>(), count, capacity);
            },
            copy: |source, target, count| unsafe {
                if size_of::<Self>() > 0 {
                    let source = source.0.as_ptr().cast::<Self>().add(source.1);
                    let target = target.0.as_ptr().cast::<Self>().add(target.1);
                    copy(source, target, count.get());
                }
            },
            drop: |data, index, count| unsafe {
                if needs_drop::<Self>() {
                    let data = data.as_ptr().cast::<Self>().add(index);
                    drop_in_place(slice_from_raw_parts_mut(data, count.get()));
                }
            },
        }
    }
}

impl Meta {
    #[inline]
    pub fn identifier(&self) -> TypeId {
        (self.identifier)()
    }

    #[inline]
    pub fn name(&self) -> &'static str {
        (self.name)()
    }
}

impl Database {
    pub fn new() -> Self {
        Self {
            keys: Keys::new(),
            tables: Tables::new(),
            resources: Resources::new(),
        }
    }

    #[inline]
    pub const fn keys(&self) -> &Keys {
        &self.keys
    }

    #[inline]
    pub const fn tables(&self) -> &Tables {
        &self.tables
    }

    #[inline]
    pub const fn resources(&self) -> &Resources {
        &self.resources
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        filter::{False, Has},
        key::Key,
        Database, Datum, Error,
    };
    use std::{
        collections::HashSet,
        simd::{f32x16, f32x2, f32x4, f32x8},
    };

    #[derive(Debug, Clone, Copy)]
    struct A;
    #[derive(Debug, Clone, Copy)]
    struct B;
    #[derive(Debug, Clone, Copy)]
    struct C(usize);
    impl Datum for A {}
    impl Datum for B {}
    impl Datum for C {}

    #[test]
    fn create_adds_a_table() -> Result<(), Error> {
        let database = Database::new();
        assert_eq!(database.tables().len(), 0);
        database.create::<()>()?;
        assert_eq!(database.tables().len(), 1);
        Ok(())
    }

    #[test]
    fn create_adds_a_table_with_datum() -> Result<(), Error> {
        let database = Database::new();
        assert_eq!(database.tables().len(), 0);
        database.create::<A>()?;
        let table = database.tables().get(0).unwrap();
        assert!(table.has::<A>());
        Ok(())
    }

    #[test]
    fn create_adds_a_table_with_data() -> Result<(), Error> {
        let database = Database::new();
        assert_eq!(database.tables().len(), 0);
        database.create::<(A, B)>()?;
        let table = database.tables().get(0).unwrap();
        assert!(table.has::<A>());
        assert!(table.has::<B>());
        Ok(())
    }

    #[test]
    fn create_one_returns_non_null_key() -> Result<(), Error> {
        let database = Database::new();
        assert_ne!(database.create()?.one(()), Key::NULL);
        Ok(())
    }

    #[test]
    fn create_all_returns_no_null_key() -> Result<(), Error> {
        let database = Database::new();
        assert!(database.create()?.all([(); 1000]).iter().all(Key::valid));
        Ok(())
    }

    #[test]
    fn create_all_returns_distinct_keys() -> Result<(), Error> {
        let database = Database::new();
        let mut set = HashSet::new();
        let mut create = database.create()?;
        assert!(create.all([(); 1000]).iter().all(|&key| set.insert(key)));
        Ok(())
    }

    #[test]
    fn create_fail_with_duplicate_datum_in_template() {
        let database = Database::new();
        let result = database.create::<(A, A)>();
        assert_eq!(result.err(), Some(Error::DuplicateMeta));
    }

    #[test]
    fn create_destroy_create_reuses_key_index() -> Result<(), Error> {
        let database = Database::new();
        let key1 = database.create()?.one(());
        assert_eq!(database.destroy().one(key1), true);
        let key2 = database.create()?.one(());
        assert_eq!(key1.index(), key2.index());
        assert_ne!(key1.generation(), key2.generation());
        Ok(())
    }

    #[test]
    fn destroy_one_fails_with_null_key() {
        let database = Database::new();
        let mut destroy = database.destroy();
        assert_eq!(
            database.keys().get(Key::NULL).err(),
            Some(Error::InvalidKey)
        );
        assert_eq!(destroy.one(Key::NULL), false);
        assert_eq!(destroy.resolve(), 0);
        assert_eq!(
            database.keys().get(Key::NULL).err(),
            Some(Error::InvalidKey)
        );
    }

    #[test]
    fn destroy_one_true_with_create_one() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create::<()>()?.one(());
        let mut destroy = database.destroy();
        assert!(database.keys().get(key).is_ok());
        assert_eq!(destroy.one(key), true);
        assert!(database.keys().get(key).is_ok());
        assert_eq!(destroy.resolve(), 1);
        assert_eq!(database.keys().get(key).err(), Some(Error::InvalidKey));
        Ok(())
    }

    #[test]
    fn destroy_all_n_with_create_all_n_resolves_n() -> Result<(), Error> {
        let database = Database::new();
        let count = |keys: &[Key]| {
            database
                .keys()
                .get_all(keys.iter().copied())
                .filter(|(_, result)| result.is_ok())
                .count()
        };

        let keys = database.create::<()>()?.all_n([(); 1000]);
        let mut destroy = database.destroy();
        assert_eq!(count(&keys), 1000);
        assert_eq!(destroy.all(keys.clone()), 1000);
        assert_eq!(count(&keys), 1000);
        assert_eq!(destroy.resolve(), 1000);
        assert_eq!(count(&keys), 0);
        Ok(())
    }

    #[test]
    fn add_resolve_is_0() -> Result<(), Error> {
        let database = Database::new();
        assert_eq!(database.add::<()>()?.resolve(), 0);
        Ok(())
    }

    #[test]
    fn add_no_double_resolve() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create()?.one(());
        let mut add = database.add()?;
        assert!(add.one(key, A));
        assert_eq!(add.resolve(), 1);
        assert_eq!(add.resolve(), 0);
        Ok(())
    }

    #[test]
    fn add_simple_template() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create()?.one(());
        let mut add = database.add()?;
        assert!(add.one(key, A));
        assert!(!database.query::<&A>()?.has(key));
        assert_eq!(add.resolve(), 1);
        assert!(database.query::<&A>()?.has(key));
        Ok(())
    }

    #[test]
    fn add_simple_template_twice() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create()?.one(());
        let mut add_a = database.add()?;
        let mut add_b = database.add()?;

        assert!(add_a.one(key, A));
        assert!(add_b.one(key, B));
        assert!(!database.query::<&A>()?.has(key));
        assert!(!database.query::<&B>()?.has(key));
        assert!(!database.query::<(&A, &B)>()?.has(key));

        assert_eq!(add_a.resolve(), 1);
        assert!(database.query::<&A>()?.has(key));
        assert!(!database.query::<&B>()?.has(key));
        assert!(!database.query::<(&A, &B)>()?.has(key));

        assert_eq!(add_b.resolve(), 1);
        assert!(database.query::<&A>()?.has(key));
        assert!(database.query::<&B>()?.has(key));
        assert!(database.query::<(&A, &B)>()?.has(key));
        Ok(())
    }

    #[test]
    fn add_composite_template() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create()?.one(());
        let mut add = database.add()?;

        assert!(add.one(key, (A, B)));
        assert!(!database.query::<&A>()?.has(key));
        assert!(!database.query::<&B>()?.has(key));
        assert!(!database.query::<(&A, &B)>()?.has(key));

        assert_eq!(add.resolve(), 1);
        assert!(database.query::<&A>()?.has(key));
        assert!(database.query::<&B>()?.has(key));
        assert!(database.query::<(&A, &B)>()?.has(key));
        Ok(())
    }

    #[test]
    fn destroy_all_resolve_0() -> Result<(), Error> {
        let database = Database::new();
        assert_eq!(database.destroy_all().resolve(), 0);
        Ok(())
    }

    #[test]
    fn destroy_all_resolve_100() -> Result<(), Error> {
        let database = Database::new();
        database.create()?.all_n([(); 100]);
        assert_eq!(database.destroy_all().resolve(), 100);
        Ok(())
    }

    #[test]
    fn destroy_all_filter() -> Result<(), Error> {
        let database = Database::new();
        database.create()?.all_n([(); 100]);
        database.create()?.all_n([A; 100]);
        assert_eq!(database.destroy_all().filter::<Has<A>>().resolve(), 100);
        assert_eq!(database.destroy_all().resolve(), 100);
        Ok(())
    }

    #[test]
    fn query_is_some_create_one_key() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create()?.one(());
        assert!(database.query::<()>()?.has(key));
        assert_eq!(database.query::<()>()?.find(key, |_| true), Ok(true));
        Ok(())
    }

    #[test]
    fn query_is_none_destroy_one_key() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create()?.one(());
        database.destroy().one(key);
        let mut query = database.query::<()>()?;
        assert_eq!(query.find(key, |_| {}).err(), Some(Error::InvalidKey));
        Ok(())
    }

    #[test]
    fn query_is_some_all_create_all() -> Result<(), Error> {
        let database = Database::new();
        let keys = database.create()?.all_n([(); 1000]);
        let mut query = database.query::<()>()?;
        assert!(keys.iter().all(|&key| query.find(key, |_| {}).is_ok()));
        Ok(())
    }

    #[test]
    fn query_is_some_remain_destroy_all() -> Result<(), Error> {
        let database = Database::new();
        let keys = database.create()?.all_n([(); 1000]);
        database.destroy().all(keys[..500].iter().copied());
        let mut query = database.query::<()>()?;
        assert!(keys[..500]
            .iter()
            .all(|&key| query.find(key, |_| {}).is_err()));
        assert!(keys[500..]
            .iter()
            .all(|&key| query.find(key, |_| {}).is_ok()));
        Ok(())
    }

    #[test]
    fn query_is_err_with_write_write() {
        let database = Database::new();
        assert_eq!(
            database.query::<(&mut A, &mut A)>().err(),
            Some(Error::WriteWriteConflict)
        );
    }

    #[test]
    fn query_is_err_with_read_write() {
        let database = Database::new();
        assert_eq!(
            database.query::<(&A, &mut A)>().err(),
            Some(Error::ReadWriteConflict)
        );
        assert_eq!(
            database.query::<(&mut A, &A)>().err(),
            Some(Error::ReadWriteConflict)
        );
    }

    #[test]
    fn query_with_false_is_always_empty() -> Result<(), Error> {
        let database = Database::new();
        let mut query = database.query::<()>()?.filter::<False>();
        assert_eq!(query.count(), 0);
        let key = database.create()?.one(());
        assert_eq!(query.count(), 0);
        assert_eq!(
            query.find(key, |_| {}).err(),
            Some(Error::KeyNotInQuery(key))
        );
        Ok(())
    }

    #[test]
    fn query_reads_same_datum_as_create_one() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create()?.one(C(1));
        assert_eq!(database.query::<&C>()?.find(key, |c| c.0), Ok(1));
        Ok(())
    }

    #[test]
    fn query1_reads_datum_written_by_query2() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create()?.one(C(1));
        database.query::<&mut C>()?.find(key, |c| c.0 += 1)?;
        assert_eq!(database.query::<&C>()?.find(key, |c| c.0), Ok(2));
        Ok(())
    }

    #[test]
    fn query_option_read() -> Result<(), Error> {
        let database = Database::new();
        let key1 = database.create()?.one(());
        let key2 = database.create()?.one(A);
        let mut query = database.query::<Option<&A>>()?;
        assert_eq!(query.count(), 2);
        assert_eq!(query.find(key1, |a| a.is_some()), Ok(false));
        assert_eq!(query.find(key2, |a| a.is_some()), Ok(true));
        assert_eq!(query.chunk().count(), 2);
        Ok(())
    }

    // #[test]
    fn multi_join() -> Result<(), Error> {
        struct A(Vec<Key>);
        impl Datum for A {}

        let database = Database::new();
        let mut create = database.create()?;
        let a = create.one(A(vec![]));
        let b = create.one(A(vec![a, a, Key::NULL]));
        create.one(A(vec![Key::NULL, Key::NULL, a, b]));
        create.resolve();

        let mut query = database.query::<&A>()?;
        assert_eq!(query.count(), 3);
        let mut by = database.by();
        assert_eq!(by.len(), 0);
        query.each(|a| {
            by.keys(a.0.iter().cloned());
        });
        assert_eq!(query.count_by(&by), 4);
        assert_eq!(by.len(), 4);

        let mut query = database.query::<Key>()?;
        assert_eq!(query.count(), 3);
        let sum = query.fold_by(&mut by, 0, |sum, _, item| match item {
            Ok(key) if key == a || key == b => sum + 1,
            _ => sum,
        });
        assert_eq!(by.len(), 0);
        assert_eq!(sum, 4);
        Ok(())
    }

    #[test]
    fn copy_to() -> Result<(), Error> {
        struct A(usize);
        struct CopyTo(Key);
        impl Datum for A {}
        impl Datum for CopyTo {}

        let database = Database::new();
        let mut a = database.create()?.one(A(1));
        let mut create = database.create()?;
        for i in 0..100 {
            a = create.one((A(i), CopyTo(a)));
        }
        create.resolve();

        let mut sources = database.query::<(&A, &CopyTo)>()?;
        let mut targets = database.query::<&mut A>()?;
        let mut by = database.by();
        sources.each(|(a, copy)| {
            by.pair(copy.0, a.0);
        });
        targets.each_by_ok(&mut by, |value, a| a.0 = value);
        // TODO: Add assertions.
        Ok(())
    }

    // #[test]
    fn copy_from() -> Result<(), Error> {
        struct A(usize);
        struct CopyFrom(Key);
        impl Datum for A {}
        impl Datum for CopyFrom {}
        const COUNT: usize = 10;

        let database = Database::new();
        let a = database.create()?.one(A(1));
        let mut create = database.create()?;
        for i in 0..COUNT {
            create.one((A(i), CopyFrom(a)));
        }
        create.resolve();

        let mut copies = database.query::<(Key, &CopyFrom)>()?;
        let mut sources = database.query::<&A>()?;
        let mut targets = database.query::<&mut A>()?;
        let mut by_source = database.by();
        let mut by_target = database.by();

        copies.each(|(key, copy)| {
            by_source.pair(copy.0, key);
        });
        sources.each_by_ok(&mut by_source, |target, a| {
            by_target.pair(target, a.0);
        });
        targets.each_by_ok(&mut by_target, |value, a| a.0 = value);

        assert_eq!(copies.count(), COUNT);
        assert_eq!(sources.count(), COUNT + 1);
        assert_eq!(targets.count(), COUNT + 1);
        assert_eq!(by_source.len(), 0);
        assert_eq!(by_target.len(), 0);
        sources.each(|a| assert_eq!(a.0, 1));
        targets.each(|a| assert_eq!(a.0, 1));
        Ok(())
    }

    #[test]
    fn swap() -> Result<(), Error> {
        struct Swap(Key, Key);
        impl Datum for Swap {}

        let database = Database::new();
        let mut a = database.create()?.one(C(1));
        let mut b = database.create()?.one(C(2));
        let mut create = database.create()?;
        for i in 0..100 {
            let c = create.one((C(i), Swap(a, b)));
            a = b;
            b = c;
        }
        create.resolve();

        let mut swaps = database.query::<&Swap>()?;
        let mut sources = database.query::<&C>()?;
        let mut targets = database.query::<&mut C>()?;
        let mut by_source = database.by();
        let mut by_target = database.by();
        swaps.each(|swap| {
            by_source.pairs([(swap.0, swap.1), (swap.1, swap.0)]);
        });
        sources.each_by_ok(&mut by_source, |target, c| {
            by_target.pair(target, c.0);
        });
        targets.each_by_ok(&mut by_target, |value, c| c.0 = value);
        // TODO: Add assertions.
        Ok(())
    }

    // #[test]
    fn simd_add() -> Result<(), Error> {
        #[repr(transparent)]
        #[derive(Clone, Copy)]
        struct A(f32);
        #[repr(transparent)]
        #[derive(Clone, Copy)]
        struct B(f32);
        impl Datum for A {}
        impl Datum for B {}

        let database = Database::new();
        database.create()?.all_n([(A(0.0), B(1.0)); 100]);
        database.query::<(&mut A, &B)>()?.chunk().each(|(a, b)| {
            debug_assert_eq!(a.len(), b.len());

            #[inline]
            unsafe fn add16(source: *const f32, target: *mut f32) {
                *(target as *mut f32x16) += *(source as *mut f32x16);
            }
            #[inline]
            unsafe fn add8(source: *const f32, target: *mut f32) {
                *(target as *mut f32x8) += *(source as *mut f32x8);
            }
            #[inline]
            unsafe fn add4(source: *const f32, target: *mut f32) {
                *(target as *mut f32x4) += *(source as *mut f32x4);
            }
            #[inline]
            unsafe fn add2(source: *const f32, target: *mut f32) {
                *(target as *mut f32x2) += *(source as *mut f32x2);
            }
            #[inline]
            unsafe fn add1(source: *const f32, target: *mut f32) {
                *target += *source
            }
            unsafe fn add(source: *const f32, target: *mut f32, count: usize) {
                match count {
                    0 => {}
                    1 => add1(source, target),
                    2 => add2(source, target),
                    3 => {
                        add2(source, target);
                        add1(source.add(2), target.add(2));
                    }
                    4 => add4(source, target),
                    5 => {
                        add4(source, target);
                        add1(source.add(4), target.add(4));
                    }
                    6 => {
                        add4(source, target);
                        add2(source.add(4), target.add(4));
                    }
                    7 => {
                        add4(source, target);
                        add2(source.add(4), target.add(4));
                        add1(source.add(6), target.add(6));
                    }
                    8 => add8(source, target),
                    9 => {
                        add8(source, target);
                        add1(source.add(8), target.add(8));
                    }
                    10 => {
                        add8(source, target);
                        add2(source.add(8), target.add(8));
                    }
                    11 => {
                        add8(source, target);
                        add2(source.add(8), target.add(8));
                        add1(source.add(10), target.add(10));
                    }
                    12 => {
                        add8(source, target);
                        add4(source.add(8), target.add(8));
                    }
                    13 => {
                        add8(source, target);
                        add4(source.add(8), target.add(8));
                        add1(source.add(12), target.add(12));
                    }
                    14 => {
                        add8(source, target);
                        add4(source.add(8), target.add(8));
                        add2(source.add(12), target.add(12));
                    }
                    15 => {
                        add8(source, target);
                        add4(source.add(8), target.add(8));
                        add2(source.add(12), target.add(12));
                        add1(source.add(14), target.add(14));
                    }
                    _ => unreachable!(),
                }
            }

            let count = a.len();
            let mut source = b.as_ptr() as _;
            let mut target = a.as_mut_ptr() as _;
            for _ in 0..count / 16 {
                unsafe { add16(source, target) };
                unsafe { source = source.add(16) };
                unsafe { target = target.add(16) };
            }
            unsafe { add(source, target, count % 16) };
        });

        Ok(())
    }
}

mod locks {
    use std::marker::PhantomData;

    pub trait Datum {}
    pub trait Item {}
    impl<D: Datum> Item for &D {}
    impl<D: Datum> Item for &mut D {}
    impl Item for () {}
    impl<I1: Item> Item for (I1,) {}
    impl<I1: Item, I2: Item + Allow<I1>> Item for (I1, I2) {}
    impl<I1: Item, I2: Item + Allow<I1>, I3: Item + Allow<I1> + Allow<I2>> Item for (I1, I2, I3) {}
    impl Datum for bool {}
    impl Datum for char {}

    pub auto trait Safe {}
    impl<T, U> Safe for (&T, &U) {}
    impl<T> !Safe for (&mut T, &T) {}
    impl<T> !Safe for (&mut T, &mut T) {}
    impl<T> !Safe for (&T, &mut T) {}

    pub trait Allow<T> {}
    impl<T, U> Allow<U> for &T where for<'a> (&'a T, U): Safe {}
    impl<T, U> Allow<U> for &mut T where for<'a> (&'a mut T, U): Safe {}
    impl<U> Allow<U> for () {}
    impl<T1, U: Allow<T1>> Allow<U> for (T1,) {}
    impl<T1, T2, U: Allow<T1> + Allow<T2>> Allow<U> for (T1, T2) {}
    impl<T1, T2, T3, U: Allow<T1> + Allow<T2> + Allow<T3>> Allow<U> for (T1, T2, T3) {}

    struct Query<I: Item>(PhantomData<I>);
    impl<I: Item, U: Allow<I>> Allow<U> for Query<I> {}
    impl<I: Item> Query<I> {
        pub fn new() -> Self {
            todo!()
        }
        pub fn each(&mut self, each: impl FnMut(I) + 'static) {}
        pub fn each_with<S: Allow<I>>(&mut self, state: S, each: impl FnMut(I, S) + 'static) {}
    }

    fn boba<T, U>(a: T, b: U)
    where
        (T, U): Item,
    {
    }
    fn jango<T: Item>(a: T) {}
    fn karl(mut query1: Query<&mut bool>) {
        let a = &mut 'a';
        let c = &mut true;
        let e = true;
        let query2 = Query::<&mut char>::new();
        query1.each(move |b| *b = e);
        query1.each_with(a, move |b, c| *b = *c == '0');
        query1.each_with(query2, move |b, mut query| {
            query.each_with(b, |c, b| *b = *c == '0')
        });
        // query.each_with(c, move |b, c| *b = *c);
        // query.each(|b| *b = *c);
        // karl([a], move |b: &bool, a| {
        //     *a[0] = *b;
        //     // *c = *a;
        //     d += 1;
        // });
    }
    fn fett<T: Datum>(value: &T) {
        // Succeed

        jango(&true);
        jango(&mut true);
        jango((&true, &true));
        jango((&mut 'a', &true));
        jango((&true, &mut 'a'));

        boba(value, &true);
        boba(&'a', &mut true);
        boba(&mut 'a', &mut true);
        boba(&true, (&'a', &true));
        boba((&'a', &true), &true);
        boba(value, value);

        // Fail
        // boba(&mut 'a', value);
        // jango(((&true, ((),)), (&mut 'a', (), &mut true)));
        // jango((&mut true, &true));
        // jango((&true, &mut true));
        // jango((&mut true, &mut true));
        // boba(&false, &mut true);
        // boba(&mut false, &true);
        // boba(&true, (&mut 'b', &mut true));
        // boba(
        //     (&'a', (&true, &'b', ((), ()))),
        //     ((), &'b', (&'a', (&mut true, ((), ())))),
        // );
    }
}

mod push_vec {
    use std::{
        mem::{forget, MaybeUninit},
        ptr::{copy_nonoverlapping, null_mut},
        sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering::*},
    };

    use crate::core::utility::get_unchecked;

    const SHIFT: usize = 8;
    const CHUNK: usize = 1 << SHIFT;
    const MASK: u32 = (CHUNK - 1) as u32;

    pub struct PushVec<T> {
        count: AtomicU64,
        uses: AtomicUsize,
        chunks: AtomicPtr<Box<[MaybeUninit<T>; CHUNK]>>,
        pending: AtomicPtr<Box<[MaybeUninit<T>; CHUNK]>>,
    }

    impl<T> PushVec<T> {
        pub fn get(&self, index: u32) -> Option<&T> {
            let (count, _, _) = decompose_count(self.count.load(Acquire));
            if index >= count as _ {
                return None;
            }

            let (chunk, item) = decompose_index(index);
            let (count, _) = decompose_index(count);
            self.increment_use(count - 1);
            let chunks = self.chunks.load(Acquire);
            let chunk = unsafe { &**chunks.add(chunk as usize) };
            self.decrement_use(count - 1);
            Some(unsafe { get_unchecked(chunk, item as usize).assume_init_ref() })
        }

        pub fn push(&self, item: T) {
            let (mut count, ended, begun) = decompose_count(self.count.fetch_add(1, AcqRel));
            let index = count + ended as u32 + begun as u32;
            let (mut old_count, _) = decompose_index(count);
            let new_count = decompose_index(index);
            self.increment_use(old_count);
            let mut old_chunks = self.chunks.load(Acquire);

            debug_assert_eq!(new_count.0 - old_count, 1);
            if old_count < new_count.0 {
                // TODO: Re-read the count here? In a loop?
                let new_chunks = {
                    let mut chunks = Vec::with_capacity(new_count.0 as usize);
                    let new_chunks = chunks.as_mut_ptr();
                    unsafe { copy_nonoverlapping(old_chunks, new_chunks, old_count as usize) };
                    forget(chunks);
                    new_chunks
                };

                match self
                    .chunks
                    .compare_exchange(old_chunks, new_chunks, AcqRel, Acquire)
                {
                    Ok(chunks) => {
                        let chunk = Box::new([(); CHUNK].map(|_| MaybeUninit::<T>::uninit()));
                        unsafe { new_chunks.add(old_count as usize).write(chunk) };
                        // It should be extremely unlikely that this call returns `true`.
                        self.try_free(old_count, old_chunks);
                        old_chunks = chunks;
                    }
                    Err(chunks) => {
                        // Another thread won the race; free this allocation.
                        drop(unsafe { Vec::from_raw_parts(new_chunks, 0, new_count.0 as usize) });
                        old_chunks = chunks;
                    }
                }
                (count, _, _) = decompose_count(self.count.load(Acquire));
                (old_count, _) = decompose_index(count);
            }

            let chunk = unsafe { &mut **old_chunks.add(new_count.0 as usize) };
            self.decrement_use(old_count - 1);
            let item = MaybeUninit::new(item);
            unsafe { chunk.as_mut_ptr().add(new_count.1 as usize).write(item) };
            let result = self.count.fetch_update(AcqRel, Acquire, |count| {
                let (count, ended, begun) = decompose_count(count);
                Some(if begun == 1 {
                    recompose_count(count + ended as u32 + begun as u32, 0, 0)
                } else {
                    debug_assert!(begun > 1);
                    recompose_count(count, ended + 1, begun - 1)
                })
            });
            debug_assert!(result.is_ok());
        }

        #[inline]
        fn increment_use(&self, count: u32) {
            if self.uses.fetch_add(1, Relaxed) == 0 {
                self.try_free(count, null_mut());
            }
        }

        #[inline]
        fn decrement_use(&self, count: u32) {
            if self.uses.fetch_sub(1, Relaxed) == 1 {
                self.try_free(count, null_mut());
            }
        }

        #[inline]
        fn try_free(&self, count: u32, swap: *mut Box<[MaybeUninit<T>; CHUNK]>) -> bool {
            let pending = self.pending.swap(swap, AcqRel);
            if pending.is_null() {
                false
            } else {
                drop(unsafe { Vec::from_raw_parts(pending, 0, count as usize) });
                true
            }
        }
    }

    #[inline]
    const fn decompose_index(index: u32) -> (u32, u32) {
        (index >> SHIFT, index & MASK)
    }

    #[inline]
    const fn recompose_count(count: u32, ended: u16, begun: u16) -> u64 {
        (count as u64) << 32 | (ended as u64) << 16 | (begun as u64)
    }

    #[inline]
    const fn decompose_count(count: u64) -> (u32, u16, u16) {
        ((count >> 32) as u32, (count >> 16) as u16, count as u16)
    }
}
