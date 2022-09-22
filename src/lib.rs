#![feature(generators)]
#![feature(iter_from_generator)]
#![feature(type_alias_impl_trait)]

pub mod bits;
pub mod create;
pub mod database;
pub mod destroy;
pub mod key;
pub mod or;
pub mod query;
pub mod resources;
pub mod table;
mod utility;
/*
COHERENCE RULES:
- Legend:
    ->: Left happens before right.
    <->: Left before or after right.
    <-: Left happens after right.
    -^: Left is used before and resolved after right.
    <-^: Left is used before or after and resolved after right.

- `Query` -> `Create`: `Query` must read the count of its overlapping tables before the `Create` begins.
- `Create` -> `Query`: `Query` must wait for the end of `Create` before reading/writing its overlapping tables.
- `Defer<Create>` <-^ `Query`: `Query` must read the count of its overlapping tables before the `Defer<Create>` is resolved.
- `Create` -> `Destroy`:

TODO: There is no need to take a `Table` lock when querying as long as the store locks are always taken from left to right.
    - By declaring used metas in `Item` it should be possible to pass the `Store`.
    - The `Lock` might need specialized methods
        `fn read_lock() -> Self::ReadGuard`
        `fn read_chunk() -> Self::ReadChunk`
        `fn read_item() -> Self::ReadItem`
        `fn write_lock() -> Self::WriteGuard`
        `fn write_chunk() -> Self::WriteChunk`
        `fn write_item() -> Self::WriteItem`
TODO: Implement drop for `Inner` and `Table`.
TODO: Allow creating with a struct with a `#[derive(Template)]`. All members will need to implement `Template`.
TODO: Allow querying with a struct or enum with a `#[derive(Item)]`. All members will need to implement `Item`.
TODO: Allow filtering with a struct or enum with a `#[derive(Filter)]`. All members will need to implement `Filter`.

- `Defer<Create>` can do most of the work of `Create` without making the changes observable.
    - Only initializing the slots and resolving the table count need to be deferred.
    - The table then must go into a state where any `Destroy` operations must consider the `reserved` count of the table when doing
    the swap and the resolution of `Create` must consider that it may have been moved or destroyed (maybe `slot.generation` can help?).

Design an ergonomic and massively parallel database with granular locking and great cache locality.
- Most of the locking would be accomplished with `parking_lot::RWLock`.
- Locking could be as granular as a single field within a `struct`.
    - Ex: Writing to `position.y` could require a read lock on the `Position` store and a write lock on `y`.
    This way, `Position` can still be read or written to by other threads.
- All accessors to the database that use locks can safely modify it immediately. The lock free accessors will defer their operations.
- A defer accessor will provide its 2 parts: `Defer` and `Resolve`.
    - They will share a special access to some data in the database to allow `Resolve` to properly resolve the deferred operations.
    - The ordering of different deferred operations is currently unclear.
*/

use std::{
    any::{type_name, TypeId},
    mem::{forget, needs_drop, size_of},
    ptr::{copy, drop_in_place, slice_from_raw_parts_mut, NonNull},
    sync::atomic::{AtomicUsize, Ordering},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    DuplicateMeta,
    ReadWriteConflict,
    WriteWriteConflict,
    WouldDeadlock,
    InvalidKey,
    WrongGeneration,
    KeyNotInQuery,
}

pub struct Meta {
    identifier: TypeId,
    name: &'static str,
    allocate: fn(usize) -> NonNull<()>,
    free: unsafe fn(NonNull<()>, usize, usize),
    copy: unsafe fn((NonNull<()>, usize), (NonNull<()>, usize), usize),
    drop: unsafe fn(NonNull<()>, usize, usize),
}

pub trait Datum: Sized + 'static {
    fn meta() -> Meta {
        Meta::new::<Self>()
    }
}

impl Meta {
    pub fn new<T: 'static>() -> Self {
        Self {
            identifier: TypeId::of::<T>(),
            name: type_name::<T>(),
            allocate: |capacity| {
                let mut items = Vec::<T>::with_capacity(capacity);
                let data = unsafe { NonNull::new_unchecked(items.as_mut_ptr().cast()) };
                forget(items);
                data
            },
            free: |data, count, capacity| unsafe {
                Vec::from_raw_parts(data.as_ptr().cast::<T>(), count, capacity);
            },
            copy: if size_of::<T>() > 0 {
                |source, target, count| unsafe {
                    if count > 0 {
                        let source = source.0.as_ptr().cast::<T>().add(source.1);
                        let target = target.0.as_ptr().cast::<T>().add(target.1);
                        copy(source, target, count);
                    }
                }
            } else {
                |_, _, _| {}
            },
            drop: if needs_drop::<T>() {
                |data, index, count| unsafe {
                    if count > 0 {
                        let data = data.as_ptr().cast::<T>().add(index);
                        drop_in_place(slice_from_raw_parts_mut(data, count));
                    }
                }
            } else {
                |_, _, _| {}
            },
        }
    }

    #[inline]
    pub const fn identifier(&self) -> TypeId {
        self.identifier
    }

    #[inline]
    pub const fn name(&self) -> &'static str {
        self.name
    }
}

pub fn identify() -> usize {
    static COUNTER: AtomicUsize = AtomicUsize::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use crate::{database::Database, key::Key, Datum, Error};
    use std::{any::TypeId, collections::HashSet, thread::scope};

    struct A;
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
        let table = database.tables().next().unwrap();
        assert!(table.has(TypeId::of::<A>()));
        Ok(())
    }

    #[test]
    fn create_adds_a_table_with_data() -> Result<(), Error> {
        let database = Database::new();
        assert_eq!(database.tables().len(), 0);
        database.create::<(A, B)>()?;
        let table = database.tables().next().unwrap();
        assert!(table.has(TypeId::of::<A>()));
        assert!(table.has(TypeId::of::<B>()));
        Ok(())
    }

    #[test]
    fn create_one_returns_non_null_key() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create()?.one(());
        assert_ne!(key, Key::NULL);
        Ok(())
    }

    #[test]
    fn create_all_n_returns_no_null_key() -> Result<(), Error> {
        let database = Database::new();
        let keys = database.create()?.all_n([(); 1000]);
        assert!(keys.iter().all(|&key| key != Key::NULL));
        Ok(())
    }

    #[test]
    fn create_all_n_returns_distinct_keys() -> Result<(), Error> {
        let database = Database::new();
        let mut set = HashSet::new();
        let keys = database.create()?.all_n([(); 1000]);
        assert!(keys.iter().all(|&key| set.insert(key)));
        Ok(())
    }

    #[test]
    fn create_fail_with_duplicate_datum_in_template() {
        let database = Database::new();
        let result = database.create::<(A, A)>();
        assert_eq!(result.err(), Some(Error::DuplicateMeta));
    }

    #[test]
    fn create_all_n_in_query_does_not_deadlock() -> Result<(), Error> {
        let database = Database::new();
        let create = database.create()?;
        let key = create.one(());
        let mut query = database.query::<Key>()?;
        let _key = query.item(key).unwrap();
        create.all_n([(); 1000]);
        Ok(())
    }

    #[test]
    fn destroy_one_fails_with_null_key() {
        let database = Database::new();
        assert_eq!(database.destroy().one(Key::NULL), false);
    }

    #[test]
    fn destroy_one_true_with_create_one() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create::<()>()?.one(());
        assert!(database.destroy().one(key));
        Ok(())
    }

    #[test]
    fn destroy_all_with_create_all_n() -> Result<(), Error> {
        let database = Database::new();
        let keys = database.create::<()>()?.all_n([(); 1000]);
        assert_eq!(database.destroy().all(keys), 1000);
        Ok(())
    }

    #[test]
    fn destroy_one_in_query_does_not_deadlock() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create()?.one(());
        let mut query = database.query::<Key>()?;
        let key = query.item(key).unwrap();
        database.destroy().one(*key);
        Ok(())
    }

    #[test]
    fn destroy_one_delays_resolution_in_query() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create()?.one(());
        let mut query1 = database.query::<()>()?;
        let mut query2 = database.query::<()>()?;
        {
            let guard = query1.item(key)?;
            assert!(database.destroy().one(key));
            // `items` will have the destroyed key.
            assert!(query2.items().next().is_some());
            // `item(key)` will not.
            assert!(query2.item(key).is_err());
            drop(guard);
        }
        assert_eq!(query1.item(key).err(), Some(Error::InvalidKey));
        assert!(query2.items().next().is_none());
        assert_eq!(query2.item(key).err(), Some(Error::InvalidKey));
        Ok(())
    }

    #[test]
    fn query_is_some_create_one_key() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create()?.one(());
        let mut query = database.query::<()>()?;
        assert!(query.item(key).is_ok());
        Ok(())
    }

    #[test]
    fn query_is_none_destroy_one_key() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create()?.one(());
        let mut query = database.query::<()>()?;
        database.destroy().one(key);
        assert_eq!(query.item(key).err(), Some(Error::InvalidKey));
        Ok(())
    }

    #[test]
    fn query_is_some_all_create_all() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create()?.all_n([(); 1000]);
        let mut query = database.query::<()>()?;
        assert!(key.into_iter().all(|key| query.item(key).is_ok()));
        Ok(())
    }

    #[test]
    fn query_is_some_remain_destroy_all() -> Result<(), Error> {
        let database = Database::new();
        let keys = database.create()?.all_n([(); 1000]);
        let mut query = database.query::<()>()?;
        database.destroy().all(keys[..500].into_iter().copied());
        assert!(keys[..500].into_iter().all(|&key| query.item(key).is_err()));
        assert!(keys[500..].into_iter().all(|&key| query.item(key).is_ok()));
        Ok(())
    }

    #[test]
    fn query_is_err_with_write_write() {
        assert_eq!(
            Database::new().query::<(&mut A, &mut A)>().err(),
            Some(Error::WriteWriteConflict)
        );
    }

    #[test]
    fn query_is_err_with_read_write() {
        assert_eq!(
            Database::new().query::<(&A, &mut A)>().err(),
            Some(Error::ReadWriteConflict)
        );
        assert_eq!(
            Database::new().query::<(&mut A, &A)>().err(),
            Some(Error::ReadWriteConflict)
        );
    }

    #[test]
    fn query_with_false_is_always_empty() -> Result<(), Error> {
        let database = Database::new();
        let mut query = database.query_with::<(), _>(false)?;
        assert!(query.items().next().is_none());
        let key = database.create()?.one(());
        assert!(query.items().next().is_none());
        assert_eq!(query.item(key).err(), Some(Error::KeyNotInQuery));
        Ok(())
    }

    #[test]
    fn query_reads_same_datum_as_create_one() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create()?.one(C(1));
        let mut query = database.query::<&C>()?;
        assert_eq!(query.item(key).unwrap().0, 1);
        Ok(())
    }

    #[test]
    fn query1_reads_datum_written_by_query2() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create()?.one(C(1));
        let mut query1 = database.query::<&C>()?;
        let mut query2 = database.query::<&mut C>()?;
        query2.item(key).unwrap().0 += 1;
        assert_eq!(query1.item(key).unwrap().0, 2);
        Ok(())
    }

    #[test]
    fn query1_item_nested_in_query2_items_does_not_deadlock() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create()?.one(C(1));
        let mut query1 = database.query::<&mut C>()?;
        let mut query2 = database.query::<&mut C>()?;
        for _item1 in query1.items() {
            assert_eq!(query2.item(key).err(), Some(Error::WouldDeadlock));
        }
        Ok(())
    }

    #[test]
    fn query1_items_nested_in_query2_items_does_not_deadlock() -> Result<(), Error> {
        let database = Database::new();
        database.create()?.one(C(1));
        let mut query1 = database.query::<&mut C>()?;
        let mut query2 = database.query::<&mut C>()?;

        // TOOD: Give a way to iterate over skipped items (see `query.skip` which is already populated).
        for _item1 in query1.items() {
            for _item2 in query2.items() {
                // Getting here means that a mutable reference would be aliased; it must never happen.
                assert!(false);
            }
        }

        Ok(())
    }

    #[test]
    fn boba() -> Result<(), Error> {
        let database = Database::new();
        let create = database.create()?;
        let mut query = database.query::<()>()?;
        let result = scope(|scope| {
            let create = &create;
            let query = &mut query;
            let mut handles = Vec::new();
            let mut counts = Vec::new();
            for _ in 0..10 {
                handles.push(scope.spawn(move || create.all_n([C(123); 10_000])));
                counts.push(query.items().count());
            }
            for handle in handles {
                match handle.join() {
                    Ok(_) => {}
                    Err(error) => return Err(error),
                }
            }
            Ok(counts)
        });
        Ok(())
    }

    fn fett() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create()?.one(C(1));
        let mut query1 = database.query::<&mut C>()?;
        let mut query2 = database.query::<&mut C>()?;
        let mut a = None;
        let mut b = None;
        for _item1 in query1.items() {
            // This is fine.
            a = Some(_item1.clone());
            // TODO: Items must not be allowed to escape like this...
            // - Link the lifetime of items to the iterator somehow.
            b = Some(_item1);
            let _item2 = query2.item(key);
            // dbg!(_item1.0);
        }
        query1.items_with(|item| {
            a = Some(item.clone());
            // b = Some(item); // The closure properly prevents this!
            let _item2 = query2.item(key);
        });
        if let Some(mut a) = a {
            a.0 += 1;
        }
        // Oh no! Item is being accessed outside its lock.
        if let Some(mut b) = b {
            b.0 += 1;
        }

        Ok(())
    }
}

mod locks {
    use parking_lot::RwLock;
    use std::rc::Weak;

    pub enum Kind {
        Read,
        Upgrade,
        Write,
    }
    pub struct Locks(Vec<Weak<RwLock<()>>>);

    impl Locks {
        fn boba(&self) {}
    }
}

mod push_vec {
    use std::{
        mem::{forget, MaybeUninit},
        ptr::{copy_nonoverlapping, null_mut},
        sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering::*},
    };

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
            Some(unsafe { chunk.get_unchecked(item as usize).assume_init_ref() })
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
