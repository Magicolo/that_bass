#![feature(generators)]
#![feature(iter_from_generator)]
#![feature(iter_order_by)]
#![feature(strict_provenance)]
#![feature(strict_provenance_atomic_ptr)]
#![feature(type_alias_impl_trait)]
#![feature(generic_associated_types)]

pub mod create;
pub mod database;
pub mod destroy;
pub mod key;
pub mod query;
pub mod slot;
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
TODO: Implement drop for `Inner` and `Table`.

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
    use std::any::TypeId;

    struct A;
    struct B;
    impl Datum for A {}
    impl Datum for B {}

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
    fn create_fail_with_duplicate_datum_in_template() {
        let database = Database::new();
        let result = database.create::<(A, A)>();
        assert_eq!(result.err(), Some(Error::DuplicateMeta));
    }

    #[test]
    fn query_has_create_one_key() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create()?.one(());
        let mut query = database.query::<()>()?;
        assert!(query.item(key).is_some());
        Ok(())
    }

    #[test]
    fn query_has_all_create_all() -> Result<(), Error> {
        let database = Database::new();
        let key = database.create()?.all_n([(); 1000]);
        let mut query = database.query::<()>()?;
        assert!(key.into_iter().all(|key| query.item(key).is_some()));
        Ok(())
    }

    #[test]
    fn query_fail_with_write_write() {
        let database = Database::new();
        let result = database.query::<(&mut A, &mut A)>();
        assert_eq!(result.err(), Some(Error::WriteWriteConflict));
    }

    #[test]
    fn query_fail_with_read_write() {
        let database = Database::new();
        let result = database.query::<(&mut A, &mut A)>();
        assert_eq!(result.err(), Some(Error::ReadWriteConflict));
    }
}
