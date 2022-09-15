#![feature(generators)]
#![feature(iter_from_generator)]

pub mod database;
pub mod modify;
pub mod query;
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

use parking_lot::{
    MappedRwLockReadGuard, MappedRwLockWriteGuard, RwLock, RwLockReadGuard,
    RwLockUpgradableReadGuard, RwLockWriteGuard,
};
use std::{
    any::{type_name, TypeId},
    cell::UnsafeCell,
    collections::{HashMap, HashSet, VecDeque},
    iter::from_generator,
    marker::PhantomData,
    mem::{forget, needs_drop, replace, size_of},
    ops::{Deref, DerefMut},
    ptr::{copy, drop_in_place, slice_from_raw_parts_mut, NonNull},
    slice::{from_raw_parts, from_raw_parts_mut, SliceIndex},
    sync::{
        atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering::*},
        Arc,
    },
};

#[derive(Debug)]
pub enum Error {}

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
}

#[cfg(test)]
mod tests {
    use crate::{
        database::{Database, Key, Table},
        Datum, Error,
    };

    #[test]
    fn main() -> Result<(), Error> {
        #[derive(Default, Clone)]
        struct Position;
        struct Velocity;
        impl Datum for Position {}
        impl Datum for Velocity {}

        let mut database = Database::new();
        fn filter(table: &Table) -> bool {
            true
        }
        let mut query1 = database.query_with::<&Position, _>(filter)?;
        let mut query2 = database.query::<Key>()?;
        let mut create = database.create()?;
        let mut destroy = database.destroy();

        for _item in query1.items() {}
        for _chunk in query1.chunks() {}
        query1.items_with(|_item| {});
        query1.chunks_with(|_chunk| {});
        query1.item_with(Key::NULL, |_item| {});
        create.one(Position);
        create.all([Position]);
        create.clones(100, Position);
        create.defaults(100);
        destroy.one(Key::NULL);
        destroy.all([Key::NULL]);
        destroy.all(query2.items());
        Ok(())
    }
}
