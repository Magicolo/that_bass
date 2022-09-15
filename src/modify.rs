use crate::{
    database::{Database, Key, Table},
    Datum, Error,
};
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

pub unsafe trait Template {
    unsafe fn apply(self, store: usize, table: &Table);
}

pub struct Spawn<T: Template>(PhantomData<T>);
pub struct With<T: Template, F: FnMut(Key) -> T>(F, PhantomData<T>);

pub struct Create<T: Template> {
    _marker: PhantomData<T>,
}

pub struct Destroy {}

unsafe impl<D: Datum> Template for D {
    unsafe fn apply(self, store: usize, table: &Table) {
        todo!()
    }
}

unsafe impl Template for () {
    unsafe fn apply(self, store: usize, table: &Table) {
        todo!()
    }
}

impl<T: Template> Create<T> {
    pub fn one(&mut self, template: T) -> Key {
        todo!()
    }

    pub fn all<I: IntoIterator<Item = T>>(&mut self, templates: I) -> &[Key] {
        todo!()
    }

    pub fn clones(&mut self, count: usize, template: T) -> &[Key]
    where
        T: Clone,
    {
        todo!()
    }

    pub fn defaults(&mut self, count: usize) -> &[Key]
    where
        T: Default,
    {
        todo!()
    }
}

impl Destroy {
    pub fn one(&mut self, key: Key) -> bool {
        todo!()
    }

    /// Destroys all provided `keys` and returns the count of the keys that were successfully destroyed.
    pub fn all<I: IntoIterator<Item = Key>>(&mut self, keys: I) -> usize {
        todo!()
    }
}

impl Database {
    pub fn create<T: Template>(&mut self) -> Result<Create<T>, Error> {
        // TODO: Fail when there are duplicate `Datum`?
        todo!()
    }

    pub fn destroy(&mut self) -> Destroy {
        todo!()
    }
}
