use crate::{
    core::{utility::get_unchecked, FullIterator},
    key::Key,
    Datum, Error, Meta,
};
use parking_lot::{RwLock, RwLockReadGuard, RwLockUpgradableReadGuard, RwLockWriteGuard};
use std::{
    any::TypeId,
    cell::UnsafeCell,
    collections::HashMap,
    num::NonZeroUsize,
    ptr::NonNull,
    slice::from_raw_parts_mut,
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc,
    },
};

pub struct Table {
    index: u32,
    indices: HashMap<TypeId, usize>,
    pub(crate) inner: RwLock<Inner>,
}

pub(crate) struct Inner {
    pub(crate) count: AtomicU32,
    pub(crate) pending: AtomicU64,
    pub(crate) keys: UnsafeCell<Vec<Key>>,
    /// Columns are ordered consistently between tables.
    pub(crate) columns: Box<[Column]>,
}

pub struct Column {
    meta: &'static Meta,
    data: RwLock<NonNull<()>>,
}

pub struct Tables {
    /// The lock is separated from the tables because once a table is dereferenced from the `tables` vector, it no longer
    /// needs to have its lifetime tied to a `RwLockReadGuard`. This is safe because the addresses of tables are stable
    /// (guaranteed by the `Arc` indirection) and no mutable references are ever given out.
    lock: RwLock<()>,
    tables: UnsafeCell<Vec<Arc<Table>>>,
}

impl Column {
    pub fn new(meta: &'static Meta, capacity: usize) -> Self {
        Self {
            data: RwLock::new(unsafe { (meta.new)(capacity) }),
            meta,
        }
    }

    #[inline]
    pub const fn meta(&self) -> &'static Meta {
        self.meta
    }

    #[inline]
    pub(crate) const fn data(&self) -> &RwLock<NonNull<()>> {
        &self.data
    }

    #[inline]
    pub unsafe fn copy_to(source: (&mut Self, usize), target: (&Self, usize), count: NonZeroUsize) {
        debug_assert_eq!(source.0.meta().identifier(), target.0.meta().identifier());
        let &Meta { copy, .. } = source.0.meta();
        copy(
            (*source.0.data.get_mut(), source.1),
            (*target.0.data.data_ptr(), target.1),
            count,
        );
    }

    pub unsafe fn grow(&mut self, old_capacity: usize, new_capacity: NonZeroUsize) {
        debug_assert!(old_capacity < new_capacity.get());
        let &Meta {
            new, free, copy, ..
        } = self.meta();
        let data = self.data.get_mut();
        let old_data = *data;
        let new_data = new(new_capacity.get());
        if let Some(old_capacity) = NonZeroUsize::new(old_capacity) {
            copy((old_data, 0), (new_data, 0), old_capacity);
        }
        // A count of 0 is sent to `free` because the values of `old_data` have been moved to `new_data`, so they must not be dropped.
        free(old_data, 0, old_capacity);
        *data = new_data;
    }

    /// SAFETY: Both the 'source' and 'target' indices must be within the bounds of the column.
    /// The ranges 'source_index..source_index + count' and 'target_index..target_index + count' must not overlap.
    #[inline]
    pub unsafe fn squash(&mut self, source_index: usize, target_index: usize, count: NonZeroUsize) {
        let &Meta { copy, drop, .. } = self.meta();
        let data = *self.data.get_mut();
        drop(data, target_index, count);
        copy((data, source_index), (data, target_index), count);
    }

    /// SAFETY: Both the 'source' and 'target' indices must be within the bounds of the column.
    /// The ranges 'source_index..source_index + count' and 'target_index..target_index + count' must not overlap.
    #[inline]
    pub unsafe fn copy(&mut self, source_index: usize, target_index: usize, count: NonZeroUsize) {
        let &Meta { copy, .. } = self.meta();
        let data = *self.data.get_mut();
        copy((data, source_index), (data, target_index), count);
    }

    #[inline]
    pub unsafe fn drop(&mut self, index: usize, count: NonZeroUsize) {
        let &Meta { drop, .. } = self.meta();
        let data = *self.data.get_mut();
        drop(data, index, count);
    }

    #[inline]
    pub unsafe fn free(&mut self, count: usize, capacity: usize) {
        let &Meta { free, .. } = self.meta();
        let data = *self.data.get_mut();
        free(data, count, capacity);
    }

    #[inline]
    pub unsafe fn get<T: 'static>(&self, index: usize) -> &mut T {
        debug_assert_eq!(TypeId::of::<T>(), self.meta().identifier());
        let data = *self.data.data_ptr();
        &mut *data.as_ptr().cast::<T>().add(index)
    }

    #[inline]
    pub unsafe fn get_all<T: 'static>(&self, count: usize) -> &mut [T] {
        debug_assert_eq!(TypeId::of::<T>(), self.meta().identifier());
        let data = *self.data.data_ptr();
        from_raw_parts_mut(data.as_ptr().cast::<T>(), count)
    }

    #[inline]
    pub unsafe fn set<T: 'static>(&self, index: usize, value: T) {
        debug_assert_eq!(TypeId::of::<T>(), self.meta().identifier());
        let data = *self.data.data_ptr();
        data.as_ptr().cast::<T>().add(index).write(value);
    }
}

impl Tables {
    #[inline]
    pub fn new() -> Self {
        Self {
            lock: RwLock::new(()),
            tables: Vec::new().into(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        let read = self.lock.read();
        let count = unsafe { &*self.tables.get() }.len();
        drop(read);
        count
    }

    #[inline]
    pub fn get(&self, index: usize) -> Result<&Table, Error> {
        let read = self.lock.read();
        let table = &**unsafe { &**self.tables.get() }
            .get(index)
            .ok_or(Error::MissingTable)?;
        drop(read);
        Ok(table)
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &Table {
        let read = self.lock.read();
        let tables = &**self.tables.get();
        debug_assert!(index < tables.len());
        let table = &**get_unchecked(tables, index);
        drop(read);
        table
    }

    #[inline]
    pub fn get_shared(&self, index: usize) -> Result<Arc<Table>, Error> {
        let read = self.lock.read();
        let table = unsafe { &**self.tables.get() }
            .get(index)
            .ok_or(Error::MissingTable)?
            .clone();
        drop(read);
        Ok(table)
    }

    #[inline]
    pub unsafe fn get_shared_unchecked(&self, index: usize) -> Arc<Table> {
        let read = self.lock.read();
        let tables = &**self.tables.get();
        debug_assert!(index < tables.len());
        let table = get_unchecked(tables, index).clone();
        drop(read);
        table
    }

    #[inline]
    pub(crate) fn find_or_add(&self, mut metas: Vec<&'static Meta>) -> Arc<Table> {
        let upgrade = self.lock.upgradable_read();
        // SAFETY: `self.tables` can be read since an upgrade lock is held. The lock will need to be upgraded
        // before any mutation to `self.tables`.
        let tables = unsafe { &*self.tables.get() };
        for table in tables.iter() {
            if table.indices.len() == metas.len()
                && metas.iter().all(|meta| table.has_with(meta.identifier()))
            {
                return table.clone();
            }
        }

        metas.sort_unstable_by_key(|meta| meta.identifier());
        let columns: Box<[Column]> = metas.into_iter().map(|meta| Column::new(meta, 0)).collect();
        let indices = columns
            .iter()
            .enumerate()
            .map(|(index, column)| (column.meta().identifier(), index))
            .collect();
        let index = tables.len();
        let inner = Inner {
            count: 0.into(),
            pending: 0.into(),
            keys: Vec::new().into(),
            columns,
        };

        let table = Arc::new(Table {
            index: index as _,
            indices,
            inner: RwLock::new(inner),
        });
        let write = RwLockUpgradableReadGuard::upgrade(upgrade);
        // SAFETY: The lock has been upgraded so `self.tables` can be mutated.
        unsafe { &mut *self.tables.get() }.push(table.clone());
        let read = RwLockWriteGuard::downgrade(write);
        let table = unsafe { get_unchecked(tables, index) }.clone();
        drop(read);
        table
    }
}

unsafe impl Send for Tables {}
unsafe impl Sync for Tables {}

impl<'a> IntoIterator for &'a Tables {
    type Item = &'a Table;
    type IntoIter = impl FullIterator<Item = Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let read = self.lock.read();
        unsafe { &**self.tables.get() }.iter().map(move |table| {
            // Keep the read guard alive.
            // SAFETY: Consumer of the iterator may keep references to tables since their address is guaranteed to remain stable.
            let _read = &read;
            &**table
        })
    }
}

impl Table {
    #[inline]
    pub const fn index(&self) -> u32 {
        self.index
    }

    #[inline]
    pub fn has<D: Datum>(&self) -> bool {
        self.has_with(TypeId::of::<D>())
    }

    #[inline]
    pub fn has_with(&self, identifier: TypeId) -> bool {
        self.indices.contains_key(&identifier)
    }

    #[inline]
    pub fn types(&self) -> impl ExactSizeIterator<Item = TypeId> + '_ {
        self.indices.keys().copied()
    }

    #[inline]
    pub fn column<D: Datum>(&self) -> Result<usize, Error> {
        self.column_with(TypeId::of::<D>())
    }

    #[inline]
    pub fn column_with(&self, identifier: TypeId) -> Result<usize, Error> {
        self.indices
            .get(&identifier)
            .copied()
            .ok_or(Error::MissingColumn)
    }
}

unsafe impl Send for Table {}
unsafe impl Sync for Table {}

impl Inner {
    #[inline]
    pub fn count(&self) -> usize {
        self.count.load(Ordering::Acquire) as _
    }

    #[inline]
    pub fn keys(&self) -> &[Key] {
        unsafe { get_unchecked(&**self.keys.get(), ..self.count()) }
    }

    #[inline]
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    #[inline]
    pub const fn decompose_pending(pending: u64) -> (u32, u32) {
        ((pending >> 32) as u32, pending as u32)
    }

    #[inline]
    pub const fn recompose_pending(begun: u32, ended: u32) -> u64 {
        ((begun as u64) << 32) | (ended as u64)
    }

    pub fn reserve<'a>(
        inner: RwLockUpgradableReadGuard<'a, Inner>,
        count: NonZeroUsize,
    ) -> (usize, RwLockReadGuard<'a, Inner>) {
        let (start, inner) = {
            let (start, _) = {
                let add = Self::recompose_pending(count.get() as _, 0);
                let pending = inner.pending.fetch_add(add, Ordering::AcqRel);
                Self::decompose_pending(pending)
            };
            // There can not be more than `u32::MAX` keys at a given time.
            assert!(start < u32::MAX - count.get() as u32);

            let old_capacity = unsafe { &*inner.keys.get() }.len();
            let new_capacity = count.saturating_add(start as _);
            let inner = if new_capacity.get() <= old_capacity {
                RwLockUpgradableReadGuard::downgrade(inner)
            } else {
                let mut inner = RwLockUpgradableReadGuard::upgrade(inner);
                let keys = inner.keys.get_mut();
                keys.resize(new_capacity.get(), Key::NULL);
                debug_assert_eq!(keys.len(), new_capacity.get());
                for column in inner.columns.iter_mut() {
                    // CHECK
                    unsafe { column.grow(old_capacity, new_capacity) };
                }
                RwLockWriteGuard::downgrade(inner)
            };
            (start as usize, inner)
        };
        (start, inner)
    }

    pub fn commit(&self, count: NonZeroUsize) {
        let add = Self::recompose_pending(0, count.get() as _);
        let pending = self.pending.fetch_add(add, Ordering::AcqRel);
        let (begun, ended) = Self::decompose_pending(pending);
        debug_assert!(begun > ended);
        if begun == ended + count.get() as u32 {
            // Only update `self.count` if it can be ensured that no other add operations are in progress.
            self.count.fetch_max(begun, Ordering::Relaxed);
        }
    }

    pub fn release(&mut self, count: NonZeroUsize) -> usize {
        let current = self.count.get_mut();
        let pending = self.pending.get_mut();
        let (begun, ended) = Self::decompose_pending(*pending);

        // Sanity checks. If this is not the case, there is a bug in the locking logic.
        debug_assert_eq!(begun, ended);
        debug_assert_eq!(begun, *current);
        debug_assert!(*current >= count.get() as u32);
        *current -= count.get() as u32;
        *pending = Self::recompose_pending(begun - count.get() as u32, ended - count.get() as u32);
        *current as usize
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let count = *self.count.get_mut() as usize;
        let capacity = self.keys.get_mut().len();
        debug_assert!(count <= capacity);
        for column in self.columns.iter_mut() {
            unsafe { column.free(count, capacity) };
        }
    }
}
