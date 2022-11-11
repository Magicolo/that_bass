use crate::{
    core::{iterate::FullIterator, utility::get_unchecked},
    key::Key,
    Datum, Error, Meta,
};
use parking_lot::{RwLock, RwLockReadGuard, RwLockUpgradableReadGuard, RwLockWriteGuard};
use std::{
    any::TypeId,
    cell::UnsafeCell,
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
    metas: Box<[&'static Meta]>,
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

    pub unsafe fn shrink(&mut self, old_capacity: NonZeroUsize, new_capacity: usize) {
        debug_assert!(old_capacity.get() > new_capacity);
        let &Meta {
            new,
            free,
            copy,
            drop,
            ..
        } = self.meta();
        let data = self.data.get_mut();
        let old_data = *data;
        let new_data = new(new_capacity);
        if let Some(new_capacity) = NonZeroUsize::new(new_capacity) {
            copy((old_data, 0), (new_data, 0), new_capacity);
        }
        if let Some(over) = NonZeroUsize::new(old_capacity.get() - new_capacity) {
            drop.1(old_data, new_capacity, over);
        }
        // A count of 0 is sent to `free` because the values of `old_data` have been moved to `new_data`, so they must not be dropped.
        free(old_data, 0, old_capacity.get());
        *data = new_data;
    }

    /// SAFETY: Both the 'source' and 'target' indices must be within the bounds of the column.
    /// The ranges 'source_index..source_index + count' and 'target_index..target_index + count' must not overlap.
    #[inline]
    pub unsafe fn squash(&mut self, source_index: usize, target_index: usize, count: NonZeroUsize) {
        let &Meta { copy, drop, .. } = self.meta();
        let data = *self.data.get_mut();
        drop.1(data, target_index, count);
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
        drop.1(data, index, count);
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
    pub fn iter(&self) -> impl FullIterator<Item = &Table> {
        let read = self.lock.read();
        unsafe { &**self.tables.get() }.iter().map(move |table| {
            // Keep the read guard alive.
            // SAFETY: Consumer of the iterator may keep references to tables since their address is guaranteed to remain stable.
            let _read = &read;
            &**table
        })
    }

    #[inline]
    pub fn get(&self, index: usize) -> Result<&Table, Error> {
        let read = self.lock.read();
        let table = &**unsafe { &**self.tables.get() }
            .get(index)
            .ok_or(Error::MissingTable(index))?;
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
            .ok_or(Error::MissingTable(index))?
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

    /// `metas` must be sorted by `meta.identifier()` and must be deduplicated.
    #[inline]
    pub(crate) fn find_or_add(&self, metas: &[&'static Meta]) -> Arc<Table> {
        // Verifies that `metas` is sorted and deduplicated.
        debug_assert!(metas
            .windows(2)
            .all(|metas| metas[0].identifier() < metas[1].identifier()));

        let upgrade = self.lock.upgradable_read();
        // SAFETY: `self.tables` can be read since an upgrade lock is held. The lock will need to be upgraded
        // before any mutation to `self.tables`.
        let tables = unsafe { &*self.tables.get() };
        for table in tables.iter() {
            if table.is_all(metas.iter().map(|meta| meta.identifier())) {
                return table.clone();
            }
        }

        let columns = metas.iter().map(|&meta| Column::new(meta, 0)).collect();
        let index = tables.len();
        let inner = Inner {
            count: 0.into(),
            pending: 0.into(),
            keys: Vec::new().into(),
            columns,
        };
        let table = Arc::new(Table {
            index: index as _,
            metas: metas.iter().copied().collect(),
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

    pub fn shrink(&self) {
        for table in self.iter() {
            table.inner.write().shrink();
        }
    }
}

unsafe impl Send for Tables {}
unsafe impl Sync for Tables {}

impl Table {
    #[inline]
    pub const fn index(&self) -> u32 {
        self.index
    }

    #[inline]
    pub fn metas(&self) -> &[&'static Meta] {
        &self.metas
    }

    #[inline]
    pub fn has<D: Datum>(&self) -> bool {
        self.has_with(TypeId::of::<D>())
    }

    #[inline]
    pub fn has_with(&self, identifier: TypeId) -> bool {
        self.metas
            .binary_search_by_key(&identifier, |meta| meta.identifier())
            .is_ok()
    }

    /// `types` must be ordered and deduplicated.
    pub(crate) fn is_all(&self, types: impl IntoIterator<Item = TypeId>) -> bool {
        let mut types = types.into_iter();
        for meta in self.metas() {
            if let Some(identifier) = types.next() {
                if meta.identifier() == identifier {
                    continue;
                }
            }
            return false;
        }
        return types.next().is_none();
    }

    /// `types` must be ordered and deduplicated.
    pub(crate) fn has_all(&self, types: impl IntoIterator<Item = TypeId>) -> bool {
        let mut types = types.into_iter();
        for meta in self.metas() {
            while let Some(identifier) = types.next() {
                if meta.identifier() == identifier {
                    continue;
                }
            }
            return false;
        }
        return true;
    }

    pub(crate) fn column<D: Datum>(&self) -> Result<(usize, &'static Meta), Error> {
        self.column_with(TypeId::of::<D>())
    }

    pub(crate) fn column_with(&self, identifier: TypeId) -> Result<(usize, &'static Meta), Error> {
        let index = self
            .metas
            .binary_search_by_key(&identifier, |meta| meta.identifier())
            .map_err(|_| Error::MissingColumn(identifier))?;
        Ok((index, self.metas[index]))
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

    pub fn shrink(&mut self) {
        let keys = self.keys.get_mut();
        let Some(old_capacity) = NonZeroUsize::new(keys.len()) else {
            return;
        };
        let new_capacity = *self.count.get_mut() as usize;
        if new_capacity == old_capacity.get() {
            return;
        }
        debug_assert!(new_capacity <= old_capacity.get());

        keys.truncate(new_capacity as _);
        keys.shrink_to_fit();
        debug_assert_eq!(keys.len(), new_capacity);

        let new_capacity = keys.len();
        for column in self.columns.iter_mut() {
            unsafe { column.shrink(old_capacity, new_capacity) };
        }
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
