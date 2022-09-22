use crate::{key::Key, utility::FullIterator, Meta};
use parking_lot::{
    MappedRwLockReadGuard, MappedRwLockWriteGuard, RwLock, RwLockReadGuard,
    RwLockUpgradableReadGuard, RwLockWriteGuard,
};
use std::{
    any::TypeId,
    cell::UnsafeCell,
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    ptr::NonNull,
    slice::{from_raw_parts, from_raw_parts_mut, SliceIndex},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

pub struct Table {
    index: u32,
    indices: HashMap<TypeId, usize>,
    pub(crate) defer: RwLock<VecDeque<Defer>>,
    pub(crate) inner: RwLock<Inner>,
}

pub(crate) struct Inner {
    pub(crate) count: AtomicU64,
    pub(crate) keys: UnsafeCell<Vec<Key>>,
    /// Stores are ordered consistently between tables.
    pub(crate) stores: Box<[Store]>,
}

pub struct Store {
    meta: Meta,
    data: RwLock<NonNull<()>>,
}

pub(crate) struct Tables {
    /// The lock is seperated from the tables because once a table is dereferenced from the `tables` vector, it no longer
    /// needs to have its lifetime tied to a `RwLockReadGuard`. This is safe because the addresses of tables are stable
    /// (guaranteed by the `Arc` indirection) and no mutable references are ever given out.
    indices: RwLock<BTreeMap<BTreeSet<TypeId>, usize>>,
    tables: UnsafeCell<Vec<Arc<Table>>>,
}

pub(crate) enum Defer {
    Add {
        keys: Box<[Key]>,
        row_index: u32,
        row_count: usize,
        initialize: Box<dyn FnMut(usize, &[Store])>,
    },
    Remove {
        row_index: u32,
    },
}

impl Store {
    pub fn new(meta: Meta, capacity: usize) -> Self {
        let data = if capacity == 0 {
            NonNull::dangling()
        } else {
            (meta.allocate)(capacity)
        };
        Self {
            meta: meta,
            data: RwLock::new(data),
        }
    }

    #[inline]
    pub const fn meta(&self) -> &Meta {
        &self.meta
    }

    #[inline]
    pub unsafe fn copy(source: (&mut Self, usize), target: (&Self, usize), count: usize) {
        debug_assert_eq!(source.0.meta().identifier, target.0.meta().identifier);
        let &Meta { copy, .. } = source.0.meta();
        copy(
            (*source.0.data.get_mut(), source.1),
            (*target.0.data.data_ptr(), target.1),
            count,
        );
    }

    pub unsafe fn grow(&mut self, old_capacity: usize, new_capacity: usize) {
        debug_assert!(old_capacity < new_capacity);
        let &Meta {
            allocate,
            free,
            copy,
            ..
        } = self.meta();
        let data = self.data.get_mut();
        let old_data = *data;
        let new_data = allocate(new_capacity);
        copy((old_data, 0), (new_data, 0), old_capacity);
        free(old_data, 0, old_capacity);
        *data = new_data;
    }

    /// SAFETY: Both the 'source' and 'target' indices must be within the bounds of the store.
    /// The ranges 'source_index..source_index + count' and 'target_index..target_index + count' must not overlap.
    #[inline]
    pub unsafe fn squash(&mut self, source_index: usize, target_index: usize, count: usize) {
        let &Meta { copy, drop, .. } = self.meta();
        let data = *self.data.get_mut();
        drop(data, target_index, count);
        copy((data, source_index), (data, target_index), count);
    }

    #[inline]
    pub unsafe fn drop(&mut self, index: usize, count: usize) {
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
    pub unsafe fn read<T: 'static, I: SliceIndex<[T]>>(
        &self,
        index: I,
        count: usize,
    ) -> MappedRwLockReadGuard<I::Output> {
        debug_assert_eq!(TypeId::of::<T>(), self.meta().identifier);
        RwLockReadGuard::map(self.data.read(), |data| unsafe {
            from_raw_parts(data.as_ptr().cast::<T>(), count).get_unchecked(index)
        })
    }

    #[inline]
    pub unsafe fn try_read<T: 'static, I: SliceIndex<[T]>>(
        &self,
        index: I,
        count: usize,
    ) -> Option<MappedRwLockReadGuard<I::Output>> {
        debug_assert_eq!(TypeId::of::<T>(), self.meta().identifier);
        let data = self.data.try_read()?;
        Some(RwLockReadGuard::map(data, |data| unsafe {
            from_raw_parts(data.as_ptr().cast::<T>(), count).get_unchecked(index)
        }))
    }

    #[inline]
    pub unsafe fn read_unlocked_at<T: 'static>(&self, index: usize) -> &T {
        self.read_unlocked(index, index + 1)
    }

    #[inline]
    pub unsafe fn read_unlocked<T: 'static, I: SliceIndex<[T]>>(
        &self,
        index: I,
        count: usize,
    ) -> &I::Output {
        debug_assert_eq!(TypeId::of::<T>(), self.meta().identifier);
        let data = *self.data.data_ptr();
        from_raw_parts(data.as_ptr().cast::<T>(), count).get_unchecked(index)
    }

    #[inline]
    pub unsafe fn write<T: 'static, I: SliceIndex<[T]>>(
        &self,
        index: I,
        count: usize,
    ) -> MappedRwLockWriteGuard<I::Output> {
        debug_assert_eq!(TypeId::of::<T>(), self.meta().identifier);
        RwLockWriteGuard::map(self.data.write(), |data| unsafe {
            from_raw_parts_mut(data.as_ptr().cast::<T>(), count).get_unchecked_mut(index)
        })
    }

    #[inline]
    pub unsafe fn write_at<T: 'static>(&self, index: usize) -> MappedRwLockWriteGuard<T> {
        self.write(index, index + 1)
    }

    #[inline]
    pub unsafe fn write_all<T: 'static>(&self, count: usize) -> MappedRwLockWriteGuard<[T]> {
        self.write(.., count)
    }

    #[inline]
    pub unsafe fn try_write<T: 'static, I: SliceIndex<[T]>>(
        &self,
        index: I,
        count: usize,
    ) -> Option<MappedRwLockWriteGuard<I::Output>> {
        debug_assert_eq!(TypeId::of::<T>(), self.meta().identifier);
        let data = self.data.try_write()?;
        Some(RwLockWriteGuard::map(data, |data| unsafe {
            from_raw_parts_mut(data.as_ptr().cast::<T>(), count).get_unchecked_mut(index)
        }))
    }

    #[inline]
    pub unsafe fn write_unlocked_at<T: 'static>(&self, index: usize, value: T) {
        let data = *self.data.data_ptr();
        data.as_ptr().cast::<T>().add(index).write(value);
    }
}

impl Tables {
    pub fn new() -> Self {
        Self {
            indices: RwLock::new(BTreeMap::new()),
            tables: Vec::new().into(),
        }
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<&Table> {
        let indices_read = self.indices.read();
        let table = &**unsafe { &**self.tables.get() }.get(index)?;
        drop(indices_read);
        Some(table)
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &Table {
        let indices_read = self.indices.read();
        let tables = &**self.tables.get();
        debug_assert!(index < tables.len());
        let table = &**tables.get_unchecked(index);
        drop(indices_read);
        table
    }

    #[inline]
    pub fn get_shared(&self, index: usize) -> Option<Arc<Table>> {
        let indices_read = self.indices.read();
        let table = unsafe { &**self.tables.get() }.get(index)?.clone();
        drop(indices_read);
        Some(table)
    }

    #[inline]
    pub unsafe fn get_shared_unchecked(&self, index: usize) -> Arc<Table> {
        let indices_read = self.indices.read();
        let tables = &**self.tables.get();
        debug_assert!(index < tables.len());
        let table = tables.get_unchecked(index).clone();
        drop(indices_read);
        table
    }

    pub fn find(&self, types: &BTreeSet<TypeId>) -> Option<Arc<Table>> {
        let indices_read = self.indices.read();
        let &index = indices_read.get(types)?;
        // SAFETY: A read lock is held.
        let table = unsafe { (&*self.tables.get()).get_unchecked(index) }.clone();
        drop(indices_read);
        Some(table)
    }

    #[inline]
    pub fn find_or_add(
        &self,
        metas: impl IntoIterator<Item = Meta>,
        types: BTreeSet<TypeId>,
        capacity: usize,
    ) -> Arc<Table> {
        let indices_upgrade = self.indices.upgradable_read();
        // SAFETY: An upgrade lock is held.
        match indices_upgrade.get(&types) {
            Some(&index) => unsafe { (&*self.tables.get()).get_unchecked(index) }.clone(),
            None => {
                // SAFETY: `tables` can be read since an upgrade lock is held. The lock will need to be upgraded
                // before any mutation to `tables`.
                let tables = unsafe { &mut *self.tables.get() };
                let stores: Box<[Store]> = metas
                    .into_iter()
                    .map(|meta| Store::new(meta, capacity))
                    .collect();
                let indices = stores
                    .iter()
                    .enumerate()
                    .map(|(index, store)| (store.meta().identifier(), index))
                    .collect();
                let index = tables.len() as _;
                let inner = Inner {
                    count: 0.into(),
                    keys: Vec::with_capacity(capacity).into(),
                    stores,
                };
                let table = Arc::new(Table {
                    index,
                    indices,
                    inner: RwLock::new(inner),
                    defer: RwLock::new(VecDeque::new()),
                });
                let mut indices_write = RwLockUpgradableReadGuard::upgrade(indices_upgrade);
                indices_write.insert(types, tables.len());
                // SAFETY: The lock has been upgraded before `tables` is mutated, which satisfy the requirement above.
                tables.push(table.clone());
                drop(indices_write);
                table
            }
        }
    }
}

unsafe impl Send for Tables {}
unsafe impl Sync for Tables {}

impl<'a> IntoIterator for &'a Tables {
    type Item = &'a Table;
    type IntoIter = impl FullIterator<Item = Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let read = self.indices.read();
        let tables = unsafe { &**self.tables.get() };
        tables.iter().map(move |table| {
            // Keep the read guard alive.
            let _ = &read;
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
    pub fn has(&self, identifier: TypeId) -> bool {
        self.indices.contains_key(&identifier)
    }

    #[inline]
    pub fn store(&self, identifier: TypeId) -> Option<usize> {
        self.indices.get(&identifier).copied()
    }

    #[inline]
    pub fn stores(&self) -> usize {
        self.indices.len()
    }

    #[inline]
    pub(crate) fn defer(&self, defer: Defer) {
        self.defer.write().push_back(defer);
    }
}

unsafe impl Send for Table {}
unsafe impl Sync for Table {}

impl Inner {
    #[inline]
    pub fn count(&self) -> u32 {
        self.count.load(Ordering::Acquire) as _
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        unsafe { &*self.keys.get() }.len()
    }

    #[inline]
    pub fn keys(&self) -> &[Key] {
        let count = self.count() as usize;
        unsafe {
            let keys = &*self.keys.get();
            debug_assert!(count <= keys.len());
            keys.get_unchecked(..count)
        }
    }

    #[inline]
    pub fn stores(&self) -> &[Store] {
        &self.stores
    }

    pub fn grow(&mut self, capacity: usize) {
        let keys = self.keys.get_mut();
        if keys.len() < capacity {
            let old_capacity = keys.len();
            keys.resize(capacity, Key::NULL);
            keys.resize(keys.capacity(), Key::NULL);
            debug_assert_eq!(keys.len(), keys.capacity());
            let new_capacity = keys.len();
            for store in self.stores.iter_mut() {
                unsafe { store.grow(old_capacity, new_capacity) };
            }
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let count = self.count() as usize;
        let capacity = self.capacity();
        debug_assert!(count <= capacity);
        for store in self.stores.iter_mut() {
            unsafe { store.free(count, capacity) };
        }
    }
}
