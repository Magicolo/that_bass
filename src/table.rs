use crate::{database::Database, key::Key, utility::FullIterator, Meta};
use parking_lot::{
    MappedRwLockReadGuard, MappedRwLockWriteGuard, RwLock, RwLockReadGuard,
    RwLockUpgradableReadGuard, RwLockWriteGuard,
};
use std::{
    any::TypeId,
    cell::UnsafeCell,
    collections::{HashMap, VecDeque},
    mem::ManuallyDrop,
    ptr::NonNull,
    slice::{from_raw_parts, from_raw_parts_mut, SliceIndex},
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
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
    pub(crate) count: AtomicU32,
    pub(crate) pending: AtomicU64,
    pub(crate) keys: UnsafeCell<Vec<Key>>,
    /// Stores are ordered consistently between tables.
    pub(crate) stores: Box<[Store]>,
}

pub struct Store {
    meta: &'static Meta,
    data: RwLock<NonNull<()>>,
}

pub struct Tables {
    /// The lock is seperated from the tables because once a table is dereferenced from the `tables` vector, it no longer
    /// needs to have its lifetime tied to a `RwLockReadGuard`. This is safe because the addresses of tables are stable
    /// (guaranteed by the `Arc` indirection) and no mutable references are ever given out.
    lock: RwLock<()>,
    tables: UnsafeCell<Vec<Arc<Table>>>,
}

pub(crate) struct TableRead<'a>(&'a Database, &'a Table, Option<RwLockReadGuard<'a, Inner>>);

pub(crate) struct TableUpgrade<'a>(
    &'a Database,
    &'a Table,
    Option<RwLockUpgradableReadGuard<'a, Inner>>,
);

pub(crate) struct TableWrite<'a>(&'a Database, &'a Table, Option<RwLockWriteGuard<'a, Inner>>);

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
    pub fn new(meta: &'static Meta, capacity: usize) -> Self {
        let data = if capacity == 0 {
            NonNull::dangling()
        } else {
            (meta.allocate)(capacity)
        };
        Self {
            meta,
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
        debug_assert_eq!(TypeId::of::<T>(), self.meta().identifier());
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
        debug_assert_eq!(TypeId::of::<T>(), self.meta().identifier());
        let data = self.data.try_read()?;
        Some(RwLockReadGuard::map(data, |data| unsafe {
            from_raw_parts(data.as_ptr().cast::<T>(), count).get_unchecked(index)
        }))
    }

    #[inline]
    pub unsafe fn get_unlocked_at<T: 'static>(&self, index: usize) -> &mut T {
        self.get_unlocked(index, index + 1)
    }

    #[inline]
    pub unsafe fn get_unlocked<T: 'static, I: SliceIndex<[T]>>(
        &self,
        index: I,
        count: usize,
    ) -> &mut I::Output {
        debug_assert_eq!(TypeId::of::<T>(), self.meta().identifier());
        let data = *self.data.data_ptr();
        from_raw_parts_mut(data.as_ptr().cast::<T>(), count).get_unchecked_mut(index)
    }

    #[inline]
    pub unsafe fn write<T: 'static, I: SliceIndex<[T]>>(
        &self,
        index: I,
        count: usize,
    ) -> MappedRwLockWriteGuard<I::Output> {
        debug_assert_eq!(TypeId::of::<T>(), self.meta().identifier());
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
        debug_assert_eq!(TypeId::of::<T>(), self.meta().identifier());
        let data = self.data.try_write()?;
        Some(RwLockWriteGuard::map(data, |data| unsafe {
            from_raw_parts_mut(data.as_ptr().cast::<T>(), count).get_unchecked_mut(index)
        }))
    }

    #[inline]
    pub unsafe fn set_unlocked_at<T: 'static>(&self, index: usize, value: T) {
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
    pub fn get(&self, index: usize) -> Option<&Table> {
        let read = self.lock.read();
        let table = &**unsafe { &**self.tables.get() }.get(index)?;
        drop(read);
        Some(table)
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &Table {
        let read = self.lock.read();
        let tables = &**self.tables.get();
        debug_assert!(index < tables.len());
        let table = &**tables.get_unchecked(index);
        drop(read);
        table
    }

    #[inline]
    pub fn get_shared(&self, index: usize) -> Option<Arc<Table>> {
        let read = self.lock.read();
        let table = unsafe { &**self.tables.get() }.get(index)?.clone();
        drop(read);
        Some(table)
    }

    #[inline]
    pub unsafe fn get_shared_unchecked(&self, index: usize) -> Arc<Table> {
        let read = self.lock.read();
        let tables = &**self.tables.get();
        debug_assert!(index < tables.len());
        let table = tables.get_unchecked(index).clone();
        drop(read);
        table
    }

    #[inline]
    pub(crate) fn find_or_add(&self, metas: Vec<&'static Meta>, capacity: usize) -> Arc<Table> {
        let upgrade = self.lock.upgradable_read();
        // SAFETY: `tables` can be read since an upgrade lock is held. The lock will need to be upgraded
        // before any mutation to `tables`.
        let tables = unsafe { &mut *self.tables.get() };
        for table in tables.iter() {
            if table.indices.len() == metas.len()
                && metas
                    .iter()
                    .all(|meta| table.indices.contains_key(&meta.identifier()))
            {
                return table.clone();
            }
        }

        let stores: Box<[Store]> = metas
            .into_iter()
            .map(|meta| Store::new(meta, capacity))
            .collect();
        let indices = stores
            .iter()
            .enumerate()
            .map(|(index, store)| (store.meta().identifier(), index))
            .collect();
        let index = tables.len();
        let inner = Inner {
            count: 0.into(),
            pending: 0.into(),
            keys: Vec::with_capacity(capacity).into(),
            stores,
        };
        let table = Arc::new(Table {
            index: index as _,
            indices,
            inner: RwLock::new(inner),
            defer: RwLock::new(VecDeque::new()),
        });
        let write = RwLockUpgradableReadGuard::upgrade(upgrade);
        // SAFETY: The lock has been upgraded before `tables` is mutated, which satisfy the requirement above.
        tables.push(table.clone());
        let read = RwLockWriteGuard::downgrade(write);
        let table = unsafe { tables.get_unchecked(index) }.clone();
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

impl<'a> TableRead<'a> {
    #[inline]
    pub(crate) fn new(
        database: &'a Database,
        table: &'a Table,
        guard: RwLockReadGuard<'a, Inner>,
    ) -> Self {
        Self(database, table, Some(guard))
    }

    #[inline]
    pub const fn database(&self) -> &'a Database {
        self.0
    }

    #[inline]
    pub const fn table(&self) -> &'a Table {
        self.1
    }

    #[inline]
    pub fn count(&self) -> u32 {
        self.inner().count()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner().capacity()
    }

    #[inline]
    pub fn keys(&self) -> &[Key] {
        self.inner().keys()
    }

    #[inline]
    pub fn stores(&self) -> &[Store] {
        self.inner().stores()
    }

    #[inline]
    pub(crate) fn inner(&self) -> &Inner {
        unsafe { self.2.as_deref().unwrap_unchecked() }
    }

    #[inline]
    pub(crate) fn forget(self) {
        drop(ManuallyDrop::new(self).2.take());
    }
}

impl<'a> TableUpgrade<'a> {
    #[inline]
    pub(crate) fn new(
        database: &'a Database,
        table: &'a Table,
        guard: RwLockUpgradableReadGuard<'a, Inner>,
    ) -> Self {
        Self(database, table, Some(guard))
    }

    #[inline]
    pub const fn database(&self) -> &'a Database {
        self.0
    }

    #[inline]
    pub const fn table(&self) -> &'a Table {
        self.1
    }

    #[inline]
    pub fn count(&self) -> u32 {
        self.inner().count()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner().capacity()
    }

    #[inline]
    pub fn keys(&self) -> &[Key] {
        self.inner().keys()
    }

    #[inline]
    pub fn stores(&self) -> &[Store] {
        self.inner().stores()
    }

    #[inline]
    pub(crate) fn inner(&self) -> &Inner {
        unsafe { self.2.as_deref().unwrap_unchecked() }
    }

    #[inline]
    pub fn upgrade(self) -> TableWrite<'a> {
        let mut guard = ManuallyDrop::new(self);
        TableWrite::new(
            guard.0,
            guard.1,
            RwLockUpgradableReadGuard::upgrade(unsafe { guard.2.take().unwrap_unchecked() }),
        )
    }

    #[inline]
    pub fn try_upgrade(self) -> Result<TableWrite<'a>, Self> {
        let mut guard = ManuallyDrop::new(self);
        match RwLockUpgradableReadGuard::try_upgrade(unsafe { guard.2.take().unwrap_unchecked() }) {
            Ok(write) => Ok(TableWrite::new(guard.0, guard.1, write)),
            Err(upgrade) => Err(Self::new(guard.0, guard.1, upgrade)),
        }
    }

    #[inline]
    pub fn downgrade(self) -> TableRead<'a> {
        let mut guard = ManuallyDrop::new(self);
        TableRead::new(
            guard.0,
            guard.1,
            RwLockUpgradableReadGuard::downgrade(unsafe { guard.2.take().unwrap_unchecked() }),
        )
    }

    #[inline]
    pub(crate) fn forget(self) {
        drop(ManuallyDrop::new(self).2.take());
    }
}

impl<'a> TableWrite<'a> {
    #[inline]
    pub(crate) fn new(
        database: &'a Database,
        table: &'a Table,
        guard: RwLockWriteGuard<'a, Inner>,
    ) -> Self {
        let mut guard = Self(database, table, Some(guard));
        guard.resolve();
        guard
    }

    #[inline]
    pub const fn database(&self) -> &'a Database {
        self.0
    }

    #[inline]
    pub const fn table(&self) -> &'a Table {
        self.1
    }

    #[inline]
    pub fn count(&self) -> u32 {
        self.inner().count()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner().capacity()
    }

    #[inline]
    pub fn keys(&self) -> &[Key] {
        self.inner().keys()
    }

    #[inline]
    pub fn stores(&self) -> &[Store] {
        self.inner().stores()
    }

    #[inline]
    pub(crate) fn inner(&self) -> &Inner {
        unsafe { self.2.as_deref().unwrap_unchecked() }
    }

    #[inline]
    pub(crate) fn inner_mut(&mut self) -> &mut Inner {
        unsafe { self.2.as_deref_mut().unwrap_unchecked() }
    }

    #[inline]
    pub fn downgrade(self) -> TableRead<'a> {
        let mut guard = ManuallyDrop::new(self);
        TableRead::new(
            guard.0,
            guard.1,
            RwLockWriteGuard::downgrade(unsafe { guard.2.take().unwrap_unchecked() }),
        )
    }

    #[inline]
    pub(crate) fn resolve(&mut self) {
        self.database().resolve_table_defers(self);
    }

    #[inline]
    pub(crate) fn forget(self) {
        drop(ManuallyDrop::new(self).2.take());
    }
}

impl<'a> Drop for TableRead<'a> {
    #[inline]
    fn drop(&mut self) {
        if match self.table().defer.try_read() {
            Some(defer) if defer.len() > 0 => true,
            None => true,
            _ => false,
        } {
            drop(self.2.take());
            self.0.table_try_write(self.1);
        }
    }
}

impl<'a> Drop for TableUpgrade<'a> {
    #[inline]
    fn drop(&mut self) {
        if match self.table().defer.try_read() {
            Some(defer) if defer.len() > 0 => true,
            None => true,
            _ => false,
        } {
            if let Some(upgrade) = self.2.take() {
                if let Ok(write) = RwLockUpgradableReadGuard::try_upgrade(upgrade) {
                    TableWrite::new(self.0, self.1, write);
                }
            }
        }
    }
}

impl<'a> Drop for TableWrite<'a> {
    #[inline]
    fn drop(&mut self) {
        self.resolve();
    }
}
