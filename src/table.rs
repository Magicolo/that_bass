use crate::{key::Key, utility::FullIterator, Meta};
use parking_lot::{
    MappedRwLockReadGuard, MappedRwLockWriteGuard, RwLock, RwLockReadGuard,
    RwLockUpgradableReadGuard, RwLockWriteGuard,
};
use std::{
    any::TypeId,
    cell::UnsafeCell,
    collections::{HashSet, VecDeque},
    iter::FusedIterator,
    ptr::NonNull,
    slice::{from_raw_parts, from_raw_parts_mut, SliceIndex},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

pub struct Table {
    index: u32,
    types: HashSet<TypeId>,
    pub(crate) inner: RwLock<Inner>,
    pub(crate) defer: RwLock<VecDeque<Defer>>,
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
    lock: RwLock<()>,
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
            lock: RwLock::new(()),
            tables: Vec::new().into(),
        }
    }

    #[inline]
    pub fn iter(&self) -> impl FullIterator<Item = &Table> {
        struct Iterate<'a, I: FullIterator<Item = &'a Table>>(I, RwLockReadGuard<'a, ()>);
        impl<'a, I: FullIterator<Item = &'a Table>> Iterator for Iterate<'a, I> {
            type Item = &'a Table;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                self.0.next()
            }
        }
        impl<'a, I: FullIterator<Item = &'a Table>> ExactSizeIterator for Iterate<'a, I> {
            #[inline]
            fn len(&self) -> usize {
                self.0.len()
            }
        }
        impl<'a, I: FullIterator<Item = &'a Table>> DoubleEndedIterator for Iterate<'a, I> {
            #[inline]
            fn next_back(&mut self) -> Option<Self::Item> {
                self.0.next_back()
            }
        }
        impl<'a, I: FullIterator<Item = &'a Table>> FusedIterator for Iterate<'a, I> {}

        let read = self.lock.read();
        let tables = unsafe { &**self.tables.get() };
        Iterate(tables.iter().map(|table| &**table), read)
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
        let table = &**(&**self.tables.get()).get_unchecked(index);
        drop(read);
        table
    }

    #[inline]
    pub fn find_or_add(
        &self,
        metas: Vec<Meta>,
        types: HashSet<TypeId>,
        capacity: usize,
    ) -> Arc<Table> {
        let upgrade = self.lock.upgradable_read();
        let tables = unsafe { &mut *self.tables.get() };

        match tables.iter().find(|table| table.types == types) {
            Some(table) => table.clone(),
            None => {
                let stores: Box<[Store]> = metas
                    .into_iter()
                    .map(|meta| Store::new(meta, capacity))
                    .collect();
                let index = tables.len() as _;
                let inner = Inner {
                    count: 0.into(),
                    keys: vec![Key::NULL; capacity].into(),
                    stores,
                };
                let table = Arc::new(Table {
                    index,
                    types,
                    inner: RwLock::new(inner),
                    defer: RwLock::new(VecDeque::new()),
                });
                let write = RwLockUpgradableReadGuard::upgrade(upgrade);
                tables.push(table.clone());
                drop(write);
                table
            }
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
    pub fn has(&self, identifier: TypeId) -> bool {
        self.types.contains(&identifier)
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
        unsafe { &*self.keys.get() }.capacity()
    }

    #[inline]
    pub fn keys(&self) -> &[Key] {
        unsafe { (&*self.keys.get()).get_unchecked(0..self.count() as usize) }
    }

    #[inline]
    pub const fn stores(&self) -> &[Store] {
        &self.stores
    }

    pub fn grow(&mut self, capacity: usize) {
        let keys = self.keys.get_mut();
        let old_capacity = keys.capacity();
        if old_capacity < capacity {
            keys.resize(capacity as _, Key::NULL);
            let new_capacity = keys.capacity();
            for store in self.stores.iter_mut() {
                unsafe { store.grow(old_capacity, new_capacity) };
            }
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let count = *self.count.get_mut() as u32 as usize;
        let capacity = self.keys.get_mut().len();
        for store in self.stores.iter_mut() {
            unsafe { store.free(count, capacity) };
        }
    }
}
