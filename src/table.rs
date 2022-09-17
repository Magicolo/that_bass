use crate::{key::Key, utility::FullIterator, Meta};
use parking_lot::{
    MappedRwLockReadGuard, MappedRwLockWriteGuard, RwLock, RwLockReadGuard,
    RwLockUpgradableReadGuard, RwLockWriteGuard,
};
use std::{
    any::TypeId,
    cell::UnsafeCell,
    collections::HashSet,
    iter::FusedIterator,
    ops::Deref,
    ptr::NonNull,
    slice::{from_raw_parts, from_raw_parts_mut, Iter, SliceIndex},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

pub struct Table {
    index: u32,
    types: HashSet<TypeId>,
    pub(crate) count: AtomicU64,
    pub(crate) keys: UnsafeCell<Vec<Key>>,
    /// Stores are ordered consistently between tables.
    pub(crate) stores: Box<[Store]>,
}

pub(crate) struct Store {
    meta: Meta,
    data: RwLock<NonNull<()>>,
}

pub(crate) struct Tables(RwLock<()>, UnsafeCell<Vec<Arc<RwLock<Table>>>>);

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
        (source.0.meta().copy)(
            (*source.0.data.get_mut(), source.1),
            (*target.0.data.data_ptr(), target.1),
            count,
        );
    }

    pub unsafe fn grow(&self, old_capacity: usize, new_capacity: usize) {
        debug_assert!(old_capacity < new_capacity);
        let mut data_write = self.data.write();
        let old_data = *data_write;
        let new_data = (self.meta().allocate)(new_capacity);
        (self.meta().copy)((old_data, 0), (new_data, 0), old_capacity);
        (self.meta().free)(old_data, 0, old_capacity);
        *data_write = new_data;
    }

    /// SAFETY: Both the 'source' and 'target' indices must be within the bounds of the store.
    /// The ranges 'source_index..source_index + count' and 'target_index..target_index + count' must not overlap.
    #[inline]
    pub unsafe fn squash(&mut self, source_index: usize, target_index: usize, count: usize) {
        let data = *self.data.get_mut();
        (self.meta().drop)(data, target_index, count);
        (self.meta().copy)((data, source_index), (data, target_index), count);
    }

    #[inline]
    pub unsafe fn drop(&mut self, index: usize, count: usize) {
        let data = *self.data.get_mut();
        (self.meta().drop)(data, index, count);
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
        Self(RwLock::new(()), Vec::new().into())
    }

    pub fn iter(&self) -> impl FullIterator<Item = impl Deref<Target = Table> + '_> {
        struct Iterate<'a>(Iter<'a, Arc<RwLock<Table>>>, RwLockReadGuard<'a, ()>);
        impl<'a> Iterator for Iterate<'a> {
            type Item = RwLockReadGuard<'a, Table>;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                Some(self.0.next()?.read())
            }
        }
        impl<'a> ExactSizeIterator for Iterate<'a> {
            #[inline]
            fn len(&self) -> usize {
                self.0.len()
            }
        }
        impl<'a> DoubleEndedIterator for Iterate<'a> {
            #[inline]
            fn next_back(&mut self) -> Option<Self::Item> {
                Some(self.0.next_back()?.read())
            }
        }
        impl<'a> FusedIterator for Iterate<'a> {}

        let read = self.0.read();
        let tables = unsafe { &**self.1.get() };
        Iterate(tables.iter(), read)
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<&RwLock<Table>> {
        let read = self.0.read();
        let table = &**unsafe { &**self.1.get() }.get(index)?;
        drop(read);
        Some(table)
    }

    #[inline]
    pub fn get_shared(&self, index: usize) -> Option<Arc<RwLock<Table>>> {
        let read = self.0.read();
        let table = unsafe { &**self.1.get() }.get(index)?.clone();
        drop(read);
        Some(table)
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &RwLock<Table> {
        let read = self.0.read();
        let table = &**(&**self.1.get()).get_unchecked(index);
        drop(read);
        table
    }

    #[inline]
    pub unsafe fn get_shared_unchecked(&self, index: usize) -> Arc<RwLock<Table>> {
        let read = self.0.read();
        let table = (&**self.1.get()).get_unchecked(index).clone();
        drop(read);
        table
    }

    #[inline]
    pub fn find_or_add(
        &self,
        metas: Vec<Meta>,
        types: HashSet<TypeId>,
        capacity: usize,
    ) -> Arc<RwLock<Table>> {
        let upgrade = self.0.upgradable_read();
        let tables = unsafe { &mut *self.1.get() };

        match tables.iter().find(|table| table.read().types == types) {
            Some(table) => table.clone(),
            None => {
                let stores: Box<[Store]> = metas
                    .into_iter()
                    .map(|meta| Store::new(meta, capacity))
                    .collect();
                let index = tables.len() as _;
                let table = Arc::new(RwLock::new(Table {
                    index,
                    count: 0.into(),
                    types,
                    keys: vec![Key::NULL; capacity].into(),
                    stores,
                }));
                let write = RwLockUpgradableReadGuard::upgrade(upgrade);
                tables.push(table.clone());
                let read = RwLockWriteGuard::downgrade(write);
                let tables = &**tables;
                drop(read);
                table
            }
        }
    }
}

impl Table {
    #[inline]
    pub const fn index(&self) -> u32 {
        self.index
    }

    #[inline]
    pub fn count(&self) -> u32 {
        self.count.load(Ordering::Acquire) as _
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        unsafe { &*self.keys.get() }.capacity()
    }

    #[inline]
    pub fn has(&self, identifier: TypeId) -> bool {
        self.types.contains(&identifier)
    }

    #[inline]
    pub fn keys(&self) -> &[Key] {
        unsafe { (&*self.keys.get()).get_unchecked(0..self.count() as usize) }
    }

    #[inline]
    pub fn metas(&self) -> impl FullIterator<Item = &Meta> {
        self.stores.iter().map(|store| store.meta())
    }

    #[inline]
    pub const fn stores(&self) -> &[Store] {
        &self.stores
    }

    pub fn grow(&mut self, capacity: u32) {
        let keys = self.keys.get_mut();
        let old_capacity = keys.capacity();
        keys.resize(capacity as _, Key::NULL);
        let new_capacity = keys.capacity();
        if old_capacity < new_capacity {
            for store in self.stores.iter() {
                unsafe { store.grow(old_capacity, new_capacity) };
            }
        }
    }
}
