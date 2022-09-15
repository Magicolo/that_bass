use crate::{Datum, Meta};
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

pub struct Database(Arc<Inner>);

pub(crate) struct Inner {
    pub free: RwLock<(Vec<Key>, AtomicI64)>,
    pub slots: (RwLock<u32>, UnsafeCell<Vec<Box<[Slot; Self::CHUNK]>>>),
    pub tables: Vec<RwLock<Table>>,
}

pub(crate) struct Store {
    meta: Meta,
    data: RwLock<NonNull<()>>,
}

pub struct Table {
    count: AtomicU64,
    indices: HashMap<TypeId, usize>,
    keys: UnsafeCell<Vec<Key>>,
    /// Stores are ordered consistently between tables.
    stores: Box<[Store]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    index: u32,
    generation: u32,
}

pub(crate) struct Slot {
    generation: AtomicU32,
    indices: AtomicU64,
}

impl Key {
    pub const NULL: Self = Self {
        index: u32::MAX,
        generation: u32::MAX,
    };

    #[inline]
    pub(crate) const fn new(index: u32) -> Self {
        Self {
            index: index,
            generation: 0,
        }
    }
}

impl Slot {
    const fn recompose_indices(table: u32, store: u32) -> u64 {
        ((table as u64) << 32) | (store as u64)
    }

    const fn decompose_indices(indices: u64) -> (u32, u32) {
        ((indices >> 32) as u32, indices as u32)
    }

    #[inline]
    pub fn new(table: u32, store: u32) -> Self {
        let indices = AtomicU64::new(Self::recompose_indices(table, store));
        Self {
            generation: 0.into(),
            indices,
        }
    }

    #[inline]
    pub fn initialize(&self, generation: u32, table: u32, store: u32) {
        self.generation.store(generation, Release);
        self.update(table, store);
    }

    #[inline]
    pub fn update(&self, table: u32, store: u32) {
        let indices = Self::recompose_indices(table, store);
        self.indices.store(indices, Release);
    }

    #[inline]
    pub fn release(&self, generation: u32) -> Option<(u32, u32)> {
        self.generation
            .compare_exchange(generation, u32::MAX, AcqRel, Acquire)
            .ok()?;
        let indices = self.indices.swap(u64::MAX, Release);
        debug_assert!(indices < u64::MAX);
        Some(Self::decompose_indices(indices))
    }

    #[inline]
    pub fn generation(&self) -> u32 {
        self.generation.load(Acquire)
    }

    #[inline]
    pub fn indices(&self) -> (u32, u32) {
        Self::decompose_indices(self.indices.load(Acquire))
    }

    #[inline]
    pub fn valid_with(&self, generation: u32) -> bool {
        self.generation() == generation
    }
}

impl Default for Slot {
    fn default() -> Self {
        Self::new(u32::MAX, u32::MAX)
    }
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
    pub unsafe fn copy(source: (&mut Self, usize), target: (&Self, usize), count: usize) {
        debug_assert_eq!(source.0.meta.identifier, target.0.meta.identifier);
        (source.0.meta.copy)(
            (*source.0.data.get_mut(), source.1),
            (*target.0.data.data_ptr(), target.1),
            count,
        );
    }

    pub unsafe fn grow(&self, old_capacity: usize, new_capacity: usize) {
        debug_assert!(old_capacity < new_capacity);
        let mut data_write = self.data.write();
        let old_data = *data_write;
        let new_data = (self.meta.allocate)(new_capacity);
        (self.meta.copy)((old_data, 0), (new_data, 0), old_capacity);
        (self.meta.free)(old_data, 0, old_capacity);
        *data_write = new_data;
    }

    /// SAFETY: Both the 'source' and 'target' indices must be within the bounds of the store.
    /// The ranges 'source_index..source_index + count' and 'target_index..target_index + count' must not overlap.
    #[inline]
    pub unsafe fn squash(&mut self, source_index: usize, target_index: usize, count: usize) {
        let data = *self.data.get_mut();
        (self.meta.drop)(data, target_index, count);
        (self.meta.copy)((data, source_index), (data, target_index), count);
    }

    #[inline]
    pub unsafe fn drop(&mut self, index: usize, count: usize) {
        let data = *self.data.get_mut();
        (self.meta.drop)(data, index, count);
    }

    #[inline]
    pub unsafe fn read<T: 'static, I: SliceIndex<[T]>>(
        &self,
        index: I,
        count: usize,
    ) -> MappedRwLockReadGuard<I::Output> {
        debug_assert_eq!(TypeId::of::<T>(), self.meta.identifier);
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
        debug_assert_eq!(TypeId::of::<T>(), self.meta.identifier);
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
        debug_assert_eq!(TypeId::of::<T>(), self.meta.identifier);
        let data = *self.data.data_ptr();
        from_raw_parts(data.as_ptr().cast::<T>(), count).get_unchecked(index)
    }

    #[inline]
    pub unsafe fn write<T: 'static, I: SliceIndex<[T]>>(
        &self,
        index: I,
        count: usize,
    ) -> MappedRwLockWriteGuard<I::Output> {
        debug_assert_eq!(TypeId::of::<T>(), self.meta.identifier);
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
        debug_assert_eq!(TypeId::of::<T>(), self.meta.identifier);
        let data = self.data.try_write()?;
        Some(RwLockWriteGuard::map(data, |data| unsafe {
            from_raw_parts_mut(data.as_ptr().cast::<T>(), count).get_unchecked_mut(index)
        }))
    }

    #[inline]
    pub unsafe fn write_unlocked_at<T: 'static>(&self, index: usize) -> &mut T {
        self.write_unlocked(index, index + 1)
    }

    #[inline]
    pub unsafe fn write_unlocked<T: 'static, I: SliceIndex<[T]>>(
        &self,
        index: I,
        count: usize,
    ) -> &mut I::Output {
        debug_assert_eq!(TypeId::of::<T>(), self.meta.identifier);
        let data = *self.data.data_ptr();
        from_raw_parts_mut(data.as_ptr().cast::<T>(), count).get_unchecked_mut(index)
    }
}

impl Table {
    pub fn new(capacity: usize, metas: impl IntoIterator<Item = Meta>) -> Self {
        let mut set = HashSet::new();
        let mut stores: Box<[Store]> = metas
            .into_iter()
            .filter_map(|meta| {
                if set.insert(meta.identifier) {
                    Some(Store::new(meta, capacity))
                } else {
                    None
                }
            })
            .collect();
        stores.sort_unstable_by_key(|store| store.meta.identifier);
        let indices = stores
            .iter()
            .enumerate()
            .map(|(index, store)| (store.meta.identifier, index))
            .collect();
        Self {
            count: 0.into(),
            indices,
            keys: vec![Key::NULL; capacity].into(),
            stores,
        }
    }

    #[inline]
    pub fn count(&self) -> u32 {
        self.count.load(Acquire) as _
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        unsafe { &*self.keys.get() }.capacity()
    }

    #[inline]
    pub fn index(&self, identifier: TypeId) -> Option<usize> {
        self.indices.get(&identifier).copied()
    }

    #[inline]
    pub fn keys(&self) -> &[Key] {
        unsafe { (&*self.keys.get()).get_unchecked(0..self.count() as usize) }
    }

    pub fn grow(&mut self, capacity: u32) {
        let keys = self.keys.get_mut();
        let old_capacity = keys.capacity();
        keys.resize(capacity as _, Key::NULL);
        let new_capacity = keys.capacity();
        for store in self.stores.iter() {
            unsafe { store.grow(old_capacity, new_capacity) };
        }
    }
}

impl Inner {
    const SHIFT: usize = 8;
    const CHUNK: usize = 1 << Self::SHIFT;

    #[inline]
    const fn decompose_index(index: u32) -> (u32, u8) {
        (index >> Self::SHIFT, index as u8)
    }

    #[inline]
    const fn decompose_count(count: u64) -> (u16, u16, u32) {
        ((count >> 48) as u16, (count >> 32) as u16, count as u32)
    }

    #[inline]
    const fn recompose_count(begun: u16, ended: u16, count: u32) -> u64 {
        ((begun as u64) << 48) | ((ended as u64) << 32) | (count as u64)
    }

    pub fn new() -> Self {
        Self {
            free: RwLock::new((Vec::new(), 0.into())),
            slots: (RwLock::new(0), Vec::new().into()),
            tables: Vec::new(),
        }
    }

    pub fn slot(&self, key: Key) -> Option<&Slot> {
        let count_read = self.slots.0.read();
        let (chunk_index, slot_index) = Self::decompose_index(key.index);
        // SAFETY: `chunks` can be read since the `count_read` lock is held.
        let chunks = unsafe { &**self.slots.1.get() };
        let chunk = &**chunks.get(chunk_index as usize)?;
        // SAFETY: As soon as the `chunk` is dereferenced, the `count_read` lock is no longer needed.
        drop(count_read);
        let slot = chunk.get(slot_index as usize)?;
        if slot.generation() == key.generation {
            // SAFETY: A shared reference to a slot can be returned safely without being tied to the lifetime of the read guard
            // because its address is stable and no mutable reference to it is ever given out.
            // The stability of the address is guaranteed by the fact that the `chunks` vector never drops its items other than
            // when `self` is dropped.
            Some(slot)
        } else {
            None
        }
    }

    pub unsafe fn slot_unchecked(&self, key: Key) -> &Slot {
        // See `slot` for safety.
        let count_read = self.slots.0.read();
        let (chunk_index, slot_index) = Self::decompose_index(key.index);
        let chunks = &**self.slots.1.get();
        let chunk = &**chunks.get_unchecked(chunk_index as usize);
        drop(count_read);
        chunk.get_unchecked(slot_index as usize)
    }

    pub fn create(
        &self,
        table_index: u32,
        mut initialize: impl FnMut(usize, usize, &Table),
        keys: &mut [Key],
    ) -> Option<()> {
        let table = self.tables.get(table_index as usize)?;
        // Create in batches to give a chance to other threads to make progress.
        for keys in keys.chunks_mut(Self::CHUNK) {
            let key_count = keys.len() as u16;
            // Hold this lock until the operation is fully complete such that no move operation are interleaved.
            let table_read = table.upgradable_read();
            let (store_index, store_count, table_read) =
                Self::create_reserve(key_count, table_read);
            let mut done = 0;
            let free_read = self.free.read();
            let tail = free_read.1.fetch_sub(key_count as _, Relaxed);
            if tail > 0 {
                let tail = tail as usize;
                let count = tail.min(key_count as _);
                let head = tail - count;
                keys.copy_from_slice(&free_read.0[head..tail]);
                drop(free_read);

                let head = done;
                done += count;
                let tail = done;
                let keys = &keys[head..tail];
                unsafe {
                    (&mut *table_read.keys.get())
                        .get_unchecked_mut(store_index as usize..store_count)
                }
                .copy_from_slice(keys);
                initialize(store_index as _, count, &table_read);
            } else {
                drop(free_read);
            }

            if done < key_count as _ {
                // Since all indices use `u32` for compactness, this index must remain under `u32::MAX`.
                // Note that 'u32::MAX' is used as a sentinel so it must be an invalid entity index.
                let keys = &mut keys[done..];
                let index = self
                    .slot_reserve(keys.len() as u32)
                    .expect("Expected slot count to be `< u32::MAX`.");
                for (i, key) in keys.iter_mut().enumerate() {
                    *key = Key::new(index + i as u32);
                }

                let head = store_index as usize + done;
                unsafe { (&mut *table_read.keys.get()).get_unchecked_mut(head..store_count) }
                    .copy_from_slice(keys);
                initialize(head, store_count, &table_read);
            }

            // Initialize the slot only after the table row has been fully initialized.
            for &key in keys.iter() {
                let slot = unsafe { self.slot_unchecked(key) };
                slot.initialize(key.generation, table_index, store_index);
            }

            Self::create_resolve(key_count, table_read);
        }
        Some(())
    }

    /// Can be used to add or remove data associated with a key.
    pub fn modify(
        &self,
        key: Key,
        target_index: u32,
        mut initialize: impl FnMut(usize, usize, &Store),
    ) -> Option<()> {
        let target_table = self.tables.get(target_index as usize)?;
        loop {
            let slot = self.slot(key)?;
            let source_indices = slot.indices();
            if source_indices.0 == target_index {
                // No move is needed.
                break Some(());
            }
            let source_table = match self.tables.get(source_indices.0 as usize) {
                Some(table) => table,
                None => return None,
            };

            // Note that 2 very synchronized threads with their `source_table` and `target_table` swapped may
            // defeat this scheme for taking 2 write locks without dead locking. It is assumed that it doesn't
            // really happen in practice.
            let source_write = source_table.write();
            let (source_write, target_read) = match target_table.try_upgradable_read() {
                Some(target_read) => (source_write, target_read),
                None => {
                    drop(source_write);
                    let target_read = target_table.upgradable_read();
                    match source_table.try_write() {
                        Some(source_write) => (source_write, target_read),
                        None => continue,
                    }
                }
            };
            if source_indices != slot.indices() {
                continue;
            }

            let (last_index, mut source_write) = Self::destroy_reserve(source_write);
            let (store_index, store_count, target_read) = Self::create_reserve(1, target_read);
            let mut store_indices = (0, 0);

            fn drop_or_squash(source: u32, target: u32, store: &mut Store) {
                if source == target {
                    unsafe { store.drop(target as _, 1) };
                } else {
                    unsafe { store.squash(source as _, target as _, 1) };
                }
            }

            loop {
                match (
                    source_write.stores.get_mut(store_indices.0),
                    target_read.stores.get(store_indices.1),
                ) {
                    (Some(source_store), Some(target_store)) => {
                        let source_identifier = source_store.meta.identifier;
                        let target_identifier = target_store.meta.identifier;
                        if source_identifier == target_identifier {
                            store_indices.0 += 1;
                            store_indices.1 += 1;
                            unsafe {
                                Store::copy(
                                    (source_store, source_indices.1 as _),
                                    (target_store, store_index as _),
                                    1,
                                );
                            };
                            drop_or_squash(last_index, source_indices.1, source_store);
                        } else if source_identifier < target_identifier {
                            drop_or_squash(last_index, source_indices.1, source_store);
                            store_indices.0 += 1;
                        } else {
                            store_indices.1 += 1;
                            initialize(store_index as _, store_count, target_store);
                        }
                    }
                    (Some(source_store), None) => {
                        store_indices.0 += 1;
                        drop_or_squash(last_index, source_indices.1, source_store);
                    }
                    (None, Some(target_store)) => {
                        store_indices.1 += 1;
                        initialize(store_index as _, store_count, target_store);
                    }
                    (None, None) => break,
                }
            }

            if last_index == source_indices.1 {
                unsafe {
                    let keys = &mut *target_read.keys.get();
                    *keys.get_unchecked_mut(store_index as usize) = key;
                    self.slot_unchecked(key).update(target_index, store_index);
                }
            } else {
                let source_keys = source_write.keys.get_mut();
                unsafe {
                    let last_key = *source_keys.get_unchecked(last_index as usize);
                    let source_key = source_keys.get_unchecked_mut(source_indices.1 as usize);
                    let source_key = replace(source_key, last_key);

                    let target_keys = &mut *target_read.keys.get();
                    *target_keys.get_unchecked_mut(store_index as usize) = source_key;
                    self.slot_unchecked(source_key)
                        .update(target_index, store_index);
                    self.slot_unchecked(last_key)
                        .update(source_indices.0, source_indices.1);
                }
            }

            Self::create_resolve(1, target_read);
            drop(source_write);
            break Some(());
        }
    }

    pub fn destroy(&self, key: Key) -> Option<()> {
        let slot = self.slot(key)?;
        let (table_index, store_index) = slot.release(key.generation)?;
        let table = unsafe { self.tables.get_unchecked(table_index as usize) };
        let mut table_write = table.write();
        let last_index = {
            let table_count = table_write.count.get_mut();
            let (begun, ended, mut count) = Self::decompose_count(*table_count);
            // Sanity checks. If this is not the case, there is a bug in the locking logic.
            debug_assert_eq!(begun, 0u16);
            debug_assert_eq!(ended, 0u16);
            count -= 1;
            *table_count = Self::recompose_count(0, 0, count);
            count
        };

        if store_index == last_index {
            for store in table_write.stores.iter_mut() {
                unsafe { store.drop(store_index as _, 1) };
            }
        } else {
            for store in table_write.stores.iter_mut() {
                unsafe { store.squash(last_index as _, store_index as _, 1) };
            }

            let keys = table_write.keys.get_mut();
            unsafe {
                let last_key = *keys.get_unchecked(last_index as usize);
                *keys.get_unchecked_mut(store_index as usize) = last_key;
                self.slot_unchecked(last_key)
                    .update(table_index, store_index);
            }
        }

        drop(table_write);
        self.destroy_resolve(key);
        Some(())
    }

    fn slot_reserve(&self, count: u32) -> Option<u32> {
        let mut count_write = self.slots.0.write();
        let index = *count_write;
        let count = index.saturating_add(count);
        if count == u32::MAX {
            return None;
        }

        // SAFETY: `chunks` can be safely written to since the `count_write` lock is held.
        let chunks = unsafe { &mut *self.slots.1.get() };
        while count as usize > chunks.len() * Self::CHUNK {
            chunks.push(Box::new([(); Self::CHUNK].map(|_| Slot::default())));
        }
        *count_write = count;
        drop(count_write);
        Some(index)
    }

    fn create_reserve(
        reserve: u16,
        table_read: RwLockUpgradableReadGuard<Table>,
    ) -> (u32, usize, RwLockReadGuard<Table>) {
        let (begun, count) = {
            let add = Self::recompose_count(reserve, 0, 0);
            let count = table_read.count.fetch_add(add, AcqRel);
            let (begun, ended, count) = Self::decompose_count(count);
            debug_assert!(begun >= ended);
            (begun, count)
        };
        let store_index = count + begun as u32;
        let store_count = store_index as usize + reserve as usize;

        // There can not be more than `u32::MAX` keys at a given time.
        assert!(store_count < u32::MAX as _);
        let table_read = if store_count > table_read.capacity() {
            let mut table_write = RwLockUpgradableReadGuard::upgrade(table_read);
            table_write.grow(store_count as _);
            RwLockWriteGuard::downgrade(table_write)
        } else if begun >= u16::MAX >> 1 {
            // A huge stream of concurrent `create` operations has been detected; force the resolution of that table's `count`
            // before `begun` or `ended` overflows. This should essentially never happen.
            let mut table_write = RwLockUpgradableReadGuard::upgrade(table_read);
            let table_count = table_write.count.get_mut();
            let (begun, ended, count) = Self::decompose_count(*table_count);
            debug_assert_eq!(begun + reserve, ended);
            *table_count = Self::recompose_count(reserve, 0, count + ended as u32);
            RwLockWriteGuard::downgrade(table_write)
        } else {
            RwLockUpgradableReadGuard::downgrade(table_read)
        };
        (store_index, store_count, table_read)
    }

    fn create_resolve(reserve: u16, table_read: RwLockReadGuard<Table>) {
        table_read
            .count
            .fetch_update(AcqRel, Acquire, |count| {
                let (begun, ended, count) = Self::decompose_count(count);
                if begun == ended + reserve {
                    Some(Self::recompose_count(0, 0, count + begun as u32))
                } else if begun > ended {
                    Some(Self::recompose_count(begun, ended + reserve, count))
                } else {
                    // If this happens, this is a bug. The `expect` below will panic.
                    None
                }
            })
            .expect("Expected the updating the count to succeed.");
    }

    fn destroy_reserve(mut table_write: RwLockWriteGuard<Table>) -> (u32, RwLockWriteGuard<Table>) {
        let index = {
            let table_count = table_write.count.get_mut();
            let (begun, ended, mut count) = Self::decompose_count(*table_count);
            // Sanity checks. If this is not the case, there is a bug in the locking logic.
            debug_assert_eq!(begun, 0u16);
            debug_assert_eq!(ended, 0u16);
            count -= 1;
            *table_count = Self::recompose_count(0, 0, count);
            count
        };
        (index, table_write)
    }

    fn destroy_resolve(&self, mut key: Key) {
        key.generation = key.generation.saturating_add(1);
        if key.generation < u32::MAX {
            let mut free_write = self.free.write();
            let free_count = *free_write.1.get_mut();
            free_write.0.truncate(free_count.max(0) as _);
            free_write.0.push(key);
            *free_write.1.get_mut() = free_write.0.len() as _;
            drop(free_write);
        }
    }
}

impl Database {
    pub fn new() -> Self {
        Self(Arc::new(Inner::new()))
    }
}
