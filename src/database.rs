use crate::{
    identify,
    key::Key,
    table::{Store, Table, Tables},
    utility::FullIterator,
};
use parking_lot::{RwLock, RwLockReadGuard, RwLockUpgradableReadGuard, RwLockWriteGuard};
use std::{
    cell::UnsafeCell,
    mem::replace,
    ops::Deref,
    sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering::*},
};

pub struct Database(Inner);

pub(crate) struct Inner {
    pub identifier: usize,
    pub free: RwLock<(Vec<Key>, AtomicI64)>,
    pub slots: Slots,
    pub tables: Tables,
}

pub(crate) struct Slot {
    generation: AtomicU32,
    indices: AtomicU64,
}

pub(crate) struct Slots(RwLock<u32>, UnsafeCell<Vec<Box<[Slot; Self::CHUNK]>>>);

impl Database {
    pub fn new() -> Self {
        Self(Inner::new())
    }

    pub fn tables(&self) -> impl FullIterator<Item = impl Deref<Target = Table> + '_> {
        self.0.tables.iter()
    }

    pub(crate) const fn inner(&self) -> &Inner {
        &self.0
    }
}

impl Slots {
    const SHIFT: usize = 8;
    const CHUNK: usize = 1 << Self::SHIFT;

    pub fn new() -> Self {
        Self(RwLock::new(0), Vec::new().into())
    }

    #[inline]
    pub const fn decompose(index: u32) -> (u32, u8) {
        (index >> Self::SHIFT, index as u8)
    }

    pub fn get(&self, key: Key) -> Option<&Slot> {
        let count_read = self.0.read();
        let (chunk_index, slot_index) = Self::decompose(key.index());
        // SAFETY: `chunks` can be read since the `count_read` lock is held.
        let chunks = unsafe { &**self.1.get() };
        let chunk = &**chunks.get(chunk_index as usize)?;
        // SAFETY: As soon as the `chunk` is dereferenced, the `count_read` lock is no longer needed.
        drop(count_read);
        let slot = chunk.get(slot_index as usize)?;
        if slot.generation() == key.generation() {
            // SAFETY: A shared reference to a slot can be returned safely without being tied to the lifetime of the read guard
            // because its address is stable and no mutable reference to it is ever given out.
            // The stability of the address is guaranteed by the fact that the `chunks` vector never drops its items other than
            // when `self` is dropped.
            Some(slot)
        } else {
            None
        }
    }

    pub unsafe fn get_unchecked(&self, key: Key) -> &Slot {
        // See `slot` for safety.
        let count_read = self.0.read();
        let (chunk_index, slot_index) = Self::decompose(key.index());
        let chunks = &**self.1.get();
        let chunk = &**chunks.get_unchecked(chunk_index as usize);
        drop(count_read);
        chunk.get_unchecked(slot_index as usize)
    }

    pub fn reserve(&self, count: u32) -> Option<u32> {
        let mut count_write = self.0.write();
        let index = *count_write;
        let count = index.saturating_add(count);
        if count == u32::MAX {
            return None;
        }

        // SAFETY: `chunks` can be safely written to since the `count_write` lock is held.
        let chunks = unsafe { &mut *self.1.get() };
        while count as usize > chunks.len() * Self::CHUNK {
            chunks.push(Box::new([(); Self::CHUNK].map(|_| Slot::default())));
        }
        *count_write = count;
        drop(count_write);
        Some(index)
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
}

impl Default for Slot {
    fn default() -> Self {
        Self::new(u32::MAX, u32::MAX)
    }
}

impl Inner {
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
            identifier: identify(),
            free: RwLock::new((Vec::new(), 0.into())),
            slots: Slots::new(),
            tables: Tables::new(),
        }
    }

    pub fn create(
        &self,
        table: &RwLock<Table>,
        keys: &mut [Key],
        mut initialize: impl FnMut((usize, usize), &Table),
    ) -> Option<()> {
        // Create in batches to give a chance to other threads to make progress.
        for keys in keys.chunks_mut(Slots::CHUNK) {
            let key_count = keys.len() as u16;
            // Hold this lock until the operation is fully complete such that no move operation are interleaved.
            let table_upgrade = table.upgradable_read();
            let (row_index, row_count, table_read) = Self::create_reserve(key_count, table_upgrade);
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
                    (&mut *table_read.keys.get()).get_unchecked_mut(row_index as usize..row_count)
                }
                .copy_from_slice(keys);
                initialize((row_index as _, keys.len()), &table_read);
            } else {
                drop(free_read);
            }

            if done < key_count as _ {
                // Since all indices use `u32` for compactness, this index must remain under `u32::MAX`.
                // Note that 'u32::MAX' is used as a sentinel so it must be an invalid entity index.
                let keys = &mut keys[done..];
                let index = self
                    .slots
                    .reserve(keys.len() as u32)
                    .expect("Expected slot count to be `< u32::MAX`.");
                for (i, key) in keys.iter_mut().enumerate() {
                    *key = Key::new(index + i as u32);
                }

                let head = row_index as usize + done;
                unsafe { (&mut *table_read.keys.get()).get_unchecked_mut(head..row_count) }
                    .copy_from_slice(keys);
                initialize((head, keys.len()), &table_read);
            }

            // Initialize the slot only after the table row has been fully initialized.
            for &key in keys.iter() {
                let slot = unsafe { self.slots.get_unchecked(key) };
                slot.initialize(key.generation(), table_read.index(), row_index);
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
            let slot = self.slots.get(key)?;
            let source_indices = slot.indices();
            if source_indices.0 == target_index {
                // No move is needed.
                break Some(());
            }
            let source_table = self.tables.get(source_indices.0 as usize)?;

            // Note that 2 very synchronized threads with their `source_table` and `target_table` swapped may
            // defeat this scheme for taking 2 write locks without dead locking. It is assumed that it doesn't
            // really happen in practice.
            let source_write = source_table.write();
            let (source_write, target_upgrade) = match target_table.try_upgradable_read() {
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
            let (row_index, store_count, target_read) = Self::create_reserve(1, target_upgrade);
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
                    source_write.stores().get_mut(store_indices.0),
                    target_read.stores().get(store_indices.1),
                ) {
                    (Some(source_store), Some(target_store)) => {
                        let source_identifier = source_store.meta().identifier;
                        let target_identifier = target_store.meta().identifier;
                        if source_identifier == target_identifier {
                            store_indices.0 += 1;
                            store_indices.1 += 1;
                            unsafe {
                                Store::copy(
                                    (source_store, source_indices.1 as _),
                                    (target_store, row_index as _),
                                    1,
                                );
                            };
                            drop_or_squash(last_index, source_indices.1, source_store);
                        } else if source_identifier < target_identifier {
                            drop_or_squash(last_index, source_indices.1, source_store);
                            store_indices.0 += 1;
                        } else {
                            store_indices.1 += 1;
                            initialize(row_index as _, store_count, target_store);
                        }
                    }
                    (Some(source_store), None) => {
                        store_indices.0 += 1;
                        drop_or_squash(last_index, source_indices.1, source_store);
                    }
                    (None, Some(target_store)) => {
                        store_indices.1 += 1;
                        initialize(row_index as _, store_count, target_store);
                    }
                    (None, None) => break,
                }
            }

            if last_index == source_indices.1 {
                unsafe {
                    let keys = &mut *target_read.keys.get();
                    *keys.get_unchecked_mut(row_index as usize) = key;
                    self.slots
                        .get_unchecked(key)
                        .update(target_index, row_index);
                }
            } else {
                let source_keys = source_write.keys.get_mut();
                unsafe {
                    let last_key = *source_keys.get_unchecked(last_index as usize);
                    let source_key = source_keys.get_unchecked_mut(source_indices.1 as usize);
                    let source_key = replace(source_key, last_key);

                    let target_keys = &mut *target_read.keys.get();
                    *target_keys.get_unchecked_mut(row_index as usize) = source_key;
                    self.slots
                        .get_unchecked(source_key)
                        .update(target_index, row_index);
                    self.slots
                        .get_unchecked(last_key)
                        .update(source_indices.0, source_indices.1);
                }
            }

            Self::create_resolve(1, target_read);
            drop(source_write);
            break Some(());
        }
    }

    pub fn destroy(&self, key: Key) -> Option<()> {
        let slot = self.slots.get(key)?;
        let (table_index, row_index) = slot.release(key.generation())?;
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

        if row_index == last_index {
            for store in table_write.stores.iter_mut() {
                unsafe { store.drop(row_index as _, 1) };
            }
        } else {
            for store in table_write.stores.iter_mut() {
                unsafe { store.squash(last_index as _, row_index as _, 1) };
            }

            let keys = table_write.keys.get_mut();
            unsafe {
                let last_key = *keys.get_unchecked(last_index as usize);
                *keys.get_unchecked_mut(row_index as usize) = last_key;
                self.slots
                    .get_unchecked(last_key)
                    .update(table_index, row_index);
            }
        }

        drop(table_write);
        self.destroy_resolve(key);
        Some(())
    }

    fn create_reserve(
        reserve: u16,
        table_upgrade: RwLockUpgradableReadGuard<Table>,
    ) -> (u32, usize, RwLockReadGuard<Table>) {
        let (begun, count) = {
            let add = Self::recompose_count(reserve, 0, 0);
            let count = table_upgrade.count.fetch_add(add, AcqRel);
            let (begun, ended, count) = Self::decompose_count(count);
            debug_assert!(begun >= ended);
            (begun, count)
        };
        let row_index = count + begun as u32;
        let row_count = row_index as usize + reserve as usize;

        // There can not be more than `u32::MAX` keys at a given time.
        assert!(row_count < u32::MAX as _);
        let table_read = if row_count > table_upgrade.capacity() {
            let mut table_write = RwLockUpgradableReadGuard::upgrade(table_upgrade);
            table_write.grow(row_count as _);
            RwLockWriteGuard::downgrade(table_write)
        } else if begun >= u16::MAX >> 1 {
            // A huge stream of concurrent `create` operations has been detected; force the resolution of that table's `count`
            // before `begun` or `ended` overflows. This should essentially never happen.
            let mut table_write = RwLockUpgradableReadGuard::upgrade(table_upgrade);
            let table_count = table_write.count.get_mut();
            let (begun, ended, count) = Self::decompose_count(*table_count);
            debug_assert_eq!(begun + reserve, ended);
            *table_count = Self::recompose_count(reserve, 0, count + ended as u32);
            RwLockWriteGuard::downgrade(table_write)
        } else {
            RwLockUpgradableReadGuard::downgrade(table_upgrade)
        };
        (row_index, row_count, table_read)
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
        key.increment();
        if key.generation() < u32::MAX {
            let mut free_write = self.free.write();
            let free_count = *free_write.1.get_mut();
            free_write.0.truncate(free_count.max(0) as _);
            free_write.0.push(key);
            *free_write.1.get_mut() = free_write.0.len() as _;
            drop(free_write);
        }
    }
}
