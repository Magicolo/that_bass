use crate::{core::utility::get_unchecked, Error};
use parking_lot::RwLock;
use std::{
    cell::UnsafeCell,
    ops::Range,
    sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering::*},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    index: u32,
    generation: u32,
}

pub struct Slot {
    // TODO: Package `table` with `generation` since they tend to be required together.
    // - Most updates to `indices` only change the `row`.
    // - `generation` has to be validated everytime the table is read to sort keys in `Add/Remove/Destroy`.
    // generation: AtomicU32,
    indices: AtomicU64,
    row: AtomicU32,
}

pub struct Keys {
    free: RwLock<(Vec<Key>, AtomicI64)>,
    /// The lock is seperated from the chunks because once a chunk is dereferenced from the `chunks` vector, it no longer
    /// needs to have its lifetime tied to a `RwLockReadGuard`. This is safe because the addresses of chunks are stable
    /// (guaranteed by the `Box` indirection) and no mutable references are ever given out.
    count: RwLock<u32>,
    chunks: UnsafeCell<Vec<Box<[Slot; Self::CHUNK]>>>,
}

pub struct State;

unsafe impl Send for Keys {}
unsafe impl Sync for Keys {}

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

    #[inline]
    pub const fn valid(&self) -> bool {
        self.index < u32::MAX && self.generation < u32::MAX
    }

    #[inline]
    pub const fn index(&self) -> u32 {
        self.index
    }

    #[inline]
    pub const fn generation(&self) -> u32 {
        self.generation
    }

    #[inline]
    pub(crate) fn increment(&mut self) -> bool {
        self.generation = self.generation.saturating_add(1);
        self.generation < u32::MAX
    }
}

impl Keys {
    const SHIFT: usize = 8;
    pub const CHUNK: usize = 1 << Self::SHIFT;

    #[inline]
    pub fn new() -> Self {
        Self {
            free: RwLock::new((Vec::new(), 0.into())),
            count: RwLock::new(0),
            chunks: Vec::new().into(),
        }
    }

    #[inline]
    const fn decompose(index: u32) -> (u32, u8) {
        (index >> Self::SHIFT, index as u8)
    }

    #[inline]
    pub fn get(&self, key: Key) -> Result<(&Slot, u32), Error> {
        let read = self.count.read();
        let (chunk_index, slot_index) = Self::decompose(key.index());
        // SAFETY: `chunks` can be read since the `count_read` lock is held.
        let chunks = unsafe { &**self.chunks.get() };
        let chunk = &**chunks.get(chunk_index as usize).ok_or(Error::InvalidKey)?;
        // SAFETY: As soon as the `chunk` is dereferenced, the `count_read` lock is no longer needed.
        drop(read);
        let slot = unsafe { get_unchecked(chunk, slot_index as usize) };
        // SAFETY: A shared reference to a slot can be returned safely without being tied to the lifetime of the read guard
        // because its address is stable and no mutable reference to it is ever given out.
        // The stability of the address is guaranteed by the fact that the `chunks` vector never drops its items other than
        // when `self` is dropped.
        Ok((slot, slot.table(key.generation())?))
    }

    /// SAFETY: The provided key must be valid.
    #[inline]
    pub unsafe fn get_unchecked(&self, key: Key) -> &Slot {
        let read = self.count.read();
        let (chunk_index, slot_index) = Self::decompose(key.index());
        // SAFETY: See `get`.
        let chunks = &**self.chunks.get();
        let chunk = &**get_unchecked(chunks, chunk_index as usize);
        drop(read);
        get_unchecked(chunk, slot_index as usize)
    }

    #[inline]
    pub fn get_all(
        &self,
        keys: impl IntoIterator<Item = Key>,
    ) -> impl Iterator<Item = (Key, Result<(&Slot, u32), Error>)> {
        let read = self.count.read();
        keys.into_iter().map(move |key| {
            // Keep the lock alive.
            let _read = &read;
            let (chunk_index, slot_index) = Keys::decompose(key.index());
            // SAFETY: See `get`.
            let chunks = unsafe { &**self.chunks.get() };
            let Some(chunk) = chunks.get(chunk_index as usize) else {
                return (key, Err(Error::InvalidKey));
            };
            let slot = unsafe { get_unchecked(&**chunk, slot_index as usize) };
            match slot.table(key.generation()) {
                Ok(table) => (key, Ok((slot, table))),
                Err(error) => (key, Err(error)),
            }
        })
    }

    /// SAFETY: The provided keys must be valid.
    #[inline]
    pub unsafe fn get_all_unchecked(
        &self,
        keys: impl IntoIterator<Item = Key>,
    ) -> impl Iterator<Item = (Key, &Slot)> {
        let read = self.count.read();
        keys.into_iter().map(move |key| {
            // Keep the lock alive.
            let _read = &read;
            let (chunk_index, slot_index) = Keys::decompose(key.index());
            // SAFETY: See `get`.
            let chunks = unsafe { &**self.chunks.get() };
            let chunk = &**unsafe { get_unchecked(chunks, chunk_index as usize) };
            (key, unsafe { get_unchecked(chunk, slot_index as usize) })
        })
    }

    pub fn reserve(&self, keys: &mut [Key]) {
        let free_read = self.free.read();
        let tail = free_read.1.fetch_sub(keys.len() as _, Relaxed);
        let done = if tail > 0 {
            let tail = tail as usize;
            let count = tail.min(keys.len());
            let head = tail - count;
            keys[..count].copy_from_slice(&free_read.0[head..tail]);
            count
        } else {
            0
        };
        drop(free_read);

        if done < keys.len() {
            let keys = &mut keys[done..];
            let mut count_write = self.count.write();
            let index = *count_write;
            let count = index.saturating_add(keys.len() as _);
            // Since all indices use `u32` for compactness, this index must remain under `u32::MAX`.
            // Note that 'u32::MAX' is used as a sentinel so it must be an invalid entity index.
            assert!(count < u32::MAX);

            // SAFETY: `chunks` can be safely written to since the `count_write` lock is held.
            let chunks = unsafe { &mut *self.chunks.get() };
            while count as usize > chunks.len() * Self::CHUNK {
                chunks.push(Box::new([(); Self::CHUNK].map(|_| Slot::default())));
            }
            *count_write = count;
            drop(count_write);

            for (i, key) in keys.iter_mut().enumerate() {
                *key = Key::new(index + i as u32);
            }
        }
    }

    #[inline]
    pub(crate) fn initialize(&self, keys: &[Key], table: u32, range: Range<usize>) {
        let mut row = range.start as u32;
        for (key, slot) in unsafe { self.get_all_unchecked(keys[range].iter().copied()) } {
            slot.initialize(key.generation(), table, row);
            row += 1;
        }
    }

    #[inline]
    pub(crate) fn update(&self, keys: &[Key], range: Range<usize>) {
        let mut row = range.start as u32;
        for (_, slot) in unsafe { self.get_all_unchecked(keys[range].iter().copied()) } {
            slot.update(row);
            row += 1;
        }
    }

    #[inline]
    pub(crate) fn release(&self, keys: &[Key]) {
        for (_, slot) in unsafe { self.get_all_unchecked(keys.iter().copied()) } {
            slot.release();
        }
        self.recycle(keys.iter().copied());
    }

    /// Assumes that the keys have had their `Slot` released and that keys are release only once.
    pub(crate) fn recycle(&self, keys: impl IntoIterator<Item = Key>) {
        let mut free_write = self.free.write();
        let (free_keys, free_count) = &mut *free_write;
        free_keys.truncate((*free_count.get_mut()).max(0) as _);

        for mut key in keys {
            // If the key reached generation `u32::MAX`, its index is discarded which results in a dead `Slot`.
            if key.increment() {
                free_keys.push(key);
            }
        }

        *free_count.get_mut() = free_keys.len() as _;
    }
}

impl Slot {
    const fn recompose_indices(generation: u32, table: u32) -> u64 {
        ((generation as u64) << 32) | (table as u64)
    }

    const fn decompose_indices(indices: u64) -> (u32, u32) {
        ((indices >> 32) as u32, indices as u32)
    }

    #[inline]
    pub fn new(table: u32, row: u32) -> Self {
        Self {
            indices: Self::recompose_indices(0, table).into(),
            row: row.into(),
        }
    }

    #[inline]
    pub fn initialize(&self, generation: u32, table: u32, row: u32) {
        debug_assert!(generation < u32::MAX);
        debug_assert!(table < u32::MAX);
        let indices = Self::recompose_indices(generation, table);
        self.indices.store(indices, Release);
        self.update(row);
    }

    #[inline]
    pub fn update(&self, row: u32) {
        debug_assert!(row < u32::MAX);
        self.row.store(row, Release);
    }

    #[inline]
    pub fn release(&self) {
        self.indices.store(u64::MAX, Release);
        self.row.store(u32::MAX, Release);
    }

    #[inline]
    pub fn table(&self, generation: u32) -> Result<u32, Error> {
        let indices = Self::decompose_indices(self.indices.load(Acquire));
        if indices.0 == generation && indices.1 < u32::MAX {
            Ok(indices.1)
        } else {
            Err(Error::InvalidKey)
        }
    }

    #[inline]
    pub fn row(&self) -> u32 {
        self.row.load(Acquire)
    }
}

impl Default for Slot {
    fn default() -> Self {
        Self::new(u32::MAX, u32::MAX)
    }
}
