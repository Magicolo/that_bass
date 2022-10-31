use crate::Error;
use parking_lot::RwLock;
use std::{
    cell::UnsafeCell,
    sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering::*},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    index: u32,
    generation: u32,
}

pub struct Slot {
    generation: AtomicU32,
    indices: AtomicU64,
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
    pub(crate) fn increment(&mut self) -> u32 {
        self.generation = self.generation.saturating_add(1);
        self.generation
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

    pub fn get(&self, key: Key) -> Result<&Slot, Error> {
        let count_read = self.count.read();
        let (chunk_index, slot_index) = Self::decompose(key.index());
        // SAFETY: `chunks` can be read since the `count_read` lock is held.
        let chunks = unsafe { &**self.chunks.get() };
        let chunk = &**chunks.get(chunk_index as usize).ok_or(Error::InvalidKey)?;
        // SAFETY: As soon as the `chunk` is dereferenced, the `count_read` lock is no longer needed.
        drop(count_read);
        let slot = unsafe { chunk.get_unchecked(slot_index as usize) };
        if slot.generation() == key.generation() {
            // SAFETY: A shared reference to a slot can be returned safely without being tied to the lifetime of the read guard
            // because its address is stable and no mutable reference to it is ever given out.
            // The stability of the address is guaranteed by the fact that the `chunks` vector never drops its items other than
            // when `self` is dropped.
            Ok(slot)
        } else {
            Err(Error::InvalidKey)
        }
    }

    pub unsafe fn get_unchecked(&self, key: Key) -> &Slot {
        // See `slot` for safety.
        let count_read = self.count.read();
        let (chunk_index, slot_index) = Self::decompose(key.index());
        let chunks = &**self.chunks.get();
        let chunk = &**chunks.get_unchecked(chunk_index as usize);
        drop(count_read);
        chunk.get_unchecked(slot_index as usize)
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

    /// Assumes that the keys have had their `Slot` released.
    pub(crate) fn release(&self, keys: impl IntoIterator<Item = Key>) {
        let mut free_write = self.free.write();
        let (free_keys, free_count) = &mut *free_write;
        free_keys.truncate((*free_count.get_mut()).max(0) as _);

        for mut key in keys {
            // If the key reached generation `u32::MAX`, its index is discarded which results in a dead `Slot`.
            if key.increment() < u32::MAX {
                free_keys.push(key);
            }
        }

        *free_count.get_mut() = free_keys.len() as _;
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
        Self {
            generation: 0.into(),
            indices: Self::recompose_indices(table, store).into(),
        }
    }

    #[inline]
    pub fn initialize(&self, generation: u32, table: u32, store: u32) {
        debug_assert!(generation < u32::MAX);
        self.generation.store(generation, Release);
        self.update(table, store);
    }

    #[inline]
    pub fn update(&self, table: u32, store: u32) {
        let indices = Self::recompose_indices(table, store);
        debug_assert!(indices < u64::MAX);
        self.indices.store(indices, Release);
    }

    #[inline]
    pub fn release(&self, generation: u32) -> Result<(u32, u32), Error> {
        self.generation
            .compare_exchange(generation, u32::MAX, AcqRel, Acquire)
            .map_err(|_| Error::WrongGeneration)?;
        let indices = self.indices.swap(u64::MAX, Release);
        debug_assert!(indices < u64::MAX);
        Ok(Self::decompose_indices(indices))
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
    pub fn table(&self) -> u32 {
        self.indices().0
    }

    #[inline]
    pub fn row(&self) -> u32 {
        self.indices().1
    }
}

impl Default for Slot {
    fn default() -> Self {
        Self::new(u32::MAX, u32::MAX)
    }
}
