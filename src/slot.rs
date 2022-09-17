use parking_lot::RwLock;
use std::{
    cell::UnsafeCell,
    sync::atomic::{AtomicU32, AtomicU64, Ordering::*},
};

use crate::key::Key;

pub(crate) struct Slot {
    generation: AtomicU32,
    indices: AtomicU64,
}

pub(crate) struct Slots(RwLock<u32>, UnsafeCell<Vec<Box<[Slot; Self::CHUNK]>>>);

impl Slots {
    const SHIFT: usize = 8;
    pub const CHUNK: usize = 1 << Self::SHIFT;

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
