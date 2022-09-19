use parking_lot::RwLock;

use crate::{
    query::{At, Context, Item},
    table::{Store, Table},
    Error,
};
use std::{
    cell::UnsafeCell,
    slice::from_raw_parts,
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

pub(crate) struct Keys {
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

unsafe impl Item for Key {
    type State = State;

    fn initialize(_: &Table) -> Option<Self::State> {
        Some(State)
    }

    fn validate(mut context: Context) -> Result<(), Error> {
        context.read::<Key>()
    }
}

impl<'a> At<'a> for State {
    type State = (*const Key, usize);
    type Chunk = &'a [Key];
    type Item = Key;

    #[inline]
    fn try_get(&self, keys: &[Key], stores: &[Store]) -> Option<Self::State> {
        Some(self.get(keys, stores))
    }
    #[inline]
    fn get(&self, keys: &[Key], _: &[Store]) -> Self::State {
        (keys.as_ptr(), keys.len())
    }
    #[inline]
    unsafe fn chunk(state: &mut Self::State) -> Self::Chunk {
        from_raw_parts(state.0, state.1)
    }
    #[inline]
    unsafe fn item(state: &mut Self::State, index: usize) -> Self::Item {
        *Self::chunk(state).get_unchecked(index)
    }
}

impl Keys {
    const SHIFT: usize = 8;
    pub const CHUNK: usize = 1 << Self::SHIFT;

    pub fn new() -> Self {
        Self {
            free: RwLock::new((Vec::new(), 0.into())),
            count: RwLock::new(0),
            chunks: Vec::new().into(),
        }
    }

    #[inline]
    pub const fn decompose(index: u32) -> (u32, u8) {
        (index >> Self::SHIFT, index as u8)
    }

    pub fn get(&self, key: Key) -> Option<&Slot> {
        let count_read = self.count.read();
        let (chunk_index, slot_index) = Self::decompose(key.index());
        // SAFETY: `chunks` can be read since the `count_read` lock is held.
        let chunks = unsafe { &**self.chunks.get() };
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
            keys.copy_from_slice(&free_read.0[head..tail]);
            drop(free_read);
            count
        } else {
            drop(free_read);
            0
        };

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

    pub fn release(&self, mut key: Key) -> Option<(u32, u32)> {
        let slot = self.get(key)?;
        let (table_index, row_index) = slot.release(key.generation())?;

        // If the key reached generation `u32::MAX`, its index is discarded which results in a dead `Slot`.
        if key.increment() < u32::MAX {
            let mut free_write = self.free.write();
            let free_count = *free_write.1.get_mut();
            free_write.0.truncate(free_count.max(0) as _);
            free_write.0.push(key);
            *free_write.1.get_mut() = free_write.0.len() as _;
            drop(free_write);
        }

        Some((table_index, row_index))
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
