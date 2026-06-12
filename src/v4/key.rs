use crate::v4::utility::IteratorExtension;
use arc_swap::{ArcSwapAny, AsRaw};
use core::{
    fmt::Debug,
    iter::empty,
    sync::atomic::{AtomicU32, AtomicU64, Ordering},
};
use triomphe::{Arc, ThinArc};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    index: u32,
    generation: u32,
}

impl Key {
    pub const NULL: Self = Self {
        index: u32::MAX,
        generation: u32::MAX,
    };

    pub(crate) const fn new(index: u32) -> Self {
        Self {
            index,
            generation: 0,
        }
    }

    pub const fn valid(&self) -> bool {
        self.index < u32::MAX && self.generation < u32::MAX
    }

    pub const fn index(&self) -> u32 {
        self.index
    }

    pub const fn generation(&self) -> u32 {
        self.generation
    }

    pub(crate) fn increment(&mut self) -> bool {
        self.generation = self.generation.saturating_add(1);
        self.generation < u32::MAX
    }
}

#[derive(Debug)]
struct State {
    slots: ArcSwapAny<ThinArc<u32, AtomicU64>>,
    next: AtomicU32,
    last: AtomicU32,
}

#[derive(Debug, Clone)]
pub struct Keys(Arc<State>);

impl Keys {
    pub(crate) fn new() -> Self {
        Self(Arc::new(State {
            slots: ArcSwapAny::new(ThinArc::from_header_and_iter(0, empty())),
            next: AtomicU32::new(u32::MAX),
            last: AtomicU32::new(0),
        }))
    }

    pub(crate) fn allocate(&self, table: u32, row: u32) -> Key {
        debug_assert!(table < u32::MAX);
        debug_assert!(row < u32::MAX);
        let target = pack(table, row);
        loop {
            if let Some(key) = self.try_pop(target) {
                return key;
            }
            if let Some(key) = self.try_extend(target) {
                return key;
            }
        }
    }

    fn try_pop(&self, target: u64) -> Option<Key> {
        let head = self.0.next.load(Ordering::Acquire);
        if head == u32::MAX {
            return None;
        }
        let slots = self.0.slots.load();
        let slot = &slots.slice[head as usize];
        let entry = slot.load(Ordering::Acquire);
        let (next, generation) = unpack(entry);
        if self
            .0
            .next
            .compare_exchange_weak(head, next, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            slot.store(target, Ordering::Release);
            Some(Key {
                index: head,
                generation,
            })
        } else {
            None
        }
    }

    fn try_extend(&self, target: u64) -> Option<Key> {
        let old = self.0.slots.load();
        let index = old.slice.len();
        if index >= u32::MAX as usize {
            return None;
        }
        let new_slot = Arc::new(AtomicU64::new(target));
        let new = ThinArc::from_header_and_iter((), old.slice.iter().cloned().and(new_slot));
        let result = self.0.slots.compare_and_swap(&*old, new);
        if old.as_raw() == result.as_raw() {
            Some(Key::new(index as u32))
        } else {
            None
        }
    }

    pub(crate) fn free(&self, mut key: Key) {
        if !key.increment() {
            let slots = self.0.slots.load();
            if let Some(slot) = slots.slice.get(key.index as usize) {
                slot.store(u64::MAX, Ordering::Release);
            }
            return;
        }
        loop {
            let head = self.0.next.load(Ordering::Acquire);
            let packed = pack(head, key.generation);
            let slots = self.0.slots.load();
            let slot = &slots.slice[key.index as usize];
            slot.store(packed, Ordering::Release);
            if self
                .0
                .next
                .compare_exchange_weak(head, key.index, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
        }
    }

    pub(crate) fn lookup(&self, key: Key) -> Option<(u32, u32)> {
        let slots = self.0.slots.load();
        let slot = slots.slice.get(key.index as usize)?;
        let (table, row) = unpack(slot.load(Ordering::Acquire));
        if table < u32::MAX && row < u32::MAX {
            Some((table, row))
        } else {
            None
        }
    }
}

impl Default for Keys {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
const fn pack(low: u32, high: u32) -> u64 {
    (low as u64) | ((high as u64) << 32)
}

#[inline]
const fn unpack(value: u64) -> (u32, u32) {
    (value as u32, (value >> 32) as u32)
}
