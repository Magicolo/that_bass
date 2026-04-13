//! Stable-key resource for `v2`.
//!
//! Storage primitives stay agnostic to whether a table carries a `Key` column. This module adds
//! the opt-in reverse mapping used by managed stable identity:
//!
//! - user code reserves `Key` values up front,
//! - keyed rows store `Key` inline like any other column,
//! - structural resolve publishes and releases `Key -> Row`,
//! - and keyed random access consults this resource first.

use crate::v2::{
    query::Access,
    schema::{Dependency, Resource, ResourceId, Row},
};
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use parking_lot::RwLock;
use std::sync::Arc;

const EMPTY_FREE_LIST: u32 = u32::MAX;
const EMPTY_ROW: u64 = u64::MAX;

/// A stable identity datum used by the managed `Keys` resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Key {
    slot_index: u32,
    generation: u32,
}

impl Key {
    pub const fn new(slot_index: u32, generation: u32) -> Self {
        Self {
            slot_index,
            generation,
        }
    }

    pub const fn slot_index(self) -> u32 {
        self.slot_index
    }

    pub const fn generation(self) -> u32 {
        self.generation
    }
}

/// The observable state of one reserved or live key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Entry {
    Reserved,
    Live(Row<'static>),
}

/// Failures produced by the managed-key resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Error {
    InvalidKey { key: Key },
    UnexpectedState { key: Key, state: StateKind },
}

/// The scheduler-visible slot state used by diagnostics and validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateKind {
    Free,
    Reserved,
    Live,
}

#[derive(Debug)]
struct Slot {
    status: AtomicU64,
    row: AtomicU64,
    next_free_slot_index: AtomicU32,
}

impl Slot {
    fn reserved(generation: u32) -> Self {
        Self {
            status: AtomicU64::new(Status::new(generation, StateKind::Reserved).packed()),
            row: AtomicU64::new(EMPTY_ROW),
            next_free_slot_index: AtomicU32::new(EMPTY_FREE_LIST),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Status {
    generation: u32,
    state: StateKind,
}

impl Status {
    const fn new(generation: u32, state: StateKind) -> Self {
        Self { generation, state }
    }

    const fn packed(self) -> u64 {
        ((self.generation as u64) << 32) | (self.state as u64)
    }

    fn from_packed(packed: u64) -> Self {
        Self {
            generation: (packed >> 32) as u32,
            state: match (packed & 0xff) as u8 {
                0 => StateKind::Free,
                1 => StateKind::Reserved,
                2 => StateKind::Live,
                _ => unreachable!("slot status should only store known state tags"),
            },
        }
    }
}

#[derive(Debug)]
struct Shared {
    slots: RwLock<Vec<Arc<Slot>>>,
    free_slot_head_index: AtomicU32,
}

impl Default for Shared {
    fn default() -> Self {
        Self {
            slots: RwLock::new(Vec::new()),
            free_slot_head_index: AtomicU32::new(EMPTY_FREE_LIST),
        }
    }
}

/// The managed stable-key resource.
#[derive(Debug, Clone, Default)]
pub struct Keys {
    shared: Arc<Shared>,
}

impl Keys {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub fn reserve(&self) -> Key {
        if let Some(key) = self.try_reserve_free_slot() {
            return key;
        }

        let mut slots = self.shared.slots.write();
        let slot_index = u32::try_from(slots.len())
            .expect("managed key slot indices must remain representable as u32");
        let generation = 0;
        slots.push(Arc::new(Slot::reserved(generation)));

        Key::new(slot_index, generation)
    }

    pub fn state(&self, key: Key) -> Result<Entry, Error> {
        let slot = self.slot(key)?;
        let status = Status::from_packed(slot.status.load(Ordering::Acquire));
        if status.generation != key.generation() {
            return Err(Error::InvalidKey { key });
        }

        match status.state {
            StateKind::Reserved => Ok(Entry::Reserved),
            StateKind::Live => {
                let packed_row = slot.row.load(Ordering::Acquire);
                debug_assert_ne!(packed_row, EMPTY_ROW);
                Ok(Entry::Live(Row::from_packed(packed_row)))
            }
            StateKind::Free => Err(Error::InvalidKey { key }),
        }
    }

    pub fn get(&self, key: Key) -> Result<Row<'static>, Error> {
        match self.state(key)? {
            Entry::Reserved => Err(Error::UnexpectedState {
                key,
                state: StateKind::Reserved,
            }),
            Entry::Live(row) => Ok(row),
        }
    }

    pub(crate) fn publish(&self, key: Key, row: Row<'static>) -> Result<(), Error> {
        let slot = self.slot(key)?;

        loop {
            let observed_status = Status::from_packed(slot.status.load(Ordering::Acquire));
            if observed_status.generation != key.generation() {
                return Err(Error::InvalidKey { key });
            }

            match observed_status.state {
                StateKind::Reserved => {
                    slot.row.store(row.packed(), Ordering::Relaxed);
                    let next_status =
                        Status::new(observed_status.generation, StateKind::Live).packed();
                    if slot
                        .status
                        .compare_exchange(
                            observed_status.packed(),
                            next_status,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return Ok(());
                    }
                }
                StateKind::Live => {
                    return Err(Error::UnexpectedState {
                        key,
                        state: StateKind::Live,
                    });
                }
                StateKind::Free => {
                    return Err(Error::InvalidKey { key });
                }
            }
        }
    }

    pub(crate) fn republish(&self, key: Key, row: Row<'static>) -> Result<(), Error> {
        let slot = self.slot(key)?;
        let observed_status = Status::from_packed(slot.status.load(Ordering::Acquire));
        if observed_status.generation != key.generation() {
            return Err(Error::InvalidKey { key });
        }

        if observed_status.state != StateKind::Live {
            return Err(Error::UnexpectedState {
                key,
                state: observed_status.state,
            });
        }

        slot.row.store(row.packed(), Ordering::Release);
        Ok(())
    }

    pub(crate) fn release(&self, key: Key) -> Result<(), Error> {
        let slot = self.slot(key)?;

        loop {
            let observed_status = Status::from_packed(slot.status.load(Ordering::Acquire));
            if observed_status.generation != key.generation() {
                return Err(Error::InvalidKey { key });
            }

            match observed_status.state {
                StateKind::Reserved | StateKind::Live => {
                    let recycled_generation = observed_status.generation.saturating_add(1);
                    let next_status = Status::new(recycled_generation, StateKind::Free).packed();
                    if slot
                        .status
                        .compare_exchange(
                            observed_status.packed(),
                            next_status,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        slot.row.store(EMPTY_ROW, Ordering::Release);
                        if recycled_generation != u32::MAX {
                            self.push_free_slot(key.slot_index(), &slot);
                        }
                        return Ok(());
                    }
                }
                StateKind::Free => {
                    return Err(Error::InvalidKey { key });
                }
            }
        }
    }

    pub fn slot_count(&self) -> usize {
        self.shared.slots.read().len()
    }

    pub fn read_dependency(root_identifier: ResourceId) -> Dependency {
        Dependency::new(
            Access::Read,
            [
                Resource::store(Some(root_identifier)),
                Resource::typed::<Self>(Some(root_identifier)),
            ],
        )
    }

    pub fn write_dependency(root_identifier: ResourceId) -> Dependency {
        Dependency::new(
            Access::Write,
            [
                Resource::store(Some(root_identifier)),
                Resource::typed::<Self>(Some(root_identifier)),
            ],
        )
    }

    fn try_reserve_free_slot(&self) -> Option<Key> {
        loop {
            let head_slot_index = self.shared.free_slot_head_index.load(Ordering::Acquire);
            if head_slot_index == EMPTY_FREE_LIST {
                return None;
            }

            let slot = self.slot_by_index(head_slot_index)?;
            let next_free_slot_index = slot.next_free_slot_index.load(Ordering::Acquire);
            if self
                .shared
                .free_slot_head_index
                .compare_exchange(
                    head_slot_index,
                    next_free_slot_index,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                continue;
            }

            loop {
                let observed_status = Status::from_packed(slot.status.load(Ordering::Acquire));
                debug_assert_eq!(observed_status.state, StateKind::Free);
                debug_assert_ne!(observed_status.generation, u32::MAX);

                let next_status =
                    Status::new(observed_status.generation, StateKind::Reserved).packed();
                if slot
                    .status
                    .compare_exchange(
                        observed_status.packed(),
                        next_status,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    slot.next_free_slot_index
                        .store(EMPTY_FREE_LIST, Ordering::Release);
                    return Some(Key::new(head_slot_index, observed_status.generation));
                }
            }
        }
    }

    fn push_free_slot(&self, slot_index: u32, slot: &Arc<Slot>) {
        loop {
            let observed_head = self.shared.free_slot_head_index.load(Ordering::Acquire);
            slot.next_free_slot_index
                .store(observed_head, Ordering::Release);
            if self
                .shared
                .free_slot_head_index
                .compare_exchange(
                    observed_head,
                    slot_index,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return;
            }
        }
    }

    fn slot(&self, key: Key) -> Result<Arc<Slot>, Error> {
        self.slot_by_index(key.slot_index())
            .ok_or(Error::InvalidKey { key })
    }

    fn slot_by_index(&self, slot_index: u32) -> Option<Arc<Slot>> {
        self.shared.slots.read().get(slot_index as usize).cloned()
    }
}
