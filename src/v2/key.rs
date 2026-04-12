//! Stable-key vocabulary for `v2`.
//!
//! Storage primitives in the rewrite are agnostic to whether a table carries a `Key` column.
//! This module only defines the datum that later extension tasks will connect to the `Keys`
//! resource and reverse mappings.

/// A stable identity datum that later extension tasks synchronize through the `Keys` resource.
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
