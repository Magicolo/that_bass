//! Managed-key vocabulary for `v2`.
//!
//! Keyless tables are the default in the rewrite. This module only defines the opt-in stable
//! identity type that later tasks will connect to keyed tables and reverse mappings.

/// An opt-in stable identity for tables that explicitly request managed keys.
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
