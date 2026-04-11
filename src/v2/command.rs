//! Deferred command vocabulary for the rewrite lane.
//!
//! The rewrite records structural work during parallel job execution and resolves it later in an
//! explicit batched phase. This module defines only the public terms needed for that model.

/// The structural command families planned for the first rewrite milestones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Insert,
    Remove,
    Set,
}

/// The selected resolve strategy for the rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Strategy {
    /// Jobs record commands independently, then a later resolve phase batches all command buffers
    /// produced by the same scheduled function.
    FunctionLevelBatch,
}
