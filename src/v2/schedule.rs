//! Scheduling vocabulary for the rewrite lane.
//!
//! The scheduler owns hot-path safety in `v2`. This module contains the first public terms for
//! ordering semantics without committing to a full executor implementation yet.

/// The ordering source that establishes a happens-before edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ordering {
    /// The default ordering induced by function declaration order when accesses conflict.
    ImplicitDeclarationOrder,
    /// A selective dependency declared by the user.
    ExplicitDependency,
    /// A user-declared barrier that forces all later work to observe all earlier work.
    ExplicitBarrier,
}
