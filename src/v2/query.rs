//! Query vocabulary for the rewrite lane.
//!
//! The selected rewrite direction exposes chunk-oriented query declarations. This foundation task
//! records the minimal access vocabulary and the first keyless row-handle request that later tasks
//! will build on.

pub use crate::v2::schema::{Row, Rows};

/// The access level a future query item may request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    Read,
    Write,
}

/// A request for transient row handles aligned with a chunk's inhabited prefix.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct RowsRequest;

pub const fn rows() -> RowsRequest {
    RowsRequest
}
