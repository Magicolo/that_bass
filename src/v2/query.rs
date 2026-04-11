//! Query vocabulary for the rewrite lane.
//!
//! The selected rewrite direction exposes chunk-oriented query declarations. This foundation task
//! records only the minimal access vocabulary that later tasks will build on.

/// The access level a future query item may request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    Read,
    Write,
}
