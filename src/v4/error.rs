use core::alloc::LayoutError;
use core::num::TryFromIntError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("duplicate metadata entry in schema definition")]
    DuplicateMeta,
    #[error("failed to push item into column storage")]
    FailedToPush,
    #[error("table initialization failed: maximum table count exceeded")]
    FailedToInitialize,
    #[error("invalid memory layout: {0}")]
    Layout(LayoutError),
    #[error("table index exceeds maximum table count")]
    TableOverflow,
    #[error("table not found for the given schema")]
    MissingTable,
    #[error("type mismatch: item does not match column schema")]
    InvalidItem,
    #[error("table count overflow: {0}")]
    TablesOverflow(TryFromIntError),
    #[error("row count overflow: {0}")]
    TooManyItems(TryFromIntError),
    #[error("failed to drop column data")]
    FailedToDrop,
    #[error("column capacity exceeded")]
    VectorOverflow,
    #[error("memory allocation failed")]
    FailedToAllocate,
    #[error("failed to resolve deferred operations")]
    FailedToResolve,
    #[error("no available row slot in table")]
    FailedToReserve,
}
