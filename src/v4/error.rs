use crate::v4::module::Resource;
use core::{alloc::LayoutError, num::TryFromIntError};

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
    #[error("read/write conflict: {0:?} -> {1:?}")]
    ReadWriteConflict(Resource, Resource),
    #[error("write/write conflict: {0:?} -> {1:?}")]
    WriteWriteConflict(Resource, Resource),
    #[error("errors: {0:?}")]
    All(Vec<Error>),
}

impl Error {
    pub fn all(errors: impl IntoIterator<Item = Error>) -> Option<Self> {
        let mut errors = errors.into_iter().flat_map(Self::errors);
        let first = errors.next()?;
        match errors.next() {
            Some(second) => {
                let mut all = vec![first, second];
                all.extend(errors);
                Some(Self::All(all))
            }
            None => Some(first),
        }
    }

    fn errors(self) -> impl Iterator<Item = Error> {
        match self {
            Self::All(errors) => None.into_iter().chain(errors),
            error => Some(error).into_iter().chain(Vec::new()),
        }
    }
}
