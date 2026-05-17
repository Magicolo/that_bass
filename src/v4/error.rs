use core::alloc::LayoutError;
use core::any::Any;
use core::num::TryFromIntError;

pub enum Error {
    DuplicateMeta,
    FailedToPush,
    FailedToInitialize,
    Layout(LayoutError),
    TableOverflow,
    MissingTable,
    InvalidItem(Box<dyn Any>),
    TablesOverflow(TryFromIntError),
    TooManyItems(TryFromIntError),
    FailedToDrop,
    VectorOverflow,
    FailedToAllocate,
    FailedToResolve,
    FailedToReserve,
}
