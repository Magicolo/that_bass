use arc_swap::ArcSwapAny;
use core::sync::atomic::{AtomicU32, AtomicU64};
use triomphe::{Arc, ThinArc};

pub struct Key {
    index: u32,
    generation: u32,
}

pub struct Keys(Arc<State>);

struct State {
    slots: ArcSwapAny<ThinArc<(), AtomicU64>>,
    next: AtomicU32,
}
