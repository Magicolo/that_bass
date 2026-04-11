//! Neutral measurement vocabulary for the rewrite lane.
//!
//! This module exists so benchmarks and diagnostics can observe the rewrite through the public API
//! without baking benchmark-specific code into the library.

use core::time::Duration;

/// The measurement categories mandated by `future/plan/00-foundation.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    ScheduleBuildCost,
    RuntimeQueueOverhead,
    ChunkAllocationCost,
    ResolveCost,
    DenseScanThroughput,
}

/// The foundation measurement plan required before tuning the rewrite.
pub const FOUNDATION_MEASUREMENT_CATEGORIES: [Category; 5] = [
    Category::ScheduleBuildCost,
    Category::RuntimeQueueOverhead,
    Category::ChunkAllocationCost,
    Category::ResolveCost,
    Category::DenseScanThroughput,
];

/// The kind of batched resolve work that completed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResolveKind {
    Insert,
    Remove,
    Mixed,
}

/// A schedule-build measurement emitted after schedule construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduleBuild {
    pub scheduled_function_count: usize,
    pub happens_before_edge_count: usize,
    pub elapsed: Duration,
}

/// A runtime-queue measurement emitted after a burst of job dispatch work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeQueue {
    pub pushed_job_count: usize,
    pub stolen_job_count: usize,
    pub elapsed: Duration,
}

/// A chunk-allocation measurement emitted after allocating storage for a new chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkAllocation {
    pub inline_row_width: usize,
    pub target_chunk_byte_count: usize,
    pub target_chunk_capacity: usize,
    pub elapsed: Duration,
}

/// A batched resolve measurement emitted after applying deferred structural commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolve {
    pub resolve_kind: ResolveKind,
    pub resolved_command_count: usize,
    pub affected_chunk_count: usize,
    pub elapsed: Duration,
}

/// A dense-scan measurement emitted after visiting chunk slices during hot iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseScan {
    pub visited_chunk_count: usize,
    pub visited_row_count: usize,
    pub visited_physical_column_count: usize,
    pub elapsed: Duration,
}

/// A single instrumentation event produced by the rewrite runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    ScheduleBuildCompleted(ScheduleBuild),
    RuntimeQueueCompleted(RuntimeQueue),
    ChunkAllocationCompleted(ChunkAllocation),
    ResolveCompleted(Resolve),
    DenseScanCompleted(DenseScan),
}

/// A thread-safe sink for instrumentation events.
pub trait Sink: Send + Sync + 'static {
    fn record(&self, event: Event);
}

/// The default sink used when callers do not request instrumentation.
#[derive(Debug, Default)]
pub struct NoopSink;

impl NoopSink {
    pub const fn new() -> Self {
        Self
    }
}

impl Sink for NoopSink {
    fn record(&self, _event: Event) {}
}
