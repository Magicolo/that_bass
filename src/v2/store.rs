//! Storage boundary for the rewrite lane.
//!
//! This module intentionally exposes only the smallest useful foundation API:
//!
//! - a `Store` root type for the rewrite boundary,
//! - a configuration object with a benchmarkable chunk-byte target,
//! - a chunk-capacity planner that follows the selected specification.

use crate::v2::instrumentation::{NoopSink, Sink};
use crate::v2::schema::{Table, TableIndex};
use core::num::NonZeroUsize;
use std::sync::Arc;

const DEFAULT_TARGET_CHUNK_BYTE_COUNT: usize = 16 * 1024;

/// The public root type for the rewrite lane.
pub struct Store {
    configuration: Configuration,
    instrumentation_sink: Arc<dyn Sink>,
    tables: Vec<Table>,
}

impl Default for Store {
    fn default() -> Self {
        Self::new()
    }
}

impl Store {
    pub fn new() -> Self {
        Self::with_configuration(Configuration::default())
    }

    pub fn with_configuration(configuration: Configuration) -> Self {
        Self::with_instrumentation(configuration, Arc::new(NoopSink::new()))
    }

    pub fn with_instrumentation(
        configuration: Configuration,
        instrumentation_sink: Arc<dyn Sink>,
    ) -> Self {
        Self {
            configuration,
            instrumentation_sink,
            tables: Vec::new(),
        }
    }

    pub fn configuration(&self) -> &Configuration {
        &self.configuration
    }

    pub fn instrumentation_sink(&self) -> &(dyn Sink + Send + Sync + 'static) {
        &*self.instrumentation_sink
    }

    pub fn tables(&self) -> &[Table] {
        &self.tables
    }

    pub fn table_count(&self) -> usize {
        self.tables.len()
    }

    pub fn table(&self, table_index: TableIndex) -> Option<&Table> {
        self.tables.get(table_index.value() as usize)
    }

    pub fn table_mut(&mut self, table_index: TableIndex) -> Option<&mut Table> {
        self.tables.get_mut(table_index.value() as usize)
    }

    pub fn plan_chunk_capacity_for_row_width(&self, inline_row_width: usize) -> ChunkPlan {
        self.configuration
            .plan_chunk_capacity_for_row_width(inline_row_width)
    }
}

/// The mutable tuning data that should remain easy to benchmark and override.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Configuration {
    target_chunk_byte_count: NonZeroUsize,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            target_chunk_byte_count: NonZeroUsize::new(DEFAULT_TARGET_CHUNK_BYTE_COUNT)
                .expect("default target chunk byte count must be non-zero"),
        }
    }
}

impl Configuration {
    pub const fn target_chunk_byte_count(self) -> NonZeroUsize {
        self.target_chunk_byte_count
    }

    pub const fn with_target_chunk_byte_count(
        mut self,
        target_chunk_byte_count: NonZeroUsize,
    ) -> Self {
        self.target_chunk_byte_count = target_chunk_byte_count;
        self
    }

    pub fn plan_chunk_capacity_for_row_width(self, inline_row_width: usize) -> ChunkPlan {
        let normalized_inline_row_width = inline_row_width.max(1);
        let raw_target_row_count =
            (self.target_chunk_byte_count.get() / normalized_inline_row_width).max(1);
        let target_chunk_capacity = previous_power_of_two(raw_target_row_count);

        ChunkPlan {
            target_chunk_byte_count: self.target_chunk_byte_count,
            inline_row_width,
            normalized_inline_row_width,
            raw_target_row_count,
            target_chunk_capacity,
        }
    }

    pub fn target_chunk_capacity_for_row_width(self, inline_row_width: usize) -> usize {
        self.plan_chunk_capacity_for_row_width(inline_row_width)
            .target_chunk_capacity()
    }
}

/// The result of applying the selected chunk-capacity formula to one row width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkPlan {
    target_chunk_byte_count: NonZeroUsize,
    inline_row_width: usize,
    normalized_inline_row_width: usize,
    raw_target_row_count: usize,
    target_chunk_capacity: usize,
}

impl ChunkPlan {
    pub const fn target_chunk_byte_count(self) -> NonZeroUsize {
        self.target_chunk_byte_count
    }

    pub const fn inline_row_width(self) -> usize {
        self.inline_row_width
    }

    pub const fn normalized_inline_row_width(self) -> usize {
        self.normalized_inline_row_width
    }

    pub const fn raw_target_row_count(self) -> usize {
        self.raw_target_row_count
    }

    pub const fn target_chunk_capacity(self) -> usize {
        self.target_chunk_capacity
    }
}

fn previous_power_of_two(value: usize) -> usize {
    1usize << value.ilog2()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_inline_row_width_uses_the_specification_fallback() {
        let configuration = Configuration::default();
        let plan = configuration.plan_chunk_capacity_for_row_width(0);

        assert_eq!(plan.normalized_inline_row_width(), 1);
        assert_eq!(
            plan.raw_target_row_count(),
            configuration.target_chunk_byte_count().get()
        );
    }
}
