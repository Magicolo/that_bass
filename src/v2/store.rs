//! Storage boundary for the rewrite lane.
//!
//! This module intentionally exposes only the smallest useful foundation API:
//!
//! - a `Store` root type for the rewrite boundary,
//! - a configuration object with a benchmarkable chunk-byte target,
//! - a chunk-capacity planner that follows the selected specification,
//! - the initialization-time table registration surface used by scheduled injections,
//! - and a direct exclusive-mode helper for singleton-like global tables.

use crate::v2::schema::{Catalog, ChunkError, DefinitionError, Meta, Table, TableIndex};
use crate::v2::{
    command::Columns,
    instrumentation::{NoopSink, Sink},
    key,
};
use core::num::NonZeroUsize;
use std::sync::Arc;

const DEFAULT_TARGET_CHUNK_BYTE_COUNT: usize = 16 * 1024;

/// The public root type for the rewrite lane.
pub struct Store {
    configuration: Configuration,
    instrumentation_sink: Arc<dyn Sink>,
    catalog: Catalog,
    keys: Option<key::Keys>,
    tables: Vec<Table>,
}

/// Errors produced while initializing or resetting a singleton-like global table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlobalError {
    Definition(DefinitionError),
    Chunk(ChunkError),
    InvalidRowCount {
        table_index: TableIndex,
        count: usize,
    },
}

impl From<DefinitionError> for GlobalError {
    fn from(error: DefinitionError) -> Self {
        Self::Definition(error)
    }
}

impl From<ChunkError> for GlobalError {
    fn from(error: ChunkError) -> Self {
        Self::Chunk(error)
    }
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

    #[doc(hidden)]
    pub fn with_instrumentation(
        configuration: Configuration,
        instrumentation_sink: Arc<dyn Sink>,
    ) -> Self {
        Self {
            configuration,
            instrumentation_sink,
            catalog: Catalog::new(),
            keys: None,
            tables: Vec::new(),
        }
    }

    pub fn configuration(&self) -> &Configuration {
        &self.configuration
    }

    #[doc(hidden)]
    pub fn instrumentation_sink(&self) -> &(dyn Sink + Send + Sync + 'static) {
        &*self.instrumentation_sink
    }

    #[doc(hidden)]
    pub fn tables(&self) -> &[Table] {
        &self.tables
    }

    pub fn keys(&self) -> Option<crate::v2::Keys> {
        self.keys.clone()
    }

    pub fn builder(&mut self) -> crate::v2::Builder<'_> {
        crate::v2::Builder::new(self)
    }

    #[doc(hidden)]
    pub fn initialize_keys(&mut self) -> key::Keys {
        let keys = self.keys.get_or_insert_with(key::Keys::new);
        keys.clone()
    }

    #[doc(hidden)]
    pub fn table_count(&self) -> usize {
        self.tables.len()
    }

    #[doc(hidden)]
    pub fn table(&self, table_index: TableIndex) -> Option<&Table> {
        self.tables.get(table_index.value() as usize)
    }

    #[doc(hidden)]
    pub fn table_mut(&mut self, table_index: TableIndex) -> Option<&mut Table> {
        self.tables.get_mut(table_index.value() as usize)
    }

    pub fn register<T>(&mut self) -> Result<TableIndex, DefinitionError>
    where
        T: Send + 'static,
    {
        self.register_table([Meta::of::<T>()])
    }

    pub fn ensure<T>(&mut self) -> Result<TableIndex, DefinitionError>
    where
        T: Send + 'static,
    {
        self.get_or_create_table([Meta::of::<T>()])
    }

    pub fn register_row<T>(&mut self) -> Result<TableIndex, DefinitionError>
    where
        T: Columns,
    {
        self.register_table(T::metas())
    }

    pub fn ensure_row<T>(&mut self) -> Result<TableIndex, DefinitionError>
    where
        T: Columns,
    {
        self.get_or_create_table(T::metas())
    }

    #[doc(hidden)]
    pub fn register_table<I>(&mut self, metas: I) -> Result<TableIndex, DefinitionError>
    where
        I: IntoIterator<Item = Meta>,
    {
        let table = self.catalog.register_table(metas, self.configuration)?;
        let table_index = table.index();
        self.tables.push(table);
        Ok(table_index)
    }

    #[doc(hidden)]
    pub fn get_or_create_table<I>(&mut self, metas: I) -> Result<TableIndex, DefinitionError>
    where
        I: IntoIterator<Item = Meta>,
    {
        self.catalog
            .get_or_create_table(&mut self.tables, metas, self.configuration)
    }

    pub fn initialize_global<T>(&mut self, value: T) -> Result<TableIndex, GlobalError>
    where
        T: Send + 'static,
    {
        let table_index = self.get_or_create_table([Meta::of::<T>()])?;
        let table = self
            .table_mut(table_index)
            .expect("registered global table should stay addressable");
        let existing_row = find_single_row(table)?;

        if let Some((chunk_index, row_index)) = existing_row {
            table.swap_remove_row(chunk_index, row_index)?;
        }

        let (chunk_index, row_index) = ensure_append_slot(table);
        // Safety: the chosen slot is the append position of a chunk inside the current
        // initialized-prefix state, and the row is published only after the addressed singleton
        // value has been written.
        unsafe {
            table.write::<T>(chunk_index, row_index, value)?;
            table.assume_initialized_prefix(chunk_index, row_index + 1)?;
        }

        Ok(table_index)
    }

    #[doc(hidden)]
    pub fn table_shape_count(&self) -> usize {
        self.catalog.table_shape_count()
    }

    pub fn plan_chunk_capacity_for_row_width(&self, inline_row_width: usize) -> ChunkPlan {
        self.configuration
            .plan_chunk_capacity_for_row_width(inline_row_width)
    }
}

fn find_single_row(
    table: &Table,
) -> Result<Option<(crate::v2::schema::ChunkIndex, usize)>, GlobalError> {
    let mut found_chunk_index = None;
    let mut total_row_count = 0usize;

    for chunk in table.chunks() {
        total_row_count += chunk.count();
        if chunk.count() != 0 {
            found_chunk_index = Some(chunk.chunk_index());
        }
    }

    match total_row_count {
        0 => Ok(None),
        1 => Ok(Some((
            found_chunk_index.expect("one-row global table should expose its inhabited chunk"),
            0,
        ))),
        count => Err(GlobalError::InvalidRowCount {
            table_index: table.index(),
            count,
        }),
    }
}

fn ensure_append_slot(table: &mut Table) -> (crate::v2::schema::ChunkIndex, usize) {
    for chunk in table.chunks() {
        if !chunk.is_full() {
            return (chunk.chunk_index(), chunk.count());
        }
    }

    let chunk_index = table.push_chunk();
    (chunk_index, 0)
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
