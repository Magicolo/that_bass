//! Deferred command vocabulary for the rewrite lane.
//!
//! The rewrite records structural work during parallel job execution and resolves it later in an
//! explicit batched phase. This module defines only the public terms needed for that model.

use crate::v2::schema::{ChunkError, Row, Table};

/// The structural command families planned for the first rewrite milestones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Insert,
    Remove,
    Set,
}

/// The selected resolve strategy for the rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Strategy {
    /// Jobs record commands independently, then a later resolve phase batches all command buffers
    /// produced by the same scheduled function.
    FunctionLevelBatch,
}

/// A local buffer of row-targeted remove commands for keyless tables.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Remove<'job> {
    rows: Vec<Row<'job>>,
}

impl<'job> Remove<'job> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn rows(&self) -> &[Row<'job>] {
        &self.rows
    }

    pub fn one(&mut self, row: Row<'job>) {
        self.rows.push(row);
    }

    pub fn array<const ROW_COUNT: usize>(&mut self, rows: [Row<'job>; ROW_COUNT]) {
        self.rows.extend(rows);
    }

    pub fn slice(&mut self, rows: &[Row<'job>]) {
        self.rows.extend_from_slice(rows);
    }

    pub fn extend<I>(&mut self, rows: I)
    where
        I: IntoIterator<Item = Row<'job>>,
    {
        self.rows.extend(rows);
    }

    pub fn clear(&mut self) {
        self.rows.clear();
    }

    pub fn resolve_on(self, table: &mut Table) -> Result<usize, ChunkError> {
        table.resolve_remove_rows(self.rows)
    }
}
