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

use crate::v2::{
    query::Access,
    schedule::{Dependency, Inject, Resource},
};

/// The command buffer injected into scheduled functions.
#[derive(Debug, Clone, Default)]
pub struct Commands {
    // For now, just handles Removes. In the future, this might hold multiple kinds of buffers.
}

impl Commands {
    pub const fn new() -> Self {
        Self {}
    }
}

use crate::v2::schedule::{Job, ResolveJob};
use std::sync::Arc;

impl<T> Inject<T> for Commands {
    type Item<'job> = &'job mut Remove<'job>;

    fn static_accesses(&self) -> Vec<crate::v2::query::DeclaredAccess> {
        Vec::new()
    }

    fn generate_jobs<F>(
        &self,
        _data: &T,
        function_id: usize,
        f: Arc<F>,
        jobs: &mut Vec<Job<T>>,
        resolve_jobs: &mut Vec<ResolveJob<T>>,
    ) where
        F: Fn(Self::Item<'_>) + Send + Sync + 'static,
    {
        // For a standalone Commands injection, we create a single job that runs `f` once,
        // providing a new command buffer.
        let run_f = f.clone();
        jobs.push(Job {
            function_id,
            dependencies: Vec::new(),
            run: Box::new(move |_data_ptr| {
                // Task 06 will correctly route this buffer to resolve jobs.
                // For now, we stub a temporary buffer just to satisfy the API.
                let mut temp_remove = Remove::default();
                run_f(&mut temp_remove);
                false
            }),
        });

        resolve_jobs.push(ResolveJob {
            function_id,
            dependencies: vec![Dependency {
                resource: Resource::Store,
                access: Access::Write,
            }],
            run: Box::new(move |_data_ptr| {
                // Task 06 resolve logic would run here.
                false
            }),
        });
    }
}
