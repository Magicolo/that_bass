//! Deferred command vocabulary for the rewrite lane.
//!
//! The rewrite records structural work during parallel job execution and resolves it later in an
//! explicit batched phase. This module defines only the public terms needed for that model.

use crate::v2::schema::{ChunkError, Meta, Row, Table, TableIndex};
use core::marker::PhantomData;

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

/// Descriptor-only insert command capability for one scheduled function.
///
/// Task 05 uses this to predeclare the target table of typed insert work before any runtime
/// command-buffer implementation exists. Task 07 will extend the same public term into the real
/// local-buffer API.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Insert<T> {
    marker: PhantomData<fn() -> T>,
}

impl<T> Insert<T> {
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    pub const fn plan(self, table_index: TableIndex) -> Plan {
        Plan::Insert { table_index }
    }
}

/// Type-level column declaration for typed insert descriptors.
pub trait Columns {
    fn metas() -> impl Iterator<Item = Meta>;
}

macro_rules! impl_columns_tuples {
    ($(($( $type_name:ident ),+)),+ $(,)?) => {
        $(
            impl<$($type_name: 'static),+> Columns for ($($type_name,)+) {
                fn metas() -> impl Iterator<Item = Meta> {
                    [$(Meta::of::<$type_name>()),+].into_iter()
                }
            }
        )+
    };
}

impl_columns_tuples!(
    (T0),
    (T0, T1),
    (T0, T1, T2),
    (T0, T1, T2, T3),
    (T0, T1, T2, T3, T4),
    (T0, T1, T2, T3, T4, T5),
    (T0, T1, T2, T3, T4, T5, T6),
    (T0, T1, T2, T3, T4, T5, T6, T7),
);

/// One planned command descriptor attached to a resolve family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Plan {
    Insert { table_index: TableIndex },
    Remove,
    Set,
}

impl Plan {
    pub const fn kind(self) -> Kind {
        match self {
            Self::Insert { .. } => Kind::Insert,
            Self::Remove => Kind::Remove,
            Self::Set => Kind::Set,
        }
    }

    pub const fn table_index(self) -> Option<TableIndex> {
        match self {
            Self::Insert { table_index } => Some(table_index),
            Self::Remove | Self::Set => None,
        }
    }
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
