//! Deferred command vocabulary and batched resolution for the rewrite lane.
//!
//! Task 07 turns command handling into a real execution boundary:
//!
//! - injected command descriptors initialize themselves against `Store`,
//! - function jobs record into local command buffers,
//! - resolve phases collect those buffers in batch,
//! - and structural visibility updates are reported back to the executor explicitly.

use crate::v2::{
    query::Filter,
    schema::{ChunkError, ChunkIndex, Meta, Row, Table, TableIndex},
    store::Store,
};
use core::{
    any::{type_name, Any, TypeId},
    marker::PhantomData,
};
use std::collections::BTreeMap;

/// Initialization of one injected item against the current store.
pub trait Initialize {
    type State;
    type Error;

    fn initialize(self, store: &mut Store) -> Result<Self::State, Self::Error>;
}

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

/// One visible chunk state produced by command resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkState {
    table_index: TableIndex,
    chunk_index: ChunkIndex,
    row_count: usize,
}

impl ChunkState {
    pub const fn new(table_index: TableIndex, chunk_index: ChunkIndex, row_count: usize) -> Self {
        Self {
            table_index,
            chunk_index,
            row_count,
        }
    }

    pub const fn table_index(self) -> TableIndex {
        self.table_index
    }

    pub const fn chunk_index(self) -> ChunkIndex {
        self.chunk_index
    }

    pub const fn row_count(self) -> usize {
        self.row_count
    }
}

/// Errors while resolving deferred command buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    MissingTable { table_index: TableIndex },
    MissingInsertBuffer { type_name: &'static str },
    UnexpectedRemoveTable { table_index: TableIndex },
    Chunk(ChunkError),
}

impl From<ChunkError> for ResolveError {
    fn from(error: ChunkError) -> Self {
        Self::Chunk(error)
    }
}

/// Descriptor-only insert command capability for one scheduled function.
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
}

impl<T> Initialize for Insert<T>
where
    T: Columns,
{
    type State = Plan;
    type Error = crate::v2::schema::DefinitionError;

    fn initialize(self, store: &mut Store) -> Result<Self::State, Self::Error> {
        let table_index = store.get_or_create_table(T::metas())?;

        Ok(Plan::Insert(InsertPlan::new::<T>(table_index)))
    }
}

/// Descriptor-only remove capability scoped by a table filter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Remove<F = crate::v2::query::AllowAll> {
    filter: F,
}

impl<F> Remove<F> {
    pub const fn new(filter: F) -> Self {
        Self { filter }
    }

    pub const fn filter(&self) -> &F {
        &self.filter
    }
}

impl<F> Initialize for Remove<F>
where
    F: Filter,
{
    type State = Plan;
    type Error = crate::v2::schema::DefinitionError;

    fn initialize(self, store: &mut Store) -> Result<Self::State, Self::Error> {
        let allowed_table_indices = store
            .tables()
            .iter()
            .filter(|table| self.filter.matches(table))
            .map(Table::index)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(Plan::Remove(RemovePlan {
            allowed_table_indices,
        }))
    }
}

/// Type-level column declaration and row application for typed inserts.
pub trait Columns: Sized + Send + 'static {
    fn metas() -> impl Iterator<Item = Meta>;

    /// # Safety
    ///
    /// The caller must ensure that `table`, `chunk_index`, and `row_index` address a valid
    /// append slot for the row shape described by `Self::metas()`, and that the row is later
    /// published only after every inline column has been initialized consistently.
    unsafe fn write_row(
        self,
        table: &mut Table,
        chunk_index: ChunkIndex,
        row_index: usize,
    ) -> Result<(), ChunkError>;
}

macro_rules! impl_columns_tuples {
    ($(($( $type_name:ident : $value_name:ident ),+)),+ $(,)?) => {
        $(
            impl<$($type_name: Send + 'static),+> Columns for ($($type_name,)+) {
                fn metas() -> impl Iterator<Item = Meta> {
                    [$(Meta::of::<$type_name>()),+].into_iter()
                }

                unsafe fn write_row(
                    self,
                    table: &mut Table,
                    chunk_index: ChunkIndex,
                    row_index: usize,
                ) -> Result<(), ChunkError> {
                    let ($($value_name,)+) = self;
                    $(
                        unsafe {
                            table.write::<$type_name>(chunk_index, row_index, $value_name)?;
                        }
                    )+

                    Ok(())
                }
            }
        )+
    };
}

impl_columns_tuples!(
    (T0: value0),
    (T0: value0, T1: value1),
    (T0: value0, T1: value1, T2: value2),
    (T0: value0, T1: value1, T2: value2, T3: value3),
    (T0: value0, T1: value1, T2: value2, T3: value3, T4: value4),
    (T0: value0, T1: value1, T2: value2, T3: value3, T4: value4, T5: value5),
    (T0: value0, T1: value1, T2: value2, T3: value3, T4: value4, T5: value5, T6: value6),
    (T0: value0, T1: value1, T2: value2, T3: value3, T4: value4, T5: value5, T6: value6, T7: value7),
);

/// One planned command descriptor attached to a resolve family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Plan {
    Insert(InsertPlan),
    Remove(RemovePlan),
    Set,
}

impl Plan {
    pub const fn kind(&self) -> Kind {
        match self {
            Self::Insert(_) => Kind::Insert,
            Self::Remove(_) => Kind::Remove,
            Self::Set => Kind::Set,
        }
    }

    pub const fn table_index(&self) -> Option<TableIndex> {
        match self {
            Self::Insert(plan) => Some(plan.table_index()),
            Self::Remove(_) | Self::Set => None,
        }
    }
}

type MakeRows = fn() -> Box<dyn Any + Send>;
type RowsAreEmpty = fn(&dyn Any) -> bool;
type MergeRows = fn(&mut dyn Any, &mut dyn Any);
type ResolveRows = fn(&mut dyn Any, &mut Table) -> Result<Box<[ChunkState]>, ResolveError>;

/// One typed insert plan resolved during initialization.
#[derive(Clone)]
pub struct InsertPlan {
    table_index: TableIndex,
    row_type_identifier: TypeId,
    row_type_name: &'static str,
    make_rows: MakeRows,
    rows_are_empty: RowsAreEmpty,
    merge_rows: MergeRows,
    resolve_rows: ResolveRows,
}

impl core::fmt::Debug for InsertPlan {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("InsertPlan")
            .field("table_index", &self.table_index)
            .field("row_type_identifier", &self.row_type_identifier)
            .field("row_type_name", &self.row_type_name)
            .finish()
    }
}

impl PartialEq for InsertPlan {
    fn eq(&self, other: &Self) -> bool {
        self.table_index == other.table_index
            && self.row_type_identifier == other.row_type_identifier
            && self.row_type_name == other.row_type_name
    }
}

impl Eq for InsertPlan {}

impl InsertPlan {
    fn new<T>(table_index: TableIndex) -> Self
    where
        T: Columns,
    {
        Self {
            table_index,
            row_type_identifier: TypeId::of::<T>(),
            row_type_name: type_name::<T>(),
            make_rows: || Box::new(Vec::<T>::new()),
            rows_are_empty: insert_rows_are_empty::<T>,
            merge_rows: merge_insert_rows::<T>,
            resolve_rows: resolve_insert_rows::<T>,
        }
    }

    pub const fn table_index(&self) -> TableIndex {
        self.table_index
    }

    pub const fn row_type_identifier(&self) -> TypeId {
        self.row_type_identifier
    }

    pub const fn row_type_name(&self) -> &'static str {
        self.row_type_name
    }
}

/// One remove plan resolved during initialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemovePlan {
    allowed_table_indices: Box<[TableIndex]>,
}

impl RemovePlan {
    pub fn allowed_table_indices(&self) -> &[TableIndex] {
        &self.allowed_table_indices
    }
}

/// A batch of local command buffers merged for one resolve phase.
#[derive(Debug)]
pub struct Batch {
    entries: Box<[Entry]>,
}

impl Batch {
    pub fn new(plans: &[Plan]) -> Self {
        Self {
            entries: plans.iter().map(Entry::from_plan).collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.iter().all(Entry::is_empty)
    }

    pub fn merge(&mut self, mut buffer: Buffer) {
        debug_assert_eq!(self.entries.len(), buffer.entries.len());

        for (batch_entry, buffer_entry) in self.entries.iter_mut().zip(buffer.entries.iter_mut()) {
            batch_entry.merge_from(buffer_entry);
        }
    }

    pub fn resolve_on(self, store: &mut Store) -> Result<Box<[ChunkState]>, ResolveError> {
        let mut changed_chunk_states = BTreeMap::<(TableIndex, ChunkIndex), usize>::new();

        for entry in self.entries {
            for chunk_state in entry.resolve_on(store)? {
                changed_chunk_states.insert(
                    (chunk_state.table_index(), chunk_state.chunk_index()),
                    chunk_state.row_count(),
                );
            }
        }

        Ok(changed_chunk_states
            .into_iter()
            .map(|((table_index, chunk_index), row_count)| {
                ChunkState::new(table_index, chunk_index, row_count)
            })
            .collect())
    }
}

/// One local per-job command buffer.
#[derive(Debug)]
pub struct Buffer {
    entries: Box<[Entry]>,
}

impl Buffer {
    pub fn new(plans: &[Plan]) -> Self {
        Self {
            entries: plans.iter().map(Entry::from_plan).collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.iter().all(Entry::is_empty)
    }

    pub fn insert<T>(&mut self) -> Option<InsertRows<'_, T>>
    where
        T: Columns,
    {
        let row_type_identifier = TypeId::of::<T>();

        self.entries.iter_mut().find_map(|entry| match entry {
            Entry::Insert(insert_entry)
                if insert_entry.plan.row_type_identifier() == row_type_identifier =>
            {
                let rows = insert_entry
                    .rows
                    .downcast_mut::<Vec<T>>()
                    .expect("insert row buffer should match its initialized row type");

                Some(InsertRows { rows })
            }
            _ => None,
        })
    }

    pub fn remove(&mut self) -> Option<&mut RemoveRows> {
        self.entries.iter_mut().find_map(|entry| match entry {
            Entry::Remove(remove_entry) => Some(&mut remove_entry.rows),
            _ => None,
        })
    }
}

impl Default for Buffer {
    fn default() -> Self {
        Self {
            entries: Box::new([]),
        }
    }
}

/// Local typed row recording for one initialized insert plan.
#[derive(Debug)]
pub struct InsertRows<'buffer, T> {
    rows: &'buffer mut Vec<T>,
}

impl<'buffer, T> InsertRows<'buffer, T> {
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn one(&mut self, row: T) {
        self.rows.push(row);
    }

    pub fn array<const ROW_COUNT: usize>(&mut self, rows: [T; ROW_COUNT]) {
        self.rows.extend(rows);
    }

    pub fn slice(&mut self, rows: &[T])
    where
        T: Copy,
    {
        self.rows.extend_from_slice(rows);
    }
}

/// A local row-targeted remove buffer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RemoveRows {
    packed_rows: Vec<u64>,
}

impl RemoveRows {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.packed_rows.is_empty()
    }

    pub fn len(&self) -> usize {
        self.packed_rows.len()
    }

    pub fn packed_rows(&self) -> &[u64] {
        &self.packed_rows
    }

    pub fn one<'job>(&mut self, row: Row<'job>) {
        self.packed_rows.push(row.packed());
    }

    pub fn array<'job, const ROW_COUNT: usize>(&mut self, rows: [Row<'job>; ROW_COUNT]) {
        self.packed_rows.extend(rows.into_iter().map(Row::packed));
    }

    pub fn slice<'job>(&mut self, rows: &[Row<'job>]) {
        self.packed_rows
            .extend(rows.iter().copied().map(Row::packed));
    }

    pub fn extend<'job, I>(&mut self, rows: I)
    where
        I: IntoIterator<Item = Row<'job>>,
    {
        self.packed_rows.extend(rows.into_iter().map(Row::packed));
    }

    pub fn clear(&mut self) {
        self.packed_rows.clear();
    }

    pub fn resolve_on(self, table: &mut Table) -> Result<usize, ChunkError> {
        table.resolve_remove_rows(
            self.packed_rows
                .into_iter()
                .map(Row::<'static>::from_packed),
        )
    }
}

#[derive(Debug)]
enum Entry {
    Insert(InsertEntry),
    Remove(RemoveEntry),
}

impl Entry {
    fn from_plan(plan: &Plan) -> Self {
        match plan {
            Plan::Insert(plan) => Self::Insert(InsertEntry {
                plan: plan.clone(),
                rows: (plan.make_rows)(),
            }),
            Plan::Remove(plan) => Self::Remove(RemoveEntry {
                plan: plan.clone(),
                rows: RemoveRows::new(),
            }),
            Plan::Set => unreachable!("set command buffers are not implemented yet"),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Insert(entry) => entry.is_empty(),
            Self::Remove(entry) => entry.rows.is_empty(),
        }
    }

    fn merge_from(&mut self, other: &mut Self) {
        match (self, other) {
            (Self::Insert(left), Self::Insert(right)) => left.merge_from(right),
            (Self::Remove(left), Self::Remove(right)) => {
                left.rows.packed_rows.append(&mut right.rows.packed_rows);
            }
            _ => unreachable!("command buffer entries are created from identical plan lists"),
        }
    }

    fn resolve_on(self, store: &mut Store) -> Result<Box<[ChunkState]>, ResolveError> {
        match self {
            Self::Insert(entry) => entry.resolve_on(store),
            Self::Remove(entry) => entry.resolve_on(store),
        }
    }
}

#[derive(Debug)]
struct InsertEntry {
    plan: InsertPlan,
    rows: Box<dyn Any + Send>,
}

impl InsertEntry {
    fn is_empty(&self) -> bool {
        (self.plan.rows_are_empty)(self.rows.as_ref())
    }

    fn merge_from(&mut self, other: &mut Self) {
        debug_assert_eq!(
            self.plan.row_type_identifier(),
            other.plan.row_type_identifier()
        );

        (self.plan.merge_rows)(self.rows.as_mut(), other.rows.as_mut());
    }

    fn resolve_on(self, store: &mut Store) -> Result<Box<[ChunkState]>, ResolveError> {
        let table = store
            .table_mut(self.plan.table_index())
            .ok_or(ResolveError::MissingTable {
                table_index: self.plan.table_index(),
            })?;
        let mut rows = self.rows;

        (self.plan.resolve_rows)(rows.as_mut(), table)
    }
}

#[derive(Debug)]
struct RemoveEntry {
    plan: RemovePlan,
    rows: RemoveRows,
}

impl RemoveEntry {
    fn resolve_on(self, store: &mut Store) -> Result<Box<[ChunkState]>, ResolveError> {
        let mut packed_rows_by_table = BTreeMap::<TableIndex, Vec<u64>>::new();

        for packed_row in self.rows.packed_rows {
            let row = Row::<'static>::from_packed(packed_row);
            let table_index = row.table_index();

            if !self.plan.allowed_table_indices().contains(&table_index) {
                return Err(ResolveError::UnexpectedRemoveTable { table_index });
            }

            packed_rows_by_table
                .entry(table_index)
                .or_default()
                .push(packed_row);
        }

        let mut changed_chunk_states = BTreeMap::<(TableIndex, ChunkIndex), usize>::new();

        for (table_index, packed_rows) in packed_rows_by_table {
            let table = store
                .table_mut(table_index)
                .ok_or(ResolveError::MissingTable { table_index })?;
            let affected_chunk_indices = packed_rows
                .iter()
                .copied()
                .map(Row::<'static>::from_packed)
                .map(|row| table.row_layout().chunk_index(row))
                .collect::<Vec<_>>();
            table.resolve_remove_rows(packed_rows.into_iter().map(Row::<'static>::from_packed))?;

            for chunk_index in affected_chunk_indices {
                let row_count = table.chunk(chunk_index).map_or(0, |chunk| chunk.count());
                changed_chunk_states.insert((table_index, chunk_index), row_count);
            }
        }

        Ok(changed_chunk_states
            .into_iter()
            .map(|((table_index, chunk_index), row_count)| {
                ChunkState::new(table_index, chunk_index, row_count)
            })
            .collect())
    }
}

fn insert_rows_are_empty<T>(rows: &dyn Any) -> bool
where
    T: Columns,
{
    rows.downcast_ref::<Vec<T>>()
        .unwrap_or_else(|| panic!("insert buffer should stay typed as {}", type_name::<T>()))
        .is_empty()
}

fn merge_insert_rows<T>(left: &mut dyn Any, right: &mut dyn Any)
where
    T: Columns,
{
    let left = left
        .downcast_mut::<Vec<T>>()
        .unwrap_or_else(|| panic!("insert buffer should stay typed as {}", type_name::<T>()));
    let right = right
        .downcast_mut::<Vec<T>>()
        .unwrap_or_else(|| panic!("insert buffer should stay typed as {}", type_name::<T>()));
    left.append(right);
}

fn resolve_insert_rows<T>(
    rows: &mut dyn Any,
    table: &mut Table,
) -> Result<Box<[ChunkState]>, ResolveError>
where
    T: Columns,
{
    let rows = rows
        .downcast_mut::<Vec<T>>()
        .ok_or(ResolveError::MissingInsertBuffer {
            type_name: type_name::<T>(),
        })?;
    if rows.is_empty() {
        return Ok(Box::new([]));
    }

    let mut changed_chunk_states = BTreeMap::<ChunkIndex, usize>::new();
    let mut current_chunk_index = table
        .chunks()
        .iter()
        .find(|chunk| !chunk.is_full())
        .map(|chunk| chunk.chunk_index());

    for row in rows.drain(..) {
        let chunk_index = match current_chunk_index {
            Some(chunk_index) => {
                let chunk = table
                    .chunk(chunk_index)
                    .ok_or(ChunkError::MissingChunk { chunk_index })?;
                if chunk.is_full() {
                    let chunk_index = table.push_chunk();
                    current_chunk_index = Some(chunk_index);
                    chunk_index
                } else {
                    chunk_index
                }
            }
            None => {
                let chunk_index = table.push_chunk();
                current_chunk_index = Some(chunk_index);
                chunk_index
            }
        };

        let row_index = table
            .chunk(chunk_index)
            .ok_or(ChunkError::MissingChunk { chunk_index })?
            .count();
        unsafe {
            T::write_row(row, table, chunk_index, row_index)?;
            table.assume_initialized_prefix(chunk_index, row_index + 1)?;
        }
        let row_count = table
            .chunk(chunk_index)
            .ok_or(ChunkError::MissingChunk { chunk_index })?
            .count();
        changed_chunk_states.insert(chunk_index, row_count);
    }

    Ok(changed_chunk_states
        .into_iter()
        .map(|(chunk_index, row_count)| ChunkState::new(table.index(), chunk_index, row_count))
        .collect())
}
