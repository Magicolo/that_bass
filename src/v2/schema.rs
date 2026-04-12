//! Table metadata, chunk layout, and scheduler-resource model for the rewrite lane.
//!
//! The current `v2` terminology is intentionally aligned with the selected storage vocabulary:
//!
//! - `Meta` describes one stored type,
//! - `Row` identifies one row inside one chunk of one table,
//! - `Column` is a runtime wrapper around a chunk pointer plus one `Meta`,
//! - `Chunk` groups column pointers for one allocation,
//! - `Table` owns `Meta` descriptors and `Chunk` storage,
//! - `Store` later owns tables.
//!
//! Task 01 established the names and resource identifiers. Task 02 turns those names into a real
//! chunk layout with:
//!
//! - precomputed single-allocation chunk layouts,
//! - geometric bootstrap chunk growth,
//! - dense inhabited prefixes,
//! - and low-level direct table/chunk operations that later tasks can build on.

use crate::v2::{
    query::Access,
    store::{ChunkPlan, Configuration},
};
use core::{
    alloc::Layout,
    any::{type_name, TypeId},
    iter::FusedIterator,
    marker::PhantomData,
    mem::{align_of, needs_drop, size_of},
    num::NonZeroUsize,
    ptr::NonNull,
    slice::{from_raw_parts, from_raw_parts_mut},
};
use std::{
    alloc::{alloc, dealloc, handle_alloc_error},
    collections::BTreeMap,
};

/// The storage class selected for one stored type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Storage {
    Inline,
    Sidecar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct MetaSignature {
    identifier: TypeId,
    name: &'static str,
    element_size: usize,
    element_alignment: usize,
    storage: Storage,
}

impl MetaSignature {
    const fn from_meta(meta: Meta) -> Self {
        Self {
            identifier: meta.identifier(),
            name: meta.name(),
            element_size: meta.element_size(),
            element_alignment: meta.element_alignment(),
            storage: meta.storage(),
        }
    }
}

/// A stable store index at the root of the scheduler resource tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StoreIndex(u32);

impl StoreIndex {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u32 {
        self.0
    }
}

/// A stable table-descriptor index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TableIndex(u32);

impl TableIndex {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u32 {
        self.0
    }
}

/// A chunk index within one table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkIndex(u32);

impl ChunkIndex {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u32 {
        self.0
    }
}

/// A stable column index inside one table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ColumnIndex(u16);

impl ColumnIndex {
    pub const fn new(value: u16) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u16 {
        self.0
    }
}

/// A scheduler-visible identifier at any scope of the store resource tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Resource {
    Store(StoreIndex),
    Table {
        store_index: StoreIndex,
        table_index: TableIndex,
    },
    Chunk {
        store_index: StoreIndex,
        table_index: TableIndex,
        chunk_index: ChunkIndex,
    },
    Column {
        store_index: StoreIndex,
        table_index: TableIndex,
        chunk_index: ChunkIndex,
        column_index: ColumnIndex,
    },
}

impl Resource {
    pub const fn store(store_index: StoreIndex) -> Self {
        Self::Store(store_index)
    }

    pub const fn table(store_index: StoreIndex, table_index: TableIndex) -> Self {
        Self::Table {
            store_index,
            table_index,
        }
    }

    pub const fn chunk(
        store_index: StoreIndex,
        table_index: TableIndex,
        chunk_index: ChunkIndex,
    ) -> Self {
        Self::Chunk {
            store_index,
            table_index,
            chunk_index,
        }
    }

    pub const fn column(
        store_index: StoreIndex,
        table_index: TableIndex,
        chunk_index: ChunkIndex,
        column_index: ColumnIndex,
    ) -> Self {
        Self::Column {
            store_index,
            table_index,
            chunk_index,
            column_index,
        }
    }

    pub const fn store_index(self) -> StoreIndex {
        match self {
            Self::Store(store_index)
            | Self::Table { store_index, .. }
            | Self::Chunk { store_index, .. }
            | Self::Column { store_index, .. } => store_index,
        }
    }

    pub const fn table_index(self) -> Option<TableIndex> {
        match self {
            Self::Table { table_index, .. }
            | Self::Chunk { table_index, .. }
            | Self::Column { table_index, .. } => Some(table_index),
            Self::Store(_) => None,
        }
    }

    pub const fn chunk_index(self) -> Option<ChunkIndex> {
        match self {
            Self::Chunk { chunk_index, .. } | Self::Column { chunk_index, .. } => Some(chunk_index),
            Self::Store(_) | Self::Table { .. } => None,
        }
    }

    pub const fn column_index(self) -> Option<ColumnIndex> {
        match self {
            Self::Column { column_index, .. } => Some(column_index),
            Self::Store(_) | Self::Table { .. } | Self::Chunk { .. } => None,
        }
    }
}

/// One scheduler dependency request expressed as an access mode plus a resource identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dependency {
    resource: Resource,
    access: Access,
}

impl Dependency {
    pub const fn new(resource: Resource, access: Access) -> Self {
        Self { resource, access }
    }

    pub const fn read(resource: Resource) -> Self {
        Self::new(resource, Access::Read)
    }

    pub const fn write(resource: Resource) -> Self {
        Self::new(resource, Access::Write)
    }

    pub const fn resource(self) -> Resource {
        self.resource
    }

    pub const fn access(self) -> Access {
        self.access
    }
}

/// Metadata-registration failures for tables and chunk layouts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefinitionError {
    DuplicateMeta {
        meta_name: &'static str,
    },
    InvalidChunkCapacity {
        capacity: usize,
    },
    TooManyTables,
    TooManyColumns,
    AllocationLayoutOverflow {
        meta_name: &'static str,
        capacity: usize,
    },
}

/// Direct chunk/table storage access failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkError {
    MissingChunk {
        chunk_index: ChunkIndex,
    },
    MissingColumnForType {
        type_name: &'static str,
    },
    ChunkIndexOutOfBounds {
        chunk_index: usize,
        capacity: usize,
    },
    ColumnIndexOutOfBounds {
        column_index: ColumnIndex,
        column_count: usize,
    },
    ColumnNotInline {
        meta_name: &'static str,
    },
    ColumnTypeMismatch {
        column_name: &'static str,
        requested_type_name: &'static str,
    },
    CountExceedsCapacity {
        count: usize,
        capacity: usize,
    },
    RowIndexOutOfBounds {
        row_index: usize,
        capacity: usize,
    },
    RowIndexOutsideInitializedPrefix {
        row_index: usize,
        count: usize,
    },
    RowTableMismatch {
        expected_table_index: TableIndex,
        actual_table_index: TableIndex,
    },
}

/// The bit partition used by the packed `Row<'job>` representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowLayout {
    row_index_bit_count: u8,
    chunk_index_bit_count: u8,
}

impl RowLayout {
    pub fn try_for_chunk_capacity(target_chunk_capacity: usize) -> Result<Self, DefinitionError> {
        if target_chunk_capacity < 1 || !target_chunk_capacity.is_power_of_two() {
            return Err(DefinitionError::InvalidChunkCapacity {
                capacity: target_chunk_capacity,
            });
        }

        if let Some(maximum_row_capacity) = 1usize.checked_shl(u32::BITS) {
            if target_chunk_capacity > maximum_row_capacity {
                return Err(DefinitionError::InvalidChunkCapacity {
                    capacity: target_chunk_capacity,
                });
            }
        }

        let row_index_bit_count = target_chunk_capacity.ilog2() as u8;
        let chunk_index_bit_count = 32u8.saturating_sub(row_index_bit_count);

        Ok(Self {
            row_index_bit_count,
            chunk_index_bit_count,
        })
    }

    pub const fn row_index_bit_count(self) -> u8 {
        self.row_index_bit_count
    }

    pub const fn chunk_index_bit_count(self) -> u8 {
        self.chunk_index_bit_count
    }

    pub fn row<'job>(
        self,
        table_index: TableIndex,
        chunk_index: ChunkIndex,
        row_index: u32,
    ) -> Result<Row<'job>, ChunkError> {
        if row_index > self.maximum_row_index() {
            return Err(ChunkError::RowIndexOutOfBounds {
                row_index: row_index as usize,
                capacity: self.maximum_row_index() as usize + 1,
            });
        }

        if chunk_index.value() > self.maximum_chunk_index() {
            return Err(ChunkError::ChunkIndexOutOfBounds {
                chunk_index: chunk_index.value() as usize,
                capacity: self.maximum_chunk_index() as usize + 1,
            });
        }

        Ok(self.pack_row(table_index, chunk_index, row_index))
    }

    pub const fn chunk_index<'job>(self, row: Row<'job>) -> ChunkIndex {
        if self.chunk_index_bit_count == 0 {
            ChunkIndex::new(0)
        } else {
            ChunkIndex::new(row.packed_chunk_and_row_index() >> self.row_index_bit_count)
        }
    }

    pub const fn row_index<'job>(self, row: Row<'job>) -> u32 {
        if self.row_index_bit_count == 32 {
            row.packed_chunk_and_row_index()
        } else {
            row.packed_chunk_and_row_index() & self.row_index_mask()
        }
    }

    const fn maximum_row_index(self) -> u32 {
        if self.row_index_bit_count == 32 {
            u32::MAX
        } else {
            self.row_index_mask()
        }
    }

    const fn maximum_chunk_index(self) -> u32 {
        if self.chunk_index_bit_count == 32 {
            u32::MAX
        } else if self.chunk_index_bit_count == 0 {
            0
        } else {
            (1u32 << self.chunk_index_bit_count) - 1
        }
    }

    const fn row_index_mask(self) -> u32 {
        if self.row_index_bit_count == 32 {
            u32::MAX
        } else if self.row_index_bit_count == 0 {
            0
        } else {
            (1u32 << self.row_index_bit_count) - 1
        }
    }

    fn pack_row<'job>(
        self,
        table_index: TableIndex,
        chunk_index: ChunkIndex,
        row_index: u32,
    ) -> Row<'job> {
        debug_assert!(row_index <= self.maximum_row_index());
        debug_assert!(chunk_index.value() <= self.maximum_chunk_index());

        let packed_chunk_and_row_index = if self.row_index_bit_count == 32 {
            row_index
        } else {
            (chunk_index.value() << self.row_index_bit_count) | row_index
        };

        let packed =
            (u64::from(table_index.value()) << u32::BITS) | u64::from(packed_chunk_and_row_index);

        Row {
            packed,
            marker: PhantomData,
        }
    }
}

/// A row index into a chunk of a table of a store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Row<'job> {
    packed: u64,
    marker: PhantomData<&'job ()>,
}

impl<'job> Row<'job> {
    pub const fn table_index(self) -> TableIndex {
        TableIndex::new((self.packed >> u32::BITS) as u32)
    }

    pub const fn packed_chunk_and_row_index(self) -> u32 {
        self.packed as u32
    }

    pub const fn packed(self) -> u64 {
        self.packed
    }
}

/// A generated chunk-aligned view of transient row handles.
///
/// `Rows<'job>` behaves like a slice-shaped view even though no physical row column exists in
/// storage. Each yielded handle is derived lazily from the table index, chunk index, and the
/// current inhabited row range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rows<'job> {
    table_index: TableIndex,
    chunk_index: ChunkIndex,
    row_layout: RowLayout,
    row_index_start: u32,
    count: usize,
    marker: PhantomData<&'job ()>,
}

impl<'job> Rows<'job> {
    const fn new(
        table_index: TableIndex,
        chunk_index: ChunkIndex,
        row_layout: RowLayout,
        row_index_start: u32,
        count: usize,
    ) -> Self {
        Self {
            table_index,
            chunk_index,
            row_layout,
            row_index_start,
            count,
            marker: PhantomData,
        }
    }

    pub const fn table_index(self) -> TableIndex {
        self.table_index
    }

    pub const fn chunk_index(self) -> ChunkIndex {
        self.chunk_index
    }

    pub const fn row_layout(self) -> RowLayout {
        self.row_layout
    }

    pub const fn len(&self) -> usize {
        self.count
    }

    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn get(&self, index: usize) -> Option<Row<'job>> {
        self.row_for_offset(index)
    }

    pub fn first(&self) -> Option<Row<'job>> {
        self.get(0)
    }

    pub fn last(&self) -> Option<Row<'job>> {
        self.count.checked_sub(1).and_then(|index| self.get(index))
    }

    pub fn iter(&self) -> RowsIter<'job> {
        RowsIter::new(*self)
    }

    pub fn split_at(&self, midpoint: usize) -> Option<(Self, Self)> {
        if midpoint > self.count {
            return None;
        }

        Some((
            self.subrows(0, midpoint),
            self.subrows(midpoint, self.count - midpoint),
        ))
    }

    pub fn zip<I>(self, other: I) -> core::iter::Zip<RowsIter<'job>, I::IntoIter>
    where
        I: IntoIterator,
    {
        self.into_iter().zip(other)
    }

    fn subrows(&self, start_offset: usize, count: usize) -> Self {
        let row_index_start = if count == 0 {
            self.row_index_start
        } else {
            self.row_index_start
                .checked_add(
                    u32::try_from(start_offset)
                        .expect("row offsets must fit inside a packed row handle"),
                )
                .expect("subrow start index must remain representable")
        };

        debug_assert!(usize::try_from(row_index_start)
            .ok()
            .and_then(|start| start.checked_add(count))
            .is_some());

        Self::new(
            self.table_index,
            self.chunk_index,
            self.row_layout,
            row_index_start,
            count,
        )
    }

    fn row_for_offset(&self, offset: usize) -> Option<Row<'job>> {
        if offset >= self.count {
            return None;
        }

        Some(self.row_for_offset_unchecked(offset))
    }

    fn row_for_offset_unchecked(&self, offset: usize) -> Row<'job> {
        debug_assert!(offset < self.count);

        let row_index = self
            .row_index_start
            .checked_add(
                u32::try_from(offset).expect("row offsets must fit inside a packed row handle"),
            )
            .expect("row indices inside a rows view must remain representable");

        self.row_layout
            .pack_row(self.table_index, self.chunk_index, row_index)
    }
}

impl<'job> IntoIterator for Rows<'job> {
    type IntoIter = RowsIter<'job>;
    type Item = Row<'job>;

    fn into_iter(self) -> Self::IntoIter {
        RowsIter::new(self)
    }
}

impl<'job> IntoIterator for &Rows<'job> {
    type IntoIter = RowsIter<'job>;
    type Item = Row<'job>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over a generated `Rows<'job>` view.
#[derive(Debug, Clone)]
pub struct RowsIter<'job> {
    rows: Rows<'job>,
    next_offset: usize,
    end_offset: usize,
}

impl<'job> RowsIter<'job> {
    const fn new(rows: Rows<'job>) -> Self {
        Self {
            rows,
            next_offset: 0,
            end_offset: rows.count,
        }
    }
}

impl<'job> Iterator for RowsIter<'job> {
    type Item = Row<'job>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_offset == self.end_offset {
            return None;
        }

        let row = self.rows.row_for_offset_unchecked(self.next_offset);
        self.next_offset += 1;
        Some(row)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end_offset.saturating_sub(self.next_offset);
        (remaining, Some(remaining))
    }
}

impl<'job> DoubleEndedIterator for RowsIter<'job> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.next_offset == self.end_offset {
            return None;
        }

        self.end_offset -= 1;
        Some(self.rows.row_for_offset_unchecked(self.end_offset))
    }
}

impl<'job> ExactSizeIterator for RowsIter<'job> {}

impl<'job> FusedIterator for RowsIter<'job> {}

/// The layout metrics that matter for chunk-capacity planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableLayout {
    row_width: usize,
    column_count: usize,
}

impl TableLayout {
    pub const fn new(row_width: usize, column_count: usize) -> Self {
        Self {
            row_width,
            column_count,
        }
    }

    pub const fn row_width(self) -> usize {
        self.row_width
    }

    pub const fn column_count(self) -> usize {
        self.column_count
    }
}

/// Type metadata for one type stored in a column.
#[derive(Debug, Clone, Copy)]
pub struct Meta {
    identifier: TypeId,
    name: &'static str,
    element_size: usize,
    element_alignment: usize,
    storage: Storage,
    drop_value: Option<unsafe fn(*mut u8)>,
    copy_value: Option<unsafe fn(*const u8, *mut u8)>,
}

impl Meta {
    pub fn of<T: 'static>() -> Self {
        Self::inline::<T>()
    }

    pub fn inline<T: 'static>() -> Self {
        Self {
            identifier: TypeId::of::<T>(),
            name: type_name::<T>(),
            element_size: size_of::<T>(),
            element_alignment: align_of::<T>(),
            storage: Storage::Inline,
            drop_value: if needs_drop::<T>() {
                Some(drop_value::<T>)
            } else {
                None
            },
            copy_value: if size_of::<T>() > 0 {
                Some(copy_value::<T>)
            } else {
                None
            },
        }
    }

    pub fn sidecar<T: 'static>() -> Self {
        Self {
            storage: Storage::Sidecar,
            ..Self::inline::<T>()
        }
    }

    pub const fn identifier(self) -> TypeId {
        self.identifier
    }

    pub const fn name(self) -> &'static str {
        self.name
    }

    pub const fn element_size(self) -> usize {
        self.element_size
    }

    pub const fn element_alignment(self) -> usize {
        self.element_alignment
    }

    pub const fn storage(self) -> Storage {
        self.storage
    }

    pub const fn is_inline(self) -> bool {
        matches!(self.storage, Storage::Inline)
    }

    pub fn layout(self) -> Layout {
        Layout::from_size_align(self.element_size, self.element_alignment)
            .expect("meta layout must be valid")
    }

    pub const fn drop_value(self) -> Option<unsafe fn(*mut u8)> {
        self.drop_value
    }

    pub const fn copy_value(self) -> Option<unsafe fn(*const u8, *mut u8)> {
        self.copy_value
    }
}

impl PartialEq for Meta {
    fn eq(&self, other: &Self) -> bool {
        self.identifier == other.identifier
            && self.name == other.name
            && self.element_size == other.element_size
            && self.element_alignment == other.element_alignment
            && self.storage == other.storage
    }
}

impl Eq for Meta {}

unsafe fn drop_value<T>(pointer: *mut u8) {
    // Safety: the caller promises that `pointer` names a valid initialized `T`.
    unsafe {
        pointer.cast::<T>().drop_in_place();
    }
}

unsafe fn copy_value<T>(source: *const u8, target: *mut u8) {
    // Safety: the caller promises that `source` and `target` are valid non-overlapping locations
    // for a `T`. This hook matches the raw storage-copy model used by `v1`; it is not a
    // user-facing clone operation.
    unsafe {
        core::ptr::copy_nonoverlapping(source.cast::<T>(), target.cast::<T>(), 1);
    }
}

/// A runtime wrapper around one column pointer inside a chunk.
#[derive(Debug, Clone, Copy)]
pub struct Column<'a> {
    pointer: NonNull<u8>,
    meta: &'a Meta,
}

impl<'a> Column<'a> {
    pub const fn new(pointer: NonNull<u8>, meta: &'a Meta) -> Self {
        Self { pointer, meta }
    }

    pub const fn pointer(self) -> NonNull<u8> {
        self.pointer
    }

    pub const fn meta(self) -> &'a Meta {
        self.meta
    }
}

/// The precomputed region information for one column inside one chunk capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnLayout {
    offset: Option<usize>,
    region_size: usize,
    alignment: usize,
    storage: Storage,
}

impl ColumnLayout {
    const fn new(
        offset: Option<usize>,
        region_size: usize,
        alignment: usize,
        storage: Storage,
    ) -> Self {
        Self {
            offset,
            region_size,
            alignment,
            storage,
        }
    }

    pub const fn offset(self) -> Option<usize> {
        self.offset
    }

    pub const fn region_size(self) -> usize {
        self.region_size
    }

    pub const fn alignment(self) -> usize {
        self.alignment
    }

    pub const fn storage(self) -> Storage {
        self.storage
    }

    pub const fn has_inline_region(self) -> bool {
        self.offset.is_some()
    }
}

/// The precomputed allocation layout for one concrete chunk capacity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkLayout {
    capacity: usize,
    allocation_size: usize,
    allocation_alignment: usize,
    columns: Box<[ColumnLayout]>,
}

impl ChunkLayout {
    fn new(metas: &[Meta], capacity: usize) -> Result<Self, DefinitionError> {
        if capacity < 1 || !capacity.is_power_of_two() {
            return Err(DefinitionError::InvalidChunkCapacity { capacity });
        }

        let mut allocation_layout = Layout::from_size_align(0, 1).expect("zero layout is valid");
        let mut columns = Vec::with_capacity(metas.len());

        for meta in metas {
            match region_layout_for_meta(*meta, capacity)? {
                Some(region_layout) => {
                    let (combined_layout, offset) = allocation_layout
                        .extend(region_layout)
                        .map_err(|_| DefinitionError::AllocationLayoutOverflow {
                            meta_name: meta.name(),
                            capacity,
                        })?;
                    allocation_layout = combined_layout;
                    columns.push(ColumnLayout::new(
                        Some(offset),
                        region_layout.size(),
                        region_layout.align(),
                        meta.storage(),
                    ));
                }
                None => {
                    columns.push(ColumnLayout::new(
                        None,
                        0,
                        meta.element_alignment(),
                        meta.storage(),
                    ));
                }
            }
        }

        let allocation_layout = allocation_layout.pad_to_align();

        Ok(Self {
            capacity,
            allocation_size: allocation_layout.size(),
            allocation_alignment: allocation_layout.align(),
            columns: columns.into_boxed_slice(),
        })
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    pub const fn allocation_size(&self) -> usize {
        self.allocation_size
    }

    pub const fn allocation_alignment(&self) -> usize {
        self.allocation_alignment
    }

    pub fn columns(&self) -> &[ColumnLayout] {
        &self.columns
    }

    pub fn column(&self, column_index: ColumnIndex) -> Option<&ColumnLayout> {
        self.columns.get(usize::from(column_index.value()))
    }

    fn allocation_layout(&self) -> Layout {
        Layout::from_size_align(self.allocation_size, self.allocation_alignment)
            .expect("stored chunk layout must remain valid")
    }
}

/// One chunk allocation described as a list of per-column pointers.
#[derive(Debug)]
pub struct Chunk {
    chunk_index: ChunkIndex,
    count: usize,
    capacity: usize,
    allocation_pointer: NonNull<u8>,
    allocation_size: usize,
    allocation_alignment: usize,
    meta_pointer: NonNull<Meta>,
    meta_count: usize,
    pointers: Box<[NonNull<u8>]>,
}

impl Chunk {
    fn new(
        chunk_index: ChunkIndex,
        chunk_layout: &ChunkLayout,
        meta_pointer: NonNull<Meta>,
        meta_count: usize,
    ) -> Self {
        let allocation_layout = chunk_layout.allocation_layout();
        let allocation_pointer = allocate_chunk_memory(allocation_layout);

        let pointers = chunk_layout
            .columns()
            .iter()
            .copied()
            .map(|column_layout| {
                column_layout.offset().map_or_else(
                    || aligned_dangling_pointer(column_layout.alignment()),
                    |offset| unsafe {
                        NonNull::new_unchecked(allocation_pointer.as_ptr().add(offset))
                    },
                )
            })
            .collect();

        Self {
            chunk_index,
            count: 0,
            capacity: chunk_layout.capacity(),
            allocation_pointer,
            allocation_size: chunk_layout.allocation_size(),
            allocation_alignment: chunk_layout.allocation_alignment(),
            meta_pointer,
            meta_count,
            pointers,
        }
    }

    pub const fn chunk_index(&self) -> ChunkIndex {
        self.chunk_index
    }

    pub const fn count(&self) -> usize {
        self.count
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn remaining_row_capacity(&self) -> usize {
        self.capacity.saturating_sub(self.count)
    }

    pub const fn allocation_size(&self) -> usize {
        self.allocation_size
    }

    pub const fn allocation_alignment(&self) -> usize {
        self.allocation_alignment
    }

    pub const fn base_pointer(&self) -> NonNull<u8> {
        self.allocation_pointer
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn is_full(&self) -> bool {
        self.count == self.capacity
    }

    pub fn pointers(&self) -> &[NonNull<u8>] {
        &self.pointers
    }

    pub fn metas(&self) -> &[Meta] {
        // Safety: each chunk stores a pointer into its owning table's boxed meta slice. Tables drop
        // chunks before dropping metas, so the pointer remains valid for the chunk lifetime.
        unsafe { from_raw_parts(self.meta_pointer.as_ptr(), self.meta_count) }
    }

    pub fn column(&self, column_index: ColumnIndex) -> Option<Column<'_>> {
        let pointer = *self.pointers.get(usize::from(column_index.value()))?;
        let meta = self.metas().get(usize::from(column_index.value()))?;

        Some(Column::new(pointer, meta))
    }

    pub fn rows<'job>(&self, table_index: TableIndex, row_layout: RowLayout) -> Rows<'job> {
        debug_assert!(self.count <= self.capacity);

        Rows::new(table_index, self.chunk_index, row_layout, 0, self.count)
    }

    /// # Safety
    ///
    /// The caller must ensure that writing `value` into `row_index` is valid for the current
    /// initialized-prefix state of the chunk. In particular:
    ///
    /// - `row_index` must not name a live value that still needs to be dropped first,
    /// - no other references may currently alias the written element,
    /// - and the caller must later update the initialized prefix only after every written column
    ///   for that row is initialized consistently.
    pub unsafe fn write<T: 'static>(
        &mut self,
        column_index: ColumnIndex,
        row_index: usize,
        value: T,
    ) -> Result<(), ChunkError> {
        if row_index >= self.capacity {
            return Err(ChunkError::RowIndexOutOfBounds {
                row_index,
                capacity: self.capacity,
            });
        }

        let (_, pointer) = self.inline_meta_and_pointer::<T>(column_index)?;

        // Safety: the caller promises that overwriting the target slot is correct for the current
        // initialized-prefix state of the chunk.
        unsafe {
            pointer.as_ptr().cast::<T>().add(row_index).write(value);
        }

        Ok(())
    }

    /// # Safety
    ///
    /// The caller must guarantee that every row in `0..count` is fully initialized for every
    /// inline column in the chunk according to the table metadata.
    pub unsafe fn assume_initialized_count(&mut self, count: usize) -> Result<(), ChunkError> {
        if count > self.capacity {
            return Err(ChunkError::CountExceedsCapacity {
                count,
                capacity: self.capacity,
            });
        }

        self.count = count;
        Ok(())
    }

    pub fn slice<T: 'static>(&self, column_index: ColumnIndex) -> Result<&[T], ChunkError> {
        let (_, pointer) = self.inline_meta_and_pointer::<T>(column_index)?;
        debug_assert!(self.count <= self.capacity);

        // Safety: `count` is the current initialized prefix length, and the type check above
        // guarantees the requested `T` matches the stored column metadata.
        Ok(unsafe { from_raw_parts(pointer.as_ptr().cast::<T>(), self.count) })
    }

    pub fn slice_mut<T: 'static>(
        &mut self,
        column_index: ColumnIndex,
    ) -> Result<&mut [T], ChunkError> {
        let (_, pointer) = self.inline_meta_and_pointer::<T>(column_index)?;
        debug_assert!(self.count <= self.capacity);

        // Safety: `&mut self` guarantees exclusive access to the chunk, and the type check above
        // guarantees the requested `T` matches the stored column metadata.
        Ok(unsafe { from_raw_parts_mut(pointer.as_ptr().cast::<T>(), self.count) })
    }

    pub fn swap_remove_row(&mut self, row_index: usize) -> Result<(), ChunkError> {
        if row_index >= self.count {
            return Err(ChunkError::RowIndexOutsideInitializedPrefix {
                row_index,
                count: self.count,
            });
        }

        let last_row_index = self.count - 1;

        for (meta, pointer) in self.metas().iter().zip(self.pointers.iter().copied()) {
            if !meta.is_inline() {
                continue;
            }

            // Safety: `row_index` and `last_row_index` are both inside the initialized prefix.
            unsafe {
                if row_index == last_row_index {
                    drop_value_at(*meta, pointer, row_index);
                } else {
                    drop_value_at(*meta, pointer, row_index);
                    copy_value_between(*meta, pointer, last_row_index, row_index);
                }
            }
        }

        self.count -= 1;
        Ok(())
    }

    fn meta_and_pointer(
        &self,
        column_index: ColumnIndex,
    ) -> Result<(&Meta, NonNull<u8>), ChunkError> {
        let column_count = self.pointers.len();
        let pointer = *self.pointers.get(usize::from(column_index.value())).ok_or(
            ChunkError::ColumnIndexOutOfBounds {
                column_index,
                column_count,
            },
        )?;
        let meta = self.metas().get(usize::from(column_index.value())).ok_or(
            ChunkError::ColumnIndexOutOfBounds {
                column_index,
                column_count,
            },
        )?;

        Ok((meta, pointer))
    }

    fn inline_meta_and_pointer<T: 'static>(
        &self,
        column_index: ColumnIndex,
    ) -> Result<(&Meta, NonNull<u8>), ChunkError> {
        let (meta, pointer) = self.meta_and_pointer(column_index)?;

        if !meta.is_inline() {
            return Err(ChunkError::ColumnNotInline {
                meta_name: meta.name(),
            });
        }

        if meta.identifier() != TypeId::of::<T>() {
            return Err(ChunkError::ColumnTypeMismatch {
                column_name: meta.name(),
                requested_type_name: type_name::<T>(),
            });
        }

        Ok((meta, pointer))
    }

    fn drop_inhabited_rows(&mut self) {
        let initialized_count = self.count;
        if initialized_count == 0 {
            return;
        }

        for (meta, pointer) in self.metas().iter().zip(self.pointers.iter().copied()) {
            if !meta.is_inline() {
                continue;
            }

            for row_index in 0..initialized_count {
                // Safety: `row_index` is inside the initialized prefix and the chunk owns the
                // corresponding storage region exclusively during drop.
                unsafe {
                    drop_value_at(*meta, pointer, row_index);
                }
            }
        }

        self.count = 0;
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        self.drop_inhabited_rows();

        if self.allocation_size > 0 {
            let allocation_layout =
                Layout::from_size_align(self.allocation_size, self.allocation_alignment)
                    .expect("stored chunk allocation layout must remain valid");

            // Safety: `allocation_pointer` was allocated with `allocation_layout` in `Chunk::new`
            // and has not been deallocated since.
            unsafe {
                dealloc(self.allocation_pointer.as_ptr(), allocation_layout);
            }
        }
    }
}

/// A table descriptor that owns stored-type metadata and chunk storage.
#[derive(Debug)]
pub struct Table {
    index: TableIndex,
    chunks: Vec<Chunk>,
    metas: Box<[Meta]>,
    row_layout: RowLayout,
    chunk_plan: ChunkPlan,
    layout: TableLayout,
    chunk_layouts: Box<[ChunkLayout]>,
}

impl Table {
    pub const fn index(&self) -> TableIndex {
        self.index
    }

    pub fn metas(&self) -> &[Meta] {
        &self.metas
    }

    pub fn meta(&self, column_index: ColumnIndex) -> Option<&Meta> {
        self.metas.get(usize::from(column_index.value()))
    }

    pub fn meta_for_identifier(&self, identifier: TypeId) -> Option<&Meta> {
        self.metas
            .iter()
            .find(|meta| meta.identifier() == identifier)
    }

    pub fn inline_meta_for_identifier(&self, identifier: TypeId) -> Option<&Meta> {
        let meta = self.meta_for_identifier(identifier)?;
        meta.is_inline().then_some(meta)
    }

    pub fn meta_for<T: 'static>(&self) -> Option<&Meta> {
        self.meta_for_identifier(TypeId::of::<T>())
    }

    pub fn inline_meta_for<T: 'static>(&self) -> Option<&Meta> {
        self.inline_meta_for_identifier(TypeId::of::<T>())
    }

    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }

    pub fn chunk(&self, chunk_index: ChunkIndex) -> Option<&Chunk> {
        self.chunks.get(chunk_index.value() as usize)
    }

    pub fn chunk_mut(&mut self, chunk_index: ChunkIndex) -> Option<&mut Chunk> {
        self.chunks.get_mut(chunk_index.value() as usize)
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn rows<'job>(&self, chunk_index: ChunkIndex) -> Result<Rows<'job>, ChunkError> {
        let chunk = self
            .chunk(chunk_index)
            .ok_or(ChunkError::MissingChunk { chunk_index })?;

        Ok(chunk.rows(self.index, self.row_layout))
    }

    pub fn chunk_layouts(&self) -> &[ChunkLayout] {
        &self.chunk_layouts
    }

    pub fn chunk_layout_for_capacity(&self, capacity: usize) -> Option<&ChunkLayout> {
        self.chunk_layouts
            .binary_search_by_key(&capacity, |chunk_layout| chunk_layout.capacity())
            .ok()
            .map(|chunk_layout_index| &self.chunk_layouts[chunk_layout_index])
    }

    pub fn full_chunk_layout(&self) -> &ChunkLayout {
        self.chunk_layouts
            .last()
            .expect("tables always precompute at least one chunk layout")
    }

    pub fn next_chunk_capacity(&self) -> usize {
        self.chunks
            .last()
            .map(|chunk| {
                if chunk.capacity() < self.chunk_plan.target_chunk_capacity() {
                    (chunk.capacity() * 2).min(self.chunk_plan.target_chunk_capacity())
                } else {
                    self.chunk_plan.target_chunk_capacity()
                }
            })
            .unwrap_or(1)
    }

    pub fn push_chunk(&mut self) -> ChunkIndex {
        let next_chunk_capacity = self.next_chunk_capacity();
        let next_chunk_layout = self
            .chunk_layout_for_capacity(next_chunk_capacity)
            .expect("table must precompute every bootstrap chunk layout");
        debug_assert!(self.chunks.len() < u32::MAX as usize);
        let chunk_index = ChunkIndex::new(
            u32::try_from(self.chunks.len()).expect("table chunk count must fit in row encoding"),
        );
        let meta_pointer = meta_pointer_from_slice(&self.metas);

        self.chunks.push(Chunk::new(
            chunk_index,
            next_chunk_layout,
            meta_pointer,
            self.metas.len(),
        ));

        chunk_index
    }

    pub fn map_access_for_identifier(
        &self,
        identifier: TypeId,
        access: Access,
    ) -> Option<ColumnAccess> {
        let column_index = self
            .metas
            .iter()
            .position(|meta| meta.identifier() == identifier)
            .and_then(column_index_value)?;

        Some(ColumnAccess::new(
            self.index,
            ColumnIndex::new(column_index),
            access,
        ))
    }

    pub fn map_access<T: 'static>(&self, access: Access) -> Option<ColumnAccess> {
        self.map_access_for_identifier(TypeId::of::<T>(), access)
    }

    pub const fn row_layout(&self) -> RowLayout {
        self.row_layout
    }

    pub const fn chunk_plan(&self) -> ChunkPlan {
        self.chunk_plan
    }

    pub const fn layout(&self) -> TableLayout {
        self.layout
    }

    pub const fn store_resource(&self, store_index: StoreIndex) -> Resource {
        Resource::store(store_index)
    }

    pub const fn table_resource(&self, store_index: StoreIndex) -> Resource {
        Resource::table(store_index, self.index)
    }

    pub const fn chunk_resource(
        &self,
        store_index: StoreIndex,
        chunk_index: ChunkIndex,
    ) -> Resource {
        Resource::chunk(store_index, self.index, chunk_index)
    }

    pub const fn column_resource(
        &self,
        store_index: StoreIndex,
        chunk_index: ChunkIndex,
        column_index: ColumnIndex,
    ) -> Resource {
        Resource::column(store_index, self.index, chunk_index, column_index)
    }

    pub const fn store_dependencies(
        &self,
        store_index: StoreIndex,
        access: Access,
    ) -> [Dependency; 1] {
        [Dependency::new(self.store_resource(store_index), access)]
    }

    pub const fn table_dependencies(
        &self,
        store_index: StoreIndex,
        access: Access,
    ) -> [Dependency; 2] {
        [
            Dependency::read(self.store_resource(store_index)),
            Dependency::new(self.table_resource(store_index), access),
        ]
    }

    pub const fn chunk_dependencies(
        &self,
        store_index: StoreIndex,
        chunk_index: ChunkIndex,
        access: Access,
    ) -> [Dependency; 3] {
        [
            Dependency::read(self.store_resource(store_index)),
            Dependency::read(self.table_resource(store_index)),
            Dependency::new(self.chunk_resource(store_index, chunk_index), access),
        ]
    }

    pub const fn column_dependency(
        &self,
        store_index: StoreIndex,
        chunk_index: ChunkIndex,
        column_index: ColumnIndex,
        access: Access,
    ) -> Dependency {
        Dependency::new(
            self.column_resource(store_index, chunk_index, column_index),
            access,
        )
    }

    /// # Safety
    ///
    /// This is the table-level forwarding form of `Chunk::write`. The caller must ensure that
    /// writing `value` into `row_index` is valid for the chunk's current initialized-prefix state
    /// and that no conflicting references to the same element exist.
    pub unsafe fn write<T: 'static>(
        &mut self,
        chunk_index: ChunkIndex,
        row_index: usize,
        value: T,
    ) -> Result<(), ChunkError> {
        let column_index = self.column_index_for_type::<T>()?;
        let chunk = self
            .chunk_mut(chunk_index)
            .ok_or(ChunkError::MissingChunk { chunk_index })?;

        unsafe { chunk.write::<T>(column_index, row_index, value) }
    }

    /// # Safety
    ///
    /// The caller must guarantee that every row in `0..count` is fully initialized for every
    /// inline column in the addressed chunk.
    pub unsafe fn assume_initialized_prefix(
        &mut self,
        chunk_index: ChunkIndex,
        count: usize,
    ) -> Result<(), ChunkError> {
        let chunk = self
            .chunk_mut(chunk_index)
            .ok_or(ChunkError::MissingChunk { chunk_index })?;

        unsafe { chunk.assume_initialized_count(count) }
    }

    pub fn slice<T: 'static>(&self, chunk_index: ChunkIndex) -> Result<&[T], ChunkError> {
        let column_index = self.column_index_for_type::<T>()?;
        let chunk = self
            .chunk(chunk_index)
            .ok_or(ChunkError::MissingChunk { chunk_index })?;

        chunk.slice::<T>(column_index)
    }

    pub fn slice_mut<T: 'static>(
        &mut self,
        chunk_index: ChunkIndex,
    ) -> Result<&mut [T], ChunkError> {
        let column_index = self.column_index_for_type::<T>()?;
        let chunk = self
            .chunk_mut(chunk_index)
            .ok_or(ChunkError::MissingChunk { chunk_index })?;

        chunk.slice_mut::<T>(column_index)
    }

    pub fn swap_remove_row(
        &mut self,
        chunk_index: ChunkIndex,
        row_index: usize,
    ) -> Result<(), ChunkError> {
        let chunk = self
            .chunk_mut(chunk_index)
            .ok_or(ChunkError::MissingChunk { chunk_index })?;

        chunk.swap_remove_row(row_index)
    }

    pub fn resolve_remove_rows<'job, I>(&mut self, rows: I) -> Result<usize, ChunkError>
    where
        I: IntoIterator<Item = Row<'job>>,
    {
        let mut row_indices_by_chunk = BTreeMap::<ChunkIndex, Vec<usize>>::new();

        for row in rows {
            let actual_table_index = row.table_index();
            if actual_table_index != self.index {
                return Err(ChunkError::RowTableMismatch {
                    expected_table_index: self.index,
                    actual_table_index,
                });
            }

            let chunk_index = self.row_layout.chunk_index(row);
            let row_index = self.row_layout.row_index(row) as usize;

            row_indices_by_chunk
                .entry(chunk_index)
                .or_default()
                .push(row_index);
        }

        if row_indices_by_chunk.is_empty() {
            return Ok(0);
        }

        let mut removed_row_count = 0usize;

        for (chunk_index, row_indices) in row_indices_by_chunk {
            let chunk = self
                .chunk_mut(chunk_index)
                .ok_or(ChunkError::MissingChunk { chunk_index })?;
            let mut row_indices = row_indices;
            row_indices.sort_unstable();
            row_indices.dedup();

            for row_index in row_indices.into_iter().rev() {
                chunk.swap_remove_row(row_index)?;
                removed_row_count += 1;
            }
        }

        Ok(removed_row_count)
    }

    fn column_index_for_type<T: 'static>(&self) -> Result<ColumnIndex, ChunkError> {
        self.map_access::<T>(Access::Read)
            .map(ColumnAccess::column_index)
            .ok_or(ChunkError::MissingColumnForType {
                type_name: type_name::<T>(),
            })
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        // Chunks keep raw pointers into `self.metas`, so they must be dropped before the meta slice.
        self.chunks.clear();
    }
}

/// One column access mapped through one table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnAccess {
    table_index: TableIndex,
    column_index: ColumnIndex,
    access: Access,
}

impl ColumnAccess {
    pub const fn new(table_index: TableIndex, column_index: ColumnIndex, access: Access) -> Self {
        Self {
            table_index,
            column_index,
            access,
        }
    }

    pub const fn table_index(self) -> TableIndex {
        self.table_index
    }

    pub const fn column_index(self) -> ColumnIndex {
        self.column_index
    }

    pub const fn access(self) -> Access {
        self.access
    }

    pub const fn resource(self, store_index: StoreIndex, chunk_index: ChunkIndex) -> Resource {
        Resource::column(
            store_index,
            self.table_index,
            chunk_index,
            self.column_index,
        )
    }

    pub const fn dependencies(
        self,
        store_index: StoreIndex,
        chunk_index: ChunkIndex,
    ) -> [Dependency; 4] {
        [
            Dependency::read(Resource::store(store_index)),
            Dependency::read(Resource::table(store_index, self.table_index)),
            Dependency::read(Resource::chunk(store_index, self.table_index, chunk_index)),
            Dependency::new(
                Resource::column(
                    store_index,
                    self.table_index,
                    chunk_index,
                    self.column_index,
                ),
                self.access,
            ),
        ]
    }
}

/// A metadata catalog that interns table shapes and issues new table indices.
#[derive(Debug, Default)]
pub struct Catalog {
    table_shape_indices_by_signature: BTreeMap<Box<[MetaSignature]>, u32>,
    table_shape_count: usize,
    next_table_index: u32,
}

impl Catalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn table_shape_count(&self) -> usize {
        self.table_shape_count
    }

    pub fn table_count(&self) -> usize {
        self.next_table_index as usize
    }

    pub fn register_table<I>(
        &mut self,
        metas: I,
        configuration: Configuration,
    ) -> Result<Table, DefinitionError>
    where
        I: IntoIterator<Item = Meta>,
    {
        let mut metas: Vec<_> = metas.into_iter().collect();
        if metas.len() > u16::MAX as usize {
            return Err(DefinitionError::TooManyColumns);
        }
        metas.sort_unstable_by_key(|meta| meta.identifier());

        for meta_pair in metas.windows(2) {
            if meta_pair[0].identifier() == meta_pair[1].identifier() {
                return Err(DefinitionError::DuplicateMeta {
                    meta_name: meta_pair[1].name(),
                });
            }
        }

        let table_shape_signature: Box<[MetaSignature]> = metas
            .iter()
            .copied()
            .map(MetaSignature::from_meta)
            .collect();

        if !self
            .table_shape_indices_by_signature
            .contains_key(&table_shape_signature)
        {
            let next_table_shape_index = u32::try_from(self.table_shape_count)
                .map_err(|_| DefinitionError::TooManyTables)?;
            self.table_shape_indices_by_signature
                .insert(table_shape_signature, next_table_shape_index);
            self.table_shape_count += 1;
        }

        let row_width = metas
            .iter()
            .filter(|meta| meta.is_inline())
            .map(|meta| meta.element_size())
            .sum();
        let layout = TableLayout::new(row_width, metas.len());
        let chunk_plan = configuration.plan_chunk_capacity_for_row_width(layout.row_width());
        let row_layout = RowLayout::try_for_chunk_capacity(chunk_plan.target_chunk_capacity())?;
        let chunk_layouts =
            chunk_layouts_for_target_capacity(&metas, chunk_plan.target_chunk_capacity())?;
        let table_index = TableIndex::new(self.next_table_index);
        self.next_table_index = self
            .next_table_index
            .checked_add(1)
            .ok_or(DefinitionError::TooManyTables)?;

        Ok(Table {
            index: table_index,
            chunks: Vec::new(),
            metas: metas.into_boxed_slice(),
            row_layout,
            chunk_plan,
            layout,
            chunk_layouts,
        })
    }
}

fn region_layout_for_meta(meta: Meta, capacity: usize) -> Result<Option<Layout>, DefinitionError> {
    if !meta.is_inline() || meta.element_size() == 0 {
        return Ok(None);
    }

    let region_size = meta.element_size().checked_mul(capacity).ok_or(
        DefinitionError::AllocationLayoutOverflow {
            meta_name: meta.name(),
            capacity,
        },
    )?;

    let region_layout =
        Layout::from_size_align(region_size, meta.element_alignment()).map_err(|_| {
            DefinitionError::AllocationLayoutOverflow {
                meta_name: meta.name(),
                capacity,
            }
        })?;

    Ok(Some(region_layout))
}

fn chunk_layouts_for_target_capacity(
    metas: &[Meta],
    target_chunk_capacity: usize,
) -> Result<Box<[ChunkLayout]>, DefinitionError> {
    chunk_capacities_for_target(target_chunk_capacity)?
        .into_iter()
        .map(|capacity| ChunkLayout::new(metas, capacity))
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}

fn chunk_capacities_for_target(
    target_chunk_capacity: usize,
) -> Result<Vec<usize>, DefinitionError> {
    if target_chunk_capacity < 1 || !target_chunk_capacity.is_power_of_two() {
        return Err(DefinitionError::InvalidChunkCapacity {
            capacity: target_chunk_capacity,
        });
    }

    let mut capacities = Vec::new();
    let mut current_capacity = 1usize;

    loop {
        capacities.push(current_capacity);
        if current_capacity == target_chunk_capacity {
            break;
        }
        current_capacity = (current_capacity * 2).min(target_chunk_capacity);
    }

    Ok(capacities)
}

fn allocate_chunk_memory(allocation_layout: Layout) -> NonNull<u8> {
    if allocation_layout.size() == 0 {
        return aligned_dangling_pointer(allocation_layout.align());
    }

    // Safety: `allocation_layout` is valid and non-zero sized.
    let allocation_pointer = unsafe { alloc(allocation_layout) };
    let Some(allocation_pointer) = NonNull::new(allocation_pointer) else {
        handle_alloc_error(allocation_layout);
    };

    allocation_pointer
}

fn aligned_dangling_pointer(alignment: usize) -> NonNull<u8> {
    debug_assert!(alignment >= 1);
    debug_assert!(alignment.is_power_of_two());

    let alignment =
        NonZeroUsize::new(alignment).expect("alignment must remain a non-zero power of two");

    // `NonNull::without_provenance` gives us an aligned dangling sentinel without relying on an
    // integer-to-pointer cast, which keeps Miri and strict provenance happy.
    NonNull::without_provenance(alignment)
}

fn meta_pointer_from_slice(metas: &[Meta]) -> NonNull<Meta> {
    if metas.is_empty() {
        NonNull::dangling()
    } else {
        // Safety: `metas.as_ptr()` is non-null for non-empty slices and remains stable because the
        // table stores metas in a boxed slice.
        unsafe { NonNull::new_unchecked(metas.as_ptr() as *mut Meta) }
    }
}

unsafe fn drop_value_at(meta: Meta, pointer: NonNull<u8>, row_index: usize) {
    let Some(drop_value) = meta.drop_value() else {
        return;
    };

    let row_pointer = row_pointer(pointer, meta, row_index);
    unsafe {
        drop_value(row_pointer.as_ptr());
    }
}

unsafe fn copy_value_between(
    meta: Meta,
    pointer: NonNull<u8>,
    source_row_index: usize,
    target_row_index: usize,
) {
    let Some(copy_value) = meta.copy_value() else {
        return;
    };

    let source_row_pointer = row_pointer(pointer, meta, source_row_index);
    let target_row_pointer = row_pointer(pointer, meta, target_row_index);

    unsafe {
        copy_value(source_row_pointer.as_ptr(), target_row_pointer.as_ptr());
    }
}

fn row_pointer(pointer: NonNull<u8>, meta: Meta, row_index: usize) -> NonNull<u8> {
    if meta.element_size() == 0 {
        return pointer;
    }

    let byte_offset = row_index
        .checked_mul(meta.element_size())
        .expect("row offset must stay within the allocated chunk region");

    // Safety: callers validate row indices against chunk capacity or initialized count before
    // asking for a row pointer, and chunk layouts guarantee the region offset is in-bounds.
    unsafe { NonNull::new_unchecked(pointer.as_ptr().add(byte_offset)) }
}

fn column_index_value(column_count: usize) -> Option<u16> {
    u16::try_from(column_count).ok()
}
