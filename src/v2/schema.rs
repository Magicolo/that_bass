//! Table metadata and scheduler-resource model for the rewrite lane.
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
//! This module does not yet implement full chunk allocation or row iteration. It establishes the
//! names, indices, metadata, and scheduler dependencies that later tasks will build on.

use crate::v2::{
    query::Access,
    store::{ChunkPlan, Configuration},
};
use core::{
    alloc::Layout,
    any::{type_name, TypeId},
    marker::PhantomData,
    mem::{align_of, size_of},
    ptr::NonNull,
};
use std::{collections::BTreeMap, rc::Rc};

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

/// The bit partition used by the packed `Row<'job>` representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowLayout {
    row_index_bit_count: u8,
    chunk_index_bit_count: u8,
}

impl RowLayout {
    pub fn for_chunk_capacity(target_chunk_capacity: usize) -> Self {
        assert!(
            target_chunk_capacity >= 1,
            "chunk capacity must be at least one row"
        );
        assert!(
            target_chunk_capacity.is_power_of_two(),
            "chunk capacity must be a power of two so row bits remain configurable"
        );

        let row_index_bit_count = target_chunk_capacity.max(1).ilog2() as u8;
        let chunk_index_bit_count = 32u8.saturating_sub(row_index_bit_count);

        Self {
            row_index_bit_count,
            chunk_index_bit_count,
        }
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
    ) -> Row<'job> {
        assert!(
            row_index <= self.maximum_row_index(),
            "row index exceeds the configured row bit budget"
        );
        assert!(
            chunk_index.value() <= self.maximum_chunk_index(),
            "chunk index exceeds the configured chunk bit budget"
        );

        Row {
            table_index,
            packed_chunk_and_row_index: (chunk_index.value() << self.row_index_bit_count)
                | row_index,
            marker: PhantomData,
        }
    }

    pub const fn chunk_index<'job>(self, row: Row<'job>) -> ChunkIndex {
        ChunkIndex::new(row.packed_chunk_and_row_index >> self.row_index_bit_count)
    }

    pub const fn row_index<'job>(self, row: Row<'job>) -> u32 {
        if self.row_index_bit_count == 32 {
            row.packed_chunk_and_row_index
        } else {
            row.packed_chunk_and_row_index & self.row_index_mask()
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
        (1u32 << self.row_index_bit_count) - 1
    }
}

/// A row index into a chunk of a table of a store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Row<'job> {
    table_index: TableIndex,
    packed_chunk_and_row_index: u32,
    marker: PhantomData<&'job ()>,
}

impl<'job> Row<'job> {
    pub const fn table_index(self) -> TableIndex {
        self.table_index
    }

    pub const fn packed_chunk_and_row_index(self) -> u32 {
        self.packed_chunk_and_row_index
    }
}

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
    drop_value: unsafe fn(*mut u8),
    copy_value: unsafe fn(*const u8, *mut u8),
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
            drop_value: drop_value::<T>,
            copy_value: copy_value::<T>,
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

    pub fn layout(self) -> Layout {
        Layout::from_size_align(self.element_size, self.element_alignment)
            .expect("meta layout must be valid")
    }

    pub const fn drop_value(self) -> unsafe fn(*mut u8) {
        self.drop_value
    }

    pub const fn copy_value(self) -> unsafe fn(*const u8, *mut u8) {
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

/// One chunk allocation described as a list of per-column pointers.
#[derive(Debug, Clone)]
pub struct Chunk {
    pointers: Box<[NonNull<u8>]>,
}

impl Chunk {
    pub fn new<I>(pointers: I) -> Self
    where
        I: IntoIterator<Item = NonNull<u8>>,
    {
        Self {
            pointers: pointers.into_iter().collect(),
        }
    }

    pub fn pointers(&self) -> &[NonNull<u8>] {
        &self.pointers
    }

    pub fn column<'a>(&self, column_index: ColumnIndex, metas: &'a [Meta]) -> Option<Column<'a>> {
        let pointer = *self.pointers.get(usize::from(column_index.value()))?;
        let meta = metas.get(usize::from(column_index.value()))?;

        Some(Column::new(pointer, meta))
    }
}

/// A table descriptor that owns stored-type metadata and chunk storage.
#[derive(Debug, Clone)]
pub struct Table {
    index: TableIndex,
    metas: Box<[Meta]>,
    chunks: Vec<Chunk>,
    row_layout: RowLayout,
    chunk_plan: ChunkPlan,
    layout: TableLayout,
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

    pub fn meta_for<T: 'static>(&self) -> Option<&Meta> {
        self.meta_for_identifier(TypeId::of::<T>())
    }

    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
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
            .map(column_index_value)?;

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

/// Metadata-registration failures for tables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefinitionError {
    DuplicateMeta { meta_name: &'static str },
}

/// A metadata catalog that interns table shapes and registers tables.
#[derive(Debug, Default)]
pub struct Catalog {
    table_shape_indices_by_signature: BTreeMap<Box<[MetaSignature]>, u32>,
    table_shape_count: usize,
    tables: Vec<Rc<Table>>,
}

impl Catalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn table_shape_count(&self) -> usize {
        self.table_shape_count
    }

    pub fn table_count(&self) -> usize {
        self.tables.len()
    }

    pub fn table(&self, table_index: TableIndex) -> Option<&Table> {
        self.tables
            .get(table_index.value() as usize)
            .map(AsRef::as_ref)
    }

    pub fn register_table<I>(
        &mut self,
        metas: I,
        configuration: Configuration,
    ) -> Result<Rc<Table>, DefinitionError>
    where
        I: IntoIterator<Item = Meta>,
    {
        let mut metas: Vec<_> = metas.into_iter().collect();
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
            let next_table_shape_index =
                u32::try_from(self.table_shape_count).expect("table shape count must fit in u32");
            self.table_shape_indices_by_signature
                .insert(table_shape_signature, next_table_shape_index);
            self.table_shape_count += 1;
        }

        let row_width = metas
            .iter()
            .filter(|meta| meta.storage() == Storage::Inline)
            .map(|meta| meta.element_size())
            .sum();
        let layout = TableLayout::new(row_width, metas.len());
        let chunk_plan = configuration.plan_chunk_capacity_for_row_width(layout.row_width());
        let row_layout = RowLayout::for_chunk_capacity(chunk_plan.target_chunk_capacity());
        let table_index = TableIndex::new(table_index_value(self.tables.len()));

        let table = Rc::new(Table {
            index: table_index,
            metas: metas.into_boxed_slice(),
            chunks: Vec::new(),
            row_layout,
            chunk_plan,
            layout,
        });

        self.tables.push(table.clone());

        Ok(table)
    }
}

fn table_index_value(table_count: usize) -> u32 {
    u32::try_from(table_count).expect("table count must fit in u32")
}

fn column_index_value(column_count: usize) -> u16 {
    u16::try_from(column_count).expect("column count must fit in u16")
}
