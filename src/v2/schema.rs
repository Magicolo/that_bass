//! Metadata and identity model for the rewrite lane.
//!
//! Task `01-identity-and-metadata.md` requires a stronger distinction than the stable engine:
//!
//! - logical schema identity,
//! - physical column layout,
//! - table identity,
//! - chunk identity,
//! - scheduler resource identity.
//!
//! This module defines that metadata model without yet implementing chunk storage or query
//! execution.

use crate::v2::{
    key::Key,
    query::Access,
    store::{ChunkPlan, Configuration},
};
use core::{
    any::{type_name, TypeId},
    mem::{align_of, size_of},
};
use std::{collections::BTreeMap, sync::Arc};

/// The identity policy selected for one table descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Identity {
    Keyless,
    ManagedKeys,
    UserKeyed,
}

/// The physical storage class selected for one physical column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Storage {
    Inline,
    Sidecar,
}

/// A stable schema registry index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SchemaIndex(u32);

impl SchemaIndex {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u32 {
        self.0
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

/// A stable logical-column index inside one schema descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LogicalColumnIndex(u16);

impl LogicalColumnIndex {
    pub const fn new(value: u16) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u16 {
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

/// A stable physical-column index inside one table descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysicalColumnIndex(u16);

impl PhysicalColumnIndex {
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
    PhysicalColumn {
        store_index: StoreIndex,
        table_index: TableIndex,
        chunk_index: ChunkIndex,
        physical_column_index: PhysicalColumnIndex,
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

    pub const fn physical_column(
        store_index: StoreIndex,
        table_index: TableIndex,
        chunk_index: ChunkIndex,
        physical_column_index: PhysicalColumnIndex,
    ) -> Self {
        Self::PhysicalColumn {
            store_index,
            table_index,
            chunk_index,
            physical_column_index,
        }
    }

    pub const fn store_index(self) -> StoreIndex {
        match self {
            Self::Store(store_index)
            | Self::Table { store_index, .. }
            | Self::Chunk { store_index, .. }
            | Self::PhysicalColumn { store_index, .. } => store_index,
        }
    }

    pub const fn table_index(self) -> Option<TableIndex> {
        match self {
            Self::Table { table_index, .. }
            | Self::Chunk { table_index, .. }
            | Self::PhysicalColumn { table_index, .. } => Some(table_index),
            Self::Store(_) => None,
        }
    }

    pub const fn chunk_index(self) -> Option<ChunkIndex> {
        match self {
            Self::Chunk { chunk_index, .. } | Self::PhysicalColumn { chunk_index, .. } => {
                Some(chunk_index)
            }
            Self::Store(_) | Self::Table { .. } => None,
        }
    }

    pub const fn physical_column_index(self) -> Option<PhysicalColumnIndex> {
        match self {
            Self::PhysicalColumn {
                physical_column_index,
                ..
            } => Some(physical_column_index),
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

/// The bit partition used by a table's future packed `Row<'job>` representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowAddressLayout {
    row_index_bit_count: u8,
    chunk_index_bit_count: u8,
}

impl RowAddressLayout {
    pub fn for_chunk_capacity(target_chunk_capacity: usize) -> Self {
        debug_assert!(target_chunk_capacity >= 1);
        debug_assert!(target_chunk_capacity.is_power_of_two());

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
}

/// The layout metrics that matter for chunk-capacity planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchemaLayout {
    inline_row_width: usize,
    physical_column_count: usize,
}

impl SchemaLayout {
    pub const fn new(inline_row_width: usize, physical_column_count: usize) -> Self {
        Self {
            inline_row_width,
            physical_column_count,
        }
    }

    pub const fn inline_row_width(self) -> usize {
        self.inline_row_width
    }

    pub const fn physical_column_count(self) -> usize {
        self.physical_column_count
    }
}

/// One logical datum requested by user-facing query or template syntax.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalColumn {
    index: LogicalColumnIndex,
    identifier: TypeId,
    name: &'static str,
}

impl LogicalColumn {
    pub const fn index(&self) -> LogicalColumnIndex {
        self.index
    }

    pub const fn identifier(&self) -> TypeId {
        self.identifier
    }

    pub const fn name(&self) -> &'static str {
        self.name
    }
}

/// A canonical logical schema descriptor interned by the catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    index: SchemaIndex,
    logical_columns: Box<[LogicalColumn]>,
}

impl Schema {
    pub const fn index(&self) -> SchemaIndex {
        self.index
    }

    pub fn logical_columns(&self) -> &[LogicalColumn] {
        &self.logical_columns
    }

    pub fn logical_column(
        &self,
        logical_column_index: LogicalColumnIndex,
    ) -> Option<&LogicalColumn> {
        self.logical_columns
            .get(usize::from(logical_column_index.value()))
    }

    pub fn logical_column_for_identifier(&self, identifier: TypeId) -> Option<&LogicalColumn> {
        self.logical_columns
            .iter()
            .find(|logical_column| logical_column.identifier() == identifier)
    }

    pub fn logical_column_for<T: 'static>(&self) -> Option<&LogicalColumn> {
        self.logical_column_for_identifier(TypeId::of::<T>())
    }
}

/// The owner category for a physical column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColumnOwner {
    LogicalColumn(LogicalColumnIndex),
    ManagedKey,
}

/// One physical storage column within one table descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalColumn {
    index: PhysicalColumnIndex,
    owner: ColumnOwner,
    name: &'static str,
    element_size: usize,
    element_alignment: usize,
    storage: Storage,
}

impl PhysicalColumn {
    pub const fn index(&self) -> PhysicalColumnIndex {
        self.index
    }

    pub const fn owner(&self) -> ColumnOwner {
        self.owner
    }

    pub const fn name(&self) -> &'static str {
        self.name
    }

    pub const fn element_size(&self) -> usize {
        self.element_size
    }

    pub const fn element_alignment(&self) -> usize {
        self.element_alignment
    }

    pub const fn storage(&self) -> Storage {
        self.storage
    }
}

/// A table descriptor that binds one logical schema to one physical layout.
#[derive(Debug, Clone)]
pub struct Table {
    index: TableIndex,
    schema: Arc<Schema>,
    physical_columns: Box<[PhysicalColumn]>,
    physical_columns_by_logical_column: Box<[Box<[PhysicalColumnIndex]>]>,
    managed_key_physical_column: Option<PhysicalColumnIndex>,
    identity: Identity,
    row_address_layout: RowAddressLayout,
    chunk_plan: ChunkPlan,
    layout: SchemaLayout,
}

impl Table {
    pub const fn index(&self) -> TableIndex {
        self.index
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    pub fn physical_columns(&self) -> &[PhysicalColumn] {
        &self.physical_columns
    }

    pub const fn managed_key_physical_column(&self) -> Option<PhysicalColumnIndex> {
        self.managed_key_physical_column
    }

    pub const fn identity(&self) -> Identity {
        self.identity
    }

    pub const fn row_address_layout(&self) -> RowAddressLayout {
        self.row_address_layout
    }

    pub const fn chunk_plan(&self) -> ChunkPlan {
        self.chunk_plan
    }

    pub const fn layout(&self) -> SchemaLayout {
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

    pub fn physical_columns_for_logical_column(
        &self,
        logical_column_index: LogicalColumnIndex,
    ) -> Option<&[PhysicalColumnIndex]> {
        self.physical_columns_by_logical_column
            .get(usize::from(logical_column_index.value()))
            .map(Box::as_ref)
    }

    pub fn physical_columns_for_identifier(
        &self,
        identifier: TypeId,
    ) -> Option<&[PhysicalColumnIndex]> {
        let logical_column = self.schema.logical_column_for_identifier(identifier)?;
        self.physical_columns_for_logical_column(logical_column.index())
    }

    pub fn physical_columns_for<T: 'static>(&self) -> Option<&[PhysicalColumnIndex]> {
        self.physical_columns_for_identifier(TypeId::of::<T>())
    }

    pub fn map_logical_access(
        &self,
        logical_access: LogicalAccess,
    ) -> Option<PhysicalAccessSet<'_>> {
        let physical_column_indices =
            self.physical_columns_for_logical_column(logical_access.logical_column_index())?;

        Some(PhysicalAccessSet {
            table_index: self.index,
            physical_column_indices,
            access: logical_access.access(),
        })
    }

    pub fn map_access_for_identifier(
        &self,
        identifier: TypeId,
        access: Access,
    ) -> Option<PhysicalAccessSet<'_>> {
        let logical_column = self.schema.logical_column_for_identifier(identifier)?;
        self.map_logical_access(LogicalAccess::new(logical_column.index(), access))
    }

    pub fn map_access<T: 'static>(&self, access: Access) -> Option<PhysicalAccessSet<'_>> {
        self.map_access_for_identifier(TypeId::of::<T>(), access)
    }

    pub const fn physical_column_resource(
        &self,
        store_index: StoreIndex,
        chunk_index: ChunkIndex,
        physical_column_index: PhysicalColumnIndex,
    ) -> Resource {
        Resource::physical_column(store_index, self.index, chunk_index, physical_column_index)
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

    pub const fn physical_column_dependency(
        &self,
        store_index: StoreIndex,
        chunk_index: ChunkIndex,
        physical_column_index: PhysicalColumnIndex,
        access: Access,
    ) -> Dependency {
        Dependency::new(
            self.physical_column_resource(store_index, chunk_index, physical_column_index),
            access,
        )
    }
}

/// A logical-access request before table-specific decomposition or resource generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicalAccess {
    logical_column_index: LogicalColumnIndex,
    access: Access,
}

impl LogicalAccess {
    pub const fn new(logical_column_index: LogicalColumnIndex, access: Access) -> Self {
        Self {
            logical_column_index,
            access,
        }
    }

    pub const fn logical_column_index(self) -> LogicalColumnIndex {
        self.logical_column_index
    }

    pub const fn access(self) -> Access {
        self.access
    }
}

/// One physical access produced by mapping a logical access through a table descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalAccess {
    physical_column_index: PhysicalColumnIndex,
    access: Access,
}

impl PhysicalAccess {
    pub const fn physical_column_index(self) -> PhysicalColumnIndex {
        self.physical_column_index
    }

    pub const fn access(self) -> Access {
        self.access
    }
}

/// The physical accesses produced by mapping one logical request through one table descriptor.
#[derive(Debug, Clone, Copy)]
pub struct PhysicalAccessSet<'a> {
    table_index: TableIndex,
    physical_column_indices: &'a [PhysicalColumnIndex],
    access: Access,
}

impl<'a> PhysicalAccessSet<'a> {
    pub const fn table_index(self) -> TableIndex {
        self.table_index
    }

    pub fn physical_column_indices(self) -> &'a [PhysicalColumnIndex] {
        self.physical_column_indices
    }

    pub const fn access(self) -> Access {
        self.access
    }

    pub fn physical_accesses(self) -> impl Iterator<Item = PhysicalAccess> + 'a {
        self.physical_column_indices
            .iter()
            .copied()
            .map(move |physical_column_index| PhysicalAccess {
                physical_column_index,
                access: self.access,
            })
    }

    pub fn resources(
        self,
        store_index: StoreIndex,
        chunk_index: ChunkIndex,
    ) -> impl Iterator<Item = Resource> + 'a {
        self.physical_column_indices
            .iter()
            .copied()
            .map(move |physical_column_index| {
                Resource::physical_column(
                    store_index,
                    self.table_index,
                    chunk_index,
                    physical_column_index,
                )
            })
    }

    pub fn dependencies(
        self,
        store_index: StoreIndex,
        chunk_index: ChunkIndex,
    ) -> impl Iterator<Item = Dependency> + 'a {
        // Descendant accesses carry read dependencies on their ancestors so broad requests such as
        // `Write(store)` or `Write(chunk)` conflict correctly with narrow leaf-column accesses.
        [
            Dependency::read(Resource::store(store_index)),
            Dependency::read(Resource::table(store_index, self.table_index)),
            Dependency::read(Resource::chunk(store_index, self.table_index, chunk_index)),
        ]
        .into_iter()
        .chain(
            self.physical_column_indices
                .iter()
                .copied()
                .map(move |physical_column_index| {
                    Dependency::new(
                        Resource::physical_column(
                            store_index,
                            self.table_index,
                            chunk_index,
                            physical_column_index,
                        ),
                        self.access,
                    )
                }),
        )
    }
}

/// One logical-column declaration used when interning a schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicalColumnDeclaration {
    identifier: TypeId,
    name: &'static str,
    element_size: usize,
    element_alignment: usize,
}

impl LogicalColumnDeclaration {
    pub fn of<T: 'static>() -> Self {
        Self {
            identifier: TypeId::of::<T>(),
            name: type_name::<T>(),
            element_size: size_of::<T>(),
            element_alignment: align_of::<T>(),
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhysicalColumnOwnerDeclaration {
    LogicalColumn {
        logical_column_identifier: TypeId,
        logical_column_name: &'static str,
    },
    ManagedKey,
}

/// One physical-column declaration used when building a table descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalColumnDeclaration {
    owner: PhysicalColumnOwnerDeclaration,
    name: &'static str,
    element_size: usize,
    element_alignment: usize,
    storage: Storage,
}

impl PhysicalColumnDeclaration {
    pub fn for_logical_column<T: 'static>(
        name: &'static str,
        element_size: usize,
        element_alignment: usize,
        storage: Storage,
    ) -> Self {
        Self {
            owner: PhysicalColumnOwnerDeclaration::LogicalColumn {
                logical_column_identifier: TypeId::of::<T>(),
                logical_column_name: type_name::<T>(),
            },
            name,
            element_size,
            element_alignment,
            storage,
        }
    }

    pub fn inline<T: 'static>() -> Self {
        Self::for_logical_column::<T>(
            type_name::<T>(),
            size_of::<T>(),
            align_of::<T>(),
            Storage::Inline,
        )
    }

    pub fn sidecar<T: 'static>() -> Self {
        Self::for_logical_column::<T>(
            type_name::<T>(),
            size_of::<T>(),
            align_of::<T>(),
            Storage::Sidecar,
        )
    }

    pub fn managed_key(storage: Storage) -> Self {
        Self {
            owner: PhysicalColumnOwnerDeclaration::ManagedKey,
            name: type_name::<Key>(),
            element_size: size_of::<Key>(),
            element_alignment: align_of::<Key>(),
            storage,
        }
    }

    pub fn inline_managed_key() -> Self {
        Self::managed_key(Storage::Inline)
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
}

/// A table-descriptor declaration before catalog indexing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableDeclaration {
    logical_columns: Box<[LogicalColumnDeclaration]>,
    physical_columns: Box<[PhysicalColumnDeclaration]>,
    identity: Identity,
}

impl TableDeclaration {
    pub fn new<I>(logical_columns: I) -> Self
    where
        I: IntoIterator<Item = LogicalColumnDeclaration>,
    {
        Self {
            logical_columns: logical_columns.into_iter().collect(),
            physical_columns: Box::new([]),
            identity: Identity::Keyless,
        }
    }

    pub fn with_physical_columns<I>(mut self, physical_columns: I) -> Self
    where
        I: IntoIterator<Item = PhysicalColumnDeclaration>,
    {
        self.physical_columns = physical_columns.into_iter().collect();
        self
    }

    pub const fn with_identity(mut self, identity: Identity) -> Self {
        self.identity = identity;
        self
    }

    pub fn logical_columns(&self) -> &[LogicalColumnDeclaration] {
        &self.logical_columns
    }

    pub fn physical_columns(&self) -> &[PhysicalColumnDeclaration] {
        &self.physical_columns
    }

    pub const fn identity(&self) -> Identity {
        self.identity
    }
}

/// Metadata-registration failures for schemas and tables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefinitionError {
    DuplicateLogicalColumn {
        logical_column_name: &'static str,
    },
    MissingPhysicalColumnsForLogicalColumn {
        logical_column_name: &'static str,
    },
    MissingLogicalColumnForPhysicalColumn {
        physical_column_name: &'static str,
        logical_column_name: &'static str,
    },
    ManagedKeyColumnRequiresManagedKeys {
        physical_column_name: &'static str,
    },
    DuplicateManagedKeyPhysicalColumn,
}

/// The metadata catalog for logical schemas and table descriptors.
#[derive(Debug, Default)]
pub struct Catalog {
    schema_indices_by_signature: BTreeMap<Box<[TypeId]>, SchemaIndex>,
    schemas: Vec<Arc<Schema>>,
    tables: Vec<Arc<Table>>,
}

impl Catalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn schema_count(&self) -> usize {
        self.schemas.len()
    }

    pub fn table_count(&self) -> usize {
        self.tables.len()
    }

    pub fn schema(&self, schema_index: SchemaIndex) -> Option<&Schema> {
        self.schemas
            .get(schema_index.value() as usize)
            .map(AsRef::as_ref)
    }

    pub fn table(&self, table_index: TableIndex) -> Option<&Table> {
        self.tables
            .get(table_index.value() as usize)
            .map(AsRef::as_ref)
    }

    pub fn register_schema<I>(
        &mut self,
        logical_column_declarations: I,
    ) -> Result<Arc<Schema>, DefinitionError>
    where
        I: IntoIterator<Item = LogicalColumnDeclaration>,
    {
        let mut logical_column_declarations: Vec<_> =
            logical_column_declarations.into_iter().collect();
        logical_column_declarations.sort_unstable_by_key(|logical_column_declaration| {
            logical_column_declaration.identifier()
        });

        for logical_column_pair in logical_column_declarations.windows(2) {
            if logical_column_pair[0].identifier() == logical_column_pair[1].identifier() {
                return Err(DefinitionError::DuplicateLogicalColumn {
                    logical_column_name: logical_column_pair[1].name(),
                });
            }
        }

        let schema_signature: Box<[TypeId]> = logical_column_declarations
            .iter()
            .map(|logical_column_declaration| logical_column_declaration.identifier())
            .collect();

        if let Some(schema_index) = self.schema_indices_by_signature.get(&schema_signature) {
            return Ok(self.schemas[schema_index.value() as usize].clone());
        }

        let schema_index = SchemaIndex::new(schema_index_value(self.schemas.len()));
        let logical_columns = logical_column_declarations
            .into_iter()
            .enumerate()
            .map(
                |(logical_column_position, logical_column_declaration)| LogicalColumn {
                    index: LogicalColumnIndex::new(logical_column_index_value(
                        logical_column_position,
                    )),
                    identifier: logical_column_declaration.identifier(),
                    name: logical_column_declaration.name(),
                },
            )
            .collect();

        let schema = Arc::new(Schema {
            index: schema_index,
            logical_columns,
        });

        self.schema_indices_by_signature
            .insert(schema_signature, schema_index);
        self.schemas.push(schema.clone());

        Ok(schema)
    }

    pub fn register_table(
        &mut self,
        table_declaration: TableDeclaration,
        configuration: Configuration,
    ) -> Result<Arc<Table>, DefinitionError> {
        let schema = self.register_schema(table_declaration.logical_columns.iter().copied())?;
        let physical_column_declarations =
            effective_physical_column_declarations(&table_declaration);

        let mut physical_columns = Vec::with_capacity(physical_column_declarations.len());
        let mut physical_columns_by_logical_column =
            vec![Vec::new(); schema.logical_columns().len()];
        let mut managed_key_physical_column = None;
        let mut inline_row_width = 0usize;

        for (physical_column_position, physical_column_declaration) in
            physical_column_declarations.iter().copied().enumerate()
        {
            let physical_column_index =
                PhysicalColumnIndex::new(physical_column_index_value(physical_column_position));

            let owner = match physical_column_declaration.owner {
                PhysicalColumnOwnerDeclaration::LogicalColumn {
                    logical_column_identifier,
                    logical_column_name,
                } => {
                    let logical_column = schema
                        .logical_column_for_identifier(logical_column_identifier)
                        .ok_or(DefinitionError::MissingLogicalColumnForPhysicalColumn {
                            physical_column_name: physical_column_declaration.name(),
                            logical_column_name,
                        })?;

                    physical_columns_by_logical_column[usize::from(logical_column.index().value())]
                        .push(physical_column_index);

                    ColumnOwner::LogicalColumn(logical_column.index())
                }
                PhysicalColumnOwnerDeclaration::ManagedKey => {
                    if table_declaration.identity() != Identity::ManagedKeys {
                        return Err(DefinitionError::ManagedKeyColumnRequiresManagedKeys {
                            physical_column_name: physical_column_declaration.name(),
                        });
                    }

                    if managed_key_physical_column.is_some() {
                        return Err(DefinitionError::DuplicateManagedKeyPhysicalColumn);
                    }

                    managed_key_physical_column = Some(physical_column_index);
                    ColumnOwner::ManagedKey
                }
            };

            if physical_column_declaration.storage() == Storage::Inline {
                inline_row_width += physical_column_declaration.element_size();
            }

            physical_columns.push(PhysicalColumn {
                index: physical_column_index,
                owner,
                name: physical_column_declaration.name(),
                element_size: physical_column_declaration.element_size(),
                element_alignment: physical_column_declaration.element_alignment(),
                storage: physical_column_declaration.storage(),
            });
        }

        for logical_column in schema.logical_columns() {
            if physical_columns_by_logical_column[usize::from(logical_column.index().value())]
                .is_empty()
            {
                return Err(DefinitionError::MissingPhysicalColumnsForLogicalColumn {
                    logical_column_name: logical_column.name(),
                });
            }
        }

        let layout = SchemaLayout::new(inline_row_width, physical_columns.len());
        let chunk_plan = configuration.plan_chunk_capacity_for_row_width(layout.inline_row_width());
        let row_address_layout =
            RowAddressLayout::for_chunk_capacity(chunk_plan.target_chunk_capacity());
        let table_index = TableIndex::new(table_index_value(self.tables.len()));

        let table = Arc::new(Table {
            index: table_index,
            schema,
            physical_columns: physical_columns.into_boxed_slice(),
            physical_columns_by_logical_column: physical_columns_by_logical_column
                .into_iter()
                .map(Vec::into_boxed_slice)
                .collect(),
            managed_key_physical_column,
            identity: table_declaration.identity(),
            row_address_layout,
            chunk_plan,
            layout,
        });

        self.tables.push(table.clone());

        Ok(table)
    }
}

fn effective_physical_column_declarations(
    table_declaration: &TableDeclaration,
) -> Vec<PhysicalColumnDeclaration> {
    let mut physical_column_declarations = if table_declaration.physical_columns().is_empty() {
        table_declaration
            .logical_columns()
            .iter()
            .map(|logical_column_declaration| {
                PhysicalColumnDeclaration::for_logical_column_by_identifier(
                    logical_column_declaration.identifier(),
                    logical_column_declaration.name(),
                    logical_column_declaration.name(),
                    logical_column_declaration.element_size(),
                    logical_column_declaration.element_alignment(),
                    Storage::Inline,
                )
            })
            .collect()
    } else {
        table_declaration.physical_columns().to_vec()
    };

    if table_declaration.identity() == Identity::ManagedKeys
        && !physical_column_declarations
            .iter()
            .any(|physical_column_declaration| {
                matches!(
                    physical_column_declaration.owner,
                    PhysicalColumnOwnerDeclaration::ManagedKey
                )
            })
    {
        physical_column_declarations.push(PhysicalColumnDeclaration::inline_managed_key());
    }

    physical_column_declarations
}

impl PhysicalColumnDeclaration {
    fn for_logical_column_by_identifier(
        logical_column_identifier: TypeId,
        logical_column_name: &'static str,
        name: &'static str,
        element_size: usize,
        element_alignment: usize,
        storage: Storage,
    ) -> Self {
        Self {
            owner: PhysicalColumnOwnerDeclaration::LogicalColumn {
                logical_column_identifier,
                logical_column_name,
            },
            name,
            element_size,
            element_alignment,
            storage,
        }
    }
}

fn schema_index_value(schema_count: usize) -> u32 {
    u32::try_from(schema_count).expect("schema count must fit in u32")
}

fn table_index_value(table_count: usize) -> u32 {
    u32::try_from(table_count).expect("table count must fit in u32")
}

fn logical_column_index_value(logical_column_count: usize) -> u16 {
    u16::try_from(logical_column_count).expect("logical column count must fit in u16")
}

fn physical_column_index_value(physical_column_count: usize) -> u16 {
    u16::try_from(physical_column_count).expect("physical column count must fit in u16")
}
