use checkito::Check;
use core::{
    any::type_name,
    mem::{align_of, size_of},
};
use that_bass::v2::{
    query::Access,
    schema::{
        Catalog, ChunkIndex, ColumnOwner, DefinitionError, Dependency, Identity,
        LogicalColumnDeclaration, PhysicalColumnDeclaration, Resource, Storage, StoreIndex,
        TableDeclaration,
    },
    Configuration,
};

#[repr(C)]
struct Position {
    x: f64,
    y: f64,
}

#[repr(C)]
struct Velocity {
    x: f32,
    y: f32,
}

#[test]
fn catalog_interns_equivalent_schemas_even_when_declarations_are_reordered() {
    let mut catalog = Catalog::new();

    let first_schema = catalog
        .register_schema([
            LogicalColumnDeclaration::of::<Velocity>(),
            LogicalColumnDeclaration::of::<Position>(),
        ])
        .expect("schema registration should succeed");
    let second_schema = catalog
        .register_schema([
            LogicalColumnDeclaration::of::<Position>(),
            LogicalColumnDeclaration::of::<Velocity>(),
        ])
        .expect("equivalent schema registration should succeed");

    assert_eq!(catalog.schema_count(), 1);
    assert_eq!(first_schema.index(), second_schema.index());
    assert_eq!(
        first_schema
            .logical_column_for::<Position>()
            .expect("schema should contain Position")
            .name(),
        type_name::<Position>()
    );
    assert_eq!(
        first_schema
            .logical_column_for::<Velocity>()
            .expect("schema should contain Velocity")
            .name(),
        type_name::<Velocity>()
    );
}

#[test]
fn table_preserves_physical_column_order_and_maps_logical_access_to_that_order() {
    let mut catalog = Catalog::new();
    let table = catalog
        .register_table(
            TableDeclaration::new([
                LogicalColumnDeclaration::of::<Position>(),
                LogicalColumnDeclaration::of::<Velocity>(),
            ])
            .with_physical_columns([
                PhysicalColumnDeclaration::inline::<Velocity>(),
                PhysicalColumnDeclaration::inline::<Position>(),
            ]),
            Configuration::default(),
        )
        .expect("table registration should succeed");

    assert_eq!(table.physical_columns()[0].name(), type_name::<Velocity>());
    assert_eq!(table.physical_columns()[1].name(), type_name::<Position>());

    let physical_access_set = table
        .map_access::<Position>(Access::Write)
        .expect("table should map Position access");

    assert_eq!(physical_access_set.access(), Access::Write);
    assert_eq!(physical_access_set.physical_column_indices().len(), 1);
    assert_eq!(physical_access_set.physical_column_indices()[0].value(), 1);
}

#[test]
fn managed_key_tables_append_the_managed_key_column_and_record_the_policy() {
    let mut catalog = Catalog::new();
    let table = catalog
        .register_table(
            TableDeclaration::new([LogicalColumnDeclaration::of::<Position>()])
                .with_identity(Identity::ManagedKeys),
            Configuration::default(),
        )
        .expect("managed-key table registration should succeed");

    assert_eq!(table.identity(), Identity::ManagedKeys);
    assert_eq!(table.layout().physical_column_count(), 2);
    assert_eq!(
        table
            .managed_key_physical_column()
            .expect("managed-key table should expose the managed key column")
            .value(),
        1
    );
    assert_eq!(table.physical_columns()[1].owner(), ColumnOwner::ManagedKey);
    assert_eq!(
        table.row_address_layout().row_index_bit_count()
            + table.row_address_layout().chunk_index_bit_count(),
        32
    );
}

#[test]
fn resource_generation_follows_the_table_chunk_and_physical_column_identity_model(
) -> Result<(), String> {
    let mut catalog = Catalog::new();
    let table = catalog
        .register_table(
            TableDeclaration::new([LogicalColumnDeclaration::of::<Position>()])
                .with_physical_columns([
                    PhysicalColumnDeclaration::for_logical_column::<Position>(
                        "Position::x",
                        size_of::<f64>(),
                        align_of::<f64>(),
                        Storage::Inline,
                    ),
                    PhysicalColumnDeclaration::for_logical_column::<Position>(
                        "Position::y",
                        size_of::<f64>(),
                        align_of::<f64>(),
                        Storage::Inline,
                    ),
                ]),
            Configuration::default(),
        )
        .expect("decomposed table registration should succeed");

    let store_index = StoreIndex::new(0);
    let chunk_index_value_generator = 0u32..1024u32;

    chunk_index_value_generator
        .check(|chunk_index_value| {
            let physical_access_set = table
                .map_access::<Position>(Access::Read)
                .expect("table should map decomposed Position access");
            let resources: Vec<_> = physical_access_set
                .resources(store_index, ChunkIndex::new(chunk_index_value))
                .collect();

            assert_eq!(resources.len(), 2);
            assert_eq!(resources[0].store_index(), store_index);
            assert_eq!(resources[0].table_index(), Some(table.index()));
            assert_eq!(
                resources[0].chunk_index(),
                Some(ChunkIndex::new(chunk_index_value))
            );
            assert_eq!(
                resources[0]
                    .physical_column_index()
                    .expect("resource should be a physical column")
                    .value(),
                0
            );
            assert_eq!(
                resources[1]
                    .physical_column_index()
                    .expect("resource should be a physical column")
                    .value(),
                1
            );
        })
        .map_or(Ok(()), |failure| Err(format!("{failure:?}")))
}

#[test]
fn dependencies_for_physical_column_access_include_read_ancestors_and_write_leaves() {
    let mut catalog = Catalog::new();
    let table = catalog
        .register_table(
            TableDeclaration::new([LogicalColumnDeclaration::of::<Position>()])
                .with_physical_columns([
                    PhysicalColumnDeclaration::for_logical_column::<Position>(
                        "Position::x",
                        size_of::<f64>(),
                        align_of::<f64>(),
                        Storage::Inline,
                    ),
                    PhysicalColumnDeclaration::for_logical_column::<Position>(
                        "Position::y",
                        size_of::<f64>(),
                        align_of::<f64>(),
                        Storage::Inline,
                    ),
                ]),
            Configuration::default(),
        )
        .expect("decomposed table registration should succeed");

    let dependencies: Vec<_> = table
        .map_access::<Position>(Access::Write)
        .expect("table should map Position access")
        .dependencies(StoreIndex::new(0), ChunkIndex::new(2))
        .collect();

    assert_eq!(
        dependencies,
        vec![
            Dependency::read(Resource::store(StoreIndex::new(0))),
            Dependency::read(Resource::table(StoreIndex::new(0), table.index())),
            Dependency::read(Resource::chunk(
                StoreIndex::new(0),
                table.index(),
                ChunkIndex::new(2)
            )),
            Dependency::write(Resource::physical_column(
                StoreIndex::new(0),
                table.index(),
                ChunkIndex::new(2),
                table.physical_columns()[0].index(),
            )),
            Dependency::write(Resource::physical_column(
                StoreIndex::new(0),
                table.index(),
                ChunkIndex::new(2),
                table.physical_columns()[1].index(),
            )),
        ]
    );
}

#[test]
fn metadata_can_name_broader_store_table_and_chunk_dependencies() {
    let mut catalog = Catalog::new();
    let table = catalog
        .register_table(
            TableDeclaration::new([LogicalColumnDeclaration::of::<Position>()]),
            Configuration::default(),
        )
        .expect("table registration should succeed");

    let store_index = StoreIndex::new(0);
    let chunk_index = ChunkIndex::new(2);

    assert_eq!(
        table.store_dependencies(store_index, Access::Write),
        [Dependency::write(Resource::store(store_index))]
    );
    assert_eq!(
        table.table_dependencies(store_index, Access::Write),
        [
            Dependency::read(Resource::store(store_index)),
            Dependency::write(Resource::table(store_index, table.index()))
        ]
    );
    assert_eq!(
        table.chunk_dependencies(store_index, chunk_index, Access::Write),
        [
            Dependency::read(Resource::store(store_index)),
            Dependency::read(Resource::table(store_index, table.index())),
            Dependency::write(Resource::chunk(store_index, table.index(), chunk_index))
        ]
    );
}

#[test]
fn table_registration_rejects_logical_columns_without_physical_coverage() {
    let mut catalog = Catalog::new();
    let error = catalog
        .register_table(
            TableDeclaration::new([
                LogicalColumnDeclaration::of::<Position>(),
                LogicalColumnDeclaration::of::<Velocity>(),
            ])
            .with_physical_columns([PhysicalColumnDeclaration::inline::<Position>()]),
            Configuration::default(),
        )
        .expect_err("table registration should reject uncovered logical columns");

    assert_eq!(
        error,
        DefinitionError::MissingPhysicalColumnsForLogicalColumn {
            logical_column_name: type_name::<Velocity>(),
        }
    );
}

#[test]
fn table_registration_rejects_orphan_physical_columns() {
    let mut catalog = Catalog::new();
    let error = catalog
        .register_table(
            TableDeclaration::new([LogicalColumnDeclaration::of::<Position>()])
                .with_physical_columns([PhysicalColumnDeclaration::inline::<Velocity>()]),
            Configuration::default(),
        )
        .expect_err("table registration should reject orphan physical columns");

    assert_eq!(
        error,
        DefinitionError::MissingLogicalColumnForPhysicalColumn {
            physical_column_name: type_name::<Velocity>(),
            logical_column_name: type_name::<Velocity>(),
        }
    );
}
