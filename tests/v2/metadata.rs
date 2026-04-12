use checkito::Check;
use core::any::type_name;
use that_bass::v2::{
    key::Key,
    query::Access,
    schema::{
        Catalog, ChunkIndex, Column, DefinitionError, Dependency, Meta, Resource, RowLayout,
        Storage, StoreIndex, TableIndex,
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
fn catalog_interns_equivalent_table_shapes_even_when_meta_declarations_are_reordered() {
    let mut catalog = Catalog::new();

    catalog
        .register_table(
            [Meta::of::<Velocity>(), Meta::of::<Position>()],
            Configuration::default(),
        )
        .expect("first table registration should succeed");
    catalog
        .register_table(
            [Meta::of::<Position>(), Meta::of::<Velocity>()],
            Configuration::default(),
        )
        .expect("equivalent table registration should succeed");

    assert_eq!(catalog.table_shape_count(), 1);
}

#[test]
fn table_keeps_meta_order_sorted_by_type_identifier_and_maps_access_by_type() {
    let mut catalog = Catalog::new();
    let table = catalog
        .register_table(
            [Meta::of::<Position>(), Meta::of::<Velocity>()],
            Configuration::default(),
        )
        .expect("table registration should succeed");

    let position_access = table
        .map_access::<Position>(Access::Write)
        .expect("table should map Position access");

    assert_eq!(table.metas().len(), 2);
    assert_eq!(position_access.access(), Access::Write);
    assert_eq!(
        table
            .meta(position_access.column_index())
            .expect("table should expose the mapped meta")
            .identifier(),
        core::any::TypeId::of::<Position>()
    );
}

#[test]
fn tables_treat_key_meta_as_ordinary_type_metadata() {
    let mut catalog = Catalog::new();
    let table = catalog
        .register_table(
            [Meta::of::<Position>(), Meta::of::<Key>()],
            Configuration::default(),
        )
        .expect("table registration with Key metadata should succeed");

    let key_meta = table.meta_for::<Key>().expect("table should contain Key");

    assert_eq!(table.layout().column_count(), 2);
    assert_eq!(key_meta.name(), type_name::<Key>());
    assert_eq!(
        table.row_layout().row_index_bit_count() + table.row_layout().chunk_index_bit_count(),
        32
    );
    assert_eq!(
        table
            .map_access::<Key>(Access::Read)
            .expect("Key access should map like any other column")
            .column_index(),
        table
            .map_access_for_identifier(key_meta.identifier(), Access::Read)
            .expect("identifier-based access should match typed access")
            .column_index(),
    );
}

#[test]
fn resource_generation_follows_the_table_chunk_and_column_identity_model() -> Result<(), String> {
    let mut catalog = Catalog::new();
    let table = catalog
        .register_table([Meta::of::<Position>()], Configuration::default())
        .expect("table registration should succeed");

    let store_index = StoreIndex::new(0);
    let chunk_index_value_generator = 0u32..1024u32;

    chunk_index_value_generator
        .check(|chunk_index_value| {
            let column_access = table
                .map_access::<Position>(Access::Read)
                .expect("table should map Position access");
            let resource = column_access.resource(store_index, ChunkIndex::new(chunk_index_value));

            assert_eq!(resource.store_index(), store_index);
            assert_eq!(resource.table_index(), Some(table.index()));
            assert_eq!(
                resource.chunk_index(),
                Some(ChunkIndex::new(chunk_index_value))
            );
            assert_eq!(resource.column_index(), Some(column_access.column_index()));
        })
        .map_or(Ok(()), |failure| Err(format!("{failure:?}")))
}

#[test]
fn dependencies_for_column_access_include_read_ancestors_and_write_leaf() {
    let mut catalog = Catalog::new();
    let table = catalog
        .register_table([Meta::of::<Position>()], Configuration::default())
        .expect("table registration should succeed");
    let column_access = table
        .map_access::<Position>(Access::Write)
        .expect("table should map Position access");

    assert_eq!(
        column_access.dependencies(StoreIndex::new(0), ChunkIndex::new(2)),
        [
            Dependency::read(Resource::store(StoreIndex::new(0))),
            Dependency::read(Resource::table(StoreIndex::new(0), table.index())),
            Dependency::read(Resource::chunk(
                StoreIndex::new(0),
                table.index(),
                ChunkIndex::new(2)
            )),
            Dependency::write(Resource::column(
                StoreIndex::new(0),
                table.index(),
                ChunkIndex::new(2),
                column_access.column_index(),
            )),
        ]
    );
}

#[test]
fn metadata_can_name_broader_store_table_and_chunk_dependencies() {
    let mut catalog = Catalog::new();
    let table = catalog
        .register_table([Meta::of::<Position>()], Configuration::default())
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
fn table_registration_rejects_duplicate_meta() {
    let mut catalog = Catalog::new();
    let error = catalog
        .register_table(
            [Meta::of::<Position>(), Meta::of::<Position>()],
            Configuration::default(),
        )
        .expect_err("table registration should reject duplicate metadata");

    assert_eq!(
        error,
        DefinitionError::DuplicateMeta {
            meta_name: type_name::<Position>(),
        }
    );
}

#[test]
fn sidecar_meta_does_not_contribute_to_row_width() {
    let mut catalog = Catalog::new();
    let table = catalog
        .register_table(
            [Meta::of::<Position>(), Meta::sidecar::<Velocity>()],
            Configuration::default(),
        )
        .expect("table registration should succeed");

    assert_eq!(table.layout().row_width(), core::mem::size_of::<Position>());
    assert_eq!(table.layout().column_count(), 2);
    assert_eq!(
        table
            .meta_for::<Velocity>()
            .expect("table should contain Velocity")
            .storage(),
        Storage::Sidecar
    );
}

#[test]
fn row_layout_packs_and_unpacks_chunk_and_row_indices() {
    let row_layout =
        RowLayout::try_for_chunk_capacity(256).expect("power-of-two chunk capacities are valid");
    let row = row_layout
        .row(TableIndex::new(7), ChunkIndex::new(19), 42)
        .expect("row values within the configured bit budget should be valid");

    assert_eq!(row.table_index(), TableIndex::new(7));
    assert_eq!(row_layout.chunk_index(row), ChunkIndex::new(19));
    assert_eq!(row_layout.row_index(row), 42);
}

#[test]
fn row_layout_rejects_non_power_of_two_chunk_capacity() {
    assert_eq!(
        RowLayout::try_for_chunk_capacity(3),
        Err(that_bass::v2::schema::DefinitionError::InvalidChunkCapacity { capacity: 3 })
    );
}

#[test]
fn row_layout_supports_a_full_u32_row_partition_when_the_platform_can_represent_it() {
    let Some(chunk_capacity) = 1usize.checked_shl(u32::BITS) else {
        return;
    };

    let row_layout = RowLayout::try_for_chunk_capacity(chunk_capacity)
        .expect("a full-row partition should be valid when the platform can represent it");
    let row = row_layout
        .row(TableIndex::new(9), ChunkIndex::new(0), u32::MAX)
        .expect("the full u32 row range should pack successfully");

    assert_eq!(row.table_index(), TableIndex::new(9));
    assert_eq!(row_layout.chunk_index(row), ChunkIndex::new(0));
    assert_eq!(row_layout.row_index(row), u32::MAX);
}

#[test]
fn row_layout_rejects_chunk_capacity_that_exceeds_the_packed_row_encoding() {
    let Some(excessive_chunk_capacity) = 1usize.checked_shl(u32::BITS + 1) else {
        return;
    };

    assert_eq!(
        RowLayout::try_for_chunk_capacity(excessive_chunk_capacity),
        Err(DefinitionError::InvalidChunkCapacity {
            capacity: excessive_chunk_capacity
        })
    );
}

#[test]
fn row_layout_rejects_row_index_that_exceeds_the_available_bits() {
    let row_layout =
        RowLayout::try_for_chunk_capacity(256).expect("power-of-two chunk capacities are valid");

    assert_eq!(
        row_layout.row(TableIndex::new(0), ChunkIndex::new(0), 256),
        Err(that_bass::v2::schema::ChunkError::RowIndexOutOfBounds {
            row_index: 256,
            capacity: 256
        })
    );
}

#[test]
fn column_wrapper_pairs_pointer_and_meta() {
    let mut bytes = 13u32;
    let pointer = core::ptr::NonNull::from(&mut bytes).cast::<u8>();
    let meta = Meta::of::<u32>();
    let column = Column::new(pointer, &meta);

    assert_eq!(column.pointer(), pointer);
    assert_eq!(column.meta().identifier(), core::any::TypeId::of::<u32>());
}
