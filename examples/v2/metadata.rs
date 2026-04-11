use core::{
    mem::{align_of, size_of},
    num::NonZeroUsize,
};
use that_bass::v2::{
    query::Access,
    schema::{
        Catalog, ChunkIndex, Identity, LogicalColumnDeclaration, PhysicalColumnDeclaration,
        Storage, StoreIndex, TableDeclaration,
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

pub fn run() {
    let mut catalog = Catalog::new();
    let table = catalog
        .register_table(
            TableDeclaration::new([
                LogicalColumnDeclaration::of::<Position>(),
                LogicalColumnDeclaration::of::<Velocity>(),
            ])
            .with_identity(Identity::ManagedKeys)
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
                PhysicalColumnDeclaration::inline::<Velocity>(),
            ]),
            Configuration::default().with_target_chunk_byte_count(non_zero_usize(8 * 1024)),
        )
        .expect("example table registration should succeed");

    let physical_access_set = table
        .map_access::<Position>(Access::Write)
        .expect("example table should map Position access");
    let store_index = StoreIndex::new(0);
    let resources: Vec<_> = physical_access_set
        .resources(store_index, ChunkIndex::new(7))
        .collect();
    let dependencies: Vec<_> = physical_access_set
        .dependencies(store_index, ChunkIndex::new(7))
        .collect();
    let chunk_write_dependencies =
        table.chunk_dependencies(store_index, ChunkIndex::new(7), Access::Write);

    println!("Metadata catalog");
    println!("  schema count: {}", catalog.schema_count());
    println!("  table count: {}", catalog.table_count());
    println!("  schema index: {}", table.schema().index().value());
    println!("  table index: {}", table.index().value());
    println!(
        "  physical columns: {} total, {} inline bytes per row",
        table.layout().physical_column_count(),
        table.layout().inline_row_width()
    );
    println!(
        "  chunk capacity: {} rows with {} row bits and {} chunk bits",
        table.chunk_plan().target_chunk_capacity(),
        table.row_address_layout().row_index_bit_count(),
        table.row_address_layout().chunk_index_bit_count()
    );
    println!(
        "  mapped Position leaf resources for chunk 7: {:?}",
        resources
    );
    println!(
        "  full Position dependencies for chunk 7: {:?}",
        dependencies
    );
    println!(
        "  broad chunk write dependencies for chunk 7: {:?}",
        chunk_write_dependencies
    );
}

fn non_zero_usize(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("example constants must be non-zero")
}
