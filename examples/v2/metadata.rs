use core::num::NonZeroUsize;
use that_bass::v2::{
    key::Key,
    query::Access,
    schema::{Catalog, ChunkIndex, Meta, StoreIndex},
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
            [
                Meta::of::<Position>(),
                Meta::of::<Velocity>(),
                Meta::of::<Key>(),
            ],
            Configuration::default().with_target_chunk_byte_count(non_zero_usize(8 * 1024)),
        )
        .expect("example table registration should succeed");

    let position_access = table
        .map_access::<Position>(Access::Write)
        .expect("example table should map Position access");
    let store_index = StoreIndex::new(0);
    let resource = position_access.resource(store_index, ChunkIndex::new(7));
    let dependencies = position_access.dependencies(store_index, ChunkIndex::new(7));
    let chunk_write_dependencies =
        table.chunk_dependencies(store_index, ChunkIndex::new(7), Access::Write);

    println!("Metadata catalog");
    println!("  table shape count: {}", catalog.table_shape_count());
    println!("  table count: {}", catalog.table_count());
    println!("  table index: {}", table.index().value());
    println!(
        "  metas: {} total, row width {} bytes",
        table.metas().len(),
        table.layout().row_width()
    );
    println!(
        "  chunk capacity: {} rows with {} row bits and {} chunk bits",
        table.chunk_plan().target_chunk_capacity(),
        table.row_layout().row_index_bit_count(),
        table.row_layout().chunk_index_bit_count()
    );
    println!("  mapped Position resource for chunk 7: {:?}", resource);
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
