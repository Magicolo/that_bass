use core::num::NonZeroUsize;
use that_bass::v2::{
    schema::{Catalog, Meta},
    Configuration,
};

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
struct Position {
    x: f64,
    y: f64,
}

pub fn run() {
    let mut catalog = Catalog::new();
    let mut table = catalog
        .register_table(
            [Meta::of::<Position>()],
            Configuration::default()
                .with_target_chunk_byte_count(NonZeroUsize::new(32).expect("non-zero constant")),
        )
        .expect("example table registration should succeed");

    let bootstrap_chunk_index = table.push_chunk();
    let full_chunk_index = table.push_chunk();

    unsafe {
        table
            .write::<Position>(full_chunk_index, 0, Position { x: 1.0, y: 2.0 })
            .expect("direct write should succeed");
        table
            .write::<Position>(full_chunk_index, 1, Position { x: 3.0, y: 4.0 })
            .expect("direct write should succeed");
        table
            .assume_initialized_prefix(full_chunk_index, 2)
            .expect("initialized prefix declaration should succeed");
    }

    let positions = table
        .slice::<Position>(full_chunk_index)
        .expect("typed slice should succeed");
    let full_chunk = table
        .chunk(full_chunk_index)
        .expect("full chunk must exist after allocation");
    let full_chunk_layout = table
        .chunk_layout_for_capacity(full_chunk.capacity())
        .expect("precomputed layout must exist for the full chunk capacity");

    println!("Chunk layout");
    println!(
        "  bootstrap chunk capacity: {} row",
        table
            .chunk(bootstrap_chunk_index)
            .expect("bootstrap chunk must exist")
            .capacity()
    );
    println!("  full chunk capacity: {} rows", full_chunk.capacity());
    println!(
        "  full chunk allocation: {} bytes at alignment {}",
        full_chunk_layout.allocation_size(),
        full_chunk_layout.allocation_alignment()
    );
    println!(
        "  first column offset: {:?}",
        full_chunk_layout.columns()[0].offset()
    );
    println!("  initialized rows: {:?}", positions);
}
