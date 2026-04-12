use std::num::NonZeroUsize;
use that_bass::v2::{
    query,
    schema::{Catalog, Meta},
    Configuration,
};

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Position {
    x: f64,
    y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Velocity {
    x: f64,
    y: f64,
}

pub fn run() {
    let mut catalog = Catalog::new();
    let mut table = catalog
        .register_table(
            [Meta::of::<Position>(), Meta::of::<Velocity>()],
            Configuration::default().with_target_chunk_byte_count(
                NonZeroUsize::new(1024).expect("example constant must be non-zero"),
            ),
        )
        .expect("table registration should succeed");

    let chunk_index = loop {
        let chunk_index = table.push_chunk();
        let chunk = table
            .chunk(chunk_index)
            .expect("newly pushed chunk must be addressable");
        if chunk.capacity() >= 2 {
            break chunk_index;
        }
    };

    unsafe {
        table
            .write::<Position>(chunk_index, 0, Position { x: 1.0, y: 2.0 })
            .expect("direct write should succeed");
        table
            .write::<Velocity>(chunk_index, 0, Velocity { x: 10.0, y: 20.0 })
            .expect("direct write should succeed");
        table
            .write::<Position>(chunk_index, 1, Position { x: 3.0, y: 4.0 })
            .expect("direct write should succeed");
        table
            .write::<Velocity>(chunk_index, 1, Velocity { x: 30.0, y: 40.0 })
            .expect("direct write should succeed");
        table
            .assume_initialized_prefix(chunk_index, 2)
            .expect("initialized prefix declaration should succeed");
    }

    let query = query::all((
        query::rows(),
        query::write::<Position>(),
        query::option(query::read::<Velocity>()),
    ))
    .expect("query declaration should succeed");

    let row_layout = table.row_layout();
    let (rows, positions, velocities) = query
        .project_chunk(&mut table, chunk_index)
        .expect("query projection should succeed");

    for ((row, position), velocity) in rows.zip(positions).zip(velocities) {
        let velocity = velocity.expect("velocity should be present");
        position.y += velocity.y;
        debug_assert_eq!(row_layout.chunk_index(row), chunk_index);
    }

    let updated_positions = table
        .slice::<Position>(chunk_index)
        .expect("dense-prefix slice should succeed");

    println!("Query surface");
    println!(
        "  conflicting self-query rejected: {:?}",
        query::all((query::write::<Position>(), query::read::<Position>()))
    );
    println!(
        "  updated positions: {:?}",
        updated_positions
            .iter()
            .map(|position| (position.x, position.y))
            .collect::<Vec<_>>()
    );
}
