use std::num::NonZeroUsize;
use that_bass::v2::{
    query::{self, Error, View},
    schema::{Catalog, ChunkIndex, Meta, Table},
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

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Dynamic;

#[test]
fn read_query_projects_a_dense_chunk_slice() {
    let mut table = make_table([Meta::of::<Position>()]);
    let chunk_index = push_chunk_with_minimum_capacity(&mut table, 2);
    write_positions(
        &mut table,
        chunk_index,
        &[Position { x: 1.0, y: 2.0 }, Position { x: 3.0, y: 4.0 }],
    );

    let query = query::all(query::read::<Position>()).expect("query declaration should succeed");
    let positions = query
        .project_chunk(&mut table, chunk_index)
        .expect("read query projection should succeed");

    assert_eq!(
        positions,
        &[Position { x: 1.0, y: 2.0 }, Position { x: 3.0, y: 4.0 }]
    );
}

#[test]
fn read_query_rejects_sidecar_only_columns() {
    let mut table = make_table([Meta::sidecar::<Position>()]);
    let chunk_index = push_chunk_with_minimum_capacity(&mut table, 1);
    let query = query::all(query::read::<Position>()).expect("query declaration should succeed");
    let table_index = table.index();

    assert_eq!(
        query.project_chunk(&mut table, chunk_index),
        Err(Error::TableDoesNotMatch { table_index })
    );
}

#[test]
fn write_query_projects_a_mutable_dense_chunk_slice() {
    let mut table = make_table([Meta::of::<Position>()]);
    let chunk_index = push_chunk_with_minimum_capacity(&mut table, 2);
    write_positions(
        &mut table,
        chunk_index,
        &[Position { x: 1.0, y: 2.0 }, Position { x: 3.0, y: 4.0 }],
    );

    let query = query::all(query::write::<Position>()).expect("query declaration should succeed");
    let positions = query
        .project_chunk(&mut table, chunk_index)
        .expect("write query projection should succeed");

    for position in positions {
        position.y += 10.0;
    }

    assert_eq!(
        table
            .slice::<Position>(chunk_index)
            .expect("dense-prefix slice should succeed"),
        &[Position { x: 1.0, y: 12.0 }, Position { x: 3.0, y: 14.0 }]
    );
}

#[test]
fn rows_and_data_queries_share_the_same_positional_indexing() {
    let mut table = make_table([Meta::of::<Position>()]);
    let chunk_index = push_chunk_with_minimum_capacity(&mut table, 2);
    write_positions(
        &mut table,
        chunk_index,
        &[Position { x: 5.0, y: 6.0 }, Position { x: 7.0, y: 8.0 }],
    );

    let query = query::all((query::rows(), query::read::<Position>()))
        .expect("query declaration should succeed");
    let row_layout = table.row_layout();
    let (rows, positions) = query
        .project_chunk(&mut table, chunk_index)
        .expect("rows query projection should succeed");

    for (row_index, (row, position)) in rows.zip(positions).enumerate() {
        assert_eq!(row_layout.row_index(row), row_index as u32);
        assert_eq!(position.x, 5.0 + row_index as f64 * 2.0);
    }
}

#[test]
fn option_query_yields_none_values_when_the_sub_query_is_missing() {
    let mut table = make_table([Meta::of::<Position>()]);
    let chunk_index = push_chunk_with_minimum_capacity(&mut table, 2);
    write_positions(
        &mut table,
        chunk_index,
        &[Position { x: 1.0, y: 2.0 }, Position { x: 3.0, y: 4.0 }],
    );

    let query = query::all((
        query::read::<Position>(),
        query::option(query::read::<Velocity>()),
    ))
    .expect("query declaration should succeed");
    let (positions, velocities) = query
        .project_chunk(&mut table, chunk_index)
        .expect("optional query projection should succeed");

    assert!(!velocities.is_present());
    assert_eq!(velocities.len(), positions.len());

    for (position, velocity) in positions.zip(velocities) {
        assert!(position.x >= 1.0);
        assert!(velocity.is_none());
    }
}

#[test]
fn option_query_yields_some_values_when_the_sub_query_is_present() {
    let mut table = make_table([Meta::of::<Position>(), Meta::of::<Velocity>()]);
    let chunk_index = push_chunk_with_minimum_capacity(&mut table, 2);

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
        query::read::<Position>(),
        query::option(query::read::<Velocity>()),
    ))
    .expect("query declaration should succeed");
    let (positions, velocities) = query
        .project_chunk(&mut table, chunk_index)
        .expect("optional query projection should succeed");

    assert!(velocities.is_present());

    for (position, velocity) in positions.zip(velocities) {
        let velocity = velocity.expect("velocity should be present");
        assert_eq!(velocity.x, position.x * 10.0);
    }
}

#[test]
fn all_query_construction_rejects_obvious_aliasing_conflicts() {
    assert_eq!(
        query::all((query::write::<Position>(), query::read::<Position>())),
        Err(Error::ConflictingAccess {
            left_type_name: core::any::type_name::<Position>(),
            left_access: query::Access::Write,
            right_type_name: core::any::type_name::<Position>(),
            right_access: query::Access::Read,
        })
    );
}

#[test]
fn disjoint_filter_split_is_accepted_for_same_written_type() {
    let left = query::all(query::write::<Position>())
        .expect("query declaration should succeed")
        .filter(query::has::<Dynamic>());
    let right = query::all(query::write::<Position>())
        .expect("query declaration should succeed")
        .filter(query::not(query::has::<Dynamic>()));

    assert!(!left.conflicts_with(&right));
}

#[test]
fn one_query_projects_one_row_as_a_single_item() {
    let mut table = make_table([Meta::of::<Position>()]);
    let chunk_index = push_chunk_with_minimum_capacity(&mut table, 1);
    write_positions(&mut table, chunk_index, &[Position { x: 9.0, y: 10.0 }]);

    let query = query::all(query::one(query::read::<Position>()))
        .expect("query declaration should succeed");
    let position = query
        .project_chunk(&mut table, chunk_index)
        .expect("one query projection should succeed");

    assert_eq!(position, &Position { x: 9.0, y: 10.0 });
}

#[test]
fn one_query_rejects_chunks_with_more_than_one_row() {
    let mut table = make_table([Meta::of::<Position>()]);
    let chunk_index = push_chunk_with_minimum_capacity(&mut table, 2);
    write_positions(
        &mut table,
        chunk_index,
        &[Position { x: 1.0, y: 2.0 }, Position { x: 3.0, y: 4.0 }],
    );

    let query = query::all(query::one(query::read::<Position>()))
        .expect("query declaration should succeed");
    let table_index = table.index();

    assert_eq!(
        query.project_chunk(&mut table, chunk_index),
        Err(Error::InvalidOneCardinality {
            table_index,
            chunk_index,
            count: 2,
        })
    );
}

fn make_table<const COLUMN_COUNT: usize>(metas: [Meta; COLUMN_COUNT]) -> Table {
    let mut catalog = Catalog::new();
    catalog
        .register_table(
            metas,
            Configuration::default().with_target_chunk_byte_count(
                NonZeroUsize::new(1024).expect("test constant must be non-zero"),
            ),
        )
        .expect("table registration should succeed")
}

fn push_chunk_with_minimum_capacity(table: &mut Table, minimum_capacity: usize) -> ChunkIndex {
    loop {
        let chunk_index = table.push_chunk();
        let chunk = table
            .chunk(chunk_index)
            .expect("newly pushed chunk must be addressable");
        if chunk.capacity() >= minimum_capacity {
            return chunk_index;
        }
    }
}

fn write_positions(table: &mut Table, chunk_index: ChunkIndex, positions: &[Position]) {
    for (row_index, position) in positions.iter().copied().enumerate() {
        unsafe {
            table
                .write::<Position>(chunk_index, row_index, position)
                .expect("direct write should succeed");
        }
    }

    unsafe {
        table
            .assume_initialized_prefix(chunk_index, positions.len())
            .expect("initialized prefix declaration should succeed");
    }
}
