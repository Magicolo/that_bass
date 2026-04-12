use checkito::Check;
use std::num::NonZeroUsize;
use that_bass::v2::{
    command::Remove,
    query,
    schema::{Catalog, ChunkError, ChunkIndex, Meta, Table},
    Configuration,
};

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Position {
    x: f64,
    y: f64,
}

#[test]
fn rows_views_align_with_dense_prefix_chunk_data() -> Result<(), String> {
    if cfg!(miri) {
        for row_count in [0usize, 1, 2, 7, 15, 31] {
            assert_rows_view_alignment(row_count);
        }

        return Ok(());
    }

    let row_count_generator = 0usize..32usize;

    row_count_generator
        .check(assert_rows_view_alignment)
        .map_or(Ok(()), |failure| Err(format!("{failure:?}")))
}

#[test]
fn batched_remove_deduplicates_targets_and_applies_descending_row_order() {
    let mut catalog = Catalog::new();
    let mut table = catalog
        .register_table(
            [Meta::of::<String>()],
            Configuration::default().with_target_chunk_byte_count(
                NonZeroUsize::new(512).expect("test constant must be non-zero"),
            ),
        )
        .expect("table registration should succeed");
    let chunk_index = push_chunk_with_minimum_capacity(&mut table, 8);

    write_strings(
        &mut table,
        chunk_index,
        &["zero", "one", "two", "three", "four"],
    );

    let rows = table
        .rows(chunk_index)
        .expect("table should expose a generated rows view");
    let mut remove = Remove::new();
    remove.one(rows.get(1).expect("row 1 should exist"));
    remove.one(rows.get(3).expect("row 3 should exist"));
    remove.one(rows.get(3).expect("row 3 should still exist"));

    let removed_row_count = remove
        .resolve_on(&mut table)
        .expect("batched remove should resolve successfully");
    let labels = table
        .slice::<String>(chunk_index)
        .expect("dense-prefix slice should succeed");

    assert_eq!(removed_row_count, 2);
    assert_eq!(
        labels,
        &["zero".to_owned(), "four".to_owned(), "two".to_owned()]
    );
}

#[test]
fn row_handles_are_transient_and_can_be_reused_after_remove_resolution() {
    let mut catalog = Catalog::new();
    let mut table = catalog
        .register_table(
            [Meta::of::<String>()],
            Configuration::default().with_target_chunk_byte_count(
                NonZeroUsize::new(512).expect("test constant must be non-zero"),
            ),
        )
        .expect("table registration should succeed");
    let chunk_index = push_chunk_with_minimum_capacity(&mut table, 4);

    write_strings(&mut table, chunk_index, &["alpha", "beta", "gamma"]);

    let rows_before = table
        .rows(chunk_index)
        .expect("table should expose a generated rows view");
    let removed_row = rows_before.first().expect("first row should exist");
    let removed_value = table
        .slice::<String>(chunk_index)
        .expect("dense-prefix slice should succeed")[0]
        .clone();

    let mut remove = Remove::new();
    remove.one(removed_row);
    remove
        .resolve_on(&mut table)
        .expect("batched remove should resolve successfully");

    let rows_after = table
        .rows(chunk_index)
        .expect("table should still expose a generated rows view");
    let labels = table
        .slice::<String>(chunk_index)
        .expect("dense-prefix slice should succeed");

    assert_eq!(rows_after.first(), Some(removed_row));
    assert_ne!(labels[0], removed_value);
    assert_eq!(labels, &["gamma".to_owned(), "beta".to_owned()]);
}

#[test]
fn remove_buffer_rejects_rows_from_a_different_table() {
    let mut catalog = Catalog::new();
    let mut left_table = catalog
        .register_table(
            [Meta::of::<Position>()],
            Configuration::default().with_target_chunk_byte_count(
                NonZeroUsize::new(256).expect("test constant must be non-zero"),
            ),
        )
        .expect("left table registration should succeed");
    let mut right_table = catalog
        .register_table(
            [Meta::of::<Position>()],
            Configuration::default().with_target_chunk_byte_count(
                NonZeroUsize::new(256).expect("test constant must be non-zero"),
            ),
        )
        .expect("right table registration should succeed");

    let left_chunk_index = push_chunk_with_minimum_capacity(&mut left_table, 1);
    unsafe {
        left_table
            .write::<Position>(left_chunk_index, 0, Position { x: 1.0, y: 2.0 })
            .expect("direct write should succeed");
        left_table
            .assume_initialized_prefix(left_chunk_index, 1)
            .expect("initialized prefix declaration should succeed");
    }

    let left_row = left_table
        .rows(left_chunk_index)
        .expect("table should expose a generated rows view")
        .first()
        .expect("the first row should exist");
    let mut remove = Remove::new();
    remove.one(left_row);

    let error = remove
        .resolve_on(&mut right_table)
        .expect_err("table resolution should reject a row from another table");

    assert_eq!(
        error,
        ChunkError::RowTableMismatch {
            expected_table_index: right_table.index(),
            actual_table_index: left_table.index(),
        }
    );
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

fn assert_rows_view_alignment(row_count: usize) {
    let mut catalog = Catalog::new();
    let mut table = catalog
        .register_table(
            [Meta::of::<Position>()],
            Configuration::default().with_target_chunk_byte_count(
                NonZeroUsize::new(1024).expect("test constant must be non-zero"),
            ),
        )
        .expect("table registration should succeed");

    let chunk_index =
        push_chunk_with_minimum_capacity(&mut table, row_count.max(1).next_power_of_two());

    for row_index in 0..row_count {
        unsafe {
            table
                .write::<Position>(
                    chunk_index,
                    row_index,
                    Position {
                        x: row_index as f64,
                        y: row_index as f64 * 10.0,
                    },
                )
                .expect("direct write should succeed");
        }
    }

    unsafe {
        table
            .assume_initialized_prefix(chunk_index, row_count)
            .expect("initialized prefix declaration should succeed");
    }

    let rows_request = query::rows();
    let rows = table
        .rows(chunk_index)
        .expect("table should expose a generated rows view");
    let positions = table
        .slice::<Position>(chunk_index)
        .expect("dense-prefix slice should succeed");

    let (prefix_rows, suffix_rows) = rows
        .split_at(row_count.min(1))
        .expect("split inside the dense prefix should succeed");

    assert_eq!(rows_request, query::RowsRequest);
    assert_eq!(rows.len(), row_count);
    assert_eq!(rows.len(), positions.len());
    assert_eq!(prefix_rows.len() + suffix_rows.len(), rows.len());
    assert_eq!(rows.first(), rows.get(0));
    assert_eq!(
        rows.last(),
        row_count.checked_sub(1).and_then(|index| rows.get(index))
    );

    for (expected_row_index, (row, position)) in rows.zip(positions).enumerate() {
        assert_eq!(table.row_layout().chunk_index(row), chunk_index);
        assert_eq!(table.row_layout().row_index(row), expected_row_index as u32);
        assert_eq!(position.x, expected_row_index as f64);
    }
}

fn write_strings(table: &mut Table, chunk_index: ChunkIndex, labels: &[&str]) {
    for (row_index, label) in labels.iter().enumerate() {
        unsafe {
            table
                .write::<String>(chunk_index, row_index, (*label).to_owned())
                .expect("direct write should succeed");
        }
    }

    unsafe {
        table
            .assume_initialized_prefix(chunk_index, labels.len())
            .expect("initialized prefix declaration should succeed");
    }
}
