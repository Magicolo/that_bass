use checkito::Check;
use std::num::NonZeroUsize;
use that_bass::v2::{
    schema::{Catalog, ColumnIndex, Meta},
    Configuration,
};

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
struct Position {
    x: f64,
    y: f64,
}

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
struct LargeRow {
    values: [u64; 8],
}

#[derive(Debug, Clone, PartialEq)]
#[repr(C, align(32))]
struct AlignedByte {
    value: u8,
}

#[test]
fn table_precomputes_bootstrap_chunk_layouts_from_one_to_the_full_capacity() {
    let mut catalog = Catalog::new();
    let table = catalog
        .register_table(
            [Meta::of::<LargeRow>()],
            Configuration::default().with_target_chunk_byte_count(
                NonZeroUsize::new(8 * 1024).expect("test constants must be non-zero"),
            ),
        )
        .expect("table registration should succeed");

    let capacities: Vec<_> = table
        .chunk_layouts()
        .iter()
        .map(|chunk_layout| chunk_layout.capacity())
        .collect();

    assert_eq!(capacities, vec![1, 2, 4, 8, 16, 32, 64, 128]);
    assert_eq!(
        table.full_chunk_layout().capacity(),
        table.chunk_plan().target_chunk_capacity()
    );
}

#[test]
fn chunk_layout_offsets_are_monotonic_and_aligned() -> Result<(), String> {
    let target_chunk_byte_count_generator = 1usize..(1 << 14);

    target_chunk_byte_count_generator
        .check(|target_chunk_byte_count| {
            let mut catalog = Catalog::new();
            let table = catalog
                .register_table(
                    [
                        Meta::of::<u8>(),
                        Meta::of::<u32>(),
                        Meta::of::<AlignedByte>(),
                        Meta::sidecar::<Position>(),
                    ],
                    Configuration::default().with_target_chunk_byte_count(
                        NonZeroUsize::new(target_chunk_byte_count)
                            .expect("generated target chunk byte count must be non-zero"),
                    ),
                )
                .expect("table registration should succeed");

            for chunk_layout in table.chunk_layouts() {
                let mut previous_inline_offset = 0usize;

                for column_layout in chunk_layout.columns().iter().copied() {
                    if let Some(offset) = column_layout.offset() {
                        assert!(offset >= previous_inline_offset);
                        assert_eq!(offset % column_layout.alignment(), 0);
                        assert!(
                            offset + column_layout.region_size() <= chunk_layout.allocation_size()
                        );
                        previous_inline_offset = offset;
                    } else {
                        assert_eq!(column_layout.region_size(), 0);
                    }
                }
            }
        })
        .map_or(Ok(()), |failure| Err(format!("{failure:?}")))
}

#[test]
fn pushing_chunks_uses_bootstrap_growth_then_repeats_the_full_capacity() {
    let mut catalog = Catalog::new();
    let mut table = catalog
        .register_table(
            [Meta::of::<LargeRow>()],
            Configuration::default().with_target_chunk_byte_count(
                NonZeroUsize::new(8 * 1024).expect("test constants must be non-zero"),
            ),
        )
        .expect("table registration should succeed");

    let capacities: Vec<_> = (0..9)
        .map(|_| {
            let chunk_index = table.push_chunk();
            table
                .chunk(chunk_index)
                .expect("newly pushed chunks must be addressable")
                .capacity()
        })
        .collect();

    assert_eq!(capacities, vec![1, 2, 4, 8, 16, 32, 64, 128, 128]);
}

#[test]
fn chunk_allocation_uses_one_block_with_offsets_matching_the_layout() {
    let mut catalog = Catalog::new();
    let mut table = catalog
        .register_table(
            [
                Meta::of::<u8>(),
                Meta::of::<u32>(),
                Meta::of::<AlignedByte>(),
                Meta::sidecar::<Position>(),
            ],
            Configuration::default(),
        )
        .expect("table registration should succeed");

    let chunk_index = table.push_chunk();
    let chunk = table
        .chunk(chunk_index)
        .expect("newly pushed chunk must be addressable");
    let chunk_layout = table
        .chunk_layout_for_capacity(chunk.capacity())
        .expect("table must expose the precomputed layout for the chunk capacity");
    let base_address = chunk.base_pointer().as_ptr() as usize;

    for (column_position, column_layout) in chunk_layout.columns().iter().copied().enumerate() {
        let column_index = ColumnIndex::new(column_position as u16);
        let column = chunk
            .column(column_index)
            .expect("chunk must expose one pointer per column");

        if let Some(offset) = column_layout.offset() {
            assert_eq!(column.pointer().as_ptr() as usize, base_address + offset);
        } else {
            assert_eq!(
                column.pointer().as_ptr() as usize % column_layout.alignment(),
                0
            );
        }
    }
}

#[test]
fn dense_prefix_slices_read_the_initialized_rows() {
    let mut catalog = Catalog::new();
    let mut table = catalog
        .register_table(
            [Meta::of::<Position>()],
            Configuration::default()
                .with_target_chunk_byte_count(NonZeroUsize::new(32).expect("non-zero constant")),
        )
        .expect("table registration should succeed");

    let _ = table.push_chunk();
    let chunk_index = table.push_chunk();

    unsafe {
        table
            .write::<Position>(chunk_index, 0, Position { x: 1.0, y: 2.0 })
            .expect("direct write should succeed");
        table
            .write::<Position>(chunk_index, 1, Position { x: 3.0, y: 4.0 })
            .expect("direct write should succeed");
        table
            .assume_initialized_prefix(chunk_index, 2)
            .expect("initialized prefix declaration should succeed");
    }

    let positions = table
        .slice::<Position>(chunk_index)
        .expect("typed dense-prefix slice should succeed");

    assert_eq!(
        positions,
        &[Position { x: 1.0, y: 2.0 }, Position { x: 3.0, y: 4.0 },]
    );
}

#[test]
fn swap_remove_row_maintains_a_dense_prefix_for_move_only_values() {
    let mut catalog = Catalog::new();
    let mut table = catalog
        .register_table(
            [Meta::of::<String>()],
            Configuration::default()
                .with_target_chunk_byte_count(NonZeroUsize::new(128).expect("non-zero constant")),
        )
        .expect("table registration should succeed");

    let _ = table.push_chunk();
    let _ = table.push_chunk();
    let chunk_index = table.push_chunk();

    unsafe {
        table
            .write::<String>(chunk_index, 0, "first".to_owned())
            .expect("direct write should succeed");
        table
            .write::<String>(chunk_index, 1, "second".to_owned())
            .expect("direct write should succeed");
        table
            .write::<String>(chunk_index, 2, "third".to_owned())
            .expect("direct write should succeed");
        table
            .assume_initialized_prefix(chunk_index, 3)
            .expect("initialized prefix declaration should succeed");
    }

    table
        .swap_remove_row(chunk_index, 1)
        .expect("swap-remove inside the initialized prefix should succeed");

    let labels = table
        .slice::<String>(chunk_index)
        .expect("typed dense-prefix slice should succeed");

    assert_eq!(labels, &["first".to_owned(), "third".to_owned()]);
    assert_eq!(
        table
            .chunk(chunk_index)
            .expect("chunk must still exist")
            .count(),
        2
    );
}
