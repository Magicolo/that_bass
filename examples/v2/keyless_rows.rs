use std::num::NonZeroUsize;
use that_bass::v2::{
    command::Remove,
    query,
    schema::{Catalog, Meta},
    Configuration,
};

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
struct Lifetime {
    frames_remaining: u32,
}

pub fn run() {
    let mut catalog = Catalog::new();
    let mut table = catalog
        .register_table(
            [Meta::of::<Lifetime>()],
            Configuration::default().with_target_chunk_byte_count(
                NonZeroUsize::new(256).expect("example constant must be non-zero"),
            ),
        )
        .expect("table registration should succeed");

    let chunk_index = loop {
        let chunk_index = table.push_chunk();
        let chunk = table
            .chunk(chunk_index)
            .expect("newly pushed chunk must be addressable");
        if chunk.capacity() >= 4 {
            break chunk_index;
        }
    };

    for (row_index, frames_remaining) in [3u32, 0, 5, 0].into_iter().enumerate() {
        unsafe {
            table
                .write::<Lifetime>(chunk_index, row_index, Lifetime { frames_remaining })
                .expect("direct write should succeed");
        }
    }

    unsafe {
        table
            .assume_initialized_prefix(chunk_index, 4)
            .expect("initialized prefix declaration should succeed");
    }

    let rows_request = query::rows();
    let rows = table
        .rows(chunk_index)
        .expect("table should expose a generated rows view");
    let lifetimes = table
        .slice::<Lifetime>(chunk_index)
        .expect("dense-prefix slice should succeed");
    let mut remove = Remove::new();

    for (row, lifetime) in rows.zip(lifetimes) {
        if lifetime.frames_remaining == 0 {
            remove.one(row);
        }
    }

    let removed_row_count = remove
        .resolve_on(&mut table)
        .expect("batched remove should resolve successfully");
    let remaining_lifetimes = table
        .slice::<Lifetime>(chunk_index)
        .expect("dense-prefix slice should succeed");

    println!("Keyless rows");
    println!("  request descriptor: {rows_request:?}");
    println!("  removed rows: {removed_row_count}");
    println!(
        "  surviving frame counts: {:?}",
        remaining_lifetimes
            .iter()
            .map(|lifetime| lifetime.frames_remaining)
            .collect::<Vec<_>>()
    );
}
