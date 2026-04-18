use core::mem::size_of;
use core::num::NonZeroUsize;
use that_bass::v2::{Configuration, Store};

#[repr(C)]
struct BodyState {
    position: [f32; 3],
    velocity: [f32; 3],
}

pub fn run() {
    let default_store = Store::new();
    let tuned_configuration =
        Configuration::default().with_target_chunk_byte_count(non_zero_usize(8 * 1024));
    let tuned_store = Store::with_configuration(tuned_configuration);

    let row_width = size_of::<BodyState>();
    let default_chunk_plan = default_store.plan_chunk_capacity_for_row_width(row_width);
    let tuned_chunk_plan = tuned_store.plan_chunk_capacity_for_row_width(row_width);

    println!("Store planning");
    println!("  row width: {row_width} bytes");
    println!(
        "  default target: {} bytes -> chunk capacity {} rows",
        default_chunk_plan.target_chunk_byte_count().get(),
        default_chunk_plan.target_chunk_capacity()
    );
    println!(
        "  tuned target: {} bytes -> chunk capacity {} rows",
        tuned_chunk_plan.target_chunk_byte_count().get(),
        tuned_chunk_plan.target_chunk_capacity()
    );
}

fn non_zero_usize(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("example constants must be non-zero")
}
