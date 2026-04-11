use core::mem::size_of;
use core::num::NonZeroUsize;
use that_bass::v2::{schema::SchemaLayout, Configuration, Store};

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

    let body_state_layout = SchemaLayout::new(size_of::<BodyState>(), 2);
    let default_chunk_plan =
        default_store.plan_chunk_capacity_for_row_width(body_state_layout.inline_row_width());
    let tuned_chunk_plan =
        tuned_store.plan_chunk_capacity_for_row_width(body_state_layout.inline_row_width());

    println!("Store planning");
    println!(
        "  row width: {} bytes across {} physical columns",
        body_state_layout.inline_row_width(),
        body_state_layout.physical_column_count()
    );
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
