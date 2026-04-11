use checkito::{Check, Generate};
use std::num::NonZeroUsize;
use that_bass::v2::{
    instrumentation::{Category, FOUNDATION_MEASUREMENT_CATEGORIES},
    ChunkPlan, Configuration, Store,
};

#[test]
fn foundation_measurement_categories_cover_the_required_plan() {
    assert_eq!(FOUNDATION_MEASUREMENT_CATEGORIES.len(), 5);
    assert!(FOUNDATION_MEASUREMENT_CATEGORIES.contains(&Category::ScheduleBuildCost));
    assert!(FOUNDATION_MEASUREMENT_CATEGORIES.contains(&Category::RuntimeQueueOverhead));
    assert!(FOUNDATION_MEASUREMENT_CATEGORIES.contains(&Category::ChunkAllocationCost));
    assert!(FOUNDATION_MEASUREMENT_CATEGORIES.contains(&Category::ResolveCost));
    assert!(FOUNDATION_MEASUREMENT_CATEGORIES.contains(&Category::DenseScanThroughput));
}

#[test]
fn store_configuration_override_changes_chunk_capacity_planning() {
    let smaller_configuration =
        Configuration::default().with_target_chunk_byte_count(NonZeroUsize::new(8 * 1024).unwrap());
    let larger_configuration = Configuration::default()
        .with_target_chunk_byte_count(NonZeroUsize::new(32 * 1024).unwrap());

    let smaller_plan = smaller_configuration.plan_chunk_capacity_for_row_width(32);
    let larger_plan = larger_configuration.plan_chunk_capacity_for_row_width(32);

    assert!(smaller_plan.target_chunk_capacity() < larger_plan.target_chunk_capacity());
}

#[test]
fn chunk_capacity_plan_is_never_zero() -> Result<(), String> {
    let target_chunk_byte_count_generator = 1usize..(1 << 20);
    let inline_row_width_generator = ..(1usize << 16);

    (
        &target_chunk_byte_count_generator,
        &inline_row_width_generator,
    )
        .check(|(target_chunk_byte_count, inline_row_width)| {
            let configuration = Configuration::default().with_target_chunk_byte_count(
                NonZeroUsize::new(target_chunk_byte_count)
                    .expect("generated target chunk byte count must be non-zero"),
            );
            let plan = configuration.plan_chunk_capacity_for_row_width(inline_row_width);

            assert!(plan.target_chunk_capacity() >= 1);
            assert!(plan.target_chunk_capacity().is_power_of_two());
        })
        .map_or(Ok(()), |failure| Err(format!("{failure:?}")))
}

#[test]
fn chunk_capacity_plan_matches_the_selected_formula() -> Result<(), String> {
    let target_chunk_byte_count_generator = 1usize..(1 << 20);
    let inline_row_width_generator = ..(1usize << 16);

    (
        &target_chunk_byte_count_generator,
        &inline_row_width_generator,
    )
        .check(|(target_chunk_byte_count, inline_row_width)| {
            let configuration = Configuration::default().with_target_chunk_byte_count(
                NonZeroUsize::new(target_chunk_byte_count)
                    .expect("generated target chunk byte count must be non-zero"),
            );
            let plan = configuration.plan_chunk_capacity_for_row_width(inline_row_width);

            assert_chunk_capacity_plan_matches_the_selected_formula(plan);
        })
        .map_or(Ok(()), |failure| Err(format!("{failure:?}")))
}

#[test]
fn increasing_target_chunk_bytes_never_decreases_chunk_capacity() -> Result<(), String> {
    let base_target_chunk_byte_count_generator = 1usize..(1 << 18);
    let additional_target_chunk_byte_count_generator = ..(1usize << 18);
    let inline_row_width_generator = ..(1usize << 16);

    (
        &base_target_chunk_byte_count_generator,
        &additional_target_chunk_byte_count_generator,
        &inline_row_width_generator,
    )
        .map(
            |(
                base_target_chunk_byte_count,
                additional_target_chunk_byte_count,
                inline_row_width,
            )| {
                (
                    base_target_chunk_byte_count,
                    base_target_chunk_byte_count + additional_target_chunk_byte_count,
                    inline_row_width,
                )
            },
        )
        .check(
            |(
                smaller_target_chunk_byte_count,
                larger_target_chunk_byte_count,
                inline_row_width,
            )| {
                let smaller_configuration = Configuration::default().with_target_chunk_byte_count(
                    NonZeroUsize::new(smaller_target_chunk_byte_count)
                        .expect("generated smaller target chunk byte count must be non-zero"),
                );
                let larger_configuration = Configuration::default().with_target_chunk_byte_count(
                    NonZeroUsize::new(larger_target_chunk_byte_count)
                        .expect("generated larger target chunk byte count must be non-zero"),
                );

                let smaller_capacity =
                    smaller_configuration.target_chunk_capacity_for_row_width(inline_row_width);
                let larger_capacity =
                    larger_configuration.target_chunk_capacity_for_row_width(inline_row_width);

                assert!(smaller_capacity <= larger_capacity);
            },
        )
        .map_or(Ok(()), |failure| Err(format!("{failure:?}")))
}

#[test]
fn store_uses_the_same_capacity_formula_as_the_configuration() {
    let configuration = Configuration::default()
        .with_target_chunk_byte_count(NonZeroUsize::new(32 * 1024).unwrap());
    let store = Store::with_configuration(configuration);

    assert_eq!(
        store.plan_chunk_capacity_for_row_width(48),
        configuration.plan_chunk_capacity_for_row_width(48)
    );
}

fn assert_chunk_capacity_plan_matches_the_selected_formula(chunk_plan: ChunkPlan) {
    let normalized_inline_row_width = chunk_plan.inline_row_width().max(1);
    let raw_target_row_count =
        (chunk_plan.target_chunk_byte_count().get() / normalized_inline_row_width).max(1);
    let expected_target_chunk_capacity = 1usize << raw_target_row_count.ilog2();

    assert_eq!(
        chunk_plan.normalized_inline_row_width(),
        normalized_inline_row_width
    );
    assert_eq!(chunk_plan.raw_target_row_count(), raw_target_row_count);
    assert_eq!(
        chunk_plan.target_chunk_capacity(),
        expected_target_chunk_capacity
    );
}
