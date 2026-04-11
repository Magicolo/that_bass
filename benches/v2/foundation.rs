use criterion::{BenchmarkId, Criterion};
use std::hint::black_box;
use std::num::NonZeroUsize;
use that_bass::{v1::Database, v2::Configuration};

pub fn benchmark(criterion: &mut Criterion) {
    benchmark_boundary_construction(criterion);
    benchmark_chunk_capacity_planning(criterion);
}

fn benchmark_boundary_construction(criterion: &mut Criterion) {
    let mut benchmark_group = criterion.benchmark_group("rewrite_foundation/boundary_construction");

    benchmark_group.bench_function("v1/database_new", |bencher| {
        bencher.iter(|| black_box(Database::new()))
    });
    benchmark_group.bench_function("v2/store_new", |bencher| {
        bencher.iter(|| black_box(that_bass::v2::Store::new()))
    });

    benchmark_group.finish();
}

fn benchmark_chunk_capacity_planning(criterion: &mut Criterion) {
    let mut benchmark_group =
        criterion.benchmark_group("rewrite_foundation/chunk_capacity_planning");
    let target_chunk_byte_counts = [8 * 1024, 16 * 1024, 32 * 1024, 64 * 1024];
    let inline_row_widths = [8usize, 16, 32, 64, 128];

    for target_chunk_byte_count in target_chunk_byte_counts {
        let configuration = Configuration::default().with_target_chunk_byte_count(
            NonZeroUsize::new(target_chunk_byte_count)
                .expect("benchmark target chunk byte count must be non-zero"),
        );

        for inline_row_width in inline_row_widths {
            benchmark_group.bench_with_input(
                BenchmarkId::new(
                    format!("{target_chunk_byte_count}_target_chunk_bytes"),
                    inline_row_width,
                ),
                &inline_row_width,
                |bencher, inline_row_width| {
                    bencher.iter(|| {
                        black_box(
                            configuration.plan_chunk_capacity_for_row_width(*inline_row_width),
                        )
                    })
                },
            );
        }
    }

    benchmark_group.finish();
}
