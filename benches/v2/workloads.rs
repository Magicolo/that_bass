use criterion::{BatchSize, BenchmarkId, Criterion};
use std::{hint::black_box, num::NonZeroUsize};
use that_bass::v2::{
    command, key, query,
    runtime::{Callbacks, Executor, FunctionContext, Options},
    schedule::{Builder, Schedule},
    schema::{Meta, Table},
    Configuration, Store,
};

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NarrowRow {
    x: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MediumRow {
    values: [f32; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct WideRow {
    values: [f32; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Position {
    x: f32,
    y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Time {
    seconds: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TagA;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TagB;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TagC;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TagD;

pub fn benchmark(criterion: &mut Criterion) {
    benchmark_scan_row_widths(criterion);
    benchmark_table_shape_balance(criterion);
    benchmark_keyed_vs_keyless_scan(criterion);
    benchmark_singleton_access(criterion);
    benchmark_remove_heavy_runtime(criterion);
}

fn benchmark_scan_row_widths(criterion: &mut Criterion) {
    let mut benchmark_group = criterion.benchmark_group("rewrite_workloads/scan_row_widths");
    let chunk_counts = [32usize, 128];

    for chunk_count in chunk_counts {
        benchmark_group.bench_with_input(
            BenchmarkId::new("narrow", chunk_count),
            &chunk_count,
            |bencher, chunk_count| {
                let mut store = Store::with_configuration(test_configuration());
                let table_index = store
                    .register_table([Meta::of::<NarrowRow>()])
                    .expect("benchmark table registration should succeed");
                populate_copy_rows(
                    store
                        .table_mut(table_index)
                        .expect("registered table should stay addressable"),
                    *chunk_count,
                    NarrowRow { x: 1.0 },
                );
                let query =
                    query::all(query::read::<NarrowRow>()).expect("benchmark query should build");

                bencher.iter(|| black_box(scan_read_query(&mut store, &query)));
            },
        );

        benchmark_group.bench_with_input(
            BenchmarkId::new("medium", chunk_count),
            &chunk_count,
            |bencher, chunk_count| {
                let mut store = Store::with_configuration(test_configuration());
                let table_index = store
                    .register_table([Meta::of::<MediumRow>()])
                    .expect("benchmark table registration should succeed");
                populate_copy_rows(
                    store
                        .table_mut(table_index)
                        .expect("registered table should stay addressable"),
                    *chunk_count,
                    MediumRow { values: [1.0; 8] },
                );
                let query =
                    query::all(query::read::<MediumRow>()).expect("benchmark query should build");

                bencher.iter(|| black_box(scan_read_query(&mut store, &query)));
            },
        );

        benchmark_group.bench_with_input(
            BenchmarkId::new("wide", chunk_count),
            &chunk_count,
            |bencher, chunk_count| {
                let mut store = Store::with_configuration(test_configuration());
                let table_index = store
                    .register_table([Meta::of::<WideRow>()])
                    .expect("benchmark table registration should succeed");
                populate_copy_rows(
                    store
                        .table_mut(table_index)
                        .expect("registered table should stay addressable"),
                    *chunk_count,
                    WideRow { values: [1.0; 32] },
                );
                let query =
                    query::all(query::read::<WideRow>()).expect("benchmark query should build");

                bencher.iter(|| black_box(scan_read_query(&mut store, &query)));
            },
        );
    }

    benchmark_group.finish();
}

fn benchmark_table_shape_balance(criterion: &mut Criterion) {
    let mut benchmark_group = criterion.benchmark_group("rewrite_workloads/table_shape_balance");

    benchmark_group.bench_function("one_dominant_table", |bencher| {
        let mut store = Store::with_configuration(test_configuration());
        let table_index = store
            .register_table([Meta::of::<Position>()])
            .expect("benchmark table registration should succeed");
        populate_copy_rows(
            store
                .table_mut(table_index)
                .expect("registered table should stay addressable"),
            256,
            Position { x: 1.0, y: 2.0 },
        );
        let query = query::all(query::read::<Position>()).expect("benchmark query should build");

        bencher.iter(|| black_box(scan_read_query(&mut store, &query)));
    });

    benchmark_group.bench_function("many_medium_tables", |bencher| {
        let mut store = Store::with_configuration(test_configuration());
        let table_indices = [
            store
                .register_table([Meta::of::<Position>(), Meta::of::<TagA>()])
                .expect("benchmark table registration should succeed"),
            store
                .register_table([Meta::of::<Position>(), Meta::of::<TagB>()])
                .expect("benchmark table registration should succeed"),
            store
                .register_table([Meta::of::<Position>(), Meta::of::<TagC>()])
                .expect("benchmark table registration should succeed"),
            store
                .register_table([Meta::of::<Position>(), Meta::of::<TagD>()])
                .expect("benchmark table registration should succeed"),
        ];
        populate_position_and_tag_rows::<TagA>(
            store
                .table_mut(table_indices[0])
                .expect("registered table should stay addressable"),
            64,
        );
        populate_position_and_tag_rows::<TagB>(
            store
                .table_mut(table_indices[1])
                .expect("registered table should stay addressable"),
            64,
        );
        populate_position_and_tag_rows::<TagC>(
            store
                .table_mut(table_indices[2])
                .expect("registered table should stay addressable"),
            64,
        );
        populate_position_and_tag_rows::<TagD>(
            store
                .table_mut(table_indices[3])
                .expect("registered table should stay addressable"),
            64,
        );
        let query = query::all(query::read::<Position>()).expect("benchmark query should build");

        bencher.iter(|| black_box(scan_read_query(&mut store, &query)));
    });

    benchmark_group.finish();
}

fn benchmark_keyed_vs_keyless_scan(criterion: &mut Criterion) {
    let mut benchmark_group = criterion.benchmark_group("rewrite_workloads/keyed_vs_keyless");

    benchmark_group.bench_function("keyless_scan", |bencher| {
        let mut store = Store::with_configuration(test_configuration());
        let table_index = store
            .register_table([Meta::of::<Position>()])
            .expect("benchmark table registration should succeed");
        populate_copy_rows(
            store
                .table_mut(table_index)
                .expect("registered table should stay addressable"),
            128,
            Position { x: 1.0, y: 2.0 },
        );
        let query = query::all(query::read::<Position>()).expect("benchmark query should build");

        bencher.iter(|| black_box(scan_read_query(&mut store, &query)));
    });

    benchmark_group.bench_function("managed_key_column_scan", |bencher| {
        let mut store = Store::with_configuration(test_configuration());
        let table_index = store
            .register_table([Meta::of::<key::Key>(), Meta::of::<Position>()])
            .expect("benchmark table registration should succeed");
        populate_keyed_position_rows(
            store
                .table_mut(table_index)
                .expect("registered table should stay addressable"),
            128,
        );
        let query = query::all(query::read::<Position>()).expect("benchmark query should build");

        bencher.iter(|| black_box(scan_read_query(&mut store, &query)));
    });

    benchmark_group.finish();
}

fn benchmark_singleton_access(criterion: &mut Criterion) {
    let mut benchmark_group = criterion.benchmark_group("rewrite_workloads/singleton_access");
    let mut store = Store::with_configuration(test_configuration());
    store
        .initialize_global(Time { seconds: 0.016 })
        .expect("benchmark global initialization should succeed");
    let singleton = query::one::<Time>();

    benchmark_group.bench_function("query_one_read", |bencher| {
        bencher.iter(|| {
            let time = singleton
                .get(&store)
                .expect("singleton query should resolve cleanly");
            black_box(time.seconds);
        });
    });

    benchmark_group.finish();
}

fn benchmark_remove_heavy_runtime(criterion: &mut Criterion) {
    let mut benchmark_group = criterion.benchmark_group("rewrite_workloads/remove_heavy_runtime");
    let worker_counts = [1usize, 4];
    let chunk_counts = [32usize, 128];
    let schedule = remove_schedule();
    let callbacks = RemoveAllCallbacks;

    for worker_count in worker_counts {
        for chunk_count in chunk_counts {
            benchmark_group.bench_with_input(
                BenchmarkId::new(format!("workers_{worker_count}"), chunk_count),
                &chunk_count,
                |bencher, chunk_count| {
                    bencher.iter_batched(
                        || seeded_remove_store(*chunk_count),
                        |mut store| {
                            let report = Executor::with_options(
                                Options::default().with_worker_count(non_zero_usize(worker_count)),
                            )
                            .run(&schedule, &mut store, &callbacks);
                            black_box(report);
                        },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }

    benchmark_group.finish();
}

struct RemoveAllCallbacks;

impl Callbacks for RemoveAllCallbacks {
    fn run_function(&self, mut context: FunctionContext<'_, '_>) {
        let rows = context.rows();
        context
            .remove()
            .expect("remove benchmark should expose a remove buffer")
            .extend(rows);
    }
}

fn remove_schedule() -> Schedule {
    let mut store = Store::with_configuration(test_configuration());
    store
        .register_table([Meta::of::<Position>()])
        .expect("benchmark table registration should succeed");

    let mut builder = Builder::new(&mut store);
    let remove_index = builder.push_query(
        "remove",
        query::all(query::rows()).expect("benchmark query should build"),
    );
    builder
        .add_remove(remove_index, command::Remove::new(query::has::<Position>()))
        .expect("remove planning should succeed");

    builder.build()
}

fn seeded_remove_store(chunk_count: usize) -> Store {
    let mut store = Store::with_configuration(test_configuration());
    let table_index = store
        .register_table([Meta::of::<Position>()])
        .expect("benchmark table registration should succeed");
    populate_copy_rows(
        store
            .table_mut(table_index)
            .expect("registered table should stay addressable"),
        chunk_count,
        Position { x: 1.0, y: 2.0 },
    );

    store
}

fn scan_read_query<T>(store: &mut Store, query: &query::All<query::Read<T>>) -> usize
where
    T: 'static,
{
    let matching_table_indices = store
        .tables()
        .iter()
        .filter(|table| query.matches_table(table))
        .map(Table::index)
        .collect::<Vec<_>>();
    let mut observed_row_count = 0usize;

    for table_index in matching_table_indices {
        let chunk_indices = store
            .table(table_index)
            .expect("matched table should stay addressable")
            .chunks()
            .iter()
            .map(|chunk| chunk.chunk_index())
            .collect::<Vec<_>>();
        let table = store
            .table_mut(table_index)
            .expect("matched table should stay addressable");

        for chunk_index in chunk_indices {
            let rows = query
                .project_chunk(table, chunk_index)
                .expect("matching benchmark query should project cleanly");
            observed_row_count += rows.len();
            black_box(rows.as_ptr());
        }
    }

    observed_row_count
}

fn populate_copy_rows<T>(table: &mut Table, chunk_count: usize, value: T)
where
    T: Copy + 'static,
{
    for _ in 0..chunk_count {
        let chunk_index = table.push_chunk();
        let row_count = table
            .chunk(chunk_index)
            .expect("new chunk should stay addressable")
            .capacity();
        for row_index in 0..row_count {
            unsafe {
                table
                    .write::<T>(chunk_index, row_index, value)
                    .expect("benchmark direct write should succeed");
            }
        }
        unsafe {
            table
                .assume_initialized_prefix(chunk_index, row_count)
                .expect("benchmark initialized prefix declaration should succeed");
        }
    }
}

fn populate_position_and_tag_rows<Tag>(table: &mut Table, chunk_count: usize)
where
    Tag: Copy + Default + 'static,
{
    for _ in 0..chunk_count {
        let chunk_index = table.push_chunk();
        let row_count = table
            .chunk(chunk_index)
            .expect("new chunk should stay addressable")
            .capacity();
        for row_index in 0..row_count {
            unsafe {
                table
                    .write::<Position>(chunk_index, row_index, Position { x: 1.0, y: 2.0 })
                    .expect("benchmark direct write should succeed");
                table
                    .write::<Tag>(chunk_index, row_index, Tag::default())
                    .expect("benchmark direct write should succeed");
            }
        }
        unsafe {
            table
                .assume_initialized_prefix(chunk_index, row_count)
                .expect("benchmark initialized prefix declaration should succeed");
        }
    }
}

fn populate_keyed_position_rows(table: &mut Table, chunk_count: usize) {
    let mut next_slot_index = 0u32;

    for _ in 0..chunk_count {
        let chunk_index = table.push_chunk();
        let row_count = table
            .chunk(chunk_index)
            .expect("new chunk should stay addressable")
            .capacity();
        for row_index in 0..row_count {
            let key = key::Key::new(next_slot_index, 0);
            next_slot_index = next_slot_index.saturating_add(1);

            unsafe {
                table
                    .write::<key::Key>(chunk_index, row_index, key)
                    .expect("benchmark direct write should succeed");
                table
                    .write::<Position>(chunk_index, row_index, Position { x: 1.0, y: 2.0 })
                    .expect("benchmark direct write should succeed");
            }
        }
        unsafe {
            table
                .assume_initialized_prefix(chunk_index, row_count)
                .expect("benchmark initialized prefix declaration should succeed");
        }
    }
}

fn non_zero_usize(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("benchmark constants must be non-zero")
}

fn test_configuration() -> Configuration {
    Configuration::default()
        .with_target_chunk_byte_count(NonZeroUsize::new(1024).expect("constant must be non-zero"))
}
