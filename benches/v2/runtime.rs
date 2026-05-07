use criterion::{BatchSize, BenchmarkId, Criterion};
use std::hint::black_box;
use std::num::NonZeroUsize;
use that_bass::v2::{
    Configuration, Store, command, query,
    runtime::{Callbacks, Executor, FunctionContext, Injection, Options, ResolveContext},
    schedule::Builder,
    schema::{Meta, Table},
};

#[repr(C)]
struct Position {
    x: f32,
    y: f32,
}

#[repr(C)]
struct Spawner {
    count: u32,
}

pub fn benchmark(criterion: &mut Criterion) {
    benchmark_many_tiny_jobs(criterion);
    benchmark_resolve_injection(criterion);
}

fn benchmark_many_tiny_jobs(criterion: &mut Criterion) {
    let mut benchmark_group = criterion.benchmark_group("rewrite_runtime/many_tiny_jobs");
    let chunk_counts = [64usize, 256, 1024];
    let worker_counts = [1usize, 4];
    let injections = [Injection::PreferProducer, Injection::SharedFirst];

    for chunk_count in chunk_counts {
        for worker_count in worker_counts {
            for injection in injections {
                let (schedule, mut store) = scan_schedule(chunk_count);
                benchmark_group.bench_with_input(
                    BenchmarkId::new(
                        format!("chunk_count/workers_{worker_count}/{injection:?}"),
                        chunk_count,
                    ),
                    &chunk_count,
                    |bencher, _chunk_count| {
                        bencher.iter(|| {
                            let report = Executor::with_options(
                                Options::default()
                                    .with_worker_count(non_zero_usize(worker_count))
                                    .with_injection(injection)
                                    .with_record_trace(false),
                            )
                            .run(&schedule, &mut store, &NoopCallbacks);

                            black_box(report);
                        });
                    },
                );
            }
        }
    }

    benchmark_group.finish();
}

fn benchmark_resolve_injection(criterion: &mut Criterion) {
    let mut benchmark_group = criterion.benchmark_group("rewrite_runtime/resolve_injection");
    let chunk_counts = [16usize, 64, 256];
    let worker_counts = [1usize, 4];
    let injections = [Injection::PreferProducer, Injection::SharedFirst];

    for chunk_count in chunk_counts {
        for worker_count in worker_counts {
            for injection in injections {
                let (schedule, spawn_index) = injection_schedule(chunk_count);
                let callbacks = InjectionCallbacks { spawn_index };
                benchmark_group.bench_with_input(
                    BenchmarkId::new(
                        format!("chunk_count/workers_{worker_count}/{injection:?}"),
                        chunk_count,
                    ),
                    &chunk_count,
                    |bencher, chunk_count| {
                        bencher.iter_batched(
                            || seeded_injection_store(*chunk_count),
                            |mut store| {
                                let report = Executor::with_options(
                                    Options::default()
                                        .with_worker_count(non_zero_usize(worker_count))
                                        .with_injection(injection)
                                        .with_record_trace(false),
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
    }

    benchmark_group.finish();
}

struct NoopCallbacks;

impl Callbacks for NoopCallbacks {
    fn run_function(&self, _context: FunctionContext<'_, '_>) {}
}

struct InjectionCallbacks {
    spawn_index: that_bass::v2::schedule::FunctionIndex,
}

impl Callbacks for InjectionCallbacks {
    fn run_function(&self, mut context: FunctionContext<'_, '_>) {
        if context.function_index() == self.spawn_index {
            context
                .insert::<(Position,)>()
                .expect("spawn function should expose its typed insert buffer")
                .one((Position { x: 1.0, y: 2.0 },));
        }
    }

    fn run_resolve(&self, _context: ResolveContext<'_>) {}
}

fn scan_schedule(chunk_count: usize) -> (that_bass::v2::schedule::Schedule, Store) {
    let mut store = Store::with_configuration(test_configuration());
    let table_index = store
        .register_table([Meta::of::<Position>()])
        .expect("benchmark table registration should succeed");
    populate_chunks(
        store
            .table_mut(table_index)
            .expect("registered table should stay addressable"),
        chunk_count,
    );

    let mut builder = Builder::new(&mut store);
    builder.push_query(
        "scan",
        query::all(query::read::<Position>()).expect("benchmark query declaration should succeed"),
    );
    let schedule = builder.build();

    (schedule, store)
}

fn injection_schedule(
    chunk_count: usize,
) -> (
    that_bass::v2::schedule::Schedule,
    that_bass::v2::schedule::FunctionIndex,
) {
    let mut store = Store::with_configuration(test_configuration());
    let table_index = store
        .register_table([Meta::of::<Spawner>()])
        .expect("benchmark table registration should succeed");
    populate_chunks(
        store
            .table_mut(table_index)
            .expect("registered table should stay addressable"),
        chunk_count,
    );

    let mut builder = Builder::new(&mut store);
    let spawn_index = builder.push_query(
        "spawn",
        query::all(query::read::<Spawner>()).expect("benchmark query declaration should succeed"),
    );
    builder.push_query(
        "clamp",
        query::all(query::read::<Position>()).expect("benchmark query declaration should succeed"),
    );
    let position_table_index = builder
        .add_insert(spawn_index, command::Insert::<(Position,)>::new())
        .expect("typed insert should resolve to one known table");
    let _ = position_table_index;
    let schedule = builder.build();

    (schedule, spawn_index)
}

fn seeded_injection_store(chunk_count: usize) -> Store {
    let mut store = Store::with_configuration(test_configuration());
    let spawner_table_index = store
        .register_table([Meta::of::<Spawner>()])
        .expect("benchmark table registration should succeed");
    let _position_table_index = store
        .register_table([Meta::of::<Position>()])
        .expect("benchmark table registration should succeed");
    populate_chunks(
        store
            .table_mut(spawner_table_index)
            .expect("registered table should stay addressable"),
        chunk_count,
    );

    store
}

fn populate_chunks(table: &mut Table, chunk_count: usize) {
    for _ in 0..chunk_count {
        black_box(table.push_chunk());
    }
}

fn non_zero_usize(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("benchmark constants must be non-zero")
}

fn test_configuration() -> Configuration {
    Configuration::default()
        .with_target_chunk_byte_count(NonZeroUsize::new(128).expect("constant must be non-zero"))
}
