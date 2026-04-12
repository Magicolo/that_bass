use criterion::{BenchmarkId, Criterion};
use std::hint::black_box;
use std::num::NonZeroUsize;
use that_bass::v2::{
    command, query,
    runtime::{
        Callbacks, Executor, FunctionContext, Options, Outcome, ResolveContext, Seed, VisibleChunk,
    },
    schedule::Builder,
    schema::{Catalog, ChunkIndex, Meta, Table, TableIndex},
    Configuration,
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

    for chunk_count in chunk_counts {
        benchmark_group.bench_with_input(
            BenchmarkId::new("chunk_count", chunk_count),
            &chunk_count,
            |bencher, chunk_count| {
                bencher.iter(|| {
                    let (schedule, seed) = scan_schedule(*chunk_count);
                    let report = Executor::with_options(
                        Options::default()
                            .with_worker_count(non_zero_usize(4))
                            .with_record_trace(false),
                    )
                    .run(&schedule, &seed, &NoopCallbacks);

                    black_box(report);
                });
            },
        );
    }

    benchmark_group.finish();
}

fn benchmark_resolve_injection(criterion: &mut Criterion) {
    let mut benchmark_group = criterion.benchmark_group("rewrite_runtime/resolve_injection");
    let chunk_counts = [16usize, 64, 256];

    for chunk_count in chunk_counts {
        benchmark_group.bench_with_input(
            BenchmarkId::new("chunk_count", chunk_count),
            &chunk_count,
            |bencher, chunk_count| {
                bencher.iter(|| {
                    let (schedule, seed, position_table_index) = injection_schedule(*chunk_count);
                    let callbacks = InjectionCallbacks {
                        table_index: position_table_index,
                        chunk_count: *chunk_count,
                    };
                    let report = Executor::with_options(
                        Options::default()
                            .with_worker_count(non_zero_usize(4))
                            .with_record_trace(false),
                    )
                    .run(&schedule, &seed, &callbacks);

                    black_box(report);
                });
            },
        );
    }

    benchmark_group.finish();
}

struct NoopCallbacks;

impl Callbacks for NoopCallbacks {
    fn run_function(&self, _context: FunctionContext<'_>) {}
}

struct InjectionCallbacks {
    table_index: TableIndex,
    chunk_count: usize,
}

impl Callbacks for InjectionCallbacks {
    fn run_function(&self, _context: FunctionContext<'_>) {}

    fn run_resolve(&self, context: ResolveContext<'_>) -> Outcome {
        if context.resolve().function_index().value() != 0 {
            return Outcome::none();
        }

        Outcome::visible_chunks(
            (0..self.chunk_count)
                .map(|chunk_offset| {
                    VisibleChunk::new(
                        self.table_index,
                        ChunkIndex::new(
                            u32::try_from(chunk_offset)
                                .expect("benchmark chunk count should fit in u32"),
                        ),
                    )
                })
                .collect::<Vec<_>>(),
        )
    }
}

fn scan_schedule(chunk_count: usize) -> (that_bass::v2::schedule::Schedule, Seed) {
    let mut catalog = Catalog::new();
    let mut tables = vec![catalog
        .register_table([Meta::of::<Position>()], test_configuration())
        .expect("benchmark table registration should succeed")];
    populate_chunks(&mut tables[0], chunk_count);

    let mut builder = Builder::new(&mut catalog, &mut tables, test_configuration());
    builder.push_query(
        "scan",
        query::all(query::read::<Position>()).expect("benchmark query declaration should succeed"),
    );
    let schedule = builder.build();
    let seed = Seed::from_tables(&tables);

    (schedule, seed)
}

fn injection_schedule(chunk_count: usize) -> (that_bass::v2::schedule::Schedule, Seed, TableIndex) {
    let mut catalog = Catalog::new();
    let mut tables = vec![catalog
        .register_table([Meta::of::<Spawner>()], test_configuration())
        .expect("benchmark table registration should succeed")];
    populate_chunks(&mut tables[0], chunk_count);

    let mut builder = Builder::new(&mut catalog, &mut tables, test_configuration());
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
    let schedule = builder.build();
    let seed = Seed::from_tables(&tables);

    (schedule, seed, position_table_index)
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
