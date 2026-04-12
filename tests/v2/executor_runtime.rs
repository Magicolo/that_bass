use checkito::Check;
use parking_lot::Mutex;
use std::{num::NonZeroUsize, thread, time::Duration};
use that_bass::v2::{
    command, query,
    runtime::{
        Callbacks, Executor, FunctionContext, Options, Outcome, ResolveContext, Seed, VisibleChunk,
    },
    schedule::Builder,
    schema::{Catalog, ChunkIndex, Meta, Table, TableIndex},
    Configuration,
};

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Spawner {
    count: u32,
}

#[test]
fn executor_runs_all_seeded_chunk_jobs_and_resolves() -> Result<(), String> {
    let chunk_count_generator = 1usize..17usize;
    let worker_count_generator = 1usize..5usize;

    (&chunk_count_generator, &worker_count_generator)
        .check(|(chunk_count, worker_count)| {
            let mut catalog = Catalog::new();
            let mut tables = vec![catalog
                .register_table([Meta::of::<Position>()], test_configuration())
                .expect("table registration should succeed")];
            populate_chunks(&mut tables[0], chunk_count);

            let mut builder = Builder::new(&mut catalog, &mut tables, test_configuration());
            builder.push_query(
                "scan",
                query::all(query::read::<Position>()).expect("query declaration should succeed"),
            );
            let schedule = builder.build();
            let seed = Seed::from_tables(&tables);
            let callbacks = Recorder::new(Duration::ZERO);
            let report = Executor::with_options(
                Options::default().with_worker_count(non_zero_usize(worker_count)),
            )
            .run(&schedule, &seed, &callbacks);

            assert_eq!(callbacks.function_call_count(), chunk_count);
            assert_eq!(callbacks.resolve_call_count(), 1);
            assert_eq!(report.function_job_count(), chunk_count);
            assert_eq!(report.resolve_job_count(), 1);
            assert_eq!(report.completed_job_count(), chunk_count + 1);
        })
        .map_or(Ok(()), |failure| Err(format!("{failure:?}")))
}

#[test]
fn executor_steals_individually_seeded_chunk_jobs() {
    if cfg!(miri) {
        return;
    }

    let chunk_count = 32usize;
    let worker_count = 2usize;
    let mut catalog = Catalog::new();
    let mut tables = vec![catalog
        .register_table([Meta::of::<Position>()], test_configuration())
        .expect("table registration should succeed")];
    populate_chunks(&mut tables[0], chunk_count);

    let mut builder = Builder::new(&mut catalog, &mut tables, test_configuration());
    builder.push_query(
        "scan",
        query::all(query::read::<Position>()).expect("query declaration should succeed"),
    );
    let schedule = builder.build();
    let seed = Seed::from_tables(&tables);
    let callbacks = Recorder::new(Duration::from_millis(1));
    let report = Executor::with_options(
        Options::default()
            .with_worker_count(non_zero_usize(worker_count))
            .with_record_trace(true),
    )
    .run(&schedule, &seed, &callbacks);

    assert!(report.steal_count() > 0);
    assert!(
        report
            .worker_execution_counts()
            .iter()
            .filter(|execution_count| **execution_count > 0)
            .count()
            > 1
    );
}

#[test]
fn resolve_can_inject_new_chunk_jobs_for_later_functions_in_the_same_frame() {
    let mut catalog = Catalog::new();
    let mut tables = vec![catalog
        .register_table([Meta::of::<Spawner>()], test_configuration())
        .expect("table registration should succeed")];
    populate_chunks(&mut tables[0], 1);

    let mut builder = Builder::new(&mut catalog, &mut tables, test_configuration());
    let spawn_index = builder.push_query(
        "spawn",
        query::all(query::read::<Spawner>()).expect("query declaration should succeed"),
    );
    let clamp_index = builder.push_query(
        "clamp",
        query::all(query::read::<Position>()).expect("query declaration should succeed"),
    );
    let position_table_index = builder
        .add_insert(spawn_index, command::Insert::<(Position,)>::new())
        .expect("typed insert should resolve to one known table");
    let schedule = builder.build();
    let seed = Seed::from_tables(&tables);
    let callbacks =
        Recorder::with_injected_chunks(Duration::ZERO, position_table_index, [ChunkIndex::new(0)]);
    let report = Executor::with_options(
        Options::default()
            .with_worker_count(non_zero_usize(2))
            .with_record_trace(true),
    )
    .run(&schedule, &seed, &callbacks);

    let clamp_calls = callbacks.function_calls_for(clamp_index);

    assert_eq!(
        schedule
            .function(clamp_index)
            .expect("clamp function should exist")
            .known_tables(),
        &[position_table_index]
    );
    assert_eq!(
        clamp_calls,
        vec![(position_table_index, ChunkIndex::new(0))]
    );
    assert_eq!(report.injected_job_count(), 1);
    assert!(report.trace().iter().any(|trace| {
        matches!(
            trace.kind(),
            that_bass::v2::runtime::TraceKind::Function {
                function_index,
                table_index,
                chunk_index,
            } if function_index == clamp_index
                && table_index == position_table_index
                && chunk_index == ChunkIndex::new(0)
        )
    }));
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RecordedFunctionCall {
    function_index: usize,
    table_index: TableIndex,
    chunk_index: ChunkIndex,
}

struct Recorder {
    sleep_duration: Duration,
    injected_table_index: Option<TableIndex>,
    injected_chunk_indices: Vec<ChunkIndex>,
    function_calls: Mutex<Vec<RecordedFunctionCall>>,
    resolve_calls: Mutex<usize>,
}

impl Recorder {
    fn new(sleep_duration: Duration) -> Self {
        Self {
            sleep_duration,
            injected_table_index: None,
            injected_chunk_indices: Vec::new(),
            function_calls: Mutex::new(Vec::new()),
            resolve_calls: Mutex::new(0),
        }
    }

    fn with_injected_chunks(
        sleep_duration: Duration,
        table_index: TableIndex,
        chunk_indices: impl IntoIterator<Item = ChunkIndex>,
    ) -> Self {
        Self {
            sleep_duration,
            injected_table_index: Some(table_index),
            injected_chunk_indices: chunk_indices.into_iter().collect(),
            function_calls: Mutex::new(Vec::new()),
            resolve_calls: Mutex::new(0),
        }
    }

    fn function_call_count(&self) -> usize {
        self.function_calls.lock().len()
    }

    fn resolve_call_count(&self) -> usize {
        *self.resolve_calls.lock()
    }

    fn function_calls_for(
        &self,
        function_index: that_bass::v2::schedule::FunctionIndex,
    ) -> Vec<(TableIndex, ChunkIndex)> {
        self.function_calls
            .lock()
            .iter()
            .filter(|call| call.function_index == function_index.value())
            .map(|call| (call.table_index, call.chunk_index))
            .collect()
    }
}

impl Callbacks for Recorder {
    fn run_function(&self, context: FunctionContext<'_>) {
        if !self.sleep_duration.is_zero() {
            thread::sleep(self.sleep_duration);
        }

        self.function_calls.lock().push(RecordedFunctionCall {
            function_index: context.function_index().value(),
            table_index: context.table_index(),
            chunk_index: context.chunk_index(),
        });
    }

    fn run_resolve(&self, _context: ResolveContext<'_>) -> Outcome {
        *self.resolve_calls.lock() += 1;

        let Some(table_index) = self.injected_table_index else {
            return Outcome::none();
        };

        Outcome::visible_chunks(
            self.injected_chunk_indices
                .iter()
                .copied()
                .map(|chunk_index| VisibleChunk::new(table_index, chunk_index))
                .collect::<Vec<_>>(),
        )
    }
}

fn populate_chunks(table: &mut Table, chunk_count: usize) {
    for _ in 0..chunk_count {
        table.push_chunk();
    }
}

fn non_zero_usize(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("test worker count must be non-zero")
}

fn test_configuration() -> Configuration {
    Configuration::default()
        .with_target_chunk_byte_count(NonZeroUsize::new(128).expect("constant must be non-zero"))
}
