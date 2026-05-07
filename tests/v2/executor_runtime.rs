use checkito::Check;
use parking_lot::Mutex;
use std::{num::NonZeroUsize, thread, time::Duration};
use that_bass::v2::{
    Configuration, Store, command, query,
    runtime::{Callbacks, Executor, FunctionContext, Options, ResolveContext},
    schedule::Builder,
    schema::{ChunkIndex, Meta, Table, TableIndex},
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
    if cfg!(miri) {
        for (chunk_count, worker_count) in [(1usize, 1usize), (4, 2), (8, 3)] {
            assert_seeded_chunk_execution(chunk_count, worker_count);
        }

        return Ok(());
    }

    let chunk_count_generator = 1usize..17usize;
    let worker_count_generator = 1usize..5usize;

    (&chunk_count_generator, &worker_count_generator)
        .check(|(chunk_count, worker_count)| {
            assert_seeded_chunk_execution(chunk_count, worker_count)
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
    let mut store = Store::with_configuration(test_configuration());
    let table_index = store
        .register_table([Meta::of::<Position>()])
        .expect("table registration should succeed");
    populate_chunks(
        store
            .table_mut(table_index)
            .expect("registered table should stay addressable"),
        chunk_count,
    );

    let mut builder = Builder::new(&mut store);
    builder.push_query(
        "scan",
        query::all(query::read::<Position>()).expect("query declaration should succeed"),
    );
    let schedule = builder.build();
    let callbacks = Recorder::new(Duration::from_millis(1));
    let report = Executor::with_options(
        Options::default()
            .with_worker_count(non_zero_usize(worker_count))
            .with_record_trace(true),
    )
    .run(&schedule, &mut store, &callbacks);

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
    let mut store = Store::with_configuration(test_configuration());
    let spawner_table_index = store
        .register_table([Meta::of::<Spawner>()])
        .expect("table registration should succeed");
    populate_chunks(
        store
            .table_mut(spawner_table_index)
            .expect("registered table should stay addressable"),
        1,
    );

    let mut builder = Builder::new(&mut store);
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
    let callbacks =
        Recorder::with_inserted_position(Duration::ZERO, spawn_index, Position { x: 1.0, y: 2.0 });
    let report = Executor::with_options(
        Options::default()
            .with_worker_count(non_zero_usize(2))
            .with_record_trace(true),
    )
    .run(&schedule, &mut store, &callbacks);

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
    insert_on_function_index: Option<usize>,
    inserted_position: Option<Position>,
    function_calls: Mutex<Vec<RecordedFunctionCall>>,
    resolve_calls: Mutex<usize>,
}

impl Recorder {
    fn new(sleep_duration: Duration) -> Self {
        Self {
            sleep_duration,
            insert_on_function_index: None,
            inserted_position: None,
            function_calls: Mutex::new(Vec::new()),
            resolve_calls: Mutex::new(0),
        }
    }

    fn with_inserted_position(
        sleep_duration: Duration,
        function_index: that_bass::v2::schedule::FunctionIndex,
        position: Position,
    ) -> Self {
        Self {
            sleep_duration,
            insert_on_function_index: Some(function_index.value()),
            inserted_position: Some(position),
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
    fn run_function(&self, mut context: FunctionContext<'_, '_>) {
        if !self.sleep_duration.is_zero() {
            thread::sleep(self.sleep_duration);
        }

        if self.insert_on_function_index == Some(context.function_index().value()) {
            context
                .insert::<(Position,)>()
                .expect("spawn function should expose its typed insert buffer")
                .one((self
                    .inserted_position
                    .expect("inserted position should be configured"),));
        }

        self.function_calls.lock().push(RecordedFunctionCall {
            function_index: context.function_index().value(),
            table_index: context.table_index(),
            chunk_index: context.chunk_index(),
        });
    }

    fn run_resolve(&self, _context: ResolveContext<'_>) {
        *self.resolve_calls.lock() += 1;
    }
}

fn populate_chunks(table: &mut Table, chunk_count: usize) {
    for _ in 0..chunk_count {
        table.push_chunk();
    }
}

fn assert_seeded_chunk_execution(chunk_count: usize, worker_count: usize) {
    let mut store = Store::with_configuration(test_configuration());
    let table_index = store
        .register_table([Meta::of::<Position>()])
        .expect("table registration should succeed");
    populate_chunks(
        store
            .table_mut(table_index)
            .expect("registered table should stay addressable"),
        chunk_count,
    );

    let mut builder = Builder::new(&mut store);
    builder.push_query(
        "scan",
        query::all(query::read::<Position>()).expect("query declaration should succeed"),
    );
    let schedule = builder.build();
    let callbacks = Recorder::new(Duration::ZERO);
    let report =
        Executor::with_options(Options::default().with_worker_count(non_zero_usize(worker_count)))
            .run(&schedule, &mut store, &callbacks);

    assert_eq!(callbacks.function_call_count(), chunk_count);
    assert_eq!(callbacks.resolve_call_count(), 1);
    assert_eq!(report.function_job_count(), chunk_count);
    assert_eq!(report.resolve_job_count(), 1);
    assert_eq!(report.completed_job_count(), chunk_count + 1);
}

fn non_zero_usize(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("test worker count must be non-zero")
}

fn test_configuration() -> Configuration {
    Configuration::default()
        .with_target_chunk_byte_count(NonZeroUsize::new(128).expect("constant must be non-zero"))
}
