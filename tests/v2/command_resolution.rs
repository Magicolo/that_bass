use parking_lot::Mutex;
use std::num::NonZeroUsize;
use that_bass::v2::{
    Configuration, Store, command, query,
    runtime::{Callbacks, Executor, FunctionContext, Options, ResolveContext},
    schedule::Builder,
    schema::Meta,
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
fn inserts_are_not_self_visible_but_are_visible_to_later_functions_in_the_same_frame() {
    let mut store = Store::with_configuration(test_configuration());
    let table_index = store
        .register_table([Meta::of::<Position>()])
        .expect("table registration should succeed");
    let chunk_index = store
        .table_mut(table_index)
        .expect("registered table should stay addressable")
        .push_chunk();
    unsafe {
        store
            .table_mut(table_index)
            .expect("registered table should stay addressable")
            .write::<Position>(chunk_index, 0, Position { x: 1.0, y: 2.0 })
            .expect("direct write should succeed");
        store
            .table_mut(table_index)
            .expect("registered table should stay addressable")
            .assume_initialized_prefix(chunk_index, 1)
            .expect("initialized prefix declaration should succeed");
    }

    let mut builder = Builder::new(&mut store);
    let producer_index = builder.push_query(
        "producer",
        query::all(query::read::<Position>()).expect("query declaration should succeed"),
    );
    builder
        .add_insert(producer_index, command::Insert::<(Position,)>::new())
        .expect("typed insert should resolve to a known table");
    let observer_index = builder.push_query(
        "observer",
        query::all(query::read::<Position>()).expect("query declaration should succeed"),
    );
    let schedule = builder.build();

    let callbacks = CommandCallbacks::for_insert(producer_index, Position { x: 3.0, y: 4.0 });
    Executor::with_options(Options::default().with_worker_count(non_zero_usize(2)))
        .run(&schedule, &mut store, &callbacks);

    assert_eq!(
        callbacks.row_lengths_for(producer_index),
        vec![RecordedRows {
            before: 1,
            after: 1,
        }]
    );
    let observer_rows = callbacks.row_lengths_for(observer_index);
    assert_eq!(observer_rows.len(), 2);
    assert_eq!(
        observer_rows
            .into_iter()
            .map(|rows| rows.before)
            .sum::<usize>(),
        2
    );
    assert_eq!(callbacks.resolve_call_count(), 2);
    assert_eq!(
        store
            .table(table_index)
            .expect("registered table should stay addressable")
            .chunks()
            .iter()
            .map(|chunk| chunk.count())
            .sum::<usize>(),
        2
    );
}

#[test]
fn remove_resolution_updates_already_seeded_later_jobs() {
    let mut store = Store::with_configuration(test_configuration());
    let table_index = store
        .register_table([Meta::of::<Position>()])
        .expect("table registration should succeed");
    let chunk_index = store
        .table_mut(table_index)
        .expect("registered table should stay addressable")
        .push_chunk();
    unsafe {
        store
            .table_mut(table_index)
            .expect("registered table should stay addressable")
            .write::<Position>(chunk_index, 0, Position { x: 1.0, y: 2.0 })
            .expect("direct write should succeed");
    }
    unsafe {
        store
            .table_mut(table_index)
            .expect("registered table should stay addressable")
            .assume_initialized_prefix(chunk_index, 1)
            .expect("initialized prefix declaration should succeed");
    }

    let mut builder = Builder::new(&mut store);
    let remover_index = builder.push_query(
        "remover",
        query::all(query::rows()).expect("query declaration should succeed"),
    );
    builder
        .add_remove(
            remover_index,
            command::Remove::new(query::has::<Position>()),
        )
        .expect("remove planning should succeed");
    let observer_index = builder.push_query(
        "observer",
        query::all(query::read::<Position>()).expect("query declaration should succeed"),
    );
    let schedule = builder.build();
    let callbacks = CommandCallbacks::for_remove(remover_index);
    Executor::with_options(Options::default().with_worker_count(non_zero_usize(2)))
        .run(&schedule, &mut store, &callbacks);

    assert_eq!(
        callbacks.row_lengths_for(observer_index),
        vec![RecordedRows {
            before: 0,
            after: 0,
        }]
    );
    assert_eq!(
        store
            .table(table_index)
            .expect("registered table should stay addressable")
            .slice::<Position>(chunk_index)
            .expect("dense-prefix slice should succeed")
            .len(),
        0
    );
}

#[test]
fn one_function_with_many_job_buffers_still_resolves_once() {
    let mut store = Store::with_configuration(test_configuration());
    let spawner_table_index = store
        .register_table([Meta::of::<Spawner>()])
        .expect("table registration should succeed");
    for _ in 0..4 {
        store
            .table_mut(spawner_table_index)
            .expect("registered table should stay addressable")
            .push_chunk();
    }

    let mut builder = Builder::new(&mut store);
    let spawn_index = builder.push_query(
        "spawn",
        query::all(query::read::<Spawner>()).expect("query declaration should succeed"),
    );
    let position_table_index = builder
        .add_insert(spawn_index, command::Insert::<(Position,)>::new())
        .expect("typed insert should resolve to a known table");
    let schedule = builder.build();

    let callbacks = CommandCallbacks::for_insert(spawn_index, Position { x: 7.0, y: 8.0 });
    Executor::with_options(Options::default().with_worker_count(non_zero_usize(4)))
        .run(&schedule, &mut store, &callbacks);

    let inserted_row_count = store
        .table(position_table_index)
        .expect("insert target table should stay addressable")
        .chunks()
        .iter()
        .map(|chunk| chunk.count())
        .sum::<usize>();

    assert_eq!(callbacks.function_call_count_for(spawn_index), 4);
    assert_eq!(callbacks.resolve_call_count(), 1);
    assert_eq!(inserted_row_count, 4);
}

#[test]
fn later_resolve_waits_for_jobs_injected_into_a_function_with_no_seeded_work() {
    let mut store = Store::with_configuration(test_configuration());
    let spawner_table_index = store
        .register_table([Meta::of::<Spawner>()])
        .expect("table registration should succeed");
    store
        .table_mut(spawner_table_index)
        .expect("registered table should stay addressable")
        .push_chunk();

    let mut builder = Builder::new(&mut store);
    let spawn_index = builder.push_query(
        "spawn",
        query::all(query::read::<Spawner>()).expect("query declaration should succeed"),
    );
    builder
        .add_insert(spawn_index, command::Insert::<(Position,)>::new())
        .expect("typed insert should resolve to a known table");
    let replicate_index = builder.push_query(
        "replicate",
        query::all(query::read::<Position>()).expect("query declaration should succeed"),
    );
    let position_table_index = builder
        .add_insert(replicate_index, command::Insert::<(Position,)>::new())
        .expect("typed insert should resolve to the same known table");
    let schedule = builder.build();
    let callbacks = CommandCallbacks::for_two_stage_insert(
        spawn_index,
        Position { x: 1.0, y: 2.0 },
        replicate_index,
        Position { x: 3.0, y: 4.0 },
    );

    Executor::with_options(Options::default().with_worker_count(non_zero_usize(2)))
        .run(&schedule, &mut store, &callbacks);

    assert_eq!(callbacks.function_call_count_for(replicate_index), 1);
    assert_eq!(callbacks.resolve_call_count(), 2);
    assert_eq!(
        store
            .table(position_table_index)
            .expect("insert target table should stay addressable")
            .chunks()
            .iter()
            .map(|chunk| chunk.count())
            .sum::<usize>(),
        2
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RecordedRows {
    before: usize,
    after: usize,
}

struct CommandCallbacks {
    insert_on_function_index: Option<usize>,
    second_insert_on_function_index: Option<usize>,
    remove_on_function_index: Option<usize>,
    inserted_position: Option<Position>,
    second_inserted_position: Option<Position>,
    row_lengths_by_function: Mutex<Vec<(usize, RecordedRows)>>,
    resolve_call_count: Mutex<usize>,
}

impl CommandCallbacks {
    fn for_insert(
        function_index: that_bass::v2::schedule::FunctionIndex,
        position: Position,
    ) -> Self {
        Self {
            insert_on_function_index: Some(function_index.value()),
            second_insert_on_function_index: None,
            remove_on_function_index: None,
            inserted_position: Some(position),
            second_inserted_position: None,
            row_lengths_by_function: Mutex::new(Vec::new()),
            resolve_call_count: Mutex::new(0),
        }
    }

    fn for_two_stage_insert(
        first_function_index: that_bass::v2::schedule::FunctionIndex,
        first_position: Position,
        second_function_index: that_bass::v2::schedule::FunctionIndex,
        second_position: Position,
    ) -> Self {
        Self {
            insert_on_function_index: Some(first_function_index.value()),
            second_insert_on_function_index: Some(second_function_index.value()),
            remove_on_function_index: None,
            inserted_position: Some(first_position),
            second_inserted_position: Some(second_position),
            row_lengths_by_function: Mutex::new(Vec::new()),
            resolve_call_count: Mutex::new(0),
        }
    }

    fn for_remove(function_index: that_bass::v2::schedule::FunctionIndex) -> Self {
        Self {
            insert_on_function_index: None,
            second_insert_on_function_index: None,
            remove_on_function_index: Some(function_index.value()),
            inserted_position: None,
            second_inserted_position: None,
            row_lengths_by_function: Mutex::new(Vec::new()),
            resolve_call_count: Mutex::new(0),
        }
    }

    fn row_lengths_for(
        &self,
        function_index: that_bass::v2::schedule::FunctionIndex,
    ) -> Vec<RecordedRows> {
        self.row_lengths_by_function
            .lock()
            .iter()
            .filter(|(recorded_function_index, _)| {
                *recorded_function_index == function_index.value()
            })
            .map(|(_, rows)| *rows)
            .collect()
    }

    fn function_call_count_for(
        &self,
        function_index: that_bass::v2::schedule::FunctionIndex,
    ) -> usize {
        self.row_lengths_for(function_index).len()
    }

    fn resolve_call_count(&self) -> usize {
        *self.resolve_call_count.lock()
    }
}

impl Callbacks for CommandCallbacks {
    fn run_function(&self, mut context: FunctionContext<'_, '_>) {
        let before = context.rows().len();

        if self.insert_on_function_index == Some(context.function_index().value()) {
            context
                .insert::<(Position,)>()
                .expect("insert-producing function should expose its typed insert buffer")
                .one((self
                    .inserted_position
                    .expect("inserted position should be configured"),));
        }

        if self.second_insert_on_function_index == Some(context.function_index().value()) {
            context
                .insert::<(Position,)>()
                .expect("second insert-producing function should expose its typed insert buffer")
                .one((self
                    .second_inserted_position
                    .expect("second inserted position should be configured"),));
        }

        if self.remove_on_function_index == Some(context.function_index().value()) {
            let rows = context.rows();
            context
                .remove()
                .expect("remove-producing function should expose its remove buffer")
                .extend(rows);
        }

        let after = context.rows().len();
        self.row_lengths_by_function.lock().push((
            context.function_index().value(),
            RecordedRows { before, after },
        ));
    }

    fn run_resolve(&self, _context: ResolveContext<'_>) {
        *self.resolve_call_count.lock() += 1;
    }
}

fn non_zero_usize(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("test constants must be non-zero")
}

fn test_configuration() -> Configuration {
    Configuration::default().with_target_chunk_byte_count(non_zero_usize(128))
}
