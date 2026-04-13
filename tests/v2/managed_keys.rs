use checkito::Check;
use parking_lot::Mutex;
use std::{
    num::NonZeroUsize,
    sync::atomic::{AtomicBool, Ordering},
};
use that_bass::v2::{
    command,
    key::{self, Entry},
    query,
    runtime::{Callbacks, Executor, FunctionContext, Options},
    schedule::{Builder, FunctionIndex},
    schema::{Meta, Row, Table},
    Configuration, Store,
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
fn reserved_keys_are_observable_before_publication() -> Result<(), String> {
    if cfg!(miri) {
        assert_reserved_keys_are_observable(4);
        return Ok(());
    }

    let reserved_key_count_generator = 1usize..17usize;
    reserved_key_count_generator
        .check(assert_reserved_keys_are_observable)
        .map_or(Ok(()), |failure| Err(format!("{failure:?}")))
}

#[test]
fn keyed_insert_publish_makes_rows_visible_to_later_lookup_paths() {
    let mut store = Store::with_configuration(test_configuration());
    let spawner_table_index = store
        .register_table([Meta::of::<Spawner>()])
        .expect("table registration should succeed");
    write_spawner_rows(
        store
            .table_mut(spawner_table_index)
            .expect("registered table should stay addressable"),
        1,
    );

    let mut builder = Builder::new(&mut store);
    let producer_index = builder.push_query(
        "spawn",
        query::all(query::read::<Spawner>()).expect("query declaration should succeed"),
    );
    builder
        .add_keys(producer_index)
        .expect("key injection should succeed");
    let position_table_index = builder
        .add_insert(
            producer_index,
            command::Insert::<(key::Key, Position)>::new(),
        )
        .expect("keyed insert planning should succeed");
    let observer_index = builder.push_query(
        "observe",
        query::all(query::read::<Position>()).expect("query declaration should succeed"),
    );
    builder
        .add_keys(observer_index)
        .expect("key injection should succeed");
    let schedule = builder.build();

    let callbacks = KeyedInsertCallbacks::new(producer_index, observer_index);
    Executor::with_options(Options::default().with_worker_count(runtime_worker_count()))
        .run(&schedule, &mut store, &callbacks);

    let inserted_key = callbacks.inserted_key();
    let observed_row = callbacks.observed_row();
    let keys = store
        .keys()
        .expect("explicit key injection should initialize the Keys resource");
    let live_row = keys
        .get(inserted_key)
        .expect("published key should be live");
    let position_query =
        query::all(query::read::<Position>()).expect("query declaration should succeed");
    let keyed_query = query::all((query::read::<key::Key>(), query::read::<Position>()))
        .expect("query declaration should succeed");

    assert_eq!(observed_row, live_row);
    assert_eq!(observed_row.table_index(), position_table_index);
    assert!(matches!(keys.state(inserted_key), Ok(Entry::Live(row)) if row == live_row));
    assert_eq!(
        *position_query
            .get(&store, &keys, inserted_key)
            .expect("keyed position lookup should succeed"),
        Position { x: 1.0, y: 2.0 }
    );
    let (stored_key, stored_position) = keyed_query
        .get(&store, &keys, inserted_key)
        .expect("keyed tuple lookup should succeed");
    assert_eq!(*stored_key, inserted_key);
    assert_eq!(*stored_position, Position { x: 1.0, y: 2.0 });
}

#[test]
fn keyed_remove_releases_deleted_keys_and_republishes_moved_rows() {
    let mut store = Store::with_configuration(test_configuration());
    let spawner_table_index = store
        .register_table([Meta::of::<Spawner>()])
        .expect("table registration should succeed");
    write_spawner_rows(
        store
            .table_mut(spawner_table_index)
            .expect("registered table should stay addressable"),
        1,
    );

    let mut seeding_builder = Builder::new(&mut store);
    let seed_index = seeding_builder.push_query(
        "seed-keyed",
        query::all(query::read::<Spawner>()).expect("query declaration should succeed"),
    );
    seeding_builder
        .add_keys(seed_index)
        .expect("key injection should succeed");
    let keyed_table_index = seeding_builder
        .add_insert(seed_index, command::Insert::<(key::Key, Position)>::new())
        .expect("keyed insert planning should succeed");
    let seeding_schedule = seeding_builder.build();
    let seeding_callbacks = SeedKeyedRowsCallbacks::new(seed_index);
    Executor::with_options(Options::default().with_worker_count(runtime_worker_count())).run(
        &seeding_schedule,
        &mut store,
        &seeding_callbacks,
    );

    let [first_key, removed_key, moved_key] = seeding_callbacks.inserted_keys();

    let mut builder = Builder::new(&mut store);
    let remover_index = builder.push_query(
        "remove-middle",
        query::all(query::rows()).expect("query declaration should succeed"),
    );
    builder
        .add_remove(
            remover_index,
            command::Remove::new(query::has::<Position>()),
        )
        .expect("remove planning should succeed");
    let observer_index = builder.push_query(
        "observe",
        query::all(query::read::<Position>()).expect("query declaration should succeed"),
    );
    builder
        .add_keys(observer_index)
        .expect("key injection should succeed");
    let schedule = builder.build();

    let callbacks =
        KeyedRemoveCallbacks::new(remover_index, observer_index, removed_key, moved_key);
    Executor::with_options(Options::default().with_worker_count(runtime_worker_count()))
        .run(&schedule, &mut store, &callbacks);

    let observed_moved_row = callbacks
        .observed_moved_row()
        .expect("observer should see the moved key");
    let keys = store
        .keys()
        .expect("explicit key injection should initialize the Keys resource");
    let live_moved_row = keys.get(moved_key).expect("moved key should stay live");
    let position_query =
        query::all(query::read::<Position>()).expect("query declaration should succeed");

    assert!(matches!(
        keys.get(removed_key),
        Err(key::Error::InvalidKey { key }) if key == removed_key
    ));
    assert_eq!(observed_moved_row, live_moved_row);
    assert_eq!(
        store
            .table(keyed_table_index)
            .expect("registered table should stay addressable")
            .row_layout()
            .row_index(live_moved_row),
        0
    );
    assert_eq!(
        *position_query
            .get(&store, &keys, moved_key)
            .expect("moved key should still resolve to its position"),
        Position { x: 3.0, y: 3.0 }
    );
    assert_eq!(
        *position_query
            .get(&store, &keys, first_key)
            .expect("first key should still resolve cleanly"),
        Position { x: 1.0, y: 1.0 }
    );
}

fn assert_reserved_keys_are_observable(reserved_key_count: usize) {
    let mut store = Store::new();
    let keys = store.initialize_keys();
    let reserved_keys = (0..reserved_key_count)
        .map(|_| keys.reserve())
        .collect::<Vec<_>>();

    assert_eq!(keys.slot_count(), reserved_key_count);
    for reserved_key in reserved_keys {
        assert_eq!(
            keys.state(reserved_key),
            Ok(Entry::Reserved),
            "reserved keys should stay observable before publication"
        );
        assert!(matches!(
            keys.get(reserved_key),
            Err(key::Error::UnexpectedState { key, state: key::StateKind::Reserved })
                if key == reserved_key
        ));
    }
}

struct KeyedInsertCallbacks {
    producer_index: FunctionIndex,
    observer_index: FunctionIndex,
    inserted_keys: Mutex<Vec<key::Key>>,
    observed_rows: Mutex<Vec<Row<'static>>>,
}

impl KeyedInsertCallbacks {
    fn new(producer_index: FunctionIndex, observer_index: FunctionIndex) -> Self {
        Self {
            producer_index,
            observer_index,
            inserted_keys: Mutex::new(Vec::new()),
            observed_rows: Mutex::new(Vec::new()),
        }
    }

    fn inserted_key(&self) -> key::Key {
        *self
            .inserted_keys
            .lock()
            .first()
            .expect("producer should reserve exactly one key")
    }

    fn observed_row(&self) -> Row<'static> {
        *self
            .observed_rows
            .lock()
            .first()
            .expect("observer should see exactly one published row")
    }
}

impl Callbacks for KeyedInsertCallbacks {
    fn run_function(&self, mut context: FunctionContext<'_, '_>) {
        if context.function_index() == self.producer_index {
            let key = context
                .keys()
                .expect("producer should receive Keys")
                .reserve();
            context
                .insert::<(key::Key, Position)>()
                .expect("producer should expose its keyed insert buffer")
                .one((key, Position { x: 1.0, y: 2.0 }));
            self.inserted_keys.lock().push(key);
        }

        if context.function_index() == self.observer_index {
            let inserted_key = self.inserted_key();
            let observed_row = context
                .keys()
                .expect("observer should receive Keys")
                .get(inserted_key)
                .expect("later lookup should see the published row");
            self.observed_rows.lock().push(observed_row);
        }
    }
}

struct SeedKeyedRowsCallbacks {
    function_index: FunctionIndex,
    inserted_keys: Mutex<Vec<key::Key>>,
}

impl SeedKeyedRowsCallbacks {
    fn new(function_index: FunctionIndex) -> Self {
        Self {
            function_index,
            inserted_keys: Mutex::new(Vec::new()),
        }
    }

    fn inserted_keys(&self) -> [key::Key; 3] {
        self.inserted_keys
            .lock()
            .clone()
            .try_into()
            .expect("seeding should insert exactly three keyed rows")
    }
}

impl Callbacks for SeedKeyedRowsCallbacks {
    fn run_function(&self, mut context: FunctionContext<'_, '_>) {
        if context.function_index() != self.function_index {
            return;
        }

        let keys = context.keys().expect("seed function should receive Keys");
        let first_key = keys.reserve();
        let removed_key = keys.reserve();
        let moved_key = keys.reserve();
        context
            .insert::<(key::Key, Position)>()
            .expect("seed function should expose its keyed insert buffer")
            .array([
                (first_key, Position { x: 1.0, y: 1.0 }),
                (removed_key, Position { x: 2.0, y: 2.0 }),
                (moved_key, Position { x: 3.0, y: 3.0 }),
            ]);
        self.inserted_keys
            .lock()
            .extend([first_key, removed_key, moved_key]);
    }
}

struct KeyedRemoveCallbacks {
    remover_index: FunctionIndex,
    observer_index: FunctionIndex,
    removed_key: key::Key,
    moved_key: key::Key,
    remove_recorded: AtomicBool,
    observed_moved_rows: Mutex<Vec<Row<'static>>>,
}

impl KeyedRemoveCallbacks {
    fn new(
        remover_index: FunctionIndex,
        observer_index: FunctionIndex,
        removed_key: key::Key,
        moved_key: key::Key,
    ) -> Self {
        Self {
            remover_index,
            observer_index,
            removed_key,
            moved_key,
            remove_recorded: AtomicBool::new(false),
            observed_moved_rows: Mutex::new(Vec::new()),
        }
    }

    fn observed_moved_row(&self) -> Option<Row<'static>> {
        self.observed_moved_rows.lock().first().copied()
    }
}

impl Callbacks for KeyedRemoveCallbacks {
    fn run_function(&self, mut context: FunctionContext<'_, '_>) {
        if context.function_index() == self.remover_index {
            if context.rows().len() < 2 || self.remove_recorded.swap(true, Ordering::AcqRel) {
                return;
            }
            let row = context
                .rows()
                .get(0)
                .expect("the targeted keyed row should still exist when remove is recorded");
            context
                .remove()
                .expect("remove-capable function should expose a remove buffer")
                .one(row);
        }

        if context.function_index() == self.observer_index {
            let keys = context.keys().expect("observer should receive Keys");
            assert!(matches!(
                keys.get(self.removed_key),
                Err(key::Error::InvalidKey { key }) if key == self.removed_key
            ));
            let moved_row = keys
                .get(self.moved_key)
                .expect("moved key should already point at its new row");
            self.observed_moved_rows.lock().push(moved_row);
        }
    }
}

fn write_spawner_rows(table: &mut Table, row_count: usize) {
    let mut chunk_index = table.push_chunk();
    for _row_index in 0..row_count {
        let chunk_is_full = table
            .chunk(chunk_index)
            .expect("seed chunk should stay addressable")
            .is_full();
        if chunk_is_full {
            chunk_index = table.push_chunk();
        }
        let row_index_in_chunk = table
            .chunk(chunk_index)
            .expect("seed chunk should stay addressable")
            .count();
        unsafe {
            table
                .write::<Spawner>(chunk_index, row_index_in_chunk, Spawner { count: 1 })
                .expect("direct write should succeed");
            table
                .assume_initialized_prefix(chunk_index, row_index_in_chunk + 1)
                .expect("initialized prefix declaration should succeed");
        }
    }
}

fn non_zero_usize(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("test constants must be non-zero")
}

fn runtime_worker_count() -> NonZeroUsize {
    if cfg!(miri) {
        non_zero_usize(1)
    } else {
        non_zero_usize(2)
    }
}

fn test_configuration() -> Configuration {
    Configuration::default().with_target_chunk_byte_count(non_zero_usize(128))
}
