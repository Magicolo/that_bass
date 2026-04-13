use parking_lot::Mutex;
use std::num::NonZeroUsize;
use that_bass::v2::{
    command, key, query,
    runtime::{Callbacks, Executor, FunctionContext, Options},
    schedule::Builder,
    schema::Meta,
    Configuration, Store,
};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Position {
    x: f32,
    y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Spawner {
    count: u32,
}

pub fn run() {
    let mut store = Store::with_configuration(test_configuration());
    let spawner_table_index = store
        .register_table([Meta::of::<Spawner>()])
        .expect("example table registration should succeed");
    let chunk_index = store
        .table_mut(spawner_table_index)
        .expect("registered table should stay addressable")
        .push_chunk();
    unsafe {
        store
            .table_mut(spawner_table_index)
            .expect("registered table should stay addressable")
            .write::<Spawner>(chunk_index, 0, Spawner { count: 1 })
            .expect("direct write should succeed");
        store
            .table_mut(spawner_table_index)
            .expect("registered table should stay addressable")
            .assume_initialized_prefix(chunk_index, 1)
            .expect("initialized prefix declaration should succeed");
    }

    let mut builder = Builder::new(&mut store);
    let producer_index = builder.push_query(
        "spawn-keyed",
        query::all(query::read::<Spawner>()).expect("example query declaration should succeed"),
    );
    builder
        .add_keys(producer_index)
        .expect("key injection should succeed");
    builder
        .add_insert(
            producer_index,
            command::Insert::<(key::Key, Position)>::new(),
        )
        .expect("keyed insert planning should succeed");
    let schedule = builder.build();

    let callbacks = DemoCallbacks {
        reserved_key: None.into(),
    };
    Executor::with_options(Options::default().with_worker_count(non_zero_usize(2)))
        .run(&schedule, &mut store, &callbacks);

    let reserved_key = callbacks
        .reserved_key
        .into_inner()
        .expect("producer should reserve one key");
    let keys = store
        .keys()
        .expect("key injection should initialize the Keys resource");
    let position_query =
        query::all(query::read::<Position>()).expect("example query declaration should succeed");
    let position = position_query
        .get(&store, &keys, reserved_key)
        .expect("keyed lookup should succeed");

    println!("Managed keys");
    println!("  reserved key: {:?}", reserved_key);
    println!("  lookup result: {:?}", position);
}

struct DemoCallbacks {
    reserved_key: Mutex<Option<key::Key>>,
}

impl Callbacks for DemoCallbacks {
    fn run_function(&self, mut context: FunctionContext<'_, '_>) {
        let keys = context.keys().expect("producer should receive Keys");
        let reserved_key = keys.reserve();
        context
            .insert::<(key::Key, Position)>()
            .expect("producer should expose its keyed insert buffer")
            .one((reserved_key, Position { x: 1.0, y: 2.0 }));
        *self.reserved_key.lock() = Some(reserved_key);
    }
}

fn non_zero_usize(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("example constants must be non-zero")
}

fn test_configuration() -> Configuration {
    Configuration::default().with_target_chunk_byte_count(non_zero_usize(128))
}
