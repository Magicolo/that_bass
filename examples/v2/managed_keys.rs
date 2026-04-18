use parking_lot::Mutex;
use std::num::NonZeroUsize;
use that_bass::v2::{
    command, query, Callbacks, Configuration, Executor, FunctionContext, Key, Options, Store,
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
    store
        .initialize_global(Spawner { count: 1 })
        .expect("global initialization should succeed");
    let mut builder = store.builder();
    let producer_index = builder
        .push("spawn-keyed", query::one::<Spawner>())
        .expect("singleton input should be valid");
    builder
        .add_keys(producer_index)
        .expect("key injection should succeed");
    builder
        .add_insert(producer_index, command::insert::<(Key, Position)>())
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
    reserved_key: Mutex<Option<Key>>,
}

impl Callbacks for DemoCallbacks {
    fn run_function(&self, mut context: FunctionContext<'_, '_>) {
        let keys = context.keys().expect("producer should receive Keys");
        let reserved_key = keys.reserve();
        context
            .insert::<(Key, Position)>()
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
