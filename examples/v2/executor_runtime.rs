use std::num::NonZeroUsize;
use that_bass::v2::{
    command, query, Callbacks, Configuration, Executor, FunctionContext, FunctionIndex, Options,
    Store,
};

#[derive(Clone, Copy)]
#[repr(C)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Spawner {
    count: u32,
}

pub fn run() {
    let mut store = Store::with_configuration(test_configuration());
    store
        .initialize_global(Spawner { count: 1 })
        .expect("global initialization should succeed");
    let mut builder = store.builder();
    let spawn_index = builder
        .push("spawn", query::one::<Spawner>())
        .expect("singleton input should be valid");
    let clamp_index = builder.push_query(
        "clamp",
        query::all(query::read::<Position>()).expect("example query declaration should succeed"),
    );
    builder
        .add_insert(spawn_index, command::insert::<(Position,)>())
        .expect("typed insert should resolve to one known table");
    let schedule = builder.build();
    let callbacks = DemoCallbacks {
        spawn_index,
        inserted_position: Position { x: 1.0, y: 2.0 },
    };
    let report = Executor::with_options(Options::default().with_worker_count(non_zero_usize(2)))
        .run(&schedule, &mut store, &callbacks);

    println!("Executor runtime");
    println!("  function count: {}", schedule.function_count());
    println!("  created job count: {}", report.created_job_count());
    println!("  injected job count: {}", report.injected_job_count());
    println!("  steal count: {}", report.steal_count());
    println!("  clamp function index: {}", clamp_index.value());
}

struct DemoCallbacks {
    spawn_index: FunctionIndex,
    inserted_position: Position,
}

impl Callbacks for DemoCallbacks {
    fn run_function(&self, mut context: FunctionContext<'_, '_>) {
        if context.function_index() == self.spawn_index {
            context
                .insert::<(Position,)>()
                .expect("spawn function should expose its typed insert buffer")
                .one((self.inserted_position,));
        }
    }
}

fn non_zero_usize(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("example constants must be non-zero")
}

fn test_configuration() -> Configuration {
    Configuration::default()
        .with_target_chunk_byte_count(NonZeroUsize::new(128).expect("constant must be non-zero"))
}
