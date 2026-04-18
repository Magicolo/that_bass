use std::num::NonZeroUsize;
use that_bass::v2::{
    command, query, Callbacks, Configuration, Executor, FunctionContext, FunctionIndex, Options,
    Store,
};

#[repr(C)]
#[derive(Clone, Copy)]
struct Position {
    x: f32,
    y: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
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
    builder.push_query(
        "observe",
        query::all(query::read::<Position>()).expect("example query declaration should succeed"),
    );
    let position_table_index = builder
        .add_insert(spawn_index, command::insert::<(Position,)>())
        .expect("typed insert should resolve to one known table");
    let schedule = builder.build();

    let callbacks = DemoCallbacks { spawn_index };
    let report = Executor::with_options(Options::default().with_worker_count(non_zero_usize(2)))
        .run(&schedule, &mut store, &callbacks);

    let inserted_row_count = store
        .table(position_table_index)
        .expect("insert target table should stay addressable")
        .chunks()
        .iter()
        .map(|chunk| chunk.count())
        .sum::<usize>();

    println!("Command resolution");
    println!("  function count: {}", schedule.function_count());
    println!("  resolve count: {}", schedule.resolve_count());
    println!("  inserted rows: {inserted_row_count}");
    println!("  created jobs: {}", report.created_job_count());
    println!("  injected jobs: {}", report.injected_job_count());
}

struct DemoCallbacks {
    spawn_index: FunctionIndex,
}

impl Callbacks for DemoCallbacks {
    fn run_function(&self, mut context: FunctionContext<'_, '_>) {
        if context.function_index() == self.spawn_index {
            context
                .insert::<(Position,)>()
                .expect("spawn function should expose its typed insert buffer")
                .one((Position { x: 1.0, y: 2.0 },));
        }
    }
}

fn non_zero_usize(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("example constants must be non-zero")
}

fn test_configuration() -> Configuration {
    Configuration::default().with_target_chunk_byte_count(non_zero_usize(128))
}
