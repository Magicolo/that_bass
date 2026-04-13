use std::num::NonZeroUsize;
use that_bass::v2::{
    command, query,
    runtime::{Callbacks, Executor, FunctionContext, Options, ResolveContext},
    schedule::Builder,
    schema::Meta,
    Configuration, Store,
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
    let spawner_table_index = store
        .register_table([Meta::of::<Spawner>()])
        .expect("example table registration should succeed");
    store
        .table_mut(spawner_table_index)
        .expect("registered table should stay addressable")
        .push_chunk();

    let mut builder = Builder::new(&mut store);
    let spawn_index = builder.push_query(
        "spawn",
        query::all(query::read::<Spawner>()).expect("example query declaration should succeed"),
    );
    let observe_index = builder.push_query(
        "observe",
        query::all(query::read::<Position>()).expect("example query declaration should succeed"),
    );
    let position_table_index = builder
        .add_insert(spawn_index, command::Insert::<(Position,)>::new())
        .expect("typed insert should resolve to one known table");
    let schedule = builder.build();

    let callbacks = DemoCallbacks {
        spawn_index,
        observe_index,
    };
    let report = Executor::with_options(
        Options::default()
            .with_worker_count(non_zero_usize(2))
            .with_record_trace(true),
    )
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
    println!("  trace: {:?}", report.trace());
}

struct DemoCallbacks {
    spawn_index: that_bass::v2::schedule::FunctionIndex,
    observe_index: that_bass::v2::schedule::FunctionIndex,
}

impl Callbacks for DemoCallbacks {
    fn run_function(&self, mut context: FunctionContext<'_, '_>) {
        println!(
            "  function {:?} chunk {:?} sees {} rows",
            context.function_index(),
            context.chunk_index(),
            context.rows().len(),
        );

        if context.function_index() == self.spawn_index {
            context
                .insert::<(Position,)>()
                .expect("spawn function should expose its typed insert buffer")
                .one((Position { x: 1.0, y: 2.0 },));
        }

        if context.function_index() == self.observe_index {
            println!("  observer rows: {}", context.rows().len());
        }
    }

    fn run_resolve(&self, context: ResolveContext<'_>) {
        println!(
            "  resolve {:?} on worker {}",
            context.resolve_index(),
            context.worker_index(),
        );
    }
}

fn non_zero_usize(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("example constants must be non-zero")
}

fn test_configuration() -> Configuration {
    Configuration::default().with_target_chunk_byte_count(non_zero_usize(128))
}
