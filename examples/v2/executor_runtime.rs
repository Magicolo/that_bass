use std::num::NonZeroUsize;
use that_bass::v2::{
    command, query,
    runtime::{
        Callbacks, Executor, FunctionContext, Options, Outcome, ResolveContext, Seed, VisibleChunk,
    },
    schedule::Builder,
    schema::{Catalog, ChunkIndex, Meta, TableIndex},
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

pub fn run() {
    let mut catalog = Catalog::new();
    let mut tables = vec![catalog
        .register_table([Meta::of::<Spawner>()], test_configuration())
        .expect("example table registration should succeed")];
    tables[0].push_chunk();

    let mut builder = Builder::new(&mut catalog, &mut tables, test_configuration());
    let spawn_index = builder.push_query(
        "spawn",
        query::all(query::read::<Spawner>()).expect("example query declaration should succeed"),
    );
    let clamp_index = builder.push_query(
        "clamp",
        query::all(query::read::<Position>()).expect("example query declaration should succeed"),
    );
    let position_table_index = builder
        .add_insert(spawn_index, command::Insert::<(Position,)>::new())
        .expect("typed insert should resolve to one known table");
    let schedule = builder.build();
    let seed = Seed::from_tables(&tables);
    let callbacks = DemoCallbacks {
        injected_table_index: position_table_index,
    };
    let report = Executor::with_options(
        Options::default()
            .with_worker_count(non_zero_usize(2))
            .with_record_trace(true),
    )
    .run(&schedule, &seed, &callbacks);

    println!("Executor runtime");
    println!("  function count: {}", schedule.function_count());
    println!("  created job count: {}", report.created_job_count());
    println!("  injected job count: {}", report.injected_job_count());
    println!("  steal count: {}", report.steal_count());
    println!(
        "  clamp known tables: {:?}",
        schedule
            .function(clamp_index)
            .expect("clamp function should exist")
            .known_tables()
    );
    println!("  trace: {:?}", report.trace());
}

struct DemoCallbacks {
    injected_table_index: TableIndex,
}

impl Callbacks for DemoCallbacks {
    fn run_function(&self, context: FunctionContext<'_>) {
        println!(
            "  function {:?} on table {:?} chunk {:?} worker {}",
            context.function_index(),
            context.table_index(),
            context.chunk_index(),
            context.worker_index(),
        );
    }

    fn run_resolve(&self, context: ResolveContext<'_>) -> Outcome {
        println!(
            "  resolve {:?} on worker {}",
            context.resolve_index(),
            context.worker_index(),
        );

        if context.resolve().function_index().value() == 0 {
            Outcome::visible_chunks([VisibleChunk::new(
                self.injected_table_index,
                ChunkIndex::new(0),
            )])
        } else {
            Outcome::none()
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
