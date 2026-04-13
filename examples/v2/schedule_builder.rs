use core::num::NonZeroUsize;
use that_bass::v2::{command, query, schedule::Builder, schema::Meta, Configuration, Store};

#[repr(C)]
struct Position {
    x: f64,
    y: f64,
}

#[repr(C)]
struct Velocity {
    x: f64,
    y: f64,
}

pub fn run() {
    let mut store = Store::with_configuration(
        Configuration::default().with_target_chunk_byte_count(non_zero_usize(8 * 1024)),
    );
    store
        .register_table([Meta::of::<Position>()])
        .expect("example table registration should succeed");
    store
        .register_table([Meta::of::<Position>(), Meta::of::<Velocity>()])
        .expect("example table registration should succeed");

    let mut builder = Builder::new(&mut store);
    let integrate_index = builder.push_query(
        "integrate",
        query::all((
            query::write::<Position>(),
            query::option(query::read::<Velocity>()),
        ))
        .expect("example query declaration should succeed"),
    );
    let spawn_index = builder.push_query(
        "spawn",
        query::all(query::rows()).expect("example query declaration should succeed"),
    );
    builder
        .add_insert(spawn_index, command::Insert::<(Position, Velocity)>::new())
        .expect("typed insert should resolve to a known table");
    let clamp_index = builder.push_query(
        "clamp",
        query::all(query::write::<Position>()).expect("example query declaration should succeed"),
    );
    let schedule = builder.build();

    println!("Schedule builder");
    println!("  function count: {}", schedule.function_count());
    println!("  resolve count: {}", schedule.resolve_count());
    println!("  edge count: {}", schedule.edge_count());
    println!(
        "  integrate known tables: {:?}",
        schedule
            .function(integrate_index)
            .expect("integrate function should exist")
            .known_tables()
    );
    println!(
        "  spawn resolve dependencies: {:?}",
        schedule
            .resolve_for_function(spawn_index)
            .expect("spawn resolve should exist")
            .dependencies()
    );
    println!(
        "  spawn planned commands: {:?}",
        schedule
            .resolve_for_function(spawn_index)
            .expect("spawn resolve should exist")
            .command_plans()
    );
    println!(
        "  clamp known tables: {:?}",
        schedule
            .function(clamp_index)
            .expect("clamp function should exist")
            .known_tables()
    );
    println!("  edges: {:?}", schedule.edges());
}

fn non_zero_usize(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("example constants must be non-zero")
}
