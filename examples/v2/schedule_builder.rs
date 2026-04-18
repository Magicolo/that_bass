use core::num::NonZeroUsize;
use that_bass::v2::{command, query, Configuration, Store};

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
        .register::<Position>()
        .expect("example table registration should succeed");
    store
        .register_row::<(Position, Velocity)>()
        .expect("example table registration should succeed");

    let mut builder = store.builder();
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
        .add_insert(spawn_index, command::insert::<(Position, Velocity)>())
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
    println!("  integrate function index: {}", integrate_index.value());
    println!("  spawn function index: {}", spawn_index.value());
    println!("  clamp function index: {}", clamp_index.value());
}

fn non_zero_usize(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("example constants must be non-zero")
}
