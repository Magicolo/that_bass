use that_bass::v2::{command, query, Configuration, Key, Store};

#[repr(C)]
#[derive(Clone, Copy)]
struct Position {
    x: f32,
    y: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Velocity {
    x: f32,
    y: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Time {
    seconds: f32,
}

pub fn run() {
    let mut store = Store::with_configuration(Configuration::default());
    let position_table_index = store
        .register::<Position>()
        .expect("position registration should succeed");
    let moving_table_index = store
        .register_row::<(Position, Velocity)>()
        .expect("row registration should succeed");
    let time_table_index = store
        .initialize_global(Time { seconds: 0.016 })
        .expect("global initialization should succeed");

    let integrate = (
        query::all((
            query::write::<Position>(),
            query::option(query::read::<Velocity>()),
        ))
        .expect("integrate query should be valid"),
        query::one::<Time>(),
    );

    let mut builder = store.builder();
    let spawn_index = builder
        .push(
            "spawn",
            query::all(query::rows()).expect("rows query should be valid"),
        )
        .expect("spawn function should be registered");
    builder
        .add_insert(spawn_index, command::insert::<(Key, Position)>())
        .expect("keyed insert planning should succeed");
    let integrate_index = builder
        .push("integrate", integrate)
        .expect("mixed stream and singleton inputs should be valid");
    let schedule = builder.build();

    println!("Recommended surface");
    println!(
        "  registered position table: {}",
        position_table_index.value()
    );
    println!("  registered moving table: {}", moving_table_index.value());
    println!("  initialized time table: {}", time_table_index.value());
    println!("  schedule functions: {}", schedule.function_count());
    println!("  schedule resolves: {}", schedule.resolve_count());
    println!("  integrate function index: {}", integrate_index.value());
}
