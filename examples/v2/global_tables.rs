use that_bass::v2::{query, schedule::Builder, Configuration, Store};

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Time {
    seconds: f32,
}

pub fn run() {
    let mut store = Store::with_configuration(Configuration::default());
    let position_table_index = store
        .register_table([that_bass::v2::schema::Meta::of::<Position>()])
        .expect("table registration should succeed");
    let time_table_index = store
        .initialize_global(Time { seconds: 0.016 })
        .expect("global initialization should succeed");
    let mut builder = Builder::new(&mut store);
    let function_index = builder
        .push(
            "integrate",
            (query::write::<Position>(), query::one::<Time>()),
        )
        .expect("mixed stream and singleton inputs should plan successfully");
    let schedule = builder.build();
    let global_time = query::one::<Time>()
        .get(&store)
        .expect("singleton query should read the initialized value");

    println!("Global tables");
    println!("  position table index: {}", position_table_index.value());
    println!("  time table index: {}", time_table_index.value());
    println!(
        "  schedule function known tables: {:?}",
        schedule
            .function(function_index)
            .expect("scheduled function should exist")
            .known_tables()
    );
    println!("  singleton value: {} seconds", global_time.seconds);
}
