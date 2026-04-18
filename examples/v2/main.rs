mod command_resolution;
mod executor_runtime;
mod global_tables;
mod managed_keys;
mod schedule_builder;
mod store_planning;
mod vocabulary;

fn main() {
    println!("that_bass v2 public API example");
    println!();

    store_planning::run();
    println!();

    managed_keys::run();
    println!();

    schedule_builder::run();
    println!();

    executor_runtime::run();
    println!();

    command_resolution::run();
    println!();

    global_tables::run();
    println!();

    vocabulary::run();
}
