mod chunk_layout;
mod instrumentation;
mod keyless_rows;
mod metadata;
mod query_surface;
mod store_planning;
mod vocabulary;

fn main() {
    println!("that_bass v2 foundation example");
    println!();

    store_planning::run();
    println!();

    metadata::run();
    println!();

    chunk_layout::run();
    println!();

    keyless_rows::run();
    println!();

    query_surface::run();
    println!();

    vocabulary::run();
    println!();

    instrumentation::run();
}
