mod chunk_layout;
mod instrumentation;
mod metadata;
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

    vocabulary::run();
    println!();

    instrumentation::run();
}
