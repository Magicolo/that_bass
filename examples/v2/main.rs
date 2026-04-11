mod instrumentation;
mod store_planning;
mod vocabulary;

fn main() {
    println!("that_bass v2 foundation example");
    println!();

    store_planning::run();
    println!();

    vocabulary::run();
    println!();

    instrumentation::run();
}
