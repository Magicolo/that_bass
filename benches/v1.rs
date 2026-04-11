#[path = "v1/mod.rs"]
mod suite;

use criterion::{criterion_group, criterion_main};

criterion_group!(benches, suite::create::benchmark);
criterion_main!(benches);
