#[path = "v2/mod.rs"]
mod suite;

use criterion::{criterion_group, criterion_main};

criterion_group!(
    benches,
    suite::foundation::benchmark,
    suite::runtime::benchmark,
    suite::workloads::benchmark
);
criterion_main!(benches);
