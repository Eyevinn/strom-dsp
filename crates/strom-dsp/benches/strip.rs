//! Placeholder; the channel-strip benchmark is filled in after all processors exist.
use criterion::{criterion_group, criterion_main, Criterion};

fn bench(_c: &mut Criterion) {}

criterion_group!(benches, bench);
criterion_main!(benches);
