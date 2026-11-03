//! Criterion benchmark of the full broadcast chain (input gain → AGC →
//! 4-band multiband compressor → limiter, plus all meters), 48 kHz stereo,
//! 480-frame (10 ms) buffers, with `take_meters` once per buffer. Throughput
//! is in frames, so the share of one core is
//! `48 000 frames/s ÷ measured frames/s`.
use std::f64::consts::PI;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use strom_dsp::broadcast::{BroadcastParams, BroadcastProcessor};
use strom_dsp::Processor;

const FS: f32 = 48_000.0;
const FRAMES: usize = 480;

/// 2 s of stereo music-like test material: a few tones spread over the
/// bands with a slow level modulation, plus a little noise.
fn programme() -> Vec<f32> {
    let n = 2 * FS as usize;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f64 / FS as f64;
        let env = 0.5 + 0.5 * (2.0 * PI * 1.5 * t).sin();
        for c in 0..2 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let nz = (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            let v = env
                * (0.3 * (2.0 * PI * 60.0 * t).sin()
                    + 0.2 * (2.0 * PI * (330.0 + 5.0 * c as f64) * t).sin()
                    + 0.1 * (2.0 * PI * 2_100.0 * t).sin()
                    + 0.05 * (2.0 * PI * 7_000.0 * t).sin())
                + 0.02 * nz;
            out.push(v as f32);
        }
    }
    out
}

fn bench(c: &mut Criterion) {
    let src = programme();
    let mut bp = BroadcastProcessor::new();
    bp.set_params(&BroadcastParams {
        num_bands: 4,
        ..BroadcastParams::default()
    });
    bp.configure(FS, 2);
    let mut buf = vec![0.0f32; FRAMES * 2];
    let mut pos = 0;
    let mut group = c.benchmark_group("broadcast");
    group.throughput(Throughput::Elements(FRAMES as u64));
    group.bench_function("full_chain_4_bands_48k_stereo_480", |b| {
        b.iter(|| {
            let len = buf.len();
            if pos + len > src.len() {
                pos = 0;
            }
            buf.copy_from_slice(&src[pos..pos + len]);
            pos += len;
            bp.process(black_box(&mut buf));
            black_box(bp.take_meters());
        })
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
