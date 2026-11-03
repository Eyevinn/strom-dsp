//! Criterion benchmark of the true-peak limiter: 48 kHz stereo, 480-frame
//! (10 ms) buffers, 5 ms lookahead, −1 dBTP ceiling, on loud music-like
//! material so it limits much of the time. Throughput is in frames, so the
//! share of one core is `48 000 frames/s ÷ measured frames/s`.
use std::f64::consts::PI;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use strom_dsp::truepeak::{TruePeakLimiter, TruePeakLimiterParams};
use strom_dsp::Processor;

const FS: f32 = 48_000.0;
const FRAMES: usize = 480;

/// 2 s of stereo music-like material, peaking well above the ceiling, with a
/// slow level modulation and a little noise.
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
            let v = 2.0
                * env
                * (0.4 * (2.0 * PI * 60.0 * t).sin()
                    + 0.3 * (2.0 * PI * (330.0 + 5.0 * c as f64) * t).sin()
                    + 0.15 * (2.0 * PI * 2_100.0 * t).sin()
                    + 0.05 * (2.0 * PI * 7_000.0 * t).sin())
                + 0.02 * nz;
            out.push(v as f32);
        }
    }
    out
}

fn bench(c: &mut Criterion) {
    let src = programme();
    let mut l = TruePeakLimiter::new();
    l.set_params(&TruePeakLimiterParams {
        lookahead_ms: 5.0,
        ..TruePeakLimiterParams::default()
    });
    l.configure(FS, 2);
    let mut buf = vec![0.0f32; FRAMES * 2];
    let mut pos = 0;
    let mut group = c.benchmark_group("truepeak");
    group.throughput(Throughput::Elements(FRAMES as u64));
    group.bench_function("48k_stereo_480_lookahead_5ms", |b| {
        b.iter(|| {
            let len = buf.len();
            if pos + len > src.len() {
                pos = 0;
            }
            buf.copy_from_slice(&src[pos..pos + len]);
            pos += len;
            l.process(black_box(&mut buf));
            black_box(&buf);
        })
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
