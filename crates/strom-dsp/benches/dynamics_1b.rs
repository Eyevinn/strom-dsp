//! Criterion benchmarks of the phase 1b dynamics processors at 48 kHz stereo
//! with 480-frame (10 ms) buffers. Throughput is in frames, so the share of
//! one core is `48 000 frames/s ÷ measured frames/s`.
use std::f64::consts::PI;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use strom_dsp::Processor;

const FS: f32 = 48_000.0;
const FRAMES: usize = 480;

/// 2 s of stereo speech-like test material: a 180 Hz tone plus 7 kHz bursts
/// (100 ms on, 100 ms off) and a little noise, so the de-esser alternates
/// between reducing and releasing.
fn sibilant_programme() -> Vec<f32> {
    let n = 2 * FS as usize;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f64 / FS as f64;
        let burst = if (i / 4800) % 2 == 0 { 0.3 } else { 0.0 };
        for c in 0..2 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let nz = (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            let v = 0.3 * (2.0 * PI * (180.0 + 20.0 * c as f64) * t).sin()
                + burst * (2.0 * PI * 7_000.0 * t).sin()
                + 0.02 * nz;
            out.push(v as f32);
        }
    }
    out
}

/// One buffer per iteration, advancing through `src` (the copy is included
/// and negligible).
fn run_buffers<P: Processor>(c: &mut Criterion, name: &str, mut dsp: P, src: &[f32]) {
    let mut buf = vec![0.0f32; FRAMES * 2];
    let mut pos = 0;
    let mut group = c.benchmark_group("dynamics_1b");
    group.throughput(Throughput::Elements(FRAMES as u64));
    group.bench_function(name, |b| {
        b.iter(|| {
            let len = buf.len();
            if pos + len > src.len() {
                pos = 0;
            }
            buf.copy_from_slice(&src[pos..pos + len]);
            pos += len;
            dsp.process(black_box(&mut buf));
            black_box(&buf);
        })
    });
    group.finish();
}

fn bench_deesser(c: &mut Criterion) {
    use strom_dsp::deesser::{Deesser, DeesserMode, DeesserParams};
    let src = sibilant_programme();
    for (name, mode) in [
        ("deesser_split_stereo", DeesserMode::SplitBand),
        ("deesser_wide_stereo", DeesserMode::Wideband),
    ] {
        let mut d = Deesser::new();
        d.set_params(&DeesserParams {
            mode,
            ..Default::default()
        });
        d.configure(FS, 2);
        run_buffers(c, name, d, &src);
    }
}

criterion_group!(benches, bench_deesser);
criterion_main!(benches);
