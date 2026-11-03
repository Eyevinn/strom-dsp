//! Criterion benchmark of the spike guard, 48 kHz stereo, 480-frame (10 ms)
//! buffers: programme without events (the common case: detector only) and
//! the same programme with a click every 50 ms (events and gain dips).
//! Throughput is in frames, so the share of one core is
//! `48 000 frames/s ÷ measured frames/s`.
use std::f64::consts::PI;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use strom_dsp::spike_guard::{SpikeGuard, SpikeGuardParams};
use strom_dsp::Processor;

const FS: f32 = 48_000.0;
const FRAMES: usize = 480;

/// 2 s of stereo music-like material (tones with a slow level modulation
/// and a little noise); with `clicks`, a single-sample click every 50 ms.
fn programme(clicks: bool) -> Vec<f32> {
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
            let click = if clicks && i % 2400 == 1200 { 1.5 } else { 0.0 };
            let v = env
                * (0.3 * (2.0 * PI * 60.0 * t).sin()
                    + 0.2 * (2.0 * PI * (330.0 + 5.0 * c as f64) * t).sin()
                    + 0.05 * (2.0 * PI * 7_000.0 * t).sin())
                + 0.02 * nz
                + click;
            out.push(v as f32);
        }
    }
    out
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("spike_guard");
    group.throughput(Throughput::Elements(FRAMES as u64));
    for (name, clicks) in [
        ("programme_48k_stereo_480", false),
        ("clicks_every_50ms_48k_stereo_480", true),
    ] {
        let src = programme(clicks);
        let mut g = SpikeGuard::new();
        g.set_params(&SpikeGuardParams::default());
        g.configure(FS, 2);
        let mut buf = vec![0.0f32; FRAMES * 2];
        let mut pos = 0;
        group.bench_function(name, |b| {
            b.iter(|| {
                let len = buf.len();
                if pos + len > src.len() {
                    pos = 0;
                }
                buf.copy_from_slice(&src[pos..pos + len]);
                pos += len;
                g.process(black_box(&mut buf));
                black_box(g.take_events());
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
