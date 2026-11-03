//! Loudness detector and AGC benchmarks at 48 kHz stereo, 480-frame (10 ms)
//! blocks:
//!
//! - `detector_process`: [`LoudnessDetector::process`] (K-weighting, energy,
//!   running sums);
//! - `detector_process_and_read`: the same plus a momentary and a short-term
//!   reading every block (what a per-block AGC or meter needs);
//! - `agc_process`: the whole AGC (detector, gate, gain ramp, gain applied);
//! - `ebur128_process_and_short_term_read`: for comparison, the
//!   `ebur128`-based `LoudnessMeter` fed and read every block, which is why
//!   the AGC has its own detector.
//!
//! ```text
//! cargo bench -p strom-dsp --bench agc
//! ```

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use strom_dsp::agc::{Agc, AgcParams};
use strom_dsp::meter::{ChannelLayout, LoudnessConfig, LoudnessDetector, LoudnessMeter};
use strom_dsp::Processor;

const FS: f32 = 48_000.0;
const FRAMES: usize = 480;

/// 1 s of stereo programme-like material (two sines plus noise) with a slow
/// level contour, so the AGC keeps moving.
fn input() -> Vec<f32> {
    let mut x = 0x2545_f491u32;
    (0..48_000)
        .flat_map(|n| {
            let t = n as f32 / FS;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let noise = (x >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
            let env = 0.5 + 0.4 * (2.0 * std::f32::consts::PI * 0.5 * t).sin();
            let a = 0.3 * (2.0 * std::f32::consts::PI * 220.0 * t).sin();
            let b = 0.2 * (2.0 * std::f32::consts::PI * 1_250.0 * t).sin();
            [env * (a + b + 0.05 * noise), env * (a - b + 0.05 * noise)]
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let sig = input();
    let blocks: Vec<&[f32]> = sig
        .as_chunks::<{ 2 * FRAMES }>()
        .0
        .iter()
        .map(|b| b.as_slice())
        .collect();

    let mut g = c.benchmark_group("agc_48k_stereo_480");
    g.throughput(Throughput::Elements(FRAMES as u64));

    let mut d = LoudnessDetector::new();
    d.configure(FS, 2, ChannelLayout::Stereo);
    let mut i = 0;
    g.bench_function("detector_process", |b| {
        b.iter(|| {
            d.process(black_box(blocks[i % blocks.len()]));
            i += 1;
        })
    });

    let mut i = 0;
    g.bench_function("detector_process_and_read", |b| {
        b.iter(|| {
            d.process(black_box(blocks[i % blocks.len()]));
            i += 1;
            black_box(d.momentary_lufs() + d.short_term_lufs())
        })
    });

    let mut a = Agc::new();
    a.configure_with_layout(FS, 2, ChannelLayout::Stereo);
    a.set_params(&AgcParams {
        target_lufs: -16.0,
        ..AgcParams::default()
    });
    let mut buf = vec![0.0f32; 2 * FRAMES];
    let mut i = 0;
    g.bench_function("agc_process", |b| {
        b.iter(|| {
            buf.copy_from_slice(blocks[i % blocks.len()]);
            a.process(black_box(&mut buf));
            i += 1;
            black_box(a.gain_db())
        })
    });

    let mut m = LoudnessMeter::new(&LoudnessConfig {
        sample_rate: FS,
        channels: 2,
        layout: ChannelLayout::Stereo,
        true_peak: false,
    })
    .unwrap();
    let mut i = 0;
    g.bench_function("ebur128_process_and_short_term_read", |b| {
        b.iter(|| {
            m.process(black_box(blocks[i % blocks.len()]));
            i += 1;
            black_box(m.momentary() + m.short_term())
        })
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
