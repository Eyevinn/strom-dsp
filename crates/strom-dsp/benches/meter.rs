//! Meter benchmarks at 48 kHz stereo, 480-frame (10 ms) blocks:
//! `LoudnessMeter::process` with and without true peak, each reading on its
//! own (their cost depends on the window, not on the block), the correlation
//! meter, the 3-band correlation meter and the mono detector.
//!
//! ```text
//! cargo bench -p strom-dsp --bench meter
//! ```

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use strom_dsp::meter::{
    BandCorrelation, ChannelLayout, CorrelationMeter, LoudnessConfig, LoudnessMeter, MonoDetector,
    MonoDetectorParams,
};

const FS: f32 = 48_000.0;
const FRAMES: usize = 480;

/// 1 s of stereo programme-like material (two sines plus noise).
fn input() -> Vec<f32> {
    let mut x = 0x2545_f491u32;
    (0..48_000)
        .flat_map(|n| {
            let t = n as f32 / FS;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let noise = (x >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
            let a = 0.3 * (2.0 * std::f32::consts::PI * 220.0 * t).sin();
            let b = 0.2 * (2.0 * std::f32::consts::PI * 1_250.0 * t).sin();
            [a + b + 0.05 * noise, a - b + 0.05 * noise]
        })
        .collect()
}

fn meter(true_peak: bool) -> LoudnessMeter {
    LoudnessMeter::new(&LoudnessConfig {
        sample_rate: FS,
        channels: 2,
        layout: ChannelLayout::Stereo,
        true_peak,
    })
    .unwrap()
}

fn bench(c: &mut Criterion) {
    let sig = input();
    let blocks: Vec<&[f32]> = sig
        .as_chunks::<{ 2 * FRAMES }>()
        .0
        .iter()
        .map(|b| b.as_slice())
        .collect();

    let mut g = c.benchmark_group("meter_48k_stereo_480");
    g.throughput(Throughput::Elements(FRAMES as u64));
    for (name, tp) in [
        ("loudness_process", false),
        ("loudness_process_true_peak", true),
    ] {
        let mut m = meter(tp);
        let mut i = 0;
        g.bench_function(name, |b| {
            b.iter(|| {
                m.process(black_box(blocks[i % blocks.len()]));
                i += 1;
            })
        });
    }
    let mut m = CorrelationMeter::new(FS, 300.0);
    let mut i = 0;
    g.bench_function("correlation", |b| {
        b.iter(|| {
            m.process_interleaved(black_box(blocks[i % blocks.len()]), 2);
            i += 1;
            black_box(m.correlation())
        })
    });
    let mut m = BandCorrelation::new(FS, 300.0);
    let mut i = 0;
    g.bench_function("band_correlation", |b| {
        b.iter(|| {
            m.process_interleaved(black_box(blocks[i % blocks.len()]), 2);
            i += 1;
            black_box(m.correlations())
        })
    });
    let mut d = MonoDetector::new(FS, &MonoDetectorParams::default());
    let mut i = 0;
    g.bench_function("mono_detector", |b| {
        b.iter(|| {
            d.process_interleaved(black_box(blocks[i % blocks.len()]), 2);
            i += 1;
            black_box(d.state())
        })
    });
    g.finish();

    // Readings after 10 s of programme (windows and histograms filled).
    let mut m = meter(true);
    for _ in 0..10 {
        m.process(&sig);
    }
    let mut g = c.benchmark_group("meter_read_48k_stereo");
    g.bench_function("momentary", |b| b.iter(|| black_box(&m).momentary()));
    g.bench_function("short_term", |b| b.iter(|| black_box(&m).short_term()));
    g.bench_function("integrated", |b| b.iter(|| black_box(&m).integrated()));
    g.bench_function("loudness_range", |b| {
        b.iter(|| black_box(&m).loudness_range())
    });
    g.bench_function("reading_all", |b| b.iter(|| black_box(&m).reading()));
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
