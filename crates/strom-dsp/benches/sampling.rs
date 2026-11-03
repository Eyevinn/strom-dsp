//! Criterion benchmarks of the sampling building blocks at 48 kHz stereo:
//!
//! - `oversample/4x_up_down/480`: one 480-frame (10 ms) buffer upsampled 4×
//!   and downsampled back, on the detected SIMD level and on the scalar
//!   fallback (to show what the explicit SIMD buys).
//! - `oversample/{2x,8x}_up_down/480` for comparison, and the 4× peak path the
//!   true-peak limiter will use.
//! - `stft/{1024_256,2048_512}_identity/480`: STFT analysis + resynthesis
//!   with an identity callback, frame/hop 1024/256 and 2048/512.
//!
//! Throughput is in base-rate frames, so the share of one core is
//! `48 000 frames/s ÷ measured frames/s`.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use strom_dsp::oversample::{OversampleFactor, Oversampler};
use strom_dsp::simd::{detect, Level};
use strom_dsp::stft::{Complex32, Stft, StftHop};

const CH: usize = 2;
const FRAMES: usize = 480;

fn programme(frames: usize, ch: usize) -> Vec<f32> {
    let mut s = 0x9e37_79b9u32;
    (0..frames * ch)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let n = (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
            0.3 * n + 0.5 * (i as f32 * 0.031).sin()
        })
        .collect()
}

fn oversample(c: &mut Criterion) {
    let x = programme(FRAMES, CH);
    let mut g = c.benchmark_group("oversample");
    g.throughput(Throughput::Elements(FRAMES as u64));
    let cases = [
        ("2x_up_down", OversampleFactor::X2, detect()),
        ("4x_up_down", OversampleFactor::X4, detect()),
        (
            "4x_up_down_fallback",
            OversampleFactor::X4,
            Level::fallback(),
        ),
        ("8x_up_down", OversampleFactor::X8, detect()),
    ];
    for (name, f, level) in cases {
        let l = f.ratio();
        let mut o = Oversampler::new();
        o.configure(f, CH);
        o.set_simd_level(level);
        let mut up = vec![0.0f32; FRAMES * l * CH];
        let mut y = vec![0.0f32; FRAMES * CH];
        g.bench_function(format!("{name}/{FRAMES}"), |b| {
            b.iter(|| {
                o.upsample(black_box(&x), &mut up);
                o.downsample(&up, &mut y);
                black_box(&y);
            })
        });
    }
    let mut o = Oversampler::new();
    o.configure(OversampleFactor::X4, CH);
    let mut pk = vec![0.0f32; FRAMES];
    g.bench_function(format!("4x_peaks/{FRAMES}"), |b| {
        b.iter(|| {
            o.upsample_peaks(black_box(&x), &mut pk);
            black_box(&pk);
        })
    });
    g.finish();
}

fn stft(c: &mut Criterion) {
    // Long input, consumed one 480-frame buffer per iteration, so every
    // hop's FFT pair is included at its real rate.
    let src = programme(FRAMES * 100, CH);
    let mut g = c.benchmark_group("stft");
    g.throughput(Throughput::Elements(FRAMES as u64));
    for (frame, hop, name) in [
        (1024, StftHop::Quarter, "1024_256"),
        (2048, StftHop::Quarter, "2048_512"),
    ] {
        let mut s = Stft::new();
        s.configure(frame, hop, CH).unwrap();
        let mut buf = vec![0.0f32; FRAMES * CH];
        let mut pos = 0;
        let mut identity = |_: usize, sp: &mut [Complex32]| {
            black_box(sp);
        };
        g.bench_function(format!("{name}_identity/{FRAMES}"), |b| {
            b.iter(|| {
                let n = FRAMES * CH;
                if pos + n > src.len() {
                    pos = 0;
                }
                buf.copy_from_slice(&src[pos..pos + n]);
                pos += n;
                s.process(&mut buf, &mut identity);
                black_box(&buf);
            })
        });
    }
    g.finish();
}

criterion_group!(benches, oversample, stft);
criterion_main!(benches);
