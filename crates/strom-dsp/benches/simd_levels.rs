//! Criterion benchmark of the vector kernels at every SIMD level the CPU
//! supports (x86-64: AVX-512, AVX2, SSE4.2; aarch64: NEON; and the scalar
//! fallback everywhere), 48 kHz, 480-frame (10 ms) buffers.
//!
//! Each case is forced to each level with `set_simd_level`, so a vector
//! level that runs no faster than the fallback stands out (the symptom of
//! code outlined from `vectorize`, #8 and #11). The deterministic guard
//! against that in CI is `simd_gungraun`; this bench gives the wall-clock
//! picture per level.
//!
//! ```text
//! cargo bench -p strom-dsp --bench simd_levels
//! cargo bench -p strom-dsp --bench simd_levels -- eq4_stereo
//! ```
use std::f64::consts::PI;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use strom_dsp::oversample::{OversampleFactor, Oversampler};
use strom_dsp::simd::{detect, Level};
use strom_dsp::truepeak::{TruePeakLimiter, TruePeakLimiterParams};
use strom_dsp::{
    BandParams, BandType, Equalizer, EqualizerParams, Limiter, LimiterParams, Processor,
};

const FS: f32 = 48_000.0;
const FRAMES: usize = 480;

/// Every level the CPU supports, fastest first, then the fallback.
fn levels() -> Vec<(&'static str, Level)> {
    let d = detect();
    #[allow(unused_mut)]
    let mut v = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = d.as_avx512() {
            v.push(("avx512", Level::Avx512(t)));
        }
        if let Some(t) = d.as_avx2() {
            v.push(("avx2", Level::Avx2(t)));
        }
        if let Some(t) = d.as_sse4_2() {
            v.push(("sse4.2", Level::Sse4_2(t)));
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = d.as_neon() {
        v.push(("neon", Level::Neon(t)));
    }
    let _ = d;
    v.push(("fallback", Level::fallback()));
    v
}

/// 2 s of programme-like material with `ch` channels: tones over the bands,
/// a slow level modulation and a little noise; `gain` 2 drives the limiters
/// into reduction much of the time.
fn programme(ch: usize, gain: f64) -> Vec<f32> {
    let n = 2 * FS as usize;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut out = Vec::with_capacity(n * ch);
    for i in 0..n {
        let t = i as f64 / FS as f64;
        let env = 0.5 + 0.5 * (2.0 * PI * 1.5 * t).sin();
        for c in 0..ch {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let nz = (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            let v = gain
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

/// Bench `process` on consecutive buffers of `src` (`ch` channels) for
/// every level, as `group/level`.
fn bench_levels(
    c: &mut Criterion,
    group: &str,
    ch: usize,
    src: &[f32],
    mut make: impl FnMut(Level) -> Box<dyn FnMut(&mut [f32])>,
) {
    let mut g = c.benchmark_group(group);
    g.throughput(Throughput::Elements(FRAMES as u64));
    let len = FRAMES * ch;
    for (name, level) in levels() {
        let mut process = make(level);
        let mut buf = vec![0.0f32; len];
        let mut pos = 0;
        let mut feed = |process: &mut dyn FnMut(&mut [f32]), buf: &mut [f32]| {
            if pos + len > src.len() {
                pos = 0;
            }
            buf.copy_from_slice(&src[pos..pos + len]);
            pos += len;
            process(black_box(buf));
        };
        // Warm up on real material (filter states, envelopes).
        for _ in 0..200 {
            feed(&mut process, &mut buf);
        }
        g.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| feed(&mut process, &mut buf))
        });
    }
    g.finish();
}

fn oversampler(c: &mut Criterion) {
    let src = programme(2, 1.0);
    for (group, factor) in [
        ("oversample_4x_up_down", OversampleFactor::X4),
        ("oversample_8x_up_down", OversampleFactor::X8),
    ] {
        bench_levels(c, group, 2, &src, |level| {
            let mut o = Oversampler::new();
            o.set_simd_level(level);
            o.configure(factor, 2);
            let mut up = vec![0.0f32; FRAMES * 2 * factor.ratio()];
            Box::new(move |buf| {
                o.upsample(buf, &mut up);
                o.downsample(&up, buf);
            })
        });
    }
}

/// HPF + two peaking bands + high shelf (as in `strip`).
fn eq_params() -> EqualizerParams {
    let mut p = EqualizerParams {
        num_bands: 4,
        ..Default::default()
    };
    let bands = [
        (BandType::HighPass, 80.0, 0.0, 0.707),
        (BandType::Peaking, 300.0, -3.0, 1.2),
        (BandType::Peaking, 3000.0, 2.5, 0.9),
        (BandType::HighShelf, 10_000.0, 2.0, 0.707),
    ];
    for (b, (band_type, frequency, gain_db, q)) in p.bands.iter_mut().zip(bands) {
        *b = BandParams {
            enabled: true,
            band_type,
            frequency,
            gain_db,
            q,
        };
    }
    p
}

fn equalizer(c: &mut Criterion) {
    for (group, ch) in [("eq4_stereo", 2), ("eq4_8ch", 8)] {
        let src = programme(ch, 0.5);
        bench_levels(c, group, ch, &src, |level| {
            let mut e = Equalizer::new();
            e.set_simd_level(level);
            e.configure(FS, ch);
            e.set_params(&eq_params());
            Box::new(move |buf| e.process(buf))
        });
    }
}

fn limiters(c: &mut Criterion) {
    let src = programme(2, 2.0);
    bench_levels(c, "limiter_stereo", 2, &src, |level| {
        let mut l = Limiter::new();
        l.set_simd_level(level);
        l.configure(FS, 2);
        l.set_params(&LimiterParams {
            enabled: true,
            threshold_db: -1.0,
            attack_ms: 1.0,
            release_ms: 80.0,
            lookahead_ms: 5.0,
        });
        Box::new(move |buf| l.process(buf))
    });
    bench_levels(c, "truepeak_stereo_lookahead_5ms", 2, &src, |level| {
        let mut l = TruePeakLimiter::new();
        l.set_simd_level(level);
        l.set_params(&TruePeakLimiterParams {
            lookahead_ms: 5.0,
            ..TruePeakLimiterParams::default()
        });
        l.configure(FS, 2);
        Box::new(move |buf| l.process(buf))
    });
}

criterion_group!(benches, oversampler, equalizer, limiters);
criterion_main!(benches);
