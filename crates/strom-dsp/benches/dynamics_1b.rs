//! Criterion benchmarks for the phase 1b dynamics processors, at 48 kHz
//! stereo programme, 480 frames (10 ms) per buffer. Share of one core = time
//! per buffer ÷ 10 ms.
//!
//! Ducker (`bench_ducker`):
//! - `ducker_stereo_key_mono_fixed`: 48 kHz mono key, 100 Hz key high-pass,
//!   fixed range;
//! - `ducker_stereo_key_mono_proportional`: the same in proportional mode
//!   (one `log10` per frame while the key is above the threshold);
//! - `ducker_stereo_key_16k`: a 16 kHz mono key (detector at the key rate);
//! - `ducker_stereo_no_key`: `process` without a key (silence path).
//!
//! The key alternates between speech-like bursts and pauses, so the ducker
//! keeps going through attack, hold and release.

use std::f64::consts::PI;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use strom_dsp::ducker::{Ducker, DuckerParams, RangeMode};
use strom_dsp::Processor;

const FS: f32 = 48_000.0;
const FRAMES: usize = 480;
const CH: usize = 2;

/// Interleaved signal of `ch` channels at `rate`: tone + noise, gated on and
/// off every `period` seconds.
fn signal(rate: f32, ch: usize, secs: f64, tone: f64, period: f64, seed: u64) -> Vec<f32> {
    let n = (rate as f64 * secs) as usize;
    let mut s = seed | 1;
    let mut out = Vec::with_capacity(n * ch);
    for i in 0..n {
        let t = i as f64 / rate as f64;
        let on = ((t / period) as usize).is_multiple_of(2);
        let env = if on { 0.5 } else { 0.002 };
        for c in 0..ch {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let nz = (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            let x = (2.0 * PI * (tone + 20.0 * c as f64) * t).sin();
            out.push((env * (0.7 * x + 0.3 * nz)) as f32);
        }
    }
    out
}

fn ducker(mode: RangeMode, key_rate: f32) -> Ducker {
    let mut d = Ducker::new();
    d.configure(FS, CH);
    d.set_key_format(key_rate, 1);
    d.set_params(&DuckerParams {
        range_mode: mode,
        hold_ms: 100.0,
        release_ms: 300.0,
        ..DuckerParams::default()
    });
    d
}

/// Benchmark one 480-frame programme buffer (and its key span) per iteration.
fn run_ducker(c: &mut Criterion, name: &str, mut d: Ducker, key_rate: f32, with_key: bool) {
    let prog = signal(FS, CH, 2.0, 220.0, 10.0, 0x9e37_79b9_7f4a_7c15);
    let key = signal(key_rate, 1, 2.0, 700.0, 0.4, 0x2545_f491_4f6c_dd1d);
    let key_per_buf = (FRAMES as f64 * key_rate as f64 / FS as f64) as usize;
    let len = FRAMES * CH;
    let mut buf = vec![0.0f32; len];
    let (mut pos, mut kpos) = (0usize, 0usize);
    let mut step = |buf: &mut [f32], d: &mut Ducker| {
        if pos + len > prog.len() || kpos + key_per_buf > key.len() {
            pos = 0;
            kpos = 0;
        }
        buf.copy_from_slice(&prog[pos..pos + len]);
        if with_key {
            d.process_with_key(buf, &key[kpos..kpos + key_per_buf], 1);
        } else {
            d.process(buf);
        }
        pos += len;
        kpos += key_per_buf;
    };
    for _ in 0..200 {
        step(&mut buf, &mut d);
    }
    let mut g = c.benchmark_group(name);
    g.throughput(Throughput::Elements(FRAMES as u64));
    g.bench_function("480", |b| {
        b.iter(|| step(black_box(&mut buf), &mut d));
    });
    g.finish();
}

fn bench_ducker(c: &mut Criterion) {
    let k = FS;
    run_ducker(
        c,
        "ducker_stereo_key_mono_fixed",
        ducker(RangeMode::Fixed, k),
        k,
        true,
    );
    run_ducker(
        c,
        "ducker_stereo_key_mono_proportional",
        ducker(RangeMode::Proportional, k),
        k,
        true,
    );
    let k16 = 16_000.0;
    run_ducker(
        c,
        "ducker_stereo_key_16k",
        ducker(RangeMode::Fixed, k16),
        k16,
        true,
    );
    run_ducker(
        c,
        "ducker_stereo_no_key",
        ducker(RangeMode::Fixed, k),
        k,
        false,
    );
}

criterion_group!(benches, bench_ducker);
criterion_main!(benches);
