//! Criterion benchmarks for the sidechain and expander paths, at 48 kHz
//! stereo, 480 frames (10 ms) per buffer:
//!
//! - `compressor_stereo_plain`: `process`, the reference;
//! - `compressor_stereo_key_hpf`: `process_with_key` with a stereo key through
//!   a 150 Hz high-pass key filter;
//! - `gate_stereo_plain`: `process` in gate mode, the reference;
//! - `gate_stereo_expander`: `process` in expander mode (ratio 2);
//! - `gate_stereo_expander_key_hpf`: expander mode, keyed through the HPF.
//!
//! The programme and key are 2 s synthetic signals (noise plus tone with a slow
//! level envelope, so the processors cross their thresholds), fed one buffer
//! per iteration as in `benches/strip.rs`. Share of one core =
//! time per buffer ÷ 10 ms.

use std::f64::consts::PI;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use strom_dsp::dynamics::KeyFilter;
use strom_dsp::gate::GateMode;
use strom_dsp::{Compressor, CompressorParams, Gate, GateParams, Processor};

const FS: f32 = 48_000.0;
const FRAMES: usize = 480;
const CH: usize = 2;

/// Interleaved stereo signal: tone + noise with a level envelope at `rate` Hz.
fn programme(secs: f64, tone: f64, rate: f64, seed: u64) -> Vec<f32> {
    let n = (FS as f64 * secs) as usize;
    let mut s = seed | 1;
    let mut out = Vec::with_capacity(n * CH);
    for i in 0..n {
        let t = i as f64 / FS as f64;
        let env = 0.005 + 0.9 * (0.5 - 0.5 * (2.0 * PI * rate * t).cos());
        for c in 0..CH {
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

fn hpf() -> KeyFilter {
    KeyFilter::HighPass {
        frequency_hz: 150.0,
        q: 0.707,
    }
}

fn gate_params() -> GateParams {
    GateParams {
        open_threshold_db: -30.0,
        close_threshold_db: -40.0,
        attack_ms: 2.0,
        hold_ms: 50.0,
        release_ms: 150.0,
        reduction_db: -40.0,
    }
}

fn compressor_params() -> CompressorParams {
    CompressorParams {
        threshold: 0.25,
        ratio: 4.0,
        attack_ms: 10.0,
        release_ms: 120.0,
        makeup_gain: 1.5,
        knee: 0.25,
    }
}

/// Benchmark `f(buf, key)` on one fresh buffer (and its key) per iteration.
fn run(c: &mut Criterion, name: &str, mut f: impl FnMut(&mut [f32], &[f32])) {
    let prog = programme(2.0, 220.0, 0.7, 0x9e37_79b9_7f4a_7c15);
    let key = programme(2.0, 1000.0, 1.3, 0x2545_f491_4f6c_dd1d);
    let len = FRAMES * CH;
    let mut buf = vec![0.0f32; len];
    let mut pos = 0usize;
    let mut step = |buf: &mut [f32], f: &mut dyn FnMut(&mut [f32], &[f32])| {
        if pos + len > prog.len() {
            pos = 0;
        }
        buf.copy_from_slice(&prog[pos..pos + len]);
        f(buf, &key[pos..pos + len]);
        pos += len;
    };
    for _ in 0..200 {
        step(&mut buf, &mut f);
    }
    let mut g = c.benchmark_group(name);
    g.throughput(Throughput::Elements(FRAMES as u64));
    g.bench_function("480", |b| {
        b.iter(|| step(black_box(&mut buf), &mut f));
    });
    g.finish();
}

fn compressor() -> Compressor {
    let mut c = Compressor::new();
    c.configure(FS, CH);
    c.set_params(&compressor_params());
    c
}

fn gate(mode: GateMode) -> Gate {
    let mut g = Gate::new();
    g.configure(FS, CH);
    g.set_params(&gate_params());
    g.set_mode(mode);
    g.set_expander_ratio(2.0);
    g
}

fn bench(c: &mut Criterion) {
    let mut p = compressor();
    run(c, "compressor_stereo_plain", |b, _| p.process(b));
    let mut p = compressor();
    p.set_key_filter(hpf());
    run(c, "compressor_stereo_key_hpf", |b, k| {
        p.process_with_key(b, k, CH)
    });
    let mut g = gate(GateMode::Gate);
    run(c, "gate_stereo_plain", |b, _| g.process(b));
    let mut g = gate(GateMode::Expander);
    run(c, "gate_stereo_expander", |b, _| g.process(b));
    let mut g = gate(GateMode::Expander);
    g.set_key_filter(hpf());
    run(c, "gate_stereo_expander_key_hpf", |b, k| {
        g.process_with_key(b, k, CH)
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
