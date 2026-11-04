//! Benchmarks of the declipper at 48 kHz stereo with 480-frame (10 ms)
//! buffers: clean material (detection only), material clipped 6 and 10 dB
//! below its peak (detection and repair), and an adversarial signal with
//! as many runs of the longest repairable length as the waveform allows.
//!
//! Two parts:
//!
//! 1. **Per buffer** (printed first): every buffer of 8 s of each case is
//!    timed on its own (the fastest of five runs from the same state, so
//!    preemption by other processes does not count), for each latency
//!    setting; the table gives the mean and the worst buffer as a share of
//!    the buffer's duration (10 ms), the fallbacks the work budget forced,
//!    and the time per unit of work (multiply-adds, as counted by the
//!    budget).
//! 2. **Criterion** throughput per case (frames per second; the share of one
//!    core is `48 000 ÷ measured frames/s`).
use std::hint::black_box;
use std::time::Instant;

use criterion::{Criterion, Throughput};
use strom_dsp::declipper::{Declipper, DeclipperLatency, DeclipperParams};
use strom_dsp::Processor;

#[path = "../tests/declipper_support/mod.rs"]
mod support;

const RATE: f32 = 48_000.0;
const FRAMES: usize = 480;
/// Timings per buffer; the fastest counts.
const TRIALS: usize = 5;

fn stereo(l: &[f64], r: &[f64]) -> Vec<f32> {
    l.iter()
        .zip(r)
        .flat_map(|(&a, &b)| [a as f32, b as f32])
        .collect()
}

/// Square-ish wave whose plateaus are `run` frames long, with one-sample
/// edges: every half period is a clipped run of the longest repairable
/// length, as densely as the detector can report them.
fn adversarial(secs: f64, run: usize) -> Vec<f64> {
    let n = (secs * f64::from(RATE)) as usize;
    let period = 2 * (run + 2);
    (0..n)
        .map(|i| {
            let k = i % period;
            let half = period / 2;
            let (k, s) = if k < half { (k, 1.0) } else { (k - half, -1.0) };
            let v = if k == 0 || k == half - 1 { 0.5 } else { 0.9 };
            s * v
        })
        .collect()
}

fn cases() -> Vec<(&'static str, Vec<f32>)> {
    use support::*;
    let secs = 8.0;
    let tones = noise_modulated_tones(secs, 0xbe_0001);
    let speech = speech_like(secs, 0xbe_0002, 140.0);
    let music = music_like(secs, 0xbe_0003);
    let scaled = |x: &[f64]| x.iter().map(|v| v * 0.9).collect::<Vec<f64>>();
    let mut d = Declipper::new();
    d.configure(RATE, 1);
    let adv = adversarial(secs, d.max_run_frames());
    vec![
        (
            "declipper_clean_stereo",
            stereo(&scaled(&tones), &scaled(&speech)),
        ),
        (
            "declipper_speech_music_6db_stereo",
            stereo(&clip(&speech, 6.0).0, &clip(&music, 6.0).0),
        ),
        (
            "declipper_speech_music_10db_stereo",
            stereo(&clip(&speech, 10.0).0, &clip(&music, 10.0).0),
        ),
        (
            "declipper_tones_10db_stereo",
            stereo(&clip(&tones, 10.0).0, &clip(&tones, 10.0).0),
        ),
        ("declipper_adversarial_stereo", stereo(&adv, &adv)),
    ]
}

fn per_buffer(cases: &[(&'static str, Vec<f32>)]) {
    println!(
        "{:<38} {:>7} {:>9} {:>9} {:>10} {:>10}",
        "case (48 kHz stereo, 10 ms buffers)",
        "latency",
        "mean %",
        "worst %",
        "fallbacks",
        "ns/unit"
    );
    for mode in [
        DeclipperLatency::Low,
        DeclipperLatency::Normal,
        DeclipperLatency::High,
    ] {
        for (name, src) in cases {
            let mut d = Declipper::new();
            d.set_params(&DeclipperParams {
                latency: mode,
                ..Default::default()
            });
            d.configure(RATE, 2);
            let mut buf = vec![0.0f32; FRAMES * 2];
            let mut trial = vec![0.0f32; FRAMES * 2];
            let (mut total, mut worst) = (0.0f64, 0.0f64);
            let mut count = 0usize;
            // Two passes: the first warms the caches and is not counted. In
            // the second, each buffer is timed on copies of the declipper
            // in the state before it, and the fastest of `TRIALS` counts:
            // that is the buffer's own cost, without preemption by other
            // processes on the machine.
            for pass in 0..2 {
                for chunk in src.as_chunks::<{ FRAMES * 2 }>().0 {
                    if pass == 1 {
                        let mut best = f64::INFINITY;
                        for _ in 0..TRIALS {
                            let mut copy = d.clone();
                            trial.copy_from_slice(chunk);
                            let t = Instant::now();
                            copy.process(black_box(&mut trial));
                            best = best.min(t.elapsed().as_secs_f64());
                        }
                        total += best;
                        worst = worst.max(best);
                        count += 1;
                    }
                    buf.copy_from_slice(chunk);
                    d.process(black_box(&mut buf));
                }
                if pass == 0 {
                    let _ = d.take_stats();
                }
            }
            let s = d.take_stats();
            let dur = FRAMES as f64 / f64::from(RATE);
            let ns_unit = if s.work > 0.0 {
                total * 1e9 / s.work
            } else {
                0.0
            };
            println!(
                "{:<38} {:>7} {:>9.3} {:>9.2} {:>10} {:>10.3}",
                name,
                format!("{mode:?}"),
                100.0 * total / count as f64 / dur,
                100.0 * worst / dur,
                s.fallbacks,
                ns_unit
            );
        }
    }
}

fn main() {
    let cases = cases();
    per_buffer(&cases);
    let mut c = Criterion::default().configure_from_args();
    let mut group = c.benchmark_group("declipper");
    group.throughput(Throughput::Elements(FRAMES as u64));
    for (name, src) in &cases {
        let mut d = Declipper::new();
        d.configure(RATE, 2);
        let mut buf = vec![0.0f32; FRAMES * 2];
        let mut pos = 0;
        group.bench_function(*name, |b| {
            b.iter(|| {
                let len = buf.len();
                if pos + len > src.len() {
                    pos = 0;
                }
                buf.copy_from_slice(&src[pos..pos + len]);
                pos += len;
                d.process(black_box(&mut buf));
                black_box(&buf);
            })
        });
    }
    group.finish();
    c.final_summary();
}
