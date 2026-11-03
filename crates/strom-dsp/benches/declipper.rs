//! Criterion benchmarks of the declipper at 48 kHz stereo with 480-frame
//! (10 ms) buffers: clean material (detection only) and material clipped
//! 6 and 10 dB below its peak (detection and repair). Throughput is in
//! frames, so the share of one core is `48 000 frames/s ÷ measured frames/s`.
//! The source loops over 4 s of synthetic material, so every buffer of the
//! clipped cases contains repairs at the rate the material produces them.
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use strom_dsp::declipper::Declipper;
use strom_dsp::Processor;

#[path = "../tests/declipper_support/mod.rs"]
mod support;

const RATE: f32 = 48_000.0;
const FRAMES: usize = 480;

fn stereo(l: &[f64], r: &[f64]) -> Vec<f32> {
    l.iter()
        .zip(r)
        .flat_map(|(&a, &b)| [a as f32, b as f32])
        .collect()
}

fn bench_declipper(c: &mut Criterion) {
    use support::*;
    let tones = noise_modulated_tones(4.0, 0xbe_0001);
    let speech = speech_like(4.0, 0xbe_0002, 140.0);
    let music = music_like(4.0, 0xbe_0003);
    let scaled = |x: &[f64]| x.iter().map(|v| v * 0.9).collect::<Vec<f64>>();
    let cases: Vec<(&str, Vec<f32>)> = vec![
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
    ];
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
}

criterion_group!(benches, bench_declipper);
criterion_main!(benches);
