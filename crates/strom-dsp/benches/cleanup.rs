//! Criterion benchmarks of the input cleanup processor at 48 kHz stereo with
//! 480-frame (10 ms) buffers. Throughput is in frames, so the share of one
//! core is `48 000 frames/s ÷ measured frames/s`.
use std::f64::consts::PI;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use strom_dsp::Processor;

const FS: f32 = 48_000.0;
const FRAMES: usize = 480;

/// 2 s of stereo material with DC, 50 Hz hum and harmonics, and a tone.
fn hummy_programme() -> Vec<f32> {
    let n = 2 * FS as usize;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f64 / FS as f64;
        for c in 0..2 {
            let w = 2.0 * PI * 50.0 * t;
            let v = 0.05
                + 0.1 * w.sin()
                + 0.03 * (3.0 * w).sin()
                + 0.3 * (2.0 * PI * (440.0 + 30.0 * c as f64) * t).sin();
            out.push(v as f32);
        }
    }
    out
}

fn bench_cleanup(c: &mut Criterion) {
    use strom_dsp::cleanup::{Cleanup, CleanupParams, HumFrequency};
    let src = hummy_programme();
    let full = CleanupParams {
        dc_block: true,
        highpass_hz: 80.0,
        highpass_order: 4,
        hum: HumFrequency::Hz50,
        hum_harmonics: 10,
        hum_q: 10.0,
    };
    let cases = [
        ("cleanup_off_stereo", CleanupParams::default()),
        (
            "cleanup_dc_hp4_stereo",
            CleanupParams {
                hum: HumFrequency::Off,
                ..full
            },
        ),
        (
            "cleanup_hum10_stereo",
            CleanupParams {
                dc_block: false,
                highpass_hz: 0.0,
                ..full
            },
        ),
        ("cleanup_all_stereo", full),
    ];
    let mut group = c.benchmark_group("channel_tools");
    group.throughput(Throughput::Elements(FRAMES as u64));
    for (name, p) in cases {
        let mut cl = Cleanup::new();
        cl.set_params(&p);
        cl.configure(FS, 2);
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
                cl.process(black_box(&mut buf));
                black_box(&buf);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_cleanup);
criterion_main!(benches);
