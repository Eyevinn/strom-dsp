//! Criterion benchmarks for the channel tools at 48 kHz, 480 frames (10 ms)
//! per buffer. Share of one core = time per buffer ÷ 10 ms.
//!
//! Delay (`bench_delay`):
//! - `delay_stereo_zero`: delay 0, the bit-exact path (ring write only);
//! - `delay_stereo_20ms`: 20 ms on both channels (bulk block read);
//! - `delay_8ch_offsets`: 8 channels, 20 ms plus a different offset per channel;
//! - `delay_stereo_xfade`: a delay change every buffer, so it always crossfades.
//!
//! Stereo tools (`bench_stereo`):
//! - `stereo_default`: defaults (identity, buffer untouched);
//! - `stereo_swap_invert`: swap + invert-left (bit-exact routing path);
//! - `stereo_width_balance`: width 1.4, balance 0.3 (general matrix path);
//! - `stereo_ramping`: a balance change every buffer (always ramping).

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use strom_dsp::delay::{Delay, DelayParams};
use strom_dsp::stereo::{StereoMode, StereoParams, StereoTools};
use strom_dsp::Processor;

const FS: f32 = 48_000.0;
const FRAMES: usize = 480;

fn noise(frames: usize, channels: usize) -> Vec<f32> {
    let mut s = 0x2545_f491_4f6c_dd1du64;
    (0..frames * channels)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * 0.5
        })
        .collect()
}

fn bench_delay(c: &mut Criterion) {
    let mut g = c.benchmark_group("delay");
    g.throughput(Throughput::Elements(FRAMES as u64));
    let cases: [(&str, usize, f32, &[f32], bool); 4] = [
        ("delay_stereo_zero", 2, 0.0, &[], false),
        ("delay_stereo_20ms", 2, 20.0, &[], false),
        (
            "delay_8ch_offsets",
            8,
            20.0,
            &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            false,
        ),
        ("delay_stereo_xfade", 2, 20.0, &[], true),
    ];
    for (name, ch, ms, offs, moving) in cases {
        let input = noise(FRAMES * 100, ch);
        let mut d = Delay::new();
        let mut p = DelayParams {
            delay_ms: ms,
            ..DelayParams::default()
        };
        p.set_offsets(offs);
        d.set_params(&p);
        d.configure(FS, ch);
        let mut buf = vec![0.0f32; FRAMES * ch];
        let mut k = 0usize;
        g.bench_function(name, |b| {
            b.iter(|| {
                let i = k % 100;
                k += 1;
                if moving {
                    p.delay_ms = if k.is_multiple_of(2) { 20.0 } else { 25.0 };
                    d.set_params(&p);
                }
                buf.copy_from_slice(&input[i * FRAMES * ch..(i + 1) * FRAMES * ch]);
                d.process(black_box(&mut buf));
            })
        });
    }
    g.finish();
}

fn bench_stereo(c: &mut Criterion) {
    let mut g = c.benchmark_group("stereotools");
    g.throughput(Throughput::Elements(FRAMES as u64));
    let cases: [(&str, StereoParams, bool); 4] = [
        ("stereo_default", StereoParams::default(), false),
        (
            "stereo_swap_invert",
            StereoParams {
                mode: StereoMode::Swap,
                invert_left: true,
                ..StereoParams::default()
            },
            false,
        ),
        (
            "stereo_width_balance",
            StereoParams {
                width: 1.4,
                balance: 0.3,
                ..StereoParams::default()
            },
            false,
        ),
        (
            "stereo_ramping",
            StereoParams {
                width: 1.4,
                ..StereoParams::default()
            },
            true,
        ),
    ];
    for (name, mut p, moving) in cases {
        let input = noise(FRAMES * 100, 2);
        let mut s = StereoTools::new();
        s.set_params(&p);
        s.configure(FS, 2);
        let mut buf = vec![0.0f32; FRAMES * 2];
        let mut k = 0usize;
        g.bench_function(name, |b| {
            b.iter(|| {
                let i = k % 100;
                k += 1;
                if moving {
                    p.balance = if k.is_multiple_of(2) { 0.2 } else { -0.2 };
                    s.set_params(&p);
                }
                buf.copy_from_slice(&input[i * FRAMES * 2..(i + 1) * FRAMES * 2]);
                s.process(black_box(&mut buf));
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench_delay, bench_stereo);
criterion_main!(benches);
