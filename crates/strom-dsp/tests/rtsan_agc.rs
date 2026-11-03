//! RealtimeSanitizer (RTSan) checks of the loudness detector and the AGC.
//!
//! `process`, every reading, `set_params`, layout changes and `reset` run
//! inside a real-time scope over many buffers of varying size, at several
//! rates, channel counts and parameter sets. Any allocation, lock, sleep or
//! blocking system call aborts with a stack trace. Setup (`new`,
//! `configure`, signal generation) happens outside the scope.
//!
//! ```text
//! RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test rtsan_agc
//! ```
//!
//! The canary that proves the sanitizer is active lives in `tests/rtsan.rs`.

#![cfg(feature = "rtsan")]

use rtsan_standalone::ScopedSanitizeRealtime;
use strom_dsp::agc::{Agc, AgcParams};
use strom_dsp::meter::{ChannelLayout, ChannelRole, LoudnessDetector};
use strom_dsp::Processor;

const BUFFERS: usize = 3000;
const SIZES: [usize; 7] = [1024, 1, 480, 7, 64, 1023, 256];

/// Deterministic test signal: noise bursts, pauses and sines at varying levels.
fn signal(frames: usize, channels: usize, fs: f32) -> Vec<f32> {
    let mut x = 0x1234_5678u32;
    let mut out = Vec::with_capacity(frames * channels);
    for n in 0..frames {
        let level = [0.0f32, 0.001, 0.1, 0.7, 1.2, 0.0][(n / 3000) % 6];
        let s = (2.0 * std::f32::consts::PI * 997.0 * n as f32 / fs).sin();
        for c in 0..channels {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let noise = (x >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
            let v = if c % 2 == 0 { s } else { noise };
            out.push(level * v);
        }
    }
    out
}

fn param_sets() -> [AgcParams; 3] {
    [
        AgcParams::default(),
        AgcParams {
            target_lufs: -5.0,
            max_gain_db: 30.0,
            rate_up_db_per_s: 20.0,
            rate_down_db_per_s: 40.0,
            gate_lufs: -80.0,
            window_ms: 400.0,
            ..AgcParams::default()
        },
        AgcParams {
            target_lufs: -40.0,
            max_cut_db: 30.0,
            rate_up_db_per_s: 0.1,
            rate_down_db_per_s: 0.1,
            gate_lufs: -20.0,
            window_ms: 10_000.0,
            ..AgcParams::default()
        },
    ]
}

#[test]
fn loudness_detector_is_realtime_safe() {
    let configs = [
        (48_000.0, 2, ChannelLayout::Stereo),
        (44_100.0, 1, ChannelLayout::Mono),
        (96_000.0, 6, ChannelLayout::Surround51),
        (48_000.0, 9, ChannelLayout::Unknown),
        (8_000.0, 2, ChannelLayout::Stereo),
        (192_000.0, 2, ChannelLayout::Stereo),
    ];
    let max = *SIZES.iter().max().unwrap();
    for (rate, channels, layout) in configs {
        let mut d = LoudnessDetector::new();
        d.configure(rate, channels, layout);
        let input = signal(max * 8, channels, rate);
        let roles = vec![ChannelRole::Front; channels];
        let mut pos = 0;
        let mut sink = 0.0f64;
        for i in 0..BUFFERS {
            let len = SIZES[i % SIZES.len()] * channels;
            if pos + len > input.len() {
                pos = 0;
            }
            let block = &input[pos..pos + len];

            let _rt = ScopedSanitizeRealtime::default();
            sink += d.process(block) as f64;
            sink += d.momentary_lufs() + d.short_term_lufs() + d.recent_mean_square(10);
            if i % 500 == 0 {
                d.set_layout(layout);
                let _ = d.set_channel_roles(&roles);
            }
            if i % 1000 == 999 {
                d.reset();
            }
            pos += len;
        }
        assert!(sink.is_finite());
    }
}

#[test]
fn agc_is_realtime_safe() {
    let configs = [
        (48_000.0, 2, ChannelLayout::Stereo),
        (44_100.0, 1, ChannelLayout::Mono),
        (96_000.0, 6, ChannelLayout::Surround51),
        (8_000.0, 3, ChannelLayout::Unknown),
        (192_000.0, 2, ChannelLayout::Stereo),
    ];
    let max = *SIZES.iter().max().unwrap();
    let sets = param_sets();
    for (rate, channels, layout) in configs {
        let mut a = Agc::new();
        a.configure_with_layout(rate, channels, layout);
        let input = signal(max * 8, channels, rate);
        let mut buf = vec![0.0f32; max * channels];
        let mut pos = 0;
        let mut sink = 0.0f64;
        for i in 0..BUFFERS {
            let len = SIZES[i % SIZES.len()] * channels;
            if pos + len > input.len() {
                pos = 0;
            }
            buf[..len].copy_from_slice(&input[pos..pos + len]);

            let _rt = ScopedSanitizeRealtime::default();
            a.process(&mut buf[..len]);
            sink += a.gain_db() + a.target_gain_db() + a.measured_lufs();
            sink += a.detector().short_term_lufs();
            if i % 200 == 0 {
                a.set_params(&sets[(i / 200) % sets.len()]);
                a.set_channel_layout(layout);
            }
            if i % 1000 == 999 {
                a.reset();
            }
            pos += len;
        }
        assert!(sink.is_finite());
    }
}
