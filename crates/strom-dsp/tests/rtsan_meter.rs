//! RealtimeSanitizer (RTSan) checks of the meters, including the wrapped
//! `ebur128` crate.
//!
//! `process`, every reading, `clear_interval_peaks`, `reset` and parameter
//! setters run inside a real-time scope over many buffers of varying size.
//! Any allocation, lock, sleep or blocking system call aborts with a stack
//! trace. Setup (`new`, signal generation) happens outside the scope.
//!
//! ```text
//! RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test rtsan_meter
//! ```
//!
//! The canary that proves the sanitizer is active lives in `tests/rtsan.rs`.

#![cfg(feature = "rtsan")]

use rtsan_standalone::ScopedSanitizeRealtime;
use strom_dsp::meter::{
    BandCorrelation, ChannelLayout, CorrelationMeter, LoudnessConfig, LoudnessMeter, MonoDetector,
    MonoDetectorParams,
};

const BUFFERS: usize = 3000;
const SIZES: [usize; 7] = [1024, 1, 480, 7, 64, 1023, 256];

/// Deterministic test signal: noise bursts and sines at varying levels.
fn signal(frames: usize, channels: usize) -> Vec<f32> {
    let mut x = 0x1234_5678u32;
    let mut out = Vec::with_capacity(frames * channels);
    for n in 0..frames {
        let level = [0.0f32, 0.001, 0.1, 0.7, 1.2][(n / 3000) % 5];
        let s = (2.0 * std::f32::consts::PI * 997.0 * n as f32 / 48_000.0).sin();
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

#[test]
fn loudness_meter_is_realtime_safe() {
    let configs = [
        (48_000.0, 2, ChannelLayout::Stereo, true),
        (48_000.0, 2, ChannelLayout::Stereo, false),
        (44_100.0, 1, ChannelLayout::Mono, true),
        (96_000.0, 6, ChannelLayout::Surround51, true),
        (48_000.0, 9, ChannelLayout::Unknown, true),
        (192_000.0, 2, ChannelLayout::Stereo, true),
    ];
    let max = *SIZES.iter().max().unwrap();
    for (rate, channels, layout, true_peak) in configs {
        let mut m = LoudnessMeter::new(&LoudnessConfig {
            sample_rate: rate,
            channels,
            layout,
            true_peak,
        })
        .unwrap();
        let input = signal(max * 8, channels);
        let mut pos = 0;
        let mut sink = 0.0f64;
        for i in 0..BUFFERS {
            let len = SIZES[i % SIZES.len()] * channels;
            if pos + len > input.len() {
                pos = 0;
            }
            let block = &input[pos..pos + len];

            let _rt = ScopedSanitizeRealtime::default();
            m.process(block);
            if i % 10 == 0 {
                let r = m.reading();
                sink += r.momentary.max(-100.0) + r.short_term.max(-100.0);
                sink += r.integrated.max(-100.0) + r.loudness_range;
                sink += m.interval_true_peak(0) + m.interval_sample_peak(channels - 1);
                sink += m.true_peak(0) + m.sample_peak(0);
                m.clear_interval_peaks();
            }
            if i % 997 == 996 {
                m.reset();
            }
            drop(_rt);
            pos += len;
        }
        assert!(sink.is_finite());
    }
}

#[test]
fn correlation_and_mono_detector_are_realtime_safe() {
    let max = *SIZES.iter().max().unwrap();
    let input = signal(max * 8, 2);
    let (l, r): (Vec<f32>, Vec<f32>) = input.chunks_exact(2).map(|f| (f[0], f[1])).unzip();
    let mut m = CorrelationMeter::new(48_000.0, 300.0);
    let mut d = MonoDetector::new(48_000.0, &MonoDetectorParams::default());
    let mut pos = 0;
    let mut sink = 0.0f64;
    for i in 0..BUFFERS {
        let n = SIZES[i % SIZES.len()];
        if pos + n > l.len() {
            pos = 0;
        }
        let _rt = ScopedSanitizeRealtime::default();
        m.set_integration_ms(5.0 + (i % 100) as f32 * 10.0);
        m.process_interleaved(&input[2 * pos..2 * (pos + n)], 2);
        m.process_pair(&l[pos..pos + n], &r[pos..pos + n]);
        d.set_params(&MonoDetectorParams {
            hold_ms: (i % 7) as f32 * 100.0,
            ..MonoDetectorParams::default()
        });
        d.process_interleaved(&input[2 * pos..2 * (pos + n)], 2);
        d.process_pair(&l[pos..pos + n], &l[pos..pos + n]);
        sink += m.correlation() + m.side_mid_ratio_db() + m.mean_square();
        sink += d.state() as u8 as f64;
        if i % 97 == 96 {
            m.reset();
            d.reset();
        }
        drop(_rt);
        pos += n;
    }
    assert!(sink.is_finite());
}

#[test]
fn band_correlation_is_realtime_safe() {
    let max = *SIZES.iter().max().unwrap();
    let input = signal(max * 8, 3);
    let (l, r): (Vec<f32>, Vec<f32>) = input.chunks_exact(3).map(|f| (f[0], f[1])).unzip();
    let mut m = BandCorrelation::new(48_000.0, 300.0);
    let mut pos = 0;
    let mut sink = 0.0f64;
    for i in 0..BUFFERS {
        let n = SIZES[i % SIZES.len()];
        if pos + n > l.len() {
            pos = 0;
        }
        let _rt = ScopedSanitizeRealtime::default();
        m.set_integration_ms(5.0 + (i % 100) as f32 * 10.0);
        if i % 50 == 0 {
            m.set_split_frequencies([200.0 + (i % 7) as f32 * 20.0, 4000.0]);
        }
        m.process_interleaved(&input[3 * pos..3 * (pos + n)], 3);
        m.process_pair(&l[pos..pos + n], &r[pos..pos + n]);
        sink += m.correlations().iter().sum::<f64>() + m.correlation(2);
        if i % 97 == 96 {
            m.reset();
        }
        drop(_rt);
        pos += n;
    }
    assert!(sink.is_finite());
}
