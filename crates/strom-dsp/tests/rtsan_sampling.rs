//! RealtimeSanitizer (RTSan) checks of the sampling building blocks
//! (oversampler and STFT), modelled on `tests/rtsan.rs`.
//!
//! Every processing call runs inside a real-time scope, over many buffers of
//! varying size (including 1 and non-powers of two). Any `malloc`/`free`, lock,
//! sleep or system call in that scope aborts the test with a stack trace. This
//! also covers the wrapped `realfft`/`rustfft` calls inside the STFT.
//!
//! ```text
//! RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test rtsan_sampling
//! ```
//!
//! Setup (`new`, `configure`, buffers, signal generation) happens outside the
//! scope.

#![cfg(feature = "rtsan")]

use rtsan_standalone::ScopedSanitizeRealtime;
use strom_dsp::oversample::{OversampleFactor, Oversampler};

/// Buffers per configuration.
const BUFFERS: usize = 400;
/// Buffer sizes in frames, cycled through.
const SIZES: [usize; 7] = [1024, 1, 480, 7, 64, 1023, 256];
/// Channel counts.
const CHANNELS: [usize; 3] = [2, 1, 8];

/// Deterministic test signal: noise bursts and a sine, up to +12 dBFS.
fn signal(frames: usize, ch: usize) -> Vec<f32> {
    let mut s = 0x1234_5678u32;
    (0..frames * ch)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let n = (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
            let level = [0.0, 0.01, 0.5, 4.0][(i / (2000 * ch)) % 4];
            level * (0.5 * n + 0.5 * (i as f32 * 0.05).sin())
        })
        .collect()
}

#[test]
fn oversampler_is_realtime_safe() {
    let max = *SIZES.iter().max().unwrap();
    for f in [
        OversampleFactor::X2,
        OversampleFactor::X4,
        OversampleFactor::X8,
    ] {
        let l = f.ratio();
        for ch in CHANNELS {
            let mut o = Oversampler::new();
            o.configure(f, ch);
            let input = signal(max * 8, ch);
            let mut up = vec![0.0f32; max * l * ch];
            let mut down = vec![0.0f32; max * ch];
            let mut peaks = vec![0.0f32; max];
            let mut pos = 0;
            for i in 0..BUFFERS {
                let n = SIZES[i % SIZES.len()];
                if (pos + n) * ch > input.len() {
                    pos = 0;
                }
                let x = &input[pos * ch..(pos + n) * ch];
                let _rt = ScopedSanitizeRealtime::default();
                o.upsample(x, &mut up[..n * l * ch]);
                o.downsample(&up[..n * l * ch], &mut down[..n * ch]);
                o.upsample_peaks(x, &mut peaks[..n]);
                let _ = o.latency_frames();
                if i % 97 == 96 {
                    o.reset();
                }
                drop(_rt);
                pos += n;
                assert!(down[..n * ch].iter().all(|v| v.is_finite()));
            }
        }
    }
}
