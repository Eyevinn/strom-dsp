//! RealtimeSanitizer (RTSan) check of the ducker, modelled on `tests/rtsan.rs`.
//!
//! The ducker runs through many buffers of varying size with keys at other
//! rates and channel counts (and no key, and mismatched keys), while every
//! parameter, the key filter and the range mode change before every buffer.
//! `set_params`, `process_with_key`, `process`, the meter getters and (now and
//! then) `reset` run inside a real-time scope; any allocation, lock or
//! blocking call there aborts the test.
//!
//! ```text
//! RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test suite rtsan_ducker::
//! ```
//!
//! Setup (`new`, `configure`, `set_key_format`, signal generation) happens
//! outside the scope: those are the documented allocating calls.

#![cfg(feature = "rtsan")]

use rtsan_standalone::ScopedSanitizeRealtime;
use strom_dsp::ducker::{Ducker, DuckerParams, RangeMode};
use strom_dsp::Processor;

const BUFFERS: usize = 600;
const SIZES: [usize; 7] = [1024, 1, 480, 7, 64, 1023, 256];
/// (programme rate, programme channels, key rate, key channels).
const CONFIGS: [(f32, usize, f32, usize); 4] = [
    (48_000.0, 2, 48_000.0, 1),
    (44_100.0, 1, 16_000.0, 2),
    (96_000.0, 8, 192_000.0, 3),
    (8_000.0, 2, 48_000.0, 2),
];

struct Rng(u32);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

/// Noise bursts from silence to +12 dBFS.
fn signal(frames: usize, channels: usize, seed: u32) -> Vec<f32> {
    let mut rng = Rng(seed);
    let levels = [0.0f32, 0.001, 0.1, 0.7, 4.0, 0.03];
    (0..frames * channels)
        .map(|i| levels[(i / channels / 1500) % levels.len()] * rng.range(-1.0, 1.0))
        .collect()
}

#[test]
fn ducker_is_realtime_safe() {
    let max_frames = *SIZES.iter().max().unwrap();
    for (rate, channels, key_rate, key_channels) in CONFIGS {
        let mut d = Ducker::new();
        d.configure(rate, channels);
        d.set_key_format(key_rate, key_channels);
        let ratio = key_rate as f64 / rate as f64;
        let input = signal(max_frames * 8, channels, 0x1234_5678);
        let key_frames = (max_frames as f64 * 8.0 * ratio) as usize + 2;
        let key = signal(key_frames, key_channels, 0x0bad_cafe);
        let mut buf = vec![0.0f32; max_frames * channels];
        let mut rng = Rng(0x9e37_79b9);
        let mut pos = 0usize;
        for i in 0..BUFFERS {
            let frames = SIZES[i % SIZES.len()];
            if (pos + frames) * channels > input.len() {
                pos = 0;
            }
            let k0 = (pos as f64 * ratio) as usize;
            let k1 = ((pos + frames) as f64 * ratio) as usize;
            let k = &key[k0 * key_channels..k1 * key_channels];
            let block = &mut buf[..frames * channels];

            let _rt = ScopedSanitizeRealtime::default();
            block.copy_from_slice(&input[pos * channels..(pos + frames) * channels]);
            d.set_params(&DuckerParams {
                threshold_db: rng.range(-80.0, 0.0),
                depth_db: rng.range(-80.0, 0.0),
                attack_ms: rng.range(0.1, 2000.0),
                hold_ms: rng.range(0.0, 5000.0),
                release_ms: rng.range(1.0, 10_000.0),
                key_highpass_hz: if i % 4 == 0 {
                    0.0
                } else {
                    rng.range(0.0, 20_000.0)
                },
                range_mode: if i % 2 == 0 {
                    RangeMode::Fixed
                } else {
                    RangeMode::Proportional
                },
            });
            match i % 9 {
                7 => d.process(block),
                8 => d.process_with_key(block, k, key_channels + 1),
                _ => d.process_with_key(block, k, key_channels),
            }
            let meters = (d.reduction_db(), d.key_active(), d.key_level_db());
            if i % 97 == 96 {
                d.reset();
            }
            pos += frames;
            drop(_rt);

            assert!(meters.0.is_finite() && meters.2.is_finite());
            assert!(block.iter().all(|x| x.is_finite()), "non-finite output");
        }
    }
}
