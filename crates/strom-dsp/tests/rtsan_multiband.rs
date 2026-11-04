//! RealtimeSanitizer (RTSan) check of the multiband compressor, modelled on
//! `tests/rtsan_filter.rs`.
//!
//! Many buffers of varying size (including more than one internal chunk),
//! with a parameter change before every buffer (drive, thresholds, ratios,
//! times, makeup, knee, detector, crossover frequencies and band count),
//! while `set_params`, `process`, the meter reads and (now and then) `reset`
//! execute inside a real-time scope. Any allocation, lock, sleep, system
//! call or other non-real-time-safe call in that scope aborts the test.
//!
//! ```text
//! RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test suite rtsan_multiband::
//! ```
//!
//! Setup (`new`, `configure`, buffers, signal generation) happens outside the
//! scope: `configure` is the one place that allocates.

#![cfg(feature = "rtsan")]

use rtsan_standalone::ScopedSanitizeRealtime;
use strom_dsp::compressor::CompressorDetector;
use strom_dsp::multiband::{MultibandCompressor, MultibandParams, MAX_BANDS};
use strom_dsp::Processor;

const BUFFERS: usize = 400;
const SIZES: [usize; 7] = [1024, 1, 480, 7, 3000, 1023, 256];
const CONFIGS: [(f32, usize); 3] = [(48_000.0, 2), (44_100.0, 1), (96_000.0, 6)];

/// Small deterministic PRNG (xorshift32).
struct Rng(u32);

impl Rng {
    fn unit(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x >> 8) as f32 / (1u32 << 24) as f32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

#[test]
fn multiband_is_realtime_safe() {
    let max = *SIZES.iter().max().unwrap();
    for (rate, ch) in CONFIGS {
        let mut m = MultibandCompressor::new();
        m.configure(rate, ch);
        let mut rng = Rng(0x1234_5678);
        let signal: Vec<f32> = (0..max * ch).map(|_| rng.range(-1.0, 1.0)).collect();
        let mut buf = vec![0.0f32; max * ch];
        let mut p = MultibandParams::default();
        let mut meter = 0.0f64;
        for i in 0..BUFFERS {
            let len = SIZES[i % SIZES.len()] * ch;
            buf[..len].copy_from_slice(&signal[..len]);
            p.num_bands = if i % 50 == 0 {
                1 + (i / 50) % MAX_BANDS
            } else {
                p.num_bands
            };
            p.drive_db = rng.range(-12.0, 12.0);
            p.detector = if i % 7 == 0 {
                CompressorDetector::Rms
            } else {
                CompressorDetector::Peak
            };
            p.frequencies_hz[i % 4] = rng.range(20.0, 20_000.0);
            for b in &mut p.bands {
                b.threshold_db = rng.range(-60.0, 0.0);
                b.ratio = rng.range(1.0, 20.0);
                b.attack_ms = rng.range(0.1, 500.0);
                b.release_ms = rng.range(5.0, 5000.0);
                b.makeup_db = rng.range(-12.0, 24.0);
                b.knee_db = rng.range(0.0, 24.0);
            }
            {
                let _rt = ScopedSanitizeRealtime::default();
                m.set_params(&p);
                m.process(&mut buf[..len]);
                meter += m.gain_reduction_db().iter().sum::<f64>();
                if i % 37 == 36 {
                    m.reset();
                }
            }
            assert!(buf[..len].iter().all(|v| v.is_finite()));
        }
        assert!(meter.is_finite());
    }
}
