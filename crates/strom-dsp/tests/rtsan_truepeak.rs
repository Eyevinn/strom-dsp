//! RealtimeSanitizer (RTSan) check of the true-peak limiter, modelled on
//! `tests/rtsan_multiband.rs`.
//!
//! Many buffers of varying size (including more than one detector chunk),
//! with a parameter change before every buffer (enable, ceiling, release,
//! lookahead), while `set_params`, `process`, the meter reads and (now and
//! then) `reset` execute inside a real-time scope. Any allocation, lock,
//! sleep, system call or other non-real-time-safe call in that scope aborts
//! the test.
//!
//! ```text
//! RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test rtsan_truepeak
//! ```
//!
//! Setup (`new`, `configure`, buffers, signal generation) happens outside the
//! scope: `configure` is the one place that allocates.

#![cfg(feature = "rtsan")]

use rtsan_standalone::ScopedSanitizeRealtime;
use strom_dsp::truepeak::{TruePeakLimiter, TruePeakLimiterParams};
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
fn truepeak_limiter_is_realtime_safe() {
    let max = *SIZES.iter().max().unwrap();
    for (rate, ch) in CONFIGS {
        let mut l = TruePeakLimiter::new();
        l.configure(rate, ch);
        let mut rng = Rng(0x1234_5678);
        let signal: Vec<f32> = (0..max * ch).map(|_| rng.range(-4.0, 4.0)).collect();
        let mut buf = vec![0.0f32; max * ch];
        let mut meter = 0.0f64;
        for i in 0..BUFFERS {
            let len = SIZES[i % SIZES.len()] * ch;
            buf[..len].copy_from_slice(&signal[..len]);
            let p = TruePeakLimiterParams {
                enabled: i % 13 != 5,
                ceiling_dbtp: rng.range(-12.0, 0.0),
                release_ms: rng.range(1.0, 2000.0),
                lookahead_ms: rng.range(0.5, 20.0),
            };
            {
                let _rt = ScopedSanitizeRealtime::default();
                l.set_params(&p);
                l.process(&mut buf[..len]);
                meter += l.gain_reduction_db() + l.current_gain();
                meter += l.latency_frames() as f64 + l.safety_clip_count() as f64;
                if i % 37 == 36 {
                    l.reset();
                }
            }
            assert!(buf[..len].iter().all(|v| v.is_finite()));
        }
        assert!(meter.is_finite());
    }
}
