//! RealtimeSanitizer (RTSan) check of the broadcast chain, modelled on
//! `tests/rtsan_multiband.rs`.
//!
//! Many buffers of varying size (including more than one internal chunk),
//! with a parameter change before every buffer (gains, spike guard, AGC,
//! multiband, limiter, every bypass and `enabled` toggle), while `set_params`,
//! `process`, `take_meters`, `reset_integrated` and (now and then) `reset`
//! execute inside a real-time scope. Any allocation, lock, sleep, system
//! call or other non-real-time-safe call in that scope aborts the test.
//!
//! ```text
//! RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test rtsan_broadcast
//! ```
//!
//! Setup (`new`, `configure`, buffers, signal generation) happens outside the
//! scope: `configure` is the one place that allocates.

#![cfg(feature = "rtsan")]

use rtsan_standalone::ScopedSanitizeRealtime;
use strom_dsp::broadcast::{BroadcastParams, BroadcastProcessor};
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
fn broadcast_is_realtime_safe() {
    let max = *SIZES.iter().max().unwrap();
    for (rate, ch) in CONFIGS {
        let mut bp = BroadcastProcessor::new();
        bp.configure(rate, ch);
        let mut rng = Rng(0x1234_5678);
        let signal: Vec<f32> = (0..max * ch).map(|_| rng.range(-1.0, 1.0)).collect();
        let mut buf = vec![0.0f32; max * ch];
        let mut p = BroadcastParams::default();
        let mut sum = 0.0f64;
        for i in 0..BUFFERS {
            let len = SIZES[i % SIZES.len()] * ch;
            buf[..len].copy_from_slice(&signal[..len]);
            p.enabled = i % 11 != 3;
            p.spike_guard_enabled = i % 6 != 0;
            p.spike_threshold_db = rng.range(6.0, 30.0);
            p.agc_enabled = i % 5 != 0;
            p.multiband_enabled = i % 7 != 0;
            p.limiter_enabled = i % 9 != 0;
            p.input_gain_db = rng.range(-24.0, 24.0);
            p.target_loudness_lufs = rng.range(-40.0, -5.0);
            p.true_peak_ceiling_dbtp = rng.range(-12.0, 0.0);
            p.agc_max_gain_db = rng.range(0.0, 30.0);
            p.agc_max_cut_db = rng.range(0.0, 30.0);
            p.agc_rate_up_db_s = rng.range(0.1, 20.0);
            p.agc_rate_down_db_s = rng.range(0.1, 40.0);
            p.agc_gate_lufs = rng.range(-80.0, -20.0);
            p.multiband_drive_db = rng.range(-12.0, 12.0);
            p.crossover_hz[i % 4] = rng.range(20.0, 20_000.0);
            for b in &mut p.bands {
                b.threshold_db = rng.range(-60.0, 0.0);
                b.ratio = rng.range(1.0, 20.0);
                b.attack_ms = rng.range(0.1, 500.0);
                b.release_ms = rng.range(5.0, 5000.0);
                b.makeup_db = rng.range(-12.0, 24.0);
            }
            p.limiter_release_ms = rng.range(1.0, 2000.0);
            // Structural: accepted at any time, applied at the next configure.
            p.num_bands = 1 + i % 5;
            p.limiter_lookahead_ms = rng.range(0.5, 20.0);
            {
                let _rt = ScopedSanitizeRealtime::default();
                bp.set_params(&p);
                bp.process(&mut buf[..len]);
                let m = bp.take_meters();
                sum += m.output_short_term_lufs + m.agc_gain_db + m.limiter_gain_reduction_db;
                if i % 23 == 22 {
                    bp.reset_integrated();
                }
                if i % 37 == 36 {
                    bp.reset();
                }
            }
            assert!(buf[..len].iter().all(|v| v.is_finite()));
        }
        assert!(sum.is_finite());
    }
}
