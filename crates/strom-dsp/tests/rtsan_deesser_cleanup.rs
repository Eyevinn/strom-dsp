//! RealtimeSanitizer (RTSan) checks of the de-esser and the input cleanup,
//! modelled on `tests/rtsan_filter.rs`.
//!
//! Both run through many buffers of varying size, with new parameters before
//! every buffer (de-esser: frequency glides, threshold/max-reduction glides,
//! mode changes; cleanup: stage switch crossfades, high-pass glides and order
//! crossfades, hum crossfades), while `set_params`, `process` and (now and
//! then) `reset` execute inside a real-time scope. Any allocation, lock,
//! sleep, system call or other non-real-time-safe call in that scope aborts
//! the test with a stack trace.
//!
//! ```text
//! RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test suite rtsan_deesser_cleanup::
//! ```
//!
//! Setup (`new`, `configure`, buffers, signal generation) happens outside the
//! scope: `configure` is documented as the one place that allocates.

#![cfg(feature = "rtsan")]

use rtsan_standalone::ScopedSanitizeRealtime;
use strom_dsp::cleanup::{Cleanup, CleanupParams, HumFrequency};
use strom_dsp::deesser::{Deesser, DeesserMode, DeesserParams};
use strom_dsp::Processor;

const BUFFERS: usize = 600;
const SIZES: [usize; 7] = [1024, 1, 480, 7, 64, 1023, 256];
const CONFIGS: [(f32, usize); 3] = [(48_000.0, 2), (44_100.0, 1), (192_000.0, 8)];

/// Small deterministic PRNG (xorshift32).
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

fn signal(len: usize) -> Vec<f32> {
    let mut rng = Rng(0x1234_5678);
    (0..len).map(|_| rng.range(-1.0, 1.0)).collect()
}

#[test]
fn deesser_is_realtime_safe() {
    let mut d = Deesser::new();
    let max = *SIZES.iter().max().unwrap();
    for (rate, ch) in CONFIGS {
        d.configure(rate, ch);
        let input = signal(max * ch * 4);
        let mut buf = vec![0.0f32; max * ch];
        let mut rng = Rng(0x9e37_79b9);
        let mut pos = 0;
        for i in 0..BUFFERS {
            let len = SIZES[i % SIZES.len()] * ch;
            if pos + len > input.len() {
                pos = 0;
            }
            let block = &mut buf[..len];
            let _rt = ScopedSanitizeRealtime::default();
            block.copy_from_slice(&input[pos..pos + len]);
            d.set_params(&DeesserParams {
                frequency_hz: rng.range(1_000.0, 20_000.0),
                threshold_db: rng.range(-60.0, 0.0),
                max_reduction_db: rng.range(0.0, 40.0),
                attack_ms: rng.range(0.01, 200.0),
                release_ms: rng.range(1.0, 2000.0),
                mode: if rng.unit() > 0.5 {
                    DeesserMode::Wideband
                } else {
                    DeesserMode::SplitBand
                },
            });
            d.process(block);
            let _ = d.gain_reduction_db();
            if i % 97 == 96 {
                d.reset();
            }
            pos += len;
            drop(_rt);
            assert!(block.iter().all(|x| x.is_finite()), "non-finite output");
        }
    }
}

#[test]
fn cleanup_is_realtime_safe() {
    let mut c = Cleanup::new();
    let max = *SIZES.iter().max().unwrap();
    let hums = [HumFrequency::Off, HumFrequency::Hz50, HumFrequency::Hz60];
    for (rate, ch) in CONFIGS {
        c.configure(rate, ch);
        let input = signal(max * ch * 4);
        let mut buf = vec![0.0f32; max * ch];
        let mut rng = Rng(0x9e37_79b9);
        let mut pos = 0;
        for i in 0..BUFFERS {
            let len = SIZES[i % SIZES.len()] * ch;
            if pos + len > input.len() {
                pos = 0;
            }
            let block = &mut buf[..len];
            let _rt = ScopedSanitizeRealtime::default();
            block.copy_from_slice(&input[pos..pos + len]);
            c.set_params(&CleanupParams {
                dc_block: rng.unit() > 0.5,
                highpass_hz: if rng.unit() > 0.3 {
                    rng.range(1.0, 1_000.0)
                } else {
                    0.0
                },
                highpass_order: 1 + rng.next_u32() as usize % 8,
                hum: hums[rng.next_u32() as usize % 3],
                hum_harmonics: 1 + rng.next_u32() as usize % 10,
                hum_q: rng.range(6.0, 100.0),
            });
            c.process(block);
            if i % 97 == 96 {
                c.reset();
            }
            pos += len;
            drop(_rt);
            assert!(block.iter().all(|x| x.is_finite()), "non-finite output");
        }
    }
}
