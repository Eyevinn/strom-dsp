//! RealtimeSanitizer (RTSan) checks of the Butterworth filter and the
//! crossover, modelled on `tests/rtsan.rs`.
//!
//! Both run through many buffers of varying size, with a parameter change
//! (cutoff/frequency glides, order, kind and band-count changes) before every
//! buffer, while `set_params`, `process` and (now and then) `reset` execute
//! inside a real-time scope. Any allocation, lock, sleep, system call or other
//! non-real-time-safe call in that scope aborts the test with a stack trace.
//!
//! ```text
//! RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test suite rtsan_filter::
//! ```
//!
//! Setup (`new`, `configure`, buffers, signal generation) happens outside the
//! scope: `configure` is documented as the one place that allocates.

#![cfg(feature = "rtsan")]

use rtsan_standalone::ScopedSanitizeRealtime;
use strom_dsp::filter::{
    Butterworth, ButterworthParams, Crossover, CrossoverParams, FilterKind, LrOrder,
    MAX_CROSSOVER_BANDS,
};
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
fn butterworth_is_realtime_safe() {
    let mut flt = Butterworth::new();
    let max = *SIZES.iter().max().unwrap();
    for (rate, ch) in CONFIGS {
        flt.configure(rate, ch);
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
            // Every other buffer only glides the cutoff; otherwise kind and
            // order change too (crossfade path).
            let structural = i % 2 == 1;
            flt.set_params(&ButterworthParams {
                kind: if structural && rng.unit() > 0.5 {
                    FilterKind::LowPass
                } else {
                    FilterKind::HighPass
                },
                order: if structural {
                    1 + rng.next_u32() as usize % 8
                } else {
                    4
                },
                cutoff_hz: rng.range(1.0, 24_000.0),
            });
            flt.process(block);
            if i % 97 == 96 {
                flt.reset();
            }
            pos += len;
            drop(_rt);
            assert!(block.iter().all(|x| x.is_finite()), "non-finite output");
        }
    }
}

#[test]
fn crossover_is_realtime_safe() {
    let mut xo = Crossover::new();
    let max = *SIZES.iter().max().unwrap();
    let orders = [LrOrder::Lr2, LrOrder::Lr4, LrOrder::Lr8];
    for (rate, ch) in CONFIGS {
        xo.configure(rate, ch);
        let input = signal(max * ch * 4);
        let mut bands = vec![vec![0.0f32; max * ch]; MAX_CROSSOVER_BANDS];
        let mut rng = Rng(0x9e37_79b9);
        let mut pos = 0;
        for i in 0..BUFFERS {
            let len = SIZES[i % SIZES.len()] * ch;
            if pos + len > input.len() {
                pos = 0;
            }
            let _rt = ScopedSanitizeRealtime::default();
            let structural = i % 2 == 1;
            let mut f = [0.0f32; 4];
            for v in f.iter_mut() {
                *v = rng.range(1.0, 24_000.0);
            }
            xo.set_params(&CrossoverParams {
                order: if structural {
                    orders[rng.next_u32() as usize % 3]
                } else {
                    LrOrder::Lr4
                },
                num_bands: if structural {
                    1 + rng.next_u32() as usize % MAX_CROSSOVER_BANDS
                } else {
                    5
                },
                frequencies_hz: f,
            });
            xo.process(&input[pos..pos + len], &mut bands);
            if i % 97 == 96 {
                xo.reset();
            }
            pos += len;
            drop(_rt);
            for b in &bands[..xo.num_bands()] {
                assert!(b[..len].iter().all(|x| x.is_finite()), "non-finite output");
            }
        }
    }
}
