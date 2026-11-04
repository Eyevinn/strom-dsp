//! RealtimeSanitizer (RTSan) check of the spike guard, modelled on
//! `tests/rtsan_broadcast.rs`.
//!
//! Many buffers of varying size (including more than one internal chunk),
//! loud clicks over noise so events and gain dips happen, with a parameter
//! change before every buffer, while `set_params`, `process`,
//! `take_events` and (now and then) `reset` execute inside a real-time
//! scope. Any allocation, lock, sleep, system call or other non-real-time
//! safe call in that scope aborts the test.
//!
//! ```text
//! RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test suite rtsan_spike_guard::
//! ```

#![cfg(feature = "rtsan")]

use rtsan_standalone::ScopedSanitizeRealtime;
use strom_dsp::spike_guard::{SpikeGuard, SpikeGuardParams};
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
}

#[test]
fn spike_guard_is_realtime_safe() {
    let max = *SIZES.iter().max().unwrap();
    for ((rate, ch), lookahead_ms) in CONFIGS.into_iter().zip([0.5, 0.0, 2.0]) {
        let mut g = SpikeGuard::new();
        g.set_lookahead_ms(lookahead_ms);
        g.configure(rate, ch);
        let mut rng = Rng(0x2468_ace1);
        let signal: Vec<f32> = (0..max * ch)
            .map(|i| {
                let click = if i % (397 * ch) == 0 { 8.0 } else { 0.0 };
                0.1 * (rng.unit() * 2.0 - 1.0) + click
            })
            .collect();
        let mut buf = vec![0.0f32; max * ch];
        let mut events = 0u64;
        for i in 0..BUFFERS {
            let len = SIZES[i % SIZES.len()] * ch;
            buf[..len].copy_from_slice(&signal[..len]);
            let p = SpikeGuardParams {
                enabled: i % 13 != 5,
                threshold_db: 6.0 + 24.0 * rng.unit(),
            };
            {
                let _rt = ScopedSanitizeRealtime::default();
                g.set_params(&p);
                g.process(&mut buf[..len]);
                events += g.take_events();
                if i % 97 == 96 {
                    g.reset();
                }
            }
            assert!(buf[..len].iter().all(|v| v.is_finite()));
        }
        assert!(events > 0, "no events at {rate} Hz");
    }
}
