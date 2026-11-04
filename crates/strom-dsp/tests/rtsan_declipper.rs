//! RealtimeSanitizer (RTSan) check of the declipper, modelled on
//! `tests/rtsan_spike_guard.rs`.
//!
//! Many buffers of varying size, of a tone mix hard-clipped hard enough
//! that runs are rebuilt, the work budget runs out and fallbacks and runs
//! too long happen, with a parameter change before every buffer, while
//! `set_params`, `process`, `take_stats` and (now and then) `reset`
//! execute inside a real-time scope, in every latency setting. Any
//! allocation, lock, sleep, system call or other non-real-time safe call in
//! that scope aborts the test.
//!
//! ```text
//! RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test rtsan_declipper
//! ```

#![cfg(feature = "rtsan")]

use rtsan_standalone::ScopedSanitizeRealtime;
use strom_dsp::declipper::{Declipper, DeclipperLatency, DeclipperParams};
use strom_dsp::Processor;

const BUFFERS: usize = 300;
const SIZES: [usize; 7] = [1024, 1, 480, 7, 3000, 1023, 256];
const CONFIGS: [(f32, usize, DeclipperLatency); 3] = [
    (48_000.0, 2, DeclipperLatency::Normal),
    (44_100.0, 1, DeclipperLatency::Low),
    (96_000.0, 3, DeclipperLatency::High),
];

#[test]
fn declipper_is_realtime_safe() {
    let max = *SIZES.iter().max().unwrap();
    for (rate, ch, latency) in CONFIGS {
        let mut d = Declipper::new();
        d.set_params(&DeclipperParams {
            latency,
            ..Default::default()
        });
        d.configure(rate, ch);
        // Three tones, clipped about 8 dB below their peak, plus a slow
        // square (long flat runs) on the last channel.
        let frames = max * 4;
        let signal: Vec<f32> = (0..frames * ch)
            .map(|i| {
                let (n, c) = ((i / ch) as f32, i % ch);
                if c == ch - 1 && ch > 1 {
                    return if (n as usize / 400).is_multiple_of(2) {
                        0.6
                    } else {
                        -0.6
                    };
                }
                let t = n / rate;
                let x = 0.5 * (2.0 * std::f32::consts::PI * 210.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 1130.0 * t + c as f32).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 3170.0 * t).sin();
                x.clamp(-0.4, 0.4)
            })
            .collect();
        let mut buf = vec![0.0f32; max * ch];
        let mut pos = 0usize;
        let mut repaired = 0u64;
        let mut fallbacks = 0u64;
        for i in 0..BUFFERS {
            let len = SIZES[i % SIZES.len()] * ch;
            if pos + len > signal.len() {
                pos = 0;
            }
            buf[..len].copy_from_slice(&signal[pos..pos + len]);
            pos += len;
            let p = DeclipperParams {
                latency,
                enabled: i % 17 != 9,
                max_overshoot_db: (i % 25) as f32,
                min_level_db: -20.0 - (i % 3) as f32,
                ..Default::default()
            };
            let stats = {
                let _rt = ScopedSanitizeRealtime::default();
                d.set_params(&p);
                d.process(&mut buf[..len]);
                if i % 97 == 96 {
                    d.reset();
                }
                d.take_stats()
            };
            repaired += stats.runs_repaired;
            fallbacks += stats.fallbacks;
            assert!(buf[..len].iter().all(|v| v.is_finite()));
        }
        assert!(repaired > 0, "nothing repaired at {rate} Hz");
        assert!(fallbacks > 0, "no fallbacks at {rate} Hz");
    }
}
