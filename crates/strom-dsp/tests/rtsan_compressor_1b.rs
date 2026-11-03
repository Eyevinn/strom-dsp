//! RealtimeSanitizer check of the compressor's detector mode, rms-window and
//! auto-makeup setters together with `process`, `process_with_key` and
//! `reset`, all inside a real-time scope (see `tests/rtsan.rs`).
//!
//! ```text
//! RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test rtsan_compressor_1b
//! ```

#![cfg(feature = "rtsan")]

use rtsan_standalone::ScopedSanitizeRealtime;
use strom_dsp::compressor::CompressorDetector;
use strom_dsp::{Compressor, CompressorParams, Processor};

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
fn compressor_1b_is_realtime_safe() {
    const SIZES: [usize; 6] = [1024, 1, 480, 7, 64, 1023];
    for (rate, ch) in [(48_000.0f32, 2usize), (44_100.0, 1), (96_000.0, 8)] {
        let mut c = Compressor::new();
        c.configure(rate, ch);
        c.set_key_channels(ch);
        let mut rng = Rng(0x1234_5678);
        let input: Vec<f32> = (0..1024 * ch * 4)
            .map(|i| {
                let level = [0.0f32, 0.01, 0.3, 2.0][(i / (2000 * ch)) % 4];
                level * rng.range(-1.0, 1.0)
            })
            .collect();
        let key = input.clone();
        let mut buf = vec![0.0f32; 1024 * ch];
        let mut pos = 0;
        for i in 0..600 {
            let len = SIZES[i % SIZES.len()] * ch;
            if pos + len > input.len() {
                pos = 0;
            }
            let block = &mut buf[..len];
            let _rt = ScopedSanitizeRealtime::default();
            block.copy_from_slice(&input[pos..pos + len]);
            c.set_detector(if i % 3 == 0 {
                CompressorDetector::Rms
            } else {
                CompressorDetector::Peak
            });
            c.set_rms_window(rng.range(0.5, 120.0));
            c.set_auto_makeup(i % 4 < 2);
            c.set_params(&CompressorParams {
                threshold: rng.range(0.0, 1.0),
                ratio: rng.range(1.0, 100.0),
                attack_ms: rng.range(0.01, 10_000.0),
                release_ms: rng.range(0.01, 10_000.0),
                makeup_gain: rng.range(0.0, 100.0),
                knee: rng.range(0.0, 1.0),
            });
            if i % 2 == 0 {
                c.process(block);
            } else {
                c.process_with_key(block, &key[pos..pos + len], ch);
            }
            if i % 97 == 96 {
                c.reset();
            }
            pos += len;
            drop(_rt);
            assert!(block.iter().all(|x| !x.is_nan()));
        }
    }
}
