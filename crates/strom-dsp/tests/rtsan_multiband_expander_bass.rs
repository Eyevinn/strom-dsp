//! RealtimeSanitizer (RTSan) check of the multiband's per-band expanders
//! and its bass clipper, modelled on `tests/rtsan_multiband.rs`.
//!
//! Many buffers of varying size, with expander settings, bass-clipper
//! settings and the upper-clipper activity changed before every buffer,
//! while the setters, `process`, the meter reads and (now and then)
//! `reset` execute inside a real-time scope.
//!
//! ```text
//! RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test rtsan_multiband_expander_bass
//! ```

#![cfg(feature = "rtsan")]

use rtsan_standalone::ScopedSanitizeRealtime;
use strom_dsp::bassclip::BassClipperParams;
use strom_dsp::dynamics::ExpanderParams;
use strom_dsp::multiband::{MultibandCompressor, MultibandParams, MAX_BANDS};
use strom_dsp::Processor;

const BUFFERS: usize = 400;
const SIZES: [usize; 7] = [1024, 1, 480, 7, 3000, 1023, 256];
const CONFIGS: [(f32, usize); 3] = [(48_000.0, 2), (44_100.0, 1), (96_000.0, 6)];

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
fn expanders_and_bass_clipper_are_realtime_safe() {
    let max = *SIZES.iter().max().unwrap();
    for (rate, ch) in CONFIGS {
        let mut m = MultibandCompressor::new();
        m.set_bass_lookahead_ms(3.0);
        m.configure(rate, ch);
        let mut rng = Rng(0x2468_ace1);
        let signal: Vec<f32> = (0..max * ch)
            .map(|i| rng.range(-1.0, 1.0) * if (i / 4000) % 2 == 0 { 1.0 } else { 0.001 })
            .collect();
        let mut buf = vec![0.0f32; max * ch];
        let mut p = MultibandParams::default();
        let mut meter = 0.0f64;
        for i in 0..BUFFERS {
            let len = SIZES[i % SIZES.len()] * ch;
            buf[..len].copy_from_slice(&signal[..len]);
            if i % 50 == 0 {
                p.num_bands = 1 + (i / 50) % MAX_BANDS;
            }
            let e: [ExpanderParams; MAX_BANDS] = core::array::from_fn(|_| ExpanderParams {
                threshold_db: rng.range(-80.0, 0.0),
                ratio: rng.range(1.0, 10.0),
                floor_db: rng.range(0.0, 60.0),
                knee_db: rng.range(0.0, 24.0),
                open_ms: rng.range(0.1, 100.0),
                close_ms: rng.range(1.0, 5000.0),
                hold_ms: rng.range(0.0, 1000.0),
            });
            let b = BassClipperParams {
                enabled: i % 30 < 20,
                threshold_db: rng.range(-40.0, 0.0),
            };
            let activity = rng.range(0.0, 0.1);
            {
                let _rt = ScopedSanitizeRealtime::default();
                m.set_params(&p);
                m.set_expanders(&e);
                m.set_bass_clipper(&b);
                m.set_upper_clip_activity(activity);
                m.process(&mut buf[..len]);
                meter += m.expander_reduction_db().iter().sum::<f64>();
                meter += m.take_bass_clip_gain_reduction_db();
                if i % 37 == 36 {
                    m.reset();
                }
            }
            assert!(buf[..len].iter().all(|v| v.is_finite()));
        }
        assert!(meter.is_finite());
    }
}
