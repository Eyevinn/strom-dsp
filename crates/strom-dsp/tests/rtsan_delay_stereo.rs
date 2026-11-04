//! RealtimeSanitizer (RTSan) checks of the delay and stereo processors, modelled
//! on `tests/rtsan.rs`.
//!
//! Both processors run through many buffers of varying size while their
//! parameters change before every buffer (delays and offsets that start and
//! queue crossfades, modes, polarity, balance and width that start matrix
//! ramps). `set_params`, `process` and (now and then) `reset` run inside a
//! real-time scope; any allocation, lock or blocking call there aborts.
//!
//! ```text
//! RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test suite rtsan_delay_stereo::
//! ```
//!
//! Setup (`new`, `configure`, signal generation) happens outside the scope.

#![cfg(feature = "rtsan")]

use rtsan_standalone::ScopedSanitizeRealtime;
use strom_dsp::delay::{Delay, DelayParams, MAX_OFFSET_CHANNELS};
use strom_dsp::stereo::{MonoLaw, Stereo, StereoMode, StereoParams};
use strom_dsp::Processor;

const BUFFERS: usize = 600;
const SIZES: [usize; 7] = [1024, 1, 480, 7, 64, 1023, 256];

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

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next_u32() >> 8) as f32 / (1u32 << 24) as f32)
    }
}

fn signal(frames: usize, channels: usize, seed: u32) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..frames * channels)
        .map(|_| rng.range(-1.0, 1.0))
        .collect()
}

fn drive<P: Processor>(
    rate: f32,
    channels: usize,
    proc: &mut P,
    mut change: impl FnMut(&mut P, &mut Rng, usize),
) {
    proc.configure(rate, channels);
    let max_frames = *SIZES.iter().max().unwrap();
    let input = signal(max_frames * 8, channels, 0x1234_5678);
    let mut buf = vec![0.0f32; max_frames * channels];
    let mut rng = Rng(0x9e37_79b9);
    let mut pos = 0usize;
    for i in 0..BUFFERS {
        let frames = SIZES[i % SIZES.len()];
        if (pos + frames) * channels > input.len() {
            pos = 0;
        }
        let block = &mut buf[..frames * channels];

        let _rt = ScopedSanitizeRealtime::default();
        block.copy_from_slice(&input[pos * channels..(pos + frames) * channels]);
        change(proc, &mut rng, i);
        proc.process(block);
        if i % 97 == 96 {
            proc.reset();
        }
        pos += frames;
        drop(_rt);

        assert!(block.iter().all(|x| x.is_finite()), "non-finite output");
    }
}

#[test]
fn delay_is_realtime_safe() {
    for (rate, channels) in [(48_000.0f32, 2usize), (44_100.0, 1), (96_000.0, 8)] {
        let mut d = Delay::new();
        let mut p = DelayParams::default();
        drive(rate, channels, &mut d, |d, rng, i| {
            p.enabled = i % 17 != 0;
            p.delay_ms = rng.range(0.0, 2000.0);
            let mut offs = [0.0f32; MAX_OFFSET_CHANNELS];
            for o in offs.iter_mut().take(channels + 1) {
                *o = rng.range(-50.0, 50.0);
            }
            p.set_offsets(&offs[..(i % (channels + 2))]);
            d.set_params(&p);
        });
    }
}

#[test]
fn stereo_is_realtime_safe() {
    let modes = [
        StereoMode::Stereo,
        StereoMode::Mono,
        StereoMode::LeftToBoth,
        StereoMode::RightToBoth,
        StereoMode::Swap,
        StereoMode::MsEncode,
        StereoMode::MsDecode,
    ];
    for rate in [48_000.0f32, 44_100.0, 192_000.0] {
        let mut s = Stereo::new();
        drive(rate, 2, &mut s, |s, rng, i| {
            let p = StereoParams {
                enabled: i % 13 != 0,
                mode: modes[(rng.next_u32() % 7) as usize],
                mono_law: if i % 2 == 0 {
                    MonoLaw::Minus3Db
                } else {
                    MonoLaw::Minus6Db
                },
                invert_left: rng.next_u32() % 2 == 0,
                invert_right: rng.next_u32() % 2 == 0,
                balance: rng.range(-1.0, 1.0),
                width: rng.range(0.0, 2.0),
            };
            s.set_params(&p);
        });
    }
}
