//! RealtimeSanitizer (RTSan) checks of the sidechain and expander paths,
//! modelled on `tests/rtsan.rs`.
//!
//! The gate (gate and expander modes), the compressor and a bare
//! `KeyDetector` run through many buffers of varying size with a key of a
//! different channel count, while the key filter, mode, ratio and parameters
//! change before every buffer. Those setters, `process_with_key`, `process`
//! and (now and then) `reset` run inside a real-time scope; any allocation,
//! lock or blocking call there aborts the test.
//!
//! ```text
//! RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test suite rtsan_sidechain::
//! ```
//!
//! Setup (`new`, `configure`, `set_key_channels`, signal generation) happens
//! outside the scope: those are the documented allocating calls.

#![cfg(feature = "rtsan")]

use rtsan_standalone::ScopedSanitizeRealtime;
use strom_dsp::dynamics::{KeyDetector, KeyFilter};
use strom_dsp::gate::GateMode;
use strom_dsp::{Compressor, CompressorParams, Gate, GateParams, Processor};

const BUFFERS: usize = 600;
const SIZES: [usize; 7] = [1024, 1, 480, 7, 64, 1023, 256];
/// (sample rate, programme channels, key channels).
const CONFIGS: [(f32, usize, usize); 3] = [(48_000.0, 2, 1), (44_100.0, 1, 2), (96_000.0, 8, 3)];

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

/// Noise bursts from silence to +12 dBFS.
fn signal(frames: usize, channels: usize, seed: u32) -> Vec<f32> {
    let mut rng = Rng(seed);
    let levels = [0.0f32, 0.001, 0.1, 0.7, 4.0, 0.03];
    (0..frames * channels)
        .map(|i| levels[(i / channels / 1500) % levels.len()] * rng.range(-1.0, 1.0))
        .collect()
}

fn random_filter(rng: &mut Rng) -> KeyFilter {
    match rng.next_u32() % 3 {
        0 => KeyFilter::Off,
        1 => KeyFilter::HighPass {
            frequency_hz: rng.range(10.0, 20_000.0),
            q: rng.range(0.1, 10.0),
        },
        _ => KeyFilter::BandPass {
            frequency_hz: rng.range(10.0, 20_000.0),
            q: rng.range(0.1, 10.0),
        },
    }
}

/// Keyed processor under test.
trait Keyed {
    fn setup(&mut self, rate: f32, channels: usize, key_channels: usize);
    fn change(&mut self, rng: &mut Rng, i: usize);
    fn run(&mut self, samples: &mut [f32], key: &[f32], key_channels: usize, i: usize);
    fn reset(&mut self);
}

fn drive<P: Keyed>(mut proc: P) {
    let max_frames = *SIZES.iter().max().unwrap();
    for (rate, channels, key_channels) in CONFIGS {
        proc.setup(rate, channels, key_channels);
        let input = signal(max_frames * 8, channels, 0x1234_5678);
        let key = signal(max_frames * 8, key_channels, 0x0bad_cafe);
        let mut buf = vec![0.0f32; max_frames * channels];
        let mut rng = Rng(0x9e37_79b9);
        let mut pos = 0usize;
        for i in 0..BUFFERS {
            let frames = SIZES[i % SIZES.len()];
            if (pos + frames) * channels > input.len() {
                pos = 0;
            }
            let block = &mut buf[..frames * channels];
            let k = &key[pos * key_channels..(pos + frames) * key_channels];

            let _rt = ScopedSanitizeRealtime::default();
            block.copy_from_slice(&input[pos * channels..(pos + frames) * channels]);
            proc.change(&mut rng, i);
            proc.run(block, k, key_channels, i);
            if i % 97 == 96 {
                proc.reset();
            }
            pos += frames;
            drop(_rt);

            assert!(block.iter().all(|x| x.is_finite()), "non-finite output");
        }
    }
}

impl Keyed for Gate {
    fn setup(&mut self, rate: f32, channels: usize, key_channels: usize) {
        self.configure(rate, channels);
        self.set_key_channels(key_channels);
    }
    fn change(&mut self, rng: &mut Rng, i: usize) {
        self.set_params(&GateParams {
            open_threshold_db: rng.range(-80.0, 0.0),
            close_threshold_db: rng.range(-80.0, 0.0),
            attack_ms: rng.range(0.01, 200.0),
            hold_ms: rng.range(0.0, 1000.0),
            release_ms: rng.range(1.0, 2000.0),
            reduction_db: rng.range(-80.0, 0.0),
        });
        self.set_key_filter(random_filter(rng));
        self.set_mode(if i.is_multiple_of(3) {
            GateMode::Gate
        } else {
            GateMode::Expander
        });
        self.set_expander_ratio(rng.range(1.0, 20.0));
    }
    fn run(&mut self, samples: &mut [f32], key: &[f32], key_channels: usize, i: usize) {
        if i % 5 == 4 {
            self.process(samples);
        } else {
            self.process_with_key(samples, key, key_channels);
        }
    }
    fn reset(&mut self) {
        Processor::reset(self);
    }
}

impl Keyed for Compressor {
    fn setup(&mut self, rate: f32, channels: usize, key_channels: usize) {
        self.configure(rate, channels);
        self.set_key_channels(key_channels);
    }
    fn change(&mut self, rng: &mut Rng, _i: usize) {
        self.set_params(&CompressorParams {
            threshold: rng.range(0.0, 1.0),
            ratio: rng.range(1.0, 100.0),
            attack_ms: rng.range(0.01, 10_000.0),
            release_ms: rng.range(0.01, 10_000.0),
            makeup_gain: rng.range(0.0, 100.0),
            knee: rng.range(0.0, 1.0),
        });
        self.set_key_filter(random_filter(rng));
    }
    fn run(&mut self, samples: &mut [f32], key: &[f32], key_channels: usize, i: usize) {
        if i % 11 == 10 {
            // Mismatched key: the silent path.
            self.process_with_key(samples, key, key_channels + 1);
        } else {
            self.process_with_key(samples, key, key_channels);
        }
    }
    fn reset(&mut self) {
        Processor::reset(self);
    }
}

/// A bare detector, as the upcoming ducker / de-esser use it.
struct Detector {
    det: KeyDetector,
    peaks: Vec<f32>,
}

impl Keyed for Detector {
    fn setup(&mut self, rate: f32, _channels: usize, key_channels: usize) {
        self.det.configure(rate, key_channels);
        self.peaks = vec![0.0; 1024];
    }
    fn change(&mut self, rng: &mut Rng, _i: usize) {
        self.det.set_filter(random_filter(rng));
    }
    fn run(&mut self, _samples: &mut [f32], key: &[f32], key_channels: usize, _i: usize) {
        let frames = key.len() / key_channels;
        self.det
            .linked_peaks(key, key_channels, &mut self.peaks[..frames]);
    }
    fn reset(&mut self) {
        self.det.reset();
    }
}

#[test]
fn keyed_gate_and_expander_are_realtime_safe() {
    drive(Gate::new());
}

#[test]
fn keyed_compressor_is_realtime_safe() {
    drive(Compressor::new());
}

#[test]
fn key_detector_is_realtime_safe() {
    drive(Detector {
        det: KeyDetector::new(),
        peaks: Vec::new(),
    });
}
