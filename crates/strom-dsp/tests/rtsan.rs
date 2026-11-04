//! RealtimeSanitizer (RTSan) checks of the real-time path.
//!
//! Every processor runs through many buffers of varying size, with a parameter
//! change before every buffer, while `set_params`, `process`, `latency_frames`
//! and (now and then) `reset` execute inside a real-time scope. Any call to
//! `malloc`/`free`, a lock, a sleep, a system call or other non-real-time-safe
//! function in that scope aborts the test with a stack trace.
//!
//! Only built with `--features rtsan`, and only effective with `RTSAN_ENABLE=1`
//! at build time (otherwise the scopes are no-ops and this is a plain
//! smoke test):
//!
//! ```text
//! RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test suite rtsan::
//! ```
//!
//! Setup (`new`, `configure`, signal generation) happens outside the scope:
//! `configure` is documented as the one place that allocates.

#![cfg(feature = "rtsan")]

use rtsan_standalone::ScopedSanitizeRealtime;
use strom_dsp::{
    BandParams, BandType, Compressor, CompressorParams, Equalizer, EqualizerParams, Gate,
    GateParams, Limiter, LimiterParams, Processor, MAX_BANDS,
};

/// Buffers per configuration.
const BUFFERS: usize = 600;
/// Buffer sizes in frames, cycled through (includes 1 and non-powers of two).
const SIZES: [usize; 7] = [1024, 1, 480, 7, 64, 1023, 256];
/// (sample rate, channels) configurations.
const CONFIGS: [(f32, usize); 3] = [(48_000.0, 2), (44_100.0, 1), (96_000.0, 8)];

/// Small deterministic PRNG (xorshift32) so the test needs no extra crates.
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

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Uniform in `[lo, hi]`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

/// Test signal: bursts of noise and sine with levels from silence to +12 dBFS,
/// so every processor goes through all of its states (gate open/closed,
/// compressor in and out of the knee, limiter engaged and released).
fn signal(frames: usize, channels: usize) -> Vec<f32> {
    let mut rng = Rng(0x1234_5678);
    let mut out = Vec::with_capacity(frames * channels);
    let segment = 2000;
    let mut level = 0.0f32;
    for n in 0..frames {
        if n % segment == 0 {
            level = match (n / segment) % 6 {
                0 => 0.0,
                1 => 0.001,
                2 => 0.1,
                3 => 0.7,
                4 => 4.0,
                _ => rng.unit(),
            };
        }
        let sine = (2.0 * std::f32::consts::PI * 997.0 * n as f32 / 48_000.0).sin();
        for c in 0..channels {
            let noise = rng.range(-1.0, 1.0);
            // Odd channels get only a fraction, to exercise channel linking.
            let scale = if c % 2 == 0 { 1.0 } else { 0.25 };
            out.push(level * scale * (0.5 * sine + 0.5 * noise));
        }
    }
    out
}

/// Run `proc` over `BUFFERS` buffers per configuration. `change` is called
/// before every buffer (with the buffer index) and must update the parameters;
/// it runs inside the real-time scope together with `process`.
fn drive<P: Processor>(mut proc: P, mut change: impl FnMut(&mut P, &mut Rng, usize)) {
    let max_frames = *SIZES.iter().max().unwrap();
    for (rate, channels) in CONFIGS {
        proc.configure(rate, channels);
        let input = signal(max_frames * 8, channels);
        let mut buf = vec![0.0f32; max_frames * channels];
        let mut rng = Rng(0x9e37_79b9);
        let mut pos = 0usize;

        for i in 0..BUFFERS {
            let frames = SIZES[i % SIZES.len()];
            let len = frames * channels;
            if pos + len > input.len() {
                pos = 0;
            }
            let block = &mut buf[..len];

            let _rt = ScopedSanitizeRealtime::default();
            block.copy_from_slice(&input[pos..pos + len]);
            change(&mut proc, &mut rng, i);
            proc.process(block);
            let _ = proc.latency_frames();
            if i % 97 == 96 {
                proc.reset();
            }
            pos += len;
            drop(_rt);

            assert!(block.iter().all(|x| x.is_finite()), "non-finite output");
        }
    }
}

#[test]
fn gate_is_realtime_safe() {
    drive(Gate::new(), |g, rng, _| {
        g.set_params(&GateParams {
            open_threshold_db: rng.range(-80.0, 0.0),
            close_threshold_db: rng.range(-80.0, 0.0),
            attack_ms: rng.range(0.01, 200.0),
            hold_ms: rng.range(0.0, 1000.0),
            release_ms: rng.range(1.0, 2000.0),
            reduction_db: rng.range(-80.0, 0.0),
        });
    });
}

#[test]
fn compressor_is_realtime_safe() {
    drive(Compressor::new(), |c, rng, _| {
        c.set_params(&CompressorParams {
            threshold: rng.range(0.0, 1.0),
            ratio: rng.range(1.0, 100.0),
            attack_ms: rng.range(0.01, 10_000.0),
            release_ms: rng.range(0.01, 10_000.0),
            makeup_gain: rng.range(0.0, 100.0),
            knee: rng.range(0.0, 1.0),
        });
    });
}

#[test]
fn equalizer_is_realtime_safe() {
    drive(Equalizer::new(), |e, rng, i| {
        let mut p = EqualizerParams {
            num_bands: 1 + (rng.next_u32() as usize % MAX_BANDS),
            bands: [BandParams::default(); MAX_BANDS],
        };
        for b in p.bands.iter_mut() {
            *b = BandParams {
                // Every other buffer keeps the types and only moves the
                // continuous parameters (glide path); otherwise types change too
                // (cross-fade path).
                enabled: i % 2 == 0 || rng.unit() > 0.2,
                band_type: if i % 2 == 0 {
                    BandType::Peaking
                } else {
                    BandType::from_i32((rng.next_u32() % 10) as i32)
                },
                frequency: rng.range(1.0, 24_000.0),
                gain_db: rng.range(-60.0, 60.0),
                q: rng.range(0.01, 100.0),
            };
        }
        e.set_params(&p);
    });
}

#[test]
fn limiter_is_realtime_safe() {
    drive(Limiter::new(), |l, rng, i| {
        l.set_params(&LimiterParams {
            enabled: i % 13 != 12,
            threshold_db: rng.range(-60.0, 0.0),
            attack_ms: rng.range(0.01, 50.0),
            release_ms: rng.range(1.0, 2000.0),
            // Lookahead changes cross-fade the delay tap; attack changes
            // rebuild the gain pipeline.
            lookahead_ms: rng.range(0.0, 20.0),
        });
    });
}

#[test]
fn channel_strip_is_realtime_safe() {
    // Gate → compressor → 4-band EQ → limiter, as one strip, parameters
    // changing on every buffer as an automated mixer would.
    struct Strip {
        gate: Gate,
        comp: Compressor,
        eq: Equalizer,
        lim: Limiter,
    }
    impl Processor for Strip {
        fn configure(&mut self, sample_rate: f32, channels: usize) {
            self.gate.configure(sample_rate, channels);
            self.comp.configure(sample_rate, channels);
            self.eq.configure(sample_rate, channels);
            self.lim.configure(sample_rate, channels);
        }
        fn process(&mut self, samples: &mut [f32]) {
            self.gate.process(samples);
            self.comp.process(samples);
            self.eq.process(samples);
            self.lim.process(samples);
        }
        fn reset(&mut self) {
            self.gate.reset();
            self.comp.reset();
            self.eq.reset();
            self.lim.reset();
        }
        fn latency_frames(&self) -> usize {
            self.lim.latency_frames()
        }
    }

    let strip = Strip {
        gate: Gate::new(),
        comp: Compressor::new(),
        eq: Equalizer::new(),
        lim: Limiter::new(),
    };
    drive(strip, |s, rng, _| {
        s.gate.set_params(&GateParams {
            open_threshold_db: rng.range(-60.0, -30.0),
            ..GateParams::default()
        });
        s.comp.set_params(&CompressorParams {
            threshold: rng.range(0.1, 0.5),
            ratio: rng.range(2.0, 8.0),
            ..CompressorParams::default()
        });
        let mut eq = EqualizerParams::default();
        eq.bands[0] = BandParams {
            band_type: BandType::HighPass,
            frequency: rng.range(40.0, 120.0),
            ..BandParams::default()
        };
        eq.bands[1] = BandParams {
            band_type: BandType::Peaking,
            frequency: rng.range(200.0, 800.0),
            gain_db: rng.range(-6.0, 6.0),
            q: 1.0,
            ..BandParams::default()
        };
        eq.bands[2] = BandParams {
            band_type: BandType::Peaking,
            frequency: rng.range(2000.0, 5000.0),
            gain_db: rng.range(-6.0, 6.0),
            q: 1.0,
            ..BandParams::default()
        };
        eq.bands[3] = BandParams {
            band_type: BandType::HighShelf,
            frequency: rng.range(8000.0, 12_000.0),
            gain_db: rng.range(-6.0, 6.0),
            ..BandParams::default()
        };
        s.eq.set_params(&eq);
        s.lim.set_params(&LimiterParams {
            threshold_db: rng.range(-6.0, -0.1),
            ..LimiterParams::default()
        });
    });
}

/// Canary: allocates inside a real-time scope, so with `RTSAN_ENABLE=1` it must
/// abort the process. Ignored by default; CI runs it and expects a failure, to
/// prove the sanitizer is really active (and the tests above are not no-ops).
#[test]
#[ignore = "must abort under RTSAN_ENABLE=1; run explicitly as a canary"]
fn canary_allocation_is_detected() {
    let _rt = ScopedSanitizeRealtime::default();
    let v = std::hint::black_box(vec![1.0f32; 64]);
    assert_eq!(v.len(), 64);
}
