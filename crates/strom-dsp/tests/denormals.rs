//! Denormal policy check.
//!
//! The library does not set FTZ/DAZ in the CPU (see PROVENANCE.md, "Engineering policies");
//! instead every recursive state is flushed explicitly. This test makes sure
//! that works: each processor gets a loud signal, then several seconds of
//! exact digital silence, with extreme parameters (long releases, very
//! low-frequency and high-Q EQ bands, maximum makeup gain). No output sample
//! and no exposed state value may ever be subnormal.
//!
//! Only values are checked, never speed, so the test is not timing-sensitive.

use strom_dsp::{
    BandParams, BandType, Compressor, CompressorParams, Equalizer, EqualizerParams, Gate,
    GateParams, Limiter, LimiterParams, Processor, MAX_BANDS,
};

/// Loud part (s).
const LOUD_SECONDS: f32 = 0.5;
/// Silent part (s).
const SILENT_SECONDS: f32 = 6.0;
const BUFFER_FRAMES: usize = 512;
/// (sample rate, channels).
const CONFIGS: [(f32, usize); 3] = [(48_000.0, 2), (8_000.0, 1), (192_000.0, 2)];

/// Loud input: a 0 dBFS low sine plus a 997 Hz sine plus an impulse train,
/// with the second channel at a different level (exercises linking).
fn loud_sample(n: usize, channel: usize, rate: f32) -> f32 {
    let t = n as f32 / rate;
    let low = (2.0 * std::f32::consts::PI * 30.0 * t).sin();
    let mid = (2.0 * std::f32::consts::PI * 997.0 * t).sin();
    let click = if n.is_multiple_of(4801) { 1.0 } else { 0.0 };
    let x = 0.6 * low + 0.3 * mid + 0.4 * click;
    if channel.is_multiple_of(2) {
        x
    } else {
        0.5 * x
    }
}

/// Feed loud signal, then silence, through `proc`. After each buffer, check
/// every output sample, and call `state` to get extra state values to check.
fn run<P: Processor>(name: &str, proc: &mut P, state: impl Fn(&P) -> Vec<(&'static str, f64)>) {
    for (rate, channels) in CONFIGS {
        proc.configure(rate, channels);
        let loud_frames = (LOUD_SECONDS * rate) as usize;
        let total_frames = loud_frames + (SILENT_SECONDS * rate) as usize;
        let mut buf = vec![0.0f32; BUFFER_FRAMES * channels];
        let mut frame = 0usize;
        while frame < total_frames {
            let frames = BUFFER_FRAMES.min(total_frames - frame);
            let block = &mut buf[..frames * channels];
            for (i, x) in block.iter_mut().enumerate() {
                let n = frame + i / channels;
                *x = if n < loud_frames {
                    loud_sample(n, i % channels, rate)
                } else {
                    0.0
                };
            }
            proc.process(block);
            for (i, x) in block.iter().enumerate() {
                assert!(
                    !x.is_subnormal(),
                    "{name} @ {rate} Hz × {channels}: subnormal output {x:e} at frame {} \
                     channel {} ({:.3} s into silence)",
                    frame + i / channels,
                    i % channels,
                    (frame + i / channels) as f32 / rate - LOUD_SECONDS,
                );
                assert!(x.is_finite(), "{name}: non-finite output");
            }
            for (what, v) in state(proc) {
                assert!(
                    !v.is_subnormal(),
                    "{name} @ {rate} Hz × {channels}: subnormal {what} = {v:e} after frame {}",
                    frame + frames,
                );
            }
            frame += frames;
        }
    }
}

#[test]
fn gate_never_goes_subnormal() {
    let cases = [
        // Slowest fades, longest hold, deepest reduction.
        GateParams {
            open_threshold_db: -80.0,
            close_threshold_db: -80.0,
            attack_ms: 200.0,
            hold_ms: 1000.0,
            release_ms: 2000.0,
            reduction_db: -80.0,
        },
        // Fastest fades, no hold.
        GateParams {
            open_threshold_db: -20.0,
            close_threshold_db: -30.0,
            attack_ms: 0.01,
            hold_ms: 0.0,
            release_ms: 1.0,
            reduction_db: -80.0,
        },
        // No reduction at all.
        GateParams {
            reduction_db: 0.0,
            ..GateParams::default()
        },
    ];
    for p in cases {
        let mut g = Gate::new();
        g.set_params(&p);
        run("gate", &mut g, |g| vec![("current_gain", g.current_gain())]);
    }
}

#[test]
fn compressor_never_goes_subnormal() {
    let cases = [
        // Longest release, maximum ratio and makeup, deep threshold, hard knee.
        CompressorParams {
            threshold: 0.001,
            ratio: 100.0,
            attack_ms: 0.01,
            release_ms: 10_000.0,
            makeup_gain: 100.0,
            knee: 1.0,
        },
        // Fastest release, widest knee.
        CompressorParams {
            threshold: 0.1,
            ratio: 20.0,
            attack_ms: 0.01,
            release_ms: 0.01,
            makeup_gain: 1.0,
            knee: 0.0,
        },
        // Longest attack and release.
        CompressorParams {
            threshold: 0.05,
            ratio: 4.0,
            attack_ms: 10_000.0,
            release_ms: 10_000.0,
            makeup_gain: 10.0,
            knee: 0.0625,
        },
    ];
    for p in cases {
        let mut c = Compressor::new();
        c.set_params(&p);
        run("compressor", &mut c, |c| {
            vec![("gain_reduction_db", c.gain_reduction_db())]
        });
    }
}

fn band(band_type: BandType, frequency: f32, gain_db: f32, q: f32) -> BandParams {
    BandParams {
        enabled: true,
        band_type,
        frequency,
        gain_db,
        q,
    }
}

#[test]
fn equalizer_never_goes_subnormal() {
    use BandType::*;
    let cases: [[BandParams; MAX_BANDS]; 3] = [
        // Very low frequencies, extreme Q and gain: slowly decaying states.
        [
            band(HighPass, 1.0, 0.0, 0.01),
            band(LowPass, 20.0, 0.0, 100.0),
            band(Peaking, 20.0, 60.0, 100.0),
            band(LowShelf, 1.0, 60.0, 0.01),
            band(HighShelf, 1.0, -60.0, 100.0),
            band(BandPassSkirt, 5.0, 0.0, 100.0),
            band(Notch, 1.0, 0.0, 0.01),
            band(AllPass, 1.0, 0.0, 100.0),
        ],
        // Same at the top of the range (clamped below Nyquist).
        [
            band(LowPass, 24_000.0, 0.0, 0.01),
            band(HighPass, 24_000.0, 0.0, 100.0),
            band(Peaking, 24_000.0, -60.0, 0.01),
            band(HighShelf, 24_000.0, 60.0, 0.01),
            band(BandPassPeak, 24_000.0, 0.0, 100.0),
            band(Notch, 24_000.0, 0.0, 100.0),
            band(AllPass, 24_000.0, 0.0, 0.01),
            band(LowShelf, 24_000.0, -60.0, 100.0),
        ],
        // A typical strip EQ (HPF + 2 peaking + high shelf) with deep cuts.
        [
            band(HighPass, 20.0, 0.0, 0.707),
            band(Peaking, 40.0, -60.0, 10.0),
            band(Peaking, 3000.0, 12.0, 1.0),
            band(HighShelf, 10_000.0, -60.0, 0.707),
            BandParams::default(),
            BandParams::default(),
            BandParams::default(),
            BandParams::default(),
        ],
    ];
    for bands in cases {
        let mut e = Equalizer::new();
        e.set_params(&EqualizerParams {
            num_bands: MAX_BANDS,
            bands,
        });
        run("equalizer", &mut e, |_| Vec::new());
    }
}

#[test]
fn limiter_never_goes_subnormal() {
    let cases = [
        // Longest release, attack and lookahead, deepest ceiling.
        LimiterParams {
            enabled: true,
            threshold_db: -60.0,
            attack_ms: 50.0,
            release_ms: 2000.0,
            lookahead_ms: 20.0,
        },
        // Shortest everything, no lookahead.
        LimiterParams {
            enabled: true,
            threshold_db: 0.0,
            attack_ms: 0.01,
            release_ms: 1.0,
            lookahead_ms: 0.0,
        },
        // Disabled: unity gain through the delay.
        LimiterParams {
            enabled: false,
            ..LimiterParams::default()
        },
    ];
    for p in cases {
        let mut l = Limiter::new();
        l.set_params(&p);
        run("limiter", &mut l, |l| {
            vec![("current_gain", l.current_gain())]
        });
    }
}
