//! De-esser tests, verified against theory and invariants: reduction on
//! high-frequency bursts and none on a low tone at the same level; the low
//! band untouched in split-band mode; the band sum flat (LR4 all-pass) at zero
//! reduction; the reduction capped at `max_reduction_db`; linked channels;
//! robustness at the parameter extremes; click-free parameter changes; zero
//! allocations in the processing path.

use std::f64::consts::PI;

use strom_dsp::deesser::{Deesser, DeesserMode, DeesserParams};
use strom_dsp::Processor;

use crate::counting_alloc::allocs;

// ---------------------------------------------------------------------------
// Helpers.

const RATES: [f32; 3] = [44_100.0, 48_000.0, 96_000.0];

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-300).log10()
}

fn deesser(fs: f32, ch: usize, p: DeesserParams) -> Deesser {
    let mut d = Deesser::new();
    d.set_params(&p);
    d.configure(fs, ch);
    d
}

/// Mono sine of amplitude `a` at `f` Hz, samples `start .. start + len`.
fn sine(f: f64, fs: f64, a: f64, start: usize, len: usize) -> Vec<f32> {
    (start..start + len)
        .map(|n| (a * (2.0 * PI * f * n as f64 / fs).sin()) as f32)
        .collect()
}

/// Amplitude of the component at `f` in `x` (least-squares projection onto
/// sin/cos over a whole number of periods of `f`, taken from the end of `x`).
fn amplitude_at(x: &[f32], f: f64, fs: f64, offset: usize) -> f64 {
    let periods = ((x.len() as f64 * f / fs).floor()).max(1.0);
    let n = ((periods * fs / f).round() as usize).min(x.len());
    let start = x.len() - n;
    let (mut s, mut c) = (0.0, 0.0);
    for (i, &v) in x[start..].iter().enumerate() {
        let ph = 2.0 * PI * f * (offset + start + i) as f64 / fs;
        s += v as f64 * ph.sin();
        c += v as f64 * ph.cos();
    }
    2.0 * (s * s + c * c).sqrt() / n as f64
}

fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64).sqrt()
}

fn ms(t: f64, fs: f32) -> usize {
    (t * 0.001 * fs as f64).round() as usize
}

// ---------------------------------------------------------------------------
// Reduction on high-frequency bursts, none on a low tone at the same level.

/// −10 dBFS bursts at 12 kHz are reduced by `max_reduction` (the detected level
/// is about 20 dB above the −30 dB threshold); a 300 Hz tone at the same level
/// is not reduced at all (the detector high-pass rejects it by ≈ 52 dB).
#[test]
fn reduces_hf_bursts_not_low_tone() {
    let a = 10f64.powf(-10.0 / 20.0);
    for fs in RATES {
        for mode in [DeesserMode::SplitBand, DeesserMode::Wideband] {
            let p = DeesserParams {
                mode,
                ..Default::default()
            };
            // Bursts: 150 ms on, 150 ms off, three times.
            let mut d = deesser(fs, 1, p);
            let on = ms(150.0, fs);
            for burst in 0..3 {
                let start = burst * 2 * on;
                let mut x = sine(12_000.0, fs as f64, a, start, on);
                d.process(&mut x);
                let red = d.gain_reduction_db();
                assert!(
                    (red - 10.0).abs() < 0.5,
                    "{fs} {mode:?}: reduction {red} dB at the end of burst {burst}"
                );
                let got = db(amplitude_at(&x[on / 2..], 12_000.0, fs as f64, start + on / 2) / a);
                // Wideband: the whole signal at −10 dB. Split-band: the LR4
                // high band (|H| = 1/(1 + (6/12)^4) ≈ 0.94 at 12 kHz, in
                // phase with the low band) at −10 dB, the low band unchanged.
                let h = 1.0 / (1.0 + (6.0f64 / 12.0).powi(4));
                let want = match mode {
                    DeesserMode::Wideband => -10.0,
                    DeesserMode::SplitBand => db((1.0 - h) + h * 10f64.powf(-0.5)),
                };
                assert!(
                    (got - want).abs() < 0.6,
                    "{fs} {mode:?}: burst gain {got} dB, expected {want} dB"
                );
                // Silence between bursts: the reduction releases with the
                // documented time constant, r(t) = r0·exp(−t/τ_release).
                let mut gap = vec![0.0f32; on];
                d.process(&mut gap);
                let want = red * (-150.0f64 / 50.0).exp();
                let got = d.gain_reduction_db();
                assert!(
                    (got - want).abs() < 0.03 * red,
                    "{fs}: released to {got} dB, expected {want} dB"
                );
            }

            // Low tone at the same level: no reduction, magnitude unchanged.
            let mut d = deesser(fs, 1, p);
            let n = ms(500.0, fs);
            let mut x = sine(300.0, fs as f64, a, 0, n);
            d.process(&mut x);
            assert_eq!(
                d.gain_reduction_db(),
                0.0,
                "{fs} {mode:?}: low tone reduced"
            );
            let got = db(amplitude_at(&x[n / 2..], 300.0, fs as f64, n / 2) / a);
            assert!(got.abs() < 0.01, "{fs} {mode:?}: low tone gain {got} dB");
        }
    }
}

/// With `max_reduction = 0` nothing is ever reduced, however loud.
#[test]
fn max_reduction_caps_the_reduction() {
    for cap in [0.0f32, 3.0, 20.0] {
        let p = DeesserParams {
            threshold_db: -60.0,
            max_reduction_db: cap,
            ..Default::default()
        };
        let mut d = deesser(48_000.0, 1, p);
        let mut x = sine(9_000.0, 48_000.0, 0.9, 0, 24_000);
        let mut peak: f64 = 0.0;
        for c in x.chunks_mut(100) {
            d.process(c);
            peak = peak.max(d.gain_reduction_db());
        }
        assert!(peak <= cap as f64 + 1e-9, "cap {cap}: {peak}");
        assert!(peak >= cap as f64 - 0.5, "cap {cap}: only {peak}");
    }
}

// ---------------------------------------------------------------------------
// Split-band leaves the low band untouched.

/// A 300 Hz tone plus 12 kHz bursts: in split-band mode the 300 Hz component of
/// the output keeps its amplitude exactly (to 0.01 dB) while the bursts are
/// reduced by 10 dB; in wideband mode the same tone is reduced (sanity check
/// that the measurement sees the gain). Also checks the output's low band
/// (the part below 1 kHz, separated with a second crossover) against the input's
/// low band sample by sample.
#[test]
fn split_band_leaves_low_band_untouched() {
    use strom_dsp::filter::{Crossover, CrossoverParams, LrOrder};
    for fs in RATES {
        let n = ms(600.0, fs);
        let low = sine(300.0, fs as f64, 0.3, 0, n);
        let high = sine(12_000.0, fs as f64, 0.3, 0, n);
        let input: Vec<f32> = low.iter().zip(&high).map(|(a, b)| a + b).collect();

        let run = |mode| {
            let mut d = deesser(
                fs,
                1,
                DeesserParams {
                    mode,
                    ..Default::default()
                },
            );
            let mut y = input.clone();
            d.process(&mut y);
            assert!((d.gain_reduction_db() - 10.0).abs() < 0.5, "{fs} {mode:?}");
            y
        };
        let split = run(DeesserMode::SplitBand);
        let wide = run(DeesserMode::Wideband);
        let tail = n / 2;
        let a_split = amplitude_at(&split[tail..], 300.0, fs as f64, tail);
        let a_wide = amplitude_at(&wide[tail..], 300.0, fs as f64, tail);
        let a_high = amplitude_at(&split[tail..], 12_000.0, fs as f64, tail);
        assert!(db(a_split / 0.3).abs() < 0.01, "{fs}: low tone {a_split}");
        assert!(db(a_wide / 0.3) < -9.0, "{fs}: wideband low tone {a_wide}");
        assert!(db(a_high / 0.3) < -8.0, "{fs}: bursts not reduced {a_high}");

        // Low band of the output vs low band of the input (LR4 at 1 kHz, far
        // below the de-esser's 6 kHz split): the reduction of the high band
        // leaves at most ≈ (1/6)^4·(1 − g) of it below 1 kHz.
        let lowband = |x: &[f32]| {
            let mut xo = Crossover::new();
            xo.set_params(&CrossoverParams {
                order: LrOrder::Lr4,
                num_bands: 2,
                frequencies_hz: [1_000.0; 4],
            });
            xo.configure(fs, 1);
            let mut bands = vec![vec![0.0f32; x.len()]; 2];
            xo.process(x, &mut bands);
            bands.swap_remove(0)
        };
        // Reference: the de-esser with no reduction (same all-pass phase).
        let mut d0 = deesser(
            fs,
            1,
            DeesserParams {
                max_reduction_db: 0.0,
                ..Default::default()
            },
        );
        let mut unreduced = input.clone();
        d0.process(&mut unreduced);
        let (l_out, l_ref) = (lowband(&split), lowband(&unreduced));
        let diff: Vec<f32> = l_out[tail..]
            .iter()
            .zip(&l_ref[tail..])
            .map(|(a, b)| a - b)
            .collect();
        let rel = db(rms(&diff) / rms(&l_ref[tail..]));
        assert!(rel < -60.0, "{fs}: low band changed by {rel} dB");
    }
}

// ---------------------------------------------------------------------------
// Flat sum at zero reduction.

/// With no reduction (`max_reduction = 0`, and separately a signal below the
/// threshold) the output magnitude equals the input's at every frequency
/// within ±0.05 dB, in both modes.
#[test]
fn flat_sum_at_zero_reduction() {
    let freqs = [
        20.0, 100.0, 1_000.0, 3_000.0, 5_000.0, 6_000.0, 7_000.0, 10_000.0, 15_000.0, 19_000.0,
    ];
    for fs in RATES {
        for mode in [DeesserMode::SplitBand, DeesserMode::Wideband] {
            for (label, p, a) in [
                (
                    "max-reduction 0",
                    DeesserParams {
                        max_reduction_db: 0.0,
                        threshold_db: -60.0,
                        mode,
                        ..Default::default()
                    },
                    0.5,
                ),
                (
                    "below threshold",
                    DeesserParams {
                        threshold_db: -20.0,
                        mode,
                        ..Default::default()
                    },
                    0.01,
                ),
            ] {
                // One channel per frequency (channels are independent).
                let ch = freqs.len();
                let mut d = deesser(fs, ch, p);
                let n = ms(400.0, fs);
                let mut x = vec![0.0f32; n * ch];
                for (c, &f) in freqs.iter().enumerate() {
                    for (i, v) in sine(f, fs as f64, a, 0, n).into_iter().enumerate() {
                        x[i * ch + c] = v;
                    }
                }
                d.process(&mut x);
                assert_eq!(d.gain_reduction_db(), 0.0, "{label}");
                for (c, &f) in freqs.iter().enumerate() {
                    let y: Vec<f32> = x.iter().skip(c).step_by(ch).copied().collect();
                    let tail = n / 2;
                    let g = db(amplitude_at(&y[tail..], f, fs as f64, tail) / a);
                    assert!(g.abs() < 0.05, "{fs} {mode:?} {label}: {f} Hz at {g} dB");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Linked channels.

/// Sibilance in channel 0 only reduces both channels by the same gain
/// (wideband), so the image does not move.
#[test]
fn channels_are_linked() {
    let fs = 48_000.0;
    let p = DeesserParams {
        mode: DeesserMode::Wideband,
        ..Default::default()
    };
    let mut d = deesser(fs, 2, p);
    let n = 24_000;
    let hf = sine(10_000.0, fs as f64, 0.3, 0, n);
    let lf = sine(500.0, fs as f64, 0.3, 0, n);
    let mut x: Vec<f32> = hf.iter().zip(&lf).flat_map(|(&a, &b)| [a, b]).collect();
    d.process(&mut x);
    let y1: Vec<f32> = x.iter().skip(1).step_by(2).copied().collect();
    let tail = n / 2;
    let g1 = db(amplitude_at(&y1[tail..], 500.0, fs as f64, tail) / 0.3);
    let r = d.gain_reduction_db();
    assert!(r > 9.0, "reduction {r}");
    assert!(
        (g1 + r).abs() < 0.3,
        "channel 1 at {g1} dB, reduction {r} dB"
    );
}

// ---------------------------------------------------------------------------
// Robustness.

#[test]
fn robust_at_parameter_extremes() {
    let extremes = [
        DeesserParams {
            frequency_hz: 1_000.0,
            threshold_db: -60.0,
            max_reduction_db: 40.0,
            attack_ms: 0.01,
            release_ms: 1.0,
            mode: DeesserMode::Wideband,
        },
        DeesserParams {
            frequency_hz: 20_000.0,
            threshold_db: 0.0,
            max_reduction_db: 0.0,
            attack_ms: 200.0,
            release_ms: 2000.0,
            mode: DeesserMode::SplitBand,
        },
        DeesserParams {
            frequency_hz: f32::NAN,
            threshold_db: f32::INFINITY,
            max_reduction_db: -5.0,
            attack_ms: f32::NEG_INFINITY,
            release_ms: 1e9,
            mode: DeesserMode::SplitBand,
        },
    ];
    for fs in [8_000.0f32, 44_100.0, 48_000.0, 96_000.0, 192_000.0] {
        for ch in [1usize, 2, 8, 64] {
            for p in &extremes {
                let mut d = deesser(fs, ch, *p);
                let mut seed = 0x1234_5678u32;
                let mut x: Vec<f32> = (0..2048 * ch)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 17;
                        seed ^= seed << 5;
                        (seed as f32 / u32::MAX as f32 - 0.5) * 4.0
                    })
                    .collect();
                d.process(&mut x);
                assert!(x.iter().all(|v| v.is_finite()), "{fs} {ch} {p:?}");
                let r = d.gain_reduction_db();
                assert!(r.is_finite() && (0.0..=40.0).contains(&r), "{r}");
                let mut silence = vec![0.0f32; 96_000 * ch / 8];
                d.process(&mut silence);
                assert!(silence.iter().all(|v| v.is_finite()));
            }
        }
    }
    // Unconfigured: passthrough. Non-finite input does not poison the state.
    let mut d = Deesser::new();
    let mut x = [0.5f32, -0.25];
    d.process(&mut x);
    assert_eq!(x, [0.5, -0.25]);
    let mut d = deesser(48_000.0, 1, DeesserParams::default());
    let mut x = [f32::NAN, f32::INFINITY, 0.5, -0.5];
    d.process(&mut x);
    let mut y = sine(500.0, 48_000.0, 0.5, 0, 4800);
    d.process(&mut y);
    assert!(y[2400..].iter().all(|v| v.is_finite()));
}

/// Parameter-change click bound.
///
/// A 4 kHz sine of amplitude `a = 0.3` (detected at all the frequency
/// settings below, so the gain is working) runs while every parameter jumps between extremes every 20 ms,
/// with the shortest attack and release (so the gain smoother gives no
/// protection). The gain is at most 1, and the LR4 bands are in phase, so the
/// output never exceeds the input envelope; a pure sine's largest sample step
/// is `2·a·sin(π·f/fs)`. Bound: no output step exceeds
/// `1.25 × 2·a·sin(π·f/fs)`. A sanity check shows an unsmoothed 40 dB
/// gain jump would exceed it.
#[test]
fn parameter_changes_do_not_click() {
    let fs = 48_000.0f32;
    let f = 4_000.0;
    let a = 0.3;
    let base = DeesserParams {
        attack_ms: 0.01,
        release_ms: 1.0,
        ..Default::default()
    };
    let sets = [
        DeesserParams { ..base },
        DeesserParams {
            threshold_db: -60.0,
            max_reduction_db: 40.0,
            mode: DeesserMode::Wideband,
            ..base
        },
        DeesserParams {
            threshold_db: 0.0,
            frequency_hz: 3_000.0,
            ..base
        },
        DeesserParams {
            threshold_db: -60.0,
            max_reduction_db: 40.0,
            frequency_hz: 10_000.0,
            ..base
        },
        DeesserParams {
            max_reduction_db: 0.0,
            mode: DeesserMode::Wideband,
            ..base
        },
    ];
    let bound = 1.25 * 2.0 * a * (PI * f / fs as f64).sin();
    let mut d = deesser(fs, 1, sets[0]);
    let block = ms(20.0, fs);
    let mut prev = 0.0f32;
    let mut max_step: f64 = 0.0;
    for i in 0..40 {
        d.set_params(&sets[i % sets.len()]);
        let mut x = sine(f, fs as f64, a, i * block, block);
        for c in x.chunks_mut(256) {
            d.process(c);
        }
        for &v in &x {
            max_step = max_step.max((v - prev).abs() as f64);
            prev = v;
        }
    }
    assert!(max_step <= bound, "max step {max_step} > bound {bound}");
    // Sanity: an instant 40 dB drop at a sine peak would step by ≈ a.
    assert!(bound < 0.99 * a, "bound {bound} not restrictive");
}

/// `reset` clears the reduction and the filters: the output after `reset`
/// equals that of a fresh instance.
#[test]
fn reset_matches_fresh_instance() {
    let fs = 48_000.0;
    let p = DeesserParams::default();
    let x = sine(9_000.0, fs as f64, 0.5, 0, 4_800);
    let mut d = deesser(fs, 1, p);
    let mut warm = x.clone();
    d.process(&mut warm);
    assert!(d.gain_reduction_db() > 1.0);
    d.reset();
    assert_eq!(d.gain_reduction_db(), 0.0);
    let mut a = x.clone();
    d.process(&mut a);
    let mut fresh = deesser(fs, 1, p);
    let mut b = x.clone();
    fresh.process(&mut b);
    assert_eq!(a, b);
}

// ---------------------------------------------------------------------------
// Real-time safety.

/// `set_params`, `process` and `reset` never allocate.
#[test]
fn no_allocation_in_processing() {
    for ch in [1usize, 2, 6] {
        let mut d = deesser(48_000.0, ch, DeesserParams::default());
        let mut x = vec![0.0f32; 1024 * ch];
        let src = sine(8_000.0, 48_000.0, 0.5, 0, 1024 * ch);
        let before = allocs();
        for i in 0..200 {
            let p = DeesserParams {
                frequency_hz: 2_000.0 + 50.0 * i as f32,
                threshold_db: -((i % 60) as f32),
                max_reduction_db: (i % 40) as f32,
                mode: if i % 3 == 0 {
                    DeesserMode::Wideband
                } else {
                    DeesserMode::SplitBand
                },
                ..Default::default()
            };
            d.set_params(&p);
            let len = [1024, 1, 480, 7][i % 4] * ch;
            x[..len].copy_from_slice(&src[..len]);
            d.process(&mut x[..len]);
            if i % 50 == 49 {
                d.reset();
            }
        }
        assert_eq!(allocs() - before, 0, "allocations in set_params/process");
    }
}
