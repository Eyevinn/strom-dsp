//! Butterworth filter and Linkwitz-Riley crossover tests, verified against
//! theory: the analytic magnitude of the bilinear-transformed (pre-warped)
//! Butterworth prototype, `1/√(1 + (tan(πf/fs)/tan(πfc/fs))^(2n))`, computed
//! independently here; the LR invariants (flat band sum, −6.02 dB band
//! edges); stability, robustness, click-free parameter changes, SIMD-level
//! equivalence and real-time safety.

use std::f64::consts::PI;

use strom_dsp::biquad::uses_fma;
use strom_dsp::filter::{
    Butterworth, ButterworthParams, Crossover, CrossoverParams, FilterKind, LrOrder,
    MAX_CROSSOVER_BANDS, MAX_SPLITS,
};
use strom_dsp::simd::{detect, Level};
use strom_dsp::Processor;

use crate::counting_alloc::allocs;

// ---------------------------------------------------------------------------
// Helpers.

const RATES: [f32; 5] = [8_000.0, 44_100.0, 48_000.0, 96_000.0, 192_000.0];
const KINDS: [FilterKind; 2] = [FilterKind::LowPass, FilterKind::HighPass];

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-300).log10()
}

/// Analytic magnitude of the digital Butterworth filter (bilinear transform
/// pre-warped at `fc`).
fn analytic(kind: FilterKind, order: usize, fc: f64, f: f64, fs: f64) -> f64 {
    let r = (PI * f / fs).tan() / (PI * fc / fs).tan();
    let r = match kind {
        FilterKind::LowPass => r,
        FilterKind::HighPass => 1.0 / r,
    };
    1.0 / (1.0 + r.powi(2 * order as i32)).sqrt()
}

fn params(kind: FilterKind, order: usize, cutoff_hz: f32) -> ButterworthParams {
    ButterworthParams {
        kind,
        order,
        cutoff_hz,
    }
}

fn filter(fs: f32, ch: usize, p: &ButterworthParams) -> Butterworth {
    let mut f = Butterworth::new();
    f.configure(fs, ch);
    f.set_params(p);
    f
}

/// Response `Σ h[n]·e^{−jωn}` of a (mono, f32) impulse response at `f` (Hz).
fn dft_at(h: &[f32], f: f64, fs: f64) -> (f64, f64) {
    let w = 2.0 * PI * f / fs;
    let (s, c) = w.sin_cos();
    // Rotating phasor e^{−jωn}, renormalised now and then.
    let (mut pr, mut pi) = (1.0f64, 0.0f64);
    let (mut re, mut im) = (0.0, 0.0);
    for (n, &x) in h.iter().enumerate() {
        let x = x as f64;
        re += x * pr;
        im += x * pi;
        let nr = pr * c + pi * s;
        let ni = pi * c - pr * s;
        pr = nr;
        pi = ni;
        if n % 1024 == 1023 {
            let m = (pr * pr + pi * pi).sqrt();
            pr /= m;
            pi /= m;
        }
    }
    (re, im)
}

fn mag(z: (f64, f64)) -> f64 {
    (z.0 * z.0 + z.1 * z.1).sqrt()
}

fn impulse_response(f: &mut Butterworth, n: usize) -> Vec<f32> {
    let mut x = vec![0.0f32; n];
    x[0] = 1.0;
    f.process(&mut x);
    x
}

fn log_freqs(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo * (hi / lo).powf(i as f64 / (n - 1) as f64))
        .collect()
}

fn sine(f: f64, fs: f64, amp: f64, start: usize, n: usize) -> Vec<f32> {
    (start..start + n)
        .map(|i| (amp * (2.0 * PI * f * i as f64 / fs).sin()) as f32)
        .collect()
}

fn noise(n: usize, seed: &mut u64, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            ((*seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * amp
        })
        .collect()
}

fn max_step(x: &[f32]) -> f64 {
    x.windows(2)
        .map(|w| (w[1] as f64 - w[0] as f64).abs())
        .fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------
// Butterworth.

/// The designed sections match the analytic magnitude for every order 1–8,
/// both kinds, five rates and several cutoffs, at 200 frequencies each:
/// within 1e-4 dB down to −120 dB (the limit of evaluating `|H|` in f64).
#[test]
fn butterworth_design_matches_analytic_magnitude() {
    for fs in RATES {
        let fsd = fs as f64;
        for fc in [20.0, 100.0, 1000.0, 0.1 * fsd, 0.45 * fsd] {
            for order in 1..=8 {
                for kind in KINDS {
                    let s = Butterworth::design(&params(kind, order, fc as f32), fs);
                    assert!(s.is_stable());
                    assert_eq!(s.len(), order.div_ceil(2));
                    let fc = fc as f32 as f64;
                    for f in log_freqs(1.0, 0.499 * fsd, 200) {
                        let want = analytic(kind, order, fc, f, fsd);
                        let got = s.magnitude_at(f, fsd);
                        if want > 1e-6 {
                            assert!(
                                (db(got) - db(want)).abs() < 1e-4,
                                "{kind:?} n={order} fc={fc} fs={fs} f={f}: {got} vs {want}"
                            );
                        } else {
                            assert!(got < 2e-6, "{kind:?} n={order} fc={fc} f={f}: {got}");
                        }
                    }
                    // −3.01 dB at the cutoff.
                    let at_fc = db(s.magnitude_at(fc, fsd));
                    assert!((at_fc + 10.0 * 2f64.log10()).abs() < 1e-6, "{at_fc}");
                }
            }
        }
    }
}

/// The processed signal (measured from the impulse response) matches the
/// analytic magnitude within 0.01 dB wherever the response is above −80 dB
/// (below that, f32 output rounding dominates).
#[test]
fn butterworth_processed_response_matches_analytic() {
    for fs in [8_000.0f32, 48_000.0, 192_000.0] {
        let fsd = fs as f64;
        for fc in [200.0, 0.2 * fsd] {
            for order in 1..=8 {
                for kind in KINDS {
                    let mut flt = filter(fs, 1, &params(kind, order, fc as f32));
                    let h = impulse_response(&mut flt, fs as usize / 2);
                    let fc = fc as f32 as f64;
                    for f in log_freqs(20.0, 0.45 * fsd, 25) {
                        let want = analytic(kind, order, fc, f, fsd);
                        if want < 1e-4 {
                            continue;
                        }
                        let got = mag(dft_at(&h, f, fsd));
                        assert!(
                            (db(got) - db(want)).abs() < 0.01,
                            "{kind:?} n={order} fc={fc} fs={fs} f={f}: {} vs {}",
                            db(got),
                            db(want)
                        );
                    }
                }
            }
        }
    }
}

/// Slope: an order-n filter falls by 6.02·n dB per octave far from the cutoff.
#[test]
fn butterworth_slope_matches_order() {
    let fs = 48_000.0;
    for order in 1..=8 {
        let s = Butterworth::design(&params(FilterKind::HighPass, order, 2000.0), fs);
        let d = db(s.magnitude_at(125.0, 48_000.0)) - db(s.magnitude_at(62.5, 48_000.0));
        assert!((d - 6.02 * order as f64).abs() < 0.05, "order {order}: {d}");
    }
}

/// 20 Hz at 192 kHz, order 8: stable, the impulse response decays to zero and
/// a step settles at the passband gain (1 for low-pass, 0 for high-pass).
#[test]
fn butterworth_stable_at_low_cutoff() {
    let fs = 192_000.0f32;
    for kind in KINDS {
        let p = params(kind, 8, 20.0);
        assert!(Butterworth::design(&p, fs).is_stable());
        let mut flt = filter(fs, 2, &p);
        // Step input for 3 s: LP settles at 1, HP at 0.
        let n = 3 * fs as usize;
        let mut x = vec![0.5f32; 2 * n];
        flt.process(&mut x);
        assert!(x.iter().all(|v| v.is_finite()));
        let last = x[2 * n - 1] as f64;
        match kind {
            FilterKind::LowPass => assert!((last - 0.5).abs() < 1e-5, "{last}"),
            FilterKind::HighPass => assert!(last.abs() < 1e-5, "{last}"),
        }
        // Silence afterwards decays to exactly zero (denormal flush).
        let mut flt = filter(fs, 1, &p);
        let h = impulse_response(&mut flt, 4 * fs as usize);
        let tail: f64 = h[3 * fs as usize..].iter().map(|v| (*v as f64).abs()).sum();
        assert!(tail < 1e-9, "{kind:?} tail {tail}");
    }
}

/// Every parameter at and beyond its extremes, every rate, several channel
/// counts: no NaN/Inf, no panic.
#[test]
fn butterworth_robust_at_extremes() {
    let cutoffs = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -1.0,
        0.0,
        1.0,
        20.0,
        24_000.0,
        1e9,
    ];
    let mut seed = 7u64;
    for fs in RATES {
        for ch in [1usize, 2, 8, 64] {
            let input = noise(512 * ch, &mut seed, 1.0);
            for &fc in &cutoffs {
                for order in [0usize, 1, 3, 8, 100] {
                    for kind in KINDS {
                        let p = params(kind, order, fc);
                        assert!(Butterworth::design(&p, fs).is_stable());
                        let mut flt = filter(fs, ch, &p);
                        let mut x = input.clone();
                        x[0] = 1.0;
                        flt.process(&mut x);
                        // Change everything mid-stream, too.
                        flt.set_params(&params(kind, 9 - order.clamp(1, 8), 1000.0));
                        flt.process(&mut x);
                        assert!(
                            x.iter().all(|v| v.is_finite()),
                            "{kind:?} n={order} fc={fc} fs={fs} ch={ch}"
                        );
                    }
                }
            }
        }
    }
    // Non-finite input does not poison later output.
    let mut flt = filter(48_000.0, 2, &params(FilterKind::LowPass, 8, 1000.0));
    let mut x = [f32::NAN, f32::INFINITY, 0.5, -0.5];
    flt.process(&mut x);
    let mut y = vec![0.25f32; 256];
    flt.process(&mut y);
    assert!(y.iter().all(|v| v.is_finite()));
    // Unconfigured: passthrough.
    let mut flt = Butterworth::new();
    let mut x = [0.5f32, -0.25];
    flt.process(&mut x);
    assert_eq!(x, [0.5, -0.25]);
}

/// Parameter-change click bound.
///
/// A sine of amplitude `a` at `f` runs through the filter while the
/// parameters change abruptly. A pure sine's largest sample step is
/// `2·a·sin(π·f/fs)`, and a Butterworth magnitude never exceeds 1. Bound:
/// no output step during or after the change exceeds
/// `1.25 × 2·a·sin(π·f/fs) + 1e-4`.
#[test]
fn butterworth_parameter_changes_do_not_click() {
    let fs = 48_000.0f32;
    let f = 1000.0;
    let a = 0.5;
    let scenarios = [
        (
            "LP cutoff 20k -> 300",
            params(FilterKind::LowPass, 4, 20_000.0),
            params(FilterKind::LowPass, 4, 300.0),
        ),
        (
            "HP cutoff 20 -> 5k",
            params(FilterKind::HighPass, 8, 20.0),
            params(FilterKind::HighPass, 8, 5000.0),
        ),
        (
            "order 2 -> 8",
            params(FilterKind::LowPass, 2, 1500.0),
            params(FilterKind::LowPass, 8, 1500.0),
        ),
        (
            "order 7 -> 1",
            params(FilterKind::HighPass, 7, 700.0),
            params(FilterKind::HighPass, 1, 700.0),
        ),
        (
            "LP -> HP",
            params(FilterKind::LowPass, 4, 1000.0),
            params(FilterKind::HighPass, 4, 1000.0),
        ),
        (
            "HP 30 -> LP 200 order 3 -> 6",
            params(FilterKind::HighPass, 3, 30.0),
            params(FilterKind::LowPass, 6, 200.0),
        ),
    ];
    for (name, before, after) in scenarios {
        let bound = 1.25 * 2.0 * a * (PI * f / fs as f64).sin() + 1e-4;
        let mut flt = filter(fs, 1, &before);
        let warm = 48_000;
        let mut x = sine(f, fs as f64, a, 0, warm);
        flt.process(&mut x);
        flt.set_params(&after);
        let mut y = sine(f, fs as f64, a, warm, 24_000);
        for chunk in y.chunks_mut(256) {
            flt.process(chunk);
        }
        let mut joined = vec![x[warm - 1]];
        joined.extend_from_slice(&y);
        let step = max_step(&joined);
        assert!(step <= bound, "{name}: step {step} > bound {bound}");
        // Ends at the new steady state.
        let tail = &y[12_000..];
        let peak = tail.iter().map(|v| v.abs() as f64).fold(0.0, f64::max);
        let s = Butterworth::design(&after, fs);
        let want = a * s.magnitude_at(f, fs as f64);
        assert!((peak - want).abs() < 0.002, "{name}: end {peak} vs {want}");
    }
}

/// Interleaved channels are filtered independently and identically.
#[test]
fn butterworth_multichannel_equals_mono() {
    let p = params(FilterKind::LowPass, 5, 3000.0);
    let mut seed = 3u64;
    for ch in [2usize, 3, 8] {
        let input = noise(1000 * ch, &mut seed, 0.5);
        let mut multi = filter(48_000.0, ch, &p);
        let mut x = input.clone();
        multi.process(&mut x);
        for c in 0..ch {
            let mut mono = filter(48_000.0, 1, &p);
            let mut m: Vec<f32> = input.iter().skip(c).step_by(ch).copied().collect();
            mono.process(&mut m);
            let got: Vec<f32> = x.iter().skip(c).step_by(ch).copied().collect();
            assert_eq!(got, m, "ch {ch} lane {c}");
        }
    }
}

/// `reset` clears the state: output after reset equals a fresh filter's.
#[test]
fn butterworth_reset_clears_state() {
    let p = params(FilterKind::HighPass, 6, 150.0);
    let mut seed = 11u64;
    let input = noise(2000, &mut seed, 0.5);
    let mut a = filter(44_100.0, 2, &p);
    let mut junk = noise(4000, &mut seed, 1.0);
    a.process(&mut junk);
    a.reset();
    let mut x = input.clone();
    a.process(&mut x);
    let mut b = filter(44_100.0, 2, &p);
    let mut y = input;
    b.process(&mut y);
    assert_eq!(x, y);
}

/// Detected SIMD level vs the scalar fallback: bit-identical when both use the
/// same FMA policy, otherwise within rounding.
#[test]
fn butterworth_simd_levels_agree() {
    let mut seed = 5u64;
    for ch in [1usize, 2, 3, 8, 64] {
        let input = noise(6000 * ch, &mut seed, 0.7);
        let run = |level: Level| {
            let mut flt = Butterworth::new();
            flt.set_simd_level(level);
            flt.configure(48_000.0, ch);
            flt.set_params(&params(FilterKind::LowPass, 8, 500.0));
            let mut x = input.clone();
            for (i, chunk) in x.chunks_mut(97 * ch).enumerate() {
                if i == 10 {
                    flt.set_params(&params(FilterKind::LowPass, 8, 9000.0));
                }
                if i == 30 {
                    flt.set_params(&params(FilterKind::HighPass, 3, 200.0));
                }
                flt.process(chunk);
            }
            x
        };
        let a = run(detect());
        let b = run(Level::fallback());
        if uses_fma(detect()) == uses_fma(Level::fallback()) {
            assert_eq!(a, b, "ch {ch}");
        } else {
            let d = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(d < 1e-5, "ch {ch}: {d}");
        }
    }
}

/// `set_params`, `process` and `reset` never allocate.
#[test]
fn butterworth_no_allocation() {
    let mut flt = filter(48_000.0, 8, &params(FilterKind::LowPass, 4, 1000.0));
    let mut seed = 9u64;
    let mut buf = noise(480 * 8, &mut seed, 0.5);
    let before = allocs();
    for i in 0..60 {
        flt.set_params(&params(
            KINDS[i % 2],
            1 + i % 8,
            20.0 + (i as f32 * 731.0) % 20_000.0,
        ));
        flt.process(&mut buf);
        if i % 17 == 16 {
            flt.reset();
        }
    }
    assert_eq!(allocs() - before, 0, "allocations in set_params/process");
}

// ---------------------------------------------------------------------------
// Crossover.

const LR_ORDERS: [LrOrder; 3] = [LrOrder::Lr2, LrOrder::Lr4, LrOrder::Lr8];

fn xparams(order: LrOrder, num_bands: usize, f: [f32; MAX_SPLITS]) -> CrossoverParams {
    CrossoverParams {
        order,
        num_bands,
        frequencies_hz: f,
    }
}

fn crossover(fs: f32, ch: usize, p: &CrossoverParams) -> Crossover {
    let mut x = Crossover::new();
    x.configure(fs, ch);
    x.set_params(p);
    x
}

fn new_bands(len: usize) -> Vec<Vec<f32>> {
    vec![vec![0.0f32; len]; MAX_CROSSOVER_BANDS]
}

/// Mono impulse responses of every band.
fn band_impulses(x: &mut Crossover, n: usize) -> Vec<Vec<f32>> {
    let mut input = vec![0.0f32; n];
    input[0] = 1.0;
    let mut bands = new_bands(n);
    x.process(&input, &mut bands);
    bands.truncate(x.num_bands());
    bands
}

fn sum_bands(bands: &[Vec<f32>]) -> Vec<f32> {
    let mut s = vec![0.0f64; bands[0].len()];
    for b in bands {
        for (s, &v) in s.iter_mut().zip(b) {
            *s += v as f64;
        }
    }
    s.into_iter().map(|v| v as f32).collect()
}

/// Two frequency sets: widely spaced, and closely spaced.
const SPLIT_SETS: [[f32; MAX_SPLITS]; 2] =
    [[100.0, 800.0, 3000.0, 9000.0], [400.0, 500.0, 630.0, 800.0]];

/// The plain sum of the bands is flat within ±0.05 dB from 20 Hz to 20 kHz,
/// for LR2/LR4/LR8 and 1–5 bands. It is in fact an all-pass, so the error is
/// rounding only: the test also asserts < 0.001 dB.
#[test]
fn crossover_band_sum_is_flat() {
    for fs in [44_100.0f32, 48_000.0, 96_000.0] {
        for order in LR_ORDERS {
            for nb in 1..=MAX_CROSSOVER_BANDS {
                for set in SPLIT_SETS {
                    let mut x = crossover(fs, 1, &xparams(order, nb, set));
                    let h = sum_bands(&band_impulses(&mut x, fs as usize));
                    let mut worst = 0.0f64;
                    for f in log_freqs(20.0, 20_000.0, 60) {
                        worst = worst.max(db(mag(dft_at(&h, f, fs as f64))).abs());
                    }
                    assert!(
                        worst < 0.05,
                        "{order:?} bands={nb} fs={fs} {set:?}: {worst} dB"
                    );
                    assert!(worst < 1e-3, "{order:?} bands={nb}: {worst} dB");
                }
            }
        }
    }
}

/// With one band, the output is the input, bit for bit.
#[test]
fn crossover_one_band_is_identity() {
    let mut seed = 1u64;
    let input = noise(3000, &mut seed, 0.9);
    let mut x = crossover(48_000.0, 3, &xparams(LrOrder::Lr8, 1, SPLIT_SETS[0]));
    let mut bands = new_bands(input.len());
    x.process(&input, &mut bands);
    assert_eq!(bands[0], input);
}

/// Both bands of a split are at −6.02 dB (`|1/√2|²`) at the crossover
/// frequency, for every order, and in phase (for LR2 because its high band
/// is polarity-inverted by definition), with the expected asymptotic slope.
/// In a 5-band split, every band matches the analytic product of its LR
/// factors, `Π_{j<k} |H_j|² · |L_k|²` (all-pass compensation has unit
/// magnitude), within 0.01 dB; with splits a decade apart every band edge is
/// at −6.02 dB within 0.05 dB for LR4/LR8 and 0.2 dB for LR2 (a neighbouring
/// LR2 split a decade away still costs `20·log10(1/(1 + 0.1²))` ≈ 0.086 dB).
#[test]
fn crossover_band_edges_are_at_minus_6_db() {
    let fs = 48_000.0f32;
    let edge = 20.0 * 0.5f64.log10();
    for order in LR_ORDERS {
        for fc in [100.0f32, 1000.0, 10_000.0] {
            let mut x = crossover(fs, 1, &xparams(order, 2, [fc, 0.0, 0.0, 0.0]));
            let b = band_impulses(&mut x, fs as usize);
            let lo = dft_at(&b[0], fc as f64, fs as f64);
            let hi = dft_at(&b[1], fc as f64, fs as f64);
            assert!((db(mag(lo)) - edge).abs() < 0.01, "{order:?} {fc}: low");
            assert!((db(mag(hi)) - edge).abs() < 0.01, "{order:?} {fc}: high");
            let phase = (lo.1.atan2(lo.0) - hi.1.atan2(hi.0)).abs();
            let phase = phase.min(2.0 * PI - phase);
            assert!(phase < 1e-3, "{order:?} {fc}: phase {phase}");
            // Low band, one octave from 2·fc to 4·fc: falls as the squared
            // analytic Butterworth (≈ 6.02·LR order dB, down to −96 dB for LR8).
            if fc <= 1000.0 {
                let n = order.butterworth_order();
                let (f2, f4) = (2.0 * fc as f64, 4.0 * fc as f64);
                let g2 = db(mag(dft_at(&b[0], f2, fs as f64)));
                let g4 = db(mag(dft_at(&b[0], f4, fs as f64)));
                let lp = |f| 2.0 * db(analytic(FilterKind::LowPass, n, fc as f64, f, fs as f64));
                let want = lp(f2) - lp(f4);
                assert!(
                    ((g2 - g4) - want).abs() < 0.05,
                    "{order:?} {fc}: slope {} vs {want}",
                    g2 - g4
                );
            }
        }
        let splits = [30.0f32, 300.0, 3000.0, 20_000.0];
        let mut x = crossover(fs, 1, &xparams(order, 5, splits));
        let b = band_impulses(&mut x, fs as usize);
        let n = order.butterworth_order();
        for (k, band) in b.iter().enumerate() {
            for f in log_freqs(10.0, 20_000.0, 50) {
                let mut want = 1.0;
                for &s in &splits[..k] {
                    want *= analytic(FilterKind::HighPass, n, s as f64, f, fs as f64).powi(2);
                }
                if k < 4 {
                    want *=
                        analytic(FilterKind::LowPass, n, splits[k] as f64, f, fs as f64).powi(2);
                }
                if want > 1e-4 {
                    let got = mag(dft_at(band, f, fs as f64));
                    assert!(
                        (db(got) - db(want)).abs() < 0.01,
                        "{order:?} band {k} at {f}: {} vs {}",
                        db(got),
                        db(want)
                    );
                }
            }
        }
        let tol = if order == LrOrder::Lr2 { 0.2 } else { 0.05 };
        for (k, band) in b.iter().enumerate() {
            let mut edges = Vec::new();
            if k > 0 {
                edges.push(splits[k - 1]);
            }
            if k < 4 {
                edges.push(splits[k]);
            }
            for f in edges {
                let g = db(mag(dft_at(band, f as f64, fs as f64)));
                assert!((g - edge).abs() < tol, "{order:?} band {k} at {f}: {g} dB");
            }
        }
    }
}

/// Stability at the lowest frequencies at 192 kHz with LR8: every band
/// decays to zero after an impulse and the sum stays flat.
#[test]
fn crossover_stable_at_low_frequencies() {
    let fs = 192_000.0f32;
    let p = xparams(LrOrder::Lr8, 5, [20.0, 25.0, 40.0, 80.0]);
    let mut x = crossover(fs, 1, &p);
    let b = band_impulses(&mut x, 4 * fs as usize);
    for (k, band) in b.iter().enumerate() {
        assert!(band.iter().all(|v| v.is_finite()));
        let tail: f64 = band[3 * fs as usize..]
            .iter()
            .map(|v| (*v as f64).abs())
            .sum();
        assert!(tail < 1e-9, "band {k} tail {tail}");
    }
    let h = sum_bands(&b);
    for f in log_freqs(10.0, 20_000.0, 40) {
        let g = db(mag(dft_at(&h, f, fs as f64)));
        assert!(g.abs() < 0.05, "{f}: {g}");
    }
}

/// Every parameter at and beyond its extremes, every rate, several channel
/// counts: no NaN/Inf, no panic; frequencies stay clamped and ordered.
#[test]
fn crossover_robust_at_extremes() {
    let sets = [
        [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0],
        [0.0, 1.0, 1e9, 5.0],
        [24_000.0, 100.0, 50.0, 20.0],
        [1.0, 1.0, 1.0, 1.0],
    ];
    let mut seed = 13u64;
    for fs in RATES {
        for ch in [1usize, 2, 8, 64] {
            let mut input = noise(300 * ch, &mut seed, 1.0);
            input[0] = 1.0;
            for set in sets {
                for order in LR_ORDERS {
                    for nb in [0usize, 1, 3, 5, 100] {
                        let mut x = crossover(fs, ch, &xparams(order, nb, set));
                        let mut bands = new_bands(input.len());
                        x.process(&input, &mut bands);
                        let next = xparams(LR_ORDERS[(nb + 1) % 3], 6 - x.num_bands(), set);
                        x.set_params(&next);
                        x.process(&input, &mut bands);
                        for b in &bands {
                            assert!(
                                b.iter().all(|v| v.is_finite()),
                                "{order:?} nb={nb} {set:?} fs={fs} ch={ch}"
                            );
                        }
                        let f = x.frequencies_hz();
                        assert!(f.windows(2).all(|w| w[0] <= w[1]), "{f:?}");
                        assert!(f
                            .iter()
                            .all(|&v| (1.0..=0.499 * fs as f64 * (1.0 + 1e-12)).contains(&v)));
                    }
                }
            }
        }
    }
    // Non-finite input does not poison later output.
    let mut x = crossover(48_000.0, 2, &CrossoverParams::default());
    let mut bands = new_bands(256);
    x.process(&[f32::NAN, f32::INFINITY, 0.5, -0.5], &mut bands);
    x.process(&[0.25f32; 256], &mut bands);
    assert!(bands.iter().flatten().all(|v| v.is_finite()));
    // Unconfigured: band 0 is the input, the others are zero.
    let mut x = Crossover::new();
    let mut bands = vec![vec![9.0f32; 4]; 3];
    x.process(&[0.5, -0.25, 0.125, 1.0], &mut bands);
    assert_eq!(bands[0], [0.5, -0.25, 0.125, 1.0]);
    assert!(bands[1..].iter().flatten().all(|&v| v == 0.0));
}

/// Structural changes faster than the crossfade (every 7-frame buffer): the
/// realised band count always follows `num_bands()`, nothing panics, and once
/// the changes stop the bands sum flat again.
#[test]
fn crossover_rapid_structural_changes() {
    let mut x = crossover(48_000.0, 2, &CrossoverParams::default());
    let mut seed = 31u64;
    let input = noise(7 * 2, &mut seed, 0.5);
    for i in 0..400 {
        let nb = 1 + (i * 3) % MAX_CROSSOVER_BANDS;
        x.set_params(&xparams(LR_ORDERS[i % 3], nb, SPLIT_SETS[i % 2]));
        assert_eq!(x.num_bands(), nb);
        let mut bands = vec![vec![f32::NAN; input.len()]; nb];
        x.process(&input, &mut bands);
        assert!(bands.iter().flatten().all(|v| v.is_finite()), "call {i}");
    }
    x.set_params(&xparams(LrOrder::Lr8, 4, SPLIT_SETS[0]));
    let mut quiet = new_bands(4800 * 2);
    x.process(&vec![0.0; 4800 * 2], &mut quiet);
    let mut input = vec![0.0f32; 48_000 * 2];
    input[0] = 1.0;
    let mut bands = new_bands(input.len());
    x.process(&input, &mut bands);
    let left: Vec<Vec<f32>> = bands[..4]
        .iter()
        .map(|b| b.iter().step_by(2).copied().collect())
        .collect();
    let h = sum_bands(&left);
    for f in log_freqs(20.0, 20_000.0, 30) {
        assert!(db(mag(dft_at(&h, f, 48_000.0))).abs() < 0.05, "{f}");
    }
}

#[test]
#[should_panic(expected = "band buffers needed")]
fn crossover_panics_on_too_few_band_buffers() {
    let mut x = crossover(48_000.0, 1, &xparams(LrOrder::Lr4, 3, SPLIT_SETS[0]));
    let mut bands = vec![vec![0.0f32; 16]; 2];
    x.process(&[0.0; 16], &mut bands);
}

/// Parameter-change click bound, as for the Butterworth filter: every band is
/// an LR low-pass/high-pass product times all-passes, so its magnitude is at
/// most 1, and the sum is an all-pass. Bound: no step of any band, nor of
/// the band sum, exceeds `1.25 × 2·a·sin(π·f/fs) + 1e-4` during or after a
/// change (frequency glide, order change, band-count increase). A band-count
/// *decrease* keeps only the sum continuous: the new top band absorbs the
/// dropped bands at the start of the crossfade (documented behaviour), so
/// only the sum is checked there.
#[test]
fn crossover_parameter_changes_do_not_click() {
    let fs = 48_000.0f32;
    let a = 0.5;
    let scenarios = [
        (
            "frequency glide",
            xparams(LrOrder::Lr4, 3, [200.0, 2000.0, 0.0, 0.0]),
            xparams(LrOrder::Lr4, 3, [900.0, 6000.0, 0.0, 0.0]),
        ),
        (
            "LR4 -> LR8",
            xparams(LrOrder::Lr4, 3, [500.0, 1500.0, 0.0, 0.0]),
            xparams(LrOrder::Lr8, 3, [500.0, 1500.0, 0.0, 0.0]),
        ),
        (
            "LR2 -> LR8",
            xparams(LrOrder::Lr2, 2, [1000.0, 0.0, 0.0, 0.0]),
            xparams(LrOrder::Lr8, 2, [1000.0, 0.0, 0.0, 0.0]),
        ),
        (
            "3 -> 5 bands",
            xparams(LrOrder::Lr4, 3, [300.0, 1200.0, 4000.0, 8000.0]),
            xparams(LrOrder::Lr4, 5, [300.0, 1200.0, 4000.0, 8000.0]),
        ),
        (
            "5 -> 2 bands",
            xparams(LrOrder::Lr8, 5, [300.0, 900.0, 1100.0, 8000.0]),
            xparams(LrOrder::Lr8, 2, [300.0, 900.0, 1100.0, 8000.0]),
        ),
    ];
    for f in [1000.0, 300.0] {
        let bound = 1.25 * 2.0 * a * (PI * f / fs as f64).sin() + 1e-4;
        for (name, before, after) in scenarios {
            let mut x = crossover(fs, 1, &before);
            let warm = 48_128; // a multiple of the 256-frame buffer
            let input = sine(f, fs as f64, a, 0, warm + 24_000);
            let mut bands = new_bands(input.len());
            let mut buf = new_bands(256);
            let mut pos = 0;
            while pos < input.len() {
                if pos == warm {
                    x.set_params(&after);
                }
                let end = (pos + 256).min(input.len());
                x.process(&input[pos..end], &mut buf);
                for (b, o) in bands.iter_mut().zip(&buf) {
                    b[pos..end].copy_from_slice(&o[..end - pos]);
                }
                pos = end;
            }
            let from = warm - 4800;
            let checked = if after.num_bands < before.num_bands {
                0
            } else {
                after.num_bands
            };
            for (k, b) in bands.iter().take(checked).enumerate() {
                let step = max_step(&b[from..]);
                assert!(
                    step <= bound,
                    "{name} f={f}: band {k} step {step} > {bound}"
                );
            }
            let sum = sum_bands(&bands[..after.num_bands]);
            let step = max_step(&sum[warm..]);
            assert!(step <= bound, "{name} f={f}: sum step {step} > {bound}");
            // The sum ends at unity gain again.
            let peak = sum[warm + 12_000..]
                .iter()
                .map(|v| v.abs() as f64)
                .fold(0.0, f64::max);
            assert!((peak - a).abs() < 1e-3, "{name} f={f}: end {peak}");
        }
    }
}

/// Interleaved channels are split independently and identically.
#[test]
fn crossover_multichannel_equals_mono() {
    let p = xparams(LrOrder::Lr8, 5, SPLIT_SETS[0]);
    let mut seed = 21u64;
    for ch in [2usize, 3, 8] {
        let input = noise(700 * ch, &mut seed, 0.5);
        let mut multi = crossover(48_000.0, ch, &p);
        let mut bands = new_bands(input.len());
        multi.process(&input, &mut bands);
        for c in 0..ch {
            let mono_in: Vec<f32> = input.iter().skip(c).step_by(ch).copied().collect();
            let mut mono = crossover(48_000.0, 1, &p);
            let mut mb = new_bands(mono_in.len());
            mono.process(&mono_in, &mut mb);
            for (k, band) in bands.iter().enumerate() {
                let got: Vec<f32> = band.iter().skip(c).step_by(ch).copied().collect();
                assert_eq!(got, mb[k], "ch {ch} lane {c} band {k}");
            }
        }
    }
}

/// `reset` clears the state: output after reset equals a fresh crossover's.
#[test]
fn crossover_reset_clears_state() {
    let p = xparams(LrOrder::Lr4, 4, SPLIT_SETS[0]);
    let mut seed = 23u64;
    let input = noise(2000, &mut seed, 0.5);
    let mut a = crossover(48_000.0, 2, &p);
    let mut ba = new_bands(input.len());
    a.process(&noise(2000, &mut seed, 1.0), &mut ba);
    a.reset();
    a.process(&input, &mut ba);
    let mut b = crossover(48_000.0, 2, &p);
    let mut bb = new_bands(input.len());
    b.process(&input, &mut bb);
    assert_eq!(ba, bb);
}

/// Detected SIMD level vs the scalar fallback, through glides and structural
/// changes: bit-identical when both use the same FMA policy.
#[test]
fn crossover_simd_levels_agree() {
    let mut seed = 25u64;
    for ch in [1usize, 2, 3, 8, 64] {
        let input = noise(5000 * ch, &mut seed, 0.7);
        let run = |level: Level| {
            let mut x = Crossover::new();
            x.set_simd_level(level);
            x.configure(48_000.0, ch);
            x.set_params(&xparams(LrOrder::Lr4, 3, SPLIT_SETS[0]));
            let mut out = new_bands(input.len());
            let mut buf = new_bands(101 * ch);
            for (i, chunk) in input.chunks(101 * ch).enumerate() {
                match i {
                    10 => x.set_params(&xparams(LrOrder::Lr4, 3, SPLIT_SETS[1])),
                    20 => x.set_params(&xparams(LrOrder::Lr8, 5, SPLIT_SETS[0])),
                    30 => x.set_params(&xparams(LrOrder::Lr2, 4, SPLIT_SETS[1])),
                    _ => {}
                }
                x.process(chunk, &mut buf);
                let at = i * 101 * ch;
                for (o, b) in out.iter_mut().zip(&buf) {
                    o[at..at + chunk.len()].copy_from_slice(&b[..chunk.len()]);
                }
            }
            out
        };
        let a = run(detect());
        let b = run(Level::fallback());
        if uses_fma(detect()) == uses_fma(Level::fallback()) {
            assert_eq!(a, b, "ch {ch}");
        } else {
            let d = a
                .iter()
                .flatten()
                .zip(b.iter().flatten())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(d < 1e-5, "ch {ch}: {d}");
        }
    }
}

/// `set_params`, `process` and `reset` never allocate.
#[test]
fn crossover_no_allocation() {
    let mut x = crossover(48_000.0, 8, &CrossoverParams::default());
    let mut seed = 27u64;
    let input = noise(480 * 8, &mut seed, 0.5);
    let mut bands = new_bands(input.len());
    let before = allocs();
    for i in 0..60 {
        x.set_params(&xparams(
            LR_ORDERS[i % 3],
            1 + i % MAX_CROSSOVER_BANDS,
            [
                20.0 + i as f32 * 10.0,
                500.0 + i as f32 * 50.0,
                3000.0,
                10_000.0 - i as f32 * 20.0,
            ],
        ));
        x.process(&input, &mut bands);
        if i % 17 == 16 {
            x.reset();
        }
    }
    assert_eq!(allocs() - before, 0, "allocations in set_params/process");
}
