//! Butterworth filter tests, verified against theory: the analytic magnitude
//! of the bilinear-transformed (pre-warped) Butterworth prototype,
//! `1/√(1 + (tan(πf/fs)/tan(πfc/fs))^(2n))`, computed independently here;
//! stability, robustness, click-free parameter changes, SIMD-level
//! equivalence and real-time safety.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::f64::consts::PI;

use strom_dsp::biquad::uses_fma;
use strom_dsp::filter::{Butterworth, ButterworthParams, FilterKind};
use strom_dsp::simd::{detect, Level};
use strom_dsp::Processor;

// ---------------------------------------------------------------------------
// Counting allocator (per thread, so parallel tests don't interfere).

struct CountingAlloc;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn allocs() -> usize {
    ALLOCS.with(|c| c.get())
}

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
