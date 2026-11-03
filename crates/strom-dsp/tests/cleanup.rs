//! Input-cleanup tests, verified against theory with steady sines at 44.1, 48
//! and 96 kHz: hum notches attenuate every enabled harmonic by ≥ 40 dB and the
//! passband between them stays within 0.5 dB; DC is removed; the high-pass
//! matches the analytic Butterworth magnitude (and so its slope) for every
//! order; defaults are a bit-exact passthrough; robustness at the parameter
//! extremes; click-free switching; zero allocations in the processing path.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::f64::consts::PI;

use strom_dsp::cleanup::{Cleanup, CleanupParams, HumFrequency, DC_BLOCK_HZ, HUM_Q_RANGE};
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

const RATES: [f32; 3] = [44_100.0, 48_000.0, 96_000.0];

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-300).log10()
}

fn cleanup(fs: f32, ch: usize, p: CleanupParams) -> Cleanup {
    let mut c = Cleanup::new();
    c.set_params(&p);
    c.configure(fs, ch);
    c
}

/// Amplitude of the component at `f` in `x` (projection onto sin/cos over a
/// whole number of periods of `f`, taken from the end of `x`; `offset` is the
/// sample index of `x[0]`).
fn amplitude_at(x: &[f32], f: f64, fs: f64, offset: usize) -> f64 {
    let periods = (x.len() as f64 * f / fs).floor().max(1.0);
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

/// Steady-state gain (dB) at each frequency in `freqs`: one channel per
/// frequency, a sine of amplitude `a`, `settle_s` seconds of settling, then
/// measured over `measure_s` seconds.
fn measured_gains_db(
    p: CleanupParams,
    fs: f32,
    freqs: &[f64],
    a: f64,
    settle_s: f64,
    measure_s: f64,
) -> Vec<f64> {
    let ch = freqs.len();
    let mut c = cleanup(fs, ch, p);
    let n_settle = (settle_s * fs as f64) as usize;
    let n_meas = (measure_s * fs as f64) as usize;
    let gen = |start: usize, len: usize| {
        let mut x = vec![0.0f32; len * ch];
        for i in 0..len {
            for (k, &f) in freqs.iter().enumerate() {
                x[i * ch + k] = (a * (2.0 * PI * f * (start + i) as f64 / fs as f64).sin()) as f32;
            }
        }
        x
    };
    for chunk_start in (0..n_settle).step_by(4096) {
        let len = 4096.min(n_settle - chunk_start);
        let mut x = gen(chunk_start, len);
        c.process(&mut x);
    }
    let mut y = gen(n_settle, n_meas);
    c.process(&mut y);
    (0..ch)
        .map(|k| {
            let yk: Vec<f32> = y.iter().skip(k).step_by(ch).copied().collect();
            db(amplitude_at(&yk, freqs[k], fs as f64, n_settle) / a)
        })
        .collect()
}

fn hum(h: HumFrequency, harmonics: usize, q: f32) -> CleanupParams {
    CleanupParams {
        hum: h,
        hum_harmonics: harmonics,
        hum_q: q,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Hum notches.

/// Every enabled harmonic is attenuated by at least 40 dB (steady sines), and
/// harmonics above the selected count are not.
#[test]
fn hum_attenuates_every_enabled_harmonic() {
    for fs in RATES {
        for h in [HumFrequency::Hz50, HumFrequency::Hz60] {
            let f0 = h.hz().unwrap();
            for (harmonics, q) in [(10, 10.0f32), (10, HUM_Q_RANGE.0), (3, 30.0)] {
                let freqs: Vec<f64> = (1..=10).map(|k| k as f64 * f0).collect();
                // Settling: notch bandwidth B = f0/q, envelope τ = 1/(π·B).
                let tau = q as f64 / (PI * f0);
                let g = measured_gains_db(hum(h, harmonics, q), fs, &freqs, 0.5, 14.0 * tau, 0.5);
                for (k, &gk) in g.iter().enumerate() {
                    let f = freqs[k];
                    if k < harmonics {
                        assert!(gk <= -40.0, "{fs} {f} Hz (q {q}): only {gk} dB");
                    } else {
                        // Not notched; also in the passband (≤ 0.5 dB loss).
                        assert!(gk > -0.5, "{fs} {f} Hz (q {q}, {harmonics} harm.): {gk} dB");
                    }
                }
            }
        }
    }
}

/// The passband between the notches (midpoints and third-points, and half a
/// spacing outside the outermost notches) stays within 0.5 dB, measured with
/// steady sines, at the lowest Q and the default Q; and the measurement agrees
/// with the analytic magnitude of the realised comb.
#[test]
fn hum_passband_between_notches_is_flat() {
    for fs in RATES {
        for h in [HumFrequency::Hz50, HumFrequency::Hz60] {
            let f0 = h.hz().unwrap();
            for q in [HUM_Q_RANGE.0, 10.0] {
                let p = hum(h, 10, q);
                let mut freqs = vec![0.5 * f0, 10.5 * f0];
                for k in 1..10 {
                    for frac in [1.0 / 3.0, 0.5, 2.0 / 3.0] {
                        freqs.push((k as f64 + frac) * f0);
                    }
                }
                let tau = q as f64 / (PI * f0);
                let g = measured_gains_db(p, fs, &freqs, 0.5, 14.0 * tau, 0.5);
                let design = Cleanup::hum_design(&p, fs);
                assert_eq!(design.as_slice().len(), 10);
                for (k, &gk) in g.iter().enumerate() {
                    let f = freqs[k];
                    assert!(gk.abs() <= 0.5, "{fs} q {q}: {f} Hz at {gk} dB");
                    let want = db(design.magnitude_at(f, fs as f64));
                    assert!((gk - want).abs() < 0.02, "{fs} {f} Hz: {gk} vs {want}");
                }
            }
        }
    }
}

/// Notches at or above 0.49·fs are skipped (the design simply has fewer
/// sections) and the comb stays stable and unity at DC.
#[test]
fn hum_notches_above_nyquist_are_skipped() {
    // 60 Hz × 10 = 600 Hz; at fs = 1 kHz only harmonics below 490 Hz remain.
    let p = hum(HumFrequency::Hz60, 10, 10.0);
    let d = Cleanup::hum_design(&p, 1_000.0);
    assert_eq!(d.as_slice().len(), 8);
    assert!(d.as_slice().iter().all(|c| c.is_stable()));
    assert!((d.magnitude_at(1e-3, 1_000.0) - 1.0).abs() < 1e-9);
    let mut c = cleanup(1_000.0, 1, p);
    let mut x: Vec<f32> = (0..5_000).map(|i| ((i as f32) * 0.37).sin()).collect();
    c.process(&mut x);
    assert!(x.iter().all(|v| v.is_finite() && v.abs() < 4.0));
    // Off: no notches.
    assert!(Cleanup::hum_design(&CleanupParams::default(), 48_000.0)
        .as_slice()
        .is_empty());
}

// ---------------------------------------------------------------------------
// DC blocker.

/// A DC offset is removed (to below −100 dB of its value after settling) while
/// a 1 kHz tone passes unchanged; the corner is at `DC_BLOCK_HZ` (−3.01 dB).
#[test]
fn dc_is_removed() {
    for fs in RATES {
        let p = CleanupParams {
            dc_block: true,
            ..Default::default()
        };
        let mut c = cleanup(fs, 1, p);
        let n = fs as usize;
        let mut x: Vec<f32> = (0..n)
            .map(|i| (0.5 + 0.25 * (2.0 * PI * 1_000.0 * i as f64 / fs as f64).sin()) as f32)
            .collect();
        c.process(&mut x);
        let tail = &x[n / 2..];
        let mean = tail.iter().map(|&v| v as f64).sum::<f64>() / tail.len() as f64;
        assert!(db(mean.abs() / 0.5) < -100.0, "{fs}: residual DC {mean}");
        let g = db(amplitude_at(tail, 1_000.0, fs as f64, n / 2) / 0.25);
        assert!(g.abs() < 0.01, "{fs}: 1 kHz at {g} dB");
        // Corner frequency and the analytic first-order response.
        let freqs = [DC_BLOCK_HZ as f64, 20.0, 50.0];
        let g = measured_gains_db(p, fs, &freqs, 0.5, 2.0, 1.0);
        for (k, &f) in freqs.iter().enumerate() {
            let want = db(butterworth_hp(f, DC_BLOCK_HZ as f64, 1, fs as f64));
            assert!(
                (g[k] - want).abs() < 0.05,
                "{fs} {f} Hz: {} vs {want}",
                g[k]
            );
        }
        assert!((g[0] + 3.0103).abs() < 0.05, "{fs}: corner at {} dB", g[0]);
    }
}

// ---------------------------------------------------------------------------
// High-pass.

/// Analytic magnitude of the digital Butterworth high-pass (bilinear transform
/// pre-warped at `fc`): `1/√(1 + (tan(π fc/fs)/tan(π f/fs))^(2n))`.
fn butterworth_hp(f: f64, fc: f64, order: usize, fs: f64) -> f64 {
    let r = (PI * fc / fs).tan() / (PI * f / fs).tan();
    1.0 / (1.0 + r.powi(2 * order as i32)).sqrt()
}

/// For every order the measured response matches the analytic Butterworth
/// magnitude (0.1 dB), and the slope between 25 and 50 Hz (one octave) matches
/// the analytic slope, which tends to 6.02·order dB/octave.
#[test]
fn highpass_slope_matches_order() {
    let fc = 100.0;
    let freqs = [25.0, 50.0, 100.0, 400.0, 2_000.0];
    for fs in RATES {
        for order in 1..=8 {
            let p = CleanupParams {
                highpass_hz: fc as f32,
                highpass_order: order,
                ..Default::default()
            };
            let g = measured_gains_db(p, fs, &freqs, 0.5, 1.0, 0.5);
            for (k, &f) in freqs.iter().enumerate() {
                let want = db(butterworth_hp(f, fc, order, fs as f64));
                assert!(
                    (g[k] - want).abs() < 0.1,
                    "{fs} order {order} {f} Hz: {} vs {want}",
                    g[k]
                );
            }
            let slope = g[1] - g[0];
            let want = db(butterworth_hp(50.0, fc, order, fs as f64))
                - db(butterworth_hp(25.0, fc, order, fs as f64));
            assert!(
                (slope - want).abs() < 0.1,
                "{fs} order {order}: slope {slope} vs {want}"
            );
            // Far below the corner the slope approaches 6.02·n dB/octave.
            assert!(
                (want - 6.0206 * order as f64).abs() < 0.8,
                "{order}: {want}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Defaults, combinations, robustness.

/// The default (everything off) leaves the samples bit-exact, also after
/// stages were switched on and off again (once their fade-out has finished).
#[test]
fn default_is_bit_exact_passthrough() {
    let x: Vec<f32> = (0..9_600)
        .map(|i| ((i as f32) * 0.123).sin() * 0.7 + 0.1)
        .collect();
    let mut c = cleanup(48_000.0, 2, CleanupParams::default());
    let mut y = x.clone();
    c.process(&mut y);
    assert_eq!(x, y);
    c.set_params(&CleanupParams {
        dc_block: true,
        highpass_hz: 80.0,
        hum: HumFrequency::Hz50,
        ..Default::default()
    });
    let mut y = x.clone();
    c.process(&mut y);
    assert_ne!(x, y);
    c.set_params(&CleanupParams::default());
    let mut y = x.clone();
    c.process(&mut y); // fade-out (10 ms) happens here
    let mut y = x.clone();
    c.process(&mut y);
    assert_eq!(x, y);
}

/// All stages at once: DC removed, hum notched, high-pass in place, the
/// passband above it unchanged.
#[test]
fn all_stages_together() {
    let fs = 48_000.0;
    let p = CleanupParams {
        dc_block: true,
        highpass_hz: 40.0,
        highpass_order: 4,
        hum: HumFrequency::Hz50,
        hum_harmonics: 10,
        hum_q: 10.0,
    };
    let freqs = [150.0, 175.0, 1_000.0];
    let g = measured_gains_db(p, fs, &freqs, 0.5, 2.0, 0.5);
    assert!(g[0] < -40.0, "150 Hz {}", g[0]);
    assert!(g[1].abs() < 0.5, "175 Hz {}", g[1]);
    assert!(g[2].abs() < 0.01, "1 kHz {}", g[2]);
}

#[test]
fn robust_at_parameter_extremes() {
    let sets = [
        CleanupParams {
            dc_block: true,
            highpass_hz: 1_000.0,
            highpass_order: 8,
            hum: HumFrequency::Hz60,
            hum_harmonics: 10,
            hum_q: 100.0,
        },
        CleanupParams {
            dc_block: true,
            highpass_hz: 0.001,
            highpass_order: 1,
            hum: HumFrequency::Hz50,
            hum_harmonics: 1,
            hum_q: HUM_Q_RANGE.0,
        },
        CleanupParams {
            dc_block: false,
            highpass_hz: f32::NAN,
            highpass_order: 0,
            hum: HumFrequency::Hz50,
            hum_harmonics: 1000,
            hum_q: f32::INFINITY,
        },
    ];
    for fs in [8_000.0f32, 44_100.0, 48_000.0, 96_000.0, 192_000.0] {
        for ch in [1usize, 2, 8, 64] {
            for p in &sets {
                let mut c = cleanup(fs, ch, *p);
                let mut seed = 0x1234_5678u32;
                let mut x: Vec<f32> = (0..2048 * ch)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 17;
                        seed ^= seed << 5;
                        (seed as f32 / u32::MAX as f32 - 0.5) * 4.0 + 1.0
                    })
                    .collect();
                c.process(&mut x);
                assert!(x.iter().all(|v| v.is_finite()), "{fs} {ch} {p:?}");
                let mut silence = vec![0.0f32; 4096 * ch];
                c.process(&mut silence);
                assert!(silence.iter().all(|v| v.is_finite()));
            }
        }
    }
    // Unconfigured: passthrough; NaN input does not poison the state.
    let mut c = Cleanup::new();
    let mut x = [0.5f32, -0.25];
    c.process(&mut x);
    assert_eq!(x, [0.5, -0.25]);
    let mut c = cleanup(48_000.0, 1, sets[0]);
    let mut x = [f32::NAN, f32::INFINITY, 0.5, -0.5];
    c.process(&mut x);
    let mut y = vec![0.25f32; 4800];
    c.process(&mut y);
    assert!(y.iter().all(|v| v.is_finite()));
}

/// Parameter-change click bound.
///
/// A 1 kHz sine of amplitude `a = 0.5` runs while the stages switch on and
/// off, the high-pass cutoff and order jump, and the hum frequency, harmonic
/// count and Q change, every 20 ms. Every stage has magnitude ≤ 1 at 1 kHz
/// and the switches crossfade over 10 ms. A pure sine's largest sample step
/// is `2·a·sin(π·f/fs)`. Bound: no output step exceeds `1.25 ×` that.
/// Sanity check: an unsmoothed jump to silence would exceed it.
#[test]
fn parameter_changes_do_not_click() {
    let fs = 48_000.0f32;
    let f = 1_000.0;
    let a = 0.5;
    let sets = [
        CleanupParams::default(),
        CleanupParams {
            dc_block: true,
            highpass_hz: 300.0,
            highpass_order: 8,
            hum: HumFrequency::Hz50,
            hum_harmonics: 10,
            hum_q: 6.0,
        },
        CleanupParams {
            dc_block: false,
            highpass_hz: 20.0,
            highpass_order: 2,
            hum: HumFrequency::Hz60,
            hum_harmonics: 3,
            hum_q: 100.0,
        },
        CleanupParams {
            dc_block: true,
            highpass_hz: 0.0,
            hum: HumFrequency::Hz60,
            hum_harmonics: 10,
            hum_q: 6.0,
            ..Default::default()
        },
        CleanupParams {
            highpass_hz: 500.0,
            highpass_order: 1,
            ..Default::default()
        },
    ];
    let bound = 1.25 * 2.0 * a * (PI * f / fs as f64).sin();
    let mut c = cleanup(fs, 1, sets[0]);
    let block = 960;
    let mut prev = 0.0f32;
    let mut max_step: f64 = 0.0;
    for i in 0..60 {
        c.set_params(&sets[(i * 3) % sets.len()]);
        let mut x: Vec<f32> = (i * block..(i + 1) * block)
            .map(|n| (a * (2.0 * PI * f * n as f64 / fs as f64).sin()) as f32)
            .collect();
        for ch in x.chunks_mut(200) {
            c.process(ch);
        }
        for &v in &x {
            max_step = max_step.max((v - prev).abs() as f64);
            prev = v;
        }
    }
    assert!(max_step <= bound, "max step {max_step} > bound {bound}");
    assert!(bound < a, "bound {bound} not restrictive");
}

/// `reset` makes the processor behave like a fresh instance.
#[test]
fn reset_matches_fresh_instance() {
    let p = CleanupParams {
        dc_block: true,
        highpass_hz: 80.0,
        hum: HumFrequency::Hz50,
        ..Default::default()
    };
    let x: Vec<f32> = (0..4_800)
        .map(|i| ((i as f32) * 0.05).sin() + 0.2)
        .collect();
    let mut c = cleanup(48_000.0, 1, CleanupParams::default());
    let mut warm = x.clone();
    c.process(&mut warm);
    c.set_params(&p);
    c.process(&mut warm);
    c.reset();
    let mut a = x.clone();
    c.process(&mut a);
    let mut fresh = cleanup(48_000.0, 1, p);
    let mut b = x.clone();
    fresh.process(&mut b);
    assert_eq!(a, b);
}

// ---------------------------------------------------------------------------
// Real-time safety.

/// `set_params`, `process` and `reset` never allocate, including switch
/// crossfades and hum crossfades.
#[test]
fn no_allocation_in_processing() {
    for ch in [1usize, 2, 6] {
        let mut c = cleanup(48_000.0, ch, CleanupParams::default());
        let src: Vec<f32> = (0..1024 * ch).map(|i| ((i as f32) * 0.01).sin()).collect();
        let mut x = vec![0.0f32; 1024 * ch];
        let before = allocs();
        for i in 0..200 {
            let p = CleanupParams {
                dc_block: i % 2 == 0,
                highpass_hz: [0.0, 40.0, 120.0][i % 3],
                highpass_order: 1 + i % 8,
                hum: [HumFrequency::Off, HumFrequency::Hz50, HumFrequency::Hz60][i % 3],
                hum_harmonics: 1 + i % 10,
                hum_q: 6.0 + (i % 50) as f32,
            };
            c.set_params(&p);
            let len = [1024, 1, 480, 7][i % 4] * ch;
            x[..len].copy_from_slice(&src[..len]);
            c.process(&mut x[..len]);
            if i % 50 == 49 {
                c.reset();
            }
        }
        assert_eq!(allocs() - before, 0, "allocations in set_params/process");
    }
}
