//! Compressor additions after the listening tests: detector mode (peak /
//! sine-calibrated RMS with an rms-window) and auto-makeup. Static curves
//! against the documented formulas, RMS timing, switching while running,
//! and zero allocations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use strom_dsp::compressor::{CompressorDetector, DEFAULT_RMS_WINDOW_MS, RMS_WINDOW_RANGE};
use strom_dsp::util::lin_to_db;
use strom_dsp::{Compressor, CompressorParams, Processor};

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

fn alloc_count() -> usize {
    ALLOCS.with(|c| c.get())
}

const FS: f32 = 48_000.0;

fn ms(n_ms: f64) -> usize {
    (n_ms * 0.001 * FS as f64).round() as usize
}

/// Static curve per phase 1 spec §3.3 (Giannoulis quadratic knee), output
/// level in dB for a detector level `x` dB, without makeup.
fn curve_out_db(x: f64, p: &CompressorParams) -> f64 {
    let k = (p.knee as f64).max(0.001);
    let t = lin_to_db(p.threshold as f64);
    let w = -40.0 * k.log10();
    let r = p.ratio as f64;
    if 2.0 * (x - t) < -w {
        x
    } else if 2.0 * (x - t).abs() <= w && w > 0.0 {
        x + (1.0 / r - 1.0) * (x - t + w / 2.0).powi(2) / (2.0 * w)
    } else {
        t + (x - t) / r
    }
}

/// Documented RMS reading of a steady sine of frequency `f` (peak level
/// `peak_db`) with averaging time `tau_ms`: the hold takes the maximum of
/// the mean square, `10·log10(1 + r)` above the peak,
/// `r = 1/sqrt(1 + (4π f τ)²)`.
fn rms_sine_reading_db(peak_db: f64, f: f64, tau_ms: f64) -> f64 {
    let x = 4.0 * std::f64::consts::PI * f * tau_ms * 1e-3;
    peak_db + 10.0 * (1.0 + 1.0 / (1.0 + x * x).sqrt()).log10()
}

/// 1 kHz sine at 48 kHz whose samples hit the exact peak.
fn sine_1k(amp: f64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (amp * (2.0 * std::f64::consts::PI * i as f64 / 48.0).sin()) as f32)
        .collect()
}

fn settings() -> Vec<CompressorParams> {
    let slow = CompressorParams {
        attack_ms: 0.01,
        release_ms: 10_000.0,
        ..Default::default()
    };
    vec![
        CompressorParams {
            threshold: 0.5,
            ratio: 4.0,
            ..slow
        },
        CompressorParams {
            threshold: 0.1,
            ratio: 4.0,
            knee: 0.5,
            ..slow
        },
        CompressorParams {
            threshold: 0.1,
            ratio: 100.0,
            knee: 1.0,
            ..slow
        },
        CompressorParams {
            threshold: 0.05,
            ratio: 2.0,
            knee: 0.25,
            makeup_gain: 2.0,
            ..slow
        },
    ]
}

/// Steady 1 kHz sines, both detectors: the output peak matches the static
/// curve at the documented reading within 0.1 dB.
#[test]
fn static_curve_both_detectors() {
    for det in [CompressorDetector::Peak, CompressorDetector::Rms] {
        for p in settings() {
            let mut level = -60.0;
            while level <= 0.0 + 1e-9 {
                let mut c = Compressor::new();
                c.set_detector(det);
                c.set_params(&p);
                c.configure(FS, 1);
                let amp = 10f64.powf(level / 20.0);
                let mut buf = sine_1k(amp, ms(300.0));
                c.process(&mut buf);
                let peak = buf[buf.len() - 96..]
                    .iter()
                    .fold(0.0f64, |m, &s| m.max(s.abs() as f64));
                let x = match det {
                    CompressorDetector::Peak => level,
                    CompressorDetector::Rms => {
                        rms_sine_reading_db(level, 1000.0, DEFAULT_RMS_WINDOW_MS as f64)
                    }
                };
                let want = level + (curve_out_db(x, &p) - x) + lin_to_db(p.makeup_gain as f64);
                let got = lin_to_db(peak);
                assert!(
                    (got - want).abs() < 0.1,
                    "{det:?} {p:?} level {level}: got {got:.3} want {want:.3}"
                );
                level += 1.0;
            }
        }
    }
}

/// Constant `|x| = c`: peak reads `c`, RMS reads `√2·c` (+3.01 dB), the
/// documented calibration for non-sine signals.
#[test]
fn dc_reading_per_detector() {
    let p = CompressorParams {
        threshold: 0.1,
        ratio: 4.0,
        knee: 1.0,
        attack_ms: 0.01,
        release_ms: 10.0,
        ..Default::default()
    };
    for (det, offset) in [
        (CompressorDetector::Peak, 0.0),
        (CompressorDetector::Rms, 10.0 * 2f64.log10()),
    ] {
        let mut c = Compressor::new();
        c.set_detector(det);
        c.set_params(&p);
        c.configure(FS, 2);
        c.process(&mut vec![0.5f32; 2 * ms(500.0)]);
        let x = lin_to_db(0.5) + offset;
        let want = x - curve_out_db(x, &p);
        assert!(
            (c.gain_reduction_db() - want).abs() < 1e-6,
            "{det:?}: {} vs {want}",
            c.gain_reduction_db()
        );
    }
}

/// RMS attack timing: a step from silence to `L` dB above the (hard-knee)
/// threshold starts reducing after `−τ·ln(1 − 10^(−L/10))`.
#[test]
fn rms_threshold_crossing_time() {
    for (tau, l_db) in [(10.0f32, 10.0f64), (50.0, 3.0), (1.0, 20.0)] {
        let p = CompressorParams {
            threshold: 0.1,
            ratio: 4.0,
            knee: 1.0,
            attack_ms: 0.01,
            release_ms: 50.0,
            ..Default::default()
        };
        let mut c = Compressor::new();
        c.set_detector(CompressorDetector::Rms);
        c.set_rms_window(tau);
        c.set_params(&p);
        c.configure(FS, 1);
        // DC reads √2·c: choose c so the reading is L dB above −20 dB.
        let level = 10f64.powf((-20.0 + l_db - 10.0 * 2f64.log10()) / 20.0) as f32;
        let n = ms(5.0 * tau as f64 + 5.0);
        let mut first = None;
        for i in 0..n {
            c.process(&mut [level]);
            if first.is_none() && c.gain_reduction_db() > 0.0 {
                first = Some(i);
            }
        }
        let want = -(tau as f64) * 1e-3 * FS as f64 * (1.0 - 10f64.powf(-l_db / 10.0)).ln();
        let got = first.expect("never crossed") as f64;
        assert!(
            (got - want).abs() <= 2.0,
            "τ {tau} L {l_db}: {got} vs {want}"
        );
    }
}

#[test]
fn rms_window_is_clamped() {
    let mut c = Compressor::new();
    assert_eq!(c.rms_window(), DEFAULT_RMS_WINDOW_MS);
    assert_eq!(c.detector(), CompressorDetector::Peak);
    c.set_rms_window(0.0);
    assert_eq!(c.rms_window(), RMS_WINDOW_RANGE.0);
    c.set_rms_window(1e9);
    assert_eq!(c.rms_window(), RMS_WINDOW_RANGE.1);
    c.set_rms_window(f32::NAN);
    assert_eq!(c.rms_window(), DEFAULT_RMS_WINDOW_MS);
}

/// Switching the detector on a steady sine while running: the detector is
/// seeded, so the gain does not jump (a sine reads the same in both modes).
#[test]
fn detector_switch_on_sine_is_continuous() {
    let p = CompressorParams {
        threshold: 0.05,
        ratio: 8.0,
        knee: 0.5,
        attack_ms: 0.01,
        release_ms: 50.0,
        ..Default::default()
    };
    let mut c = Compressor::new();
    c.set_params(&p);
    c.configure(FS, 1);
    let amp = 0.5;
    let input = sine_1k(amp, ms(2000.0));
    let mut prev_red = None;
    let mut max_step: f64 = 0.0;
    for (k, chunk) in input.chunks(ms(100.0)).enumerate() {
        if k > 2 {
            c.set_detector(if k % 2 == 0 {
                CompressorDetector::Peak
            } else {
                CompressorDetector::Rms
            });
        }
        for &s in chunk {
            c.process(&mut [s]);
            let r = c.gain_reduction_db();
            if let Some(pr) = prev_red {
                if k > 2 {
                    max_step = max_step.max((r - pr as f64).abs());
                }
            }
            prev_red = Some(r);
        }
    }
    // The RMS reading of a 1 kHz sine at τ = 10 ms is 0.035 dB above the
    // peak reading (see the module docs): steps stay well below 0.1 dB.
    assert!(max_step < 0.1, "max reduction step {max_step} dB");
}

#[test]
fn no_allocation_in_new_setters_and_process() {
    let mut c = Compressor::new();
    c.configure(48_000.0, 2);
    let mut buf: Vec<f32> = (0..2 * 1024)
        .map(|i| ((i as f32) * 0.37).sin() * 0.8)
        .collect();
    let before = alloc_count();
    for i in 0..60 {
        c.set_detector(if i % 3 == 0 {
            CompressorDetector::Rms
        } else {
            CompressorDetector::Peak
        });
        c.set_rms_window(1.0 + i as f32);
        c.set_params(&CompressorParams {
            threshold: 0.1,
            ratio: 4.0,
            release_ms: 1.0 + i as f32,
            ..Default::default()
        });
        c.process(&mut buf);
        if i % 17 == 16 {
            c.reset();
        }
    }
    assert_eq!(alloc_count() - before, 0);
}

#[test]
fn extremes_stay_finite_in_rms_mode() {
    for fs in [8_000.0f32, 192_000.0] {
        for (thr, ratio, att, rel, win) in [
            (0.0f32, 100.0f32, 0.01f32, 0.01f32, 1.0f32),
            (1.0, 1.0, 10_000.0, 10_000.0, 100.0),
            (0.0, 100.0, 0.01, 10_000.0, 100.0),
        ] {
            let mut c = Compressor::new();
            c.set_detector(CompressorDetector::Rms);
            c.set_rms_window(win);
            c.configure(fs, 3);
            c.set_params(&CompressorParams {
                threshold: thr,
                ratio,
                attack_ms: att,
                release_ms: rel,
                makeup_gain: 100.0,
                knee: 0.0,
            });
            let mut buf: Vec<f32> = (0..3 * 4000)
                .map(|i| match i / 3000 {
                    0 => 1e4 * ((i as f32) * 0.1).sin(),
                    1 => 0.0,
                    2 => f32::MAX,
                    _ => 1e-30,
                })
                .collect();
            c.process(&mut buf);
            assert!(buf.iter().all(|v| !v.is_nan()));
            assert!(c.gain_reduction_db().is_finite());
        }
    }
}
