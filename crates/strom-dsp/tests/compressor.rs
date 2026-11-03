//! Compressor tests: static curve (incl. soft knee) against the gain-computer
//! formula of spec 3.3, attack/release time constants, unity at ratio 1, linking,
//! robustness, parameter-change jump bound and real-time safety.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use proptest::prelude::*;
use strom_dsp::compressor::{DETECTOR_HOLD_MS, PARAM_SMOOTHING_MS};
use strom_dsp::util::{lin_to_db, one_pole_coef};
use strom_dsp::{Compressor, CompressorParams, Processor};

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

fn alloc_count() -> usize {
    ALLOCS.with(|c| c.get())
}

// ---------------------------------------------------------------------------
// Helpers.

const FS: f32 = 48_000.0;
const PROBE: f32 = 1e-5;

fn comp(fs: f32, ch: usize, p: CompressorParams) -> Compressor {
    let mut c = Compressor::new();
    c.configure(fs, ch);
    c.set_params(&p);
    c
}

fn ms(n_ms: f64) -> usize {
    (n_ms * 0.001 * FS as f64).round() as usize
}

/// Static output level (dB) per spec 3.3, written directly from the definition:
/// knee region `threshold·k … threshold/k`, `W = −40·log10(k)`, Giannoulis quadratic
/// interpolation inside the region, `T + (x − T)/R` above it.
fn expected_out_db(x: f64, p: &CompressorParams) -> f64 {
    let k = (p.knee as f64).max(0.001);
    let t = lin_to_db(p.threshold as f64);
    let w = -40.0 * k.log10();
    let r = p.ratio as f64;
    let y = if 2.0 * (x - t) < -w {
        x
    } else if 2.0 * (x - t).abs() <= w && w > 0.0 {
        x + (1.0 / r - 1.0) * (x - t + w / 2.0).powi(2) / (2.0 * w)
    } else {
        t + (x - t) / r
    };
    y + lin_to_db(p.makeup_gain as f64)
}

/// 1 kHz sine at 48 kHz whose samples hit the exact peak (n ≡ 12 mod 48).
fn sine_1k(amp: f64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (amp * (2.0 * std::f64::consts::PI * i as f64 / 48.0).sin()) as f32)
        .collect()
}

// ---------------------------------------------------------------------------

#[test]
fn static_curve_matches_gain_computer() {
    let settings = [
        // Default knee 0.0625 (W ≈ 48.2 dB): most of the range is in the knee.
        CompressorParams {
            threshold: 0.5,
            ratio: 4.0,
            ..Default::default()
        },
        CompressorParams {
            threshold: 0.1,
            ratio: 4.0,
            knee: 0.5,
            ..Default::default()
        },
        CompressorParams {
            threshold: 0.1,
            ratio: 100.0,
            knee: 1.0,
            ..Default::default()
        },
        CompressorParams {
            threshold: 0.05,
            ratio: 2.0,
            knee: 0.25,
            makeup_gain: 2.0,
            ..Default::default()
        },
        CompressorParams {
            threshold: 0.0,
            ratio: 10.0,
            knee: 0.0,
            makeup_gain: 100.0,
            ..Default::default()
        },
    ];
    for base in settings {
        // Instant attack, very slow release: the smoothed reduction sits at the
        // reduction of the sine's peak (the documented detector is a peak detector).
        let p = CompressorParams {
            attack_ms: 0.01,
            release_ms: 10_000.0,
            ..base
        };
        let mut level = -60.0;
        while level <= 0.0 + 1e-9 {
            let mut c = comp(FS, 1, p);
            let amp = 10f64.powf(level / 20.0);
            let mut buf = sine_1k(amp, ms(100.0));
            c.process(&mut buf);
            let peak = buf[buf.len() - 96..]
                .iter()
                .fold(0.0f64, |m, &s| m.max(s.abs() as f64));
            let got = lin_to_db(peak);
            let want = expected_out_db(level, &p);
            assert!(
                (got - want).abs() < 0.1,
                "{p:?} level {level}: got {got:.3} want {want:.3}"
            );
            level += 0.5;
        }
    }
}

/// Feed DC at `lo` then step to `hi`; return the smoothed reduction (dB) after
/// each frame, read from output/input.
fn step_reduction(c: &mut Compressor, level: f32, frames: usize) -> Vec<f64> {
    let mut buf = vec![level; frames];
    c.process(&mut buf);
    buf.iter()
        .map(|&o| -lin_to_db(o as f64 / level as f64))
        .collect()
}

/// Time constant in frames at FS (the sample grid may not hit τ exactly, so the
/// expectation uses the actual frame count `n`: covered fraction `1 − exp(−n/(τ·fs))`).
fn tau_frames(t_ms: f32) -> f64 {
    t_ms as f64 * 0.001 * FS as f64
}

#[test]
fn attack_time_constant() {
    for attack in [0.1f32, 1.0, 20.0, 1000.0] {
        let p = CompressorParams {
            threshold: 0.1,
            ratio: 4.0,
            knee: 1.0,
            attack_ms: attack,
            ..Default::default()
        };
        let mut c = comp(FS, 1, p);
        step_reduction(&mut c, 0.01, 1000); // below threshold: no reduction
        let r = step_reduction(&mut c, 1.0, ms(attack as f64) + 10);
        let final_r = 0.75 * 20.0; // S·(x − T) = 0.75·(0 − (−20))
        let n = ms(attack as f64);
        // After τ·fs frames the reduction covered 1 − 1/e of the step (in dB).
        let frac = r[n - 1] / final_r;
        assert!(
            (frac - (1.0 - (-(n as f64) / tau_frames(attack)).exp())).abs() < 1e-3,
            "attack {attack}: {frac}"
        );
    }
}

#[test]
fn release_time_constant() {
    for release in [0.1f32, 1.0, 100.0, 5000.0] {
        let p = CompressorParams {
            threshold: 0.1,
            ratio: 4.0,
            knee: 1.0,
            attack_ms: 0.01,
            release_ms: release,
            ..Default::default()
        };
        let mut c = comp(FS, 1, p);
        let r0 = step_reduction(&mut c, 1.0, 1000);
        assert!((r0.last().unwrap() - 15.0).abs() < 1e-6);
        // The detector holds the level for H = min(DETECTOR_HOLD_MS, release)
        // frames before the static reduction steps down (listening-test THD
        // fix); the documented time constant applies from there.
        // A window of H frames includes the current one, so the static
        // reduction steps down H − 1 frames after the input.
        let delay = ms((release as f64).min(DETECTOR_HOLD_MS)) - 1;
        let r = step_reduction(&mut c, 0.01, delay + ms(release as f64) + 10);
        assert!((r[delay - 1] - 15.0).abs() < 1e-6, "held through the hold");
        assert!(r[delay] < 15.0 - 1e-6, "released after the hold");
        let n = ms(release as f64);
        let frac = r[delay + n - 1] / 15.0;
        assert!(
            (frac - (-(n as f64) / tau_frames(release)).exp()).abs() < 1e-3,
            "release {release}: {frac}"
        );
    }
}

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

#[test]
fn ratio_one_makeup_one_is_unity() {
    let mut rng = Rng(7);
    for (threshold, knee) in [(0.5f32, 0.0625f32), (0.0, 0.0), (1.0, 1.0), (0.01, 0.5)] {
        let p = CompressorParams {
            threshold,
            knee,
            ratio: 1.0,
            makeup_gain: 1.0,
            attack_ms: 0.01,
            release_ms: 0.01,
        };
        let mut c = comp(FS, 2, p);
        let input: Vec<f32> = (0..20_000).map(|_| 4.0 * rng.next_f32()).collect();
        let mut buf = input.clone();
        c.process(&mut buf);
        assert_eq!(buf, input, "threshold {threshold} knee {knee}");
        // Makeup alone: exact multiplication.
        c.set_params(&CompressorParams {
            makeup_gain: 2.0,
            ..p
        });
        c.reset();
        let mut buf = input.clone();
        c.process(&mut buf);
        assert!(buf.iter().zip(&input).all(|(o, i)| *o == 2.0 * *i));
    }
}

#[test]
fn linked_channels() {
    for ch in [2usize, 8] {
        let p = CompressorParams {
            threshold: 0.1,
            ratio: 8.0,
            attack_ms: 1.0,
            release_ms: 10.0,
            ..Default::default()
        };
        let mut c = comp(FS, ch, p);
        let frames = ms(100.0);
        let mut buf = vec![PROBE; frames * ch];
        for f in 0..frames {
            buf[f * ch] = if f < frames / 2 { 0.9 } else { 0.001 };
        }
        let input = buf.clone();
        c.process(&mut buf);
        let mut max_red: f64 = 0.0;
        for f in 0..frames {
            let g0 = buf[f * ch] as f64 / input[f * ch] as f64;
            max_red = max_red.max(-lin_to_db(g0));
            for k in 1..ch {
                let gk = buf[f * ch + k] as f64 / input[f * ch + k] as f64;
                assert!((gk - g0).abs() <= 1e-6 * g0, "frame {f} ch {k}");
            }
        }
        assert!(max_red > 10.0, "compression happened: {max_red}");
    }
}

// ---------------------------------------------------------------------------
// Robustness.

fn corner_sets() -> Vec<CompressorParams> {
    (0u32..64)
        .map(|bits| {
            let pick = |i: u32, lo: f32, hi: f32| if bits & (1 << i) != 0 { hi } else { lo };
            CompressorParams {
                threshold: pick(0, 0.0, 1.0),
                ratio: pick(1, 1.0, 100.0),
                attack_ms: pick(2, 0.01, 10_000.0),
                release_ms: pick(3, 0.01, 10_000.0),
                makeup_gain: pick(4, 0.0, 100.0),
                knee: pick(5, 0.0, 1.0),
            }
        })
        .collect()
}

/// Loud noise (+20 dBFS), silence, an impulse, then quiet DC.
fn torture_signal(frames: usize, ch: usize, rng: &mut Rng) -> Vec<f32> {
    let mut v = vec![0.0f32; frames * ch];
    for (i, s) in v.iter_mut().enumerate() {
        let f = i / ch;
        *s = match (f * 4) / frames {
            0 => 10.0 * rng.next_f32(),
            1 => 0.0,
            2 => {
                if f == frames / 2 {
                    1.0
                } else {
                    0.0
                }
            }
            _ => 1e-3,
        };
    }
    v
}

#[test]
fn extremes_no_nan_inf_all_rates_and_channels() {
    let mut rng = Rng(0xdead_beef_1234_5678);
    let sets = corner_sets();
    for fs in [8_000.0f32, 44_100.0, 48_000.0, 96_000.0, 192_000.0] {
        for ch in [1usize, 2, 8, 64] {
            let chosen: Vec<usize> = if ch == 2 {
                (0..64).collect()
            } else {
                (0..64).step_by(9).collect()
            };
            for i in chosen {
                let p = sets[i];
                let mut c = comp(fs, ch, p);
                let frames = (fs as usize) / 50;
                let mut buf = torture_signal(frames, ch, &mut rng);
                c.process(&mut buf);
                assert!(buf.iter().all(|x| x.is_finite()), "fs {fs} ch {ch} {p:?}");
                let q = sets[63 - i];
                c.set_params(&q);
                let mut buf = torture_signal(frames, ch, &mut rng);
                c.process(&mut buf);
                assert!(buf.iter().all(|x| x.is_finite()), "fs {fs} ch {ch} {q:?}");
                assert!(c.gain_reduction_db().is_finite());
            }
        }
    }
}

#[test]
fn out_of_range_and_nan_params_are_clamped() {
    let mut c = comp(FS, 2, CompressorParams::default());
    c.set_params(&CompressorParams {
        threshold: -1.0,
        ratio: 0.5,
        attack_ms: 0.0,
        release_ms: 1e9,
        makeup_gain: f32::INFINITY,
        knee: f32::NAN,
    });
    let p = *c.params();
    assert_eq!(p.threshold, 0.0);
    assert_eq!(p.ratio, 1.0);
    assert_eq!(p.attack_ms, 0.01);
    assert_eq!(p.release_ms, 10_000.0);
    assert_eq!(p.makeup_gain, 100.0);
    assert_eq!(p.knee, CompressorParams::default().knee);
    let mut buf = vec![0.3f32; 2000];
    c.process(&mut buf);
    assert!(buf.iter().all(|x| x.is_finite()));
}

/// Jump bound for parameter changes. Threshold (dB), slope `1 − 1/ratio`, knee width
/// and makeup glide with a one-pole of τ_p = `PARAM_SMOOTHING_MS`, whose per-frame
/// step is the fraction `k = 1 − exp(−1/(τ_p·fs))` (≈ 1/480 at 48 kHz) of the remaining
/// distance. We require: per-frame change of the applied gain (dB) ≤ `3·k·ΔG`, where
/// ΔG is the largest change of the static gain (dB) between consecutive parameter
/// sets (factor 3 covers the product terms of slope × (level − threshold) and the
/// linear-domain makeup glide). The attack time is the minimum (0.01 ms), so the
/// attack smoother provides no protection: the glide alone has to do it.
#[test]
fn parameter_changes_do_not_jump() {
    let base = CompressorParams {
        attack_ms: 0.01,
        release_ms: 0.01,
        ..Default::default()
    };
    let sets = [
        CompressorParams { ..base },
        CompressorParams {
            threshold: 0.05,
            ratio: 20.0,
            knee: 1.0,
            makeup_gain: 4.0,
            ..base
        },
        CompressorParams {
            threshold: 0.2,
            ratio: 100.0,
            knee: 0.001,
            makeup_gain: 0.5,
            ..base
        },
        CompressorParams {
            threshold: 1.0,
            ratio: 2.0,
            knee: 0.5,
            makeup_gain: 1.0,
            ..base
        },
    ];
    let level = 0.5f32;
    let x_db = lin_to_db(level as f64);
    let mut dg_max: f64 = 0.0;
    for i in 0..sets.len() {
        let a = expected_out_db(x_db, &sets[i]);
        let b = expected_out_db(x_db, &sets[(i + 1) % sets.len()]);
        dg_max = dg_max.max((a - b).abs());
    }
    let k = 1.0 - one_pole_coef(PARAM_SMOOTHING_MS, FS as f64);
    let bound = 3.0 * k * dg_max;

    let mut c = comp(FS, 1, sets[0]);
    let mut prev = expected_out_db(x_db, &sets[0]) - x_db;
    let mut max_step: f64 = 0.0;
    for i in 0..40 {
        c.set_params(&sets[i % sets.len()]);
        let mut buf = vec![level; ms(53.0)];
        c.process(&mut buf);
        for o in buf {
            let g = lin_to_db(o as f64 / level as f64);
            max_step = max_step.max((g - prev).abs());
            prev = g;
        }
    }
    assert!(
        max_step <= bound,
        "max step {max_step} dB > bound {bound} dB (ΔG {dg_max})"
    );
    // Sanity: the bound is a meaningful restriction compared with an unsmoothed jump.
    assert!(bound < 0.2 && dg_max > 20.0, "bound {bound}, ΔG {dg_max}");
}

#[test]
fn reset_clears_state() {
    let p = CompressorParams {
        threshold: 0.1,
        ratio: 4.0,
        release_ms: 10_000.0,
        ..Default::default()
    };
    let mut c = comp(FS, 1, p);
    c.process(&mut vec![1.0f32; 10_000]);
    assert!(c.gain_reduction_db() > 1.0);
    c.reset();
    assert_eq!(c.gain_reduction_db(), 0.0);
    let mut buf = vec![0.001f32; 10];
    c.process(&mut buf);
    assert!(buf.iter().all(|&s| s == 0.001));
}

#[test]
fn unconfigured_is_passthrough() {
    let mut c = Compressor::new();
    c.set_params(&CompressorParams {
        ratio: 10.0,
        threshold: 0.01,
        makeup_gain: 3.0,
        ..Default::default()
    });
    let mut buf = vec![0.25f32, -0.5, 0.9];
    c.process(&mut buf);
    assert_eq!(buf, vec![0.25, -0.5, 0.9]);
}

#[test]
fn no_allocation_in_process_or_set_params() {
    let mut c = Compressor::new();
    c.configure(48_000.0, 8);
    let mut rng = Rng(99);
    let mut buf: Vec<f32> = (0..8 * 1024).map(|_| rng.next_f32()).collect();
    let a = CompressorParams {
        ratio: 8.0,
        threshold: 0.1,
        ..Default::default()
    };
    let b = CompressorParams::default();
    let before = alloc_count();
    for i in 0..50 {
        c.set_params(if i % 2 == 0 { &a } else { &b });
        c.process(&mut buf);
    }
    c.reset();
    c.process(&mut buf);
    assert_eq!(alloc_count() - before, 0);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// With fixed parameters a downward compressor never raises the level beyond
    /// the makeup gain, and never produces NaN/Inf.
    #[test]
    fn never_amplifies_beyond_makeup(
        threshold in 0.0f32..=1.0,
        ratio in 1.0f32..=100.0,
        attack in 0.01f32..=10_000.0,
        release in 0.01f32..=10_000.0,
        makeup in 0.0f32..=100.0,
        knee in 0.0f32..=1.0,
        seed in 1u64..u64::MAX,
        scale in 0.0f32..=10.0,
    ) {
        let p = CompressorParams { threshold, ratio, attack_ms: attack, release_ms: release, makeup_gain: makeup, knee };
        let mut c = comp(44_100.0, 2, p);
        let mut rng = Rng(seed);
        let input: Vec<f32> = (0..4096).map(|_| scale * rng.next_f32()).collect();
        let mut buf = input.clone();
        c.process(&mut buf);
        for (o, i) in buf.iter().zip(&input) {
            prop_assert!(o.is_finite());
            prop_assert!(o.abs() <= i.abs() * makeup * (1.0 + 1e-6) + 1e-30);
        }
    }
}

// ---------------------------------------------------------------------------
// Call order: set_params before or after configure/reset gives identical output.

#[test]
fn set_params_before_or_after_configure_is_identical() {
    let p = CompressorParams {
        threshold: 0.1,
        ratio: 8.0,
        attack_ms: 1.0,
        release_ms: 30.0,
        makeup_gain: 2.0,
        knee: 0.25,
    };
    let mut rng = Rng(11);
    let input: Vec<f32> = (0..2 * 9_600)
        .map(|i| rng.next_f32() * if (i / 2400) % 2 == 0 { 0.9 } else { 0.01 })
        .collect();

    let mut a = Compressor::new();
    a.set_params(&p);
    a.configure(48_000.0, 2);
    let mut b = Compressor::new();
    b.configure(48_000.0, 2);
    b.set_params(&p);
    let mut c = Compressor::new();
    c.configure(48_000.0, 2);
    c.process(&mut input.clone());
    c.reset();
    c.set_params(&p);

    let (mut ya, mut yb, mut yc) = (input.clone(), input.clone(), input.clone());
    a.process(&mut ya);
    b.process(&mut yb);
    c.process(&mut yc);
    assert!(ya.iter().zip(&yb).all(|(x, y)| x.to_bits() == y.to_bits()));
    assert!(ya.iter().zip(&yc).all(|(x, y)| x.to_bits() == y.to_bits()));
}
