//! Ducker: timing of attack, hold and release against the documented
//! linear-in-dB fades, the proportional range curve, the key high-pass, a key
//! at another rate and channel count, the missing-key path, robustness at the
//! extremes, SIMD-level bit identity and zero allocations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use strom_dsp::biquad::uses_fma;
use strom_dsp::ducker::{Ducker, DuckerParams, RangeMode};
use strom_dsp::gate::DETECTOR_RMS_MS;
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

fn alloc_count() -> usize {
    ALLOCS.with(|c| c.get())
}

// ---------------------------------------------------------------------------
// Helpers.

const FS: f32 = 48_000.0;

fn frames(ms: f64, rate: f32) -> usize {
    (ms * 0.001 * rate as f64).round() as usize
}

fn sine(amp: f32, freq: f64, n: usize, ch: usize, rate: f32) -> Vec<f32> {
    (0..n)
        .flat_map(|i| {
            let v = amp * (2.0 * std::f64::consts::PI * freq * i as f64 / rate as f64).sin() as f32;
            std::iter::repeat_n(v, ch)
        })
        .collect()
}

/// Parameters with the key filter off (so a constant key reaches the
/// detector unchanged) and easy-to-check times.
fn params() -> DuckerParams {
    DuckerParams {
        threshold_db: -40.0,
        depth_db: -20.0,
        attack_ms: 20.0,
        hold_ms: 100.0,
        release_ms: 200.0,
        key_highpass_hz: 0.0,
        range_mode: RangeMode::Fixed,
    }
}

fn ducker(rate: f32, ch: usize, p: &DuckerParams) -> Ducker {
    let mut d = Ducker::new();
    d.set_params(p);
    d.configure(rate, ch);
    d
}

/// Run a mono programme of ones (so the output is the gain) against `key`
/// (mono, same rate) in buffers of `buf` frames; returns the gain in dB per frame.
fn gain_trace(d: &mut Ducker, key: &[f32], buf: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(key.len());
    for k in key.chunks(buf) {
        let mut x = vec![1.0f32; k.len()];
        d.process_with_key(&mut x, k, 1);
        out.extend(x.iter().map(|&g| 20.0 * (g as f64).log10()));
    }
    out
}

// ---------------------------------------------------------------------------

/// A constant key `c` (30 dB above the threshold) switched on at frame 0 and
/// off after 300 ms. Expected from the module docs:
/// - the detector crosses the threshold within 0.1 ms, then the reduction
///   ramps linearly to the depth and reaches it exactly at the attack time;
/// - after the key stops, the detector power `2c²·exp(−t/τ)` crosses the
///   threshold at `t_c = τ·ln(2c² / 10^(T/10))`; the reduction stays at the
///   depth for `hold` after that, then recovers linearly over `release`.
#[test]
fn attack_hold_release_timing() {
    let p = params();
    let c = 10f32.powf(-10.0 / 20.0) / 2f32.sqrt(); // reads as −10 dB
    let on = frames(300.0, FS);
    let total = frames(1000.0, FS);
    let key: Vec<f32> = (0..total).map(|i| if i < on { c } else { 0.0 }).collect();
    for buf in [480, 64, 1, 1000] {
        let mut d = ducker(FS, 1, &p);
        let g = gain_trace(&mut d, &key, buf);
        let block = 64usize; // envelope interpolation granularity
                             // Attack: full depth within attack time (+ detector delay + one block).
        let a = frames(p.attack_ms as f64, FS);
        assert!(
            (g[a + block] - p.depth_db as f64).abs() < 1e-4,
            "buf {buf}: {}",
            g[a + block]
        );
        // ... and not before: halfway there at half the attack time (± a block).
        let half = g[a / 2];
        let slope = p.depth_db as f64 / a as f64;
        assert!(
            (half - p.depth_db as f64 / 2.0).abs() <= slope.abs() * (block as f64 + 2.0),
            "buf {buf}: half-way {half}"
        );
        // Held at depth while the key is on.
        assert!(g[a + block..on]
            .iter()
            .all(|&v| (v - p.depth_db as f64).abs() < 1e-4));

        // Release timing.
        let tau = DETECTOR_RMS_MS * 1e-3;
        let p0 = 2.0 * (c as f64).powi(2);
        let t_c = tau * (p0 / 10f64.powf(p.threshold_db as f64 / 10.0)).ln();
        let hold_end = on + (t_c * FS as f64).round() as usize + frames(p.hold_ms as f64, FS);
        let rel = frames(p.release_ms as f64, FS);
        let tol = 2 * block;
        assert!(
            (g[hold_end - tol] - p.depth_db as f64).abs() < 1e-4,
            "buf {buf}: released before the hold ended"
        );
        let mid = g[hold_end + rel / 2];
        assert!(
            (mid - p.depth_db as f64 / 2.0).abs()
                <= slope.abs() * (rel as f64 / a as f64) * tol as f64 + 0.05,
            "buf {buf}: mid-release {mid}"
        );
        assert_eq!(g[hold_end + rel + tol], 0.0, "buf {buf}: not back at unity");
        assert!(!d.key_active());
        assert_eq!(d.reduction_db(), 0.0);
    }
}

#[test]
fn metering_follows_the_state() {
    let p = params();
    let mut d = ducker(FS, 2, &p);
    let mut x = vec![0.1f32; 2 * 4800];
    let key = vec![0.5f32; 2 * 4800];
    d.process_with_key(&mut x, &key, 2);
    assert!(d.key_active());
    assert_eq!(d.reduction_db(), -20.0);
    // Constant 0.5 reads as √2·0.5 → −3.01 dB.
    assert!(
        (d.key_level_db() + 3.0103).abs() < 0.01,
        "{}",
        d.key_level_db()
    );
    assert!((d.current_gain() - 0.1).abs() < 1e-12);
    // Linked: both channels get the same gain.
    assert!(x[x.len() - 2..].iter().all(|&v| (v - 0.01).abs() < 1e-7));
    d.reset();
    assert!(!d.key_active());
    assert_eq!(d.reduction_db(), 0.0);
    assert!(d.key_level_db() <= -199.0);
}

/// Proportional mode: steady keys at several levels settle at
/// `min(depth, L − T)`; fixed mode gives the full depth for all of them.
#[test]
fn proportional_range_curve() {
    let t = -40.0f64;
    let depth = -30.0f64;
    for level in [-45.0f64, -38.0, -30.0, -20.0, -12.0, -5.0] {
        let amp = 10f32.powf(level as f32 / 20.0);
        let key = sine(amp, 1000.0, frames(1000.0, FS), 1, FS);
        for mode in [RangeMode::Proportional, RangeMode::Fixed] {
            let p = DuckerParams {
                threshold_db: t as f32,
                depth_db: depth as f32,
                range_mode: mode,
                ..params()
            };
            let mut d = ducker(FS, 1, &p);
            let g = gain_trace(&mut d, &key, 480);
            let settled = &g[g.len() - frames(200.0, FS)..];
            let expected = if level <= t {
                0.0
            } else if mode == RangeMode::Fixed {
                depth
            } else {
                -(level - t).min(-depth)
            };
            for &v in settled {
                // Detector ripple on a 1 kHz sine: about ±0.07 dB.
                assert!(
                    (v - expected).abs() < 0.15,
                    "{mode:?} level {level}: {v} vs {expected}"
                );
            }
        }
    }
}

/// Key high-pass: a 50 Hz key 30 dB above the threshold ducks with the filter
/// off, but not through a 1 kHz high-pass (−52 dB at 50 Hz); a 5 kHz key
/// passes the filter.
#[test]
fn key_highpass() {
    let lo = sine(0.3, 50.0, frames(500.0, FS), 1, FS);
    let hi = sine(0.3, 5000.0, frames(500.0, FS), 1, FS);
    let run = |hp: f32, key: &[f32]| {
        let p = DuckerParams {
            threshold_db: -30.0,
            key_highpass_hz: hp,
            ..params()
        };
        let mut d = ducker(FS, 1, &p);
        let g = gain_trace(&mut d, key, 480);
        let m = g.iter().cloned().fold(0.0, f64::min);
        // The output is f32: compare within its precision.
        (m * 1e4).round() / 1e4
    };
    assert_eq!(run(0.0, &lo), -20.0);
    assert_eq!(run(1000.0, &lo), 0.0);
    assert_eq!(run(1000.0, &hi), -20.0);
    // The default 100 Hz high-pass keeps 20 Hz rumble (−10.5 dB, −28 dB
    // through the filter) below a −30 dB threshold.
    let rumble = sine(0.3, 20.0, frames(500.0, FS), 1, FS);
    assert_eq!(run(0.0, &rumble), -20.0);
    assert_eq!(run(DuckerParams::default().key_highpass_hz, &rumble), 0.0);
}

/// A key at another rate and channel count: a 16 kHz stereo key ducks a
/// 48 kHz programme with the same timing as a 48 kHz mono key of the same
/// level (the detector runs at the key's rate; only its envelope is
/// interpolated).
#[test]
fn key_at_other_rate_and_channels() {
    let p = params();
    let c = 0.2f32;
    let ms_on = 300.0;
    let ms_total = 1000.0;
    let reference = {
        let key: Vec<f32> = (0..frames(ms_total, FS))
            .map(|i| if i < frames(ms_on, FS) { c } else { 0.0 })
            .collect();
        gain_trace(&mut ducker(FS, 1, &p), &key, 480)
    };
    for key_rate in [16_000.0f32, 44_100.0, 192_000.0] {
        let mut d = ducker(FS, 1, &p);
        d.set_key_format(key_rate, 2);
        assert_eq!(d.key_channels(), 2);
        let kn = frames(ms_total, key_rate);
        let key: Vec<f32> = (0..kn)
            .flat_map(|i| {
                let v = if i < frames(ms_on, key_rate) { c } else { 0.0 };
                [v, -v * 0.5]
            })
            .collect();
        // 10 ms programme buffers with the matching key spans.
        let mut g = Vec::new();
        let bufs = (ms_total / 10.0) as usize;
        for b in 0..bufs {
            let k0 = frames(b as f64 * 10.0, key_rate);
            let k1 = frames((b + 1) as f64 * 10.0, key_rate);
            let mut x = vec![1.0f32; 480];
            d.process_with_key(&mut x, &key[k0 * 2..k1 * 2], 2);
            g.extend(x.iter().map(|&v| 20.0 * (v as f64).log10()));
        }
        // Same curve within a couple of blocks of timing.
        let slope = 20.0 / frames(p.attack_ms as f64, FS) as f64;
        let rel_slope = 20.0 / frames(p.release_ms as f64, FS) as f64;
        for (i, (a, b)) in g.iter().zip(&reference).enumerate() {
            let tol = 128.0 * slope.max(rel_slope) + 1e-3;
            assert!(
                (a - b).abs() <= tol,
                "rate {key_rate}, frame {i}: {a} vs {b}"
            );
        }
    }
}

/// No key (or a zero channel count): the key reads as silence for the
/// programme's duration, so an active duck holds, then releases to unity.
#[test]
fn missing_key_releases_to_unity() {
    let p = params();
    for key_rate in [48_000.0f32, 8_000.0] {
        let mut d = ducker(FS, 2, &p);
        d.set_key_format(key_rate, 1);
        let kn = frames(200.0, key_rate);
        let mut x = vec![1.0f32; 2 * frames(200.0, FS)];
        d.process_with_key(&mut x, &vec![0.5f32; kn], 1);
        assert_eq!(d.reduction_db(), -20.0);
        // Key gone: 10 ms buffers without a key. Gain never rises by more
        // than one release step per frame, and is unity after detector decay +
        // hold + release.
        let step = 20.0 / frames(p.release_ms as f64, FS) as f64;
        let mut last = -20.0f64;
        for i in 0..60 {
            let mut x = vec![1.0f32; 2 * 480];
            if i % 2 == 0 {
                d.process(&mut x);
            } else {
                d.process_with_key(&mut x, &[], 0);
            }
            for f in x.chunks(2) {
                let db = 20.0 * (f[0] as f64).log10();
                assert!(db - last <= step + 1e-5, "jump {last} -> {db}");
                last = db;
            }
        }
        assert_eq!(d.reduction_db(), 0.0, "key rate {key_rate}");
        assert!(!d.key_active());
    }
}

/// Depth changes while ducked apply at once (capped hold) but are faded.
#[test]
fn depth_change_while_ducked_is_faded() {
    let p = params();
    let mut d = ducker(FS, 1, &p);
    let key = vec![0.5f32; 4800];
    let mut x = vec![1.0f32; 4800];
    d.process_with_key(&mut x, &key, 1);
    assert_eq!(d.reduction_db(), -20.0);
    d.set_params(&DuckerParams {
        depth_db: -6.0,
        ..p
    });
    let mut x = vec![1.0f32; 4800];
    d.process_with_key(&mut x, &key, 1);
    let step = 14.0 / frames(p.release_ms as f64, FS) as f64;
    let mut last = -20.0;
    for &v in &x {
        let db = 20.0 * (v as f64).log10();
        assert!((db - last).abs() <= step + 1e-5);
        last = db;
    }
    assert!((d.reduction_db() + 6.0).abs() < 1e-9 || d.reduction_db() < -6.0);
}

#[test]
fn robust_at_extremes() {
    let extremes = [
        DuckerParams {
            threshold_db: -80.0,
            depth_db: -80.0,
            attack_ms: 0.1,
            hold_ms: 0.0,
            release_ms: 1.0,
            key_highpass_hz: 20_000.0,
            range_mode: RangeMode::Proportional,
        },
        DuckerParams {
            threshold_db: 0.0,
            depth_db: 0.0,
            attack_ms: 2000.0,
            hold_ms: 5000.0,
            release_ms: 10_000.0,
            key_highpass_hz: 0.0,
            range_mode: RangeMode::Fixed,
        },
        DuckerParams {
            threshold_db: f32::NAN,
            depth_db: f32::INFINITY,
            attack_ms: -1.0,
            hold_ms: f32::NAN,
            release_ms: f32::NEG_INFINITY,
            key_highpass_hz: 5.0,
            range_mode: RangeMode::Proportional,
        },
    ];
    for rate in [8_000.0f32, 44_100.0, 48_000.0, 96_000.0, 192_000.0] {
        for (ch, kc) in [(1usize, 1usize), (2, 1), (8, 2), (64, 64)] {
            for p in &extremes {
                let mut d = ducker(rate, ch, p);
                d.set_key_format(if kc == 64 { 8_000.0 } else { rate }, kc);
                let n = 3000;
                let key_amps = [0.0f32, 1e-30, 1.0, 1e30, f32::INFINITY, f32::NAN];
                for (b, &ka) in key_amps.iter().enumerate() {
                    let mut x: Vec<f32> =
                        (0..n * ch).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
                    let key: Vec<f32> = (0..n * kc)
                        .map(|i| if i % 3 == 0 { ka } else { -ka * 0.5 })
                        .collect();
                    d.process_with_key(&mut x, &key, kc);
                    assert!(
                        x.iter().all(|v| v.is_finite()),
                        "rate {rate} ch {ch} block {b}"
                    );
                    assert!(d.reduction_db().is_finite() && d.reduction_db() <= 0.0);
                    assert!(d.key_level_db().is_finite());
                }
                // Mismatched key channels read as silence: no panic.
                let mut x = vec![0.5f32; 7 * ch];
                d.process_with_key(&mut x, &[1.0; 3], kc + 1);
                d.process(&mut x[..ch]);
                assert!(x.iter().all(|v| v.is_finite()));
            }
        }
    }
    // Unconfigured: passthrough, no panic.
    let mut d = Ducker::new();
    let mut x = vec![0.25f32; 8];
    d.process_with_key(&mut x, &[1.0; 8], 1);
    assert!(x.iter().all(|&v| v == 0.25));
}

/// Every SIMD level gives bit-identical output (key filter off: always; on:
/// when the levels share the biquad's FMA policy).
#[test]
fn simd_fallback_is_bit_identical() {
    let (lo, hi) = (Level::fallback(), detect());
    for hp in [0.0f32, 150.0] {
        if hp > 0.0 && uses_fma(lo) != uses_fma(hi) {
            continue;
        }
        for mode in [RangeMode::Fixed, RangeMode::Proportional] {
            let p = DuckerParams {
                key_highpass_hz: hp,
                range_mode: mode,
                ..params()
            };
            let run = |level: Level| {
                let mut d = Ducker::new();
                d.set_simd_level(level);
                d.set_params(&p);
                d.configure(FS, 3);
                d.set_key_format(22_050.0, 2);
                let mut prog = sine(0.5, 330.0, 48_000, 3, FS);
                let key: Vec<f32> = sine(1.0, 700.0, 22_050, 2, 22_050.0)
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v * if (i / 4410) % 2 == 0 { 0.3 } else { 0.001 })
                    .collect();
                for (pc, kc) in prog.chunks_mut(480 * 3).zip(key.chunks(2 * 2205 / 10)) {
                    d.process_with_key(pc, kc, 2);
                }
                prog
            };
            let a = run(lo);
            let b = run(hi);
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "hp {hp} {mode:?}"
            );
        }
    }
}

#[test]
fn zero_allocations_after_setup() {
    let mut d = Ducker::new();
    d.configure(48_000.0, 2);
    d.set_key_format(16_000.0, 1);
    let prog = sine(0.5, 220.0, 48_000, 2, FS);
    let key = sine(0.5, 1000.0, 16_000, 1, 16_000.0);
    let mut buf = prog.clone();
    let before = alloc_count();
    for (i, (pc, kc)) in buf.chunks_mut(960).zip(key.chunks(160)).enumerate() {
        d.set_params(&DuckerParams {
            threshold_db: -20.0 - (i % 30) as f32,
            depth_db: -((i % 40) as f32),
            key_highpass_hz: (i % 3) as f32 * 200.0,
            range_mode: if i % 2 == 0 {
                RangeMode::Fixed
            } else {
                RangeMode::Proportional
            },
            ..DuckerParams::default()
        });
        match i % 4 {
            0 => d.process(pc),
            1 => d.process_with_key(pc, &[], 0),
            _ => d.process_with_key(pc, kc, 1),
        }
        if i % 17 == 16 {
            d.reset();
        }
        let _ = (d.reduction_db(), d.key_active(), d.key_level_db());
    }
    assert_eq!(alloc_count() - before, 0, "process path allocated");
}
