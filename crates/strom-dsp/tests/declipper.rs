//! Declipper behaviour: latency, bit-exactness on unclipped material, the
//! repair constraints, the plausibility check, robustness to processing
//! after the clip, channel independence, long runs, bypass, extreme inputs
//! and real-time safety.
//!
//! Quality per clip depth is in `declipper_quality.rs`.

use std::f64::consts::PI;

use crate::declipper_support::*;
use strom_dsp::declipper::{
    Declipper, DeclipperLatency, DeclipperParams, WORK_BURST, WORK_RATE, XFADE_MS,
};
use strom_dsp::Processor;

use crate::counting_alloc::allocs;

// ---------------------------------------------------------------------------
// Helpers.

fn quantise(x: &[f64], bits: u32) -> Vec<f32> {
    let s = (1u64 << (bits - 1)) as f64 - 1.0;
    x.iter().map(|v| ((v * s).round() / s) as f32).collect()
}

fn interleave(chs: &[&[f32]]) -> Vec<f32> {
    let n = chs[0].len();
    let mut out = Vec::with_capacity(n * chs.len());
    for i in 0..n {
        for c in chs {
            out.push(c[i]);
        }
    }
    out
}

/// Run interleaved `x` through `d` in blocks of `block` frames; returns the
/// raw (delayed) output.
fn run(d: &mut Declipper, x: &[f32], channels: usize, block: usize) -> Vec<f32> {
    let mut buf = x.to_vec();
    for b in buf.chunks_mut(block * channels) {
        d.process(b);
    }
    buf
}

fn declipper(fs: f32, channels: usize, p: DeclipperParams) -> Declipper {
    let mut d = Declipper::new();
    d.set_params(&p);
    d.configure(fs, channels);
    d
}

/// Assert `y` is `x` delayed by `lat` frames, bit for bit (interleaved).
fn assert_delayed_exact(x: &[f32], y: &[f32], lat: usize, channels: usize, what: &str) {
    let l = lat * channels;
    assert!(
        y[..l].iter().all(|&v| v == 0.0),
        "{what}: pre-roll not silent"
    );
    for i in l..y.len() {
        assert!(
            y[i].to_bits() == x[i - l].to_bits(),
            "{what}: sample {i}: {} vs {}",
            y[i],
            x[i - l]
        );
    }
}

/// Unclipped material, including full-scale tones (at sample-aligned and
/// half-sample peak phases), band-limited squares and soft-saturated sines.
fn unclipped_set(secs: f64) -> Vec<(String, Vec<f64>)> {
    let n = (secs * FS) as usize;
    let mut v: Vec<(String, Vec<f64>)> = Vec::new();
    for f in [
        20.0, 50.0, 100.0, 440.0, 997.0, 1000.0, 3000.0, 6000.0, 12_000.0, 20_000.0,
    ] {
        for ph in [0.0, 0.5] {
            v.push((
                format!("sine {f} Hz phase {ph}"),
                (0..n)
                    .map(|i| (2.0 * PI * f * (i as f64 + ph) / FS).sin())
                    .collect(),
            ));
            v.push((
                format!("cosine {f} Hz phase {ph}"),
                (0..n)
                    .map(|i| (2.0 * PI * f * (i as f64 + ph) / FS).cos())
                    .collect(),
            ));
        }
    }
    for f0 in [50.0, 100.0, 1000.0] {
        let mut sq: Vec<f64> = (0..n)
            .map(|i| {
                let mut s = 0.0;
                let mut k = 1.0;
                while k * f0 < 20_000.0 {
                    s += (2.0 * PI * k * f0 * i as f64 / FS).sin() / k;
                    k += 2.0;
                }
                s
            })
            .collect();
        normalize(&mut sq, 1.0);
        v.push((format!("band-limited square {f0} Hz"), sq));
        v.push((
            format!("tanh-saturated sine {f0} Hz"),
            (0..n)
                .map(|i| (4.0 * (2.0 * PI * f0 * i as f64 / FS).sin()).tanh())
                .collect(),
        ));
    }
    v.push(("multitone".into(), multitone(secs, 7)));
    v.push((
        "noise-modulated tones".into(),
        noise_modulated_tones(secs, 8),
    ));
    v.push(("band-limited noise".into(), noise_modulated(secs, 9)));
    v.push(("speech-like".into(), speech_like(secs, 10, 120.0)));
    v.push(("music-like".into(), music_like(secs, 11)));
    v
}

// ---------------------------------------------------------------------------
// Tests.

const MODES: [DeclipperLatency; 3] = [
    DeclipperLatency::Low,
    DeclipperLatency::Normal,
    DeclipperLatency::High,
];

/// An impulse comes out exactly `latency_frames` later, enabled or not, in
/// every latency setting, and the latency is the documented function of the
/// rate and setting (144, 240 and 480 frames at 48 kHz): at most 3 ms for
/// `Low`, 5 ms for the default.
#[test]
fn latency_is_exact() {
    assert_eq!(DeclipperParams::default().latency, DeclipperLatency::Normal);
    assert_eq!(
        Declipper::latency_frames_for(48_000.0, DeclipperLatency::Low),
        144
    );
    assert_eq!(
        Declipper::latency_frames_for(48_000.0, DeclipperLatency::Normal),
        240
    );
    assert_eq!(
        Declipper::latency_frames_for(48_000.0, DeclipperLatency::High),
        480
    );
    for (fs, mode, enabled) in [8_000.0f32, 44_100.0, 48_000.0, 96_000.0, 192_000.0]
        .into_iter()
        .flat_map(|fs| MODES.into_iter().map(move |m| (fs, m)))
        .flat_map(|(fs, m)| [true, false].into_iter().map(move |e| (fs, m, e)))
    {
        {
            let p = DeclipperParams {
                enabled,
                latency: mode,
                ..Default::default()
            };
            let mut d = declipper(fs, 1, p);
            let lat = d.latency_frames();
            assert_eq!(lat, Declipper::latency_frames_for(fs, mode));
            assert!(
                lat as f32 <= (mode.ms() / 1000.0 * fs).round().max(16.0),
                "{fs}: latency {lat}"
            );
            assert!(d.max_run_frames() >= 1 && d.max_run_frames() < lat);
            let mut x = vec![0.0f32; lat + 1000];
            x[100] = 0.5;
            let y = run(&mut d, &x, 1, 64);
            for (i, &v) in y.iter().enumerate() {
                let want = if i == 100 + lat { 0.5 } else { 0.0 };
                assert_eq!(v, want, "fs {fs} {mode:?} enabled {enabled} frame {i}");
            }
            assert_eq!(d.latency_frames(), lat);
        }
    }
    // Unconfigured: passthrough, no latency.
    let mut d = Declipper::new();
    assert_eq!(d.latency_frames(), 0);
    let mut x = [0.5f32, -0.25];
    d.process(&mut x);
    assert_eq!(x, [0.5, -0.25]);
}

/// Material without clipping comes out bit-exact (delayed), as float and
/// quantised to 16 and 24 bits, with tilt detection on and off. The false
/// positive rate is printed (it is zero on this set).
#[test]
fn unclipped_material_is_bit_exact() {
    let set = unclipped_set(2.0);
    let mut total = 0usize;
    let mut flagged = 0usize;
    for (name, x) in &set {
        for (fmt, xs) in [
            ("f32", to_f32(x)),
            ("int16", quantise(x, 16)),
            ("int24", quantise(x, 24)),
        ] {
            for tilt_hz in [0.0, 50.0, 100.0] {
                let p = DeclipperParams {
                    tilt_hz,
                    ..Default::default()
                };
                let mut d = declipper(48_000.0, 1, p);
                let lat = d.latency_frames();
                let mut input = xs.clone();
                input.extend(std::iter::repeat_n(0.0, lat));
                let y = run(&mut d, &input, 1, 480);
                let s = d.take_stats();
                total += xs.len();
                flagged += s.samples_repaired as usize;
                assert_eq!(
                    s.runs_repaired + s.runs_too_long,
                    0,
                    "{name} {fmt} tilt {tilt_hz}: detected clipping"
                );
                assert_delayed_exact(&input, &y, lat, 1, name);
            }
        }
    }
    println!(
        "false positives: {flagged} of {total} samples ({} signals x 3 formats x 3 tilt settings)",
        set.len()
    );
}

/// Every repaired sample keeps its sign, lies beyond the observed clipped
/// value and no further than `max_overshoot_db` beyond it. With a cap of
/// 0 dB nothing can change.
#[test]
fn repairs_respect_sign_and_cap() {
    let x = multitone(2.0, 21);
    let (xc, _) = clip(&x, 8.0);
    let xin = to_f32(&xc);
    for cap in [0.0f32, 1.0, 3.0, 12.0, 24.0] {
        let p = DeclipperParams {
            max_overshoot_db: cap,
            ..Default::default()
        };
        let (y, mut d) = declip(&xin, &p);
        let lim = 10f32.powf(cap / 20.0) * (1.0 + 1e-6);
        let mut changed = 0;
        for (&o, &v) in xin.iter().zip(&y) {
            if o != v {
                changed += 1;
                assert!(o * v > 0.0, "sign flipped: {o} -> {v}");
                assert!(v.abs() >= o.abs(), "below clip level: {o} -> {v}");
                assert!(v.abs() <= o.abs() * lim, "cap {cap}: {o} -> {v}");
            }
        }
        let s = d.take_stats();
        assert!(s.max_overshoot_db <= cap + 1e-3);
        if cap == 0.0 {
            assert_eq!(changed, 0);
        } else {
            assert!(changed > 1000, "cap {cap}: only {changed} changed");
        }
    }
}

/// Gain, DC, and requantisation applied after the clip keep the plateau flat,
/// so detection and repair still work. The reference is the original with
/// the same processing.
/// A low tone under broadband noise (like music with a bass line and dense
/// high-frequency content), clipped 6 dB: long runs whose edges say little
/// about the hidden peak. Before the plausibility check and the 6 dB
/// default cap, the rebuild filled such runs with humps at the 12 dB cap:
/// on these seeds 1.1 dB further from the original on the clipped samples
/// than the clipped input, and output peaks 6 dB above the original's. Now
/// no seed comes out worse than its clipped input, no output peak lies more
/// than 1 dB above the original's, and runs are rejected.
#[test]
fn noisy_long_runs_are_never_worse_than_clipped() {
    for seed in [1u32, 2, 3] {
        let mut rng = Rng(0x5eed_0100 + seed);
        let mut x: Vec<f64> = (0..24_000)
            .map(|i| (2.0 * PI * 110.0 * i as f64 / FS).sin() + 0.1 * rng.gauss())
            .collect();
        normalize(&mut x, 1.0);
        let (xc, c) = clip(&x, 6.0);
        let xc32 = to_f32(&xc);
        let truth = truth_mask(&x, c);
        let (y, mut d) = declip(&xc32, &DeclipperParams::default());
        let st = d.take_stats();
        let gain = sdr(&x, &y, Some(&truth)) - sdr(&x, &xc32, Some(&truth));
        let peak = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            gain >= 0.0,
            "seed {seed}: SDR gain on the clipped samples {gain} dB"
        );
        assert!(
            20.0 * f64::from(peak).log10() <= 1.0,
            "seed {seed}: output peak {peak} over the original's 1.0"
        );
        assert!(
            st.runs_rejected > 0 && st.runs_repaired > 0,
            "seed {seed}: {st:?}"
        );
    }
}

#[test]
fn robust_to_gain_dc_and_requantisation_after_clip() {
    let x = multitone(3.0, 31);
    let (xc, c) = clip(&x, 6.0);
    let truth = truth_mask(&x, c);
    let cases: [(&str, f64, f64, Option<u32>); 5] = [
        ("plain", 1.0, 0.0, None),
        ("gain -6 dB", 0.5, 0.0, None),
        ("DC +0.05", 1.0, 0.05, None),
        ("gain -3 dB, 16-bit", 0.708, 0.0, Some(16)),
        ("gain -1 dB, DC -0.02, 24-bit", 0.891, -0.02, Some(24)),
    ];
    for (name, g, dc, bits) in cases {
        let fx = |v: &[f64]| -> Vec<f64> { v.iter().map(|s| s * g + dc).collect() };
        let r = fx(&x);
        let xin = match bits {
            Some(b) => quantise(&fx(&xc), b),
            None => to_f32(&fx(&xc)),
        };
        let det = detect_mask(&xin, default_detector());
        let (prec, rec) = precision_recall(&det, &truth);
        let (y, _) = declip(&xin, &DeclipperParams::default());
        let d = sdr(&r, &y, Some(&truth)) - sdr(&r, &xin, Some(&truth));
        println!("{name}: precision {prec:.3} recall {rec:.3} ΔSDRc {d:.2} dB");
        assert!(prec > 0.99 && rec > 0.9, "{name}: {prec} {rec}");
        assert!(d > 9.0, "{name}: ΔSDRc {d}");
    }
}

fn highpass(x: &[f64], fc: f64) -> Vec<f64> {
    // First-order high-pass y[n] = a·(y[n−1] + x[n] − x[n−1]).
    let a = (-2.0 * PI * fc / FS).exp();
    let (mut y1, mut x1) = (0.0, 0.0);
    x.iter()
        .map(|&v| {
            let y = a * (y1 + v - x1);
            y1 = y;
            x1 = v;
            y
        })
        .collect()
}

/// A first-order high-pass after the clip tilts the plateau. Flat detection
/// finds (almost) nothing; the default tilt tolerance (50 Hz) finds most of
/// it at low corners, less as the corner rises.
#[test]
fn tilted_plateau_after_highpass() {
    let x = multitone(3.0, 41);
    let (xc, c) = clip(&x, 6.0);
    let truth = truth_mask(&x, c);
    for fc in [10.0, 20.0, 40.0] {
        let r = highpass(&x, fc);
        let xin = to_f32(&highpass(&xc, fc));
        let flat = DeclipperParams {
            tilt_hz: 0.0,
            ..Default::default()
        };
        let (y0, mut d0) = declip(&xin, &flat);
        let (y1, _) = declip(&xin, &DeclipperParams::default());
        let d_flat = sdr(&r, &y0, Some(&truth)) - sdr(&r, &xin, Some(&truth));
        let d_tilt = sdr(&r, &y1, Some(&truth)) - sdr(&r, &xin, Some(&truth));
        let mut dp = default_detector();
        let (_, rec) = precision_recall(&detect_mask(&xin, dp), &truth);
        dp.max_tilt = 0.0;
        let (_, rec0) = precision_recall(&detect_mask(&xin, dp), &truth);
        println!(
            "high-pass {fc} Hz after clip: flat recall {rec0:.3} ΔSDRc {d_flat:.2}; \
             tilt recall {rec:.3} ΔSDRc {d_tilt:.2}"
        );
        let flat_runs = d0.take_stats().runs_repaired;
        assert!(
            rec0 < 0.01,
            "{fc} Hz: flat mode recall {rec0} ({flat_runs} runs)"
        );
        assert!(rec > 0.25, "{fc} Hz: recall {rec}");
        assert!(d_tilt > 1.0, "{fc} Hz: ΔSDRc {d_tilt}");
    }
}

/// Channels are processed independently: a clean channel next to a clipped
/// one stays bit-exact, and a clipped channel comes out as it would alone.
#[test]
fn channels_are_independent() {
    let a = to_f32(&clip(&multitone(1.5, 51), 6.0).0);
    let b = to_f32(&speech_like(1.5, 52, 150.0));
    let (alone, _) = declip(&a, &DeclipperParams::default());
    for channels in [2usize, 6] {
        let mut chs: Vec<&[f32]> = vec![&a, &b];
        while chs.len() < channels {
            chs.push(&b);
        }
        let mut d = declipper(48_000.0, channels, DeclipperParams::default());
        let lat = d.latency_frames();
        let mut x = interleave(&chs);
        x.extend(std::iter::repeat_n(0.0, lat * channels));
        let y = run(&mut d, &x, channels, 333);
        for i in 0..a.len() {
            assert_eq!(
                y[(i + lat) * channels],
                alone[i],
                "{channels} ch, frame {i}"
            );
            for c in 1..channels {
                assert_eq!(
                    y[(i + lat) * channels + c],
                    b[i],
                    "{channels} ch {c} frame {i}"
                );
            }
        }
    }
}

/// The block size does not change the output.
#[test]
fn block_size_does_not_matter() {
    let x = to_f32(&clip(&speech_like(1.0, 61, 180.0), 10.0).0);
    let mut d1 = declipper(48_000.0, 1, DeclipperParams::default());
    let mut d2 = declipper(48_000.0, 1, DeclipperParams::default());
    let y1 = run(&mut d1, &x, 1, 1);
    let y2 = run(&mut d2, &x, 1, 4096);
    assert_eq!(y1, y2);
    assert!(d1.take_stats().runs_repaired > 0);
}

/// Runs longer than `max_run_ms` pass through unchanged and are counted.
#[test]
fn long_runs_pass_through_and_are_counted() {
    // 50 Hz clipped 10 dB below its peak: plateaus of about 7.8 ms.
    let x: Vec<f64> = (0..48_000)
        .map(|i| (2.0 * PI * 50.0 * i as f64 / FS).sin())
        .collect();
    let xin = to_f32(&clip(&x, 10.0).0);
    let mut d = declipper(48_000.0, 1, DeclipperParams::default());
    let lat = d.latency_frames();
    let mut input = xin.clone();
    input.extend(std::iter::repeat_n(0.0, lat));
    let y = run(&mut d, &input, 1, 480);
    let s = d.take_stats();
    assert!(s.runs_too_long >= 90, "too long: {}", s.runs_too_long);
    assert_eq!(s.runs_repaired, 0);
    assert_delayed_exact(&input, &y, lat, 1, "long runs");
}

/// `enabled = false` fades to the dry signal; after the fade the output is
/// the input delayed by the (unchanged) latency, and re-enabling returns to
/// exactly what an always-enabled declipper produces.
#[test]
fn bypass_fades_to_exact_dry() {
    let x = to_f32(&clip(&multitone(2.0, 71), 8.0).0);
    let mut on = declipper(48_000.0, 1, DeclipperParams::default());
    let mut d = declipper(48_000.0, 1, DeclipperParams::default());
    let lat = d.latency_frames();
    let fade = (XFADE_MS * 48.0) as usize;
    let ref_out = run(&mut on, &x, 1, 480);
    let mut y = x.clone();
    let third = x.len() / 3;
    d.process(&mut y[..third]);
    d.set_params(&DeclipperParams {
        enabled: false,
        ..Default::default()
    });
    d.process(&mut y[third..2 * third]);
    d.set_params(&DeclipperParams::default());
    d.process(&mut y[2 * third..]);
    assert_eq!(d.latency_frames(), lat);
    assert_eq!(y[..third], ref_out[..third]);
    for i in third + fade..2 * third {
        assert_eq!(y[i], x[i - lat], "dry at {i}");
    }
    for i in 2 * third + fade..x.len() {
        assert_eq!(y[i], ref_out[i], "wet at {i}");
    }
    // The bypass really bypassed something.
    assert!((third + fade..2 * third).any(|i| ref_out[i] != x[i - lat]));
    // Starting disabled is bit-exact from the first frame.
    let mut off = declipper(
        48_000.0,
        1,
        DeclipperParams {
            enabled: false,
            ..Default::default()
        },
    );
    let mut input = x.clone();
    input.extend(std::iter::repeat_n(0.0, lat));
    let y = run(&mut off, &input, 1, 480);
    assert_delayed_exact(&input, &y, lat, 1, "disabled");
}

/// `reset` returns to the state after `configure`.
#[test]
fn reset_restarts() {
    let x = to_f32(&clip(&multitone(0.5, 81), 6.0).0);
    let mut d = declipper(48_000.0, 1, DeclipperParams::default());
    let a = run(&mut d, &x, 1, 480);
    d.reset();
    assert_eq!(d.stats(), Default::default());
    let b = run(&mut d, &x, 1, 480);
    assert_eq!(a, b);
}

/// Finite input gives finite output for any input, at any parameter
/// extreme; non-finite samples pass through delayed and do not poison the
/// state.
#[test]
fn extreme_inputs_stay_finite() {
    let n = 9600;
    let mut rng = Rng(0xfeed);
    let signals: Vec<(&str, Vec<f32>)> = vec![
        (
            "max square",
            (0..n)
                .map(|i| {
                    if (i / 7) % 2 == 0 {
                        f32::MAX
                    } else {
                        -f32::MAX
                    }
                })
                .collect(),
        ),
        (
            "alternating max",
            (0..n)
                .map(|i| if i % 2 == 0 { f32::MAX } else { -f32::MAX })
                .collect(),
        ),
        ("dc full scale", vec![1.0; n]),
        ("denormals", vec![1e-40; n]),
        (
            "naive square 1 kHz",
            (0..n)
                .map(|i| if (i / 24) % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
        ),
        (
            "noise clipped 20 dB",
            (0..n)
                .map(|_| (rng.gauss() as f32).clamp(-0.3, 0.3))
                .collect(),
        ),
        (
            "huge clipped tone",
            (0..n)
                .map(|i| (1e30 * (i as f32 * 0.05).sin()).clamp(-5e29, 5e29))
                .collect(),
        ),
        (
            "non-finite mix",
            (0..n)
                .map(|i| match i % 97 {
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    2 => f32::NEG_INFINITY,
                    _ => ((i as f32 * 0.03).sin() * 2.0).clamp(-1.0, 1.0),
                })
                .collect(),
        ),
    ];
    let params = [
        DeclipperParams::default(),
        DeclipperParams {
            min_level_db: -60.0,
            tolerance: 0.01,
            max_run_ms: 4.0,
            max_overshoot_db: 24.0,
            tilt_hz: 200.0,
            enabled: true,
            latency: DeclipperLatency::Low,
        },
        DeclipperParams {
            min_level_db: f32::NAN,
            tolerance: f32::INFINITY,
            max_run_ms: f32::NEG_INFINITY,
            max_overshoot_db: f32::NAN,
            tilt_hz: f32::NAN,
            enabled: true,
            latency: DeclipperLatency::High,
        },
    ];
    for p in &params {
        for fs in [8_000.0f32, 48_000.0, 192_000.0] {
            for (name, x) in &signals {
                let mut d = declipper(fs, 2, *p);
                let stereo = interleave(&[x, x]);
                let y = run(&mut d, &stereo, 2, 257);
                let lat = d.latency_frames();
                for (i, v) in y.iter().enumerate() {
                    let src = i.checked_sub(lat * 2).map(|j| stereo[j]);
                    if src.is_none_or(f32::is_finite) {
                        assert!(v.is_finite(), "{name} fs {fs} {p:?}: sample {i}");
                    }
                }
                // Afterwards a clean signal is bit-exact again.
                let clean: Vec<f32> = (0..4 * lat)
                    .map(|i| 0.3 * (i as f32 * 0.01).sin())
                    .collect();
                let clean = interleave(&[&clean, &clean]);
                let y = run(&mut d, &clean, 2, 100);
                for i in 2 * lat * 2..y.len() {
                    assert_eq!(y[i], clean[i - lat * 2], "{name} fs {fs}: recovery");
                }
            }
        }
    }
}

/// Square-ish wave whose plateaus are `run` frames long with one-sample
/// edges: a clipped run of that length every half period.
fn adversarial(n: usize, run: usize) -> Vec<f32> {
    let half = run + 2;
    (0..n)
        .map(|i| {
            let k = i % (2 * half);
            let (k, s) = if k < half { (k, 1.0) } else { (k - half, -1.0) };
            let v = if k == 0 || k == half - 1 { 0.5 } else { 0.9 };
            s * v
        })
        .collect()
}

/// The work budget holds on any input: on runs of the longest repairable
/// length back to back, on dense clipped tones and on clipped speech, no
/// buffer (of several sizes) spends more than
/// `channels · (WORK_BURST + WORK_RATE · d)` plus 12 units per sample for
/// the fallbacks. The fallback is used on the heavy inputs and keeps every
/// rebuilt sample on the clipped side, within the overshoot cap.
#[test]
fn work_budget_bounds_every_buffer() {
    for mode in MODES {
        let p = DeclipperParams {
            latency: mode,
            ..Default::default()
        };
        let probe = declipper(48_000.0, 1, p);
        let n = 96_000;
        let adv = adversarial(n, probe.max_run_frames());
        let tones = to_f32(&clip(&multitone(2.0, 0xb0d6), 10.0).0);
        let speech = to_f32(&clip(&speech_like(2.0, 0xb0d7, 150.0), 10.0).0);
        for (name, a, b) in [
            ("adversarial", &adv, &adv),
            ("tones + adversarial", &tones, &adv),
            ("speech + tones", &speech, &tones),
        ] {
            for block in [64usize, 480, 2048] {
                let mut d = declipper(48_000.0, 2, p);
                let x = interleave(&[a, b]);
                let mut y = x.clone();
                let mut fallbacks = 0;
                for chunk in y.chunks_mut(block * 2) {
                    let frames = chunk.len() / 2;
                    d.process(chunk);
                    let s = d.take_stats();
                    let bound = 2.0 * (WORK_BURST + WORK_RATE * frames as f64 / 48_000.0)
                        + 12.0 * s.samples_repaired as f64;
                    assert!(
                        s.work <= bound,
                        "{mode:?} {name} block {block}: work {} > {bound}",
                        s.work
                    );
                    fallbacks += s.fallbacks;
                }
                if name != "speech + tones" {
                    assert!(fallbacks > 0, "{mode:?} {name}: budget never ran out");
                }
                let lat = d.latency_frames() * 2;
                let lim = 10f32.powf(12.0 / 20.0) * (1.0 + 1e-6);
                for (i, &v) in y[lat..].iter().enumerate() {
                    let o = x[i];
                    if o != v {
                        assert!(o * v > 0.0 && v.abs() >= o.abs() && v.abs() <= o.abs() * lim);
                    }
                }
            }
        }
    }
}

/// No allocation after `configure`, while heavily clipped stereo material
/// runs and parameters change between buffers.
#[test]
fn no_allocation_after_configure() {
    let a = to_f32(&clip(&multitone(2.0, 91), 10.0).0);
    let b = to_f32(&clip(&speech_like(2.0, 92, 130.0), 10.0).0);
    let x = interleave(&[&a, &b]);
    let mut d = Declipper::new();
    d.configure(48_000.0, 2);
    let mut buf = x.clone();
    let sizes = [480usize, 1, 64, 1023, 7, 2048];
    let before = allocs();
    let mut pos = 0;
    let mut k = 0;
    while pos < buf.len() / 2 {
        let frames = sizes[k % sizes.len()].min(buf.len() / 2 - pos);
        d.set_params(&DeclipperParams {
            enabled: k % 5 != 3,
            max_overshoot_db: [12.0, 3.0, 24.0][k % 3],
            max_run_ms: [2.0, 0.5, 4.0][k % 3],
            tilt_hz: [50.0, 0.0][k % 2],
            ..Default::default()
        });
        d.process(&mut buf[pos * 2..(pos + frames) * 2]);
        let _ = d.take_stats();
        if k == 40 {
            d.reset();
        }
        pos += frames;
        k += 1;
    }
    assert_eq!(allocs(), before, "allocated in the processing path");
    assert!(buf.iter().all(|v| v.is_finite()));
}
