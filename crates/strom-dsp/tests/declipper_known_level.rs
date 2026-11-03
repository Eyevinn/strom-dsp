//! Known clip level: single samples and pairs at a level that clipping has
//! shown are repaired, and nothing else is.
//!
//! False positives are counted as detected samples that were not clipped,
//! in material that follows clipped material, so the clip level is known
//! when the unclipped part starts.

mod declipper_support;

use std::f64::consts::PI;

use declipper_support::*;
use strom_dsp::declipper::{ClipDetector, DetectorParams};

fn quantise(x: &[f64], bits: u32) -> Vec<f32> {
    let s = (1u64 << (bits - 1)) as f64 - 1.0;
    x.iter().map(|v| ((v * s).round() / s) as f32).collect()
}

/// Indices (into `x`) of samples in detected runs.
fn detected(x: &[f32], p: DetectorParams) -> Vec<usize> {
    let mask = detect_mask(x, p);
    (0..x.len()).filter(|&i| mask[i]).collect()
}

/// Detected samples in `x[from..]` (the unclipped part).
fn false_positives(x: &[f32], from: usize, p: DetectorParams) -> usize {
    detected(x, p).into_iter().filter(|&i| i >= from).count()
}

/// Clipped material (6 dB), then unclipped material whose peak is `g` times
/// the clip level, as float and as 16- and 24-bit. The unclipped part has
/// local maxima right at the old clip level (exactly at it when `g` is 1,
/// up to quantisation in the integer formats).
fn clipped_then_unclipped(g: f64, seed: u32) -> (Vec<f64>, usize) {
    let a = speech_like(2.0, seed, 140.0);
    let (ac, c) = clip(&a, 6.0);
    let mut b = speech_like(3.0, seed ^ 0x55, 190.0);
    normalize(&mut b, g * c);
    let mut x = ac;
    let from = x.len();
    x.extend_from_slice(&b);
    (x, from)
}

/// The detector finds singles at the known level once it is known, and
/// the known level makes the difference (singles off: not found).
#[test]
fn singles_found_at_known_level() {
    // A 1 kHz tone clipped 6 dB (long runs: the level becomes known), then
    // the same tone clipped only 0.05 dB below its peak: one sample per
    // peak reaches the clip level.
    let n = 48_000;
    let c = 0.5;
    let mut x: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * 1000.0 * i as f64 / FS).sin())
        .collect();
    let g = c * 10f64.powf(0.05 / 20.0);
    for (i, v) in x.iter_mut().enumerate().skip(n / 2) {
        *v = g * (2.0 * PI * 997.0 * i as f64 / FS + 0.3).sin();
    }
    let x: Vec<f32> = x.iter().map(|v| v.clamp(-c, c) as f32).collect();
    let mut p = default_detector();
    let mut d = ClipDetector::new(p);
    let mut singles = 0;
    for &v in x.iter().chain([0.0; 4].iter()) {
        if let Some(r) = d.push(v) {
            if r.len == 1 && r.start as usize >= n / 2 {
                singles += 1;
                assert_eq!(r.level.abs(), c as f32);
            }
        }
    }
    assert!(singles > 100, "singles found: {singles}");
    p.singles = false;
    let mut d = ClipDetector::new(p);
    for &v in &x {
        if let Some(r) = d.push(v) {
            assert!(r.len >= 2, "single without the known-level rule");
        }
    }
}

/// After clipped material, unclipped material whose peaks reach the old
/// clip level, at several gains and formats: nothing in it is detected.
#[test]
fn no_false_positives_after_gain_change() {
    let p = default_detector();
    let mut total = 0;
    for (k, g) in [0.9, 0.97, 0.995, 1.0, 1.003, 1.03, 1.1, 1.5]
        .into_iter()
        .enumerate()
    {
        let (x, from) = clipped_then_unclipped(g, 0x6a1e_0000 + k as u32);
        for (fmt, xs) in [
            ("f32", to_f32(&x)),
            ("int16", quantise(&x, 16)),
            ("int24", quantise(&x, 24)),
        ] {
            // The clipped part is found (the level is known).
            let found = detected(&xs, p).into_iter().filter(|&i| i < from).count();
            assert!(found > 20, "gain {g} {fmt}: clipped part found {found}");
            let fp = false_positives(&xs, from, p);
            total += fp;
            assert_eq!(fp, 0, "gain {g} {fmt}: {fp} false positives");
        }
    }
    println!("false positives after a gain change: {total}");
}

/// A gain change after the clip moves the plateaus: the new level is learnt
/// from the plateaus, and the old level finds nothing.
#[test]
fn gain_change_after_clip_moves_the_level() {
    let a = multitone(2.0, 0x6a1e_1000);
    let (ac, c) = clip(&a, 6.0);
    let truth = truth_mask(&a, c);
    let g = 10f64.powf(-1.5 / 20.0);
    let mut x: Vec<f64> = ac.clone();
    x.extend(ac.iter().map(|v| v * g));
    let mut t = truth.clone();
    t.extend_from_slice(&truth);
    let xs = to_f32(&x);
    let det = detect_mask(&xs, default_detector());
    let (prec, rec) = precision_recall(&det[a.len()..], &t[a.len()..]);
    println!("after -1.5 dB: precision {prec:.4} recall {rec:.3}");
    assert!(prec > 0.999, "precision {prec}");
    assert!(rec > 0.85, "recall {rec}");
}

/// Full-scale unclipped material, square-ish signals and soft saturation
/// right after clipped material at full scale: nothing detected.
#[test]
fn no_false_positives_on_full_scale_and_square_ish() {
    let p = default_detector();
    let n = (2.0 * FS) as usize;
    let lead = to_f32(&clip(&multitone(1.0, 0x6a1e_2000), 6.0).0);
    let lead: Vec<f32> = {
        // Clipped at exactly full scale.
        let m = lead.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        lead.iter().map(|v| v / m).collect()
    };
    let mut cases: Vec<(String, Vec<f64>)> = Vec::new();
    for f0 in [50.0, 100.0, 440.0, 1000.0] {
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
        cases.push((format!("band-limited square {f0} Hz"), sq));
        cases.push((
            format!("tanh-saturated sine {f0} Hz"),
            (0..n)
                .map(|i| (6.0 * (2.0 * PI * f0 * i as f64 / FS).sin()).tanh())
                .collect(),
        ));
        for ph in [0.0, 0.25, 0.5] {
            cases.push((
                format!("full-scale sine {f0} Hz phase {ph}"),
                (0..n)
                    .map(|i| (2.0 * PI * f0 * (i as f64 + ph) / FS).cos())
                    .collect(),
            ));
        }
    }
    let mut sp = speech_like(2.0, 0x6a1e_2001, 120.0);
    normalize(&mut sp, 1.0);
    cases.push(("full-scale speech".into(), sp));
    let mut mu = music_like(2.0, 0x6a1e_2002);
    normalize(&mut mu, 1.0);
    cases.push(("full-scale music".into(), mu));
    for (name, b) in &cases {
        for (fmt, bs) in [
            ("f32", to_f32(b)),
            ("int16", quantise(b, 16)),
            ("int24", quantise(b, 24)),
        ] {
            let mut x = lead.clone();
            x.extend_from_slice(&bs);
            let fp = false_positives(&x, lead.len(), p);
            assert_eq!(fp, 0, "{name} {fmt}: {fp} false positives");
        }
    }
}

/// Speech-like material clipped 3 and 6 dB: the known level finds more of
/// the clipped samples than plateaus alone.
#[test]
fn known_level_raises_speech_recall() {
    for (seed, f0) in [(0x6a1e_3000u32, 115.0), (0x6a1e_3001, 210.0)] {
        let x = speech_like(8.0, seed, f0);
        for depth in [3.0, 6.0] {
            let (xc, c) = clip(&x, depth);
            let xs = to_f32(&xc);
            let truth = truth_mask(&x, c);
            let mut p = default_detector();
            let (prec_on, rec_on) = precision_recall(&detect_mask(&xs, p), &truth);
            p.singles = false;
            let (_, rec_off) = precision_recall(&detect_mask(&xs, p), &truth);
            println!(
                "speech f0 {f0} depth {depth}: recall {rec_off:.3} -> {rec_on:.3} \
                 (precision {prec_on:.4})"
            );
            assert!(prec_on > 0.99);
            assert!(rec_on >= rec_off);
        }
    }
}
