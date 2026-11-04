//! Declipper quality on synthetic material: detector precision and recall,
//! SDR and SDRc improvement per clip depth (Záviška et al. 2021 protocol).
//!
//! Run with `--nocapture` to see the table.

use crate::declipper_support::*;
use strom_dsp::declipper::{DeclipperLatency, DeclipperParams};

const SECS: f64 = 4.0;
const DEPTHS: [f64; 4] = [1.0, 3.0, 6.0, 10.0];

struct Row {
    signal: &'static str,
    depth: f64,
    clipped_pct: f64,
    precision: f64,
    recall: f64,
    recall_ge2: f64,
    sdr_in: f64,
    sdr_out: f64,
    dsdrc: f64,
    max_over_db: f32,
    too_long: u64,
    fallbacks: u64,
}

fn evaluate(signal: &'static str, x: &[f64], depth: f64) -> Row {
    evaluate_with(signal, x, depth, &DeclipperParams::default())
}

fn evaluate_with(signal: &'static str, x: &[f64], depth: f64, params: &DeclipperParams) -> Row {
    let (xc, c) = clip(x, depth);
    let xc32 = to_f32(&xc);
    let truth = truth_mask(x, c);
    let det = detect_mask(&xc32, default_detector());
    let (precision, recall) = precision_recall(&det, &truth);
    let (_, recall_ge2) = precision_recall(&det, &truth_runs_ge2(&truth));
    let (y, mut d) = declip(&xc32, params);
    let stats = d.take_stats();
    let dsdrc = sdr(x, &y, Some(&truth)) - sdr(x, &xc32, Some(&truth));
    assert!(y.iter().all(|v| v.is_finite()));
    Row {
        signal,
        depth,
        clipped_pct: 100.0 * truth.iter().filter(|&&t| t).count() as f64 / x.len() as f64,
        precision,
        recall,
        recall_ge2,
        sdr_in: sdr(x, &xc32, None),
        sdr_out: sdr(x, &y, None),
        dsdrc,
        max_over_db: stats.max_overshoot_db,
        too_long: stats.runs_too_long,
        fallbacks: stats.fallbacks,
    }
}

fn table() -> Vec<Row> {
    let signals: [(&'static str, Vec<f64>); 5] = [
        ("multitone", multitone(SECS, 0x5eed_0001)),
        ("noise-mod", noise_modulated_tones(SECS, 0x5eed_0002)),
        ("noise", noise_modulated(SECS, 0x5eed_0005)),
        ("speech-m", speech_like(SECS, 0x5eed_0003, 115.0)),
        ("speech-f", speech_like(SECS, 0x5eed_0004, 210.0)),
    ];
    let mut rows = Vec::new();
    for (name, x) in &signals {
        for &depth in &DEPTHS {
            rows.push(evaluate(name, x, depth));
        }
    }
    println!(
        "{:<10} {:>5} {:>8} {:>6} {:>6} {:>8} {:>7} {:>7} {:>7} {:>7} {:>6} {:>9}",
        "signal",
        "depth",
        "clipped%",
        "prec",
        "recall",
        "rec(≥2)",
        "SDR in",
        "SDR out",
        "ΔSDRc",
        "over dB",
        "long",
        "fallbacks"
    );
    for r in &rows {
        println!(
            "{:<10} {:>5.0} {:>8.2} {:>6.3} {:>6.3} {:>8.3} {:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>6} {:>9}",
            r.signal,
            r.depth,
            r.clipped_pct,
            r.precision,
            r.recall,
            r.recall_ge2,
            r.sdr_in,
            r.sdr_out,
            r.dsdrc,
            r.max_over_db,
            r.too_long,
            r.fallbacks
        );
    }
    rows
}

/// Bounds a little below the measured values (fixed seeds, so the numbers
/// are reproducible); they catch a regression in detection or repair.
///
/// - precision ≥ 0.995 everywhere (no repairs of unclipped samples);
/// - repair never makes the clipped samples worse (ΔSDRc ≥ 0);
/// - recall of the clipped samples in runs of ≥ 2 samples ≥ 0.9 at 6 and
///   10 dB on tones and speech, and of all clipped samples ≥ 0.8 on speech
///   at 6 dB and ≥ 0.4 at 3 dB (single samples count only at the known
///   clip level);
/// - ΔSDRc on tones ≥ 10 dB up to 6 dB depth and ≥ 3 dB at 10 dB, on
///   speech ≥ 2.5 dB at 3 and 6 dB and ≥ 1.5 dB at 10 dB. At 10 dB the
///   clipping is dense enough (a third of the samples on the multitone)
///   that the work budget forces fallbacks, which the table shows.
#[test]
fn quality_table() {
    let rows = table();
    for r in &rows {
        let what = format!("{} {} dB", r.signal, r.depth);
        assert!(r.precision >= 0.995, "{what}: precision {}", r.precision);
        assert!(r.dsdrc >= -0.01, "{what}: ΔSDRc {}", r.dsdrc);
        let tones = r.signal == "multitone" || r.signal == "noise-mod";
        let speech = r.signal.starts_with("speech");
        if (tones || speech) && r.depth >= 6.0 {
            assert!(r.recall_ge2 >= 0.9, "{what}: recall(≥2) {}", r.recall_ge2);
        }
        if speech && r.depth == 6.0 {
            assert!(r.recall >= 0.8, "{what}: recall {}", r.recall);
        }
        if speech && r.depth == 3.0 {
            assert!(r.recall >= 0.4, "{what}: recall {}", r.recall);
        }
        if tones && r.clipped_pct > 0.05 {
            let min = if r.depth <= 6.0 { 10.0 } else { 3.0 };
            assert!(r.dsdrc >= min, "{what}: ΔSDRc {}", r.dsdrc);
        }
        if speech && r.depth >= 3.0 {
            let min = if r.depth <= 6.0 { 2.5 } else { 1.5 };
            assert!(r.dsdrc >= min, "{what}: ΔSDRc {}", r.dsdrc);
        }
    }
}

/// What each latency setting costs: ΔSDRc and recall on speech-like
/// material and tones at 3, 6 and 10 dB for Low (3 ms), Normal (5 ms) and
/// High (10 ms). Detection does not depend on the setting; the rebuild
/// gets less future context in the shorter ones. Averaged over the speech
/// rows, the low setting stays within 1 dB of the high one (single rows
/// at 3 dB hang on a handful of runs and vary more).
#[test]
fn latency_settings() {
    let secs = 12.0;
    let signals: [(&'static str, Vec<f64>); 3] = [
        ("multitone", multitone(secs, 0x5eed_0001)),
        ("speech-m", speech_like(secs, 0x5eed_0003, 115.0)),
        ("speech-f", speech_like(secs, 0x5eed_0004, 210.0)),
    ];
    let mut speech_sum = [0.0f64; 3];
    let modes = [
        DeclipperLatency::Low,
        DeclipperLatency::Normal,
        DeclipperLatency::High,
    ];
    println!(
        "{:<10} {:>5} {:>18} {:>18} {:>18}",
        "signal", "depth", "Low ΔSDRc/recall", "Normal", "High"
    );
    for (name, x) in &signals {
        for depth in [3.0, 6.0, 10.0] {
            let rows: Vec<Row> = modes
                .iter()
                .map(|&latency| {
                    let p = DeclipperParams {
                        latency,
                        ..Default::default()
                    };
                    evaluate_with(name, x, depth, &p)
                })
                .collect();
            println!(
                "{:<10} {:>5.0} {:>11.2} / {:.3} {:>11.2} / {:.3} {:>11.2} / {:.3}",
                name,
                depth,
                rows[0].dsdrc,
                rows[0].recall,
                rows[1].dsdrc,
                rows[1].recall,
                rows[2].dsdrc,
                rows[2].recall
            );
            for r in &rows {
                assert!(r.dsdrc >= -0.01, "{name} {depth}: ΔSDRc {}", r.dsdrc);
            }
            if name.starts_with("speech") {
                for (k, r) in rows.iter().enumerate() {
                    speech_sum[k] += r.dsdrc / 6.0;
                }
            }
        }
    }
    println!(
        "speech mean ΔSDRc: low {:.2}, normal {:.2}, high {:.2}",
        speech_sum[0], speech_sum[1], speech_sum[2]
    );
    assert!(speech_sum[0] >= speech_sum[2] - 1.0);
    assert!(speech_sum[1] >= speech_sum[2] - 1.0);
}
