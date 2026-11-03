//! Declipper quality on synthetic material: detector precision and recall,
//! SDR and SDRc improvement per clip depth (Záviška et al. 2021 protocol).
//!
//! Run with `--nocapture` to see the table.

mod declipper_support;

use declipper_support::*;
use strom_dsp::declipper::DeclipperParams;

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
}

fn evaluate(signal: &'static str, x: &[f64], depth: f64) -> Row {
    let (xc, c) = clip(x, depth);
    let xc32 = to_f32(&xc);
    let truth = truth_mask(x, c);
    let det = detect_mask(&xc32, default_detector());
    let (precision, recall) = precision_recall(&det, &truth);
    let (_, recall_ge2) = precision_recall(&det, &truth_runs_ge2(&truth));
    let (y, mut d) = declip(&xc32, &DeclipperParams::default());
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
        "{:<10} {:>5} {:>8} {:>6} {:>6} {:>8} {:>7} {:>7} {:>7} {:>7} {:>6}",
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
        "long"
    );
    for r in &rows {
        println!(
            "{:<10} {:>5.0} {:>8.2} {:>6.3} {:>6.3} {:>8.3} {:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>6}",
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
            r.too_long
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
///   10 dB on tones and speech (single-sample clips have no plateau);
/// - ΔSDRc on tones ≥ 10 dB up to 6 dB depth and ≥ 4 dB at 10 dB, on
///   speech ≥ 2 dB at 6 and 10 dB.
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
        if tones && r.clipped_pct > 0.05 {
            let min = if r.depth <= 6.0 { 10.0 } else { 4.0 };
            assert!(r.dsdrc >= min, "{what}: ΔSDRc {}", r.dsdrc);
        }
        if speech && r.depth >= 6.0 {
            assert!(r.dsdrc >= 2.0, "{what}: ΔSDRc {}", r.dsdrc);
        }
    }
}
