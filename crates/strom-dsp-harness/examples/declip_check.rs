//! The declipper's corpus check ([`strom_dsp_harness::declip`]) with
//! options for measuring other settings, and a look at single runs.
//!
//! ```sh
//! cargo run --release -p strom-dsp-harness --example declip_check -- \
//!     <corpus dir> [--max-overshoot DB] [--latency low|normal|high] \
//!     [--jobs N] [--no-speech] [--runs CLIP:DEPTH]
//! ```
//!
//! Prints the table and the failures of the check. `--runs CLIP:DEPTH`
//! instead lists the rebuilt runs of one case (clip id, depth in dB) that
//! moved furthest from the original, and those with the highest peak
//! against the original's.

use std::path::PathBuf;

use strom_dsp::declipper::{DeclipperLatency, DeclipperParams};
use strom_dsp_harness::declip::{self, corpus_cases, failures, speech_dsdrc, table};
use strom_dsp_harness::listen::{clip_below_peak, read_manifest, run_declipper, set_original};
use strom_dsp_harness::signals::CH;

/// A stretch of rebuilt samples of one channel.
struct Run {
    ch: usize,
    start: usize,
    len: usize,
    level: f64,
    /// Peaks of the original and the output over the clip level.
    orig: f64,
    out: f64,
    /// Squared error against the original: output, clipped input.
    err_out: f64,
    err_clipped: f64,
}

fn print_runs(title: &str, runs: &[Run]) {
    println!("{title}");
    println!("ch  start (s)  len   level  orig/lvl  out/lvl   err out  err clip");
    for r in runs.iter().take(12) {
        println!(
            "{}  {:>9.4} {:>4} {:>7.3} {:>9.2} {:>8.2} {:>9.2e} {:>9.2e}",
            r.ch,
            r.start as f64 / 48_000.0,
            r.len,
            r.level,
            r.orig,
            r.out,
            r.err_out,
            r.err_clipped
        );
    }
}

fn show_runs(corpus: &std::path::Path, id: &str, depth: f64, params: &DeclipperParams) {
    let manifest = read_manifest(&corpus.join("manifest.tsv")).expect("manifest");
    let clip = manifest.iter().find(|c| c.id == id).expect("clip id");
    let (x, _) = set_original(corpus, clip, declip::MAX_SECONDS).expect("decode");
    let xc = clip_below_peak(&x, depth);
    let (y, st, _) = run_declipper(&xc, params);
    println!("{st:?}");
    let n = x.len() / CH;
    let mut runs = Vec::new();
    for c in 0..CH {
        let changed = |i: usize| y[i * CH + c].to_bits() != xc[i * CH + c].to_bits();
        let mut i = 0;
        while i < n {
            if !changed(i) {
                i += 1;
                continue;
            }
            let start = i;
            while i < n && changed(i) {
                i += 1;
            }
            let level = f64::from(xc[start * CH + c].abs());
            let mut r = Run {
                ch: c,
                start,
                len: i - start,
                level,
                orig: 0.0,
                out: 0.0,
                err_out: 0.0,
                err_clipped: 0.0,
            };
            for j in start..i {
                let k = j * CH + c;
                r.err_out += f64::from(y[k] - x[k]).powi(2);
                r.err_clipped += f64::from(xc[k] - x[k]).powi(2);
                r.orig = r.orig.max(f64::from(x[k].abs()) / level);
                r.out = r.out.max(f64::from(y[k].abs()) / level);
            }
            runs.push(r);
        }
    }
    let worse = runs.iter().filter(|r| r.err_out > r.err_clipped).count();
    println!("runs rebuilt: {}, worse than clipped: {worse}", runs.len());
    runs.sort_by(|a, b| (b.err_out - b.err_clipped).total_cmp(&(a.err_out - a.err_clipped)));
    print_runs(
        "furthest from the original against the clipped input:",
        &runs,
    );
    runs.sort_by(|a, b| (b.out / b.orig).total_cmp(&(a.out / a.orig)));
    print_runs("highest peak against the original's:", &runs);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let corpus = PathBuf::from(args.next().expect("corpus dir"));
    let mut params = DeclipperParams::default();
    let mut jobs = 8;
    let mut speech = true;
    let mut runs: Option<(String, f64)> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--max-overshoot" => {
                params.max_overshoot_db = args.next().and_then(|v| v.parse().ok()).expect("dB")
            }
            "--latency" => {
                params.latency = match args.next().as_deref() {
                    Some("low") => DeclipperLatency::Low,
                    Some("normal") => DeclipperLatency::Normal,
                    Some("high") => DeclipperLatency::High,
                    _ => panic!("--latency low|normal|high"),
                }
            }
            "--jobs" => jobs = args.next().and_then(|v| v.parse().ok()).expect("N"),
            "--no-speech" => speech = false,
            "--runs" => {
                let v = args.next().expect("CLIP:DEPTH");
                let (c, d) = v.rsplit_once(':').expect("CLIP:DEPTH");
                runs = Some((c.to_string(), d.parse().expect("depth")));
            }
            _ => panic!("unknown option {a}"),
        }
    }
    println!("settings: {params:?}");
    if let Some((id, depth)) = runs {
        show_runs(&corpus, &id, depth, &params);
        return;
    }
    let cases = corpus_cases(&corpus, &params, jobs).expect("corpus");
    let sp = if speech {
        speech_dsdrc(&corpus, &params, jobs).expect("speech")
    } else {
        [f64::NAN; 3]
    };
    print!("{}", table(&cases, &sp));
    for f in failures(&cases, &sp) {
        println!("FAIL {f}");
    }
}
