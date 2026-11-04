//! Measure the declipper on recordings: clip each file at 3, 6 and 10 dB
//! below its peak, declip it, and report detection recall and precision,
//! SDR improvement on the clipped samples (ΔSDRc, Záviška et al. 2021) and
//! the time spent per 10 ms buffer.
//!
//! ```text
//! cargo run --release --example declip_eval -- [options] FILE.wav...
//! ```
//!
//! Each file is mixed to mono (the mean of its channels). Use 48 kHz files
//! (other rates are processed at their own rate). Options:
//!
//! ```text
//! --depths 3,6,10   clip depths below the file's peak (dB)
//! --latency MODE    low, normal (default) or high
//! --max-run MS      longest run that is rebuilt (default 2)
//! --quiet           only the summary
//! --show-fp         detections on the unclipped files, and missed runs
//! ```
//!
//! The summary pools the samples of all files for recall and precision, and
//! averages ΔSDRc over the files (files without clipped samples at a depth
//! are skipped for that depth).

#[path = "../tests/declipper_support/mod.rs"]
mod support;

use std::path::Path;
use std::time::Instant;

use strom_dsp::declipper::{Declipper, DeclipperLatency, DeclipperParams};
use strom_dsp::Processor;

fn usage() -> ! {
    eprintln!("usage: declip_eval [--depths 3,6,10] [--latency low|normal|high] [--max-run MS] [--quiet] [--show-fp] FILE.wav...");
    std::process::exit(2)
}

fn read_mono(path: &Path) -> (Vec<f64>, u32) {
    let mut r = hound::WavReader::open(path).expect("open input WAV");
    let spec = r.spec();
    let data: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(|s| s.expect("read")).collect(),
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.expect("read") as f32 * scale)
                .collect()
        }
    };
    let ch = usize::from(spec.channels).max(1);
    let mono = data
        .chunks_exact(ch)
        .map(|f| f.iter().map(|&v| f64::from(v)).sum::<f64>() / ch as f64)
        .collect();
    (mono, spec.sample_rate)
}

/// Declip mono `x` at `rate` in 10 ms buffers; output aligned with the
/// input, plus the time per buffer (s).
fn run(x: &[f32], rate: u32, params: &DeclipperParams) -> (Vec<f32>, Vec<f64>, Declipper) {
    let mut d = Declipper::new();
    d.set_params(params);
    d.configure(rate as f32, 1);
    let lat = d.latency_frames();
    let mut buf = x.to_vec();
    buf.extend(std::iter::repeat_n(0.0, lat));
    let block = (rate as usize / 100).max(1);
    let mut times = Vec::with_capacity(buf.len() / block + 1);
    for b in buf.chunks_mut(block) {
        let t = Instant::now();
        d.process(b);
        times.push(t.elapsed().as_secs_f64());
    }
    (buf[lat..].to_vec(), times, d)
}

#[derive(Default, Clone, Copy)]
struct Acc {
    files: usize,
    truth: usize,
    tp: usize,
    fp: usize,
    dsdrc: f64,
    buffers: usize,
    time: f64,
    worst: f64,
    runs: u64,
    fallbacks: u64,
    work: f64,
}

fn main() {
    let mut depths = vec![3.0, 6.0, 10.0];
    let mut quiet = false;
    let mut show_fp = false;
    let mut files = Vec::new();
    let mut params = DeclipperParams::default();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--depths" => {
                depths = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .split(',')
                    .map(|s| s.parse().unwrap_or_else(|_| usage()))
                    .collect();
            }
            "--quiet" => quiet = true,
            "--latency" => {
                params.latency = match args.next().as_deref() {
                    Some("low") => DeclipperLatency::Low,
                    Some("normal") => DeclipperLatency::Normal,
                    Some("high") => DeclipperLatency::High,
                    _ => usage(),
                }
            }
            "--max-run" => {
                params.max_run_ms = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage())
            }
            "--show-fp" => show_fp = true,
            s if s.starts_with("--") => usage(),
            _ => files.push(a),
        }
    }
    if files.is_empty() {
        usage();
    }
    println!("settings: {params:?}");
    let mut acc = vec![Acc::default(); depths.len()];
    for f in &files {
        let (x, rate) = read_mono(Path::new(f));
        let mut x = x;
        support::normalize(&mut x, 1.0);
        let name = Path::new(f)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        if show_fp {
            let x32 = support::to_f32(&x);
            let mut det = strom_dsp::declipper::ClipDetector::new(support::default_detector());
            for &v in x32.iter().chain(std::iter::once(&0.0)) {
                if let Some(r) = det.push(v) {
                    let s = r.start as usize;
                    let a = s.saturating_sub(3);
                    let b = (s + r.len + 3).min(x32.len());
                    println!(
                        "{name}: unclipped run at {s} len {} too_long {}: {:?}",
                        r.len,
                        r.too_long,
                        &x32[a..b]
                    );
                }
            }
        }
        for (k, &depth) in depths.iter().enumerate() {
            let (xc, c) = support::clip(&x, depth);
            let xc32 = support::to_f32(&xc);
            let truth = support::truth_mask(&x, c);
            let n_truth = truth.iter().filter(|&&t| t).count();
            let det = support::detect_mask(&xc32, support::default_detector());
            let tp = det.iter().zip(&truth).filter(|(d, t)| **d && **t).count();
            let fp = det.iter().zip(&truth).filter(|(d, t)| **d && !**t).count();
            if show_fp {
                // Missed clipped runs by length.
                let mut hist = [0usize; 8];
                let mut i = 0;
                while i < truth.len() {
                    if truth[i] {
                        let mut j = i;
                        while j < truth.len() && truth[j] {
                            j += 1;
                        }
                        if !det[i] {
                            hist[(j - i).min(7)] += 1;
                            if j - i >= 3 && std::env::var_os("SHOW_MISSED").is_some() {
                                println!(
                                    "  missed {i} len {}: {:?}",
                                    j - i,
                                    &xc32[i.saturating_sub(3)..(j + 3).min(xc32.len())]
                                );
                            }
                        }
                        i = j;
                    } else {
                        i += 1;
                    }
                }
                println!(
                    "{name} {depth} dB: missed runs by length (1..7+): {:?}",
                    &hist[1..]
                );
            }
            let (y, times, d) = run(&xc32, rate, &params);
            let a = &mut acc[k];
            a.truth += n_truth;
            a.tp += tp;
            a.fp += fp;
            let buf_s = (rate as f64 / 100.0).floor() / rate as f64;
            for &t in &times {
                a.time += t / buf_s;
                a.worst = a.worst.max(t / buf_s);
            }
            a.buffers += times.len();
            let st = d.stats();
            a.runs += st.runs_repaired;
            a.fallbacks += st.fallbacks;
            a.work += st.work;
            if n_truth == 0 {
                continue;
            }
            let dsdrc = support::sdr(&x, &y, Some(&truth)) - support::sdr(&x, &xc32, Some(&truth));
            a.files += 1;
            a.dsdrc += dsdrc;
            if !quiet {
                println!(
                    "{name:<28} {depth:>4.0} dB  clipped {n_truth:>6}  recall {:.3}  fp {fp:>3}  ΔSDRc {dsdrc:>6.2}",
                    tp as f64 / n_truth as f64
                );
            }
        }
    }
    println!(
        "\n{:>5} {:>6} {:>9} {:>7} {:>9} {:>7} {:>8} {:>8} {:>8} {:>9} {:>10}",
        "depth",
        "files",
        "clipped",
        "recall",
        "fp",
        "ΔSDRc",
        "cpu mean",
        "cpu max",
        "runs",
        "fallbacks",
        "work/run"
    );
    for (k, &depth) in depths.iter().enumerate() {
        let a = acc[k];
        println!(
            "{:>5.0} {:>6} {:>9} {:>7.3} {:>9} {:>7.2} {:>7.3}% {:>7.2}% {:>8} {:>9} {:>10.0}",
            depth,
            a.files,
            a.truth,
            a.tp as f64 / a.truth.max(1) as f64,
            a.fp,
            a.dsdrc / a.files.max(1) as f64,
            100.0 * a.time / a.buffers.max(1) as f64,
            100.0 * a.worst,
            a.runs,
            a.fallbacks,
            a.work / a.runs.max(1) as f64,
        );
    }
}
