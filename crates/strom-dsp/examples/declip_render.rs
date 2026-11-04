//! Declip a WAV file, or render a synthetic A/B demo set.
//!
//! ```text
//! cargo run --release --example declip_render -- IN.wav OUT.wav [options]
//! cargo run --release --example declip_render -- --demo OUT_DIR
//! ```
//!
//! Options (defaults as in `DeclipperParams::default()`):
//!
//! ```text
//! --max-overshoot-db X   how far beyond the clip level a rebuilt sample may go
//! --min-level-db X       lowest plateau level that counts as clipping (dBFS)
//! --max-run-ms X         longest run that is rebuilt
//! --tolerance X          plateau flatness relative to its level
//! --tilt-hz X            tolerate a high-pass droop up to this corner (0 = off)
//! ```
//!
//! The output is 32-bit float, so rebuilt peaks above 0 dBFS are kept; the
//! report says how much headroom they need. A text report is written next to
//! the output (`OUT.wav.txt`) and printed.
//!
//! `--demo` writes, for synthetic speech-like and music-like material clipped
//! 3, 6 and 10 dB below its peak, the original, the clipped and the declipped
//! signal (48 kHz, 32-bit float, the same gain for all three), plus
//! `README.txt` with the numbers.

#[path = "../tests/declipper_support/mod.rs"]
mod support;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use strom_dsp::declipper::{Declipper, DeclipperParams, DeclipperStats};
use strom_dsp::Processor;

fn usage() -> ! {
    eprintln!(
        "usage: declip_render IN.wav OUT.wav [--max-overshoot-db X] [--min-level-db X] \
         [--max-run-ms X] [--tolerance X] [--tilt-hz X]\n       declip_render --demo OUT_DIR"
    );
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--demo") {
        let dir = args.get(1).map(PathBuf::from).unwrap_or_else(|| usage());
        demo(&dir);
        return;
    }
    if args.len() < 2 {
        usage();
    }
    let mut params = DeclipperParams::default();
    let mut i = 2;
    while i < args.len() {
        let v: f32 = args
            .get(i + 1)
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| usage());
        match args[i].as_str() {
            "--max-overshoot-db" => params.max_overshoot_db = v,
            "--min-level-db" => params.min_level_db = v,
            "--max-run-ms" => params.max_run_ms = v,
            "--tolerance" => params.tolerance = v,
            "--tilt-hz" => params.tilt_hz = v,
            _ => usage(),
        }
        i += 2;
    }
    let (input, rate, channels) = read_wav(Path::new(&args[0]));
    let (output, stats, latency) = process(&input, rate, channels, &params);
    write_wav(Path::new(&args[1]), &output, rate, channels);
    let report = report(
        &args[0], &input, &output, rate, channels, latency, &params, &stats,
    );
    let report_path = format!("{}.txt", args[1]);
    std::fs::write(&report_path, &report).expect("write report");
    print!("{report}");
}

fn read_wav(path: &Path) -> (Vec<f32>, u32, usize) {
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
    (data, spec.sample_rate, usize::from(spec.channels))
}

fn write_wav(path: &Path, data: &[f32], rate: u32, channels: usize) {
    let spec = hound::WavSpec {
        channels: channels as u16,
        sample_rate: rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec).expect("create output WAV");
    for &s in data {
        w.write_sample(s).expect("write");
    }
    w.finalize().expect("finalize");
}

/// Declip interleaved `input`; the output is aligned with the input.
fn process(
    input: &[f32],
    rate: u32,
    channels: usize,
    params: &DeclipperParams,
) -> (Vec<f32>, DeclipperStats, usize) {
    let mut d = Declipper::new();
    d.set_params(params);
    d.configure(rate as f32, channels);
    let lat = d.latency_frames();
    let mut buf = input.to_vec();
    buf.extend(std::iter::repeat_n(0.0, lat * channels));
    let block = (rate as usize / 100).max(1) * channels;
    for b in buf.chunks_mut(block) {
        d.process(b);
    }
    (buf[lat * channels..].to_vec(), d.take_stats(), lat)
}

fn peak(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |a, v| a.max(v.abs()))
}

fn dbfs(x: f32) -> f64 {
    20.0 * f64::from(x).max(1e-10).log10()
}

#[allow(clippy::too_many_arguments)]
fn report(
    name: &str,
    input: &[f32],
    output: &[f32],
    rate: u32,
    channels: usize,
    latency: usize,
    params: &DeclipperParams,
    s: &DeclipperStats,
) -> String {
    let frames = input.len() / channels.max(1);
    let changed = input.iter().zip(output).filter(|(a, b)| a != b).count();
    let out_peak = peak(output);
    let mut r = String::new();
    let _ = writeln!(r, "input            {name}");
    let _ = writeln!(
        r,
        "format           {rate} Hz, {channels} ch, {:.2} s",
        frames as f64 / f64::from(rate)
    );
    let _ = writeln!(
        r,
        "latency          {latency} frames ({:.2} ms), compensated in the file",
        latency as f64 * 1000.0 / f64::from(rate)
    );
    let _ = writeln!(r, "settings         {params:?}");
    let _ = writeln!(r, "runs repaired    {}", s.runs_repaired);
    let _ = writeln!(
        r,
        "samples repaired {} ({:.3} % of all samples; {} samples differ from the input)",
        s.samples_repaired,
        100.0 * s.samples_repaired as f64 / input.len().max(1) as f64,
        changed
    );
    let _ = writeln!(r, "runs too long    {} (passed through)", s.runs_too_long);
    let _ = writeln!(r, "runs failed      {}", s.runs_failed);
    let _ = writeln!(
        r,
        "max overshoot    {:.2} dB above the clip level",
        s.max_overshoot_db
    );
    let _ = writeln!(
        r,
        "peaks            input {:.2} dBFS, output {:.2} dBFS",
        dbfs(peak(input)),
        dbfs(out_peak)
    );
    if out_peak > 1.0 {
        let _ = writeln!(
            r,
            "headroom         the output peaks {:.2} dB over full scale: lower the gain \
             by that much, or put a limiter after the declipper, before converting to integer",
            dbfs(out_peak)
        );
    }
    r
}

fn demo(dir: &Path) {
    use support::*;
    std::fs::create_dir_all(dir).expect("create demo dir");
    let secs = 8.0;
    let signals: [(&str, Vec<f64>); 4] = [
        ("speech-low", speech_like(secs, 0xdead_0001, 110.0)),
        ("speech-high", speech_like(secs, 0xdead_0002, 205.0)),
        ("music", music_like(secs, 0xdead_0003)),
        ("tones", noise_modulated_tones(secs, 0xdead_0004)),
    ];
    let params = DeclipperParams::default();
    let mut readme = String::new();
    let _ = writeln!(
        readme,
        "Declipper A/B demo (synthetic material, 48 kHz, 32-bit float WAV).\n\
         For each signal and clip depth (clip level that many dB below the peak):\n\
         *_1-original.wav, *_2-clipped.wav, *_3-declipped.wav, all with the same gain.\n\
         Default declipper settings: {params:?}\n\n\
         ΔSDRc: SDR improvement on the clipped samples (dB). Recall: share of the\n\
         clipped samples found. Overshoot: largest rebuilt peak over the clip level.\n"
    );
    let _ = writeln!(
        readme,
        "{:<12} {:>5} {:>9} {:>7} {:>7} {:>9} {:>10}",
        "signal", "depth", "clipped%", "recall", "ΔSDRc", "SDR out", "overshoot"
    );
    for (name, x) in &signals {
        for depth in [3.0, 6.0, 10.0] {
            let (xc, c) = clip(x, depth);
            let xc32 = to_f32(&xc);
            let truth = truth_mask(x, c);
            let (y, stats, _) = process(&xc32, 48_000, 1, &params);
            let det: Vec<bool> = xc32.iter().zip(&y).map(|(a, b)| a != b).collect();
            let (_, recall) = precision_recall(&det, &truth);
            let dsdrc = sdr(x, &y, Some(&truth)) - sdr(x, &xc32, Some(&truth));
            let x32 = to_f32(x);
            let g = 0.9 / peak(&x32).max(peak(&y)).max(1e-9);
            let scaled = |v: &[f32]| v.iter().map(|s| s * g).collect::<Vec<f32>>();
            let base = format!("{name}_{depth:02.0}dB");
            write_wav(
                &dir.join(format!("{base}_1-original.wav")),
                &scaled(&x32),
                48_000,
                1,
            );
            write_wav(
                &dir.join(format!("{base}_2-clipped.wav")),
                &scaled(&xc32),
                48_000,
                1,
            );
            write_wav(
                &dir.join(format!("{base}_3-declipped.wav")),
                &scaled(&y),
                48_000,
                1,
            );
            let line = format!(
                "{:<12} {:>5.0} {:>9.2} {:>7.3} {:>7.2} {:>9.2} {:>9.2}",
                name,
                depth,
                100.0 * truth.iter().filter(|&&t| t).count() as f64 / x.len() as f64,
                recall,
                dsdrc,
                sdr(x, &y, None),
                stats.max_overshoot_db
            );
            println!("{line}");
            let _ = writeln!(readme, "{line}");
        }
    }
    std::fs::write(dir.join("README.txt"), readme).expect("write README");
}
