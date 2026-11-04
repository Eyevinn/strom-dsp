//! Crackle hunt: the broadcast chain on corpus clips, per configuration,
//! with NMR and a click count, plus a buffer-size check and WAV excerpts
//! of the worst regions.
//!
//! ```sh
//! cargo run --release --example crackle -- <corpus dir> <out dir> [jobs]
//! ```
//!
//! # Click metric
//!
//! The error is the processed output minus the linear reference (the
//! input through the chain with every dynamic stage neutral, as for NMR),
//! both high-passed at 4 kHz (4th-order Butterworth), with the reference
//! gain matched by least squares over 10 ms windows (hop 5 ms, the gain
//! interpolated linearly between window centres), so slow gain changes of
//! the AGC and the upper bands are not counted. Per 1 ms window `k`, `E_k`
//! is the mean square of that error. A window is a **click** when
//!
//! - `E_k` is more than [`CLICK_OVER_MEDIAN_DB`] over the median of `E`
//!   over ±50 ms (a burst, not the steady residue of a gain change), and
//! - `E_k` is no more than [`CLICK_UNDER_PROGRAMME_DB`] under the full-band
//!   reference mean square over ±10 ms (loud enough to matter against the
//!   programme around it), and above −100 dBFS.
//!
//! Neighbouring click windows count once (an event). Reported per minute
//! of measured audio, with the largest burst relative to the programme.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use strom_dsp::biquad::{Biquad, BiquadCoefficients};
use strom_dsp::broadcast::{BroadcastParams, BroadcastProcessor, LatencyMode};
use strom_dsp::clipper::ClipperLatency;
use strom_dsp::presets::BroadcastPreset;
use strom_dsp::spike_guard;
use strom_dsp::Processor;
use strom_dsp_harness::chain::{linear_params, measure_run, run_chain, FS};
use strom_dsp_harness::listen::audio;
use strom_dsp_harness::metrics;
use strom_dsp_harness::signals::CH;

const CLICK_OVER_MEDIAN_DB: f64 = 15.0;
const CLICK_UNDER_PROGRAMME_DB: f64 = -45.0;
const SKIP_S: f64 = 3.0;
const MAX_S: f64 = 30.0;

const CLIPS: [&str; 8] = [
    "librivox_ssc113_02_beeswaxcandle",
    "librispeech_2277-149874-0000",
    "vctk_p225_mic1_concat",
    "vctk_p226_g711u_narrowband",
    "fs189821_goal_red_bull_arena",
    "kmacleod_boogie_party",
    "kmacleod_magistar",
    "musan_us-gov-0001_hearing",
];

type Tweak = (&'static str, fn(&mut BroadcastParams));

fn tweaks_low() -> Vec<Tweak> {
    vec![
        ("all", |_| {}),
        ("spike-la-std", |p| {
            p.spike_guard_lookahead_ms = spike_guard::STANDARD_LOOKAHEAD_MS
        }),
        ("bass-la-std", |p| {
            p.bass_clipper_lookahead_ms = LatencyMode::Standard.bass_clipper_lookahead_ms()
        }),
        ("clipper-std", |p| {
            p.clipper_latency = ClipperLatency::Standard
        }),
        ("limiter-la-std", |p| {
            p.limiter_lookahead_ms = LatencyMode::Standard.limiter_lookahead_ms()
        }),
        ("no-spike", |p| p.spike_guard_enabled = false),
        ("no-agc", |p| p.agc_enabled = false),
        ("no-multiband", |p| p.multiband_enabled = false),
        ("no-expander", |p| {
            p.band_expanders.iter_mut().for_each(|e| e.ratio = 1.0)
        }),
        ("no-bassclip", |p| p.bass_clipper_enabled = false),
        ("no-clipper", |p| p.clipper_enabled = false),
        ("no-limiter", |p| p.limiter_enabled = false),
        ("no-governor", |p| p.governor_enabled = false),
    ]
}

fn sweep_tweaks() -> Vec<Tweak> {
    vec![
        ("la-0.00", |p| p.spike_guard_lookahead_ms = 0.0),
        ("la-0.05", |p| p.spike_guard_lookahead_ms = 0.05),
        ("la-0.10", |p| p.spike_guard_lookahead_ms = 0.10),
        ("la-0.15", |p| p.spike_guard_lookahead_ms = 0.15),
        ("la-0.20", |p| p.spike_guard_lookahead_ms = 0.20),
        ("la-0.25", |p| p.spike_guard_lookahead_ms = 0.25),
        ("la-0.30", |p| p.spike_guard_lookahead_ms = 0.30),
        ("la-0.35", |p| p.spike_guard_lookahead_ms = 0.35),
        ("la-0.42", |p| p.spike_guard_lookahead_ms = 0.42),
        ("la-0.50", |p| p.spike_guard_lookahead_ms = 0.50),
    ]
}

fn tweaks_std() -> Vec<Tweak> {
    vec![
        ("all", |_| {}),
        ("spike-la-0", |p| p.spike_guard_lookahead_ms = 0.0),
        ("no-spike", |p| p.spike_guard_enabled = false),
        ("no-expander", |p| {
            p.band_expanders.iter_mut().for_each(|e| e.ratio = 1.0)
        }),
        ("no-bassclip", |p| p.bass_clipper_enabled = false),
        ("no-clipper", |p| p.clipper_enabled = false),
        ("no-limiter", |p| p.limiter_enabled = false),
        ("no-multiband", |p| p.multiband_enabled = false),
    ]
}

fn hp4k(x: &[f32]) -> Vec<f32> {
    let mut y = x.to_vec();
    for q in [0.541_196, 1.306_563] {
        let mut b = Biquad::new(BiquadCoefficients::high_pass(4_000.0, q, FS), CH);
        b.process(&mut y);
    }
    y
}

/// Error of `y` against `r` (stereo), gain matched per 10 ms (see module).
fn matched_error(r: &[f32], y: &[f32]) -> Vec<f32> {
    let frames = r.len().min(y.len()) / CH;
    let win = 480;
    let hop = 240;
    let mut out = vec![0.0f32; frames * CH];
    for c in 0..CH {
        let mut centres = Vec::new();
        let mut gains = Vec::new();
        let mut s = 0;
        while s + win <= frames {
            let (mut ry, mut rr) = (0.0f64, 0.0f64);
            for i in s..s + win {
                let a = r[i * CH + c] as f64;
                ry += a * y[i * CH + c] as f64;
                rr += a * a;
            }
            centres.push(s + win / 2);
            gains.push(if rr > 1e-20 { ry / rr } else { 1.0 });
            s += hop;
        }
        let mut k = 0;
        for i in 0..frames {
            while k + 1 < centres.len() && centres[k + 1] <= i {
                k += 1;
            }
            let g = if centres.is_empty() {
                1.0
            } else if i <= centres[0] {
                gains[0]
            } else if k + 1 >= centres.len() {
                gains[centres.len() - 1]
            } else {
                let t = (i - centres[k]) as f64 / (centres[k + 1] - centres[k]) as f64;
                gains[k] * (1.0 - t) + gains[k + 1] * t
            };
            out[i * CH + c] = (y[i * CH + c] as f64 - g * r[i * CH + c] as f64) as f32;
        }
    }
    out
}

struct Clicks {
    per_min: f64,
    worst_db: f64,
    /// Frame of the largest burst.
    worst_at: usize,
    /// Click event start frames.
    at: Vec<usize>,
}

fn clicks(reference: &[f32], output: &[f32], skip: usize) -> Clicks {
    let e = matched_error(&hp4k(reference), &hp4k(output));
    let frames = e.len() / CH;
    let w = 48;
    let n = frames / w;
    let ms = |x: &[f32], k: usize| {
        let s = &x[k * w * CH..(k + 1) * w * CH];
        s.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / s.len() as f64
    };
    let en: Vec<f64> = (0..n).map(|k| ms(&e, k)).collect();
    let rn: Vec<f64> = (0..n).map(|k| ms(reference, k)).collect();
    let (mut count, mut worst_db, mut worst_at, mut prev) = (0usize, -200.0f64, 0usize, false);
    let mut at = Vec::new();
    let first = skip / w;
    for k in first..n {
        let lo = k.saturating_sub(50);
        let hi = (k + 51).min(n);
        let mut med: Vec<f64> = en[lo..hi].to_vec();
        med.sort_by(f64::total_cmp);
        let m = med[med.len() / 2].max(1e-30);
        let prog = rn[k.saturating_sub(10)..(k + 11).min(n)]
            .iter()
            .sum::<f64>()
            / (k + 11).min(n).saturating_sub(k.saturating_sub(10)) as f64;
        let rel = 10.0 * (en[k] / prog.max(1e-30)).log10();
        let hit = 10.0 * (en[k] / m).log10() > CLICK_OVER_MEDIAN_DB
            && rel > CLICK_UNDER_PROGRAMME_DB
            && en[k] > 1e-10;
        if hit {
            if !prev {
                count += 1;
                at.push(k * w);
            }
            if rel > worst_db {
                worst_db = rel;
                worst_at = k * w;
            }
        }
        prev = hit;
    }
    let minutes = (frames - skip.min(frames)) as f64 / FS / 60.0;
    Clicks {
        per_min: count as f64 / minutes.max(1e-9),
        worst_db,
        worst_at,
        at,
    }
}

fn load(corpus: &Path, id: &str) -> Vec<f32> {
    let manifest = std::fs::read_to_string(corpus.join("../manifest.tsv"))
        .or_else(|_| std::fs::read_to_string(corpus.join("manifest.tsv")))
        .expect("manifest");
    let line = manifest
        .lines()
        .find(|l| l.split('\t').next() == Some(id))
        .expect("clip in manifest");
    let f: Vec<&str> = line.split('\t').collect();
    let ch: usize = f[4].parse().unwrap();
    let x = audio::decode(&corpus.join(f[2]), ch).expect("decode");
    let frames = (x.len() / ch).min((MAX_S * FS) as usize);
    let mut y = Vec::with_capacity(frames * 2);
    for fr in x[..frames * ch].chunks_exact(ch) {
        let (l, r) = if ch == 1 {
            (fr[0], fr[0])
        } else {
            (fr[0], fr[1])
        };
        y.push(l);
        y.push(r);
    }
    y
}

fn at_target(x: &[f32], lufs: f64) -> Vec<f32> {
    let now = metrics::loudness(x, CH, FS).integrated_lufs;
    let g = 10f64.powf((lufs - now) / 20.0);
    x.iter().map(|&v| (v as f64 * g) as f32).collect()
}

struct Row {
    clip: String,
    preset: &'static str,
    mode: &'static str,
    tweak: &'static str,
    nmr_mean: f64,
    nmr_max: f64,
    clicks_min: f64,
    click_worst: f64,
    spike_events: f64,
    clip_depth: f64,
    lim_gr: f64,
}

fn run_one(
    x: &[f32],
    p: &BroadcastParams,
) -> (Vec<f32>, Vec<f32>, usize, strom_dsp_harness::chain::Numbers) {
    let mut run = run_chain(p, x.to_vec(), Vec::new());
    run.linear = run_chain(&linear_params(p), x.to_vec(), Vec::new()).output;
    let n = measure_run(None, true, p, &run, SKIP_S);
    (run.linear, run.output, run.latency, n)
}

fn process_blocks(p: &BroadcastParams, x: &[f32], block: usize) -> Vec<f32> {
    let mut bp = BroadcastProcessor::new();
    bp.set_params(p);
    bp.configure(FS as f32, CH);
    let mut out = x.to_vec();
    for c in out.chunks_mut(block * CH) {
        bp.process(c);
    }
    out
}

/// Worst 2 s region (start frame) around `at`, clamped.
fn region(at: usize, frames: usize) -> (usize, usize) {
    let len = (2.0 * FS) as usize;
    let s = at.saturating_sub(len / 2).min(frames.saturating_sub(len));
    (s, len.min(frames))
}

fn write_excerpts(dir: &Path, tag: &str, r: &[f32], y: &[f32], at: usize) {
    let frames = r.len() / CH;
    let (s, len) = region(at, frames);
    let sl = |x: &[f32]| x[s * CH..(s + len) * CH].to_vec();
    let (r2, y2) = (sl(r), sl(y));
    let e = sl(&matched_error(r, y));
    let peak = |x: &[f32]| x.iter().fold(0.0f32, |a, &v| a.max(v.abs())) as f64;
    let g = 10f64.powf(-1.0 / 20.0) / peak(&r2).max(peak(&y2)).max(1e-9);
    let ge = 10f64.powf(-6.0 / 20.0) / peak(&e).max(1e-12);
    let gain_db = 20.0 * (ge / g).log10();
    audio::write_wav24(&dir.join(format!("{tag}_reference.wav")), &r2, CH, g).unwrap();
    audio::write_wav24(&dir.join(format!("{tag}_processed.wav")), &y2, CH, g).unwrap();
    audio::write_wav24(
        &dir.join(format!("{tag}_error_plus{:.0}dB.wav", gain_db)),
        &e,
        CH,
        ge,
    )
    .unwrap();
    println!(
        "excerpt {tag}: {:.2}–{:.2} s (output time), error amplified {:+.0} dB re processed",
        s as f64 / FS,
        (s + len) as f64 / FS,
        gain_db
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus = PathBuf::from(args.get(1).expect("corpus dir"));
    let out = PathBuf::from(args.get(2).expect("out dir"));
    let jobs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    let only: Option<String> = args.get(4).cloned();
    std::fs::create_dir_all(&out).unwrap();

    let presets = [
        BroadcastPreset::EbuR128,
        BroadcastPreset::Streaming,
        BroadcastPreset::Speech,
    ];
    let clips: Vec<(String, Vec<f32>)> = CLIPS
        .iter()
        .map(|id| (id.to_string(), load(&corpus, id)))
        .collect();

    // Work list.
    let mut work = Vec::new();
    for (ci, _) in clips.iter().enumerate() {
        for preset in presets {
            for (mode, tw) in [
                (LatencyMode::Low, tweaks_low()),
                (LatencyMode::Standard, tweaks_std()),
            ] {
                for t in tw {
                    work.push((ci, preset, mode, t));
                }
            }
        }
    }
    if only.as_deref() == Some("buffers")
        || only.as_deref().is_some_and(|o| o.starts_with("excerpt:"))
    {
        work.clear();
    }
    if only.as_deref() == Some("sweep") {
        work.clear();
        for (ci, _) in clips.iter().enumerate() {
            for preset in [BroadcastPreset::EbuR128, BroadcastPreset::Streaming] {
                for t in sweep_tweaks() {
                    work.push((ci, preset, LatencyMode::Low, t));
                }
            }
        }
    }
    let rows = Mutex::new(Vec::new());
    let next = Mutex::new(0usize);
    std::thread::scope(|s| {
        for _ in 0..jobs {
            s.spawn(|| loop {
                let i = {
                    let mut n = next.lock().unwrap();
                    let i = *n;
                    *n += 1;
                    i
                };
                let Some(&(ci, preset, mode, (tname, tf))) = work.get(i) else {
                    break;
                };
                let mut p = BroadcastParams::preset(preset, 4);
                mode.apply(&mut p);
                tf(&mut p);
                let x = at_target(&clips[ci].1, p.target_loudness_lufs as f64);
                let (lin, y, _lat, n) = run_one(&x, &p);
                let c = clicks(&lin, &y, (SKIP_S * FS) as usize);
                let g = |k: &str| n.get(k).copied().flatten().unwrap_or(f64::NAN);
                rows.lock().unwrap().push(Row {
                    clip: clips[ci].0.clone(),
                    preset: preset.nick(),
                    mode: mode.nick(),
                    tweak: tname,
                    nmr_mean: g("nmr_mean_db"),
                    nmr_max: g("nmr_max_frame_db"),
                    clicks_min: c.per_min,
                    click_worst: c.worst_db,
                    spike_events: g("spike_events"),
                    clip_depth: g("clip_depth_db"),
                    lim_gr: g("limiter_gr_programme_mean_db"),
                });
            });
        }
    });
    let mut rows = rows.into_inner().unwrap();
    rows.sort_by(|a, b| {
        (a.clip.as_str(), a.preset, a.mode, a.tweak).cmp(&(
            b.clip.as_str(),
            b.preset,
            b.mode,
            b.tweak,
        ))
    });
    let mut tsv = String::from(
        "clip\tpreset\tmode\tconfig\tnmr_mean_db\tnmr_max_frame_db\tclicks_per_min\tclick_worst_db\tspike_events\tclip_depth_db\tlimiter_gr_db\n",
    );
    for r in &rows {
        tsv += &format!(
            "{}\t{}\t{}\t{}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.0}\t{:.1}\t{:.2}\n",
            r.clip,
            r.preset,
            r.mode,
            r.tweak,
            r.nmr_mean,
            r.nmr_max,
            r.clicks_min,
            r.click_worst,
            r.spike_events,
            r.clip_depth,
            r.lim_gr
        );
    }
    if !rows.is_empty() {
        let name = only.clone().unwrap_or_else(|| "ablation".into());
        std::fs::write(out.join(format!("{name}.tsv")), &tsv).unwrap();
        print!("{tsv}");
    }

    // Buffer-size check.
    if only.is_none() || only.as_deref() == Some("buffers") {
        println!("\nbuffer-size check (max |difference| against 1-frame buffers)");
        for (id, x) in clips.iter().take(3).chain(clips.iter().skip(5).take(1)) {
            let variants: [Tweak; 4] = [
                ("all", |_| {}),
                ("no-agc", |p| p.agc_enabled = false),
                ("no-governor", |p| p.governor_enabled = false),
                ("no-agc-governor", |p| {
                    p.agc_enabled = false;
                    p.governor_enabled = false
                }),
            ];
            for mode in LatencyMode::ALL {
                for (vname, vf) in variants {
                    let mut p = BroadcastParams::preset(BroadcastPreset::EbuR128, 4);
                    mode.apply(&mut p);
                    vf(&mut p);
                    let x = at_target(x, p.target_loudness_lufs as f64);
                    let base = process_blocks(&p, &x, 1);
                    let mut line = format!("{id}\t{}\t{vname}", mode.nick());
                    for b in [64, 480, 4096, 48, 511, 513] {
                        let y = process_blocks(&p, &x, b);
                        let d = base
                            .iter()
                            .zip(&y)
                            .fold(0.0f32, |a, (u, v)| a.max((u - v).abs()));
                        line += &format!("\t{b}: {d:.3e}");
                    }
                    println!("{line}");
                }
            }
        }
    }

    // Excerpts: worst burst for the given (clip, preset, mode, tweak).
    if let Some(spec) = only.as_deref().and_then(|o| o.strip_prefix("excerpt:")) {
        // spec: clip,preset,mode,tweak
        let f: Vec<&str> = spec.split(',').collect();
        let (ci, _) = clips.iter().enumerate().find(|(_, c)| c.0 == f[0]).unwrap();
        let preset = presets.into_iter().find(|p| p.nick() == f[1]).unwrap();
        let mode = LatencyMode::ALL
            .into_iter()
            .find(|m| m.nick() == f[2])
            .unwrap();
        let tw = if mode == LatencyMode::Low {
            tweaks_low()
        } else {
            tweaks_std()
        };
        let (_, tf) = tw.into_iter().find(|t| t.0 == f[3]).unwrap();
        let mut p = BroadcastParams::preset(preset, 4);
        mode.apply(&mut p);
        tf(&mut p);
        let x = at_target(&clips[ci].1, p.target_loudness_lufs as f64);
        let (lin, y, _, _) = run_one(&x, &p);
        let c = clicks(&lin, &y, (SKIP_S * FS) as usize);
        println!(
            "{spec}: {:.1} clicks/min, worst {:.1} dB at {:.3} s; first events at {:?} s",
            c.per_min,
            c.worst_db,
            c.worst_at as f64 / FS,
            c.at.iter()
                .take(12)
                .map(|&a| (a as f64 / FS * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>()
        );
        let tag = format!("{}_{}_{}_{}", f[0], f[1], f[2], f[3]);
        // Optional fifth field: centre of the region (s) instead of the worst burst.
        let at = f
            .get(4)
            .and_then(|v| v.parse::<f64>().ok())
            .map_or(c.worst_at, |s| (s * FS) as usize);
        write_excerpts(&out, &tag, &lin, &y, at);
    }
}
