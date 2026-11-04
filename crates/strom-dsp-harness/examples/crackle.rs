//! Crackle hunt: the broadcast chain on corpus clips, per configuration,
//! with NMR and a click count, plus a buffer-size check and WAV excerpts
//! of the worst regions.
//!
//! ```sh
//! cargo run --release --example crackle -- <corpus dir> <out dir> [jobs]
//! ```
//!
//! The click metric and the acceptance check are in
//! [`strom_dsp_harness::crackle`]. `accept` as the fourth argument prints
//! the acceptance table (worst-frame NMR per preset × latency mode over
//! the speech clips); `accept:no-spike` and `accept:spike-la-std` print it
//! with the spike guard off or at the standard lookahead in every mode;
//! `accept:no-adapt`, `accept:no-guard` and `accept:no-peak-control` with
//! the clipper's content-adaptive depth, its bandwidth guard or the
//! limiter's peak control off. `diag:clip,preset,mode[,switch…]` prints the
//! worst NMR frames of one run and the Bark bands that make them
//! (`DIAG_AT=<s>` adds the levels around a time); `predict` the
//! linear-prediction gain of every corpus clip.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use strom_dsp::broadcast::{BroadcastParams, BroadcastProcessor, LatencyMode};
use strom_dsp::clipper::ClipperLatency;
use strom_dsp::presets::BroadcastPreset;
use strom_dsp::spike_guard;
use strom_dsp::Processor;
use strom_dsp_harness::chain::FS;
use strom_dsp_harness::crackle::{
    self, acceptance_with, at_loudness, clicks, matched_error, verdict, MAX_SECONDS,
    SKIP_SECONDS as SKIP_S,
};
use strom_dsp_harness::listen::audio;
use strom_dsp_harness::signals::CH;

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

fn load(corpus: &Path, id: &str) -> Vec<f32> {
    let clip = crackle::manifest(corpus)
        .expect("manifest")
        .into_iter()
        .find(|c| c.id == id)
        .expect("clip in manifest");
    crackle::load(corpus, &clip, MAX_SECONDS).expect("decode")
}

fn at_target(x: &[f32], lufs: f64) -> Vec<f32> {
    at_loudness(x, lufs)
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
) -> (Vec<f32>, Vec<f32>, strom_dsp_harness::chain::Numbers) {
    let r = crackle::run(p, x);
    (r.linear, r.output, r.numbers)
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

/// One stage switch of the `accept:` and `diag:` modes.
fn apply_switch(p: &mut BroadcastParams, name: &str) {
    match name {
        "no-spike" => p.spike_guard_enabled = false,
        "spike-on" => p.spike_guard_enabled = true,
        "spike-la-std" => {
            p.spike_guard_enabled = true;
            p.spike_guard_lookahead_ms = spike_guard::STANDARD_LOOKAHEAD_MS
        }
        "no-clipper" => p.clipper_enabled = false,
        "clipper-on" => p.clipper_enabled = true,
        "no-expander" => p.band_expanders.iter_mut().for_each(|e| e.ratio = 1.0),
        "no-limiter" => p.limiter_enabled = false,
        "no-agc" => p.agc_enabled = false,
        "no-multiband" => p.multiband_enabled = false,
        "no-bassclip" => p.bass_clipper_enabled = false,
        "no-governor" => p.governor_enabled = false,
        "limiter-la-std" => p.limiter_lookahead_ms = LatencyMode::Standard.limiter_lookahead_ms(),
        "clipper-std" => p.clipper_latency = ClipperLatency::Standard,
        "no-adapt" | "no-guard" | "no-peak-control" => {}
        v => {
            if let Some(ms) = v.strip_prefix("limiter-la=") {
                p.limiter_lookahead_ms = ms.parse().expect("ms");
            } else if let Some(ms) = v.strip_prefix("limiter-rel=") {
                p.limiter_release_ms = ms.parse().expect("ms");
            } else {
                panic!("unknown switch {v}")
            }
        }
    }
}

/// The stage switches of the `accept:` and `diag:` modes that are not
/// parameters.
fn apply_processor_switch(bp: &mut BroadcastProcessor, name: &str) {
    match name {
        "no-adapt" => bp.set_clipper_adaptive_depth(false),
        "no-guard" => bp.set_clipper_bandwidth_guard(false),
        "no-peak-control" => bp.set_limiter_peak_control(false),
        _ => {}
    }
}

/// `diag:clip,preset,mode[,switch…]`: the worst NMR frames of one run, the
/// Bark bands that make them, and the limiter's reduction around them.
fn diag(corpus: &Path, spec: &str) {
    let f: Vec<&str> = spec.split(',').collect();
    let x = load(corpus, f[0]);
    let preset = BroadcastPreset::ALL
        .into_iter()
        .find(|p| p.nick() == f[1])
        .expect("preset");
    let mode = LatencyMode::ALL
        .into_iter()
        .find(|m| m.nick() == f[2])
        .expect("mode");
    let mut p = BroadcastParams::preset(preset, 4);
    mode.apply(&mut p);
    for s in &f[3..] {
        apply_switch(&mut p, s);
    }
    let x = at_target(&x, p.target_loudness_lufs as f64);
    let switches: Vec<String> = f[3..].iter().map(|v| v.to_string()).collect();
    let mut r = strom_dsp_harness::chain::run_chain_with(&p, x.clone(), Vec::new(), |bp| {
        for s in &switches {
            apply_processor_switch(bp, s);
        }
    });
    r.linear = strom_dsp_harness::chain::run_chain(
        &strom_dsp_harness::chain::linear_params(&p),
        x,
        Vec::new(),
    )
    .output;
    let mut frames = Vec::new();
    strom_dsp_harness::metrics::nmr_frames(
        &r.linear,
        &r.output,
        CH,
        FS,
        (SKIP_S * FS) as usize,
        |fr| frames.push(*fr),
    );
    if let Ok(t) = std::env::var("DIAG_AT") {
        let t: f64 = t.parse().expect("seconds");
        let s0 = ((t - 0.05) * FS) as usize;
        let blk = 96;
        for k in 0..30 {
            let a = s0 + k * blk;
            let pk = |x: &[f32]| {
                x[a * CH..(a + blk) * CH]
                    .iter()
                    .fold(0.0f32, |m, &v| m.max(v.abs())) as f64
            };
            let gr =
                r.limiter_gr[(a / strom_dsp_harness::chain::BLOCK).min(r.limiter_gr.len() - 1)];
            println!(
                "  {:.3} s  linear {:6.1}  out {:6.1}  limiter GR {:4.1}",
                a as f64 / FS,
                20.0 * pk(&r.linear).log10(),
                20.0 * pk(&r.output).log10(),
                gr
            );
        }
    }
    frames.sort_by(|a, b| b.ratio.total_cmp(&a.ratio));
    let db = |v: f64| 10.0 * v.max(1e-30).log10();
    for fr in frames.iter().take(4) {
        let t = fr.start as f64 / FS;
        println!(
            "frame at {t:.3} s ch {}: NMR {:+.1} dB",
            fr.channel,
            db(fr.ratio)
        );
        let mut bands: Vec<usize> = (0..strom_dsp_harness::metrics::BARK_BANDS).collect();
        bands.sort_by(|&a, &b| {
            (fr.error[b] / fr.threshold[b]).total_cmp(&(fr.error[a] / fr.threshold[a]))
        });
        for &b in bands.iter().take(6) {
            println!(
                "  bark {b:2}: ref {:6.1}  err {:6.1}  thr {:6.1}  nmr {:+5.1}",
                db(fr.reference[b]),
                db(fr.error[b]),
                db(fr.threshold[b]),
                db(fr.error[b] / fr.threshold[b])
            );
        }
        let b0 = fr.start / strom_dsp_harness::chain::BLOCK;
        let b1 = (fr.start + 2048) / strom_dsp_harness::chain::BLOCK;
        let gr = r.limiter_gr[b0.min(r.limiter_gr.len())..b1.min(r.limiter_gr.len())]
            .iter()
            .fold(0.0f64, |a, &v| a.max(v));
        let peak = r.output[fr.start * CH..((fr.start + 2048) * CH).min(r.output.len())]
            .iter()
            .fold(0.0f32, |a, &v| a.max(v.abs()));
        let lpeak = r.linear[fr.start * CH..((fr.start + 2048) * CH).min(r.linear.len())]
            .iter()
            .fold(0.0f32, |a, &v| a.max(v.abs()));
        println!(
            "  limiter GR max {gr:.2} dB; sample peak out {:.1} dBFS, linear {:.1} dBFS",
            20.0 * (peak as f64).log10(),
            20.0 * (lpeak as f64).log10()
        );
    }
}

/// `predict`: linear-prediction gain per 5 ms block over every corpus
/// clip (blocks over −50 dBFS), percentiles per clip.
fn predict(corpus: &Path) {
    use strom_dsp::predictability::Predictability;
    let clips = crackle::manifest(corpus).expect("manifest");
    for clip in &clips {
        let Ok(x) = crackle::load(corpus, clip, MAX_SECONDS) else {
            continue;
        };
        let x = at_target(&x, -23.0);
        let mut p = Predictability::new(FS);
        let mut guard = strom_dsp::bandwidth_guard::BandwidthGuard::new(FS);
        let mut edges = [0usize; 4];
        let mut ab: Vec<[f64; 3]> = Vec::new();
        let blk = 240;
        let mut g = Vec::new();
        for c in x.chunks(blk * CH) {
            p.push(c, CH);
            guard.push(c, CH);
            guard.end_block();
            edges[guard.edge().unwrap_or(3)] += 1;
            ab.push([guard.above_db(0), guard.above_db(1), guard.above_db(2)]);
            let ms = c.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / c.len() as f64;
            if ms > 1e-5 {
                g.push(p.gain_db());
            }
        }
        g.sort_by(f64::total_cmp);
        let q = |f: f64| {
            g.get(((g.len() as f64 - 1.0) * f) as usize)
                .copied()
                .unwrap_or(f64::NAN)
        };
        let total: usize = edges.iter().sum();
        let med = |k: usize| {
            let mut v: Vec<f64> = ab.iter().map(|a| a[k]).collect();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        print!(
            "above 4/8/12k {:6.1} {:6.1} {:6.1}  ",
            med(0),
            med(1),
            med(2)
        );
        println!(
            "{:<26} {:<42} p10 {:5.1}  p50 {:5.1}  p90 {:5.1}  edge 4k/8k/12k/full {:3.0} {:3.0} {:3.0} {:3.0} %",
            clip.category,
            clip.id,
            q(0.1),
            q(0.5),
            q(0.9),
            100.0 * edges[0] as f64 / total as f64,
            100.0 * edges[1] as f64 / total as f64,
            100.0 * edges[2] as f64 / total as f64,
            100.0 * edges[3] as f64 / total as f64
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus = PathBuf::from(args.get(1).expect("corpus dir"));
    let out = PathBuf::from(args.get(2).expect("out dir"));
    let jobs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    let only: Option<String> = args.get(4).cloned();
    std::fs::create_dir_all(&out).unwrap();
    if only.as_deref() == Some("predict") {
        predict(&corpus);
        return;
    }
    if let Some(spec) = only.as_deref().and_then(|o| o.strip_prefix("diag:")) {
        diag(&corpus, spec);
        return;
    }
    if let Some(variant) = only.as_deref().and_then(|o| o.strip_prefix("accept")) {
        // Comma-separated stage switches applied to every configuration.
        let names: Vec<String> = variant
            .trim_start_matches(':')
            .split(',')
            .filter(|v| !v.is_empty())
            .map(String::from)
            .collect();
        let tweak = |p: &mut BroadcastParams| {
            for n in &names {
                apply_switch(p, n);
            }
        };
        let setup = |bp: &mut BroadcastProcessor| {
            for n in &names {
                apply_processor_switch(bp, n);
            }
        };
        let runs = acceptance_with(&corpus, jobs, tweak, setup).expect("corpus");
        let (table, problems) = verdict(&runs);
        print!("{table}");
        for p in &problems {
            println!("problem: {p}");
        }
        if std::env::var_os("CRACKLE_PER_CLIP").is_some() {
            for r in &runs {
                println!(
                    "  {}\t{}\t{}\t{:+.1}\t{:+.1}",
                    r.preset.nick(),
                    r.mode.nick(),
                    r.clip,
                    r.worst_nmr_db,
                    r.mean_nmr_db
                );
            }
        }
        return;
    }

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
                let (lin, y, n) = run_one(&x, &p);
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
        let (lin, y, _) = run_one(&x, &p);
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
