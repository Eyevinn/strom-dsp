//! Crackle on speech: the broadcast chain on corpus clips, measured with
//! the worst-frame NMR and a click count, and the acceptance check "no
//! measurable crackle on speech" over every preset and latency mode.
//!
//! # Acceptance check
//!
//! Over the corpus clips of [`SPEECH_CATEGORIES`] (clean speech, phone and
//! codec speech), each scaled to the preset's target loudness, the first
//! [`MAX_SECONDS`] through every preset × latency mode at 4 bands, measured
//! from [`SKIP_SECONDS`]: the largest per-frame NMR
//! ([`metrics::nmr`] against the linear reference,
//! [`crate::chain::linear_params`]) over every clip stays under
//! [`NMR_MAX_FRAME_BOUND_DB`].
//!
//! **Why 0 dB on the worst frame.** NMR 0 dB is where the error added in a
//! frame reaches the masked threshold of the programme in it: below it the
//! error is masked (inaudible by the model), above it a listener can hear
//! it. A slow gain change (AGC, band compressors, expander) is gain
//! matched per Bark band and per frame, so it costs nothing; what remains
//! is distortion inside a frame: a gain dip at a word onset, a clipped
//! peak. Those are short, so the mean over a clip hides them, and the
//! worst frame is the number that shows one. Measured with the spike
//! guard and the final clipper off at `ebu-r128` and `atsc-a85` (AGC,
//! multiband, expander, bass clipper and limiter at work), the worst frame
//! of every speech clip sits between −18.5 and −8.3 dB, so 0 dB leaves
//! 8 dB of margin for the dynamics doing their work, while the defects
//! found (an onset dipped by the spike guard without lookahead, speech
//! peaks in the final clipper, plosives taken for clicks, speech peaks in
//! the limiter at −16 and −18 LUFS) reach +1 … +37 dB.
//!
//! **Known failures** ([`KNOWN_FAILURES`]) are listed with their cause,
//! for a whole configuration or for named clips: they are reported, not
//! failed. The check fails when any run that no entry covers is over the
//! bound, and when an entry covers no failing run any more (it passes now:
//! take it off the list), so the list cannot go stale.

//! # Click metric
//!
//! The error is the processed output minus the linear reference, both
//! high-passed at 4 kHz (4th-order Butterworth), with the reference gain
//! matched by least squares over 10 ms windows (hop 5 ms, the gain
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

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use strom_dsp::biquad::{Biquad, BiquadCoefficients};
use strom_dsp::broadcast::{BroadcastParams, LatencyMode};
use strom_dsp::presets::BroadcastPreset;

use crate::chain::{linear_params, measure_run, run_chain, Numbers, FS};
use crate::listen::{audio, read_manifest, to_stereo, Clip};
use crate::metrics;
use crate::signals::CH;

/// A window is a click when its error is this far over the local median (dB).
pub const CLICK_OVER_MEDIAN_DB: f64 = 15.0;
/// … and no further than this under the programme around it (dB).
pub const CLICK_UNDER_PROGRAMME_DB: f64 = -45.0;
/// Measured from here (s): the AGC, the trim and the band detectors settle.
pub const SKIP_SECONDS: f64 = 3.0;
/// Longest part of a clip that is run (s).
pub const MAX_SECONDS: f64 = 30.0;
/// Corpus categories of the acceptance check.
pub const SPEECH_CATEGORIES: [&str; 2] = ["1_speech_clean", "2_phone_narrowband"];
/// Bound on the worst-frame NMR of every speech clip (dB): the error
/// stays under the masked threshold (see the module docs).
pub const NMR_MAX_FRAME_BOUND_DB: f64 = 0.0;

/// A configuration (or some of its clips) known to fail the acceptance
/// check, and why.
#[derive(Debug, Clone, Copy)]
pub struct KnownFailure {
    pub preset: BroadcastPreset,
    pub mode: LatencyMode,
    /// The clips it covers; `None`: every clip.
    pub clips: Option<&'static [&'static str]>,
    /// The cause, as measured.
    pub reason: &'static str,
}

impl KnownFailure {
    /// Whether this entry covers `clip` in `preset` × `mode`.
    pub fn covers(&self, preset: BroadcastPreset, mode: LatencyMode, clip: &str) -> bool {
        self.preset == preset && self.mode == mode && self.clips.is_none_or(|c| c.contains(&clip))
    }
}

const SHORTWAVE: &[&str] = &["fs259175_shortwave_websdr"];
const CLIPPER_ON_STATIC: &str = "final clipper on the static crashes of the shortwave clip";

/// The known failures (see the module docs). Measured with
/// `cargo run --release --example crackle -- corpus/data <out> 8 accept`
/// and its stage switches (`accept:no-clipper`, `accept:no-limiter`, …).
/// The spike guard is not among the causes: with it off every number is
/// the same.
pub const KNOWN_FAILURES: &[KnownFailure] = &[
    KnownFailure {
        preset: BroadcastPreset::EbuR128,
        mode: LatencyMode::Standard,
        clips: Some(SHORTWAVE),
        reason: CLIPPER_ON_STATIC,
    },
    KnownFailure {
        preset: BroadcastPreset::EbuR128,
        mode: LatencyMode::Low,
        clips: Some(SHORTWAVE),
        reason: CLIPPER_ON_STATIC,
    },
    KnownFailure {
        preset: BroadcastPreset::AtscA85,
        mode: LatencyMode::Standard,
        clips: Some(SHORTWAVE),
        reason: CLIPPER_ON_STATIC,
    },
    KnownFailure {
        preset: BroadcastPreset::AtscA85,
        mode: LatencyMode::Low,
        clips: Some(SHORTWAVE),
        reason: CLIPPER_ON_STATIC,
    },
    KnownFailure {
        preset: BroadcastPreset::Streaming,
        mode: LatencyMode::Standard,
        clips: None,
        reason: "final clipper clips speech peaks at −16 LUFS (up to +35.7 dB; pending clipper \
                 work); without it the limiter (up to +6.3 dB)",
    },
    KnownFailure {
        preset: BroadcastPreset::Streaming,
        mode: LatencyMode::Low,
        clips: None,
        reason: "final clipper clips speech peaks at −16 LUFS (up to +35.7 dB); without it the \
                 limiter at 0.6 ms lookahead (up to +13.9 dB)",
    },
    KnownFailure {
        preset: BroadcastPreset::Speech,
        mode: LatencyMode::Standard,
        clips: Some(&["librispeech_2035-152373-0005"]),
        reason: "limiter on loud speech peaks at −18 LUFS (+2.7 dB)",
    },
    KnownFailure {
        preset: BroadcastPreset::Speech,
        mode: LatencyMode::Low,
        clips: Some(&[
            "librispeech_2035-152373-0005",
            "fs259175_shortwave_websdr",
            "fs524806_frs_walkie_kids",
        ]),
        reason: "limiter at 0.6 ms lookahead on loud speech peaks at −18 LUFS (up to +7.6 dB)",
    },
];

/// The corpus directory: `$STROM_DSP_CORPUS`, else `corpus/data` in the
/// repository.
pub fn corpus_dir() -> PathBuf {
    std::env::var_os("STROM_DSP_CORPUS")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/data"))
}

/// The manifest of the corpus in `corpus` (its own copy, or the
/// repository's next to it).
pub fn manifest(corpus: &Path) -> io::Result<Vec<Clip>> {
    read_manifest(&corpus.join("manifest.tsv"))
        .or_else(|_| read_manifest(&corpus.join("../manifest.tsv")))
}

/// `clip` decoded to 48 kHz stereo, at most `max_seconds`.
pub fn load(corpus: &Path, clip: &Clip, max_seconds: f64) -> io::Result<Vec<f32>> {
    let ch = clip.channels.max(1);
    let x = audio::decode(&corpus.join(&clip.file), ch)?;
    let frames = (x.len() / ch).min((max_seconds * FS) as usize);
    Ok(to_stereo(&x[..frames * ch], ch))
}

/// `x` scaled to an integrated loudness of `lufs`.
pub fn at_loudness(x: &[f32], lufs: f64) -> Vec<f32> {
    let now = metrics::loudness(x, CH, FS).integrated_lufs;
    let g = 10f64.powf((lufs - now) / 20.0);
    x.iter().map(|&v| (v as f64 * g) as f32).collect()
}

/// One chain run on a recording.
pub struct Run {
    /// The input through [`linear_params`].
    pub linear: Vec<f32>,
    /// The chain output.
    pub output: Vec<f32>,
    /// The harness numbers ([`measure_run`], from [`SKIP_SECONDS`]).
    pub numbers: Numbers,
}

impl Run {
    /// A number by key (NaN when absent).
    pub fn get(&self, key: &str) -> f64 {
        self.numbers.get(key).copied().flatten().unwrap_or(f64::NAN)
    }
}

/// Run `x` through `p` and through its linear reference.
pub fn run(p: &BroadcastParams, x: &[f32]) -> Run {
    let mut r = run_chain(p, x.to_vec(), Vec::new());
    r.linear = run_chain(&linear_params(p), x.to_vec(), Vec::new()).output;
    let numbers = measure_run(None, true, p, &r, SKIP_SECONDS);
    Run {
        linear: r.linear,
        output: r.output,
        numbers,
    }
}

/// `x` high-passed at 4 kHz (4th-order Butterworth).
pub fn hp4k(x: &[f32]) -> Vec<f32> {
    let mut y = x.to_vec();
    for q in [0.541_196, 1.306_563] {
        let mut b = Biquad::new(BiquadCoefficients::high_pass(4_000.0, q, FS), CH);
        b.process(&mut y);
    }
    y
}

/// Error of `y` against `r` (stereo), gain matched per 10 ms (see module).
pub fn matched_error(r: &[f32], y: &[f32]) -> Vec<f32> {
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

/// Click count of a run (see the module docs).
pub struct Clicks {
    /// Click events per minute of measured audio.
    pub per_min: f64,
    /// Largest burst relative to the programme around it (dB).
    pub worst_db: f64,
    /// Frame of the largest burst.
    pub worst_at: usize,
    /// Click event start frames.
    pub at: Vec<usize>,
}

/// Clicks of `output` against `reference`, from frame `skip`.
pub fn clicks(reference: &[f32], output: &[f32], skip: usize) -> Clicks {
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
        let (plo, phi) = (k.saturating_sub(10), (k + 11).min(n));
        let prog = rn[plo..phi].iter().sum::<f64>() / (phi - plo) as f64;
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

/// The parameters of `preset` (4 bands) in `mode`.
pub fn params(preset: BroadcastPreset, mode: LatencyMode) -> BroadcastParams {
    let mut p = BroadcastParams::preset(preset, 4);
    mode.apply(&mut p);
    p
}

/// Run `work` items on `jobs` threads; results in the order of `work`.
pub fn parallel<T: Sync, R: Send>(work: &[T], jobs: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = Mutex::new(0usize);
    let out: Mutex<Vec<Option<R>>> = Mutex::new((0..work.len()).map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1) {
            s.spawn(|| loop {
                let i = {
                    let mut n = next.lock().unwrap();
                    let i = *n;
                    *n += 1;
                    i
                };
                let Some(w) = work.get(i) else { break };
                let r = f(w);
                out.lock().unwrap()[i] = Some(r);
            });
        }
    });
    out.into_inner()
        .unwrap()
        .into_iter()
        .map(|r| r.expect("every item ran"))
        .collect()
}

/// One run of the acceptance check: a clip through a configuration.
#[derive(Debug, Clone)]
pub struct AcceptanceRun {
    pub preset: BroadcastPreset,
    pub mode: LatencyMode,
    pub clip: String,
    /// Largest per-frame NMR (dB).
    pub worst_nmr_db: f64,
    /// Mean NMR (dB).
    pub mean_nmr_db: f64,
}

impl AcceptanceRun {
    /// Under [`NMR_MAX_FRAME_BOUND_DB`].
    pub fn pass(&self) -> bool {
        self.worst_nmr_db < NMR_MAX_FRAME_BOUND_DB
    }

    /// The known failure that covers this run, if any.
    pub fn known(&self) -> Option<&'static KnownFailure> {
        KNOWN_FAILURES
            .iter()
            .find(|k| k.covers(self.preset, self.mode, &self.clip))
    }
}

/// Every run of the acceptance check over the speech clips of the corpus in
/// `corpus` (presets × latency modes × clips, in that order), with `tweak`
/// applied to every configuration (identity for the check itself; a stage
/// switched off to measure what it costs).
pub fn acceptance(
    corpus: &Path,
    jobs: usize,
    tweak: impl Fn(&mut BroadcastParams) + Sync,
) -> io::Result<Vec<AcceptanceRun>> {
    let clips: Vec<Clip> = manifest(corpus)?
        .into_iter()
        .filter(|c| SPEECH_CATEGORIES.contains(&c.category.as_str()))
        .collect();
    if clips.is_empty() {
        return Err(io::Error::other("no speech clips in the corpus manifest"));
    }
    let audio: Vec<Vec<f32>> = parallel(&clips, jobs, |c| load(corpus, c, MAX_SECONDS))
        .into_iter()
        .collect::<io::Result<_>>()?;
    let mut work = Vec::new();
    for preset in BroadcastPreset::ALL {
        for mode in LatencyMode::ALL {
            for ci in 0..clips.len() {
                work.push((preset, mode, ci));
            }
        }
    }
    Ok(parallel(&work, jobs, |&(preset, mode, ci)| {
        let mut p = params(preset, mode);
        tweak(&mut p);
        let x = at_loudness(&audio[ci], p.target_loudness_lufs as f64);
        let r = run(&p, &x);
        AcceptanceRun {
            preset,
            mode,
            clip: clips[ci].id.clone(),
            worst_nmr_db: r.get("nmr_max_frame_db"),
            mean_nmr_db: r.get("nmr_mean_db"),
        }
    }))
}

/// The verdict over `runs`: the table (one line per preset × mode: worst
/// NMR over every clip and over the clips no known failure covers, and the
/// result) and the problems (empty when the check passes).
pub fn verdict(runs: &[AcceptanceRun]) -> (String, Vec<String>) {
    let mut table = String::from(
        "preset     mode      worst NMR  clip                              uncovered worst  result\n",
    );
    let mut problems = Vec::new();
    for preset in BroadcastPreset::ALL {
        for mode in LatencyMode::ALL {
            let rs: Vec<&AcceptanceRun> = runs
                .iter()
                .filter(|r| r.preset == preset && r.mode == mode)
                .collect();
            let Some(worst) = rs
                .iter()
                .max_by(|a, b| a.worst_nmr_db.total_cmp(&b.worst_nmr_db))
            else {
                continue;
            };
            let uncovered = rs
                .iter()
                .filter(|r| r.known().is_none())
                .map(|r| r.worst_nmr_db)
                .fold(f64::NEG_INFINITY, f64::max);
            let mut result = "pass".to_string();
            for r in rs.iter().filter(|r| !r.pass() || r.worst_nmr_db.is_nan()) {
                if r.known().is_none() {
                    result = "FAIL".into();
                    problems.push(format!(
                        "{} {} {}: worst-frame NMR {:+.1} dB (bound {NMR_MAX_FRAME_BOUND_DB} dB)",
                        preset.nick(),
                        mode.nick(),
                        r.clip,
                        r.worst_nmr_db
                    ));
                }
            }
            for k in KNOWN_FAILURES
                .iter()
                .filter(|k| k.preset == preset && k.mode == mode)
            {
                let covered: Vec<&&AcceptanceRun> = rs
                    .iter()
                    .filter(|r| k.covers(preset, mode, &r.clip))
                    .collect();
                if covered.iter().all(|r| r.pass()) {
                    result = "FAIL".into();
                    problems.push(format!(
                        "{} {}: listed as a known failure ({}) but passes now: take it off \
                         the list",
                        preset.nick(),
                        mode.nick(),
                        k.reason
                    ));
                } else if result == "pass" {
                    result = "known failure".into();
                }
            }
            let uncovered = if uncovered.is_finite() {
                format!("{uncovered:+.1}")
            } else {
                "(all listed)".into()
            };
            table += &format!(
                "{:<10} {:<9} {:>+8.1}  {:<33} {uncovered:>15}  {result}\n",
                preset.nick(),
                mode.nick(),
                worst.worst_nmr_db,
                worst.clip,
            );
        }
    }
    (table, problems)
}
