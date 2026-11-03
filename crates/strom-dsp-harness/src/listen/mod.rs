//! The listening kit: processor configurations run over the test corpus,
//! with before/after audio for listening and the harness numbers per clip
//! and configuration.
//!
//! # What is rendered
//!
//! A **group** is one processor (or one question about a processor), a
//! list of **configurations** and the corpus clips it runs on. Each
//! (group, clip) pair is a **set**: the original and one processed version
//! per configuration, rendered at the same gain so they can be switched
//! while playing. [`groups`] lists them.
//!
//! - Every clip is processed as stereo (a mono clip as dual mono) and
//!   written back with its own channel count.
//! - The original of a set is the clip scaled to [`REFERENCE_LUFS`]
//!   (EBU R 128 alignment, so the element defaults see the level they are
//!   designed for). A configuration may scale its own input further (the
//!   broadcast presets run at their own target, like the harness cases).
//! - Every version is time-aligned with the original: the processor's
//!   latency is removed, so versions can be switched at the same position.
//! - A set is written at one gain for all its versions, chosen so the
//!   loudest sample peak of the set is at [`LISTEN_PEAK_DBFS`]. Loudness
//!   differences between versions are kept; the page can match loudness
//!   on playback.
//!
//! # Adding a processor
//!
//! Add a [`Group`] to [`groups`]: an id, a label, the clips
//! ([`ClipSet`]), the numbers to show ([`Group::columns`]), and one
//! [`Config`] per configuration. A configuration is a closure from the
//! set's stereo input to a [`Rendered`] (output aligned with the input,
//! and its numbers); [`strip_config`] wraps any [`Processor`]. A group
//! whose processor needs a damaged input (e.g. a declipper on clipped
//! speech) sets [`Group::pre`]: its output becomes an extra version of the
//! set and the input of every configuration.

pub mod audio;
pub mod page;

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use serde_json::{json, Value};
use strom_dsp::broadcast::{BroadcastParams, LatencyMode};
use strom_dsp::cleanup::{Cleanup, CleanupParams, HumFrequency};
use strom_dsp::compressor::Compressor;
use strom_dsp::deesser::{Deesser, DeesserParams};
use strom_dsp::gate::{Gate, GateMode};
use strom_dsp::presets::{
    BroadcastPreset, COMPRESSOR_BROADCAST_AUTO_MAKEUP, COMPRESSOR_BROADCAST_DETECTOR,
    COMPRESSOR_BROADCAST_RMS_WINDOW_MS, GATE_BROADCAST_RATIO,
};
use strom_dsp::{CompressorParams, GateParams, Processor};

use crate::chain::{linear_params, measure_run, run_chain, Numbers, FS};
use crate::metrics;
use crate::signals::{sample_peak, CH};

/// Loudness (LUFS) of every set's original.
pub const REFERENCE_LUFS: f64 = -23.0;
/// Largest sample peak of a set as written (dBFS).
pub const LISTEN_PEAK_DBFS: f64 = -1.0;
/// The broadcast numbers are measured from here (s): the AGC, the trim
/// and the band detectors settle first (as in the harness cases).
pub const BROADCAST_SKIP_S: f64 = 3.0;
/// Buffer size for the channel-strip elements (frames, 10 ms).
const STRIP_BLOCK: usize = 480;

/// One clip of the corpus manifest (`manifest.tsv`).
#[derive(Debug, Clone)]
pub struct Clip {
    pub id: String,
    pub category: String,
    /// File name in the corpus directory.
    pub file: String,
    pub channels: usize,
    pub seconds: f64,
    pub source_url: String,
    pub licence: String,
    pub licence_url: String,
    pub attribution: String,
    pub extract: String,
    pub processing: String,
}

/// Read the corpus manifest (tab-separated, header line first).
pub fn read_manifest(path: &Path) -> io::Result<Vec<Clip>> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap_or_default().split('\t').collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .ok_or_else(|| io::Error::other(format!("manifest has no column {name}")))
    };
    let c = [
        col("id")?,
        col("category")?,
        col("file")?,
        col("channels")?,
        col("seconds")?,
        col("source_url")?,
        col("licence")?,
        col("licence_url")?,
        col("attribution")?,
        col("extract")?,
        col("processing")?,
    ];
    let mut clips = Vec::new();
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split('\t').collect();
        let get = |i: usize| f.get(c[i]).copied().unwrap_or_default().to_string();
        clips.push(Clip {
            id: get(0),
            category: get(1),
            file: get(2),
            channels: get(3).parse().unwrap_or(2),
            seconds: get(4).parse().unwrap_or(0.0),
            source_url: get(5),
            licence: get(6),
            licence_url: get(7),
            attribution: get(8),
            extract: get(9),
            processing: get(10),
        });
    }
    Ok(clips)
}

/// What a configuration returns.
pub struct Rendered {
    /// Stereo output, aligned with the input (latency removed), same length.
    pub output: Vec<f32>,
    /// Its numbers.
    pub numbers: Numbers,
}

/// A configuration's processing: stereo input (48 kHz, the set's original
/// or [`Group::pre`] output) to a [`Rendered`].
pub type RunFn = Box<dyn Fn(&[f32]) -> Rendered + Send + Sync>;

/// One configuration of a group.
pub struct Config {
    pub id: String,
    pub label: String,
    /// One sentence: what is set, and why it is here.
    pub about: String,
    pub run: RunFn,
}

/// A step that makes a set's input from its original (e.g. clipping), shown
/// as its own version.
pub struct Pre {
    pub id: &'static str,
    pub label: &'static str,
    pub about: &'static str,
    pub run: fn(&[f32]) -> Vec<f32>,
}

/// The clips a group runs on.
pub enum ClipSet {
    /// These clip ids, in this order.
    Ids(&'static [&'static str]),
    /// Every clip of these categories.
    Categories(&'static [&'static str]),
}

impl ClipSet {
    fn select<'a>(&self, clips: &'a [Clip]) -> Vec<&'a Clip> {
        match self {
            ClipSet::Ids(ids) => ids
                .iter()
                .filter_map(|id| clips.iter().find(|c| c.id == *id))
                .collect(),
            ClipSet::Categories(cats) => clips
                .iter()
                .filter(|c| cats.contains(&c.category.as_str()))
                .collect(),
        }
    }
}

/// One processor (or one question about it): configurations × clips.
pub struct Group {
    pub id: &'static str,
    pub label: &'static str,
    pub about: &'static str,
    pub clips: ClipSet,
    /// Numbers shown in the page's table, as (key, label). Every number is
    /// in the JSON; these are the ones worth reading first.
    pub columns: Vec<(&'static str, &'static str)>,
    pub pre: Option<Pre>,
    pub configs: Vec<Config>,
}

/// Numbers every version gets (measured on the version as rendered,
/// before the set gain).
const COMMON_COLUMNS: [(&str, &str); 3] = [
    ("lufs", "Loudness (LUFS)"),
    ("lra_lu", "LRA (LU)"),
    ("true_peak_dbtp", "True peak (dBTP)"),
];

fn db_to_lin(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

fn scaled(x: &[f32], g: f64) -> Vec<f32> {
    x.iter().map(|&v| (v as f64 * g) as f32).collect()
}

/// `x` advanced by `latency` frames (zeros at the end), same length.
fn advance(x: &[f32], latency: usize) -> Vec<f32> {
    let shift = (latency * CH).min(x.len());
    let mut y = x[shift..].to_vec();
    y.resize(x.len(), 0.0);
    y
}

/// A broadcast processor configuration: the input scaled from
/// [`REFERENCE_LUFS`] to the preset target (the AGC starts on target, as
/// in the harness cases), the harness chain numbers
/// ([`crate::chain::measure_run`], from [`BROADCAST_SKIP_S`]).
pub fn broadcast_config(id: &str, label: &str, about: &str, p: BroadcastParams) -> Config {
    Config {
        id: id.into(),
        label: label.into(),
        about: about.into(),
        run: Box::new(move |input| {
            let trim = p.target_loudness_lufs as f64 - REFERENCE_LUFS;
            let x = scaled(input, db_to_lin(trim));
            let mut run = run_chain(&p, x.clone(), Vec::new());
            run.linear = run_chain(&linear_params(&p), x, Vec::new()).output;
            let mut numbers = measure_run(None, true, &p, &run, BROADCAST_SKIP_S);
            numbers.insert("input_trim_db".into(), Some(trim));
            Rendered {
                output: advance(&run.output, run.latency),
                numbers,
            }
        }),
    }
}

/// A configuration of any [`Processor`] (`make` builds it with its
/// settings), run in 10 ms buffers. Numbers: `latency_ms`; `change_db`,
/// the energy of output − input relative to the input (how much the
/// processor changed, dB); and with `gain_reduction`, its mean and largest
/// value over the buffers (`gain_reduction_mean_db`,
/// `gain_reduction_max_db`).
pub fn strip_config<P: Processor + 'static>(
    id: &str,
    label: &str,
    about: &str,
    make: impl Fn() -> P + Send + Sync + 'static,
    gain_reduction: Option<fn(&P) -> f64>,
) -> Config {
    Config {
        id: id.into(),
        label: label.into(),
        about: about.into(),
        run: Box::new(move |input| {
            let mut p = make();
            p.configure(FS as f32, CH);
            let latency = p.latency_frames();
            let mut out = input.to_vec();
            let mut gr = Vec::with_capacity(out.len() / (STRIP_BLOCK * CH) + 1);
            for c in out.chunks_mut(STRIP_BLOCK * CH) {
                p.process(c);
                if let Some(f) = gain_reduction {
                    gr.push(f(&p));
                }
            }
            let output = advance(&out, latency);
            let mut n = Numbers::new();
            n.insert("latency_ms".into(), Some(latency as f64 * 1000.0 / FS));
            let d = metrics::clip_depth(input, &output, CH, FS);
            n.insert("change_db".into(), Some(d.overall_db));
            if gain_reduction.is_some() {
                let mean = gr.iter().sum::<f64>() / gr.len().max(1) as f64;
                let max = gr.iter().copied().fold(0.0, f64::max);
                n.insert("gain_reduction_mean_db".into(), Some(mean + 0.0));
                n.insert("gain_reduction_max_db".into(), Some(max + 0.0));
            }
            Rendered { output, numbers: n }
        }),
    }
}

/// Every group of the listening kit.
pub fn groups() -> Vec<Group> {
    let broadcast_columns = vec![
        ("latency_ms", "Latency (ms)"),
        ("nmr_mean_db", "NMR (dB)"),
        ("clip_depth_db", "Clip depth (dB)"),
        ("limiter_gr_programme_mean_db", "Limiter GR (dB)"),
        ("expander_gr_mean_db", "Expander GR (dB)"),
        ("bass_clip_gr_mean_db", "Bass clip GR (dB)"),
        ("governor_deficit_lu", "Governor deficit (LU)"),
        ("spike_events", "Spikes"),
    ];

    // Presets × latency modes.
    let mut presets = Vec::new();
    for preset in [
        BroadcastPreset::EbuR128,
        BroadcastPreset::Streaming,
        BroadcastPreset::Speech,
    ] {
        for mode in LatencyMode::ALL {
            let mut p = BroadcastParams::preset(preset, 4);
            mode.apply(&mut p);
            presets.push(broadcast_config(
                &format!("{}-{}", preset.nick(), mode.nick()),
                &format!("{} · {} latency", preset.nick(), mode.nick()),
                &format!(
                    "Preset {} ({} LUFS, {} dBTP), {} latency mode, every stage at the preset's setting.",
                    preset.nick(),
                    p.target_loudness_lufs,
                    p.true_peak_ceiling_dbtp,
                    mode.nick()
                ),
                p,
            ));
        }
    }

    // Stage A/B: every stage on, then each stage off on its own.
    let stage_base = BroadcastParams::preset(BroadcastPreset::Streaming, 4);
    type StageOff = (&'static str, &'static str, fn(&mut BroadcastParams));
    let stages: [StageOff; 5] = [
        ("no-spike-guard", "spike guard off", |p| {
            p.spike_guard_enabled = false
        }),
        ("no-expander", "expander off", |p| {
            p.band_expanders.iter_mut().for_each(|e| e.ratio = 1.0)
        }),
        ("no-bass-clipper", "bass clipper off", |p| {
            p.bass_clipper_enabled = false
        }),
        ("no-final-clipper", "final clipper off", |p| {
            p.clipper_enabled = false
        }),
        ("no-governor", "governor off", |p| {
            p.governor_enabled = false
        }),
    ];
    let mut stage_configs = vec![broadcast_config(
        "all-stages",
        "all stages",
        "Preset streaming, standard latency, every stage on.",
        stage_base,
    )];
    for (id, label, off) in stages {
        let mut p = stage_base;
        off(&mut p);
        stage_configs.push(broadcast_config(
            id,
            label,
            &format!("Preset streaming, standard latency, {label}; everything else as in \"all stages\"."),
            p,
        ));
    }

    // Channel strip on speech, at the element defaults.
    let strip = vec![
        strip_config(
            "gate",
            "gate",
            "stromdsp-gate defaults: gate mode, open −45 / close −50 dBFS, 3 ms attack, 120 ms hold, 200 ms release, −35 dB reduction.",
            || {
                let mut g = Gate::new();
                g.set_params(&GateParams::broadcast());
                g
            },
            Some(|g: &Gate| -20.0 * g.current_gain().max(1e-9).log10()),
        ),
        strip_config(
            "expander",
            "expander",
            "stromdsp-gate in expander mode: 1:2 below −45 dBFS, floored at −35 dB, other settings as the gate.",
            || {
                let mut g = Gate::new();
                g.set_params(&GateParams::broadcast());
                g.set_mode(GateMode::Expander);
                g.set_expander_ratio(GATE_BROADCAST_RATIO);
                g
            },
            Some(|g: &Gate| -20.0 * g.current_gain().max(1e-9).log10()),
        ),
        strip_config(
            "compressor",
            "compressor",
            "stromdsp-compressor defaults: RMS detector, threshold −18 dBFS, 3:1, 12 dB soft knee, 10 ms attack, 150 ms release, no makeup.",
            || {
                let mut c = Compressor::new();
                c.set_params(&CompressorParams::broadcast());
                c.set_detector(COMPRESSOR_BROADCAST_DETECTOR);
                c.set_rms_window(COMPRESSOR_BROADCAST_RMS_WINDOW_MS);
                c.set_auto_makeup(COMPRESSOR_BROADCAST_AUTO_MAKEUP);
                c
            },
            Some(|c: &Compressor| c.gain_reduction_db()),
        ),
        strip_config(
            "deesser",
            "de-esser",
            "stromdsp-deesser defaults: split-band at 6 kHz, threshold −30 dBFS, at most 10 dB, 1 ms attack, 50 ms release.",
            || {
                let mut d = Deesser::new();
                d.set_params(&DeesserParams::default());
                d
            },
            Some(|d: &Deesser| d.gain_reduction_db()),
        ),
        strip_config(
            "cleanup",
            "cleanup",
            "stromdsp-cleanup with the usual speech settings (the element default is all off): DC blocker, 80 Hz 2nd-order high-pass, 50 Hz hum notches (5 harmonics, Q 10).",
            || {
                let mut c = Cleanup::new();
                c.set_params(&CleanupParams {
                    dc_block: true,
                    highpass_hz: 80.0,
                    highpass_order: 2,
                    hum: HumFrequency::Hz50,
                    hum_harmonics: 5,
                    hum_q: 10.0,
                });
                c
            },
            None,
        ),
    ];

    vec![
        Group {
            id: "broadcast-presets",
            label: "Broadcast processor: presets and latency modes",
            about: "stromdsp-broadcastprocessor per preset in the standard and the low latency mode. Each configuration gets the clip at its preset's target loudness.",
            clips: ClipSet::Ids(&[
                "vctk_p226_mic1_concat",
                "librivox_ssc113_02_beeswaxcandle",
                "musan_us-gov-0001_hearing",
                "vctk_p226_g711u_narrowband",
                "fs209301_police_walkies",
                "mix_librivox_mkirkpat_stadium_snr0LU",
                "fs328623_college_football",
                "kmacleod_raving_energy",
                "kmacleod_boogie_party",
                "kmacleod_magistar",
                "fs637102_voice_broken_mic",
                "fs273815_shouted_nonsense_talk",
                "pinknoise_level_steps_0_+20_+10_-10dB_40s",
            ]),
            columns: broadcast_columns.clone(),
            pre: None,
            configs: presets,
        },
        Group {
            id: "broadcast-stages",
            label: "Broadcast processor: each stage on and off",
            about: "stromdsp-broadcastprocessor, preset streaming (the loudest target, where the stages work hardest), standard latency: all stages, then the spike guard, expander, bass clipper, final clipper and governor switched off one at a time.",
            clips: ClipSet::Ids(&[
                "musan_us-gov-0001_hearing",
                "mix_vctk_p225_nycstreet_snr10LU",
                "fs189821_goal_red_bull_arena",
                "fs273815_shouted_nonsense_talk",
                "kmacleod_raving_energy",
                "kmacleod_boogie_party",
            ]),
            columns: broadcast_columns,
            pre: None,
            configs: stage_configs,
        },
        Group {
            id: "channel-strip",
            label: "Channel strip on speech",
            about: "The speech elements on their own, at the element defaults: gate, gate as expander, compressor, de-esser, and cleanup.",
            clips: ClipSet::Ids(&[
                "vctk_p225_mic1_concat",
                "vctk_p245_mic1_concat",
                "librivox_ssc113_10_mkirkpat",
                "librispeech_2277-149874-0000",
                "musan_us-gov-0001_hearing",
                "fs24025_cathedral_tour_guide",
                "fs223292_tour_guide_bus",
                "mix_vctk_p245_musan0064_snr5LU",
            ]),
            columns: vec![
                ("latency_ms", "Latency (ms)"),
                ("gain_reduction_mean_db", "GR mean (dB)"),
                ("gain_reduction_max_db", "GR max (dB)"),
                ("change_db", "Change (dB)"),
            ],
            pre: None,
            configs: strip,
        },
    ]
}

/// What to render.
#[derive(Debug, Clone)]
pub struct RenderOptions {
    /// The unpacked corpus (holds `manifest.tsv` and the clips).
    pub corpus: PathBuf,
    /// Output directory.
    pub out: PathBuf,
    /// Worker threads.
    pub jobs: usize,
    /// Clips longer than this are cut (s).
    pub max_seconds: f64,
    /// Also keep the WAV files (the MP4 files are always written).
    pub wav: bool,
    /// Only these groups (ids); all if `None`.
    pub groups: Option<Vec<String>>,
    /// Only these clips (ids); all of each group's if `None`.
    pub clips: Option<Vec<String>>,
}

/// Mono or multichannel `x` as stereo (mono duplicated, beyond two
/// channels the first two).
fn to_stereo(x: &[f32], ch: usize) -> Vec<f32> {
    let mut y = Vec::with_capacity(x.len() / ch * 2);
    for f in x.chunks_exact(ch) {
        let (l, r) = if ch == 1 { (f[0], f[0]) } else { (f[0], f[1]) };
        y.push(l);
        y.push(r);
    }
    y
}

/// Stereo `x` back to `ch` channels (mono: the mean of both).
fn from_stereo(x: &[f32], ch: usize) -> Vec<f32> {
    if ch != 1 {
        return x.to_vec();
    }
    x.chunks_exact(2).map(|f| 0.5 * (f[0] + f[1])).collect()
}

/// The numbers every version gets.
fn common_numbers(x: &[f32], n: &mut Numbers) {
    let l = metrics::loudness(x, CH, FS);
    let fin = |v: f64| v.is_finite().then_some(v);
    n.insert("lufs".into(), fin(l.integrated_lufs));
    n.insert("lra_lu".into(), fin(l.lra_lu));
    n.insert("true_peak_dbtp".into(), fin(l.max_true_peak_dbtp));
    n.insert(
        "sample_peak_dbfs".into(),
        fin(20.0 * sample_peak(x).max(1e-12).log10()),
    );
}

fn numbers_json(n: &Numbers) -> Value {
    Value::Object(
        n.iter()
            .map(|(k, v)| {
                let v = v
                    .filter(|x| x.is_finite())
                    .map_or(Value::Null, |x| json!((x * 1000.0).round() / 1000.0));
                (k.clone(), v)
            })
            .collect(),
    )
}

/// Render one set: returns its JSON entry.
fn render_set(group: &Group, clip: &Clip, opts: &RenderOptions) -> io::Result<Value> {
    let native = audio::decode(&opts.corpus.join(&clip.file), clip.channels)?;
    let ch = clip.channels.max(1);
    let max_frames = (opts.max_seconds * FS) as usize;
    let frames = (native.len() / ch).min(max_frames);
    let mut x = to_stereo(&native[..frames * ch], ch);
    let now = metrics::loudness(&x, CH, FS).integrated_lufs;
    if now.is_finite() && now > -70.0 {
        x = scaled(&x, db_to_lin(REFERENCE_LUFS - now));
    }

    // (id, label, stereo audio, numbers)
    let mut versions: Vec<(String, String, Vec<f32>, Numbers)> = Vec::new();
    let mut n = Numbers::new();
    n.insert("clip_lufs".into(), now.is_finite().then_some(now));
    versions.push(("original".into(), "original".into(), x.clone(), n));
    let input = match &group.pre {
        Some(pre) => {
            let y = (pre.run)(&x);
            versions.push((pre.id.into(), pre.label.into(), y.clone(), Numbers::new()));
            y
        }
        None => x,
    };
    for c in &group.configs {
        let r = (c.run)(&input);
        if r.output.len() != input.len() || r.output.iter().any(|v| !v.is_finite()) {
            return Err(io::Error::other(format!(
                "{}/{}/{}: output has the wrong length or is not finite",
                group.id, clip.id, c.id
            )));
        }
        versions.push((c.id.clone(), c.label.clone(), r.output, r.numbers));
    }

    let peak = versions
        .iter()
        .map(|v| sample_peak(&v.2))
        .fold(1e-9, f64::max);
    let gain_db = LISTEN_PEAK_DBFS - 20.0 * peak.log10();
    let gain = db_to_lin(gain_db);

    let rel = format!("sets/{}/{}", group.id, clip.id);
    let dir = opts.out.join(&rel);
    std::fs::create_dir_all(&dir)?;
    let mut out_versions = Vec::new();
    for (id, label, audio_, mut numbers) in versions {
        common_numbers(&audio_, &mut numbers);
        let wav = dir.join(format!("{id}.wav"));
        let mp4 = dir.join(format!("{id}.mp4"));
        audio::write_wav24(&wav, &from_stereo(&audio_, ch), ch, gain)?;
        audio::encode_mp4(&wav, &mp4, ch)?;
        if !opts.wav {
            std::fs::remove_file(&wav)?;
        }
        let mut files = json!({ "mp4": format!("{rel}/{id}.mp4") });
        if opts.wav {
            files["wav"] = json!(format!("{rel}/{id}.wav"));
        }
        out_versions.push(json!({
            "id": id,
            "label": label,
            "files": files,
            "numbers": numbers_json(&numbers),
        }));
    }
    let set = json!({
        "group": group.id,
        "clip": clip.id,
        "dir": rel,
        "seconds": frames as f64 / FS,
        "gain_db": (gain_db * 100.0).round() / 100.0,
        "versions": out_versions,
    });
    std::fs::write(
        dir.join("metrics.json"),
        serde_json::to_string_pretty(&set).map_err(io::Error::other)?,
    )?;
    Ok(set)
}

/// Run every selected group over the corpus into `opts.out`, and write
/// `listen-data.json` (groups, clips, sets with every number). Progress
/// goes to stderr.
pub fn render(opts: &RenderOptions) -> io::Result<Value> {
    let clips = read_manifest(&opts.corpus.join("manifest.tsv"))?;
    let groups: Vec<Group> = groups()
        .into_iter()
        .filter(|g| {
            opts.groups
                .as_ref()
                .is_none_or(|ids| ids.iter().any(|i| i == g.id))
        })
        .collect();
    let mut jobs = Vec::new();
    for (gi, g) in groups.iter().enumerate() {
        let selected = g.clips.select(&clips);
        if selected.is_empty() {
            eprintln!("warning: group {} has no clips in the corpus", g.id);
        }
        for c in selected {
            if opts.clips.as_ref().is_none_or(|ids| ids.contains(&c.id)) {
                jobs.push((gi, c));
            }
        }
    }
    std::fs::create_dir_all(&opts.out)?;
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<(usize, io::Result<Value>)>> = Mutex::new(Vec::new());
    let start = Instant::now();
    let total = jobs.len();
    std::thread::scope(|s| {
        for _ in 0..opts.jobs.max(1) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(gi, clip)) = jobs.get(i) else {
                    break;
                };
                let t = Instant::now();
                let r = render_set(&groups[gi], clip, opts);
                eprintln!(
                    "[{:>3}/{total}] {}/{} {:.1} s{}",
                    i + 1,
                    groups[gi].id,
                    clip.id,
                    t.elapsed().as_secs_f64(),
                    match &r {
                        Ok(_) => String::new(),
                        Err(e) => format!(" FAILED: {e}"),
                    }
                );
                results.lock().expect("results").push((i, r));
            });
        }
    });
    let mut results = results.into_inner().expect("results");
    results.sort_by_key(|r| r.0);
    let mut sets = Vec::new();
    for (_, r) in results {
        sets.push(r?);
    }

    let used: BTreeMap<&str, &Clip> = jobs.iter().map(|(_, c)| (c.id.as_str(), *c)).collect();
    let data = json!({
        "reference_lufs": REFERENCE_LUFS,
        "listen_peak_dbfs": LISTEN_PEAK_DBFS,
        "broadcast_skip_s": BROADCAST_SKIP_S,
        "render_seconds": (start.elapsed().as_secs_f64() * 10.0).round() / 10.0,
        "groups": groups.iter().map(|g| json!({
            "id": g.id,
            "label": g.label,
            "about": g.about,
            "columns": COMMON_COLUMNS.iter().chain(g.columns.iter())
                .map(|(k, l)| json!([k, l])).collect::<Vec<_>>(),
            "pre": g.pre.as_ref().map(|p| json!({"id": p.id, "label": p.label, "about": p.about})),
            "configs": g.configs.iter().map(|c| json!({
                "id": c.id, "label": c.label, "about": c.about,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "clips": used.values().map(|c| json!({
            "id": c.id,
            "category": c.category,
            "channels": c.channels,
            "source_url": c.source_url,
            "licence": c.licence,
            "licence_url": c.licence_url,
            "attribution": c.attribution,
            "extract": c.extract,
            "processing": c.processing,
        })).collect::<Vec<_>>(),
        "sets": sets,
    });
    std::fs::write(
        opts.out.join("listen-data.json"),
        serde_json::to_string_pretty(&data).map_err(io::Error::other)?,
    )?;
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_removes_latency() {
        let x: Vec<f32> = (0..10).map(|v| v as f32).collect();
        assert_eq!(advance(&x, 2), vec![4., 5., 6., 7., 8., 9., 0., 0., 0., 0.]);
    }

    #[test]
    fn stereo_round_trip() {
        let m = [0.1f32, 0.2, 0.3];
        let s = to_stereo(&m, 1);
        assert_eq!(s, vec![0.1, 0.1, 0.2, 0.2, 0.3, 0.3]);
        assert_eq!(from_stereo(&s, 1), m.to_vec());
    }

    /// Every configuration returns output aligned with its input: a
    /// broadband burst comes out at the same frame (the cross-correlation
    /// of input and output peaks at lag 0).
    #[test]
    fn configurations_are_time_aligned() {
        let frames = (5.0 * FS) as usize;
        let (at, len) = ((4.0 * FS) as usize, (0.03 * FS) as usize);
        let mut seed = 1u32;
        let mut noise = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        // Quiet noise, then a 30 ms Hann-shaped noise burst.
        let mut x: Vec<f32> = (0..frames * CH).map(|_| noise() * 0.01).collect();
        for i in 0..len {
            let w = (std::f32::consts::PI * i as f32 / len as f32).sin().powi(2);
            let v = noise() * 0.6 * w;
            x[(at + i) * CH] = v;
            x[(at + i) * CH + 1] = v;
        }
        let corr = |y: &[f32], lag: isize| -> f64 {
            (at..at + len)
                .map(|i| x[i * CH] as f64 * y[((i as isize + lag) as usize) * CH] as f64)
                .sum()
        };
        for g in groups() {
            for c in g.configs.iter() {
                let r = (c.run)(&x);
                assert_eq!(r.output.len(), x.len(), "{}/{}", g.id, c.id);
                let best = (-480isize..=480)
                    .max_by(|&a, &b| corr(&r.output, a).total_cmp(&corr(&r.output, b)))
                    .unwrap();
                // A few frames of slack for the phase of minimum-phase filters
                // (de-esser split, crossovers); any lookahead left uncompensated
                // is 0.6 ms (29 frames) or more.
                assert!(
                    best.abs() <= 8,
                    "{}/{}: output lags the input by {best} frames",
                    g.id,
                    c.id
                );
            }
        }
    }

    #[test]
    fn group_and_config_ids_are_unique_and_clips_exist() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/manifest.tsv");
        let clips = read_manifest(&manifest).expect("corpus manifest");
        let mut gids = Vec::new();
        for g in groups() {
            gids.push(g.id);
            let mut cids: Vec<&str> = g.configs.iter().map(|c| c.id.as_str()).collect();
            cids.push("original");
            let n = cids.len();
            cids.sort();
            cids.dedup();
            assert_eq!(cids.len(), n, "{}: duplicate configuration ids", g.id);
            if let ClipSet::Ids(ids) = g.clips {
                for id in ids {
                    assert!(
                        clips.iter().any(|c| c.id == *id),
                        "{}: clip {id} not in the manifest",
                        g.id
                    );
                }
            }
        }
        let n = gids.len();
        gids.sort();
        gids.dedup();
        assert_eq!(gids.len(), n);
    }
}
