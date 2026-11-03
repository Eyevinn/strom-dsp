//! The broadcast chain cases: each harness signal through each preset,
//! reduced to the harness numbers.
//!
//! # Cases
//!
//! A case is a signal, a level and the AGC state:
//!
//! - **Programme cases** (AGC on): the signal is scaled to the preset's
//!   target loudness (pre-levelled, so the AGC need not travel and the run
//!   can stay short), then the preset runs as it is.
//! - **Low-latency cases** (`*_low_latency`): the same with the shortest
//!   lookaheads (spike guard: none).
//! - **Drive cases** (AGC off): the signal's sample peak is set this many dB
//!   over the true-peak ceiling, so the multiband and the limiter work hard
//!   on a steady signal (aliasing and intermodulation).
//!
//! # Numbers per case
//!
//! Measured on the output from `skip` seconds on (the AGC, the trim and the
//! band detectors settle first):
//!
//! | Key | Meaning |
//! |---|---|
//! | `latency_ms` | total latency of the chain (ms) in this configuration |
//! | `loudness_error_lu` | integrated output − target (programme cases) |
//! | `lra_lu` | loudness range of the output |
//! | `max_true_peak_dbtp` | largest output true peak |
//! | `limiter_gr_peaks_mean_db`, `limiter_gr_peaks_max_db` | limiter gain reduction around each mark (−lookahead − 1 ms … +2 ms); mean over marks of the largest value, and the largest (cases with marks) |
//! | `limiter_gr_programme_mean_db`, `limiter_gr_programme_max_db` | limiter gain reduction outside −lookahead − 1 ms … +50 ms of every mark |
//! | `nmr_mean_db`, `nmr_max_frame_db` | [`metrics::nmr`] of the output against the linear reference (below) |
//! | `alias_spur_db` | [`metrics::alias_spur_db`] (15 kHz tone) |
//! | `smpte_imd_db`, `ccif_imd_db` | [`metrics::smpte_imd_db`], [`metrics::ccif_imd_db`] |
//! | `sideband_50hz_dbc`, `sideband_100hz_dbc` | [`metrics::sideband_dbc`] at 3 kHz ± 50 / ± 100 Hz (bass into highs) |
//!
//! Spectral metrics are floored at [`SPECTRAL_FLOOR_DB`] and NMR at
//! [`NMR_FLOOR_DB`]: below those a result is clean, and the exact value is
//! float rounding that differs between machines.
//!
//! The limiter gain reduction is read from the chain's meter after every
//! 1 ms buffer (its value at the end of the buffer).
//!
//! **Linear reference.** The NMR reference is the input through the same
//! chain with every dynamic stage neutral ([`linear_params`]): the same
//! delay and the same crossover all-pass phase, no gain change. A new
//! stage adds its neutral setting there. For signals with clicks the
//! reference is the click-free programme ([`crate::signals::Signal::clean`]),
//! so clicks left in the output count as error and clicks removed do not.
//!
//! Stages that only remove peaks report their clip depth with
//! [`metrics::clip_depth`] under their own keys:
//!
//! | Key | Meaning |
//! |---|---|
//! | `spike_removed_db` | spike guard: removed energy relative to the programme over the measured part (dB; −200 = nothing) |
//! | `spike_removed_max_db` | spike guard: largest per-10 ms-block depth (dB) |
//! | `spike_touched_pct` | spike guard: share of samples changed (%) |
//! | `spike_events` | spike guard: events in the measured part |
//!
//! The spike guard is the first stage after the input gain (0 dB in every
//! case), so these come from a [`SpikeGuard`] with the preset's settings
//! run on the case input, against that input delayed by its latency.

use std::collections::BTreeMap;

use strom_dsp::broadcast::{BroadcastParams, BroadcastProcessor};
use strom_dsp::presets::BroadcastPreset;
use strom_dsp::spike_guard::{self, SpikeGuard, SpikeGuardParams};
use strom_dsp::Processor;

use crate::agc_support::reference_loudness;
use crate::metrics::{self, FFT_LEN};
use crate::signals::{sample_peak, SignalKind, CH};
use strom_dsp::meter::ChannelLayout;

/// Floor of the spectral metrics (dB): below it a signal counts as clean,
/// and float rounding (which differs between SIMD levels) does not show.
pub const SPECTRAL_FLOOR_DB: f64 = -120.0;
/// Floor of the NMR numbers (dB), for the same reason.
pub const NMR_FLOOR_DB: f64 = -80.0;

/// Sample rate of every case.
pub const FS: f64 = 48_000.0;
/// Frames per buffer when running the chain (1 ms).
pub const BLOCK: usize = 48;

/// The numbers of one case, by key. `None` = not applicable.
pub type Numbers = BTreeMap<String, Option<f64>>;

/// Level of a case's input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Level {
    /// Integrated loudness equal to the preset target, AGC on.
    AtTarget,
    /// Sample peak this many dB over the preset's true-peak ceiling, AGC
    /// off.
    OverCeiling(f64),
}

/// One case.
#[derive(Debug, Clone, Copy)]
pub struct Case {
    pub name: &'static str,
    pub signal: SignalKind,
    pub level: Level,
    /// Run the low-latency configuration (shortest lookaheads) instead of
    /// the standard one.
    pub low_latency: bool,
}

/// Every case of the harness, per preset.
pub fn cases() -> Vec<Case> {
    use Level::*;
    use SignalKind::*;
    let c = |name, signal, level| Case {
        name,
        signal,
        level,
        low_latency: false,
    };
    let low = |name, signal, level| Case {
        name,
        signal,
        level,
        low_latency: true,
    };
    vec![
        c("music", Music, AtTarget),
        c("peaky", Peaky(1.1), AtTarget),
        c("peaky_clicks_300ms", Peaky(0.3), AtTarget),
        c("speech_clicks_100ms", SpeechClicks(0.1), AtTarget),
        c("speech_clicks_300ms", SpeechClicks(0.3), AtTarget),
        c("speech_clicks_1s", SpeechClicks(1.0), AtTarget),
        c("speech_3k4", SpeechNoise(3_400.0), AtTarget),
        c("speech_7k", SpeechNoise(7_000.0), AtTarget),
        c("speech_full", SpeechNoise(20_000.0), AtTarget),
        c("sweep", LogSweep, AtTarget),
        c("sine_1k", Sine(1_000.0), AtTarget),
        c("tilted_square", TiltedSquare, AtTarget),
        c("clipped_sine", ClippedSine, AtTarget),
        c("tone_15k_over", Sine(15_000.0), OverCeiling(6.0)),
        c("smpte_over", SmptePair, OverCeiling(6.0)),
        c("ccif_over", CcifPair, OverCeiling(6.0)),
        c("bass_into_highs_over", BassIntoHighs, OverCeiling(6.0)),
        low("music_low_latency", Music, AtTarget),
        low("peaky_clicks_300ms_low_latency", Peaky(0.3), AtTarget),
        low(
            "speech_clicks_300ms_low_latency",
            SpeechClicks(0.3),
            AtTarget,
        ),
    ]
}

/// Run length of a case: total seconds and the seconds skipped before
/// measuring.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunLength {
    pub seconds: f64,
    pub skip_seconds: f64,
}

/// The default (CI) run: 8 s per case, measured from 3 s.
pub const DEFAULT_RUN: RunLength = RunLength {
    seconds: 8.0,
    skip_seconds: 3.0,
};

/// The parameters of `preset` for a case (4 bands, the element default).
pub fn case_params(preset: BroadcastPreset, case: &Case) -> BroadcastParams {
    let mut p = BroadcastParams::preset(preset, 4);
    if let Level::OverCeiling(_) = case.level {
        p.agc_enabled = false;
    }
    if case.low_latency {
        p.spike_guard_lookahead_ms = spike_guard::LOW_LATENCY_LOOKAHEAD_MS;
    }
    p
}

/// `p` with every dynamic stage neutral: spike guard, AGC and limiter
/// bypassed (their
/// dry paths are exact delays), the multiband kept at ratio 1 with no
/// drive or makeup (an all-pass with the same crossovers), input gain 0.
/// The output is the input with the chain's delay and phase only.
pub fn linear_params(p: &BroadcastParams) -> BroadcastParams {
    let mut q = *p;
    q.input_gain_db = 0.0;
    q.spike_guard_enabled = false;
    q.agc_enabled = false;
    q.multiband_enabled = true;
    q.multiband_drive_db = 0.0;
    for b in &mut q.bands {
        b.ratio = 1.0;
        b.makeup_db = 0.0;
    }
    q.limiter_enabled = false;
    q
}

/// What a chain run returns.
#[derive(Debug, Clone)]
pub struct ChainRun {
    /// Input as fed to the chain (scaled).
    pub input: Vec<f32>,
    /// Chain output.
    pub output: Vec<f32>,
    /// Chain latency (frames).
    pub latency: usize,
    /// Limiter gain reduction (dB) after every [`BLOCK`]-frame buffer.
    pub limiter_gr: Vec<f64>,
    /// Event frames of the input.
    pub marks: Vec<usize>,
    /// The input through [`linear_params`] (aligned with `output`).
    pub linear: Vec<f32>,
}

/// Scale and run one case through `params` for `len`.
pub fn run_case(
    preset: BroadcastPreset,
    case: &Case,
    len: RunLength,
) -> (BroadcastParams, ChainRun) {
    let p = case_params(preset, case);
    let frames = (len.seconds * FS) as usize;
    let sig = case.signal.generate(frames, FS);
    let mut input = sig.data;
    let g = match case.level {
        Level::AtTarget => {
            let now = reference_loudness(&input, CH, FS, ChannelLayout::Unknown);
            10f64.powf((p.target_loudness_lufs as f64 - now) / 20.0)
        }
        Level::OverCeiling(db) => {
            10f64.powf((p.true_peak_ceiling_dbtp as f64 + db) / 20.0) / sample_peak(&input)
        }
    };
    let scale = |x: &mut [f32]| x.iter_mut().for_each(|v| *v = (*v as f64 * g) as f32);
    scale(&mut input);
    // The NMR reference: the click-free programme where there is one.
    let reference = match sig.clean {
        Some(mut clean) => {
            scale(&mut clean);
            clean
        }
        None => input.clone(),
    };
    let linear = run_chain(&linear_params(&p), reference, Vec::new()).output;
    let mut run = run_chain(&p, input, sig.marks);
    run.linear = linear;
    (p, run)
}

/// Run `input` through a fresh chain with `params` in [`BLOCK`]-frame
/// buffers, reading the limiter gain reduction after each.
pub fn run_chain(params: &BroadcastParams, input: Vec<f32>, marks: Vec<usize>) -> ChainRun {
    let mut bp = BroadcastProcessor::new();
    bp.set_params(params);
    bp.configure(FS as f32, CH);
    let latency = bp.latency_frames();
    let mut output = input.clone();
    let mut limiter_gr = Vec::with_capacity(output.len() / (BLOCK * CH) + 1);
    for c in output.chunks_mut(BLOCK * CH) {
        bp.process(c);
        limiter_gr.push(bp.take_meters().limiter_gain_reduction_db);
    }
    ChainRun {
        input,
        output,
        latency,
        limiter_gr,
        marks,
        linear: Vec::new(),
    }
}

/// The input delayed by `latency` frames (zeros first), same length.
pub fn delayed(input: &[f32], latency: usize) -> Vec<f32> {
    let mut d = vec![0.0f32; input.len()];
    let shift = (latency * CH).min(input.len());
    d[shift..].copy_from_slice(&input[..input.len() - shift]);
    d
}

/// The harness numbers of one case (see the module docs).
pub fn measure(case: &Case, p: &BroadcastParams, run: &ChainRun, len: RunLength) -> Numbers {
    let skip = (len.skip_seconds * FS) as usize;
    let out = &run.output[skip * CH..];
    let mut n = Numbers::new();
    let mut put = |k: &str, v: Option<f64>| {
        n.insert(k.to_string(), v);
    };

    put("latency_ms", Some(run.latency as f64 * 1000.0 / FS));
    let l = metrics::loudness(out, CH, FS);
    let programme = matches!(case.level, Level::AtTarget);
    put(
        "loudness_error_lu",
        programme.then_some(l.integrated_lufs - p.target_loudness_lufs as f64),
    );
    put("lra_lu", Some(l.lra_lu));
    put("max_true_peak_dbtp", Some(l.max_true_peak_dbtp));

    // Limiter gain reduction, marks in output time from `skip` on.
    let la = (p.limiter_lookahead_ms as f64 * 0.001 * FS).round() as usize;
    let ms = (0.001 * FS) as usize;
    let first_block = skip / BLOCK;
    let series = &run.limiter_gr[first_block.min(run.limiter_gr.len())..];
    let base = first_block * BLOCK;
    let marks: Vec<usize> = run
        .marks
        .iter()
        .map(|&m| m + run.latency)
        .filter(|&m| m >= base + la + ms)
        .map(|m| m - base)
        .collect();
    let gr = metrics::split_by_marks(series, BLOCK, &marks, la + ms, 2 * ms, 50 * ms);
    let has_marks = !run.marks.is_empty();
    put(
        "limiter_gr_peaks_mean_db",
        has_marks.then_some(gr.peaks_mean),
    );
    put("limiter_gr_peaks_max_db", has_marks.then_some(gr.peaks_max));
    put("limiter_gr_programme_mean_db", Some(gr.programme_mean));
    put("limiter_gr_programme_max_db", Some(gr.programme_max));

    let nmr = metrics::nmr(&run.linear, &run.output, CH, FS, skip);
    put("nmr_mean_db", Some(nmr.mean_db.max(NMR_FLOOR_DB)));
    put("nmr_max_frame_db", Some(nmr.max_frame_db.max(NMR_FLOOR_DB)));

    let frames = run.output.len() / CH;
    let from = frames.saturating_sub(FFT_LEN).max(skip);
    let o = &run.output;
    let mut put = |k: &str, v: Option<f64>| put(k, v.map(|x| x.max(SPECTRAL_FLOOR_DB)));
    put(
        "alias_spur_db",
        match case.signal {
            SignalKind::Sine(f) if f > 10_000.0 => {
                Some(metrics::alias_spur_db(o, CH, FS, from, f, 200.0))
            }
            _ => None,
        },
    );
    put(
        "smpte_imd_db",
        (case.signal == SignalKind::SmptePair).then(|| metrics::smpte_imd_db(o, CH, FS, from)),
    );
    put(
        "ccif_imd_db",
        (case.signal == SignalKind::CcifPair).then(|| metrics::ccif_imd_db(o, CH, FS, from)),
    );
    let bih = case.signal == SignalKind::BassIntoHighs;
    put(
        "sideband_50hz_dbc",
        bih.then(|| metrics::sideband_dbc(o, CH, FS, from, 3_000.0, 50.0)),
    );
    put(
        "sideband_100hz_dbc",
        bih.then(|| metrics::sideband_dbc(o, CH, FS, from, 3_000.0, 100.0)),
    );
    spike_numbers(p, run, skip, &mut n);
    n
}

/// The spike guard's numbers (module docs), `None` when it is off.
fn spike_numbers(p: &BroadcastParams, run: &ChainRun, skip: usize, n: &mut Numbers) {
    let keys = [
        "spike_removed_db",
        "spike_removed_max_db",
        "spike_touched_pct",
        "spike_events",
    ];
    if !p.spike_guard_enabled {
        for k in keys {
            n.insert(k.to_string(), None);
        }
        return;
    }
    let mut g = SpikeGuard::new();
    g.set_params(&SpikeGuardParams {
        enabled: true,
        threshold_db: p.spike_threshold_db,
    });
    g.set_lookahead_ms(p.spike_guard_lookahead_ms);
    g.configure(FS as f32, CH);
    let lat = g.latency_frames();
    let mut out = run.input.clone();
    let mut events = 0;
    for (i, c) in out.chunks_mut(BLOCK * CH).enumerate() {
        g.process(c);
        let e = g.take_events();
        if i * BLOCK >= skip + lat {
            events += e;
        }
    }
    let reference = delayed(&run.input, lat);
    let d = metrics::clip_depth(&reference[skip * CH..], &out[skip * CH..], CH, FS);
    let vals = [
        d.overall_db,
        d.max_db,
        100.0 * d.samples_share,
        events as f64,
    ];
    for (k, v) in keys.into_iter().zip(vals) {
        n.insert(k.to_string(), Some(v));
    }
}

/// Run and measure every case of `preset`. Keys are `"<preset>/<case>"`.
pub fn run_preset(preset: BroadcastPreset, len: RunLength) -> BTreeMap<String, Numbers> {
    cases()
        .iter()
        .map(|case| {
            let (p, run) = run_case(preset, case, len);
            (
                format!("{}/{}", preset.nick(), case.name),
                measure(case, &p, &run, len),
            )
        })
        .collect()
}
