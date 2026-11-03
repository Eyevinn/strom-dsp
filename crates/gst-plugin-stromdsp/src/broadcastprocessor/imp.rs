//! `stromdsp-broadcastprocessor` implementation. See the module
//! documentation in `mod.rs` for the interface.
//!
//! Settings hand-over follows `common.rs` (a `Mutex` plus a generation
//! counter, picked up by the streaming thread with `try_lock`). The
//! structural parameters (`num-bands`, `limiter-lookahead`) are copied into
//! the processor only in `set_caps`, before `configure`; the per-buffer
//! update keeps the configured values, so a write while streaming takes
//! effect at the next negotiation. The `reset` signal sets an atomic flag
//! that the streaming thread consumes. The meter message is built (and
//! allocates) only when it is due.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{LazyLock, Mutex};

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::prelude::*;
use gst_base::subclass::prelude::*;
use strom_dsp::bassclip::BASS_CLIP_THRESHOLD_RANGE_DB;
use strom_dsp::broadcast::{BroadcastMeters, BroadcastParams, BroadcastProcessor, MAX_CROSSOVERS};
use strom_dsp::dynamics::expander::{
    EXPANDER_FLOOR_RANGE_DB, EXPANDER_HOLD_RANGE_MS, EXPANDER_RATIO_RANGE,
    EXPANDER_THRESHOLD_RANGE_DB,
};
use strom_dsp::multiband::{
    ATTACK_RANGE_MS, MAKEUP_RANGE_DB, MAX_BANDS, RATIO_RANGE, RELEASE_RANGE_MS, THRESHOLD_RANGE_DB,
};
use strom_dsp::presets::BroadcastPreset;
use strom_dsp::spike_guard::THRESHOLD_RANGE_DB as SPIKE_THRESHOLD_RANGE_DB;
use strom_dsp::Processor;

use super::{StromDspBroadcastPreset as Preset, MESSAGE_NAME, MIN_CROSSOVER_GAP_HZ, SILENCE_DB};
use crate::common::{self, Engine, SharedSettings};

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "stromdsp-broadcastprocessor",
        gst::DebugColorFlags::empty(),
        Some("strom-dsp broadcast final processor"),
    )
});

/// Crossover frequency range (Hz).
const XO_RANGE: (f32, f32) = (20.0, 20_000.0);
/// Default band count (1 … 5).
const DEFAULT_NUM_BANDS: usize = 4;
const DEFAULT_METER_INTERVAL_MS: u32 = 100;
const MAX_METER_INTERVAL_MS: u32 = 10_000;
/// Rate assumed for the latency before caps are negotiated.
const FALLBACK_RATE: u32 = 48_000;

/// Element settings: the chain parameters plus what belongs to the element.
#[derive(Debug, Clone, Copy)]
struct Settings {
    params: BroadcastParams,
    preset: Preset,
    meter_interval_ms: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            params: BroadcastParams::preset(BroadcastPreset::EbuR128, DEFAULT_NUM_BANDS),
            preset: Preset::EbuR128,
            meter_interval_ms: DEFAULT_METER_INTERVAL_MS,
        }
    }
}

// ---------------------------------------------------------------------------
// Property table
// ---------------------------------------------------------------------------

/// A chain parameter exposed as an `f32` property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FloatField {
    InputGain,
    SpikeThreshold,
    Target,
    Ceiling,
    AgcMaxGain,
    AgcMaxCut,
    AgcRateUp,
    AgcRateDown,
    AgcGate,
    Drive,
    Crossover(usize),
    BandThreshold(usize),
    BandRatio(usize),
    BandAttack(usize),
    BandRelease(usize),
    BandMakeup(usize),
    BandExpanderThreshold(usize),
    BandExpanderRatio(usize),
    BandExpanderFloor(usize),
    ExpanderHold,
    BassClipThreshold,
    LimiterRelease,
    Lookahead,
}

impl FloatField {
    fn get_mut(self, p: &mut BroadcastParams) -> &mut f32 {
        match self {
            Self::InputGain => &mut p.input_gain_db,
            Self::SpikeThreshold => &mut p.spike_threshold_db,
            Self::Target => &mut p.target_loudness_lufs,
            Self::Ceiling => &mut p.true_peak_ceiling_dbtp,
            Self::AgcMaxGain => &mut p.agc_max_gain_db,
            Self::AgcMaxCut => &mut p.agc_max_cut_db,
            Self::AgcRateUp => &mut p.agc_rate_up_db_s,
            Self::AgcRateDown => &mut p.agc_rate_down_db_s,
            Self::AgcGate => &mut p.agc_gate_lufs,
            Self::Drive => &mut p.multiband_drive_db,
            Self::Crossover(k) => &mut p.crossover_hz[k],
            Self::BandThreshold(k) => &mut p.bands[k].threshold_db,
            Self::BandRatio(k) => &mut p.bands[k].ratio,
            Self::BandAttack(k) => &mut p.bands[k].attack_ms,
            Self::BandRelease(k) => &mut p.bands[k].release_ms,
            Self::BandMakeup(k) => &mut p.bands[k].makeup_db,
            Self::BandExpanderThreshold(k) => &mut p.band_expanders[k].threshold_db,
            Self::BandExpanderRatio(k) => &mut p.band_expanders[k].ratio,
            Self::BandExpanderFloor(k) => &mut p.band_expanders[k].floor_db,
            Self::ExpanderHold => &mut p.expander_hold_ms,
            Self::BassClipThreshold => &mut p.bass_clip_threshold_db,
            Self::LimiterRelease => &mut p.limiter_release_ms,
            Self::Lookahead => &mut p.limiter_lookahead_ms,
        }
    }

    fn get(self, p: &BroadcastParams) -> f32 {
        let mut q = *p;
        *self.get_mut(&mut q)
    }
}

/// A chain parameter exposed as a boolean property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoolField {
    Enabled,
    SpikeGuard,
    Agc,
    Multiband,
    BassClipper,
    Limiter,
}

impl BoolField {
    fn get_mut(self, p: &mut BroadcastParams) -> &mut bool {
        match self {
            Self::Enabled => &mut p.enabled,
            Self::SpikeGuard => &mut p.spike_guard_enabled,
            Self::Agc => &mut p.agc_enabled,
            Self::Multiband => &mut p.multiband_enabled,
            Self::BassClipper => &mut p.bass_clipper_enabled,
            Self::Limiter => &mut p.limiter_enabled,
        }
    }

    fn get(self, p: &BroadcastParams) -> bool {
        let mut q = *p;
        *self.get_mut(&mut q)
    }
}

/// Every property, by kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Prop {
    Float(FloatField),
    Bool(BoolField),
    Preset,
    NumBands,
    MeterInterval,
    CrossoverArray,
}

/// One float property: name, nick, blurb, range.
struct FloatSpec {
    name: String,
    nick: String,
    blurb: String,
    range: (f32, f32),
    field: FloatField,
}

fn float_specs() -> &'static [FloatSpec] {
    static SPECS: LazyLock<Vec<FloatSpec>> = LazyLock::new(|| {
        let s = |name: &str, nick: &str, blurb: &str, range: (f32, f32), field| FloatSpec {
            name: name.to_string(),
            nick: nick.to_string(),
            blurb: blurb.to_string(),
            range,
            field,
        };
        let mut v = vec![
            s(
                "input-gain",
                "Input gain",
                "Gain before the chain (dB)",
                (-24.0, 24.0),
                FloatField::InputGain,
            ),
            s(
                "spike-threshold",
                "Spike threshold",
                "Spikes (at most 0.25 ms, isolated) more than this far above the \
                 programme's own peak envelope are brought down to it (dB)",
                SPIKE_THRESHOLD_RANGE_DB,
                FloatField::SpikeThreshold,
            ),
            s(
                "target-loudness",
                "Target loudness",
                "Loudness the AGC steers the programme to (LUFS)",
                (-40.0, -5.0),
                FloatField::Target,
            ),
            s(
                "true-peak-ceiling",
                "True-peak ceiling",
                "Ceiling of the true-peak limiter (dBTP, BS.1770 4x oversampled)",
                (-12.0, 0.0),
                FloatField::Ceiling,
            ),
            s(
                "agc-max-gain",
                "AGC max gain",
                "Largest boost the AGC applies (dB)",
                (0.0, 30.0),
                FloatField::AgcMaxGain,
            ),
            s(
                "agc-max-cut",
                "AGC max cut",
                "Largest cut the AGC applies (dB)",
                (0.0, 30.0),
                FloatField::AgcMaxCut,
            ),
            s(
                "agc-rate-up",
                "AGC rate up",
                "Speed at which the AGC gain rises (dB/s)",
                (0.1, 20.0),
                FloatField::AgcRateUp,
            ),
            s(
                "agc-rate-down",
                "AGC rate down",
                "Speed at which the AGC gain falls (dB/s)",
                (0.1, 40.0),
                FloatField::AgcRateDown,
            ),
            s(
                "agc-gate",
                "AGC gate",
                "Below this input loudness (LUFS) the AGC gain holds (pauses, room tone)",
                (-80.0, -20.0),
                FloatField::AgcGate,
            ),
            s(
                "multiband-drive",
                "Multiband drive",
                "Gain into all bands of the multiband compressor (dB)",
                (-12.0, 12.0),
                FloatField::Drive,
            ),
        ];
        for k in 0..MAX_CROSSOVERS {
            v.push(s(
                &format!("crossover{k}-frequency"),
                &format!("Crossover {k} frequency"),
                &format!(
                    "Crossover {k} between band {k} and band {} (Hz); kept strictly \
                     ascending: a value crossing a neighbour is clamped to just inside it",
                    k + 1
                ),
                XO_RANGE,
                FloatField::Crossover(k),
            ));
        }
        for k in 0..MAX_BANDS {
            v.push(s(
                &format!("band{k}-threshold"),
                &format!("Band {k} threshold"),
                &format!("Compressor threshold of band {k} (dB, sine-calibrated band level)"),
                THRESHOLD_RANGE_DB,
                FloatField::BandThreshold(k),
            ));
            v.push(s(
                &format!("band{k}-ratio"),
                &format!("Band {k} ratio"),
                &format!("Compressor ratio of band {k} (1 = no compression)"),
                RATIO_RANGE,
                FloatField::BandRatio(k),
            ));
            v.push(s(
                &format!("band{k}-attack"),
                &format!("Band {k} attack"),
                &format!("Compressor attack time of band {k} (ms)"),
                ATTACK_RANGE_MS,
                FloatField::BandAttack(k),
            ));
            v.push(s(
                &format!("band{k}-release"),
                &format!("Band {k} release"),
                &format!("Compressor release time of band {k} (ms)"),
                RELEASE_RANGE_MS,
                FloatField::BandRelease(k),
            ));
            v.push(s(
                &format!("band{k}-makeup"),
                &format!("Band {k} makeup"),
                &format!("Makeup gain of band {k} (dB)"),
                MAKEUP_RANGE_DB,
                FloatField::BandMakeup(k),
            ));
            v.push(s(
                &format!("band{k}-expander-threshold"),
                &format!("Band {k} expander threshold"),
                &format!(
                    "Downward expander threshold of band {k} (dB, sine-calibrated band level \
                     like the band threshold); below it the band is attenuated"
                ),
                EXPANDER_THRESHOLD_RANGE_DB,
                FloatField::BandExpanderThreshold(k),
            ));
            v.push(s(
                &format!("band{k}-expander-ratio"),
                &format!("Band {k} expander ratio"),
                &format!(
                    "Downward expander ratio of band {k}: 1 dB below the threshold gives this \
                     many dB below it at the output (1 = off)"
                ),
                EXPANDER_RATIO_RANGE,
                FloatField::BandExpanderRatio(k),
            ));
            v.push(s(
                &format!("band{k}-expander-floor"),
                &format!("Band {k} expander floor"),
                &format!("Largest attenuation of band {k}'s downward expander (dB)"),
                EXPANDER_FLOOR_RANGE_DB,
                FloatField::BandExpanderFloor(k),
            ));
        }
        v.push(s(
            "expander-hold",
            "Expander hold",
            "Time the level must stay low before a band's expander starts to close (ms); \
             keeps word endings and short gaps intact",
            EXPANDER_HOLD_RANGE_MS,
            FloatField::ExpanderHold,
        ));
        v.push(s(
            "bass-clip-threshold",
            "Bass clip threshold",
            "Threshold of the bass clipper on band 0 (dB, analytic envelope: a steady sine \
             reads its peak)",
            BASS_CLIP_THRESHOLD_RANGE_DB,
            FloatField::BassClipThreshold,
        ));
        v.push(s(
            "limiter-release",
            "Limiter release",
            "Release time of the true-peak limiter (ms)",
            (1.0, 2000.0),
            FloatField::LimiterRelease,
        ));
        v
    });
    SPECS.as_ref()
}

/// Names of the boolean properties a preset sets.
const PRESET_BOOLS: [(&str, BoolField); 5] = [
    ("spike-guard-enabled", BoolField::SpikeGuard),
    ("agc-enabled", BoolField::Agc),
    ("multiband-enabled", BoolField::Multiband),
    ("bass-clipper-enabled", BoolField::BassClipper),
    ("limiter-enabled", BoolField::Limiter),
];

/// Every property a preset sets, in interface order: (name, field).
fn preset_props() -> &'static [(String, Prop)] {
    static P: LazyLock<Vec<(String, Prop)>> = LazyLock::new(|| {
        let mut v: Vec<(String, Prop)> = Vec::new();
        let floats = float_specs();
        let push_float = |v: &mut Vec<(String, Prop)>, f: &FloatSpec| {
            v.push((f.name.clone(), Prop::Float(f.field)));
        };
        // Interface order: input, spike guard, AGC, multiband, limiter.
        for f in floats.iter().filter(|f| {
            matches!(
                f.field,
                FloatField::InputGain | FloatField::Target | FloatField::Ceiling
            )
        }) {
            push_float(&mut v, f);
        }
        v.push((
            "spike-guard-enabled".into(),
            Prop::Bool(BoolField::SpikeGuard),
        ));
        for f in floats
            .iter()
            .filter(|f| f.field == FloatField::SpikeThreshold)
        {
            push_float(&mut v, f);
        }
        v.push(("agc-enabled".into(), Prop::Bool(BoolField::Agc)));
        for f in floats.iter().filter(|f| {
            matches!(
                f.field,
                FloatField::AgcMaxGain
                    | FloatField::AgcMaxCut
                    | FloatField::AgcRateUp
                    | FloatField::AgcRateDown
                    | FloatField::AgcGate
            )
        }) {
            push_float(&mut v, f);
        }
        v.push(("multiband-enabled".into(), Prop::Bool(BoolField::Multiband)));
        for f in floats.iter().filter(|f| {
            matches!(
                f.field,
                FloatField::Drive
                    | FloatField::Crossover(_)
                    | FloatField::BandThreshold(_)
                    | FloatField::BandRatio(_)
                    | FloatField::BandAttack(_)
                    | FloatField::BandRelease(_)
                    | FloatField::BandMakeup(_)
                    | FloatField::BandExpanderThreshold(_)
                    | FloatField::BandExpanderRatio(_)
                    | FloatField::BandExpanderFloor(_)
                    | FloatField::ExpanderHold
            )
        }) {
            push_float(&mut v, f);
        }
        v.push((
            "bass-clipper-enabled".into(),
            Prop::Bool(BoolField::BassClipper),
        ));
        for f in floats
            .iter()
            .filter(|f| f.field == FloatField::BassClipThreshold)
        {
            push_float(&mut v, f);
        }
        v.push(("limiter-enabled".into(), Prop::Bool(BoolField::Limiter)));
        for f in floats
            .iter()
            .filter(|f| f.field == FloatField::LimiterRelease)
        {
            push_float(&mut v, f);
        }
        debug_assert_eq!(v.len(), floats.len() + PRESET_BOOLS.len());
        v
    });
    P.as_ref()
}

/// Map a property name to its field.
fn lookup(name: &str) -> Prop {
    match name {
        "enabled" => Prop::Bool(BoolField::Enabled),
        "preset" => Prop::Preset,
        "num-bands" => Prop::NumBands,
        "meter-interval" => Prop::MeterInterval,
        "crossover-frequencies" => Prop::CrossoverArray,
        "limiter-lookahead" => Prop::Float(FloatField::Lookahead),
        _ => {
            if let Some((_, f)) = PRESET_BOOLS.iter().find(|(n, _)| *n == name) {
                return Prop::Bool(*f);
            }
            float_specs()
                .iter()
                .find(|s| s.name == name)
                .map(|s| Prop::Float(s.field))
                .unwrap_or_else(|| unreachable!("unknown property {name}"))
        }
    }
}

// ---------------------------------------------------------------------------
// Crossover ordering
// ---------------------------------------------------------------------------

/// Clamp crossover `k` to lie strictly between its neighbours (at least
/// [`MIN_CROSSOVER_GAP_HZ`] away) and inside the range.
fn clamp_crossover(xo: &[f32; MAX_CROSSOVERS], k: usize, v: f32) -> f32 {
    let lo = if k == 0 {
        XO_RANGE.0
    } else {
        xo[k - 1] + MIN_CROSSOVER_GAP_HZ
    };
    let hi = if k + 1 == MAX_CROSSOVERS {
        XO_RANGE.1
    } else {
        xo[k + 1] - MIN_CROSSOVER_GAP_HZ
    };
    v.clamp(lo, hi.max(lo))
}

/// Make a crossover set strictly ascending (≥ [`MIN_CROSSOVER_GAP_HZ`] apart)
/// inside the range: push up from the lowest, then down from the highest.
fn make_ascending(mut xo: [f32; MAX_CROSSOVERS]) -> [f32; MAX_CROSSOVERS] {
    let n = MAX_CROSSOVERS;
    for k in 0..n {
        let floor = if k == 0 {
            XO_RANGE.0
        } else {
            xo[k - 1] + MIN_CROSSOVER_GAP_HZ
        };
        xo[k] = xo[k].max(floor);
    }
    for k in (0..n).rev() {
        let ceil = if k + 1 == n {
            XO_RANGE.1
        } else {
            xo[k + 1] - MIN_CROSSOVER_GAP_HZ
        };
        xo[k] = xo[k].min(ceil);
    }
    xo
}

/// A numeric array entry as Hz, or `None` if it is not a finite number.
fn entry_hz(v: &glib::SendValue) -> Option<f32> {
    let hz = if let Ok(f) = v.get::<f32>() {
        f
    } else if let Ok(d) = v.get::<f64>() {
        d as f32
    } else if let Ok(i) = v.get::<i32>() {
        i as f32
    } else if let Ok(u) = v.get::<u32>() {
        u as f32
    } else {
        return None;
    };
    hz.is_finite().then(|| hz.clamp(XO_RANGE.0, XO_RANGE.1))
}

// ---------------------------------------------------------------------------
// Streaming state
// ---------------------------------------------------------------------------

/// What the streaming thread owns; built in `set_caps` (allocates).
#[derive(Debug)]
struct Stream {
    engine: Engine<BroadcastProcessor>,
    rate: u32,
    /// Band count in effect (from the last `configure`).
    num_bands: usize,
    /// Frames between meter messages (0 = off).
    interval_frames: u64,
    /// Frames accumulated toward the next message.
    phase: u64,
}

impl Stream {
    fn set_interval(&mut self, ms: u32) {
        self.interval_frames = (ms as f64 * 0.001 * self.rate as f64).round() as u64;
        if ms > 0 {
            self.interval_frames = self.interval_frames.max(1);
        } else {
            self.phase = 0;
        }
    }
}

/// Per-buffer parameter update: everything except the structural fields,
/// which keep the values of the last `configure`.
fn apply(dsp: &mut BroadcastProcessor, s: &Settings) {
    let cur = dsp.params();
    let mut p = s.params;
    p.num_bands = cur.num_bands;
    p.limiter_lookahead_ms = cur.limiter_lookahead_ms;
    dsp.set_params(&p);
}

/// Latency of the chain as time, rounded up to whole nanoseconds.
fn latency_time(rate: u32, lookahead_ms: f32) -> gst::ClockTime {
    let rate = if rate == 0 { FALLBACK_RATE } else { rate };
    let frames = BroadcastProcessor::latency_frames_for(rate as f32, lookahead_ms) as u64;
    gst::ClockTime::from_nseconds(
        (frames * gst::ClockTime::SECOND.nseconds()).div_ceil(u64::from(rate)),
    )
}

/// A level (LUFS, dBTP) for the message: finite, ≥ [`SILENCE_DB`].
fn level(x: f64) -> f64 {
    if x.is_finite() {
        x.max(SILENCE_DB)
    } else {
        SILENCE_DB
    }
}

/// A gain reduction for the message: finite, ≥ 0.
fn reduction(x: f64) -> f64 {
    if x.is_finite() {
        x.max(0.0)
    } else {
        0.0
    }
}

/// The meter message structure (allocates; once per interval).
fn meter_structure(
    m: &BroadcastMeters,
    num_bands: usize,
    running_time: Option<gst::ClockTime>,
    stream_time: Option<gst::ClockTime>,
) -> gst::Structure {
    let ns = |t: Option<gst::ClockTime>| t.map_or(u64::MAX, gst::ClockTime::nseconds);
    let n = num_bands.clamp(1, MAX_BANDS);
    let bands = &m.band_gain_reduction_db[..n];
    let expanders = &m.band_expander_gain_reduction_db[..n];
    gst::Structure::builder(MESSAGE_NAME)
        .field("running-time", ns(running_time))
        .field("stream-time", ns(stream_time))
        .field("input-momentary", level(m.input_momentary_lufs))
        .field("input-short-term", level(m.input_short_term_lufs))
        .field("output-momentary", level(m.output_momentary_lufs))
        .field("output-short-term", level(m.output_short_term_lufs))
        .field("output-integrated", level(m.output_integrated_lufs))
        .field("input-true-peak", level(m.input_true_peak_dbtp))
        .field("output-true-peak", level(m.output_true_peak_dbtp))
        .field("spike-events", m.spike_events)
        .field(
            "agc-gain",
            if m.agc_gain_db.is_finite() {
                m.agc_gain_db
            } else {
                0.0
            },
        )
        .field(
            "band-gain-reduction",
            gst::Array::new(bands.iter().map(|&v| reduction(v))),
        )
        .field(
            "limiter-gain-reduction",
            reduction(m.limiter_gain_reduction_db),
        )
        .field(
            "band-expander-gain-reduction",
            gst::Array::new(expanders.iter().map(|&v| reduction(v))),
        )
        .field("bass-clip-gr", reduction(m.bass_clip_gain_reduction_db))
        .build()
}

/// The structure returned by `get-preset-values` and
/// `get-preset-values-for-bands` for `preset` at `num_bands`.
fn preset_structure(preset: BroadcastPreset, num_bands: usize) -> gst::Structure {
    let p = BroadcastParams::preset(preset, num_bands);
    let mut s = gst::Structure::new_empty(preset.nick());
    for (name, prop) in preset_props() {
        match prop {
            Prop::Float(f) => s.set(name.as_str(), f.get(&p)),
            Prop::Bool(b) => s.set(name.as_str(), b.get(&p)),
            _ => unreachable!(),
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Element
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct StromDspBroadcastProcessor {
    settings: SharedSettings<Settings>,
    stream: Mutex<Option<Stream>>,
    /// Negotiated rate (0 = none) and configured lookahead (f32 bits), for
    /// the latency without touching the streaming lock.
    rate: AtomicU32,
    lookahead_bits: AtomicU32,
    /// `reset` signal: restart the output integrated loudness.
    reset_integrated: AtomicBool,
}

impl StromDspBroadcastProcessor {
    fn latency(&self) -> gst::ClockTime {
        let rate = self.rate.load(Ordering::Acquire);
        let lookahead = if rate == 0 {
            self.settings.get().params.limiter_lookahead_ms
        } else {
            f32::from_bits(self.lookahead_bits.load(Ordering::Acquire))
        };
        latency_time(rate, lookahead)
    }

    fn streaming(&self) -> bool {
        self.rate.load(Ordering::Acquire) != 0
    }

    fn notify_preset_props(&self) {
        let obj = self.obj();
        for (name, _) in preset_props() {
            obj.notify(name);
        }
        obj.notify("crossover-frequencies");
    }

    /// Write a chain parameter. Returns (preset before, preset after,
    /// whether the value changed).
    fn write_param(&self, f: impl FnOnce(&mut BroadcastParams)) -> (Preset, Preset) {
        self.settings.update(|s| {
            let before = s.preset;
            let old = s.params;
            f(&mut s.params);
            if s.params != old {
                s.preset = Preset::Custom;
            }
            (before, s.preset)
        })
    }
}

#[glib::object_subclass]
impl ObjectSubclass for StromDspBroadcastProcessor {
    const NAME: &'static str = "StromDspBroadcastProcessor";
    type Type = super::StromDspBroadcastProcessor;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for StromDspBroadcastProcessor {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPS: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            let d = Settings::default().params;
            let float = |name: &str| {
                let s = float_specs().iter().find(|s| s.name == name).unwrap();
                common::float_prop(
                    &s.name,
                    &s.nick,
                    &s.blurb,
                    s.range.0,
                    s.range.1,
                    s.field.get(&d),
                )
            };
            let mut v = vec![
                common::bool_prop(
                    "enabled",
                    "Enabled",
                    "Process; when false the output is the input at unity gain with the same \
                     delay (latency kept), crossfaded click-free",
                    d.enabled,
                ),
                glib::ParamSpecEnum::builder_with_default("preset", Preset::EbuR128)
                    .nick("Preset")
                    .blurb(
                        "Starting point: sets every property except enabled, meter-interval, \
                         num-bands and limiter-lookahead (for the current num-bands). Reads \
                         custom after any of those properties is changed; writing custom \
                         changes nothing else",
                    )
                    .readwrite()
                    .controllable()
                    .mutable_playing()
                    .build(),
                float("input-gain"),
                float("target-loudness"),
                float("true-peak-ceiling"),
                common::bool_prop(
                    "spike-guard-enabled",
                    "Spike guard enabled",
                    "Remove clicks and other short, isolated peaks before the AGC (bypassed: \
                     unity gain, same delay)",
                    d.spike_guard_enabled,
                ),
                float("spike-threshold"),
                common::bool_prop(
                    "agc-enabled",
                    "AGC enabled",
                    "Run the loudness-targeting AGC (bypassed: unity gain, same delay)",
                    d.agc_enabled,
                ),
                float("agc-max-gain"),
                float("agc-max-cut"),
                float("agc-rate-up"),
                float("agc-rate-down"),
                float("agc-gate"),
                common::bool_prop(
                    "multiband-enabled",
                    "Multiband enabled",
                    "Run the multiband compressor (bypassed: unity gain, same delay)",
                    d.multiband_enabled,
                ),
                glib::ParamSpecUInt::builder("num-bands")
                    .nick("Number of bands")
                    .blurb(
                        "Bands of the multiband compressor (structural: takes effect at the \
                         next caps negotiation); the first num-bands - 1 crossovers are used",
                    )
                    .minimum(1)
                    .maximum(MAX_BANDS as u32)
                    .default_value(DEFAULT_NUM_BANDS as u32)
                    .readwrite()
                    .mutable_ready()
                    .build(),
                float("multiband-drive"),
            ];
            for k in 0..MAX_CROSSOVERS {
                v.push(float(&format!("crossover{k}-frequency")));
            }
            let entry = glib::ParamSpecFloat::builder("crossover-frequency")
                .nick("Crossover frequency")
                .blurb("One crossover frequency (Hz)")
                .minimum(XO_RANGE.0)
                .maximum(XO_RANGE.1)
                .default_value(d.crossover_hz[0])
                .readwrite()
                .build();
            v.push(
                gst::ParamSpecArray::builder("crossover-frequencies")
                    .nick("Crossover frequencies")
                    .blurb(
                        "All crossovers at once (Hz), e.g. <150.0, 1000.0, 5000.0>: entries \
                         (20 .. 20000) are sorted and set as crossover 0, 1, ...; \
                         the rest keep their values; the set is kept strictly ascending",
                    )
                    .element_spec(&entry)
                    .readwrite()
                    .mutable_playing()
                    .build(),
            );
            for k in 0..MAX_BANDS {
                for f in [
                    "threshold",
                    "ratio",
                    "attack",
                    "release",
                    "makeup",
                    "expander-threshold",
                    "expander-ratio",
                    "expander-floor",
                ] {
                    v.push(float(&format!("band{k}-{f}")));
                }
            }
            v.push(float("expander-hold"));
            v.push(common::bool_prop(
                "bass-clipper-enabled",
                "Bass clipper enabled",
                "Run the bass clipper on band 0 (needs 2 or more bands; off: band 0 only \
                 delayed, same latency), crossfaded click-free",
                d.bass_clipper_enabled,
            ));
            v.push(float("bass-clip-threshold"));
            v.push(common::bool_prop(
                "limiter-enabled",
                "Limiter enabled",
                "Run the true-peak limiter (bypassed: unity gain, same delay)",
                d.limiter_enabled,
            ));
            v.push(float("limiter-release"));
            v.push(
                glib::ParamSpecFloat::builder("limiter-lookahead")
                    .nick("Limiter lookahead")
                    .blurb(
                        "Lookahead of the true-peak limiter (ms); part of the reported latency \
                         (structural: takes effect at the next caps negotiation)",
                    )
                    .minimum(0.5)
                    .maximum(20.0)
                    .default_value(d.limiter_lookahead_ms)
                    .readwrite()
                    .mutable_ready()
                    .build(),
            );
            v.push(
                glib::ParamSpecUInt::builder("meter-interval")
                    .nick("Meter interval")
                    .blurb("Time between two meter messages (ms of stream time); 0 = no messages")
                    .minimum(0)
                    .maximum(MAX_METER_INTERVAL_MS)
                    .default_value(DEFAULT_METER_INTERVAL_MS)
                    .readwrite()
                    .controllable()
                    .mutable_playing()
                    .build(),
            );
            v
        });
        PROPS.as_ref()
    }

    fn signals() -> &'static [glib::subclass::Signal] {
        static SIGNALS: LazyLock<Vec<glib::subclass::Signal>> = LazyLock::new(|| {
            vec![
                glib::subclass::Signal::builder("get-preset-values")
                    .action()
                    .param_types([String::static_type()])
                    .return_type::<gst::Structure>()
                    .class_handler(|args| {
                        let obj = args[0]
                            .get::<super::StromDspBroadcastProcessor>()
                            .expect("signal emitted on the broadcast processor");
                        let nick = args[1].get::<Option<String>>().ok().flatten();
                        let preset = nick.as_deref().and_then(BroadcastPreset::from_nick);
                        let s = preset.map(|p| {
                            let n = obj.imp().settings.get().params.num_bands;
                            preset_structure(p, n)
                        });
                        Some(s.to_value())
                    })
                    .build(),
                glib::subclass::Signal::builder("get-preset-values-for-bands")
                    .action()
                    .param_types([String::static_type(), u32::static_type()])
                    .return_type::<gst::Structure>()
                    .class_handler(|args| {
                        let nick = args[1].get::<Option<String>>().ok().flatten();
                        let preset = nick.as_deref().and_then(BroadcastPreset::from_nick);
                        let n = args[2].get::<u32>().unwrap_or(0) as usize;
                        let s = preset
                            .filter(|_| (1..=MAX_BANDS).contains(&n))
                            .map(|p| preset_structure(p, n));
                        Some(s.to_value())
                    })
                    .build(),
                glib::subclass::Signal::builder("reset")
                    .action()
                    .class_handler(|args| {
                        let obj = args[0]
                            .get::<super::StromDspBroadcastProcessor>()
                            .expect("signal emitted on the broadcast processor");
                        let imp = obj.imp();
                        gst::debug!(CAT, imp = imp, "reset of the integrated loudness requested");
                        imp.reset_integrated.store(true, Ordering::Release);
                        None
                    })
                    .build(),
            ]
        });
        SIGNALS.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let name = pspec.name();
        let (before, after) = match lookup(name) {
            Prop::Bool(BoolField::Enabled) => {
                let v: bool = value.get().expect("type checked upstream");
                self.settings.update(|s| s.params.enabled = v);
                gst::debug!(CAT, imp = self, "enabled set to {v}");
                return;
            }
            Prop::MeterInterval => {
                let v: u32 = value.get().expect("type checked upstream");
                self.settings.update(|s| s.meter_interval_ms = v);
                gst::debug!(CAT, imp = self, "meter-interval set to {v} ms");
                return;
            }
            Prop::Float(FloatField::Lookahead) => {
                let v: f32 = value.get().expect("type checked upstream");
                self.settings.update(|s| s.params.limiter_lookahead_ms = v);
                if self.streaming() {
                    gst::warning!(
                        CAT,
                        imp = self,
                        "limiter-lookahead is READY-only: {v} ms takes effect at the next \
                         caps negotiation"
                    );
                } else {
                    gst::debug!(CAT, imp = self, "limiter-lookahead set to {v} ms");
                }
                return;
            }
            Prop::NumBands => {
                let v: u32 = value.get().expect("type checked upstream");
                let preset = self.settings.update(|s| {
                    s.params.num_bands = v as usize;
                    if let Some(p) = s.preset.library() {
                        s.params.apply_preset(p);
                    }
                    s.preset
                });
                if self.streaming() {
                    gst::warning!(
                        CAT,
                        imp = self,
                        "num-bands is READY-only: {v} takes effect at the next caps negotiation"
                    );
                } else {
                    gst::debug!(CAT, imp = self, "num-bands set to {v}");
                }
                if preset != Preset::Custom {
                    self.notify_preset_props();
                }
                return;
            }
            Prop::Preset => {
                let v: Preset = value.get().expect("type checked upstream");
                self.settings.update(|s| {
                    s.preset = v;
                    if let Some(p) = v.library() {
                        s.params.apply_preset(p);
                    }
                });
                gst::debug!(CAT, imp = self, "preset set to {v:?}");
                if v != Preset::Custom {
                    self.notify_preset_props();
                }
                return;
            }
            Prop::CrossoverArray => {
                let arr: gst::Array = value.get().expect("type checked upstream");
                if arr.len() > MAX_CROSSOVERS {
                    gst::warning!(
                        CAT,
                        imp = self,
                        "crossover-frequencies: only the first {MAX_CROSSOVERS} entries are used"
                    );
                }
                let mut entries = [0.0f32; MAX_CROSSOVERS];
                let n = arr.len().min(MAX_CROSSOVERS);
                for (e, v) in entries.iter_mut().zip(arr.iter()) {
                    let Some(hz) = entry_hz(v) else {
                        gst::warning!(
                            CAT,
                            imp = self,
                            "crossover-frequencies: entry {v:?} is not a finite number; \
                             write ignored"
                        );
                        return;
                    };
                    *e = hz;
                }
                if n == 0 {
                    gst::warning!(CAT, imp = self, "crossover-frequencies: empty; ignored");
                    return;
                }
                let new = &mut entries[..n];
                new.sort_by(f32::total_cmp);
                let r = self.write_param(|p| {
                    p.crossover_hz[..n].copy_from_slice(new);
                    p.crossover_hz = make_ascending(p.crossover_hz);
                });
                let xo = self.settings.get().params.crossover_hz;
                gst::debug!(CAT, imp = self, "crossover-frequencies set to {xo:?}");
                let obj = self.obj();
                for k in 0..MAX_CROSSOVERS {
                    obj.notify(&format!("crossover{k}-frequency"));
                }
                r
            }
            Prop::Float(FloatField::Crossover(k)) => {
                let v: f32 = value.get().expect("type checked upstream");
                let mut clamped = v;
                let r = self.write_param(|p| {
                    clamped = clamp_crossover(&p.crossover_hz, k, v);
                    p.crossover_hz[k] = clamped;
                });
                if clamped != v {
                    gst::debug!(
                        CAT,
                        imp = self,
                        "{name}: {v} Hz would break the ascending order; clamped to {clamped} Hz"
                    );
                } else {
                    gst::debug!(CAT, imp = self, "{name} set to {v} Hz");
                }
                self.obj().notify("crossover-frequencies");
                r
            }
            Prop::Float(f) => {
                let v: f32 = value.get().expect("type checked upstream");
                let r = self.write_param(|p| *f.get_mut(p) = v);
                gst::debug!(CAT, imp = self, "{name} set to {v}");
                r
            }
            Prop::Bool(b) => {
                let v: bool = value.get().expect("type checked upstream");
                let r = self.write_param(|p| *b.get_mut(p) = v);
                gst::debug!(CAT, imp = self, "{name} set to {v}");
                r
            }
        };
        if before != after {
            gst::debug!(CAT, imp = self, "preset is now custom");
            self.obj().notify("preset");
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let s = self.settings.get();
        match lookup(pspec.name()) {
            Prop::Float(f) => f.get(&s.params).to_value(),
            Prop::Bool(b) => b.get(&s.params).to_value(),
            Prop::Preset => s.preset.to_value(),
            Prop::NumBands => (s.params.num_bands as u32).to_value(),
            Prop::MeterInterval => s.meter_interval_ms.to_value(),
            Prop::CrossoverArray => gst::Array::new(s.params.crossover_hz).to_value(),
        }
    }
}

impl GstObjectImpl for StromDspBroadcastProcessor {}

impl ElementImpl for StromDspBroadcastProcessor {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static META: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "strom-dsp broadcast processor",
                "Filter/Effect/Audio",
                "Broadcast/streaming final processor: input gain, spike guard, \
                 loudness-targeting AGC, multiband compressor (1-5 bands, LR4) with a \
                 downward expander per band and a bass clipper on band 0, and true-peak \
                 limiter, with presets for EBU R 128, ATSC A/85, streaming and speech. \
                 Reports its latency (spike guard + bass lookahead + limiter lookahead + \
                 oversampling filter); enabled=false keeps the delay. Posts \
                 'stromdsp-broadcastprocessor' meter messages every meter-interval ms.",
                "Eyevinn Technology",
            )
        });
        Some(&*META)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static T: LazyLock<Vec<gst::PadTemplate>> = LazyLock::new(common::pad_templates);
        T.as_ref()
    }
}

impl BaseTransformImpl for StromDspBroadcastProcessor {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        *self.stream.lock().unwrap() = None;
        self.rate.store(0, Ordering::Release);
        gst::debug!(CAT, imp = self, "stopped");
        Ok(())
    }

    fn set_caps(&self, incaps: &gst::Caps, outcaps: &gst::Caps) -> Result<(), gst::LoggableError> {
        let (rate, channels) = common::parse_caps(incaps)?;
        let old_latency = self.latency();
        let (num_bands, lookahead) = {
            let mut guard = self.stream.lock().unwrap();
            let mut dsp = guard.take().map(|s| s.engine.dsp).unwrap_or_default();
            let (s, generation) = self.settings.snapshot();
            // Structural parameters first: `configure` builds the chain from them.
            dsp.set_params(&s.params);
            let mut engine = Engine::new(dsp, rate, channels);
            engine.applied = generation;
            let p = *engine.dsp.params();
            let mut st = Stream {
                engine,
                rate,
                num_bands: p.num_bands,
                interval_frames: 0,
                phase: 0,
            };
            st.set_interval(s.meter_interval_ms);
            *guard = Some(st);
            (p.num_bands, p.limiter_lookahead_ms)
        };
        self.lookahead_bits
            .store(lookahead.to_bits(), Ordering::Release);
        self.rate.store(rate, Ordering::Release);
        let new_latency = self.latency();
        gst::debug!(
            CAT,
            imp = self,
            "configured for {rate} Hz, {channels} channels, {num_bands} bands, \
             lookahead {lookahead} ms: latency {new_latency}"
        );
        if new_latency != old_latency {
            let obj = self.obj();
            let _ = obj.post_message(gst::message::Latency::builder().src(&*obj).build());
        }
        self.parent_set_caps(incaps, outcaps)
    }

    fn query(&self, direction: gst::PadDirection, query: &mut gst::QueryRef) -> bool {
        if direction == gst::PadDirection::Src {
            if let gst::QueryViewMut::Latency(q) = query.view_mut() {
                let sinkpad = self.obj().sink_pad().clone();
                let mut upstream = gst::query::Latency::new();
                if !sinkpad.peer_query(&mut upstream) {
                    return false;
                }
                let (live, min, max) = upstream.result();
                let ours = self.latency();
                let min = min + ours;
                let max = max.map(|m| m + ours);
                gst::debug!(
                    CAT,
                    imp = self,
                    "latency: upstream live={live}, adding {ours}: min {min}, max {}",
                    max.display()
                );
                q.set(live, min, max);
                return true;
            }
        }
        BaseTransformImplExt::parent_query(self, direction, query)
    }

    fn sink_event(&self, event: gst::Event) -> bool {
        if let gst::EventView::FlushStop(_) = event.view() {
            gst::debug!(CAT, imp = self, "flush-stop: resetting state");
            if let Some(st) = self.stream.lock().unwrap().as_mut() {
                st.engine.dsp.reset();
                st.phase = 0;
            }
        }
        self.parent_sink_event(event)
    }

    fn before_transform(&self, inbuf: &gst::BufferRef) {
        common::sync_controlled_values(&*self.obj(), inbuf);
    }

    fn transform_ip(&self, buf: &mut gst::BufferRef) -> Result<gst::FlowSuccess, gst::FlowError> {
        // Process; decide under the lock whether a meter message is due.
        let due = {
            let mut guard = self.stream.lock().unwrap();
            let st = guard.as_mut().ok_or(gst::FlowError::NotNegotiated)?;
            if let Some(s) = self.settings.fetch_if_changed(&mut st.engine.applied) {
                apply(&mut st.engine.dsp, &s);
                st.set_interval(s.meter_interval_ms);
            }
            if buf.flags().contains(gst::BufferFlags::DISCONT) {
                st.engine.dsp.reset();
            }
            if self.reset_integrated.swap(false, Ordering::AcqRel) {
                st.engine.dsp.reset_integrated();
            }
            let frames = {
                let mut map = buf.map_writable().map_err(|_| gst::FlowError::Error)?;
                st.engine.process_bytes(map.as_mut_slice());
                (map.len() / (4 * st.engine.channels)) as u64
            };
            if st.interval_frames == 0 {
                None
            } else {
                st.phase += frames;
                if st.phase >= st.interval_frames {
                    st.phase -= st.interval_frames;
                    if st.phase >= st.interval_frames {
                        // Buffers longer than the interval: one message each.
                        st.phase = 0;
                    }
                    Some((st.engine.dsp.take_meters(), st.num_bands, st.rate, frames))
                } else {
                    None
                }
            }
        };
        let Some((meters, num_bands, rate, frames)) = due else {
            return Ok(gst::FlowSuccess::Ok);
        };
        // Once per interval: times from the segment (takes the object lock),
        // build and post the message.
        let end = buf.pts().map(|pts| {
            pts + buf.duration().unwrap_or_else(|| {
                gst::ClockTime::SECOND
                    .mul_div_floor(frames, u64::from(rate))
                    .unwrap_or(gst::ClockTime::ZERO)
            })
        });
        let obj = self.obj();
        let segment = obj.segment();
        let (running_time, stream_time) = match (end, segment.downcast_ref::<gst::ClockTime>()) {
            (Some(end), Some(seg)) => (seg.to_running_time(end), seg.to_stream_time(end)),
            _ => (None, None),
        };
        let s = meter_structure(&meters, num_bands, running_time, stream_time);
        let _ = obj.post_message(gst::message::Element::builder(s).src(&*obj).build());
        Ok(gst::FlowSuccess::Ok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossover_clamp_keeps_order() {
        let xo = [120.0, 800.0, 4000.0, 10_000.0];
        assert_eq!(clamp_crossover(&xo, 1, 50.0), 121.0);
        assert_eq!(clamp_crossover(&xo, 1, 5000.0), 3999.0);
        assert_eq!(clamp_crossover(&xo, 1, 1000.0), 1000.0);
        assert_eq!(clamp_crossover(&xo, 0, 20.0), 20.0);
        assert_eq!(clamp_crossover(&xo, 3, 20_000.0), 20_000.0);
        assert_eq!(clamp_crossover(&xo, 3, 100.0), 4001.0);
    }

    #[test]
    fn make_ascending_is_strict_and_in_range() {
        for xo in [
            [20.0, 20.0, 20.0, 20.0],
            [20_000.0; 4],
            [5000.0, 12_000.0, 4000.0, 10_000.0],
            [100.0, 200.0, 300.0, 400.0],
        ] {
            let r = make_ascending(xo);
            assert!(r[0] >= XO_RANGE.0 && r[3] <= XO_RANGE.1, "{r:?}");
            for w in r.windows(2) {
                assert!(w[1] - w[0] >= MIN_CROSSOVER_GAP_HZ, "{r:?}");
            }
        }
        // Already ascending: unchanged.
        let ok = [100.0, 200.0, 300.0, 400.0];
        assert_eq!(make_ascending(ok), ok);
        assert_eq!(
            make_ascending([5000.0, 12_000.0, 4000.0, 10_000.0]),
            [5000.0, 12_000.0, 12_001.0, 12_002.0]
        );
    }

    #[test]
    fn message_values_are_finite() {
        gst::init().unwrap();
        let m = BroadcastMeters {
            input_momentary_lufs: f64::NEG_INFINITY,
            output_true_peak_dbtp: f64::NAN,
            agc_gain_db: f64::NAN,
            limiter_gain_reduction_db: -1.0,
            ..BroadcastMeters::default()
        };
        let s = meter_structure(&m, 3, None, None);
        assert_eq!(s.get::<f64>("input-momentary").unwrap(), SILENCE_DB);
        assert_eq!(s.get::<f64>("output-true-peak").unwrap(), SILENCE_DB);
        assert_eq!(s.get::<f64>("agc-gain").unwrap(), 0.0);
        assert_eq!(s.get::<f64>("limiter-gain-reduction").unwrap(), 0.0);
        assert_eq!(s.get::<u64>("running-time").unwrap(), u64::MAX);
        assert_eq!(s.get::<gst::Array>("band-gain-reduction").unwrap().len(), 3);
        assert_eq!(
            s.get::<gst::Array>("band-expander-gain-reduction")
                .unwrap()
                .len(),
            3
        );
        assert_eq!(s.get::<f64>("bass-clip-gr").unwrap(), 0.0);
    }

    #[test]
    fn preset_props_cover_every_preset_field() {
        let names: Vec<&str> = preset_props().iter().map(|(n, _)| n.as_str()).collect();
        // 10 scalars + 4 crossovers + 40 band values (5 x 8) + expander-hold +
        // bass-clip-threshold + limiter-release + 5 booleans.
        assert_eq!(names.len(), 10 + 4 + 40 + 1 + 1 + 1 + 5);
        for excluded in [
            "enabled",
            "preset",
            "meter-interval",
            "num-bands",
            "limiter-lookahead",
            "crossover-frequencies",
        ] {
            assert!(!names.contains(&excluded), "{excluded}");
        }
    }
}
