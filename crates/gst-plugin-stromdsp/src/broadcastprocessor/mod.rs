//! `stromdsp-broadcastprocessor`: broadcast/streaming final processor.
//!
//! ```text
//! input gain → spike guard → wideband AGC → multiband compressor → final clipper → true-peak limiter
//!                                           (expander per band,   ↑                        │
//!                                            bass clipper band 0) └─ clip budget governor ←┘
//! ```
//!
//! An in-place `BaseTransform` on interleaved F32LE, any rate and channel
//! count, around [`strom_dsp::broadcast::BroadcastProcessor`].
//!
//! # Properties
//!
//! Units are dB, LUFS, dBTP, dB/s, ms and Hz. Every property is readable,
//! writable, controllable and mutable in PLAYING, except where noted.
//!
//! | Property | Type | Unit | Range | Default | Mutable |
//! |---|---|---|---|---|---|
//! | `enabled` | Boolean | | | `true` | PLAYING |
//! | `preset` | enum [`StromDspBroadcastPreset`] | | `ebu-r128`, `atsc-a85`, `streaming`, `speech`, `custom` | `ebu-r128` | PLAYING |
//! | `input-gain` | Float | dB | −24 … 24 | 0 | PLAYING |
//! | `target-loudness` | Float | LUFS | −40 … −5 | −23 | PLAYING |
//! | `true-peak-ceiling` | Float | dBTP | −12 … 0 | −1 | PLAYING |
//! | `spike-guard-enabled` | Boolean | | | `true` | PLAYING |
//! | `spike-threshold` | Float | dB | 6 … 30 | 8 | PLAYING |
//! | `agc-enabled` | Boolean | | | `true` | PLAYING |
//! | `agc-max-gain` | Float | dB | 0 … 30 | 24 | PLAYING |
//! | `agc-max-cut` | Float | dB | 0 … 30 | 24 | PLAYING |
//! | `agc-rate-up` | Float | dB/s | 0.1 … 20 | 1 | PLAYING |
//! | `agc-rate-down` | Float | dB/s | 0.1 … 40 | 3 | PLAYING |
//! | `agc-gate` | Float | LUFS | −80 … −20 | −50 | PLAYING |
//! | `multiband-enabled` | Boolean | | | `true` | PLAYING |
//! | `num-bands` | UInt | | 1 … 5 | 4 | READY (not controllable) |
//! | `multiband-drive` | Float | dB | −12 … 12 | 0 | PLAYING |
//! | `crossoverN-frequency` (N = 0 … 3) | Float | Hz | 20 … 20000 | 120, 800, 4000, 10000 | PLAYING |
//! | `crossover-frequencies` | `GstValueArray` of Float | Hz | 20 … 20000 each | `<120, 800, 4000, 10000>` | PLAYING (not controllable) |
//! | `bandN-threshold` (N = 0 … 4) | Float | dB | −60 … 0 | −18 | PLAYING |
//! | `bandN-ratio` | Float | | 1 … 20 | 2 | PLAYING |
//! | `bandN-attack` | Float | ms | 0.1 … 500 | 30, 15, 8, 5, 3 | PLAYING |
//! | `bandN-release` | Float | ms | 5 … 5000 | 300, 200, 150, 100, 80 | PLAYING |
//! | `bandN-makeup` | Float | dB | −12 … 24 | 0 | PLAYING |
//! | `bandN-expander-threshold` | Float | dB | −80 … 0 | −47 | PLAYING |
//! | `bandN-expander-ratio` | Float | | 1 … 10 (1 = off) | 1.5 | PLAYING |
//! | `bandN-expander-floor` | Float | dB | 0 … 40 | 4 | PLAYING |
//! | `expander-hold` | Float | ms | 0 … 1000 | 100 | PLAYING |
//! | `bass-clipper-enabled` | Boolean | | | `true` | PLAYING |
//! | `bass-clip-threshold` | Float | dB | −40 … 0 | −14 | PLAYING |
//! | `clipper-enabled` | Boolean | | | `true` | PLAYING |
//! | `clipper-bandwidth` | Float | Hz | 8000 … 22000 | 20000 | PLAYING |
//! | `governor-enabled` | Boolean | | | `true` | PLAYING |
//! | `clip-budget` | Float | dB | 0 … 6 | 2 | PLAYING |
//! | `limiter-enabled` | Boolean | | | `true` | PLAYING |
//! | `limiter-release` | Float | ms | 1 … 2000 | 200 | PLAYING |
//! | `latency-mode` | enum [`StromDspBroadcastLatencyMode`] | | `standard`, `low` | `standard` | READY (not controllable) |
//! | `limiter-lookahead` | Float | ms | 0.5 … 20 | 1.5 | READY (not controllable) |
//! | `meter-interval` | UInt | ms | 0 … 10000 (0 = off) | 100 | PLAYING |
//!
//! The defaults are the `ebu-r128` preset at 4 bands
//! ([`strom_dsp::presets::BroadcastPreset`] documents every value and its
//! reason).
//!
//! **READY-only properties.** `num-bands`, `latency-mode` and
//! `limiter-lookahead` change the structure of the chain (and the
//! latency). A value written while a stream is configured is stored and
//! read back at once, but takes effect at the next caps negotiation (a
//! warning is logged).
//!
//! **`latency-mode` and `limiter-lookahead`.** Writing `latency-mode` sets
//! the spike guard lookahead, the bass clipper lookahead, the final
//! clipper's structure and `limiter-lookahead` to the mode's values
//! ([`strom_dsp::broadcast::LatencyMode`]); `limiter-lookahead` reads the
//! new value (1.5 ms standard, 0.6 ms low). A later write of
//! `limiter-lookahead` overrides the limiter's lookahead only, and
//! `latency-mode` keeps reading the mode written. Writing `latency-mode`
//! again sets all four again. So set `latency-mode` first and
//! `limiter-lookahead` after it, if at all. Presets change neither.
//!
//! **Crossovers** are kept strictly ascending, at least
//! [`MIN_CROSSOVER_GAP_HZ`] apart. A write of `crossoverN-frequency` that
//! would cross a neighbour is clamped to just inside it (one `debug!` line
//! per such write); the read-back gives the clamped value. To move several
//! crossovers one at a time, write them in the direction of travel: when
//! **lowering**, write `crossover0-frequency` first, then 1, 2, 3 (the
//! lowest moves out of the way first); when **raising**, write
//! `crossover3-frequency` first, then 2, 1, 0. Or use
//! `crossover-frequencies`, which sets them all at once. Its entries are
//! validated by GObject against the element spec (`gfloat`, 20 … 20000 Hz:
//! an array with an entry of another type, out of range or NaN is rejected
//! as a whole with a GLib warning, like any out-of-range property value; an
//! empty array is ignored). At most 4 entries are used (extra entries are
//! ignored with a warning). They are sorted and become crossovers 0 … k−1;
//! the others keep their values. The whole set is then made strictly
//! ascending (pushed up from the lowest, then down from the highest, by at
//! most what is needed), so duplicates end up 1 Hz apart. Only the first
//! `num-bands − 1` crossovers are used.
//!
//! # Presets
//!
//! Setting `preset` to anything but `custom` sets every property except
//! `enabled`, `meter-interval`, `num-bands` and `limiter-lookahead`, for the
//! current `num-bands` (band settings and crossovers depend on the band
//! count). `limiter-lookahead` is left out because it is READY-only.
//!
//! - Writing any other property afterwards changes only that property, and
//!   `preset` then reads `custom`. A write that leaves the value as it was
//!   (the same value, or a crossover clamped back to where it was) does not
//!   switch to `custom`. Writing `enabled` or `meter-interval` never does
//!   (the preset does not set them).
//! - Writing `num-bands` while `preset` is not `custom` re-applies that
//!   preset's band settings and crossovers for the new band count, so the
//!   element keeps matching the preset it reports. With `preset=custom` it
//!   changes only `num-bands`.
//! - Writing `custom` changes nothing else; `preset` then reads `custom`.
//! - Every property a preset changes is notified (`notify::<name>`), as is
//!   `preset` when it becomes `custom`.
//!
//! # Action signals
//!
//! - `get-preset-values(nick: gchararray) -> GstStructure`: the values the
//!   preset `nick` would set at the element's current `num-bands`, in a
//!   structure named `nick`, with one field per property the preset sets
//!   (same names and types as the properties: `gfloat` and `gboolean`;
//!   `crossover-frequencies`, which only mirrors the individual crossovers,
//!   is not included). Returns `NULL` for `custom` or an unknown nick.
//! - `get-preset-values-for-bands(nick: gchararray, num_bands: guint) ->
//!   GstStructure`: the same structure, but for `num_bands` bands instead of
//!   the current `num-bands` (band settings and crossovers depend on the
//!   band count), so a host can preview a preset at another band count. It
//!   changes nothing on the element. Returns `NULL` for `custom`, an unknown
//!   nick, or `num_bands` outside 1 … 5.
//! - `reset()`: restarts the output integrated loudness
//!   (`output-integrated` in the meter message).
//!
//! # Meter message
//!
//! Every `meter-interval` ms of stream time (counted in processed frames;
//! at most once per buffer, after the buffer that completes the interval;
//! 0 = off) an element message is posted with this structure:
//!
//! | Field | Type | Unit | Meaning |
//! |---|---|---|---|
//! | `running-time` | `guint64` | ns | running time of the end of the buffer (`GST_CLOCK_TIME_NONE` without timestamps) |
//! | `stream-time` | `guint64` | ns | stream time of the end of the buffer (likewise) |
//! | `input-momentary` | `gdouble` | LUFS | input loudness, 400 ms, before `input-gain` |
//! | `input-short-term` | `gdouble` | LUFS | input loudness, 3 s, before `input-gain` |
//! | `output-momentary` | `gdouble` | LUFS | output loudness, 400 ms |
//! | `output-short-term` | `gdouble` | LUFS | output loudness, 3 s |
//! | `output-integrated` | `gdouble` | LUFS | gated output integrated loudness since start, flush or `reset` |
//! | `input-true-peak` | `gdouble` | dBTP | max input true peak since the last message, before `input-gain` |
//! | `output-true-peak` | `gdouble` | dBTP | max output true peak since the last message |
//! | `spike-events` | `guint64` | | spikes the spike guard removed since the last message (0 while it is off) |
//! | `agc-gain` | `gdouble` | dB | AGC gain, signed (+ = boost) |
//! | `band-gain-reduction` | `GstValueArray` of `gdouble` | dB | positive dB of reduction, one per active band, lowest first |
//! | `limiter-gain-reduction` | `gdouble` | dB | positive dB of reduction |
//! | `band-expander-gain-reduction` | `GstValueArray` of `gdouble` | dB | positive dB of expander attenuation, one per active band, lowest first |
//! | `bass-clip-gr` | `gdouble` | dB | positive dB of bass-clipper reduction, largest since the last message |
//! | `clip-depth` | `gdouble` | dB | energy the final clipper removed relative to its input, since the last message (−200 when nothing) |
//! | `loudness-deficit` | `gdouble` | LU | how far the governor lets the output fall short of the target (≥ 0) |
//! | `governor-state` | `gchararray` | | `off`, `frozen`, `free`, `hold`, `density`, `cut` or `release` |
//!
//! Silence (and anything not yet measured) reads [`SILENCE_DB`] (−200),
//! never NaN or −∞.
//!
//! # Latency
//!
//! The `LATENCY` query adds the configured chain's
//! `BroadcastProcessor::latency_frames_for_params` frames, as time rounded
//! up to whole nanoseconds:
//!
//! `24 + round(spike lookahead · fs) + round(bass lookahead · fs) + C +
//! round(limiter-lookahead · fs) + 32` frames, with the lookaheads in
//! seconds and `C` the final clipper's 69 (standard) or 48 (low) frames.
//! `standard`: 269 frames (5.60 ms) at 48 kHz, 5.83 ms at 44.1 kHz,
//! 4.30 ms at 96 kHz (budget 6 ms). `low`: 133 frames (2.77 ms) at
//! 48 kHz, 2.95 ms at 44.1 kHz, 1.69 ms at 96 kHz (budget 3 ms).
//!
//! Before caps are negotiated it assumes 48 kHz. It does not depend on
//! `enabled` or on stage bypass. A `Latency` message is posted when it
//! changes, which only happens at a caps negotiation (the structural
//! properties are READY-only). Timestamps pass through unchanged.
//!
//! `enabled=false` gives unity gain with the same delay (the chain
//! crossfades click-free); it is never a BaseTransform passthrough.

use gst::glib;
use gst::prelude::*;

mod imp;

/// Name of the meter message structure.
pub const MESSAGE_NAME: &str = "stromdsp-broadcastprocessor";

/// Value reported for silence / no measurement (dB, LUFS, dBTP).
pub const SILENCE_DB: f64 = strom_dsp::broadcast::SILENCE_DB;

/// Smallest distance (Hz) between two neighbouring crossover frequencies.
pub const MIN_CROSSOVER_GAP_HZ: f32 = 1.0;

/// Value type of the `preset` property (GType `StromDspBroadcastPreset`).
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone, Copy, Default, glib::Enum)]
#[repr(i32)]
#[enum_type(name = "StromDspBroadcastPreset")]
pub enum StromDspBroadcastPreset {
    /// EBU R 128 broadcast delivery: −23 LUFS, −1 dBTP.
    #[default]
    #[enum_value(name = "EBU R 128: -23 LUFS, -1 dBTP", nick = "ebu-r128")]
    EbuR128 = 0,
    /// ATSC A/85 (US broadcast): −24 LKFS, −2 dBTP.
    #[enum_value(name = "ATSC A/85: -24 LKFS, -2 dBTP", nick = "atsc-a85")]
    AtscA85 = 1,
    /// Online/streaming (AES TD1004/TD1008 range): −16 LUFS, −1 dBTP.
    #[enum_value(name = "Streaming: -16 LUFS, -1 dBTP", nick = "streaming")]
    Streaming = 2,
    /// Talk and commentary: −18 LUFS, −1 dBTP, gentler multiband.
    #[enum_value(
        name = "Speech: -18 LUFS, -1 dBTP, gentler multiband, faster AGC gate",
        nick = "speech"
    )]
    Speech = 3,
    /// Read-only state after manual edits; writing it changes nothing else.
    #[enum_value(name = "Custom: values edited after a preset", nick = "custom")]
    Custom = 4,
}

/// Value type of the `latency-mode` property (GType
/// `StromDspBroadcastLatencyMode`).
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone, Copy, Default, glib::Enum)]
#[repr(i32)]
#[enum_type(name = "StromDspBroadcastLatencyMode")]
pub enum StromDspBroadcastLatencyMode {
    /// Every stage at its standard latency: at most 6 ms (5.60 ms at
    /// 48 kHz).
    #[default]
    #[enum_value(name = "Standard: at most 6 ms (5.6 ms at 48 kHz)", nick = "standard")]
    Standard = 0,
    /// The shortest settings that keep the true-peak ceiling: at most 3 ms
    /// (2.77 ms at 48 kHz), for two-way use.
    #[enum_value(
        name = "Low: at most 3 ms (2.8 ms at 48 kHz), for commentary, talkback and calls",
        nick = "low"
    )]
    Low = 1,
}

impl StromDspBroadcastLatencyMode {
    /// The library's latency mode.
    pub fn library(self) -> strom_dsp::broadcast::LatencyMode {
        use strom_dsp::broadcast::LatencyMode as M;
        match self {
            Self::Standard => M::Standard,
            Self::Low => M::Low,
        }
    }
}

impl StromDspBroadcastPreset {
    /// The library preset, or `None` for `custom`.
    pub fn library(self) -> Option<strom_dsp::presets::BroadcastPreset> {
        use strom_dsp::presets::BroadcastPreset as P;
        match self {
            Self::EbuR128 => Some(P::EbuR128),
            Self::AtscA85 => Some(P::AtscA85),
            Self::Streaming => Some(P::Streaming),
            Self::Speech => Some(P::Speech),
            Self::Custom => None,
        }
    }
}

glib::wrapper! {
    /// Broadcast final processor element (`stromdsp-broadcastprocessor`).
    pub struct StromDspBroadcastProcessor(ObjectSubclass<imp::StromDspBroadcastProcessor>)
        @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

/// Register `stromdsp-broadcastprocessor` with `plugin`.
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "stromdsp-broadcastprocessor",
        gst::Rank::NONE,
        StromDspBroadcastProcessor::static_type(),
    )
}
