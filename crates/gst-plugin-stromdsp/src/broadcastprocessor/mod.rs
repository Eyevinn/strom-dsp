//! `stromdsp-broadcastprocessor`: broadcast/streaming final processor
//! (phase 2 spec §3).
//!
//! ```text
//! input gain → wideband AGC → multiband compressor → true-peak limiter
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
//! | `agc-enabled` | Boolean | | | `true` | PLAYING |
//! | `agc-max-gain` | Float | dB | 0 … 30 | 12 | PLAYING |
//! | `agc-max-cut` | Float | dB | 0 … 30 | 12 | PLAYING |
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
//! | `limiter-enabled` | Boolean | | | `true` | PLAYING |
//! | `limiter-release` | Float | ms | 1 … 2000 | 200 | PLAYING |
//! | `limiter-lookahead` | Float | ms | 0.5 … 20 | 5 | READY (not controllable) |
//! | `meter-interval` | UInt | ms | 0 … 10000 (0 = off) | 100 | PLAYING |
//!
//! The defaults are the `ebu-r128` preset at 4 bands
//! ([`strom_dsp::presets::BroadcastPreset`] documents every value and its
//! reason).
//!
//! **READY-only properties.** `num-bands` and `limiter-lookahead` change the
//! structure of the chain (and the latency). A value written while a stream
//! is configured is stored and read back at once, but takes effect at the
//! next caps negotiation (a warning is logged).
//!
//! **Crossovers** are kept strictly ascending, at least
//! [`MIN_CROSSOVER_GAP_HZ`] apart. A write of `crossoverN-frequency` that
//! would cross a neighbour is clamped to just inside it (one `debug!` line
//! per such write); the read-back gives the clamped value. To move several
//! crossovers one at a time, write them in the direction of travel: when
//! **lowering**, write `crossover0-frequency` first, then 1, 2, 3 (the
//! lowest moves out of the way first); when **raising**, write
//! `crossover3-frequency` first, then 2, 1, 0. Or use
//! `crossover-frequencies`, which sets them all at once: its entries (at
//! most 4; extra entries are ignored with a warning; a non-numeric or NaN
//! entry rejects the whole write) are clamped to 20 … 20000 Hz, sorted, and
//! become crossovers 0 … k−1; the others keep their values. The whole set is
//! then made strictly ascending (pushed up from the lowest, then down from
//! the highest, by at most what is needed). Only the first `num-bands − 1`
//! crossovers are used.
//!
//! # Presets
//!
//! Setting `preset` to anything but `custom` sets every property except
//! `enabled`, `meter-interval`, `num-bands` and `limiter-lookahead`, for the
//! current `num-bands` (band settings and crossovers depend on the band
//! count). Leaving out `limiter-lookahead` is a documented deviation from
//! phase 2 spec §3.2: it is READY-only.
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
//! | `input-momentary` | `gdouble` | LUFS | input loudness, 400 ms |
//! | `input-short-term` | `gdouble` | LUFS | input loudness, 3 s |
//! | `output-momentary` | `gdouble` | LUFS | output loudness, 400 ms |
//! | `output-short-term` | `gdouble` | LUFS | output loudness, 3 s |
//! | `output-integrated` | `gdouble` | LUFS | gated output integrated loudness since start, flush or `reset` |
//! | `input-true-peak` | `gdouble` | dBTP | max input true peak since the last message |
//! | `output-true-peak` | `gdouble` | dBTP | max output true peak since the last message |
//! | `agc-gain` | `gdouble` | dB | AGC gain, signed (+ = boost) |
//! | `band-gain-reduction` | `GstValueArray` of `gdouble` | dB | positive dB of reduction, one per active band, lowest first |
//! | `limiter-gain-reduction` | `gdouble` | dB | positive dB of reduction |
//!
//! Silence (and anything not yet measured) reads [`SILENCE_DB`] (−200),
//! never NaN or −∞.
//!
//! # Latency
//!
//! The `LATENCY` query adds
//! `BroadcastProcessor::latency_frames_for(rate, limiter-lookahead)` frames
//! (lookahead plus the oversampling filter delay), as time rounded up to
//! whole nanoseconds. Before caps are negotiated it assumes 48 kHz. It does
//! not depend on `enabled` or on stage bypass. A `Latency` message is posted
//! when it changes, which only happens at a caps negotiation (both
//! structural properties are READY-only). Timestamps pass through
//! unchanged.
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
