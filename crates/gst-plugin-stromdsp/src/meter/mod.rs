//! `stromdsp-meter`: loudness, peak, RMS and stereo-correlation meter.
//!
//! A passthrough element: samples are never changed (bit-exact, buffers pass
//! through untouched). Every `interval` ms of stream time it posts an element
//! message on the bus with the structure described below.
//!
//! # Properties
//!
//! | Property | Type | Unit | Range | Default | Mutable |
//! |---|---|---|---|---|---|
//! | `interval` | UInt (u32) | ms | 10 … 60 000 | 100 | PLAYING |
//! | `true-peak` | Boolean | | | `true` | READY (takes effect at the next caps negotiation) |
//!
//! Action signal `reset` (no arguments, no return value): restarts the
//! loudness measurement (integrated loudness, loudness range, the momentary
//! and short-term windows and the programme peaks) before the next buffer.
//!
//! # Message format
//!
//! An element message whose structure is named [`MESSAGE_NAME`]
//! (`stromdsp-meter`), with these fields:
//!
//! | Field | GType | Unit | Meaning |
//! |---|---|---|---|
//! | `stream-time` | `guint64` (GstClockTime) | ns | stream time of the end of the last buffer measured (`GST_CLOCK_TIME_NONE` if unknown) |
//! | `running-time` | `guint64` (GstClockTime) | ns | running time of the same instant (`GST_CLOCK_TIME_NONE` if unknown) |
//! | `duration` | `guint64` (GstClockTime) | ns | audio covered by the interval values (`rms`, `sample-peak`, `true-peak`) |
//! | `channels` | `gint` | | channel count |
//! | `momentary` | `gdouble` | LUFS | BS.1770 momentary loudness (400 ms) |
//! | `short-term` | `gdouble` | LUFS | short-term loudness (3 s) |
//! | `integrated` | `gdouble` | LUFS | gated integrated loudness since start/`reset` |
//! | `loudness-range` | `gdouble` | LU | EBU Tech 3342 loudness range since start/`reset` (0 without data) |
//! | `sample-peak` | `GstValueArray` of `gdouble` | dBFS | per channel, highest absolute sample in the interval |
//! | `sample-peak-max` | `gdouble` | dBFS | maximum of `sample-peak` over the channels |
//! | `true-peak` | `GstValueArray` of `gdouble` | dBTP | per channel, highest 4×-oversampled peak in the interval (only with `true-peak=true`) |
//! | `true-peak-max` | `gdouble` | dBTP | maximum of `true-peak` (only with `true-peak=true`) |
//! | `programme-sample-peak` | `gdouble` | dBFS | highest sample peak over all channels since start/`reset` |
//! | `programme-true-peak` | `gdouble` | dBTP | highest true peak over all channels since start/`reset` (only with `true-peak=true`) |
//! | `rms` | `GstValueArray` of `gdouble` | dBFS | per channel, `20·log10(√(mean x²))` over the interval (unweighted; a full-scale sine reads −3.01) |
//!
//! Only for exactly 2 channels (L = channel 0, R = channel 1):
//!
//! | Field | GType | Unit | Meaning |
//! |---|---|---|---|
//! | `correlation` | `gdouble` | −1 … +1 | broadband phase correlation (300 ms integration) |
//! | `correlation-low` | `gdouble` | −1 … +1 | correlation below 250 Hz |
//! | `correlation-mid` | `gdouble` | −1 … +1 | correlation 250 Hz … 4 kHz |
//! | `correlation-high` | `gdouble` | −1 … +1 | correlation above 4 kHz |
//! | `stereo-state` | `gchararray` | | `"mono"`, `"stereo"` or `"inverted"` (mono detector with hysteresis) |
//!
//! **Silence and missing values:** every dB/LUFS/dBTP value is finite. A value
//! that would be −∞ (digital silence, or integrated loudness before the first
//! gating block above −70 LUFS) is reported as [`SILENCE_DB`] = **−200.0**,
//! and every such value is clamped to at least −200.0. NaN is never posted.
//! A correlation is 0.0 when either channel (or band) is below −100 dBFS RMS.
//!
//! # Timing and cost
//!
//! Messages are rate-limited by stream time, counted in samples: after every
//! buffer the element adds the buffer's frames to an accumulator, and when the
//! accumulator reaches `round(interval·rate/1000)` frames it posts one message
//! and subtracts that amount (at most one message per buffer; if a buffer is
//! longer than two intervals, the remainder is dropped). Over N frames, the
//! element posts `⌊N / interval_frames⌋` messages when buffers are at most one
//! interval long.
//!
//! Per buffer, the streaming thread only feeds the meters (BS.1770 filters,
//! peaks and optional true peak, RMS sums, correlation, 3-band crossover and
//! mono detector): no allocation, no logging. The loudness *readings* are
//! taken only when a message is due: a short-term read costs about 150 µs and
//! a momentary read about 20 µs at 48 kHz stereo (integrated and LRA < 1 µs),
//! so at the default 100 ms interval reading costs ≈ 0.2 % of one core.
//! Building and posting the message allocates (GstStructure, arrays); this
//! happens once per interval, never per buffer.
//!
//! # Channel positions
//!
//! When the caps carry a non-zero `channel-mask`, each channel's GStreamer
//! position maps to a BS.1770 [`strom_dsp::meter::ChannelRole`]:
//!
//! | GStreamer position | Role (weight) |
//! |---|---|
//! | `LFE1`, `LFE2` | `Lfe` (excluded) |
//! | `SIDE_LEFT`, `SIDE_RIGHT`, `SURROUND_LEFT`, `SURROUND_RIGHT` | `Surround` (1.41) |
//! | `REAR_LEFT`, `REAR_RIGHT` | `Surround` (1.41) if the layout has no side pair (5.x), else `Front` (1.0, 7.x back pair) |
//! | anything else (front, centre, rear centre, wide, top, bottom) | `Front` (1.0) |
//!
//! Without a channel mask (or with mask 0) every channel weighs 1.0
//! (`ChannelLayout::Unknown`).

use gst::glib;
use gst::prelude::*;

mod imp;

/// Name of the message structure posted by `stromdsp-meter`.
pub const MESSAGE_NAME: &str = "stromdsp-meter";

/// Value posted for −∞ dB / LUFS (silence, or no measurement yet); also the
/// lower clamp of every level field.
pub const SILENCE_DB: f64 = -200.0;

glib::wrapper! {
    /// Loudness / peak / correlation meter element (`stromdsp-meter`).
    pub struct StromDspMeter(ObjectSubclass<imp::StromDspMeter>)
        @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

/// Register `stromdsp-meter` with `plugin`.
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "stromdsp-meter",
        gst::Rank::NONE,
        StromDspMeter::static_type(),
    )
}
