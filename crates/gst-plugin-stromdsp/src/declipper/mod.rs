//! `stromdsp-declipper`: repairs hard-clipped audio.
//!
//! An in-place `BaseTransform` on interleaved F32LE, any rate and channel
//! count, around [`strom_dsp::declipper::Declipper`]. It finds runs of
//! clipped samples on the waveform (a flat plateau with sharp corners, or
//! single samples at a clip level the input has already shown) and rebuilds
//! them by constrained AR interpolation, each rebuilt sample beyond its
//! clipped value. Everything else passes bit-exact, delayed by the latency.
//! Channels are independent.
//!
//! **Opt-in, one per input.** Put it on an input that is known or likely to
//! clip (an overdriven microphone, a contribution that clipped before it
//! reached Strom). No chain enables it on its own.
//!
//! **Headroom.** The rebuilt peaks lie beyond the old clip level, so the
//! output can exceed the input's peak (by up to `max-overshoot` over the
//! clip level). Put a limiter, or the broadcast processor, after it.
//!
//! # Properties
//!
//! | Property | Type | Unit | Range | Default | Mutable |
//! |---|---|---|---|---|---|
//! | `enabled` | Boolean | | | `true` | PLAYING |
//! | `latency-mode` | enum [`StromDspDeclipperLatency`] | | `low`, `standard`, `high` | `standard` | READY (not controllable) |
//! | `max-overshoot` | Float | dB | 0 … 24 | 12 | PLAYING |
//! | `min-level` | Float | dBFS | −60 … 0 | −20 | PLAYING |
//! | `meter-interval` | UInt | ms | 0 … 10000 (0 = off) | 1000 | PLAYING |
//!
//! `latency-mode` written while a stream is configured is stored and read
//! back at once, but takes effect at the next caps negotiation (a warning
//! is logged): the delay line and the solver are sized for it.
//! `max-overshoot` and `min-level` apply to runs found after the change.
//!
//! # Meter message
//!
//! Every `meter-interval` ms of stream time (counted in processed frames; at
//! most once per buffer, after the buffer that completes the interval;
//! 0 = off) an element message named [`MESSAGE_NAME`] is posted:
//!
//! | Field | Type | Unit | Meaning |
//! |---|---|---|---|
//! | `running-time` | `guint64` | ns | running time of the end of the buffer (`GST_CLOCK_TIME_NONE` without timestamps) |
//! | `stream-time` | `guint64` | ns | stream time of the end of the buffer (likewise) |
//! | `runs-repaired` | `guint64` | | clipped runs rebuilt since the last message |
//! | `samples-repaired` | `guint64` | | samples rebuilt |
//! | `runs-too-long` | `guint64` | | runs longer than the declipper repairs, passed through unchanged |
//! | `fallbacks` | `guint64` | | runs rebuilt by the cheap fallback because the work budget was spent (included in `runs-repaired`) |
//! | `largest-overshoot` | `gdouble` | dB | largest rebuilt peak over its clip level (0 when nothing was rebuilt) |
//!
//! # Latency
//!
//! The `LATENCY` query adds `max(16, round(ms · fs))` frames, as time
//! rounded up to whole nanoseconds, with `ms` the `latency-mode`: `low`
//! 3 ms, `standard` 5 ms, `high` 10 ms (144, 240 and 480 frames at
//! 48 kHz). Before caps are negotiated it assumes 48 kHz. It does not depend
//! on `enabled`. A `Latency` message is posted when it changes, which only
//! happens at a caps negotiation. Timestamps pass through unchanged.
//!
//! `enabled=false` crossfades (10 ms raised cosine) to the input delayed by
//! the same latency; it is never a BaseTransform passthrough.

use gst::glib;
use gst::prelude::*;
use strom_dsp::declipper::DeclipperLatency;

mod imp;

/// Name of the meter message structure.
pub const MESSAGE_NAME: &str = "stromdsp-declipper";

/// Value type of the `latency-mode` property (GType
/// `StromDspDeclipperLatency`).
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone, Copy, Default, glib::Enum)]
#[repr(i32)]
#[enum_type(name = "StromDspDeclipperLatency")]
pub enum StromDspDeclipperLatency {
    /// 3 ms: less context after a run, for two-way use.
    #[enum_value(name = "Low: 3 ms, for commentary, talkback and calls", nick = "low")]
    Low = 0,
    /// 5 ms.
    #[default]
    #[enum_value(name = "Standard: 5 ms", nick = "standard")]
    Standard = 1,
    /// 10 ms: the full context after runs up to 5 ms long.
    #[enum_value(name = "High: 10 ms", nick = "high")]
    High = 2,
}

impl StromDspDeclipperLatency {
    /// The library's latency setting.
    pub fn library(self) -> DeclipperLatency {
        match self {
            Self::Low => DeclipperLatency::Low,
            Self::Standard => DeclipperLatency::Normal,
            Self::High => DeclipperLatency::High,
        }
    }
}

glib::wrapper! {
    /// Declipper element (`stromdsp-declipper`).
    pub struct StromDspDeclipper(ObjectSubclass<imp::StromDspDeclipper>)
        @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

/// Register `stromdsp-declipper` with `plugin`.
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "stromdsp-declipper",
        gst::Rank::NONE,
        StromDspDeclipper::static_type(),
    )
}
