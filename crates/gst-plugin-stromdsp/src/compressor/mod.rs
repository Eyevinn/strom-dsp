//! `stromdsp-compressor`: compressor element.
//!
//! Additions after the listening tests (additive; every phase 1 property,
//! default and the static curve are unchanged):
//!
//! | Property | Type | Range | Default | Mutable |
//! |---|---|---|---|---|
//! | `detector` | enum [`StromDspCompressorDetector`] (`peak` = 0, `rms` = 1) | | `peak` | PLAYING |
//! | `rms-window` | Float (f32), ms | 1 … 100 | 10 | PLAYING |
//! | `auto-makeup` | Boolean | | `false` | PLAYING |
//!
//! Both detectors are sine-calibrated (a steady sine reads its peak), so the
//! threshold means the same in both; see `strom_dsp::compressor`.

use gst::glib;
use gst::prelude::*;

mod imp;

/// Value type of the compressor's `detector` property (GType
/// `StromDspCompressorDetector`).
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone, Copy, Default, glib::Enum)]
#[repr(i32)]
#[enum_type(name = "StromDspCompressorDetector")]
pub enum StromDspCompressorDetector {
    /// Linked sample peak with a short hold (phase 1 calibration).
    #[default]
    #[enum_value(name = "Peak: linked sample peak", nick = "peak")]
    Peak = 0,
    /// Sine-calibrated RMS over `rms-window`.
    #[enum_value(name = "RMS: sine-calibrated RMS over rms-window", nick = "rms")]
    Rms = 1,
}

glib::wrapper! {
    /// Compressor element (`stromdsp-compressor`).
    pub struct StromDspCompressor(ObjectSubclass<imp::StromDspCompressor>)
        @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

/// Register `stromdsp-compressor` with `plugin`.
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "stromdsp-compressor",
        gst::Rank::NONE,
        StromDspCompressor::static_type(),
    )
}
