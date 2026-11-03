//! `stromdsp-compressor`: compressor element.

use gst::glib;
use gst::prelude::*;

mod imp;

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
