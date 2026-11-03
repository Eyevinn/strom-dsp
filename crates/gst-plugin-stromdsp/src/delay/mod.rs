//! `stromdsp-delay`: intentional delay for content alignment (mic alignment,
//! lip sync).
//!
//! The delay is **not** latency: it is not reported in the LATENCY query and
//! timestamps pass through unchanged, so the audio content moves later
//! relative to its timestamps (that is the point: a latency-compensating
//! pipeline would otherwise undo it).

use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    /// Intentional delay element (`stromdsp-delay`).
    pub struct StromDspDelay(ObjectSubclass<imp::StromDspDelay>)
        @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

/// Register `stromdsp-delay` with `plugin`.
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "stromdsp-delay",
        gst::Rank::NONE,
        StromDspDelay::static_type(),
    )
}
