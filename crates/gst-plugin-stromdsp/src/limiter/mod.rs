//! `stromdsp-limiter`: lookahead limiter element.

use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    /// Lookahead limiter element (`stromdsp-limiter`).
    pub struct StromDspLimiter(ObjectSubclass<imp::StromDspLimiter>)
        @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

/// Register `stromdsp-limiter` with `plugin`.
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "stromdsp-limiter",
        gst::Rank::NONE,
        StromDspLimiter::static_type(),
    )
}
