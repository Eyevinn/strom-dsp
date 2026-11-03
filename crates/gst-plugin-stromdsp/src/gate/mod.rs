//! `stromdsp-gate`: noise gate element.

use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    /// Noise gate element (`stromdsp-gate`).
    pub struct StromDspGate(ObjectSubclass<imp::StromDspGate>)
        @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

/// Register `stromdsp-gate` with `plugin`.
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "stromdsp-gate",
        gst::Rank::NONE,
        StromDspGate::static_type(),
    )
}
