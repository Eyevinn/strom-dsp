//! `stromdsp-equalizer`: parametric equalizer element.

use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    /// Parametric equalizer element (`stromdsp-equalizer`).
    pub struct StromDspEqualizer(ObjectSubclass<imp::StromDspEqualizer>)
        @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

/// Register `stromdsp-equalizer` with `plugin`.
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "stromdsp-equalizer",
        gst::Rank::NONE,
        StromDspEqualizer::static_type(),
    )
}
