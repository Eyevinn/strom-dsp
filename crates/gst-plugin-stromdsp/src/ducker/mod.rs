//! `stromdsp-ducker`: lowers the programme (pad `sink`) while a key (request
//! pad `sidechain`) is present. A `GstAggregator`; see `imp.rs` for the pad,
//! timing and live-pipeline behaviour.

use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    /// Ducker element (`stromdsp-ducker`).
    pub struct StromDspDucker(ObjectSubclass<imp::StromDspDucker>)
        @extends gst_base::Aggregator, gst::Element, gst::Object;
}

/// Value of the `range-mode` property (GLib enum `StromDspDuckerRangeMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, glib::Enum)]
#[repr(i32)]
#[enum_type(name = "StromDspDuckerRangeMode")]
pub enum RangeMode {
    /// Full `depth` whenever the key is above the threshold.
    #[default]
    #[enum_value(
        name = "Fixed: full depth whenever the key is above the threshold",
        nick = "fixed"
    )]
    Fixed = 0,
    /// 1 dB of reduction per dB the key is above the threshold, up to `depth`.
    #[enum_value(
        name = "Proportional: 1 dB of reduction per dB above the threshold, up to depth",
        nick = "proportional"
    )]
    Proportional = 1,
}

impl From<RangeMode> for strom_dsp::ducker::RangeMode {
    fn from(m: RangeMode) -> Self {
        match m {
            RangeMode::Fixed => Self::Fixed,
            RangeMode::Proportional => Self::Proportional,
        }
    }
}

impl From<strom_dsp::ducker::RangeMode> for RangeMode {
    fn from(m: strom_dsp::ducker::RangeMode) -> Self {
        match m {
            strom_dsp::ducker::RangeMode::Fixed => Self::Fixed,
            strom_dsp::ducker::RangeMode::Proportional => Self::Proportional,
        }
    }
}

/// Register `stromdsp-ducker` with `plugin`.
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    RangeMode::static_type().mark_as_plugin_api(gst::PluginAPIFlags::empty());
    gst::Element::register(
        Some(plugin),
        "stromdsp-ducker",
        gst::Rank::NONE,
        StromDspDucker::static_type(),
    )
}
