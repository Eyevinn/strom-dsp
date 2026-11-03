//! `stromdsp-cleanup`: input cleanup element (DC blocker, high-pass, hum notches).

use gst::glib;
use gst::prelude::*;

mod imp;

/// Value of the `hum` property (GLib enum type `StromDspCleanupHum`). The
/// integer values are the mains frequency in Hz (0 = off), and the nicks are
/// `off`, `50` and `60`.
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone, Copy, Default, glib::Enum)]
#[repr(i32)]
#[enum_type(name = "StromDspCleanupHum")]
pub enum CleanupHum {
    /// No hum notches.
    #[default]
    #[enum_value(name = "Off", nick = "off")]
    Off = 0,
    /// 50 Hz mains and harmonics.
    #[enum_value(name = "50 Hz mains", nick = "50")]
    Hz50 = 50,
    /// 60 Hz mains and harmonics.
    #[enum_value(name = "60 Hz mains", nick = "60")]
    Hz60 = 60,
}

impl From<CleanupHum> for strom_dsp::cleanup::HumFrequency {
    fn from(h: CleanupHum) -> Self {
        match h {
            CleanupHum::Off => Self::Off,
            CleanupHum::Hz50 => Self::Hz50,
            CleanupHum::Hz60 => Self::Hz60,
        }
    }
}

glib::wrapper! {
    /// Input cleanup element (`stromdsp-cleanup`).
    pub struct StromDspCleanup(ObjectSubclass<imp::StromDspCleanup>)
        @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

/// Register `stromdsp-cleanup` with `plugin`.
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "stromdsp-cleanup",
        gst::Rank::NONE,
        StromDspCleanup::static_type(),
    )
}
