//! `stromdsp-deesser`: de-esser element (split-band or wideband).

use gst::glib;
use gst::prelude::*;

mod imp;

/// Value of the `mode` property (GLib enum type `StromDspDeesserMode`).
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone, Copy, Default, glib::Enum)]
#[repr(i32)]
#[enum_type(name = "StromDspDeesserMode")]
pub enum DeesserMode {
    /// Reduce only the band above `frequency` (LR4 split).
    #[default]
    #[enum_value(
        name = "Split-band: reduce only the band above frequency",
        nick = "split-band"
    )]
    SplitBand = 0,
    /// Reduce the whole signal.
    #[enum_value(name = "Wideband: reduce the whole signal", nick = "wideband")]
    Wideband = 1,
}

impl From<DeesserMode> for strom_dsp::deesser::DeesserMode {
    fn from(m: DeesserMode) -> Self {
        match m {
            DeesserMode::SplitBand => Self::SplitBand,
            DeesserMode::Wideband => Self::Wideband,
        }
    }
}

glib::wrapper! {
    /// De-esser element (`stromdsp-deesser`).
    pub struct StromDspDeesser(ObjectSubclass<imp::StromDspDeesser>)
        @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

/// Register `stromdsp-deesser` with `plugin`.
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "stromdsp-deesser",
        gst::Rank::NONE,
        StromDspDeesser::static_type(),
    )
}
