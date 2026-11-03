//! `stromdsp-gate`: noise gate element.
//!
//! Phase 1b adds two properties (phase 1b spec §3.3); every
//! phase 1 property, default and behaviour is unchanged:
//!
//! | Property | Type | Range | Default | Mutable |
//! |---|---|---|---|---|
//! | `mode` | enum [`StromDspGateMode`] (`gate` = 0, `expander` = 1) | | `gate` | PLAYING |
//! | `ratio` | Float (f32), expansion ratio (2 = 1:2) | 1 … 20 | 2 | PLAYING |
//!
//! In `expander` mode the closed gain follows the downward-expander curve
//! `max(reduction, (ratio − 1)·(level − open-threshold))` dB instead of the
//! fixed `reduction` (see `strom_dsp::gate`). `ratio` is ignored in `gate`
//! mode, which is bit-identical to the phase 1 element.

use gst::glib;
use gst::prelude::*;

mod imp;

/// Value type of the gate's `mode` property (GType `StromDspGateMode`).
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone, Copy, Default, glib::Enum)]
#[repr(i32)]
#[enum_type(name = "StromDspGateMode")]
pub enum StromDspGateMode {
    /// Classic gate: fixed attenuation `reduction` when closed (phase 1).
    #[default]
    #[enum_value(name = "Gate: fixed attenuation when closed", nick = "gate")]
    Gate = 0,
    /// Downward expander below `open-threshold` with slope `ratio`, floored
    /// at `reduction`.
    #[enum_value(
        name = "Expander: downward expansion below the open threshold",
        nick = "expander"
    )]
    Expander = 1,
}

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
