//! `stromdsp-stereo`: stereo mode matrix, polarity, constant-power
//! balance and M/S width for two-channel audio.

use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    /// Stereo element (`stromdsp-stereo`).
    pub struct StromDspStereo(ObjectSubclass<imp::StromDspStereo>)
        @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

/// Values of the `mode` property (GLib enum `StromDspStereoMode`).
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy, Default, glib::Enum)]
#[repr(i32)]
#[enum_type(name = "StromDspStereoMode")]
pub enum StereoMode {
    /// Unchanged.
    #[default]
    #[enum_value(name = "Stereo: unchanged", nick = "stereo")]
    Stereo = 0,
    /// L+R on both channels, scaled by `mono-law`.
    #[enum_value(name = "Mono: (L+R) times mono-law on both channels", nick = "mono")]
    Mono = 1,
    /// Left input on both outputs.
    #[enum_value(name = "Left to both", nick = "left-to-both")]
    LeftToBoth = 2,
    /// Right input on both outputs.
    #[enum_value(name = "Right to both", nick = "right-to-both")]
    RightToBoth = 3,
    /// Left and right exchanged.
    #[enum_value(name = "Swap left and right", nick = "swap")]
    Swap = 4,
    /// L/R in, M = (L+R)/2 on the left and S = (L-R)/2 on the right out.
    #[enum_value(
        name = "M/S encode: M=(L+R)/2 left, S=(L-R)/2 right",
        nick = "ms-encode"
    )]
    MsEncode = 5,
    /// M (left) / S (right) in, L = M+S and R = M-S out.
    #[enum_value(name = "M/S decode: L=M+S, R=M-S", nick = "ms-decode")]
    MsDecode = 6,
}

/// Values of the `mono-law` property (GLib enum `StromDspStereoMonoLaw`).
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy, Default, glib::Enum)]
#[repr(i32)]
#[enum_type(name = "StromDspStereoMonoLaw")]
pub enum MonoLaw {
    /// Sum times 1/sqrt(2) (-3.01 dB).
    #[default]
    #[enum_value(name = "-3 dB: (L+R)/sqrt(2)", nick = "minus-3db")]
    Minus3Db = 0,
    /// Sum times 1/2 (-6.02 dB).
    #[enum_value(name = "-6 dB: (L+R)/2", nick = "minus-6db")]
    Minus6Db = 1,
}

impl From<StereoMode> for strom_dsp::stereo::StereoMode {
    fn from(m: StereoMode) -> Self {
        use strom_dsp::stereo::StereoMode as L;
        match m {
            StereoMode::Stereo => L::Stereo,
            StereoMode::Mono => L::Mono,
            StereoMode::LeftToBoth => L::LeftToBoth,
            StereoMode::RightToBoth => L::RightToBoth,
            StereoMode::Swap => L::Swap,
            StereoMode::MsEncode => L::MsEncode,
            StereoMode::MsDecode => L::MsDecode,
        }
    }
}

impl From<MonoLaw> for strom_dsp::stereo::MonoLaw {
    fn from(m: MonoLaw) -> Self {
        match m {
            MonoLaw::Minus3Db => strom_dsp::stereo::MonoLaw::Minus3Db,
            MonoLaw::Minus6Db => strom_dsp::stereo::MonoLaw::Minus6Db,
        }
    }
}

/// Register `stromdsp-stereo` with `plugin`.
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    StereoMode::static_type().mark_as_plugin_api(gst::PluginAPIFlags::empty());
    MonoLaw::static_type().mark_as_plugin_api(gst::PluginAPIFlags::empty());
    gst::Element::register(
        Some(plugin),
        "stromdsp-stereo",
        gst::Rank::NONE,
        StromDspStereo::static_type(),
    )
}
