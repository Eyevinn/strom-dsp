//! `stromdsp-stereo` implementation.
//!
//! - Caps: interleaved F32LE, any rate, **exactly 2 channels**.
//! - `mode` and `mono-law` are GLib enums ([`super::StereoMode`],
//!   [`super::MonoLaw`]); the other properties are booleans and floats.
//! - Never a BaseTransform passthrough: every change, including
//!   `enabled=false`, ramps the library's 2×2 matrix over 10 ms (a linear
//!   crossfade between old and new processing), so nothing clicks. At the
//!   defaults, and when disabled once the ramp is over, the matrix is the
//!   identity and the buffer is left untouched (bit-exact).
//! - No latency.

use std::sync::LazyLock;

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::subclass::prelude::*;
use strom_dsp::stereo::{Stereo, StereoParams};

use super::{MonoLaw, StereoMode};
use crate::common::{self, EngineSlot, SharedSettings};

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "stromdsp-stereo",
        gst::DebugColorFlags::empty(),
        Some("strom-dsp stereo"),
    )
});

/// Property values as set (enums kept as the GLib types for read-back).
#[derive(Debug, Clone, Copy)]
struct Settings {
    enabled: bool,
    mode: StereoMode,
    mono_law: MonoLaw,
    invert_left: bool,
    invert_right: bool,
    balance: f32,
    width: f32,
}

impl Default for Settings {
    fn default() -> Self {
        let d = StereoParams::default();
        Self {
            enabled: d.enabled,
            mode: StereoMode::default(),
            mono_law: MonoLaw::default(),
            invert_left: d.invert_left,
            invert_right: d.invert_right,
            balance: d.balance,
            width: d.width,
        }
    }
}

impl Settings {
    fn params(&self) -> StereoParams {
        StereoParams {
            enabled: self.enabled,
            mode: self.mode.into(),
            mono_law: self.mono_law.into(),
            invert_left: self.invert_left,
            invert_right: self.invert_right,
            balance: self.balance,
            width: self.width,
        }
    }
}

#[derive(Debug, Default)]
pub struct StromDspStereo {
    settings: SharedSettings<Settings>,
    engine: EngineSlot<Stereo>,
}

fn apply(dsp: &mut Stereo, s: &Settings) {
    dsp.set_params(&s.params());
}

/// Readable, writable, controllable, mutable-in-PLAYING GLib enum property
/// whose default is `E::default()`.
fn enum_prop<E>(name: &str, nick: &str, blurb: &str) -> glib::ParamSpec
where
    E: StaticType
        + Default
        + glib::translate::FromGlib<i32>
        + glib::translate::IntoGlib<GlibType = i32>,
{
    glib::ParamSpecEnum::builder_with_default(name, E::default())
        .nick(nick)
        .blurb(blurb)
        .readwrite()
        .controllable()
        .mutable_playing()
        .build()
}

#[glib::object_subclass]
impl ObjectSubclass for StromDspStereo {
    const NAME: &'static str = "StromDspStereo";
    type Type = super::StromDspStereo;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for StromDspStereo {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPS: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            let d = StereoParams::default();
            vec![
                common::bool_prop(
                    "enabled",
                    "Enabled",
                    "Process; when false the matrix ramps to the identity (then bit-exact \
                     passthrough)",
                    d.enabled,
                ),
                enum_prop::<StereoMode>(
                    "mode",
                    "Mode",
                    "Channel routing: stereo, mono, left-to-both, right-to-both, swap, \
                     ms-encode, ms-decode",
                ),
                enum_prop::<MonoLaw>(
                    "mono-law",
                    "Mono law",
                    "Gain of the L+R sum in mono mode: minus-3db (1/sqrt 2) or minus-6db (1/2)",
                ),
                common::bool_prop(
                    "invert-left",
                    "Invert left",
                    "Invert the polarity of the left input (before the mode matrix)",
                    d.invert_left,
                ),
                common::bool_prop(
                    "invert-right",
                    "Invert right",
                    "Invert the polarity of the right input (before the mode matrix)",
                    d.invert_right,
                ),
                common::float_prop(
                    "balance",
                    "Balance",
                    "Balance, -1 (left) .. 0 .. 1 (right), constant power: \
                     gL = sqrt2 cos t, gR = sqrt2 sin t, t = (b+1) pi/4 (unity at 0)",
                    -1.0,
                    1.0,
                    d.balance,
                ),
                common::float_prop(
                    "width",
                    "Width",
                    "Stereo width through M/S: S' = width * S; 0 = mono, 1 = unchanged, 2 = wide",
                    0.0,
                    2.0,
                    d.width,
                ),
            ]
        });
        PROPS.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let name = pspec.name();
        self.settings.update(|s| match name {
            "enabled" => s.enabled = value.get().expect("type checked upstream"),
            "mode" => s.mode = value.get().expect("type checked upstream"),
            "mono-law" => s.mono_law = value.get().expect("type checked upstream"),
            "invert-left" => s.invert_left = value.get().expect("type checked upstream"),
            "invert-right" => s.invert_right = value.get().expect("type checked upstream"),
            "balance" => s.balance = value.get().expect("type checked upstream"),
            "width" => s.width = value.get().expect("type checked upstream"),
            _ => unreachable!("unknown property {name}"),
        });
        gst::debug!(CAT, imp = self, "{name} set to {value:?}");
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let s = self.settings.get();
        match pspec.name() {
            "enabled" => s.enabled.to_value(),
            "mode" => s.mode.to_value(),
            "mono-law" => s.mono_law.to_value(),
            "invert-left" => s.invert_left.to_value(),
            "invert-right" => s.invert_right.to_value(),
            "balance" => s.balance.to_value(),
            "width" => s.width.to_value(),
            name => unreachable!("unknown property {name}"),
        }
    }
}

impl GstObjectImpl for StromDspStereo {}

impl ElementImpl for StromDspStereo {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static META: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "strom-dsp stereo",
                "Filter/Effect/Audio",
                "Stereo utility for 2 channels: mode (stereo, mono with -3/-6 dB law, \
                 left/right to both, swap, M/S encode/decode), input polarity, \
                 constant-power balance and M/S width. Changes ramp over 10 ms (click-free). \
                 The defaults, and enabled=false once ramped, are a bit-exact passthrough.",
                "Eyevinn Technology",
            )
        });
        Some(&*META)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static T: LazyLock<Vec<gst::PadTemplate>> = LazyLock::new(|| {
            let caps = gst_audio::AudioCapsBuilder::new_interleaved()
                .format(gst_audio::AudioFormat::F32le)
                .rate_range(1..=i32::MAX)
                .channels(2)
                .build();
            vec![
                gst::PadTemplate::new(
                    "src",
                    gst::PadDirection::Src,
                    gst::PadPresence::Always,
                    &caps,
                )
                .unwrap(),
                gst::PadTemplate::new(
                    "sink",
                    gst::PadDirection::Sink,
                    gst::PadPresence::Always,
                    &caps,
                )
                .unwrap(),
            ]
        });
        T.as_ref()
    }
}

impl BaseTransformImpl for StromDspStereo {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        *self.engine.lock().unwrap() = None;
        gst::debug!(CAT, imp = self, "stopped");
        Ok(())
    }

    fn set_caps(&self, incaps: &gst::Caps, outcaps: &gst::Caps) -> Result<(), gst::LoggableError> {
        let (rate, channels) = common::parse_caps(incaps)?;
        if channels != 2 {
            return Err(gst::loggable_error!(
                CAT,
                "the stereo element needs 2 channels, got {channels}"
            ));
        }
        gst::debug!(CAT, imp = self, "configuring for {rate} Hz");
        common::configure_engine(&self.engine, &self.settings, rate, channels, apply);
        self.parent_set_caps(incaps, outcaps)
    }

    fn sink_event(&self, event: gst::Event) -> bool {
        if let gst::EventView::FlushStop(_) = event.view() {
            gst::debug!(CAT, imp = self, "flush-stop: resetting state");
            common::reset_engine(&self.engine);
        }
        self.parent_sink_event(event)
    }

    fn before_transform(&self, inbuf: &gst::BufferRef) {
        common::sync_controlled_values(&*self.obj(), inbuf);
    }

    fn transform_ip(&self, buf: &mut gst::BufferRef) -> Result<gst::FlowSuccess, gst::FlowError> {
        common::transform_buffer(&self.engine, &self.settings, buf, apply)
    }
}
