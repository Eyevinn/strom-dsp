//! `stromdsp-cleanup` implementation.

use std::sync::LazyLock;

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::prelude::*;
use gst_base::subclass::prelude::*;
use strom_dsp::cleanup::{
    Cleanup, CleanupParams, HIGHPASS_RANGE_HZ, HUM_Q_RANGE, MAX_HUM_HARMONICS,
};
use strom_dsp::filter::MAX_ORDER;

use super::CleanupHum;
use crate::common::{self, EngineSlot, SharedSettings};

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "stromdsp-cleanup",
        gst::DebugColorFlags::empty(),
        Some("strom-dsp input cleanup"),
    )
});

impl common::Bypassable for Settings {
    fn enabled(&self) -> bool {
        self.enabled
    }
}

#[derive(Debug, Clone, Copy)]
struct Settings {
    enabled: bool,
    params: CleanupParams,
    hum: CleanupHum,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            enabled: true,
            params: CleanupParams::default(),
            hum: CleanupHum::Off,
        }
    }
}

#[derive(Debug, Default)]
pub struct StromDspCleanup {
    settings: SharedSettings<Settings>,
    engine: EngineSlot<Cleanup>,
}

fn apply(dsp: &mut Cleanup, s: &Settings) {
    dsp.set_params(&CleanupParams {
        hum: s.hum.into(),
        ..s.params
    });
}

/// Readable, writable, controllable, mutable-in-PLAYING `u32` property.
fn uint_prop(
    name: &str,
    nick: &str,
    blurb: &str,
    min: u32,
    max: u32,
    default: u32,
) -> glib::ParamSpec {
    glib::ParamSpecUInt::builder(name)
        .nick(nick)
        .blurb(blurb)
        .minimum(min)
        .maximum(max)
        .default_value(default)
        .readwrite()
        .controllable()
        .mutable_playing()
        .build()
}

#[glib::object_subclass]
impl ObjectSubclass for StromDspCleanup {
    const NAME: &'static str = "StromDspCleanup";
    type Type = super::StromDspCleanup;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for StromDspCleanup {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPS: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            let d = CleanupParams::default();
            vec![
                common::bool_prop(
                    "enabled",
                    "Enabled",
                    "Process audio; when false the element crossfades to dry (20 ms), then is a \
                     bit-exact passthrough",
                    true,
                ),
                common::bool_prop(
                    "dc-block",
                    "DC block",
                    "Remove DC with a first-order high-pass at 5 Hz (-3 dB)",
                    d.dc_block,
                ),
                common::float_prop(
                    "highpass",
                    "High-pass",
                    "Butterworth high-pass cutoff (Hz, -3 dB); 0 = off",
                    HIGHPASS_RANGE_HZ.0,
                    HIGHPASS_RANGE_HZ.1,
                    d.highpass_hz,
                ),
                uint_prop(
                    "highpass-order",
                    "High-pass order",
                    "Butterworth order of the high-pass (6 dB/octave per order)",
                    1,
                    MAX_ORDER as u32,
                    d.highpass_order as u32,
                ),
                glib::ParamSpecEnum::builder_with_default("hum", CleanupHum::Off)
                    .nick("Hum")
                    .blurb("Mains hum fundamental to notch out (off, 50 or 60 Hz)")
                    .readwrite()
                    .controllable()
                    .mutable_playing()
                    .build(),
                uint_prop(
                    "hum-harmonics",
                    "Hum harmonics",
                    "Number of hum notches: the fundamental and harmonics up to this number",
                    1,
                    MAX_HUM_HARMONICS as u32,
                    d.hum_harmonics as u32,
                ),
                common::float_prop(
                    "hum-q",
                    "Hum Q",
                    "Q of the fundamental notch; every notch has the same bandwidth \
                     (fundamental / hum-q Hz)",
                    HUM_Q_RANGE.0,
                    HUM_Q_RANGE.1,
                    d.hum_q,
                ),
            ]
        });
        PROPS.as_ref()
    }

    fn constructed(&self) {
        self.parent_constructed();
        self.obj().set_passthrough(!self.settings.get().enabled);
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let name = pspec.name();
        match name {
            "enabled" => {
                let enabled: bool = value.get().expect("type checked upstream");
                let was = self
                    .settings
                    .update(|s| std::mem::replace(&mut s.enabled, enabled));
                common::set_enabled_idle(&*self.obj(), &self.engine, enabled);
                gst::debug!(CAT, imp = self, "enabled: {was} -> {enabled}");
            }
            "dc-block" => {
                let v: bool = value.get().expect("type checked upstream");
                self.settings.update(|s| s.params.dc_block = v);
                gst::debug!(CAT, imp = self, "dc-block set to {v}");
            }
            "highpass" | "hum-q" => {
                let v: f32 = value.get().expect("type checked upstream");
                self.settings.update(|s| match name {
                    "highpass" => s.params.highpass_hz = v,
                    _ => s.params.hum_q = v,
                });
                gst::debug!(CAT, imp = self, "{name} set to {v}");
            }
            "highpass-order" | "hum-harmonics" => {
                let v: u32 = value.get().expect("type checked upstream");
                self.settings.update(|s| match name {
                    "highpass-order" => s.params.highpass_order = v as usize,
                    _ => s.params.hum_harmonics = v as usize,
                });
                gst::debug!(CAT, imp = self, "{name} set to {v}");
            }
            "hum" => {
                let hum: CleanupHum = value.get().expect("type checked upstream");
                self.settings.update(|s| s.hum = hum);
                gst::debug!(CAT, imp = self, "hum set to {hum:?}");
            }
            _ => unreachable!("unknown property {name}"),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let s = self.settings.get();
        let p = &s.params;
        match pspec.name() {
            "enabled" => s.enabled.to_value(),
            "dc-block" => p.dc_block.to_value(),
            "highpass" => p.highpass_hz.to_value(),
            "highpass-order" => (p.highpass_order as u32).to_value(),
            "hum" => s.hum.to_value(),
            "hum-harmonics" => (p.hum_harmonics as u32).to_value(),
            "hum-q" => p.hum_q.to_value(),
            name => unreachable!("unknown property {name}"),
        }
    }
}

impl GstObjectImpl for StromDspCleanup {}

impl ElementImpl for StromDspCleanup {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static META: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "strom-dsp input cleanup",
                "Filter/Effect/Audio",
                "Input cleanup: DC blocker (5 Hz), Butterworth high-pass (order 1-8) and \
                 mains-hum notches (50/60 Hz plus harmonics, constant bandwidth). Each stage \
                 is optional and off by default (bit-exact passthrough); switching crossfades \
                 over 10 ms. No latency. Toggling enabled crossfades over 20 ms; enabled=false is then a bit-exact passthrough.",
                "Eyevinn Technology",
            )
        });
        Some(&*META)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static T: LazyLock<Vec<gst::PadTemplate>> = LazyLock::new(common::pad_templates);
        T.as_ref()
    }
}

impl BaseTransformImpl for StromDspCleanup {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = true;

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        *self.engine.lock().unwrap() = None;
        gst::debug!(CAT, imp = self, "stopped");
        Ok(())
    }

    fn set_caps(&self, incaps: &gst::Caps, outcaps: &gst::Caps) -> Result<(), gst::LoggableError> {
        let (rate, channels) = common::parse_caps(incaps)?;
        gst::debug!(
            CAT,
            imp = self,
            "configuring for {rate} Hz, {channels} channels"
        );
        common::configure_engine_bypass(
            &*self.obj(),
            &self.engine,
            &self.settings,
            rate,
            channels,
            apply,
        );
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

    fn transform_ip_passthrough(
        &self,
        buf: &gst::Buffer,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        common::transform_passthrough_bypass(&*self.obj(), &self.engine, &self.settings, buf, apply)
    }

    fn transform_ip(&self, buf: &mut gst::BufferRef) -> Result<gst::FlowSuccess, gst::FlowError> {
        common::transform_buffer_bypass(&*self.obj(), &self.engine, &self.settings, buf, apply)
    }
}
