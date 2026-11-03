//! `stromdsp-compressor` implementation.

use std::sync::LazyLock;

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::prelude::*;
use gst_base::subclass::prelude::*;
use strom_dsp::{Compressor, CompressorParams};

use crate::common::{self, EngineSlot, SharedSettings};

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "stromdsp-compressor",
        gst::DebugColorFlags::empty(),
        Some("strom-dsp compressor"),
    )
});

#[derive(Debug, Clone, Copy)]
struct Settings {
    enabled: bool,
    params: CompressorParams,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            enabled: true,
            params: CompressorParams::default(),
        }
    }
}

#[derive(Debug, Default)]
pub struct StromDspCompressor {
    settings: SharedSettings<Settings>,
    engine: EngineSlot<Compressor>,
}

fn apply(dsp: &mut Compressor, s: &Settings) {
    dsp.set_params(&s.params);
}

#[glib::object_subclass]
impl ObjectSubclass for StromDspCompressor {
    const NAME: &'static str = "StromDspCompressor";
    type Type = super::StromDspCompressor;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for StromDspCompressor {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPS: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            let d = CompressorParams::default();
            vec![
                common::bool_prop(
                    "enabled",
                    "Enabled",
                    "Process audio; when false the element is a bit-exact passthrough",
                    true,
                ),
                common::float_prop(
                    "threshold",
                    "Threshold",
                    "Threshold as linear amplitude",
                    0.0,
                    1.0,
                    d.threshold,
                ),
                common::float_prop(
                    "ratio",
                    "Ratio",
                    "Compression ratio (4 = 4:1; 1 = no compression)",
                    1.0,
                    100.0,
                    d.ratio,
                ),
                common::float_prop(
                    "attack",
                    "Attack",
                    "Attack time (ms)",
                    0.01,
                    10000.0,
                    d.attack_ms,
                ),
                common::float_prop(
                    "release",
                    "Release",
                    "Release time (ms)",
                    0.01,
                    10000.0,
                    d.release_ms,
                ),
                common::float_prop(
                    "makeup-gain",
                    "Makeup gain",
                    "Makeup gain as linear multiplier",
                    0.0,
                    100.0,
                    d.makeup_gain,
                ),
                common::float_prop(
                    "knee",
                    "Knee",
                    "Soft-knee factor k (linear): the knee spans threshold*k to threshold/k, \
                     width -40*log10(k) dB; 1 = hard knee, k < 0.001 is treated as 0.001",
                    0.0,
                    1.0,
                    d.knee,
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
        if name == "enabled" {
            let enabled: bool = value.get().expect("type checked upstream");
            let was = self
                .settings
                .update(|s| std::mem::replace(&mut s.enabled, enabled));
            if enabled && !was {
                self.settings.request_reset();
            }
            self.obj().set_passthrough(!enabled);
            gst::debug!(CAT, imp = self, "enabled: {was} -> {enabled}");
            return;
        }
        let v: f32 = value.get().expect("type checked upstream");
        self.settings.update(|s| {
            let p = &mut s.params;
            match name {
                "threshold" => p.threshold = v,
                "ratio" => p.ratio = v,
                "attack" => p.attack_ms = v,
                "release" => p.release_ms = v,
                "makeup-gain" => p.makeup_gain = v,
                // Stored verbatim so it reads back exactly as written.
                "knee" => p.knee = v,
                _ => unreachable!("unknown property {name}"),
            }
        });
        gst::debug!(CAT, imp = self, "{name} set to {v}");
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let s = self.settings.get();
        let p = &s.params;
        match pspec.name() {
            "enabled" => s.enabled.to_value(),
            "threshold" => p.threshold.to_value(),
            "ratio" => p.ratio.to_value(),
            "attack" => p.attack_ms.to_value(),
            "release" => p.release_ms.to_value(),
            "makeup-gain" => p.makeup_gain.to_value(),
            "knee" => p.knee.to_value(),
            name => unreachable!("unknown property {name}"),
        }
    }
}

impl GstObjectImpl for StromDspCompressor {}

impl ElementImpl for StromDspCompressor {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static META: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "strom-dsp compressor",
                "Filter/Effect/Audio",
                "Feed-forward downward compressor with soft knee and linked channels. \
                 enabled=false is a bit-exact passthrough.",
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

impl BaseTransformImpl for StromDspCompressor {
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
        gst::debug!(
            CAT,
            imp = self,
            "configuring for {rate} Hz, {channels} channels"
        );
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
