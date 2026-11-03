//! `stromdsp-deesser` implementation.

use std::sync::LazyLock;

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::prelude::*;
use gst_base::subclass::prelude::*;
use strom_dsp::deesser::{
    Deesser, DeesserParams, ATTACK_RANGE_MS, FREQUENCY_RANGE_HZ, MAX_REDUCTION_RANGE_DB,
    RELEASE_RANGE_MS, THRESHOLD_RANGE_DB,
};

use super::DeesserMode;
use crate::common::{self, EngineSlot, SharedSettings};

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "stromdsp-deesser",
        gst::DebugColorFlags::empty(),
        Some("strom-dsp de-esser"),
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
    params: DeesserParams,
    mode: DeesserMode,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            enabled: true,
            params: DeesserParams::default(),
            mode: DeesserMode::SplitBand,
        }
    }
}

#[derive(Debug, Default)]
pub struct StromDspDeesser {
    settings: SharedSettings<Settings>,
    engine: EngineSlot<Deesser>,
}

fn apply(dsp: &mut Deesser, s: &Settings) {
    dsp.set_params(&DeesserParams {
        mode: s.mode.into(),
        ..s.params
    });
}

#[glib::object_subclass]
impl ObjectSubclass for StromDspDeesser {
    const NAME: &'static str = "StromDspDeesser";
    type Type = super::StromDspDeesser;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for StromDspDeesser {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPS: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            let d = DeesserParams::default();
            vec![
                common::bool_prop(
                    "enabled",
                    "Enabled",
                    "Process audio; when false the element crossfades to dry (20 ms), then is a \
                     bit-exact passthrough",
                    true,
                ),
                common::float_prop(
                    "frequency",
                    "Frequency",
                    "Sibilance band (Hz): detector high-pass corner and split frequency \
                     (limited to 0.45 x sample rate)",
                    FREQUENCY_RANGE_HZ.0,
                    FREQUENCY_RANGE_HZ.1,
                    d.frequency_hz,
                ),
                common::float_prop(
                    "threshold",
                    "Threshold",
                    "Detector peak level (dBFS) above which the band is reduced",
                    THRESHOLD_RANGE_DB.0,
                    THRESHOLD_RANGE_DB.1,
                    d.threshold_db,
                ),
                common::float_prop(
                    "max-reduction",
                    "Maximum reduction",
                    "Largest gain reduction (dB, positive = attenuation)",
                    MAX_REDUCTION_RANGE_DB.0,
                    MAX_REDUCTION_RANGE_DB.1,
                    d.max_reduction_db,
                ),
                common::float_prop(
                    "attack",
                    "Attack",
                    "Attack time constant (ms)",
                    ATTACK_RANGE_MS.0,
                    ATTACK_RANGE_MS.1,
                    d.attack_ms,
                ),
                common::float_prop(
                    "release",
                    "Release",
                    "Release time constant (ms)",
                    RELEASE_RANGE_MS.0,
                    RELEASE_RANGE_MS.1,
                    d.release_ms,
                ),
                glib::ParamSpecEnum::builder_with_default("mode", DeesserMode::SplitBand)
                    .nick("Mode")
                    .blurb("split-band: reduce only the band above frequency; wideband: reduce everything")
                    .readwrite()
                    .controllable()
                    .mutable_playing()
                    .build(),
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
            "mode" => {
                let mode: DeesserMode = value.get().expect("type checked upstream");
                self.settings.update(|s| s.mode = mode);
                gst::debug!(CAT, imp = self, "mode set to {mode:?}");
            }
            _ => {
                let v: f32 = value.get().expect("type checked upstream");
                self.settings.update(|s| {
                    let p = &mut s.params;
                    match name {
                        "frequency" => p.frequency_hz = v,
                        "threshold" => p.threshold_db = v,
                        "max-reduction" => p.max_reduction_db = v,
                        "attack" => p.attack_ms = v,
                        "release" => p.release_ms = v,
                        _ => unreachable!("unknown property {name}"),
                    }
                });
                gst::debug!(CAT, imp = self, "{name} set to {v}");
            }
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let s = self.settings.get();
        let p = &s.params;
        match pspec.name() {
            "enabled" => s.enabled.to_value(),
            "frequency" => p.frequency_hz.to_value(),
            "threshold" => p.threshold_db.to_value(),
            "max-reduction" => p.max_reduction_db.to_value(),
            "attack" => p.attack_ms.to_value(),
            "release" => p.release_ms.to_value(),
            "mode" => s.mode.to_value(),
            name => unreachable!("unknown property {name}"),
        }
    }
}

impl GstObjectImpl for StromDspDeesser {}

impl ElementImpl for StromDspDeesser {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static META: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "strom-dsp de-esser",
                "Filter/Effect/Audio",
                "De-esser with linked channels: a high-pass detector at `frequency` drives a \
                 gain reduction (up to max-reduction dB above threshold) applied to the band \
                 above `frequency` (split-band, LR4 crossover) or to the whole signal \
                 (wideband). No latency. Toggling enabled crossfades over 20 ms; enabled=false is then a bit-exact passthrough.",
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

impl BaseTransformImpl for StromDspDeesser {
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
