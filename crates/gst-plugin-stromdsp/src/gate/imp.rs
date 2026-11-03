//! `stromdsp-gate` implementation.

use std::sync::LazyLock;

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::prelude::*;
use gst_base::subclass::prelude::*;
use strom_dsp::gate::{GateMode, DEFAULT_EXPANDER_RATIO, EXPANDER_RATIO_RANGE};
use strom_dsp::{Gate, GateParams};

use super::StromDspGateMode;
use crate::common::{self, EngineSlot, SharedSettings};

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "stromdsp-gate",
        gst::DebugColorFlags::empty(),
        Some("strom-dsp noise gate"),
    )
});

#[derive(Debug, Clone, Copy)]
struct Settings {
    enabled: bool,
    params: GateParams,
    /// Phase 1b: gate or expander.
    mode: StromDspGateMode,
    /// Phase 1b: expander ratio (1 … 20).
    ratio: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            enabled: true,
            params: GateParams::default(),
            mode: StromDspGateMode::Gate,
            ratio: DEFAULT_EXPANDER_RATIO,
        }
    }
}

#[derive(Debug, Default)]
pub struct StromDspGate {
    settings: SharedSettings<Settings>,
    engine: EngineSlot<Gate>,
}

fn apply(dsp: &mut Gate, s: &Settings) {
    dsp.set_params(&s.params);
    dsp.set_mode(match s.mode {
        StromDspGateMode::Gate => GateMode::Gate,
        StromDspGateMode::Expander => GateMode::Expander,
    });
    dsp.set_expander_ratio(s.ratio);
}

#[glib::object_subclass]
impl ObjectSubclass for StromDspGate {
    const NAME: &'static str = "StromDspGate";
    type Type = super::StromDspGate;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for StromDspGate {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPS: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            let d = GateParams::default();
            vec![
                common::bool_prop(
                    "enabled",
                    "Enabled",
                    "Process audio; when false the element is a bit-exact passthrough",
                    true,
                ),
                common::float_prop(
                    "open-threshold",
                    "Open threshold",
                    "Level (dBFS) above which the gate opens",
                    -80.0,
                    0.0,
                    d.open_threshold_db,
                ),
                common::float_prop(
                    "close-threshold",
                    "Close threshold",
                    "Level (dBFS) below which the gate closes (treated as open-threshold if larger)",
                    -80.0,
                    0.0,
                    d.close_threshold_db,
                ),
                common::float_prop(
                    "attack",
                    "Attack",
                    "Opening time (ms)",
                    0.01,
                    200.0,
                    d.attack_ms,
                ),
                common::float_prop(
                    "hold",
                    "Hold",
                    "Time the gate stays open after the level falls below the close threshold (ms)",
                    0.0,
                    1000.0,
                    d.hold_ms,
                ),
                common::float_prop(
                    "release",
                    "Release",
                    "Closing time (ms)",
                    1.0,
                    2000.0,
                    d.release_ms,
                ),
                common::float_prop(
                    "reduction",
                    "Reduction",
                    "Attenuation when closed (dB)",
                    -80.0,
                    0.0,
                    d.reduction_db,
                ),
                // Phase 1b additions (STROM_DSP_PHASE1B_SPEC.md 3.3).
                glib::ParamSpecEnum::builder_with_default("mode", StromDspGateMode::Gate)
                    .nick("Mode")
                    .blurb(
                        "gate: fixed attenuation 'reduction' when closed; expander: \
                         downward expansion below open-threshold with slope 'ratio', \
                         floored at 'reduction'",
                    )
                    .readwrite()
                    .controllable()
                    .mutable_playing()
                    .build(),
                common::float_prop(
                    "ratio",
                    "Ratio",
                    "Expansion ratio below open-threshold in expander mode (2 = 1:2); \
                     ignored in gate mode",
                    EXPANDER_RATIO_RANGE.0,
                    EXPANDER_RATIO_RANGE.1,
                    DEFAULT_EXPANDER_RATIO,
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
        if name == "mode" {
            let mode: StromDspGateMode = value.get().expect("type checked upstream");
            self.settings.update(|s| s.mode = mode);
            gst::debug!(CAT, imp = self, "mode set to {mode:?}");
            return;
        }
        let v: f32 = value.get().expect("type checked upstream");
        self.settings.update(|s| {
            let p = &mut s.params;
            match name {
                "open-threshold" => p.open_threshold_db = v,
                "close-threshold" => p.close_threshold_db = v,
                "attack" => p.attack_ms = v,
                "hold" => p.hold_ms = v,
                "release" => p.release_ms = v,
                "reduction" => p.reduction_db = v,
                "ratio" => s.ratio = v,
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
            "open-threshold" => p.open_threshold_db.to_value(),
            "close-threshold" => p.close_threshold_db.to_value(),
            "attack" => p.attack_ms.to_value(),
            "hold" => p.hold_ms.to_value(),
            "release" => p.release_ms.to_value(),
            "reduction" => p.reduction_db.to_value(),
            "mode" => s.mode.to_value(),
            "ratio" => s.ratio.to_value(),
            name => unreachable!("unknown property {name}"),
        }
    }
}

impl GstObjectImpl for StromDspGate {}

impl ElementImpl for StromDspGate {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static META: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "strom-dsp noise gate",
                "Filter/Effect/Audio",
                "Downward noise gate with hysteresis, hold and linked channels, \
                 with an optional downward-expander mode (mode, ratio). \
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

impl BaseTransformImpl for StromDspGate {
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
