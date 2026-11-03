//! `stromdsp-equalizer` implementation.

use std::sync::LazyLock;

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::prelude::*;
use gst_base::subclass::prelude::*;
use strom_dsp::{BandParams, BandType, Equalizer, EqualizerParams, MAX_BANDS};

use crate::common::{self, EngineSlot, SharedSettings};

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "stromdsp-equalizer",
        gst::DebugColorFlags::empty(),
        Some("strom-dsp parametric equalizer"),
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
    params: EqualizerParams,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            enabled: true,
            params: EqualizerParams::default(),
        }
    }
}

#[derive(Debug, Default)]
pub struct StromDspEqualizer {
    settings: SharedSettings<Settings>,
    engine: EngineSlot<Equalizer>,
}

fn apply(dsp: &mut Equalizer, s: &Settings) {
    dsp.set_params(&s.params);
}

#[glib::object_subclass]
impl ObjectSubclass for StromDspEqualizer {
    const NAME: &'static str = "StromDspEqualizer";
    type Type = super::StromDspEqualizer;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for StromDspEqualizer {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPS: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            let d = BandParams::default();
            let mut v = vec![
                common::bool_prop(
                    "enabled",
                    "Enabled",
                    "Process audio; when false the element crossfades to dry (20 ms), then is a \
                     bit-exact passthrough",
                    true,
                ),
                glib::ParamSpecUInt::builder("num-bands")
                    .nick("Number of bands")
                    .blurb("Number of active bands; bands at index >= num-bands are inactive")
                    .minimum(1)
                    .maximum(MAX_BANDS as u32)
                    .default_value(EqualizerParams::default().num_bands as u32)
                    .readwrite()
                    .controllable()
                    .mutable_ready()
                    .build(),
            ];
            for n in 0..MAX_BANDS {
                v.push(common::bool_prop(
                    &format!("band{n}-enabled"),
                    &format!("Band {n} enabled"),
                    &format!("Enable band {n}"),
                    d.enabled,
                ));
                v.push(
                    glib::ParamSpecInt::builder(&format!("band{n}-type"))
                        .nick(&format!("Band {n} type"))
                        .blurb(&format!(
                            "Filter type of band {n}: 0 off, 1 low-pass, 2 high-pass, \
                             3 band-pass (constant skirt gain), 4 band-pass (0 dB peak), \
                             5 notch, 6 all-pass, 7 peaking, 8 low shelf, 9 high shelf"
                        ))
                        .minimum(0)
                        .maximum(9)
                        .default_value(d.band_type as i32)
                        .readwrite()
                        .controllable()
                        .mutable_playing()
                        .build(),
                );
                v.push(common::float_prop(
                    &format!("band{n}-frequency"),
                    &format!("Band {n} frequency"),
                    &format!("Centre/corner frequency of band {n} (Hz)"),
                    1.0,
                    24000.0,
                    d.frequency,
                ));
                v.push(common::float_prop(
                    &format!("band{n}-gain"),
                    &format!("Band {n} gain"),
                    &format!("Gain of band {n} (dB; peaking and shelves only)"),
                    -60.0,
                    60.0,
                    d.gain_db,
                ));
                v.push(common::float_prop(
                    &format!("band{n}-q"),
                    &format!("Band {n} Q"),
                    &format!("Q of band {n} (RBJ cookbook Q, also for shelves)"),
                    0.01,
                    100.0,
                    d.q,
                ));
            }
            v
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
            "num-bands" => {
                let n: u32 = value.get().expect("type checked upstream");
                self.settings.update(|s| s.params.num_bands = n as usize);
                gst::debug!(CAT, imp = self, "num-bands set to {n}");
            }
            _ => {
                let (band, field) = parse_band_property(name).expect("unknown property");
                self.settings.update(|s| {
                    let b = &mut s.params.bands[band];
                    match field {
                        "enabled" => b.enabled = value.get().expect("type checked upstream"),
                        "type" => {
                            b.band_type =
                                BandType::from_i32(value.get().expect("type checked upstream"))
                        }
                        "frequency" => b.frequency = value.get().expect("type checked upstream"),
                        "gain" => b.gain_db = value.get().expect("type checked upstream"),
                        "q" => b.q = value.get().expect("type checked upstream"),
                        _ => unreachable!("unknown property {name}"),
                    }
                });
                gst::debug!(CAT, imp = self, "{name} set to {value:?}");
            }
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let s = self.settings.get();
        match pspec.name() {
            "enabled" => s.enabled.to_value(),
            "num-bands" => (s.params.num_bands as u32).to_value(),
            name => {
                let (band, field) = parse_band_property(name).expect("unknown property");
                let b = &s.params.bands[band];
                match field {
                    "enabled" => b.enabled.to_value(),
                    "type" => (b.band_type as i32).to_value(),
                    "frequency" => b.frequency.to_value(),
                    "gain" => b.gain_db.to_value(),
                    "q" => b.q.to_value(),
                    _ => unreachable!("unknown property {name}"),
                }
            }
        }
    }
}

/// Split `bandN-field` into `(N, field)`.
fn parse_band_property(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("band")?;
    let (idx, field) = rest.split_once('-')?;
    let idx: usize = idx.parse().ok()?;
    (idx < MAX_BANDS).then_some((idx, field))
}

impl GstObjectImpl for StromDspEqualizer {}

impl ElementImpl for StromDspEqualizer {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static META: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "strom-dsp parametric equalizer",
                "Filter/Effect/Audio",
                "Parametric equalizer with up to 8 RBJ-cookbook biquad bands in series. \
                 Toggling enabled crossfades over 20 ms; enabled=false is then a bit-exact passthrough.",
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

impl BaseTransformImpl for StromDspEqualizer {
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
