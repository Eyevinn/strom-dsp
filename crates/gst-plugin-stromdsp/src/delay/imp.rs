//! `stromdsp-delay` implementation.
//!
//! - The delay is intentional (content alignment), so it is **not** latency:
//!   the LATENCY query is left to BaseTransform (upstream's values unchanged)
//!   and timestamps pass through unchanged.
//! - Never a BaseTransform passthrough: `enabled=false` is handed to the
//!   library, which crossfades every channel to delay 0 (20 ms) and is then a
//!   bit-exact copy, while it keeps recording history so re-enabling fades
//!   back without a gap. Toggling is therefore click-free.
//! - `channel-delays` is a `GstValueArray` of per-channel offsets in ms
//!   (−2000 … 2000), added to `delay`; the total per channel is clamped to
//!   0 … 2000 ms. Missing entries are 0, entries beyond the channel count (or
//!   beyond 64) are ignored. The array
//!   is validated by GLib against its float element spec, so elements must be
//!   floats in range (gst-launch deserialises `<1, 2.5>` with that spec, so
//!   untyped numbers work there). The setter still converts defensively.

use std::sync::LazyLock;

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::subclass::prelude::*;
use strom_dsp::delay::{Delay, DelayParams, MAX_DELAY_MS, MAX_OFFSET_CHANNELS};

use crate::common::{self, EngineSlot, SharedSettings};

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "stromdsp-delay",
        gst::DebugColorFlags::empty(),
        Some("strom-dsp intentional delay"),
    )
});

#[derive(Debug, Default)]
pub struct StromDspDelay {
    settings: SharedSettings<DelayParams>,
    engine: EngineSlot<Delay>,
}

fn apply(dsp: &mut Delay, p: &DelayParams) {
    dsp.set_params(p);
}

/// One `channel-delays` entry as ms (defensive: GLib validation already
/// guarantees an in-range float), clamped to ±2000; `None` for an
/// unsupported value type.
fn entry_ms(v: &glib::SendValue) -> Option<f32> {
    let ms = if let Ok(f) = v.get::<f32>() {
        f
    } else if let Ok(d) = v.get::<f64>() {
        d as f32
    } else if let Ok(i) = v.get::<i32>() {
        i as f32
    } else {
        return None;
    };
    Some(if ms.is_nan() {
        0.0
    } else {
        ms.clamp(-MAX_DELAY_MS, MAX_DELAY_MS)
    })
}

#[glib::object_subclass]
impl ObjectSubclass for StromDspDelay {
    const NAME: &'static str = "StromDspDelay";
    type Type = super::StromDspDelay;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for StromDspDelay {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPS: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            let d = DelayParams::default();
            let entry = glib::ParamSpecFloat::builder("channel-delay")
                .nick("Channel delay")
                .blurb("Offset of one channel (ms), added to delay")
                .minimum(-MAX_DELAY_MS)
                .maximum(MAX_DELAY_MS)
                .default_value(0.0)
                .readwrite()
                .build();
            vec![
                common::bool_prop(
                    "enabled",
                    "Enabled",
                    "Delay; when false every channel crossfades to delay 0 (then bit-exact \
                     passthrough), and back when re-enabled",
                    d.enabled,
                ),
                common::float_prop(
                    "delay",
                    "Delay",
                    "Delay for all channels (ms). Intentional: NOT reported as latency",
                    0.0,
                    MAX_DELAY_MS,
                    d.delay_ms,
                ),
                gst::ParamSpecArray::builder("channel-delays")
                    .nick("Channel delays")
                    .blurb(
                        "Per-channel offsets (ms, -2000 .. 2000) added to delay, as an array \
                         of floats, e.g. <0.0, 2.5>. Missing entries are 0, extra entries are \
                         ignored; each channel's total is clamped to 0 .. 2000 ms",
                    )
                    .element_spec(&entry)
                    .readwrite()
                    .mutable_playing()
                    .build(),
            ]
        });
        PROPS.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        match pspec.name() {
            "enabled" => {
                let enabled: bool = value.get().expect("type checked upstream");
                self.settings.update(|p| p.enabled = enabled);
                gst::debug!(CAT, imp = self, "enabled set to {enabled}");
            }
            "delay" => {
                let v: f32 = value.get().expect("type checked upstream");
                self.settings.update(|p| p.delay_ms = v);
                gst::debug!(CAT, imp = self, "delay set to {v} ms");
            }
            "channel-delays" => {
                let arr: gst::Array = value.get().expect("type checked upstream");
                let mut offs = [0.0f32; MAX_OFFSET_CHANNELS];
                if arr.len() > MAX_OFFSET_CHANNELS {
                    gst::warning!(
                        CAT,
                        imp = self,
                        "channel-delays: only the first {MAX_OFFSET_CHANNELS} entries are used"
                    );
                }
                let n = arr.len().min(MAX_OFFSET_CHANNELS);
                for (o, v) in offs.iter_mut().zip(arr.iter()) {
                    *o = entry_ms(v).unwrap_or_else(|| {
                        gst::warning!(
                            CAT,
                            imp = self,
                            "channel-delays: entry of type {} is not a number, using 0",
                            v.type_()
                        );
                        0.0
                    });
                }
                self.settings.update(|p| p.set_offsets(&offs[..n]));
                gst::debug!(CAT, imp = self, "channel-delays set to {:?}", &offs[..n]);
            }
            name => unreachable!("unknown property {name}"),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let p = self.settings.get();
        match pspec.name() {
            "enabled" => p.enabled.to_value(),
            "delay" => p.delay_ms.to_value(),
            "channel-delays" => gst::Array::new(p.offsets().iter().copied()).to_value(),
            name => unreachable!("unknown property {name}"),
        }
    }
}

impl GstObjectImpl for StromDspDelay {}

impl ElementImpl for StromDspDelay {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static META: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "strom-dsp delay",
                "Filter/Effect/Audio",
                "Intentional delay for content alignment (mic alignment, lip sync): delay \
                 plus per-channel channel-delays, up to 2000 ms. The delay is NOT latency: \
                 it is not reported in the LATENCY query and timestamps are unchanged, so \
                 the audio content moves later relative to its timestamps. Changes crossfade \
                 over 20 ms. enabled=false crossfades to delay 0 and is then a bit-exact \
                 passthrough.",
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

impl BaseTransformImpl for StromDspDelay {
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
            "configuring for {rate} Hz, {channels} channels (allocating {MAX_DELAY_MS} ms)"
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
