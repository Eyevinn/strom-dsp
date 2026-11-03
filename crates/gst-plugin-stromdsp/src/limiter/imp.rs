//! `stromdsp-limiter` implementation.
//!
//! Unlike the other elements the limiter never uses BaseTransform passthrough:
//! `enabled=false` is handed to the library, which then applies unity gain but
//! keeps the lookahead delay, so the latency does not jump when toggled.
//!
//! Latency: the LATENCY query is answered by adding the lookahead to upstream's
//! min and max. The lookahead is converted to frames by the library's own
//! `strom_dsp::limiter::lookahead_frames` and then to time, rounded up, so
//! the reported latency is never smaller than the real delay. Before caps are
//! known it is reported from the millisecond value directly. Timestamps pass
//! through unchanged.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::LazyLock;

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::prelude::*;
use gst_base::subclass::prelude::*;
use strom_dsp::{Limiter, LimiterParams};

use crate::common::{self, EngineSlot, SharedSettings};

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "stromdsp-limiter",
        gst::DebugColorFlags::empty(),
        Some("strom-dsp lookahead limiter"),
    )
});

/// The element's settings: the library parameters, starting from the
/// broadcast preset (`strom_dsp::presets`) instead of `LimiterParams::default()`.
#[derive(Debug, Clone, Copy)]
struct Settings(LimiterParams);

impl Default for Settings {
    fn default() -> Self {
        Self(LimiterParams::broadcast())
    }
}

impl std::ops::Deref for Settings {
    type Target = LimiterParams;
    fn deref(&self) -> &LimiterParams {
        &self.0
    }
}

impl std::ops::DerefMut for Settings {
    fn deref_mut(&mut self) -> &mut LimiterParams {
        &mut self.0
    }
}

#[derive(Debug, Default)]
pub struct StromDspLimiter {
    settings: SharedSettings<Settings>,
    engine: EngineSlot<Limiter>,
    /// Negotiated sample rate (0 = not negotiated), for latency reporting
    /// without touching the DSP lock.
    rate: AtomicU32,
}

fn apply(dsp: &mut Limiter, s: &Settings) {
    dsp.set_params(s);
}

/// Lookahead delay as time. With a known `rate` the frame count comes from the
/// library (`lookahead_frames`), then it is rounded up to whole nanoseconds.
pub(crate) fn lookahead_latency(lookahead_ms: f32, rate: u32) -> gst::ClockTime {
    let ms = f64::from(lookahead_ms.max(0.0));
    if rate == 0 {
        return gst::ClockTime::from_nseconds((ms * 1_000_000.0).ceil() as u64);
    }
    let frames = strom_dsp::limiter::lookahead_frames(lookahead_ms, rate as f32) as u64;
    gst::ClockTime::from_nseconds(
        (frames * gst::ClockTime::SECOND.nseconds()).div_ceil(u64::from(rate)),
    )
}

impl StromDspLimiter {
    fn latency(&self) -> gst::ClockTime {
        lookahead_latency(
            self.settings.get().lookahead_ms,
            self.rate.load(Ordering::Acquire),
        )
    }
}

#[glib::object_subclass]
impl ObjectSubclass for StromDspLimiter {
    const NAME: &'static str = "StromDspLimiter";
    type Type = super::StromDspLimiter;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for StromDspLimiter {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPS: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            let d = LimiterParams::broadcast();
            vec![
                common::bool_prop(
                    "enabled",
                    "Enabled",
                    "Limit; when false the gain is unity but the lookahead delay (latency) is kept",
                    d.enabled,
                ),
                common::float_prop(
                    "threshold",
                    "Threshold",
                    "Ceiling (dBFS); no output sample exceeds it",
                    -60.0,
                    0.0,
                    d.threshold_db,
                ),
                common::float_prop(
                    "attack",
                    "Attack",
                    "Attack time (ms)",
                    0.01,
                    50.0,
                    d.attack_ms,
                ),
                common::float_prop(
                    "release",
                    "Release",
                    "Release time (ms)",
                    1.0,
                    2000.0,
                    d.release_ms,
                ),
                common::float_prop(
                    "lookahead",
                    "Lookahead",
                    "Lookahead (ms); the audio is delayed by this much and it is reported as latency",
                    0.0,
                    20.0,
                    d.lookahead_ms,
                ),
            ]
        });
        PROPS.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let name = pspec.name();
        if name == "enabled" {
            let enabled: bool = value.get().expect("type checked upstream");
            self.settings.update(|s| s.enabled = enabled);
            gst::debug!(CAT, imp = self, "enabled set to {enabled}");
            return;
        }
        let v: f32 = value.get().expect("type checked upstream");
        let old_latency = self.latency();
        self.settings.update(|p| match name {
            "threshold" => p.threshold_db = v,
            "attack" => p.attack_ms = v,
            "release" => p.release_ms = v,
            "lookahead" => p.lookahead_ms = v,
            _ => unreachable!("unknown property {name}"),
        });
        gst::debug!(CAT, imp = self, "{name} set to {v}");

        if name == "lookahead" {
            let new_latency = self.latency();
            let obj = self.obj();
            if new_latency != old_latency && obj.current_state() >= gst::State::Paused {
                gst::debug!(
                    CAT,
                    imp = self,
                    "latency changed {old_latency} -> {new_latency}, posting latency message"
                );
                let _ = obj.post_message(gst::message::Latency::builder().src(&*obj).build());
            }
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let p = self.settings.get();
        match pspec.name() {
            "enabled" => p.enabled.to_value(),
            "threshold" => p.threshold_db.to_value(),
            "attack" => p.attack_ms.to_value(),
            "release" => p.release_ms.to_value(),
            "lookahead" => p.lookahead_ms.to_value(),
            name => unreachable!("unknown property {name}"),
        }
    }
}

impl GstObjectImpl for StromDspLimiter {}

impl ElementImpl for StromDspLimiter {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static META: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "strom-dsp limiter",
                "Filter/Effect/Audio",
                "Lookahead sample-peak limiter with linked channels. Adds latency equal to \
                 the lookahead and reports it. enabled=false gives unity gain but keeps \
                 the lookahead delay, so the latency stays constant when toggled.",
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

impl BaseTransformImpl for StromDspLimiter {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        *self.engine.lock().unwrap() = None;
        self.rate.store(0, Ordering::Release);
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
        let old_rate = self.rate.load(Ordering::Acquire);
        let old_latency = self.latency();
        common::configure_engine(&self.engine, &self.settings, rate, channels, apply);
        self.rate.store(rate, Ordering::Release);
        let new_latency = self.latency();
        if old_rate != 0 && new_latency != old_latency {
            // Renegotiation: frame rounding at the new rate changed the latency.
            let obj = self.obj();
            let _ = obj.post_message(gst::message::Latency::builder().src(&*obj).build());
        }
        self.parent_set_caps(incaps, outcaps)
    }

    fn query(&self, direction: gst::PadDirection, query: &mut gst::QueryRef) -> bool {
        if direction == gst::PadDirection::Src {
            if let gst::QueryViewMut::Latency(q) = query.view_mut() {
                let sinkpad = self.obj().sink_pad().clone();
                let mut upstream = gst::query::Latency::new();
                if !sinkpad.peer_query(&mut upstream) {
                    return false;
                }
                let (live, min, max) = upstream.result();
                let ours = self.latency();
                let min = min + ours;
                let max = max.map(|m| m + ours);
                gst::debug!(
                    CAT,
                    imp = self,
                    "latency: upstream live={live}, adding {ours}: min {min}, max {}",
                    max.display()
                );
                q.set(live, min, max);
                return true;
            }
        }
        BaseTransformImplExt::parent_query(self, direction, query)
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
