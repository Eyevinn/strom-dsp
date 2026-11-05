//! `stromdsp-declipper` implementation. See the module documentation in
//! `mod.rs` for the interface.
//!
//! Settings hand-over follows `common.rs`. The latency setting is copied
//! into the processor only in `set_caps`, before `configure`; the per-buffer
//! update keeps the configured one, so a write while streaming takes effect
//! at the next negotiation. `enabled` is handed to the library, which
//! crossfades to its own delayed dry signal, so the element never switches
//! to BaseTransform passthrough and the latency never changes with it. The
//! meter message is built (and allocates) only when it is due.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{LazyLock, Mutex};

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::prelude::*;
use gst_base::subclass::prelude::*;
use strom_dsp::declipper::{
    Declipper, DeclipperParams, DeclipperStats, MAX_OVERSHOOT_DB_RANGE, MIN_LEVEL_DB_RANGE,
};
use strom_dsp::Processor;

use super::{StromDspDeclipperLatency as LatencyEnum, MESSAGE_NAME};
use crate::common::{self, Engine, SharedSettings};

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "stromdsp-declipper",
        gst::DebugColorFlags::empty(),
        Some("strom-dsp declipper"),
    )
});

const DEFAULT_METER_INTERVAL_MS: u32 = 1000;
const MAX_METER_INTERVAL_MS: u32 = 10_000;
/// Rate assumed for the latency before caps are negotiated.
const FALLBACK_RATE: u32 = 48_000;

#[derive(Debug, Clone, Copy)]
struct Settings {
    params: DeclipperParams,
    latency: LatencyEnum,
    meter_interval_ms: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            params: DeclipperParams::default(),
            latency: LatencyEnum::Standard,
            meter_interval_ms: DEFAULT_METER_INTERVAL_MS,
        }
    }
}

/// What the streaming thread owns; built in `set_caps` (allocates).
#[derive(Debug)]
struct Stream {
    engine: Engine<Declipper>,
    rate: u32,
    /// Frames between meter messages (0 = off).
    interval_frames: u64,
    /// Frames accumulated toward the next message.
    phase: u64,
}

impl Stream {
    fn set_interval(&mut self, ms: u32) {
        self.interval_frames = (f64::from(ms) * 0.001 * f64::from(self.rate)).round() as u64;
        if ms > 0 {
            self.interval_frames = self.interval_frames.max(1);
        } else {
            self.phase = 0;
        }
    }
}

/// Per-buffer parameter update; the latency setting keeps the configured one.
fn apply(dsp: &mut Declipper, s: &Settings) {
    let mut p = s.params;
    p.latency = dsp.latency_mode();
    dsp.set_params(&p);
}

/// `frames` at `rate` as time, rounded up to whole nanoseconds.
fn latency_time(rate: u32, frames: u32) -> gst::ClockTime {
    gst::ClockTime::from_nseconds(
        (u64::from(frames) * gst::ClockTime::SECOND.nseconds()).div_ceil(u64::from(rate)),
    )
}

/// The meter message structure (allocates; once per interval).
fn meter_structure(
    st: &DeclipperStats,
    running_time: Option<gst::ClockTime>,
    stream_time: Option<gst::ClockTime>,
) -> gst::Structure {
    let ns = |t: Option<gst::ClockTime>| t.map_or(u64::MAX, gst::ClockTime::nseconds);
    let over = f64::from(st.max_overshoot_db);
    gst::Structure::builder(MESSAGE_NAME)
        .field("running-time", ns(running_time))
        .field("stream-time", ns(stream_time))
        .field("runs-repaired", st.runs_repaired)
        .field("samples-repaired", st.samples_repaired)
        .field("runs-too-long", st.runs_too_long)
        .field("fallbacks", st.fallbacks)
        .field("runs-rejected", st.runs_rejected)
        .field("runs-failed", st.runs_failed)
        .field(
            "largest-overshoot",
            if over.is_finite() { over.max(0.0) } else { 0.0 },
        )
        .build()
}

#[derive(Debug, Default)]
pub struct StromDspDeclipper {
    settings: SharedSettings<Settings>,
    stream: Mutex<Option<Stream>>,
    /// Negotiated rate (0 = none) and the configured latency (frames), for
    /// the latency query without touching the streaming lock.
    rate: AtomicU32,
    latency_frames: AtomicU32,
}

impl StromDspDeclipper {
    /// The reported latency: the configured one, or before caps the current
    /// setting's at [`FALLBACK_RATE`].
    fn latency(&self) -> gst::ClockTime {
        let rate = self.rate.load(Ordering::Acquire);
        if rate == 0 {
            let mode = self.settings.get().latency.library();
            let frames = Declipper::latency_frames_for(FALLBACK_RATE as f32, mode);
            latency_time(FALLBACK_RATE, frames as u32)
        } else {
            latency_time(rate, self.latency_frames.load(Ordering::Acquire))
        }
    }
}

#[glib::object_subclass]
impl ObjectSubclass for StromDspDeclipper {
    const NAME: &'static str = "StromDspDeclipper";
    type Type = super::StromDspDeclipper;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for StromDspDeclipper {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPS: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            let d = DeclipperParams::default();
            vec![
                common::bool_prop(
                    "enabled",
                    "Enabled",
                    "Repair clipped runs; when false the input passes delayed by the same \
                     latency, crossfaded click-free",
                    d.enabled,
                ),
                glib::ParamSpecEnum::builder_with_default("latency-mode", LatencyEnum::Standard)
                    .nick("Latency mode")
                    .blurb(
                        "Delay available for finding and rebuilding a run: low (3 ms), \
                         standard (5 ms) or high (10 ms). Structural: takes effect at the next \
                         caps negotiation",
                    )
                    .readwrite()
                    .mutable_ready()
                    .build(),
                common::float_prop(
                    "max-overshoot",
                    "Max overshoot",
                    "How far beyond the clip level a rebuilt sample may go (dB); the output \
                     can exceed the input's peak by this much",
                    MAX_OVERSHOOT_DB_RANGE.0,
                    MAX_OVERSHOOT_DB_RANGE.1,
                    d.max_overshoot_db,
                ),
                common::float_prop(
                    "min-level",
                    "Min level",
                    "Lowest clip level that is repaired (dBFS); a flat run below it is left \
                     alone",
                    MIN_LEVEL_DB_RANGE.0,
                    MIN_LEVEL_DB_RANGE.1,
                    d.min_level_db,
                ),
                glib::ParamSpecUInt::builder("meter-interval")
                    .nick("Meter interval")
                    .blurb("Time between two meter messages (ms of stream time); 0 = no messages")
                    .minimum(0)
                    .maximum(MAX_METER_INTERVAL_MS)
                    .default_value(DEFAULT_METER_INTERVAL_MS)
                    .readwrite()
                    .controllable()
                    .mutable_playing()
                    .build(),
            ]
        });
        PROPS.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let name = pspec.name();
        match name {
            "enabled" => {
                let v: bool = value.get().expect("type checked upstream");
                self.settings.update(|s| s.params.enabled = v);
            }
            "latency-mode" => {
                let v: LatencyEnum = value.get().expect("type checked upstream");
                self.settings.update(|s| {
                    s.latency = v;
                    s.params.latency = v.library();
                });
                if self.rate.load(Ordering::Acquire) != 0 {
                    gst::warning!(
                        CAT,
                        imp = self,
                        "latency-mode is READY-only: {v:?} takes effect at the next caps \
                         negotiation"
                    );
                }
            }
            "max-overshoot" => {
                let v: f32 = value.get().expect("type checked upstream");
                self.settings.update(|s| s.params.max_overshoot_db = v);
            }
            "min-level" => {
                let v: f32 = value.get().expect("type checked upstream");
                self.settings.update(|s| s.params.min_level_db = v);
            }
            "meter-interval" => {
                let v: u32 = value.get().expect("type checked upstream");
                self.settings.update(|s| s.meter_interval_ms = v);
            }
            _ => unreachable!("unknown property {name}"),
        }
        gst::debug!(CAT, imp = self, "{name} set to {value:?}");
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let s = self.settings.get();
        match pspec.name() {
            "enabled" => s.params.enabled.to_value(),
            "latency-mode" => s.latency.to_value(),
            "max-overshoot" => s.params.max_overshoot_db.to_value(),
            "min-level" => s.params.min_level_db.to_value(),
            "meter-interval" => s.meter_interval_ms.to_value(),
            name => unreachable!("unknown property {name}"),
        }
    }
}

impl GstObjectImpl for StromDspDeclipper {}

impl ElementImpl for StromDspDeclipper {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static META: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "strom-dsp declipper",
                "Filter/Effect/Audio",
                "Repairs hard-clipped audio: finds clipped runs on the waveform and rebuilds \
                 them by constrained AR interpolation; everything else passes bit-exact. \
                 Reports its latency (latency-mode: 3, 5 or 10 ms). Rebuilt peaks can exceed \
                 the input's peak: put a limiter after it. Posts 'stromdsp-declipper' meter \
                 messages every meter-interval ms.",
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

impl BaseTransformImpl for StromDspDeclipper {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        *self.stream.lock().unwrap() = None;
        self.rate.store(0, Ordering::Release);
        gst::debug!(CAT, imp = self, "stopped");
        Ok(())
    }

    fn set_caps(&self, incaps: &gst::Caps, outcaps: &gst::Caps) -> Result<(), gst::LoggableError> {
        let (rate, channels) = common::parse_caps(incaps)?;
        let old_latency = self.latency();
        let frames = {
            let mut guard = self.stream.lock().unwrap();
            let mut dsp = guard.take().map(|s| s.engine.dsp).unwrap_or_default();
            let (s, generation) = self.settings.snapshot();
            // The latency setting first: `configure` sizes the delay for it.
            dsp.set_params(&s.params);
            let mut engine = Engine::new(dsp, rate, channels);
            engine.applied = generation;
            let mut st = Stream {
                engine,
                rate,
                interval_frames: 0,
                phase: 0,
            };
            st.set_interval(s.meter_interval_ms);
            let frames = st.engine.dsp.latency_frames() as u32;
            *guard = Some(st);
            frames
        };
        self.latency_frames.store(frames, Ordering::Release);
        self.rate.store(rate, Ordering::Release);
        let new_latency = self.latency();
        gst::debug!(
            CAT,
            imp = self,
            "configured for {rate} Hz, {channels} channels: latency {frames} frames, {new_latency}"
        );
        if new_latency != old_latency {
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
            if let Some(st) = self.stream.lock().unwrap().as_mut() {
                st.engine.dsp.reset();
                st.phase = 0;
            }
        }
        self.parent_sink_event(event)
    }

    fn before_transform(&self, inbuf: &gst::BufferRef) {
        common::sync_controlled_values(&*self.obj(), inbuf);
    }

    fn transform_ip(&self, buf: &mut gst::BufferRef) -> Result<gst::FlowSuccess, gst::FlowError> {
        // Process; decide under the lock whether a meter message is due.
        let due = {
            let mut guard = self.stream.lock().unwrap();
            let st = guard.as_mut().ok_or(gst::FlowError::NotNegotiated)?;
            if let Some(s) = self.settings.fetch_if_changed(&mut st.engine.applied) {
                apply(&mut st.engine.dsp, &s);
                st.set_interval(s.meter_interval_ms);
            }
            if buf.flags().contains(gst::BufferFlags::DISCONT) {
                st.engine.dsp.reset();
            }
            let frames = {
                let mut map = buf.map_writable().map_err(|_| gst::FlowError::Error)?;
                st.engine.process_bytes(map.as_mut_slice());
                (map.len() / (4 * st.engine.channels)) as u64
            };
            if st.interval_frames == 0 {
                None
            } else {
                st.phase += frames;
                if st.phase >= st.interval_frames {
                    st.phase -= st.interval_frames;
                    if st.phase >= st.interval_frames {
                        // Buffers longer than the interval: one message each.
                        st.phase = 0;
                    }
                    Some((st.engine.dsp.take_stats(), st.rate, frames))
                } else {
                    None
                }
            }
        };
        let Some((stats, rate, frames)) = due else {
            return Ok(gst::FlowSuccess::Ok);
        };
        // Once per interval: times from the segment (takes the object lock),
        // build and post the message.
        let end = buf.pts().map(|pts| {
            pts + buf.duration().unwrap_or_else(|| {
                gst::ClockTime::SECOND
                    .mul_div_floor(frames, u64::from(rate))
                    .unwrap_or(gst::ClockTime::ZERO)
            })
        });
        let obj = self.obj();
        let segment = obj.segment();
        let (running_time, stream_time) = match (end, segment.downcast_ref::<gst::ClockTime>()) {
            (Some(end), Some(seg)) => (seg.to_running_time(end), seg.to_stream_time(end)),
            _ => (None, None),
        };
        let s = meter_structure(&stats, running_time, stream_time);
        let _ = obj.post_message(gst::message::Element::builder(s).src(&*obj).build());
        Ok(gst::FlowSuccess::Ok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_values_are_finite() {
        gst::init().unwrap();
        let st = DeclipperStats {
            runs_repaired: 3,
            max_overshoot_db: f32::NAN,
            ..DeclipperStats::default()
        };
        let s = meter_structure(&st, None, None);
        assert_eq!(s.get::<u64>("runs-repaired").unwrap(), 3);
        assert_eq!(s.get::<f64>("largest-overshoot").unwrap(), 0.0);
        assert_eq!(s.get::<u64>("running-time").unwrap(), u64::MAX);
    }
}
