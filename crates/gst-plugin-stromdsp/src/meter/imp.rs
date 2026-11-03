//! `stromdsp-meter` implementation. See the module documentation in
//! `mod.rs` for the message format.

use std::sync::{LazyLock, Mutex};

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_audio::AudioChannelPosition as Pos;
use gst_base::prelude::*;
use gst_base::subclass::prelude::*;
use strom_dsp::meter::{
    BandCorrelation, ChannelLayout, ChannelRole, CorrelationMeter, LoudnessConfig, LoudnessMeter,
    MonoDetector, MonoDetectorParams, StereoState, DEFAULT_CORRELATION_MS, MAX_LOUDNESS_CHANNELS,
};

use super::{MESSAGE_NAME, SILENCE_DB};
use crate::common::{self, SharedSettings};

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "stromdsp-meter",
        gst::DebugColorFlags::empty(),
        Some("strom-dsp loudness/peak/correlation meter"),
    )
});

const DEFAULT_INTERVAL_MS: u32 = 100;
const MIN_INTERVAL_MS: u32 = 10;
const MAX_INTERVAL_MS: u32 = 60_000;

/// Samples converted from bytes per step.
const SCRATCH_SAMPLES: usize = 4096;

#[derive(Debug, Clone, Copy)]
struct Settings {
    interval_ms: u32,
    true_peak: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            interval_ms: DEFAULT_INTERVAL_MS,
            true_peak: true,
        }
    }
}

/// Stereo-only meters.
#[derive(Debug)]
struct Stereo {
    correlation: CorrelationMeter,
    bands: BandCorrelation,
    mono: MonoDetector,
}

/// Everything the streaming thread owns; built in `set_caps` (allocates).
#[derive(Debug)]
struct State {
    rate: u32,
    channels: usize,
    true_peak: bool,
    loudness: LoudnessMeter,
    stereo: Option<Stereo>,
    /// Per-channel Σx² since the last message.
    sum_sq: Vec<f64>,
    scratch: Vec<f32>,
    interval_frames: u64,
    /// Frames accumulated toward the next message.
    phase: u64,
    /// Frames measured since the last message.
    since_post: u64,
    /// Settings generation last applied.
    applied: u64,
}

impl State {
    fn new(
        rate: u32,
        channels: usize,
        roles: Option<Vec<ChannelRole>>,
        settings: Settings,
        generation: u64,
    ) -> Result<Self, gst::LoggableError> {
        let mut loudness = LoudnessMeter::new(&LoudnessConfig {
            sample_rate: rate as f32,
            channels,
            layout: ChannelLayout::Unknown,
            true_peak: settings.true_peak,
        })
        .map_err(|e| gst::loggable_error!(CAT, "cannot set up the meter: {e}"))?;
        if let Some(roles) = roles {
            loudness
                .set_channel_roles(&roles)
                .map_err(|e| gst::loggable_error!(CAT, "invalid channel roles: {e}"))?;
        }
        let fs = rate as f32;
        let stereo = (channels == 2).then(|| Stereo {
            correlation: CorrelationMeter::new(fs, DEFAULT_CORRELATION_MS),
            bands: BandCorrelation::new(fs, DEFAULT_CORRELATION_MS),
            mono: MonoDetector::new(fs, &MonoDetectorParams::default()),
        });
        let mut s = Self {
            rate,
            channels,
            true_peak: settings.true_peak,
            loudness,
            stereo,
            sum_sq: vec![0.0; channels],
            scratch: vec![0.0; (SCRATCH_SAMPLES / channels).max(1) * channels],
            interval_frames: 1,
            phase: 0,
            since_post: 0,
            applied: generation,
        };
        s.set_interval(settings.interval_ms);
        Ok(s)
    }

    fn set_interval(&mut self, ms: u32) {
        let frames = (ms as f64 * 0.001 * self.rate as f64).round() as u64;
        self.interval_frames = frames.max(1);
    }

    /// Restart the per-interval values (RMS, interval peaks, scheduling).
    fn clear_interval(&mut self) {
        self.sum_sq.fill(0.0);
        self.loudness.clear_interval_peaks();
        self.since_post = 0;
        self.phase = 0;
    }

    /// Feed one buffer's payload (F32LE, interleaved). Returns the number of
    /// whole frames measured. No allocation.
    fn measure(&mut self, data: &[u8]) -> u64 {
        let ch = self.channels;
        let usable = data.len() / (4 * ch) * (4 * ch);
        for chunk in data[..usable].chunks(self.scratch.len() * 4) {
            let samples = &mut self.scratch[..chunk.len() / 4];
            for (s, b) in samples.iter_mut().zip(chunk.chunks_exact(4)) {
                *s = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
            self.loudness.process(samples);
            for frame in samples.chunks_exact(ch) {
                for (acc, &x) in self.sum_sq.iter_mut().zip(frame) {
                    *acc += x as f64 * x as f64;
                }
            }
            if let Some(st) = self.stereo.as_mut() {
                st.correlation.process_interleaved(samples, 2);
                st.bands.process_interleaved(samples, 2);
                st.mono.process_interleaved(samples, 2);
            }
        }
        (usable / (4 * ch)) as u64
    }

    /// Read every value and build the message structure (allocates; called
    /// once per interval). Clears the interval values.
    fn build(
        &mut self,
        stream_time: Option<gst::ClockTime>,
        running_time: Option<gst::ClockTime>,
    ) -> gst::Structure {
        let ch = self.channels;
        let duration = gst::ClockTime::SECOND
            .mul_div_floor(self.since_post, self.rate as u64)
            .unwrap_or(gst::ClockTime::ZERO);
        let mut s = gst::Structure::new_empty(MESSAGE_NAME);
        s.set("stream-time", stream_time);
        s.set("running-time", running_time);
        s.set("duration", duration);
        s.set("channels", ch as i32);
        let m = &self.loudness;
        s.set("momentary", lufs(m.momentary()));
        s.set("short-term", lufs(m.short_term()));
        s.set("integrated", lufs(m.integrated()));
        s.set("loudness-range", finite_or(m.loudness_range(), 0.0));

        let sp: Vec<f64> = (0..ch).map(|c| m.interval_sample_peak(c)).collect();
        s.set(
            "sample-peak-max",
            lin_to_db(sp.iter().copied().fold(0.0, f64::max)),
        );
        s.set(
            "sample-peak",
            gst::Array::new(sp.into_iter().map(lin_to_db)),
        );
        if self.true_peak {
            let tp: Vec<f64> = (0..ch).map(|c| m.interval_true_peak(c)).collect();
            s.set(
                "true-peak-max",
                lin_to_db(tp.iter().copied().fold(0.0, f64::max)),
            );
            s.set("true-peak", gst::Array::new(tp.into_iter().map(lin_to_db)));
        }
        let prog_sp = (0..ch).map(|c| m.sample_peak(c)).fold(0.0, f64::max);
        s.set("programme-sample-peak", lin_to_db(prog_sp));
        if self.true_peak {
            let prog_tp = (0..ch).map(|c| m.true_peak(c)).fold(0.0, f64::max);
            s.set("programme-true-peak", lin_to_db(prog_tp));
        }
        let n = self.since_post.max(1) as f64;
        s.set(
            "rms",
            gst::Array::new(self.sum_sq.iter().map(|&e| lin_to_db((e / n).sqrt()))),
        );

        if let Some(st) = &self.stereo {
            s.set("correlation", st.correlation.correlation());
            let [low, mid, high] = st.bands.correlations();
            s.set("correlation-low", low);
            s.set("correlation-mid", mid);
            s.set("correlation-high", high);
            s.set("stereo-state", stereo_state_name(st.mono.state()));
        }
        self.sum_sq.fill(0.0);
        self.loudness.clear_interval_peaks();
        self.since_post = 0;
        s
    }
}

/// `20·log10(x)`, clamped to ≥ [`SILENCE_DB`]; 0 and non-finite → `SILENCE_DB`.
pub(crate) fn lin_to_db(x: f64) -> f64 {
    if x > 0.0 && x.is_finite() {
        (20.0 * x.log10()).max(SILENCE_DB)
    } else {
        SILENCE_DB
    }
}

/// A loudness in LUFS with −∞ (and NaN) mapped to [`SILENCE_DB`].
fn lufs(x: f64) -> f64 {
    if x.is_finite() {
        x.max(SILENCE_DB)
    } else {
        SILENCE_DB
    }
}

fn finite_or(x: f64, fallback: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        fallback
    }
}

/// The `stereo-state` string of a [`StereoState`].
fn stereo_state_name(s: StereoState) -> &'static str {
    match s {
        StereoState::Mono => "mono",
        StereoState::Stereo => "stereo",
        StereoState::Inverted => "inverted",
    }
}

/// BS.1770 roles from the caps' channel positions, or `None` (weights 1.0)
/// without a non-zero channel mask. See the table in the module docs.
fn roles_from_caps(caps: &gst::Caps) -> Option<Vec<ChannelRole>> {
    let s = caps.structure(0)?;
    let mask = s.get::<gst::Bitmask>("channel-mask").ok()?;
    if mask.0 == 0 {
        return None;
    }
    let info = gst_audio::AudioInfo::from_caps(caps).ok()?;
    let positions = info.positions()?;
    Some(roles_from_positions(positions))
}

fn roles_from_positions(positions: &[Pos]) -> Vec<ChannelRole> {
    let has_side = positions
        .iter()
        .any(|p| matches!(p, Pos::SideLeft | Pos::SideRight));
    positions
        .iter()
        .map(|p| match p {
            Pos::Lfe1 | Pos::Lfe2 => ChannelRole::Lfe,
            Pos::SideLeft | Pos::SideRight | Pos::SurroundLeft | Pos::SurroundRight => {
                ChannelRole::Surround
            }
            Pos::RearLeft | Pos::RearRight if !has_side => ChannelRole::Surround,
            _ => ChannelRole::Front,
        })
        .collect()
}

/// Caps: interleaved F32LE, any rate, 1 … 64 channels.
fn pad_templates() -> Vec<gst::PadTemplate> {
    let caps = gst_audio::AudioCapsBuilder::new_interleaved()
        .format(gst_audio::AudioFormat::F32le)
        .rate_range(1..=i32::MAX)
        .channels_range(1..=MAX_LOUDNESS_CHANNELS as i32)
        .build();
    [gst::PadDirection::Src, gst::PadDirection::Sink]
        .into_iter()
        .map(|dir| {
            let name = if dir == gst::PadDirection::Src {
                "src"
            } else {
                "sink"
            };
            gst::PadTemplate::new(name, dir, gst::PadPresence::Always, &caps).unwrap()
        })
        .collect()
}

#[derive(Debug, Default)]
pub struct StromDspMeter {
    settings: SharedSettings<Settings>,
    state: Mutex<Option<State>>,
}

#[glib::object_subclass]
impl ObjectSubclass for StromDspMeter {
    const NAME: &'static str = "StromDspMeter";
    type Type = super::StromDspMeter;
    type ParentType = gst_base::BaseTransform;
}

impl ObjectImpl for StromDspMeter {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPS: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            vec![
                glib::ParamSpecUInt::builder("interval")
                    .nick("Interval")
                    .blurb("Time between two meter messages (ms of stream time)")
                    .minimum(MIN_INTERVAL_MS)
                    .maximum(MAX_INTERVAL_MS)
                    .default_value(DEFAULT_INTERVAL_MS)
                    .readwrite()
                    .mutable_playing()
                    .build(),
                glib::ParamSpecBoolean::builder("true-peak")
                    .nick("True peak")
                    .blurb(
                        "Measure true peak (4x oversampling, costs CPU); takes effect \
                         at the next caps negotiation",
                    )
                    .default_value(true)
                    .readwrite()
                    .mutable_ready()
                    .build(),
            ]
        });
        PROPS.as_ref()
    }

    fn signals() -> &'static [glib::subclass::Signal] {
        static SIGNALS: LazyLock<Vec<glib::subclass::Signal>> = LazyLock::new(|| {
            vec![glib::subclass::Signal::builder("reset")
                .action()
                .class_handler(|args| {
                    let obj = args[0]
                        .get::<super::StromDspMeter>()
                        .expect("signal emitted on the meter");
                    let imp = obj.imp();
                    gst::debug!(CAT, imp = imp, "reset requested");
                    imp.settings.request_reset();
                    None
                })
                .build()]
        });
        SIGNALS.as_ref()
    }

    fn constructed(&self) {
        self.parent_constructed();
        // Never touches the samples: always passthrough, measured read-only in
        // `transform_ip_passthrough`.
        self.obj().set_passthrough(true);
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        match pspec.name() {
            "interval" => {
                let v: u32 = value.get().expect("type checked upstream");
                self.settings.update(|s| s.interval_ms = v);
                gst::debug!(CAT, imp = self, "interval set to {v} ms");
            }
            "true-peak" => {
                let v: bool = value.get().expect("type checked upstream");
                self.settings.update(|s| s.true_peak = v);
                gst::debug!(CAT, imp = self, "true-peak set to {v}");
            }
            name => unreachable!("unknown property {name}"),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let s = self.settings.get();
        match pspec.name() {
            "interval" => s.interval_ms.to_value(),
            "true-peak" => s.true_peak.to_value(),
            name => unreachable!("unknown property {name}"),
        }
    }
}

impl GstObjectImpl for StromDspMeter {}

impl ElementImpl for StromDspMeter {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static META: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "strom-dsp meter",
                "Filter/Analyzer/Audio",
                "EBU R 128 loudness (momentary, short-term, integrated, LRA), sample and \
                 true peak, RMS, stereo correlation (broadband and 3 bands) and mono \
                 detection, posted as 'stromdsp-meter' element messages every interval. \
                 Passthrough: samples are never changed.",
                "Eyevinn Technology",
            )
        });
        Some(&*META)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static T: LazyLock<Vec<gst::PadTemplate>> = LazyLock::new(pad_templates);
        T.as_ref()
    }
}

impl BaseTransformImpl for StromDspMeter {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = true;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = true;

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        *self.state.lock().unwrap() = None;
        gst::debug!(CAT, imp = self, "stopped");
        Ok(())
    }

    fn set_caps(&self, incaps: &gst::Caps, outcaps: &gst::Caps) -> Result<(), gst::LoggableError> {
        let (rate, channels) = common::parse_caps(incaps)?;
        let roles = roles_from_caps(incaps);
        gst::debug!(
            CAT,
            imp = self,
            "configuring for {rate} Hz, {channels} channels, roles {roles:?}"
        );
        let (settings, generation) = self.settings.snapshot();
        let state = State::new(rate, channels, roles, settings, generation)?;
        // Fresh measurement; a pending reset is moot.
        self.settings.take_reset();
        *self.state.lock().unwrap() = Some(state);
        self.parent_set_caps(incaps, outcaps)
    }

    fn sink_event(&self, event: gst::Event) -> bool {
        if let gst::EventView::FlushStop(_) = event.view() {
            gst::debug!(CAT, imp = self, "flush-stop: restarting the interval");
            if let Some(st) = self.state.lock().unwrap().as_mut() {
                st.clear_interval();
            }
        }
        self.parent_sink_event(event)
    }

    fn transform_ip_passthrough(
        &self,
        buf: &gst::Buffer,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let due = {
            let mut guard = self.state.lock().unwrap();
            let st = guard.as_mut().ok_or(gst::FlowError::NotNegotiated)?;
            if let Some(s) = self.settings.fetch_if_changed(&mut st.applied) {
                st.set_interval(s.interval_ms);
            }
            if self.settings.take_reset() {
                st.loudness.reset();
            }
            let map = buf.map_readable().map_err(|_| gst::FlowError::Error)?;
            let frames = st.measure(&map);
            drop(map);
            st.phase += frames;
            st.since_post += frames;
            if st.phase >= st.interval_frames {
                st.phase -= st.interval_frames;
                if st.phase >= st.interval_frames {
                    st.phase = 0;
                }
                // End of this buffer = the instant the message describes.
                let end = buf.pts().map(|pts| {
                    pts + buf.duration().unwrap_or_else(|| {
                        gst::ClockTime::SECOND
                            .mul_div_floor(frames, st.rate as u64)
                            .unwrap_or(gst::ClockTime::ZERO)
                    })
                });
                Some(end)
            } else {
                None
            }
        };
        let Some(end) = due else {
            return Ok(gst::FlowSuccess::Ok);
        };
        // Times from the segment (takes the object lock; once per interval).
        let segment = self.obj().segment();
        let (stream_time, running_time) = match (end, segment.downcast_ref::<gst::ClockTime>()) {
            (Some(end), Some(seg)) => (seg.to_stream_time(end), seg.to_running_time(end)),
            _ => (None, None),
        };
        let structure = {
            let mut guard = self.state.lock().unwrap();
            let Some(st) = guard.as_mut() else {
                return Ok(gst::FlowSuccess::Ok);
            };
            st.build(stream_time, running_time)
        };
        let obj = self.obj();
        let _ = obj.post_message(gst::message::Element::builder(structure).src(&*obj).build());
        Ok(gst::FlowSuccess::Ok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_mapping() {
        use ChannelRole::{Front, Lfe, Surround};
        let p51 = [
            Pos::FrontLeft,
            Pos::FrontRight,
            Pos::FrontCenter,
            Pos::Lfe1,
            Pos::RearLeft,
            Pos::RearRight,
        ];
        assert_eq!(
            roles_from_positions(&p51),
            [Front, Front, Front, Lfe, Surround, Surround]
        );
        let p71 = [
            Pos::FrontLeft,
            Pos::FrontRight,
            Pos::FrontCenter,
            Pos::Lfe1,
            Pos::RearLeft,
            Pos::RearRight,
            Pos::SideLeft,
            Pos::SideRight,
        ];
        assert_eq!(
            roles_from_positions(&p71),
            [Front, Front, Front, Lfe, Front, Front, Surround, Surround]
        );
        assert_eq!(roles_from_positions(&[Pos::Mono]), [Front], "mono position");
        assert_eq!(
            roles_from_positions(&[Pos::TopFrontLeft, Pos::Lfe2, Pos::SurroundRight]),
            [Front, Lfe, Surround]
        );
    }

    #[test]
    fn db_conversions_are_finite() {
        assert_eq!(lin_to_db(0.0), SILENCE_DB);
        assert_eq!(lin_to_db(1e-30), SILENCE_DB);
        assert_eq!(lin_to_db(f64::NAN), SILENCE_DB);
        assert!((lin_to_db(0.5) + 6.0206).abs() < 1e-3);
        assert_eq!(lufs(f64::NEG_INFINITY), SILENCE_DB);
        assert_eq!(lufs(f64::NAN), SILENCE_DB);
        assert_eq!(lufs(-23.0), -23.0);
    }
}
