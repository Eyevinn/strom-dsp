//! `stromdsp-ducker` implementation: a `GstAggregator` around
//! [`strom_dsp::ducker::Ducker`].
//!
//! # Pads
//!
//! - `sink` (always): the programme. Its caps (F32LE interleaved, any rate and
//!   channel count) become the `src` caps; its segment becomes the output
//!   segment, and its buffers are output with their timestamps, durations,
//!   offsets and flags unchanged ("timestamps follow the programme").
//! - `sidechain` (request, at most one): the key, F32LE interleaved at any
//!   rate and channel count. A request pad so that a ducker without a key is
//!   simply an element without that pad, and the aggregator never waits for
//!   it.
//! - `src` (always).
//!
//! # Aligning the key with the programme
//!
//! Key buffers go into a FIFO (allocated at the key's caps, 2 s) indexed by
//! key frame number. The first key buffer anchors frame 0 at its running
//! time; later buffers are placed by their running time: a gap of more than
//! 1 ms is filled with silence, an overlap of more than 1 ms is dropped,
//! smaller jitter is ignored. For each programme buffer (split into chunks of
//! [`CHUNK_FRAMES`]), the key frames covering the same running-time span are
//! taken from the FIFO; missing frames are zeros (silence). The library runs
//! its detector at the key's rate and interpolates only the envelope.
//!
//! # Live pipelines, missing and stalled keys
//!
//! - [`AggregatorImpl::next_time`] returns the running time of the next
//!   programme buffer (the end of the last one, or the first buffer's start).
//!   In a live pipeline the aggregator waits for the key until that time plus
//!   the pipeline latency (upstream latency plus the `latency` property
//!   inherited from `GstAggregator`), then calls `aggregate` with
//!   `timeout = true`, and the programme goes out with whatever key arrived,
//!   the rest read as silence. A stalled or unlinked key therefore costs no
//!   more than that latency and never blocks.
//! - `ignore-inactive-pads` is on: a requested sidechain that never received
//!   data is not waited for at all in live pipelines.
//! - Without a timeout (all pads have data), the programme buffer waits until
//!   the key covers its whole span, the key is EOS, or the key is inactive.
//! - The key reading as silence makes the ducker hold, then release to unity
//!   with the release fade: no jump.
//! - In **non-live** pipelines the aggregator has no timeout: a requested
//!   `sidechain` must be fed (or reach EOS). Don't request it if there is no
//!   key.
//! - If the programme itself stalls, `next_time` advances in steps of
//!   [`PROGRAMME_POLL`], so the aggregator does not spin.
//!
//! # Per-buffer path
//!
//! Settings are copied in once per buffer through [`SharedSettings`]
//! (generation counter, `try_lock`). The DSP, FIFO and conversion buffers are
//! allocated in the CAPS handlers only. No logging per buffer. The meter
//! message (a `gst::Structure`) is the only allocation, at most once per
//! `meter-interval`.

use std::sync::{LazyLock, Mutex, OnceLock};

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::prelude::*;
use gst_base::subclass::prelude::*;
use strom_dsp::ducker::{Ducker, DuckerParams};
use strom_dsp::Processor;

use super::RangeMode;
use crate::common::{self, SharedSettings};

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "stromdsp-ducker",
        gst::DebugColorFlags::empty(),
        Some("strom-dsp ducker"),
    )
});

/// Programme frames per DSP call.
const CHUNK_FRAMES: usize = 1024;
/// Key FIFO length (seconds of key).
const KEY_FIFO_SECONDS: usize = 2;
/// How far `next_time` advances when the programme stalls.
const PROGRAMME_POLL: gst::ClockTime = gst::ClockTime::from_mseconds(100);
/// Default `meter-interval` (ms).
pub(crate) const DEFAULT_METER_INTERVAL_MS: u32 = 100;
/// Name of the meter message structure.
pub(crate) const METER_MESSAGE: &str = "stromdsp-ducker";

#[derive(Debug, Clone, Copy)]
struct Settings {
    enabled: bool,
    params: DuckerParams,
    meter_interval_ms: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            enabled: true,
            params: DuckerParams::default(),
            meter_interval_ms: DEFAULT_METER_INTERVAL_MS,
        }
    }
}

/// Key samples indexed by key frame number (see the module docs).
#[derive(Debug)]
struct KeyFifo {
    rate: u32,
    channels: usize,
    /// `data[..len·channels]` holds frames `head .. head + len`.
    data: Vec<f32>,
    len: usize,
    head: u64,
    /// Running time (ns) of key frame 0; `None` until the first buffer.
    anchor: Option<u64>,
}

impl KeyFifo {
    fn new(rate: u32, channels: usize) -> Self {
        let cap = (rate as usize * KEY_FIFO_SECONDS).max(CHUNK_FRAMES);
        Self {
            rate,
            channels,
            data: vec![0.0; cap * channels],
            len: 0,
            head: 0,
            anchor: None,
        }
    }

    fn capacity(&self) -> usize {
        self.data.len() / self.channels
    }

    fn clear(&mut self) {
        self.len = 0;
        self.head = 0;
        self.anchor = None;
    }

    fn tail(&self) -> i128 {
        self.head as i128 + self.len as i128
    }

    /// Key frame index at running time `rt` (ns), rounded to nearest.
    fn index_at(&self, anchor: u64, rt: u64) -> i128 {
        let num = (rt as i128 - anchor as i128) * self.rate as i128;
        (num + 500_000_000).div_euclid(1_000_000_000)
    }

    /// Key frame index matching programme frame `j` of a buffer starting at
    /// running time `t0` (ns), at programme rate `prog_rate`.
    fn index_for(&self, anchor: u64, t0: u64, j: usize, prog_rate: u32) -> i128 {
        let r = prog_rate as i128;
        let num =
            ((t0 as i128 - anchor as i128) * r + j as i128 * 1_000_000_000) * self.rate as i128;
        let den = 1_000_000_000 * r;
        (num + den / 2).div_euclid(den)
    }

    /// Prepare to append a buffer of key frames starting at running time
    /// `rt` (ns): anchor, fill a gap with silence, or report an overlap.
    /// Returns how many leading frames of the buffer to skip (overlap).
    fn align(&mut self, rt: u64) -> usize {
        let cap = self.capacity() as i128;
        let e = match self.anchor {
            Some(a) => self.index_at(a, rt),
            None => -1,
        };
        // First buffer, time going back before the buffered data, or a gap
        // longer than the FIFO: start over, anchored at this buffer.
        if self.anchor.is_none() || e < self.head as i128 || e > self.tail() + cap {
            self.clear();
            self.anchor = Some(rt);
            return 0;
        }
        let tol = (self.rate as i128 / 1000).max(1);
        let tail = self.tail();
        if e > tail + tol {
            self.append_zeros((e - tail) as usize);
        } else if e < tail - tol {
            return (tail - e) as usize;
        }
        0
    }

    fn make_room(&mut self, frames: usize) {
        let cap = self.capacity();
        if self.len + frames > cap {
            let drop = (self.len + frames - cap).min(self.len);
            let ch = self.channels;
            self.data.copy_within(drop * ch..self.len * ch, 0);
            self.len -= drop;
            self.head += drop as u64;
        }
    }

    fn append_zeros(&mut self, frames: usize) {
        let frames = frames.min(self.capacity());
        self.make_room(frames);
        let ch = self.channels;
        self.data[self.len * ch..(self.len + frames) * ch].fill(0.0);
        self.len += frames;
    }

    fn append(&mut self, src: &[f32]) {
        let ch = self.channels;
        let mut frames = src.len() / ch;
        let mut src = src;
        let cap = self.capacity();
        if frames > cap {
            // Only the newest `cap` frames fit.
            let skip = frames - cap;
            self.head += (self.len + skip) as u64;
            self.len = 0;
            src = &src[skip * ch..];
            frames = cap;
        }
        self.make_room(frames);
        self.data[self.len * ch..(self.len + frames) * ch].copy_from_slice(src);
        self.len += frames;
    }

    /// Whether frames up to (excluding) `i1` are buffered.
    fn covers(&self, i1: i128) -> bool {
        self.anchor.is_some() && self.tail() >= i1
    }

    /// Copy key frames `i0 .. i0 + out.len()/channels` into `out` (zeros where
    /// missing) and drop everything before the end of that range. Returns
    /// the number of real (not missing) frames.
    fn take(&mut self, i0: i128, out: &mut [f32]) -> usize {
        let ch = self.channels;
        let i1 = i0 + (out.len() / ch) as i128;
        out.fill(0.0);
        let (head, tail) = (self.head as i128, self.tail());
        let (a, b) = (i0.max(head), i1.min(tail));
        let real = if a < b {
            let (sa, sb) = ((a - head) as usize, (b - head) as usize);
            let oa = (a - i0) as usize;
            out[oa * ch..(oa + sb - sa) * ch].copy_from_slice(&self.data[sa * ch..sb * ch]);
            sb - sa
        } else {
            0
        };
        let consumed = (i1 - head).clamp(0, self.len as i128) as usize;
        if consumed > 0 {
            self.data.copy_within(consumed * ch..self.len * ch, 0);
            self.len -= consumed;
            self.head += consumed as u64;
        }
        real
    }
}

/// Streaming state (behind a mutex used almost only by the aggregate thread).
#[derive(Debug, Default)]
struct State {
    dsp: Ducker,
    /// Programme (rate, channels) once caps are known.
    prog: Option<(u32, usize)>,
    key: Option<KeyFifo>,
    prog_scratch: Vec<f32>,
    key_scratch: Vec<f32>,
    applied: u64,
    settings: Settings,
    /// Running time (ns) of the last meter message.
    last_meter: Option<u64>,
}

impl State {
    fn apply(&mut self, s: &Settings) {
        if s.enabled && !self.settings.enabled {
            self.dsp.reset();
        }
        self.dsp.set_params(&s.params);
        self.settings = *s;
    }

    /// (Re)allocate the key conversion buffer for the current formats.
    fn alloc_key_scratch(&mut self) {
        if let (Some((pr, _)), Some(k)) = (self.prog, self.key.as_ref()) {
            let frames = (CHUNK_FRAMES as u64 * k.rate as u64).div_ceil(pr as u64) as usize + 2;
            self.key_scratch = vec![0.0; frames * k.channels];
        }
    }

    fn reset(&mut self) {
        self.dsp.reset();
        if let Some(k) = self.key.as_mut() {
            k.clear();
        }
        self.last_meter = None;
    }
}

/// What the meter message reports.
struct Meter {
    running_time: gst::ClockTime,
    stream_time: Option<gst::ClockTime>,
    reduction: f64,
    key_level: f64,
    key_active: bool,
    key_present: bool,
}

#[derive(Debug, Default)]
pub struct StromDspDucker {
    settings: SharedSettings<Settings>,
    state: Mutex<State>,
    sinkpad: OnceLock<gst_base::AggregatorPad>,
    sidechain: Mutex<Option<gst_base::AggregatorPad>>,
    src_caps: Mutex<Option<gst::Caps>>,
    /// Running time the aggregator waits for (see `next_time`).
    deadline: Mutex<Option<gst::ClockTime>>,
}

#[glib::object_subclass]
impl ObjectSubclass for StromDspDucker {
    const NAME: &'static str = "StromDspDucker";
    type Type = super::StromDspDucker;
    type ParentType = gst_base::Aggregator;
}

impl ObjectImpl for StromDspDucker {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPS: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            use strom_dsp::ducker::{
                ATTACK_RANGE_MS, DEPTH_RANGE_DB, HOLD_RANGE_MS, KEY_HIGHPASS_RANGE_HZ,
                RELEASE_RANGE_MS, THRESHOLD_RANGE_DB,
            };
            let d = DuckerParams::default();
            vec![
                common::bool_prop(
                    "enabled",
                    "Enabled",
                    "Duck the programme; when false the programme passes bit-exact \
                     (the key is still consumed)",
                    true,
                ),
                common::float_prop(
                    "threshold",
                    "Threshold",
                    "Key level (dBFS, sine-calibrated RMS) above which the programme is ducked",
                    THRESHOLD_RANGE_DB.0,
                    THRESHOLD_RANGE_DB.1,
                    d.threshold_db,
                ),
                common::float_prop(
                    "depth",
                    "Depth",
                    "Programme gain at full ducking (dB)",
                    DEPTH_RANGE_DB.0,
                    DEPTH_RANGE_DB.1,
                    d.depth_db,
                ),
                common::float_prop(
                    "attack",
                    "Attack",
                    "Fade time from unity to full depth (ms, linear in dB)",
                    ATTACK_RANGE_MS.0,
                    ATTACK_RANGE_MS.1,
                    d.attack_ms,
                ),
                common::float_prop(
                    "hold",
                    "Hold",
                    "Time the reduction is held after the key drops (ms)",
                    HOLD_RANGE_MS.0,
                    HOLD_RANGE_MS.1,
                    d.hold_ms,
                ),
                common::float_prop(
                    "release",
                    "Release",
                    "Fade time from full depth back to unity (ms, linear in dB)",
                    RELEASE_RANGE_MS.0,
                    RELEASE_RANGE_MS.1,
                    d.release_ms,
                ),
                common::float_prop(
                    "key-highpass",
                    "Key high-pass",
                    "High-pass on the key detector path (Hz, 0 = off)",
                    KEY_HIGHPASS_RANGE_HZ.0,
                    KEY_HIGHPASS_RANGE_HZ.1,
                    d.key_highpass_hz,
                ),
                glib::ParamSpecEnum::builder_with_default("range-mode", RangeMode::Fixed)
                    .nick("Range mode")
                    .blurb(
                        "fixed: full depth above the threshold; proportional: 1 dB of \
                         reduction per dB above the threshold, up to depth",
                    )
                    .readwrite()
                    .controllable()
                    .mutable_playing()
                    .build(),
                glib::ParamSpecUInt::builder("meter-interval")
                    .nick("Meter interval")
                    .blurb(
                        "Interval of the stromdsp-ducker element message (ms of \
                         running time, 0 = off)",
                    )
                    .minimum(0)
                    .maximum(10_000)
                    .default_value(DEFAULT_METER_INTERVAL_MS)
                    .readwrite()
                    .mutable_playing()
                    .build(),
            ]
        });
        PROPS.as_ref()
    }

    fn constructed(&self) {
        self.parent_constructed();
        let obj = self.obj();
        let templ = obj.pad_template("sink").expect("sink template");
        let pad = gst::PadBuilder::<gst_base::AggregatorPad>::from_template(&templ).build();
        // The first programme buffer (after start or a flush) sets the first
        // deadline, in the upstream thread as it arrives; afterwards
        // `aggregate` moves it. (The aggregator does not call `clip` before
        // queueing, so it cannot be used for this.)
        let weak = obj.downgrade();
        pad.add_probe(gst::PadProbeType::BUFFER, move |pad, info| {
            let (Some(el), Some(buf)) = (weak.upgrade(), info.buffer()) else {
                return gst::PadProbeReturn::Ok;
            };
            let imp = el.imp();
            let mut d = imp.deadline.lock().unwrap();
            if d.is_none() {
                let seg = pad.sticky_event::<gst::event::Segment>(0);
                *d = seg.and_then(|s| {
                    let s = s.segment().downcast_ref::<gst::ClockTime>()?.clone();
                    s.to_running_time(buf.pts()?)
                });
            }
            gst::PadProbeReturn::Ok
        });
        obj.add_pad(&pad).expect("add sink pad");
        let _ = self.sinkpad.set(pad);
        // Live: a sidechain that never received data is not waited for.
        obj.set_ignore_inactive_pads(true);
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let name = pspec.name();
        self.settings.update(|s| {
            let p = &mut s.params;
            match name {
                "enabled" => s.enabled = value.get().expect("type checked upstream"),
                "threshold" => p.threshold_db = value.get().expect("type checked upstream"),
                "depth" => p.depth_db = value.get().expect("type checked upstream"),
                "attack" => p.attack_ms = value.get().expect("type checked upstream"),
                "hold" => p.hold_ms = value.get().expect("type checked upstream"),
                "release" => p.release_ms = value.get().expect("type checked upstream"),
                "key-highpass" => p.key_highpass_hz = value.get().expect("type checked upstream"),
                "range-mode" => {
                    p.range_mode = value
                        .get::<RangeMode>()
                        .expect("type checked upstream")
                        .into()
                }
                "meter-interval" => {
                    s.meter_interval_ms = value.get().expect("type checked upstream")
                }
                _ => unreachable!("unknown property {name}"),
            }
        });
        gst::debug!(CAT, imp = self, "{name} set to {value:?}");
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let s = self.settings.get();
        let p = &s.params;
        match pspec.name() {
            "enabled" => s.enabled.to_value(),
            "threshold" => p.threshold_db.to_value(),
            "depth" => p.depth_db.to_value(),
            "attack" => p.attack_ms.to_value(),
            "hold" => p.hold_ms.to_value(),
            "release" => p.release_ms.to_value(),
            "key-highpass" => p.key_highpass_hz.to_value(),
            "range-mode" => RangeMode::from(p.range_mode).to_value(),
            "meter-interval" => s.meter_interval_ms.to_value(),
            name => unreachable!("unknown property {name}"),
        }
    }
}

impl GstObjectImpl for StromDspDucker {}

impl ElementImpl for StromDspDucker {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static META: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "strom-dsp ducker",
                "Filter/Effect/Audio",
                "Lowers the programme (sink) while the key (request pad sidechain, any \
                 rate and channel count) is above a threshold. Output has the programme's \
                 caps and timestamps. In live pipelines a missing or stalled key never \
                 blocks: the programme passes and recovers to unity. Posts \
                 'stromdsp-ducker' element messages every meter-interval ms.",
                "Eyevinn Technology",
            )
        });
        Some(&*META)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static T: LazyLock<Vec<gst::PadTemplate>> = LazyLock::new(|| {
            let caps = common::audio_caps();
            let ty = gst_base::AggregatorPad::static_type();
            let make = |name, dir, presence| {
                gst::PadTemplate::with_gtype(name, dir, presence, &caps, ty).unwrap()
            };
            vec![
                make("src", gst::PadDirection::Src, gst::PadPresence::Always),
                make("sink", gst::PadDirection::Sink, gst::PadPresence::Always),
                make(
                    "sidechain",
                    gst::PadDirection::Sink,
                    gst::PadPresence::Request,
                ),
            ]
        });
        T.as_ref()
    }

    fn release_pad(&self, pad: &gst::Pad) {
        {
            let mut side = self.sidechain.lock().unwrap();
            if side.as_ref().map(|p| p.upcast_ref::<gst::Pad>()) == Some(pad) {
                *side = None;
                gst::debug!(CAT, imp = self, "sidechain released");
            }
        }
        self.parent_release_pad(pad);
    }
}

impl StromDspDucker {
    fn sinkpad(&self) -> &gst_base::AggregatorPad {
        self.sinkpad.get().expect("sink pad created in constructed")
    }

    fn is_sidechain(&self, pad: &gst_base::AggregatorPad) -> bool {
        self.sidechain.lock().unwrap().as_ref() == Some(pad)
    }

    /// Running-time span (ns) of a programme buffer, or `None` without a
    /// usable timestamp.
    fn span(
        segment: &gst::Segment,
        buf: &gst::BufferRef,
        frames: usize,
        rate: u32,
    ) -> Option<(u64, u64)> {
        let seg = segment.downcast_ref::<gst::ClockTime>()?;
        let t0 = seg.to_running_time(buf.pts()?)?.nseconds();
        let dur = gst::ClockTime::SECOND.mul_div_floor(frames as u64, rate as u64)?;
        Some((t0, t0 + dur.nseconds()))
    }

    /// Running time (ns) of the start of a buffer on `pad`.
    fn start_rt(pad: &gst_base::AggregatorPad, buf: &gst::BufferRef) -> Option<u64> {
        let seg = pad.segment();
        let seg = seg.downcast_ref::<gst::ClockTime>()?;
        Some(seg.to_running_time(buf.pts()?)?.nseconds())
    }

    fn set_deadline(&self, t: Option<gst::ClockTime>) {
        *self.deadline.lock().unwrap() = t;
    }

    fn post_meter(&self, m: Meter) {
        let s = gst::Structure::builder(METER_MESSAGE)
            .field("running-time", m.running_time)
            .field("stream-time", m.stream_time)
            .field("reduction", m.reduction)
            .field("key-level", m.key_level)
            .field("key-active", m.key_active)
            .field("key-present", m.key_present)
            .build();
        let obj = self.obj();
        let _ = obj.post_message(gst::message::Element::builder(s).src(&*obj).build());
    }

    /// Duck one programme buffer in place. Returns the meter reading if one
    /// is due.
    fn process_buffer(
        &self,
        state: &mut State,
        buf: &mut gst::BufferRef,
        span: Option<(u64, u64)>,
        has_sidechain: bool,
        stream_time: Option<gst::ClockTime>,
    ) -> Result<Option<Meter>, gst::FlowError> {
        let (rate, ch) = state.prog.ok_or(gst::FlowError::NotNegotiated)?;
        if let Some(s) = self.settings.fetch_if_changed(&mut state.applied) {
            state.apply(&s);
        }
        if buf.flags().contains(gst::BufferFlags::DISCONT) {
            state.dsp.reset();
        }
        let enabled = state.settings.enabled;
        let mut map = buf.map_writable().map_err(|_| gst::FlowError::Error)?;
        let data = map.as_mut_slice();
        let frames = data.len() / (4 * ch);
        let State {
            dsp,
            key,
            prog_scratch,
            key_scratch,
            ..
        } = state;
        // Key available for this buffer: a sidechain with caps and data.
        let key = key
            .as_mut()
            .filter(|k| has_sidechain && k.anchor.is_some() && span.is_some());
        let mut real = 0usize;
        match key {
            Some(k) => {
                let (t0, _) = span.unwrap();
                let anchor = k.anchor.unwrap();
                let kc = k.channels;
                let max_key = key_scratch.len() / kc;
                let mut j = 0usize;
                while j < frames {
                    let n = (frames - j).min(CHUNK_FRAMES);
                    let i0 = k.index_for(anchor, t0, j, rate);
                    let i1 = k.index_for(anchor, t0, j + n, rate);
                    let kn = ((i1 - i0).max(0) as usize).min(max_key);
                    let ks = &mut key_scratch[..kn * kc];
                    real += k.take(i0, ks);
                    let bytes = &mut data[j * ch * 4..(j + n) * ch * 4];
                    if enabled {
                        let ps = &mut prog_scratch[..n * ch];
                        from_le(bytes, ps);
                        dsp.process_with_key(ps, ks, kc);
                        to_le(ps, bytes);
                    }
                    j += n;
                }
            }
            None if enabled => {
                for bytes in data[..frames * ch * 4].chunks_mut(CHUNK_FRAMES * ch * 4) {
                    let ps = &mut prog_scratch[..bytes.len() / 4];
                    from_le(bytes, ps);
                    dsp.process(ps);
                    to_le(ps, bytes);
                }
            }
            None => {}
        }
        drop(map);

        let interval = state.settings.meter_interval_ms as u64 * 1_000_000;
        let Some((_, t1)) = span else {
            return Ok(None);
        };
        if interval == 0 || state.last_meter.is_some_and(|l| t1 < l + interval) {
            return Ok(None);
        }
        state.last_meter = Some(t1);
        let (reduction, key_level, key_active) = if enabled {
            (
                state.dsp.reduction_db(),
                state.dsp.key_level_db(),
                state.dsp.key_active(),
            )
        } else {
            (0.0, state.dsp.key_level_db(), false)
        };
        Ok(Some(Meter {
            running_time: gst::ClockTime::from_nseconds(t1),
            stream_time,
            reduction,
            key_level,
            key_active,
            key_present: real > 0,
        }))
    }
}

fn from_le(bytes: &[u8], out: &mut [f32]) {
    for (s, b) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *s = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
}

fn to_le(samples: &[f32], bytes: &mut [u8]) {
    for (b, s) in bytes.chunks_exact_mut(4).zip(samples) {
        b.copy_from_slice(&s.to_le_bytes());
    }
}

impl AggregatorImpl for StromDspDucker {
    fn start(&self) -> Result<(), gst::ErrorMessage> {
        self.set_deadline(None);
        self.parent_start()
    }

    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        {
            let mut st = self.state.lock().unwrap();
            st.reset();
        }
        self.set_deadline(None);
        gst::debug!(CAT, imp = self, "stopped");
        self.parent_stop()
    }

    fn flush(&self) -> Result<gst::FlowSuccess, gst::FlowError> {
        gst::debug!(CAT, imp = self, "flush: resetting state");
        self.state.lock().unwrap().reset();
        self.set_deadline(None);
        self.parent_flush()
    }

    fn next_time(&self) -> Option<gst::ClockTime> {
        *self.deadline.lock().unwrap()
    }

    fn create_new_pad(
        &self,
        templ: &gst::PadTemplate,
        _req_name: Option<&str>,
        _caps: Option<&gst::Caps>,
    ) -> Option<gst_base::AggregatorPad> {
        if templ.name_template() != "sidechain" {
            return None;
        }
        let mut side = self.sidechain.lock().unwrap();
        if side.is_some() {
            gst::warning!(CAT, imp = self, "sidechain pad already exists");
            return None;
        }
        let pad = gst::PadBuilder::<gst_base::AggregatorPad>::from_template(templ)
            .name("sidechain")
            .build();
        *side = Some(pad.clone());
        gst::debug!(CAT, imp = self, "sidechain requested");
        Some(pad)
    }

    fn sink_event(&self, pad: &gst_base::AggregatorPad, event: gst::Event) -> bool {
        match event.view() {
            gst::EventView::Caps(c) => {
                let caps = c.caps_owned();
                let Ok((rate, channels)) = common::parse_caps(&caps) else {
                    gst::warning!(CAT, imp = self, "unsupported caps {caps}");
                    return false;
                };
                let mut st = self.state.lock().unwrap();
                if pad == self.sinkpad() {
                    gst::debug!(CAT, imp = self, "programme: {rate} Hz, {channels} ch");
                    st.prog = Some((rate, channels));
                    st.dsp.configure(rate as f32, channels);
                    st.prog_scratch = vec![0.0; CHUNK_FRAMES * channels];
                    let (s, generation) = self.settings.snapshot();
                    st.dsp.set_params(&s.params);
                    st.settings = s;
                    st.applied = generation;
                    st.alloc_key_scratch();
                    drop(st);
                    *self.src_caps.lock().unwrap() = Some(caps);
                    self.obj().src_pad().mark_reconfigure();
                } else if self.is_sidechain(pad) {
                    gst::debug!(CAT, imp = self, "key: {rate} Hz, {channels} ch");
                    st.dsp.set_key_format(rate as f32, channels);
                    st.key = Some(KeyFifo::new(rate, channels));
                    st.alloc_key_scratch();
                }
            }
            gst::EventView::Segment(s) if pad == self.sinkpad() => {
                if let Some(seg) = s.segment().downcast_ref::<gst::ClockTime>() {
                    self.obj().update_segment(seg);
                }
            }
            _ => {}
        }
        self.parent_sink_event(pad, event)
    }

    fn update_src_caps(&self, _caps: &gst::Caps) -> Result<gst::Caps, gst::FlowError> {
        self.src_caps
            .lock()
            .unwrap()
            .clone()
            .ok_or(gst_base::AGGREGATOR_FLOW_NEED_DATA)
    }

    fn aggregate(&self, timeout: bool) -> Result<gst::FlowSuccess, gst::FlowError> {
        let sink = self.sinkpad();
        let Some(buf) = sink.peek_buffer() else {
            if sink.is_eos() {
                return Err(gst::FlowError::Eos);
            }
            if timeout {
                // The programme stalled: poll instead of spinning.
                let mut d = self.deadline.lock().unwrap();
                *d = d.map(|t| t + PROGRAMME_POLL);
            }
            return Err(gst_base::AGGREGATOR_FLOW_NEED_DATA);
        };
        let Some((rate, ch)) = self.state.lock().unwrap().prog else {
            return Err(gst::FlowError::NotNegotiated);
        };
        let frames = buf.size() / (4 * ch);
        let segment = sink.segment();
        let span = Self::span(&segment, &buf, frames, rate);

        if let (true, Some((t0, _))) = (timeout, span) {
            // A deadline left over from a programme stall: give the key its
            // full latency budget for this buffer first.
            let mut d = self.deadline.lock().unwrap();
            if d.is_some_and(|d| d.nseconds() < t0) {
                *d = Some(gst::ClockTime::from_nseconds(t0));
                return Err(gst_base::AGGREGATOR_FLOW_NEED_DATA);
            }
        }

        // Move key buffers that start before the end of this programme
        // buffer into the FIFO.
        let side = self.sidechain.lock().unwrap().clone();
        let mut key_complete = true;
        if let (Some(side), Some((_, t1))) = (side.as_ref(), span) {
            loop {
                match side.peek_buffer() {
                    Some(kb) => {
                        let start = Self::start_rt(side, &kb);
                        if start.is_some_and(|s| s >= t1) {
                            break;
                        }
                        side.drop_buffer();
                        let (Some(start), Ok(map)) = (start, kb.map_readable()) else {
                            continue;
                        };
                        let mut st = self.state.lock().unwrap();
                        if let Some(k) = st.key.as_mut() {
                            push_le(k, start, map.as_slice());
                        }
                    }
                    None => {
                        let st = self.state.lock().unwrap();
                        let covered = st.key.as_ref().is_some_and(|k| match k.anchor {
                            Some(a) => k.covers(k.index_for(a, span.unwrap().0, frames, rate)),
                            None => false,
                        });
                        key_complete = covered || side.is_eos() || side.is_inactive();
                        break;
                    }
                }
            }
        }
        if !key_complete && !timeout {
            return Err(gst_base::AGGREGATOR_FLOW_NEED_DATA);
        }

        let mut buf = sink
            .pop_buffer()
            .ok_or(gst_base::AGGREGATOR_FLOW_NEED_DATA)?;
        let time_seg = segment.downcast_ref::<gst::ClockTime>();
        if let Some(st) = time_seg.and_then(|s| s.to_stream_time(buf.pts()?)) {
            let _ = self.obj().sync_values(st);
        }
        // The meter reports the end of the buffer, in stream and running time.
        let stream_time = time_seg.and_then(|s| {
            let dur = gst::ClockTime::SECOND.mul_div_floor(frames as u64, rate as u64)?;
            s.to_stream_time(buf.pts()? + dur)
        });
        let meter = {
            let mut st = self.state.lock().unwrap();
            self.process_buffer(&mut st, buf.make_mut(), span, side.is_some(), stream_time)?
        };
        if let Some((_, t1)) = span {
            self.set_deadline(Some(gst::ClockTime::from_nseconds(t1)));
        }
        if let Some(m) = meter {
            self.post_meter(m);
        }
        self.finish_buffer(buf)
    }
}

/// Append the F32LE key buffer `bytes` starting at running time `rt` to the
/// FIFO, converting through a stack buffer (no allocation).
fn push_le(k: &mut KeyFifo, rt: u64, bytes: &[u8]) {
    let ch = k.channels;
    let mut tmp = [0.0f32; 1024];
    let per = tmp.len() / ch;
    if per == 0 {
        // More than 1024 key channels: not supported, the key reads as silence.
        return;
    }
    let frames = bytes.len() / (4 * ch);
    let mut done = k.align(rt).min(frames);
    while done < frames {
        let m = (frames - done).min(per);
        let t = &mut tmp[..m * ch];
        from_le(&bytes[done * ch * 4..(done + m) * ch * 4], t);
        k.append(t);
        done += m;
    }
}
