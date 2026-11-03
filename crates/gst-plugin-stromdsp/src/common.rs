//! Shared plumbing for the four elements: caps, settings hand-over between the
//! application thread and the streaming thread, and the per-buffer DSP driver.
//!
//! # Settings hand-over and locking
//!
//! Property setters (application thread) write into a [`SharedSettings`]: a
//! plain-old-data settings struct behind a `Mutex`, plus an atomic generation
//! counter that is bumped while the mutex is held. The streaming thread keeps
//! the generation it last applied. Once per buffer it compares that with the
//! atomic counter (a single relaxed-cost atomic load when nothing changed). Only
//! when the counter moved does it take the mutex, and then with `try_lock`: if a
//! property setter happens to hold the lock at that instant, the streaming thread
//! keeps the previous parameters for this buffer and picks the new ones up on the
//! next buffer, so it never blocks on the application thread. The copy is a
//! `Copy` struct, so nothing is allocated.
//!
//! The DSP processor itself lives in a second `Mutex` ([`Engine`] inside an
//! `Option`) that is used almost only by the streaming thread (`set_caps`,
//! `transform_ip`, `FLUSH_STOP`); `stop()` takes it once at shutdown. It is
//! therefore practically uncontended.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use gst::glib;
use gst::prelude::*;
use gst_base::prelude::*;
use strom_dsp::Processor;

/// Caps accepted and produced by every element: interleaved F32LE, any rate,
/// any channel count.
pub(crate) fn audio_caps() -> gst::Caps {
    gst_audio::AudioCapsBuilder::new_interleaved()
        .format(gst_audio::AudioFormat::F32le)
        .rate_range(1..=i32::MAX)
        .channels_range(1..=i32::MAX)
        .build()
}

/// Always-present sink and src pad templates with [`audio_caps`].
pub(crate) fn pad_templates() -> Vec<gst::PadTemplate> {
    let caps = audio_caps();
    let src = gst::PadTemplate::new(
        "src",
        gst::PadDirection::Src,
        gst::PadPresence::Always,
        &caps,
    )
    .unwrap();
    let sink = gst::PadTemplate::new(
        "sink",
        gst::PadDirection::Sink,
        gst::PadPresence::Always,
        &caps,
    )
    .unwrap();
    vec![src, sink]
}

/// Settings written by property setters and read by the streaming thread.
/// See the module documentation for the locking scheme.
#[derive(Debug, Default)]
pub(crate) struct SharedSettings<T: Copy> {
    settings: Mutex<T>,
    generation: AtomicU64,
    reset_requested: AtomicBool,
}

impl<T: Copy> SharedSettings<T> {
    /// Copy of the current settings (application thread).
    pub(crate) fn get(&self) -> T {
        *self.settings.lock().unwrap()
    }

    /// Modify the settings and mark them as changed. Returns the closure's result.
    pub(crate) fn update<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        let mut s = self.settings.lock().unwrap();
        let r = f(&mut s);
        self.generation.fetch_add(1, Ordering::Release);
        r
    }

    /// Current settings together with their generation (blocking lock).
    pub(crate) fn snapshot(&self) -> (T, u64) {
        let s = self.settings.lock().unwrap();
        (*s, self.generation.load(Ordering::Acquire))
    }

    /// Streaming thread: if the settings changed since generation `*applied`,
    /// return a copy and advance `*applied`. Never blocks (uses `try_lock`); on
    /// contention returns `None` and the change is picked up on the next call.
    #[inline]
    pub(crate) fn fetch_if_changed(&self, applied: &mut u64) -> Option<T> {
        if self.generation.load(Ordering::Acquire) == *applied {
            return None;
        }
        let guard = self.settings.try_lock().ok()?;
        // Read the generation under the lock: it cannot move while we hold it.
        *applied = self.generation.load(Ordering::Acquire);
        Some(*guard)
    }

    /// Ask the streaming thread to clear the DSP state before the next buffer
    /// it processes (used when leaving passthrough, where state went stale).
    pub(crate) fn request_reset(&self) {
        self.reset_requested.store(true, Ordering::Release);
    }

    /// Streaming thread: consume a pending reset request.
    #[inline]
    pub(crate) fn take_reset(&self) -> bool {
        self.reset_requested.swap(false, Ordering::AcqRel)
    }
}

/// Number of interleaved samples converted per chunk (see [`Engine::process_bytes`]).
const SCRATCH_SAMPLES: usize = 4096;

/// Length (ms) of the `enabled` crossfade between processed and dry signal.
///
/// Calibrated with the knob test (`tests/knobs.rs`): a raised-cosine fade of
/// a full-scale swing on a 1 kHz tone leaves an annihilator residual of about
/// −61 dB at 10 ms and −67 dB at 20 ms. 20 ms keeps a +12 dB EQ boost (a
/// three-fold amplitude swing) and a 7 kHz de-esser band below −50 dB, and is
/// still an immediate reaction for an operator. Same length as the delay
/// element's crossfade.
pub(crate) const BYPASS_FADE_MS: f64 = 20.0;

/// Warm-up (ms) when leaving full bypass: the processor is reset and runs on
/// the live input for this long, while the output stays dry, before the
/// crossfade to its output starts. Filters settle and detectors catch up with
/// the current level, so the crossfade starts from valid state.
pub(crate) const BYPASS_WARMUP_MS: f64 = 10.0;

/// Click-free `enabled` switch: a raised-cosine crossfade between the
/// processor's output ("wet") and its input ("dry").
///
/// The wet weight is `w = (1 − cos(π·k/L))/2` for fade position `k = 0 … L`
/// ([`BYPASS_FADE_MS`]): zero slope at both ends, so the gain change has no
/// kink. Disabling moves `k` down to 0 (then the element switches to
/// BaseTransform passthrough, bit-exact); enabling moves it up to `L` (then
/// the output is the processor's output, bit-exact, with no mixing cost).
/// A toggle during a running fade reverses it from where it is.
///
/// Leaving full bypass ([`BypassFade::set_enabled`] returns true) needs a
/// processor reset: its state is from whenever it was bypassed, arbitrarily
/// long ago. The processor then runs for [`BYPASS_WARMUP_MS`] with the output
/// still dry, and only then fades in, so it is never faded in from a stale
/// or a just-reset state.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BypassFade {
    len: u32,
    warmup_len: u32,
    /// Fade position `k`, `0 ..= len`.
    pos: u32,
    /// Remaining warm-up frames (only while enabling from full bypass).
    warm: u32,
    enabled: bool,
}

impl Default for BypassFade {
    /// Enabled, at 48 kHz (replaced when the format is known).
    fn default() -> Self {
        Self::new(48_000, true)
    }
}

impl BypassFade {
    /// Settled at `enabled` (no fade running).
    pub(crate) fn new(rate: u32, enabled: bool) -> Self {
        let frames = |ms: f64| ((ms * 0.001 * rate as f64).round() as u32).max(1);
        let len = frames(BYPASS_FADE_MS);
        Self {
            len,
            warmup_len: frames(BYPASS_WARMUP_MS),
            pos: if enabled { len } else { 0 },
            warm: 0,
            enabled,
        }
    }

    /// Set the target state. Returns true when the processor leaves full
    /// bypass and must be reset (its state is stale).
    pub(crate) fn set_enabled(&mut self, enabled: bool) -> bool {
        if enabled == self.enabled {
            return false;
        }
        self.enabled = enabled;
        self.warm = 0;
        if enabled && self.pos == 0 {
            self.warm = self.warmup_len;
            return true;
        }
        false
    }

    /// Disabled and fully faded out: the output is the dry input.
    #[inline]
    pub(crate) fn is_bypassed(&self) -> bool {
        !self.enabled && self.pos == 0
    }

    /// Enabled and fully faded in: the output is the processor's output.
    #[inline]
    pub(crate) fn is_wet(&self) -> bool {
        self.enabled && self.pos == self.len
    }

    /// Count `frames` of warm-up that were output dry by the caller.
    pub(crate) fn advance_dry(&mut self, frames: usize) {
        if self.enabled {
            self.warm = self
                .warm
                .saturating_sub(frames.min(u32::MAX as usize) as u32);
        }
    }

    /// Mix one chunk in place: `wet` (the processor's output) becomes the
    /// output, `dry` is the unprocessed input. Advances the fade per frame.
    #[inline]
    pub(crate) fn mix(&mut self, wet: &mut [f32], dry: &[f32], channels: usize) {
        let inv_len = std::f64::consts::PI / self.len as f64;
        for (w, d) in wet
            .chunks_exact_mut(channels)
            .zip(dry.chunks_exact(channels))
        {
            if self.enabled {
                if self.warm > 0 {
                    self.warm -= 1;
                } else if self.pos < self.len {
                    self.pos += 1;
                }
            } else if self.pos > 0 {
                self.pos -= 1;
            }
            if self.pos == 0 {
                w.copy_from_slice(d);
            } else if self.pos < self.len {
                let g = (0.5 - 0.5 * (self.pos as f64 * inv_len).cos()) as f32;
                for (a, &b) in w.iter_mut().zip(d) {
                    *a = b + g * (*a - b);
                }
            }
        }
    }
}

/// Settings of an element with an `enabled` property handled by [`BypassFade`].
pub(crate) trait Bypassable {
    fn enabled(&self) -> bool;
}

/// A configured DSP processor plus the scratch buffers used to convert between
/// the little-endian byte payload and `f32` samples, and the `enabled`
/// crossfade state.
#[derive(Debug)]
pub(crate) struct Engine<P: Processor> {
    pub(crate) dsp: P,
    pub(crate) channels: usize,
    /// Settings generation last applied to `dsp`.
    pub(crate) applied: u64,
    /// `enabled` crossfade (always wet for elements that do not use it).
    pub(crate) fade: BypassFade,
    scratch: Vec<f32>,
    /// Copy of the input while a crossfade runs.
    dry: Vec<f32>,
}

impl<P: Processor> Engine<P> {
    /// Configure `dsp` for `rate`/`channels`. Allocates (only called from `set_caps`).
    pub(crate) fn new(mut dsp: P, rate: u32, channels: usize) -> Self {
        dsp.configure(rate as f32, channels);
        let frames = (SCRATCH_SAMPLES / channels).max(1);
        Self {
            dsp,
            channels,
            applied: u64::MAX,
            fade: BypassFade::new(rate, true),
            scratch: vec![0.0; frames * channels],
            dry: vec![0.0; frames * channels],
        }
    }

    /// Process every complete frame in `data` (F32LE interleaved) in place.
    /// Trailing bytes that do not form a whole frame are left untouched.
    ///
    /// The payload is converted chunk-wise through a scratch buffer allocated in
    /// [`Engine::new`]: this keeps the code free of `unsafe` and of alignment
    /// assumptions on the buffer memory, is endian-correct, and never allocates.
    /// While the `enabled` crossfade runs, the input is kept in a second
    /// scratch buffer and mixed with the processor's output ([`BypassFade::mix`]).
    #[inline]
    pub(crate) fn process_bytes(&mut self, data: &mut [u8]) {
        let frame_bytes = self.channels * 4;
        let usable = data.len() / frame_bytes * frame_bytes;
        let chunk_bytes = self.scratch.len() * 4;
        for chunk in data[..usable].chunks_mut(chunk_bytes) {
            let samples = &mut self.scratch[..chunk.len() / 4];
            for (s, b) in samples.iter_mut().zip(chunk.chunks_exact(4)) {
                *s = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
            if self.fade.is_wet() {
                self.dsp.process(samples);
            } else {
                let dry = &mut self.dry[..samples.len()];
                dry.copy_from_slice(samples);
                self.dsp.process(samples);
                self.fade.mix(samples, dry, self.channels);
            }
            for (b, s) in chunk.chunks_exact_mut(4).zip(samples.iter()) {
                b.copy_from_slice(&s.to_le_bytes());
            }
        }
    }

    /// Run the processor on a copy of `data` (warm-up while the buffer passes
    /// through untouched).
    fn warm_up_bytes(&mut self, data: &[u8]) {
        let frame_bytes = self.channels * 4;
        let usable = data.len() / frame_bytes * frame_bytes;
        let chunk_bytes = self.scratch.len() * 4;
        for chunk in data[..usable].chunks(chunk_bytes) {
            let samples = &mut self.scratch[..chunk.len() / 4];
            for (s, b) in samples.iter_mut().zip(chunk.chunks_exact(4)) {
                *s = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
            self.dsp.process(samples);
            self.fade.advance_dry(samples.len() / self.channels);
        }
    }
}

/// Parse negotiated caps into (rate, channels).
///
/// Reads the fields directly instead of going through `AudioInfo`, which
/// rejects more than two channels without a `channel-mask`; the elements do not
/// care about channel positions (all channels are processed alike).
pub(crate) fn parse_caps(caps: &gst::Caps) -> Result<(u32, usize), gst::LoggableError> {
    let s = caps
        .structure(0)
        .ok_or_else(|| gst::loggable_error!(gst::CAT_RUST, "Empty caps"))?;
    let rate = s.get::<i32>("rate").ok().filter(|&r| r > 0);
    let channels = s.get::<i32>("channels").ok().filter(|&c| c > 0);
    let format_ok = s.get::<&str>("format").ok() == Some("F32LE");
    match (rate, channels, format_ok) {
        (Some(r), Some(c), true) => Ok((r as u32, c as usize)),
        _ => Err(gst::loggable_error!(
            gst::CAT_RUST,
            "Unsupported caps {caps}"
        )),
    }
}

/// Let attached GstController bindings update the properties for this buffer
/// (this is what makes the properties "controllable"). A cheap no-op when no
/// control bindings are attached.
pub(crate) fn sync_controlled_values<O: IsA<gst::Object> + IsA<gst_base::BaseTransform>>(
    obj: &O,
    buf: &gst::BufferRef,
) {
    let Some(pts) = buf.pts() else {
        return;
    };
    let segment = obj.upcast_ref::<gst_base::BaseTransform>().segment();
    let Some(segment) = segment.downcast_ref::<gst::ClockTime>() else {
        return;
    };
    if let Some(stream_time) = segment.to_stream_time(pts) {
        let _ = obj.upcast_ref::<gst::Object>().sync_values(stream_time);
    }
}

/// Shorthand for a readable, writable, controllable, mutable-in-PLAYING `f32` property.
pub(crate) fn float_prop(
    name: &str,
    nick: &str,
    blurb: &str,
    min: f32,
    max: f32,
    default: f32,
) -> glib::ParamSpec {
    glib::ParamSpecFloat::builder(name)
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

/// Shorthand for a readable, writable, controllable, mutable-in-PLAYING boolean property.
pub(crate) fn bool_prop(name: &str, nick: &str, blurb: &str, default: bool) -> glib::ParamSpec {
    glib::ParamSpecBoolean::builder(name)
        .nick(nick)
        .blurb(blurb)
        .default_value(default)
        .readwrite()
        .controllable()
        .mutable_playing()
        .build()
}

/// DSP slot of an element: `None` until caps are negotiated.
pub(crate) type EngineSlot<P> = Mutex<Option<Engine<P>>>;

/// `set_caps` helper: (re)configure the processor for new caps. Reuses the
/// existing processor object if there is one, and applies the current settings
/// (blocking lock: `set_caps` is not on the per-buffer path).
pub(crate) fn configure_engine<P: Processor + Default, S: Copy>(
    slot: &EngineSlot<P>,
    settings: &SharedSettings<S>,
    rate: u32,
    channels: usize,
    apply: impl FnOnce(&mut P, &S),
) {
    let mut guard = slot.lock().unwrap();
    let dsp = guard.take().map(|e| e.dsp).unwrap_or_default();
    let mut engine = Engine::new(dsp, rate, channels);
    let (s, generation) = settings.snapshot();
    apply(&mut engine.dsp, &s);
    engine.applied = generation;
    // State is fresh after `configure`; a pending reset request is moot.
    settings.take_reset();
    *guard = Some(engine);
}

// ---------------------------------------------------------------------------
// `enabled` with a crossfade, for the BaseTransform passthrough elements.
//
// Steady states: enabled → the processor's output (no mixing); disabled →
// BaseTransform passthrough, bit-exact. In between, the
// streaming thread crossfades ([`BypassFade`]) and owns the passthrough flag:
//
// - disabling: `transform_ip` keeps running and fades out; when the fade
//   reaches dry it sets passthrough itself;
// - enabling: the element is in passthrough, so it gets the buffers in
//   `transform_ip_passthrough` (TRANSFORM_IP_ON_PASSTHROUGH = true), which
//   sees the change, resets the processor, warms it up on a copy of the
//   (untouched) buffer and leaves passthrough; `transform_ip` fades in.
//
// The application thread sets passthrough directly only while no stream is
// configured (no engine), so `enabled=false` maps to passthrough at once on an
// idle element. It never calls into BaseTransform while streaming, so a
// controller-driven `enabled` (set from the streaming thread with the object
// lock held) cannot deadlock.
// ---------------------------------------------------------------------------

/// `set_property("enabled")` helper: on an idle element (no stream
/// configured) apply passthrough at once; while streaming the streaming
/// thread fades and switches (see above). Never blocks.
pub(crate) fn set_enabled_idle<P: Processor>(
    obj: &impl IsA<gst_base::BaseTransform>,
    slot: &EngineSlot<P>,
    enabled: bool,
) {
    let idle = matches!(slot.try_lock(), Ok(g) if g.is_none());
    if idle {
        obj.set_passthrough(!enabled);
    }
}

/// `set_caps` helper for elements with a crossfaded `enabled`: like
/// [`configure_engine`], and the fade starts settled at the current setting.
pub(crate) fn configure_engine_bypass<P: Processor + Default, S: Copy + Bypassable>(
    obj: &impl IsA<gst_base::BaseTransform>,
    slot: &EngineSlot<P>,
    settings: &SharedSettings<S>,
    rate: u32,
    channels: usize,
    apply: impl FnOnce(&mut P, &S),
) {
    configure_engine(slot, settings, rate, channels, apply);
    let enabled = settings.get().enabled();
    if let Some(e) = slot.lock().unwrap().as_mut() {
        e.fade = BypassFade::new(rate, enabled);
    }
    obj.set_passthrough(!enabled);
}

/// Apply changed settings, and the `enabled` target to the fade (resetting
/// the processor when it leaves full bypass); honour reset requests/DISCONT.
#[inline]
fn update_bypass_engine<P: Processor, S: Copy + Bypassable>(
    engine: &mut Engine<P>,
    settings: &SharedSettings<S>,
    discont: bool,
    apply: impl FnOnce(&mut P, &S),
) {
    if let Some(s) = settings.fetch_if_changed(&mut engine.applied) {
        apply(&mut engine.dsp, &s);
        if engine.fade.set_enabled(s.enabled()) {
            engine.dsp.reset();
        }
    }
    if settings.take_reset() || discont {
        engine.dsp.reset();
    }
}

/// `transform_ip` helper for elements with a crossfaded `enabled`: like
/// [`transform_buffer`], plus the fade, and the switch to passthrough once a
/// fade-out is complete. No allocation, no logging.
#[inline]
pub(crate) fn transform_buffer_bypass<P: Processor, S: Copy + Bypassable>(
    obj: &impl IsA<gst_base::BaseTransform>,
    slot: &EngineSlot<P>,
    settings: &SharedSettings<S>,
    buf: &mut gst::BufferRef,
    apply: impl FnOnce(&mut P, &S),
) -> Result<gst::FlowSuccess, gst::FlowError> {
    let mut guard = slot.lock().unwrap();
    let engine = guard.as_mut().ok_or(gst::FlowError::NotNegotiated)?;
    let discont = buf.flags().contains(gst::BufferFlags::DISCONT);
    update_bypass_engine(engine, settings, discont, apply);
    let mut map = buf.map_writable().map_err(|_| gst::FlowError::Error)?;
    engine.process_bytes(map.as_mut_slice());
    if engine.fade.is_bypassed() {
        // Faded out: from the next buffer on, bit-exact passthrough.
        obj.set_passthrough(true);
    }
    Ok(gst::FlowSuccess::Ok)
}

/// `transform_ip_passthrough` helper for elements with a crossfaded
/// `enabled`: the buffer always passes untouched. When `enabled` became true,
/// reset the processor, warm it up on this buffer and leave passthrough (the
/// fade-in runs in [`transform_buffer_bypass`]). Otherwise only a generation
/// check. No allocation, no logging.
#[inline]
pub(crate) fn transform_passthrough_bypass<P: Processor, S: Copy + Bypassable>(
    obj: &impl IsA<gst_base::BaseTransform>,
    slot: &EngineSlot<P>,
    settings: &SharedSettings<S>,
    buf: &gst::Buffer,
    apply: impl FnOnce(&mut P, &S),
) -> Result<gst::FlowSuccess, gst::FlowError> {
    let mut guard = slot.lock().unwrap();
    let engine = guard.as_mut().ok_or(gst::FlowError::NotNegotiated)?;
    let discont = buf.flags().contains(gst::BufferFlags::DISCONT);
    update_bypass_engine(engine, settings, discont, apply);
    if !engine.fade.is_bypassed() {
        let map = buf.map_readable().map_err(|_| gst::FlowError::Error)?;
        engine.warm_up_bytes(map.as_slice());
        obj.set_passthrough(false);
    }
    Ok(gst::FlowSuccess::Ok)
}

/// Clear the processor state (FLUSH_STOP).
pub(crate) fn reset_engine<P: Processor>(slot: &EngineSlot<P>) {
    if let Some(engine) = slot.lock().unwrap().as_mut() {
        engine.dsp.reset();
    }
}

/// `transform_ip` helper: apply changed settings, honour reset requests and
/// DISCONT, then process the buffer in place. No allocation, no logging.
#[inline]
pub(crate) fn transform_buffer<P: Processor, S: Copy>(
    slot: &EngineSlot<P>,
    settings: &SharedSettings<S>,
    buf: &mut gst::BufferRef,
    apply: impl FnOnce(&mut P, &S),
) -> Result<gst::FlowSuccess, gst::FlowError> {
    let mut guard = slot.lock().unwrap();
    let engine = guard.as_mut().ok_or(gst::FlowError::NotNegotiated)?;
    if let Some(s) = settings.fetch_if_changed(&mut engine.applied) {
        apply(&mut engine.dsp, &s);
    }
    let discont = buf.flags().contains(gst::BufferFlags::DISCONT);
    if settings.take_reset() || discont {
        engine.dsp.reset();
    }
    let mut map = buf.map_writable().map_err(|_| gst::FlowError::Error)?;
    engine.process_bytes(map.as_mut_slice());
    Ok(gst::FlowSuccess::Ok)
}

/// Tests of the plumbing with a fake processor, so they are meaningful
/// independently of the real `strom-dsp` processors.
#[cfg(test)]
mod tests {
    use super::*;

    /// Fake processor: multiplies by `gain`, adds a running sample counter
    /// (stateful, so chunking/reset mistakes show up), counts resets.
    #[derive(Debug, Default)]
    struct Fake {
        channels: usize,
        gain: f32,
        counter: f32,
        resets: u32,
        configured: u32,
    }

    impl Processor for Fake {
        fn configure(&mut self, _rate: f32, channels: usize) {
            self.channels = channels;
            self.counter = 0.0;
            self.configured += 1;
        }
        fn process(&mut self, samples: &mut [f32]) {
            assert_eq!(samples.len() % self.channels, 0, "partial frame");
            for frame in samples.chunks_exact_mut(self.channels) {
                for s in frame.iter_mut() {
                    *s = *s * self.gain + self.counter;
                }
                self.counter += 1.0;
            }
        }
        fn reset(&mut self) {
            self.counter = 0.0;
            self.resets += 1;
        }
    }

    fn apply(p: &mut Fake, gain: &f32) {
        p.gain = *gain;
    }

    fn to_bytes(s: &[f32]) -> Vec<u8> {
        s.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn from_bytes(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn bypass_fade_raised_cosine_warm_up_and_reversal() {
        let rate = 48_000;
        let len = (BYPASS_FADE_MS * 48.0) as usize;
        let warm = (BYPASS_WARMUP_MS * 48.0) as usize;
        let mut f = BypassFade::new(rate, true);
        assert!(f.is_wet() && !f.is_bypassed());
        // wet = 1, dry = 0 on one channel: the output is the wet weight.
        let run = |f: &mut BypassFade, n: usize| -> Vec<f32> {
            let mut wet = vec![1.0f32; n];
            f.mix(&mut wet, &vec![0.0; n], 1);
            wet
        };
        assert!(!f.set_enabled(false), "fading out needs no reset");
        let out = run(&mut f, len + 10);
        for (k, v) in out.iter().enumerate().take(len) {
            let pos = (len - 1 - k) as f64;
            let w = 0.5 - 0.5 * (std::f64::consts::PI * pos / len as f64).cos();
            assert!((*v as f64 - w).abs() < 1e-6, "{k}: {v} vs {w}");
        }
        // Fully dry, bit-exact, from the end of the fade on.
        assert!(out[len - 1..].iter().all(|&v| v == 0.0));
        assert!(f.is_bypassed());
        // Leaving full bypass: reset requested, then warm-up (dry), then fade in.
        assert!(f.set_enabled(true));
        let out = run(&mut f, warm + len + 10);
        assert!(out[..warm].iter().all(|&v| v == 0.0), "dry during warm-up");
        assert!(out[warm] > 0.0 && out[warm] < 1e-4, "smooth start");
        assert!(
            out[warm + len - 1..].iter().all(|&v| v == 1.0),
            "bit-exact wet"
        );
        assert!(f.is_wet());
        // Reversal mid-fade continues from where it is: no jump.
        f.set_enabled(false);
        let a = run(&mut f, len / 2);
        assert!(!f.set_enabled(true), "not bypassed: no reset");
        let b = run(&mut f, 1);
        assert!((a[a.len() - 1] - b[0]).abs() < 2e-3);
        // advance_dry only counts warm-up.
        let mut g = BypassFade::new(rate, false);
        assert!(g.set_enabled(true));
        g.advance_dry(warm);
        assert!(run(&mut g, 1)[0] > 0.0);
    }

    #[test]
    fn shared_settings_generation() {
        let s = SharedSettings::<f32>::default();
        let mut applied = u64::MAX;
        assert_eq!(s.fetch_if_changed(&mut applied), Some(0.0));
        assert_eq!(s.fetch_if_changed(&mut applied), None);
        s.update(|v| *v = 2.0);
        s.update(|v| *v = 3.0);
        assert_eq!(s.fetch_if_changed(&mut applied), Some(3.0));
        assert_eq!(s.fetch_if_changed(&mut applied), None);

        // Contended: the streaming side does not block and retries later.
        s.update(|v| *v = 4.0);
        {
            let _held = s.settings.lock().unwrap();
            assert_eq!(s.fetch_if_changed(&mut applied), None);
        }
        assert_eq!(s.fetch_if_changed(&mut applied), Some(4.0));

        assert!(!s.take_reset());
        s.request_reset();
        assert!(s.take_reset());
        assert!(!s.take_reset());
    }

    #[test]
    fn process_bytes_matches_direct_processing() {
        for channels in [1usize, 2, 3, 7, 5000] {
            let frames = 3001;
            let input: Vec<f32> = (0..frames * channels)
                .map(|i| (i as f32 * 0.37).sin())
                .collect();
            let mut direct = Fake::default();
            direct.configure(48_000.0, channels);
            direct.gain = 0.5;
            let mut expected = input.clone();
            direct.process(&mut expected);

            let mut engine = Engine::new(Fake::default(), 48_000, channels);
            engine.dsp.gain = 0.5;
            let mut bytes = to_bytes(&input);
            // A trailing partial frame is left untouched.
            bytes.extend_from_slice(&[1, 2, 3]);
            engine.process_bytes(&mut bytes);
            assert_eq!(&bytes[bytes.len() - 3..], &[1, 2, 3]);
            let out = from_bytes(&bytes[..bytes.len() - 3]);
            assert_eq!(out, expected, "channels {channels}");
        }
    }

    #[test]
    fn transform_applies_settings_and_resets() {
        gst::init().unwrap();
        let settings = SharedSettings::<f32>::default();
        settings.update(|g| *g = 2.0);
        let slot: EngineSlot<Fake> = Mutex::new(None);

        let mut buf = gst::Buffer::from_mut_slice(to_bytes(&[1.0, 1.0]));
        assert_eq!(
            transform_buffer(&slot, &settings, buf.get_mut().unwrap(), apply),
            Err(gst::FlowError::NotNegotiated)
        );

        configure_engine(&slot, &settings, 48_000, 1, apply);
        let run = |samples: &[f32], flags: gst::BufferFlags| {
            let mut buf = gst::Buffer::from_mut_slice(to_bytes(samples));
            buf.get_mut().unwrap().set_flags(flags);
            transform_buffer(&slot, &settings, buf.get_mut().unwrap(), apply).unwrap();
            let map = buf.map_readable().unwrap();
            from_bytes(&map)
        };
        // Gain applied at configure time; counter runs on.
        assert_eq!(run(&[1.0, 1.0], gst::BufferFlags::empty()), [2.0, 3.0]);
        settings.update(|g| *g = 10.0);
        assert_eq!(run(&[1.0], gst::BufferFlags::empty()), [12.0]);
        // DISCONT resets state before processing.
        assert_eq!(run(&[1.0], gst::BufferFlags::DISCONT), [10.0]);
        // A reset request does too.
        settings.request_reset();
        assert_eq!(run(&[1.0, 0.0], gst::BufferFlags::empty()), [10.0, 1.0]);
        // FLUSH_STOP path.
        reset_engine(&slot);
        assert_eq!(run(&[0.0], gst::BufferFlags::empty()), [0.0]);
        // Renegotiation reconfigures the same processor and re-applies settings.
        configure_engine(&slot, &settings, 44_100, 2, apply);
        let e = slot.lock().unwrap();
        let e = e.as_ref().unwrap();
        assert_eq!(e.dsp.configured, 2);
        assert_eq!(e.dsp.channels, 2);
        assert_eq!(e.dsp.gain, 10.0);
    }
}
