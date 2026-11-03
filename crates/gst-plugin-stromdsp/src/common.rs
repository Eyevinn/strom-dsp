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

/// A configured DSP processor plus the scratch buffer used to convert between
/// the little-endian byte payload and `f32` samples.
#[derive(Debug)]
pub(crate) struct Engine<P: Processor> {
    pub(crate) dsp: P,
    pub(crate) channels: usize,
    /// Settings generation last applied to `dsp`.
    pub(crate) applied: u64,
    scratch: Vec<f32>,
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
            scratch: vec![0.0; frames * channels],
        }
    }

    /// Process every complete frame in `data` (F32LE interleaved) in place.
    /// Trailing bytes that do not form a whole frame are left untouched.
    ///
    /// The payload is converted chunk-wise through a scratch buffer allocated in
    /// [`Engine::new`]: this keeps the code free of `unsafe` and of alignment
    /// assumptions on the buffer memory, is endian-correct, and never allocates.
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
            self.dsp.process(samples);
            for (b, s) in chunk.chunks_exact_mut(4).zip(samples.iter()) {
                b.copy_from_slice(&s.to_le_bytes());
            }
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
