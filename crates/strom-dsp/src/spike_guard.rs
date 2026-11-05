//! Spike guard: removes short, isolated peaks (clicks, a microphone tick, a
//! spike from a bad link) so the gain control after it does not react to
//! them.
//!
//! # Signal flow
//!
//! ```text
//! input ──┬── delay D ───────────────────────────────────── × gain ──→ output
//!         │                                                   ▲
//!         └─ 4× up ─ peak per frame and channel ─ event detector ─ gain ring (La ahead)
//!                       │                           ▲
//!                       └─ delayed envelope ────────┘ (non-event frames only)
//! ```
//!
//! Each channel is handled on its own (a click on one channel leaves the
//! other untouched).
//!
//! - **Detector.** The input is interpolated 4× ([`Oversampler`]). Per base
//!   frame and channel, `p` is the largest `|x|` over the 4 sub-samples (the
//!   inter-sample peak). A peak envelope `e` ([`PeakEnvelope`]: instant
//!   attack, [`ENVELOPE_RELEASE_MS`] release) follows `p` delayed by
//!   [`ENVELOPE_DELAY_MS`], leaving out the frames of events, the frames up
//!   to that delay before an event (the interpolator rings ahead of a large
//!   click) and the ramp after one, so the programme level it tracks is not
//!   raised by the spikes it judges. It counts as at least
//!   [`ENVELOPE_FLOOR`] (−60 dBFS).
//! - **Rise gate.** With `T` the threshold (dB) as a factor, the cap is
//!   `L = e·T`. A frame with `p > L` is an excursion; it starts an event
//!   only if it looks like a click in the base-rate input that the
//!   interpolator's delay already buffers (the 24 frames from the frame
//!   the detector sees, `n`, to the newest; no added latency). With `j`
//!   the frame of the largest `|x|` from `n − 1` to the end of the longest
//!   core ([`EVENT_MAX_MS`]):
//!   1. it rises from the programme level to its peak within `R` samples:
//!      for some `1 ≤ r ≤ R` (the smallest), `s = j − r` and the sample
//!      before it are at or under the cap ([`RISE_FROM_PROGRAMME_FRAMES`],
//!      2, samples at the programme level), and the samples between rise
//!      in one stroke: on the peak's side, each at least as large as the
//!      one before (a swell on a high tone alternates in sign on its way
//!      up, and a bipolar click's first lobe is left to the limiter). `R` is [`RISE_MAX_FRAMES_48K`] (3) at 48 kHz, scaled with
//!      the sample rate (`max(1, round(3·fs/48000))`: 3 at 44.1 kHz, 6 at
//!      96 kHz, 12 at 192 kHz), so a click is the same length in time at
//!      every rate;
//!   2. the rise `|x[j] − x[s]|` exceeds the cap, by [`RISE_OVER_CAP_DB`]
//!      (0 dB), once per 48 kHz sample it takes (`r·48000/fs` times, at
//!      least once): steep all the way, a click and not a swell. With
//!      `r = 1` at 48 kHz this is the one-sample step of a single-sample
//!      click. (A plain rise over the cap within three samples, without
//!      the per-sample steepness, also took speech onsets in the corpus
//!      crackle check: up to +19 dB worst-frame NMR in the low-latency
//!      mode.)
//!   3. nothing above its core level comes back in the isolation window
//!      ([`EVENT_MAX_MS`] … [`EVENT_MAX_MS`] + [`ISOLATION_MS`] after `n`).
//!      Up to 48 kHz the buffer holds all of it. Above, where it reaches
//!      only 0.25 ms, the window starts after the click's own fall (its
//!      last sample over the core level within `R` samples after the peak
//!      `j`: a click may fall as long as it may rise) and runs to the
//!      newest buffered frame, so it never covers the click itself; fewer
//!      than 7 frames there refuse the excursion.
//!
//!   Speech, tones, noise bursts and drum onsets build up over more than
//!   `R` samples or recur, so they fail it; a refused excursion's peak goes
//!   into the envelope at once, so the rest of it is judged against
//!   itself. A click that peaks over one to three samples (at 48 kHz)
//!   passes.
//! - **Event (causal).** The knee is `a = L·K` (`K` = −[`KNEE_DB`]). An
//!   excursion that passes the rise gate starts an event at once; `L` and
//!   `a` are frozen for it.
//!   The event's **core** is its frames above `max(a, P·C)`, with `P` its
//!   largest peak so far and `C` = [`CORE_DB`] (−20 dB, which leaves out
//!   the interpolator ringing of a large click). The event is
//!   - **abandoned** (programme: a held note, a shout, a drum onset, a
//!     noise burst) when a core frame comes more than [`EVENT_MAX_MS`]
//!     after its start, before it is confirmed. The envelope then takes its
//!     peak at once, and the dip is lifted;
//!   - **confirmed** as a spike once [`ISOLATION_MS`] have passed after
//!     [`EVENT_MAX_MS`] without that. It then **ends** (counted) when the
//!     core has stayed away for [`GAP_MS`], or when a new core frame
//!     arrives, which is judged as a new excursion.
//! - **Soft knee.** The gain is `g = f(P)/P` with the static curve
//!   `f(P) = a + (L − a)·tanh((P − a)/(L − a))`: identity up to `a`,
//!   continuous in value, slope and curvature there, never above `L`. A
//!   larger peak later in the event lowers `g` from there on.
//! - **Band-limited gain with lookahead.** The gain is written into a ring
//!   that the output reads `La` frames ([`SpikeGuard::set_lookahead_ms`])
//!   behind the detector. A dip starts with a raised-cosine ramp before the
//!   peak of `La − (EVENT_MAX_MS + ISOLATION_MS)` (at most [`RAMP_MS`]),
//!   holds through the event, and ends with a [`RAMP_MS`] ramp (look-ahead,
//!   band-limited gain control: US 5,737,434, expired). Overlapping dips
//!   take the lower gain. Outside dips the gain is exactly 1 and the input
//!   passes untouched: the output is **bit-exact** with the input delayed
//!   by [`SpikeGuard::latency_frames`].
//! - **Abandoned events.** An event is abandoned at most `EVENT_MAX_MS +
//!   ISOLATION_MS` (0.4 ms) after it starts. With the standard 0.5 ms
//!   lookahead the output has not read any of its dip by then, so the dip
//!   is lifted completely. With less lookahead the part already read
//!   stays, so the rise gate is what keeps programme untouched there: it
//!   refuses an excursion that will be abandoned before an event starts.
//!   Measured with no lookahead: a programme of chords, dense drums with
//!   stick transients and sudden tones is bit-exact at 44.1, 48 and
//!   96 kHz (without the rise gate 430 samples were cut, by up to
//!   22.8 dB), and corpus speech through the broadcast chain measures the
//!   same with the guard as without it.
//!
//! **Why a gain dip and not a waveform clipper.** A clipper at the 4× rate
//! whose clippings are band-limited back to the base rate (US 6,337,999,
//! expired) leaves most of a single-sample click: the band-limited
//! clippings of such a narrow pulse are much smaller than the clippings
//! themselves, so the peak grows back. Measured with the 4× oversampler and
//! the same soft clipper at an 8 dB cap: a 1.0 click on a 440 Hz tone of
//! 0.1 came out at 0.40 (cap 0.25), a 10.0 click on a tone of 0.05 at 0.87
//! (cap 0.13). The gain dip scales the click as a whole, so its
//! inter-sample peak lands at `f(P)`, and it creates no harmonics: a smooth
//! gain times a band-limited signal stays band-limited, so nothing aliases.
//!
//! Basis: impulsive-noise detection against a signal envelope (Godsill &
//! Rayner, *Digital Audio Restoration*, 1998), with our own rise gate
//! (rise time and isolation in the buffered base-rate input; see
//! `PROVENANCE.md`); peak control ahead of the
//! gain control so it does not react, with a band-limited gain signal
//! (US 5,737,434, expired); inter-sample peak detection by 4×
//! interpolation (ITU-R BS.1770 Annex 2).
//!
//! # Latency
//!
//! `D = U + La` frames: the 4× interpolator's delay `U = 24`
//! ([`Oversampler`]) and the lookahead `La = round(lookahead · fs)`
//! ([`latency_frames_for`]). Standard ([`STANDARD_LOOKAHEAD_MS`], 0.5 ms):
//! 24 + 24 = 48 frames (1.0 ms) at 48 kHz; low latency
//! ([`LOW_LATENCY_LOOKAHEAD_MS`], 0): 24 frames (0.5 ms); the rise gate
//! keeps programme untouched there (see Abandoned events), and the dip of
//! a click starts without a ramp before it. The lookahead is
//! structural (applied at [`Processor::configure`]); the latency is
//! constant for a configuration and independent of the parameters.
//!
//! # Parameters
//!
//! `threshold_db` changes take effect at the next event (an event keeps
//! its cap), so a change never steps the output. `enabled = false` stops
//! new events (one in progress finishes); the output is then the delayed
//! input, sample for sample.

use crate::dynamics::{DelayLine, PeakEnvelope};
use crate::oversample::{OversampleFactor, Oversampler};
use crate::util::db_to_lin;
use crate::Processor;

/// Longest core that still counts as a spike (ms).
pub const EVENT_MAX_MS: f64 = 0.25;
/// After its core, a spike must stay below its core level this long before
/// it is confirmed (ms); a core frame within it abandons the event.
pub const ISOLATION_MS: f64 = 0.15;
/// A confirmed spike ends when its core has stayed away this long (ms).
pub const GAP_MS: f64 = 0.5;
/// Longest raised-cosine ramp of a gain dip (ms).
pub const RAMP_MS: f64 = 0.5;
/// Delay of the envelope's input behind the detector (ms).
pub const ENVELOPE_DELAY_MS: f64 = 0.33;
/// Release of the programme envelope (ms).
pub const ENVELOPE_RELEASE_MS: f64 = 40.0;
/// Floor of the programme envelope (linear, −60 dBFS).
pub const ENVELOPE_FLOOR: f32 = 1e-3;
/// Width of the soft knee below the cap (dB). The threshold range starts
/// at the knee width, so the knee never starts below the envelope.
pub const KNEE_DB: f32 = 6.0;
/// Extent of an event's core relative to its peak (dB).
pub const CORE_DB: f32 = -20.0;
/// Range of the threshold (dB above the envelope).
pub const THRESHOLD_RANGE_DB: (f32, f32) = (6.0, 30.0);
/// Default threshold (dB above the envelope).
pub const DEFAULT_THRESHOLD_DB: f32 = 8.0;
/// Standard lookahead (ms): 1.0 ms of latency in all at 48 kHz.
pub const STANDARD_LOOKAHEAD_MS: f32 = 0.5;
/// Low-latency lookahead (ms): only the interpolator's 0.5 ms remain.
pub const LOW_LATENCY_LOOKAHEAD_MS: f32 = 0.0;
/// Largest lookahead (ms).
pub const MAX_LOOKAHEAD_MS: f32 = 2.0;

/// Rise gate: the rise into an excursion's base-rate peak, from the last
/// sample at or under the cap, must exceed the cap by this much (dB) for
/// the excursion to start an event.
pub const RISE_OVER_CAP_DB: f32 = 0.0;
/// Rise gate: base-rate samples at the start of that rise that must sit
/// at or under the cap (the excursion rises from the programme level).
pub const RISE_FROM_PROGRAMME_FRAMES: usize = 2;
/// Rise gate: most samples the rise from the programme level to the peak
/// may take at 48 kHz (62.5 µs); scaled with the sample rate.
pub const RISE_MAX_FRAMES_48K: usize = 3;
/// Rise gate: fewest buffered frames the isolation window is checked over
/// (the window's length at 48 kHz).
const PREVIEW_MIN_FRAMES: usize = 7;
/// Rise gate: base-rate frames before the aligned frame searched for the
/// excursion's peak (the interpolator's ringing reaches the detector a
/// frame before the click).
const RISE_BEFORE: usize = 1;

/// Oversampling factor of the detector.
const FACTOR: OversampleFactor = OversampleFactor::X4;
/// Sub-samples per base frame.
const OS: usize = 4;
/// Delay of the 4× interpolator ([`Oversampler::up_latency_frames`]).
const UP_LATENCY: usize = 24;
/// Base frames per internal chunk.
const CHUNK: usize = 256;

/// Parameters of the [`SpikeGuard`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpikeGuardParams {
    /// When false no new event starts; the output is the delayed input.
    pub enabled: bool,
    /// Cap above the programme envelope (dB, 6 … 30).
    pub threshold_db: f32,
}

impl Default for SpikeGuardParams {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold_db: DEFAULT_THRESHOLD_DB,
        }
    }
}

impl SpikeGuardParams {
    /// Copy with the threshold clamped to its range (NaN → default).
    pub fn clamped(&self) -> Self {
        Self {
            enabled: self.enabled,
            threshold_db: if self.threshold_db.is_nan() {
                DEFAULT_THRESHOLD_DB
            } else {
                self.threshold_db
                    .clamp(THRESHOLD_RANGE_DB.0, THRESHOLD_RANGE_DB.1)
            },
        }
    }
}

/// A lookahead clamped to `0 … MAX_LOOKAHEAD_MS` (NaN → standard).
pub fn clamp_lookahead_ms(ms: f32) -> f32 {
    if ms.is_nan() {
        STANDARD_LOOKAHEAD_MS
    } else {
        ms.clamp(0.0, MAX_LOOKAHEAD_MS)
    }
}

fn rate_or_default(sample_rate: f32) -> f64 {
    if sample_rate.is_finite() && sample_rate > 0.0 {
        sample_rate as f64
    } else {
        48_000.0
    }
}

/// Frame counts at one sample rate and lookahead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Timing {
    la: usize,
    /// Ramp before a dip: what the lookahead leaves after the longest event
    /// and its isolation window, so an abandoned event is lifted before the
    /// output reads any of it.
    pre: usize,
    max_len: usize,
    isolation: usize,
    gap: usize,
    ramp: usize,
    env_delay: usize,
    /// Rise gate: most samples from the programme level to the peak.
    rise: usize,
}

impl Timing {
    fn at(sample_rate: f32, lookahead_ms: f32) -> Self {
        let fs = rate_or_default(sample_rate);
        let frames = |ms: f64| (ms * 0.001 * fs).round() as usize;
        let la = frames(clamp_lookahead_ms(lookahead_ms) as f64);
        let max_len = frames(EVENT_MAX_MS).max(1);
        let isolation = frames(ISOLATION_MS).max(1);
        let ramp = frames(RAMP_MS).max(1);
        Self {
            la,
            pre: la.saturating_sub(max_len + isolation + 1).min(ramp),
            max_len,
            isolation,
            gap: frames(GAP_MS).max(1),
            ramp,
            env_delay: frames(ENVELOPE_DELAY_MS).max(1),
            rise: ((RISE_MAX_FRAMES_48K as f64 * fs / 48_000.0).round() as usize).max(1),
        }
    }

    /// Frames in the base-rate input ring: back to the aligned frame minus
    /// [`RISE_BEFORE`], the longest rise, and the sample before its start.
    fn base_len(&self) -> usize {
        UP_LATENCY + RISE_BEFORE + self.rise + RISE_FROM_PROGRAMME_FRAMES
    }
}

/// Latency of the spike guard (frames) at `sample_rate` with a lookahead
/// of `lookahead_ms` (clamped): 24 + round(lookahead · fs).
pub fn latency_frames_for(sample_rate: f32, lookahead_ms: f32) -> usize {
    UP_LATENCY + Timing::at(sample_rate, lookahead_ms).la
}

/// Static soft-knee curve: identity up to `a`, then
/// `a + (L − a)·tanh((x − a)/(L − a))` (never above `L`).
#[inline]
fn soft_knee(x: f32, a: f32, level: f32) -> f32 {
    if x <= a {
        x
    } else {
        let span = level - a;
        a + span * ((x - a) / span).tanh()
    }
}

/// One channel's event in progress.
#[derive(Debug, Clone, Copy, Default)]
struct Event {
    active: bool,
    first: u64,
    last_core: u64,
    peak: f32,
    level: f32,
    knee: f32,
    gain: f32,
}

/// The spike guard. See the module documentation.
#[derive(Debug)]
pub struct SpikeGuard {
    params: SpikeGuardParams,
    channels: usize,
    /// Lookahead for the next configure (ms).
    lookahead_ms: f32,
    timing: Timing,
    latency: usize,
    os: Oversampler,
    /// Main path: input delayed by `latency`.
    main: DelayLine,
    /// Interpolated chunk (`CHUNK · OS · channels`).
    hi: Vec<f32>,
    /// Gain ring (`gain_len` frames × channels), indexed by frame label,
    /// 1 outside dips.
    gain: Vec<f32>,
    gain_len: usize,
    /// Raised-cosine ramp weights, from the dip outwards.
    ramp_w: Vec<f32>,
    /// Envelope input ring (`env_delay + 1` frames × channels): peak, and
    /// whether the frame is left out.
    env_p: Vec<f32>,
    env_skip: Vec<bool>,
    /// Label of the next frame.
    frame: u64,
    env: Vec<PeakEnvelope>,
    ev: Vec<Event>,
    /// Per channel: frames after an event still left out of the envelope.
    hold_left: Vec<usize>,
    /// Last frame covered by a counted event (any channel).
    counted_until: u64,
    threshold_lin: f32,
    knee_lin: f32,
    core_lin: f32,
    /// Rise gate factor ([`RISE_OVER_CAP_DB`], linear, re the cap).
    rise_lin: f32,
    /// Base-rate input ring (`base_len` frames × channels), by frame label.
    base: Vec<f32>,
    base_len: usize,
    /// Rise gate: 48 kHz samples per sample (`48000 / fs`).
    rise_scale: f32,
    /// The current chunk's input (`CHUNK · channels`).
    input: Vec<f32>,
    /// Spikes since the last [`take_events`](Self::take_events).
    events: u64,
}

impl Default for SpikeGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl SpikeGuard {
    /// Unconfigured spike guard with default parameters and the standard
    /// lookahead.
    pub fn new() -> Self {
        Self {
            params: SpikeGuardParams::default(),
            channels: 0,
            lookahead_ms: STANDARD_LOOKAHEAD_MS,
            timing: Timing::at(48_000.0, STANDARD_LOOKAHEAD_MS),
            latency: 0,
            os: Oversampler::new(),
            main: DelayLine::default(),
            hi: Vec::new(),
            gain: Vec::new(),
            gain_len: 1,
            ramp_w: Vec::new(),
            env_p: Vec::new(),
            env_skip: Vec::new(),
            frame: 0,
            env: Vec::new(),
            ev: Vec::new(),
            hold_left: Vec::new(),
            counted_until: 0,
            threshold_lin: db_to_lin(DEFAULT_THRESHOLD_DB as f64) as f32,
            knee_lin: db_to_lin(-KNEE_DB as f64) as f32,
            core_lin: db_to_lin(CORE_DB as f64) as f32,
            rise_lin: db_to_lin(RISE_OVER_CAP_DB as f64) as f32,
            base: Vec::new(),
            base_len: 1,
            rise_scale: 1.0,
            input: Vec::new(),
            events: 0,
        }
    }

    /// Update the parameters (clamped). Never allocates.
    pub fn set_params(&mut self, params: &SpikeGuardParams) {
        self.params = params.clamped();
        self.threshold_lin = db_to_lin(self.params.threshold_db as f64) as f32;
    }

    /// Current (clamped) parameters.
    pub fn params(&self) -> &SpikeGuardParams {
        &self.params
    }

    /// Set the lookahead (ms, 0 … [`MAX_LOOKAHEAD_MS`]). Structural: takes
    /// effect at the next [`Processor::configure`].
    pub fn set_lookahead_ms(&mut self, ms: f32) {
        self.lookahead_ms = clamp_lookahead_ms(ms);
    }

    /// Lookahead set for the next configure (ms).
    pub fn lookahead_ms(&self) -> f32 {
        self.lookahead_ms
    }

    /// Number of spikes since the last call (or configure/reset). A click
    /// on several channels at once counts once.
    pub fn take_events(&mut self) -> u64 {
        core::mem::take(&mut self.events)
    }

    /// Current programme envelope of channel `c` (linear peak, ≥
    /// [`ENVELOPE_FLOOR`]).
    pub fn envelope(&self, c: usize) -> f32 {
        self.env
            .get(c)
            .map_or(ENVELOPE_FLOOR, |e| (e.value() as f32).max(ENVELOPE_FLOOR))
    }

    #[inline]
    fn gain_index(&self, f: u64, c: usize) -> usize {
        (f % self.gain_len as u64) as usize * self.channels + c
    }

    /// Lower the gain of channel `c` at frame `f` to `v` (min-combined).
    #[inline]
    fn lower(&mut self, c: usize, f: u64, v: f32) {
        let i = self.gain_index(f, c);
        if v < self.gain[i] {
            self.gain[i] = v;
        }
    }

    /// A ramp from `g` (at `f`) up to 1 over the `pre` frames before `f`,
    /// with its own raised-cosine shape.
    fn ramp_before(&mut self, c: usize, f: u64, g: f32) {
        let n = self.timing.pre;
        for d in 0..n {
            let w = 0.5 + 0.5 * (core::f32::consts::PI * (d + 1) as f32 / (n + 1) as f32).cos();
            self.lower(c, f - 1 - d as u64, 1.0 - (1.0 - g) * w);
        }
    }

    /// A ramp from `g` (at `f`) back toward 1 over the frames after `f`.
    fn ramp_after(&mut self, c: usize, f: u64, g: f32) {
        for d in 0..self.ramp_w.len() {
            let v = 1.0 - (1.0 - g) * self.ramp_w[d];
            self.lower(c, f + 1 + d as u64, v);
        }
    }

    /// Count a spike from `first` to `end` (once across channels), and keep
    /// the frames of its ramp out of the envelope.
    fn count(&mut self, first: u64, end: u64) {
        if first > self.counted_until {
            self.events += 1;
        }
        self.counted_until = self.counted_until.max(end);
    }

    /// Detector step for the newest frame `t` of channel `c`, peak `p`.
    fn detect(&mut self, c: usize, t: u64, p: f32) {
        let Timing {
            la,
            max_len,
            isolation,
            gap,
            ..
        } = self.timing;
        let mut e = self.ev[c];
        // Past the isolation window the spike is confirmed; a new core
        // frame after that is a new excursion.
        let confirmed = e.active && t > e.first + (max_len + isolation) as u64;
        if e.active && confirmed && p > e.knee.max(e.peak * self.core_lin) {
            e.active = false;
            self.ramp_after(c, t - 1, e.gain);
            self.hold_left[c] = self.ramp_w.len();
            self.count(e.first, t);
            self.ev[c] = e;
        } else if e.active {
            e.peak = e.peak.max(p);
            let core = e.knee.max(e.peak * self.core_lin);
            if p > core {
                e.last_core = t;
            }
            let g = soft_knee(e.peak, e.knee, e.level) / e.peak;
            if g < e.gain {
                e.gain = g;
                self.ramp_before(c, t, g);
            }
            if e.last_core - e.first + 1 > max_len as u64 {
                // Programme, not a spike: lift what the output has not
                // read yet, and let the envelope take the peak.
                let read = t - la as u64;
                let g_read = self.gain[self.gain_index(read, c)];
                for f in read + 1..=t {
                    let i = self.gain_index(f, c);
                    self.gain[i] = 1.0;
                }
                if g_read < 1.0 {
                    self.ramp_after(c, read, g_read);
                }
                self.env[c].step(e.peak as f64);
                e.active = false;
                self.ev[c] = e;
                return;
            }
            if t - e.last_core > gap as u64 {
                // A spike: the dip has held through the gap; ramp back.
                e.active = false;
                self.ramp_after(c, t - 1, e.gain);
                self.hold_left[c] = self.ramp_w.len();
                self.count(e.first, t);
                self.ev[c] = e;
                return;
            }
            self.lower(c, t, e.gain);
            self.ev[c] = e;
            return;
        }
        if !self.params.enabled {
            return;
        }
        let env = (self.env[c].value() as f32).max(ENVELOPE_FLOOR);
        let level = env * self.threshold_lin;
        if p > level && !self.looks_like_click(c, t, level) {
            // Programme (it builds up, or recurs): no event, and the
            // envelope takes this frame's peak at once, so the rest of the
            // excursion is judged against it.
            self.env[c].step(p as f64);
            return;
        }
        if p > level {
            let knee = level * self.knee_lin;
            let g = soft_knee(p, knee, level) / p;
            self.ev[c] = Event {
                active: true,
                first: t,
                last_core: t,
                peak: p,
                level,
                knee,
                gain: g,
            };
            self.lower(c, t, g);
            self.ramp_before(c, t, g);
            // The frames just before it (ringing ahead of a click) stay out
            // of the envelope.
            let ch = self.channels;
            for k in 0..self.env_skip.len() / ch {
                self.env_skip[k * ch + c] = true;
            }
        }
    }

    /// Rise gate (see the module docs): whether the excursion the detector
    /// sees at `t` (input frame `n = t − 24`) looks like a click in the
    /// base-rate input already buffered (`n − 3 − (R − 1) … t`). `level` is
    /// the cap.
    #[inline]
    fn looks_like_click(&self, c: usize, t: u64, level: f32) -> bool {
        let ch = self.channels;
        let n = t - UP_LATENCY as u64;
        let len = self.base_len as u64;
        let at = |f: u64| self.base[(f % len) as usize * ch + c];
        // The excursion's base-rate peak within the longest core.
        let reach = n + self.timing.max_len.min(UP_LATENCY) as u64;
        let (mut j, mut peak) = (n, 0.0f32);
        for f in n - RISE_BEFORE as u64..=reach {
            if at(f).abs() > peak {
                peak = at(f).abs();
                j = f;
            }
        }
        // Short and steep: at most R samples take it from the programme
        // level (`s` and the sample before it at or under the cap; the
        // latest such `s`) to the peak, by more than the cap per 48 kHz
        // sample of the rise.
        let rise = self.timing.rise as u64;
        let at_programme =
            |s: u64| (0..RISE_FROM_PROGRAMME_FRAMES as u64).all(|k| at(s - k).abs() <= level);
        let Some(s) = (1..=rise).map(|r| j - r).find(|&s| at_programme(s)) else {
            return false;
        };
        let steps = (j - s) as f32 * self.rise_scale;
        if (at(j) - at(s)).abs() <= self.rise_lin * level * steps.max(1.0) {
            return false;
        }
        // One stroke: the samples between rise toward the peak on its side.
        // (A swell on a high tone alternates in sign on the way up.)
        if !(s + 1..j).all(|k| at(k) * at(j) > 0.0 && at(k).abs() <= at(k + 1).abs()) {
            return false;
        }
        // Isolated: its core does not come back in the isolation window.
        // Up to 48 kHz all of it is buffered. Above, the window starts
        // after the click's own fall (its last sample over the core level
        // within R samples after the peak) and runs to the newest frame,
        // and needs at least PREVIEW_MIN_FRAMES of it.
        let core = (level * self.knee_lin).max(peak * self.core_lin);
        let full_from = n + self.timing.max_len as u64 + 1;
        let to = (n + (self.timing.max_len + self.timing.isolation) as u64).min(t);
        let from = if full_from + PREVIEW_MIN_FRAMES as u64 <= to + 1 {
            full_from
        } else {
            let fall_end = (j + rise).min(t);
            let end = (j + 1..=fall_end)
                .rev()
                .find(|&f| at(f).abs() > core)
                .unwrap_or(j)
                + 1;
            if end + PREVIEW_MIN_FRAMES as u64 > to + 1 {
                return false;
            }
            end
        };
        (from..=to).all(|f| at(f).abs() <= core)
    }

    fn process_chunk(&mut self, chunk: &mut [f32]) {
        let ch = self.channels;
        let frames = chunk.len() / ch;
        let w = OS * ch;
        let la = self.timing.la as u64;
        let ed = (self.timing.env_delay + 1) as u64;
        // Interpolate the new input, then replace it by the delayed input.
        self.os.upsample(chunk, &mut self.hi[..frames * w]);
        self.input[..frames * ch].copy_from_slice(chunk);
        self.main.write_frames(chunk);
        self.main.read_frames(self.latency, chunk);
        for i in 0..frames {
            let t = self.frame;
            self.frame += 1;
            let slot = (t % ed) as usize;
            let bslot = (t % self.base_len as u64) as usize * ch;
            self.base[bslot..bslot + ch].copy_from_slice(&self.input[i * ch..(i + 1) * ch]);
            for c in 0..ch {
                let mut p = 0.0f32;
                for s in 0..OS {
                    p = p.max(self.hi[i * w + s * ch + c].abs());
                }
                // Envelope: the frame `env_delay` back leaves the ring (its
                // slot is the one the new frame takes).
                let j = slot * ch + c;
                if !self.env_skip[j] {
                    self.env[c].step(self.env_p[j] as f64);
                }
                self.env_p[j] = p;
                self.env_skip[j] = false;
                let held = self.hold_left[c] > 0;
                if held {
                    self.hold_left[c] -= 1;
                }
                self.detect(c, t, p);
                if held || self.ev[c].active {
                    self.env_skip[j] = true;
                }
            }
            // The output frame takes gain frame `t − La` (input frame
            // `t − La − 24`).
            let read = t - la;
            for c in 0..ch {
                let gi = self.gain_index(read, c);
                let g = self.gain[gi];
                if g != 1.0 {
                    chunk[i * ch + c] *= g;
                    self.gain[gi] = 1.0;
                }
            }
        }
    }
}

impl Processor for SpikeGuard {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        self.channels = channels;
        if channels == 0 {
            return;
        }
        let fs = rate_or_default(sample_rate);
        self.os.configure(FACTOR, channels);
        debug_assert_eq!(self.os.up_latency_frames(), UP_LATENCY);
        self.timing = Timing::at(sample_rate, self.lookahead_ms);
        self.latency = UP_LATENCY + self.timing.la;
        self.main.allocate(self.latency + CHUNK, channels);
        self.hi = vec![0.0; CHUNK * OS * channels];
        self.input = vec![0.0; CHUNK * channels];
        self.base_len = self.timing.base_len();
        self.rise_scale = (48_000.0 / fs) as f32;
        self.base = vec![0.0; self.base_len * channels];
        // Writes reach `ramp` frames past the newest; reads are `la` behind.
        self.gain_len = self.timing.la + self.timing.ramp + 2;
        self.gain = vec![1.0; self.gain_len * channels];
        let n = self.timing.ramp;
        self.ramp_w = (1..=n)
            .map(|d| (0.5 + 0.5 * (core::f64::consts::PI * d as f64 / (n + 1) as f64).cos()) as f32)
            .collect();
        let ed = self.timing.env_delay + 1;
        self.env_p = vec![0.0; ed * channels];
        self.env_skip = vec![false; ed * channels];
        self.env = vec![PeakEnvelope::new(ENVELOPE_RELEASE_MS, fs); channels];
        self.ev = vec![Event::default(); channels];
        self.hold_left = vec![0; channels];
        self.reset();
    }

    fn process(&mut self, samples: &mut [f32]) {
        let ch = self.channels;
        if ch == 0 || self.hi.is_empty() {
            return;
        }
        let usable = samples.len() / ch * ch;
        for chunk in samples[..usable].chunks_mut(CHUNK * ch) {
            self.process_chunk(chunk);
        }
    }

    fn reset(&mut self) {
        if self.channels == 0 {
            return;
        }
        self.os.reset();
        self.main.clear();
        self.gain.fill(1.0);
        self.env_p.fill(0.0);
        self.env_skip.fill(false);
        self.base.fill(0.0);
        // Labels start past every backward reach (lookahead and ramps).
        self.frame = (self.gain_len + self.timing.ramp + 1 + self.base_len) as u64;
        self.counted_until = 0;
        for e in &mut self.env {
            e.reset();
        }
        self.ev.fill(Event::default());
        self.hold_left.fill(0);
        self.events = 0;
    }

    fn latency_frames(&self) -> usize {
        self.latency
    }
}
