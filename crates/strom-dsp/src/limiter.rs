//! Lookahead sample-peak limiter.
//!
//! # Signal flow
//!
//! The audio is delayed by the lookahead `L` frames ([`DelayLine`]). The gain is
//! computed from the *undelayed* input, so it is already down when a peak reaches
//! the output. All channels share one gain (linked), computed from the frame peak
//! `p[j] = max_ch |x_ch[j]|`.
//!
//! For every input frame `j` the limiter stores the *required gain*
//! `r[j] = min(1, c / p[j])`, where `c` is the linear ceiling. The output frame
//! `n` is the delayed input `x[n − L]` multiplied by a gain `g[n]`; the ceiling
//! holds exactly when `g[n] ≤ r[n − L]`.
//!
//! # Gain computation (derived from the requirement above)
//!
//! With attack length `A ≤ L` frames, split as `A = A1 + A2`, and hold
//! `H = min(`[`GAIN_HOLD_MS`]`, release)` frames:
//!
//! 1. **Sliding minimum** ([`SegmentedMin`])
//!    `h[m] = min { r[j] : m − L − H ≤ j ≤ m − L + A }`
//!    (a window of `A + H + 1` required gains, ending `L − A` frames in the
//!    past, so it is causal). The *base* part `[m − L, m − L + A]` makes the
//!    gain reach the required gain in time; the *hold* part `[m − L − H, m − L)`
//!    keeps it down for `H` frames after a peak has been output, so on a
//!    steady low-frequency tone the gain does not release between waveform
//!    peaks (a window of at least half a period always contains a peak of
//!    `|x|`). Without it the gain rippled with every half-cycle: 3.4 % THD at
//!    50 Hz with attack 1 ms / release 50 ms. The release therefore starts `H`
//!    frames after the last frame that required the reduction.
//!    Computed with the van Herk / Gil–Werman segment algorithm
//!    (prefix and suffix minima of fixed-length segments): a constant number of
//!    comparisons per frame, independent of the material. It uses comparisons
//!    only, so `h[m]` is the *exact* minimum of the window, which is all the
//!    argument below needs.
//! 2. **Two cascaded moving averages** ([`RampSmoother`]) of lengths `A1 + 1` and
//!    `A2 + 1` over `h`. Their combined kernel is a trapezoid/triangle spanning offsets
//!    `0 … A`, so a step in `h` becomes a smooth S-shaped ramp of `A` frames.
//! 3. **Release**: `g[n] = t[n]` when the box output `t[n]` is below the previous
//!    gain (follow downwards instantly; the ramp is already smooth), otherwise a
//!    one-pole rise towards `t[n]` with the release time constant, capped at `t[n]`.
//!
//! *Why the ceiling is guaranteed*: the box output at `n` is a weighted mean
//! (non-negative weights summing to 1) of `h[n − k]` for `0 ≤ k ≤ A`. The window
//! of `h[n − k]` is `[n − k − L − H, n − k − L + A]`. It contains `n − L` for
//! every such `k`: its start `n − k − L − H ≤ n − L` because `k, H ≥ 0` (the hold
//! only moves the start further back), and its end `n − k − L + A ≥ n − L`
//! because `k ≤ A`. In fact the *base* part `[n − k − L, n − k − L + A]` alone
//! contains `n − L`. Each `h` is the exact minimum over a set that includes its
//! whole base window (right after a rebuild, the segment minimum is pre-filled
//! from the stored history; only values in the hold part may be replaced by 1,
//! which is `≥` every `r`, see the rebuild rule below), so each
//! `h[n − k] ≤ r[n − L]`, the mean is `≤ r[n − L]`, and the release stage never
//! exceeds the box output. The gain reaches its target exactly when the peak
//! is output, starting `A` frames earlier.
//!
//! To make this hold *exactly* in floating point the box stage works on
//! integers: `h` is quantised downwards to a fixed-point value with
//! [`RAMP_FRAC_BITS`](crate::dynamics::RAMP_FRAC_BITS) fractional bits, the running sums are exact, and each mean
//! is an integer division (rounding down). Rounding down only ever lowers the
//! gain. The ceiling itself is the largest `f32` not above `10^(dB/20)`, so the
//! final `f64 → f32` rounding of `g · x` cannot cross it.
//!
//! With lookahead 0 the windows degenerate to `A = 0`: the gain drops instantly
//! to `r[n]` (a hard limiter with hold and smooth release).
//!
//! # Parameter changes
//!
//! - Threshold and enable state are applied per *input* frame: each frame stores
//!   the required gain and ceiling in force when it entered the lookahead window,
//!   so a change takes effect with the same latency as the audio and is ramped
//!   by the normal gain smoothing (no clicks, guarantee unaffected).
//!   When disabled, incoming frames require unity gain; the gain returns to
//!   exactly 1.0 by a linear ramp of 10 ms (full scale) after the frames that
//!   were limited have left, after which the output is a bit-exact delay.
//! - Attack changes, and release changes that change the hold `H`, rebuild the
//!   min/box pipeline from the stored history of required gains (O(lookahead +
//!   hold), no allocation). The base window is replayed *as if* the new attack
//!   had always been in force, so the guarantee holds immediately. The hold
//!   part never reaches further back than the previous pipeline did: required
//!   gains older than that count as 1. A longer hold therefore grows one frame
//!   per frame instead of pulling in old, lower required gains and dropping a
//!   releasing gain at once; a shorter hold lets the release start earlier,
//!   smoothly.
//! - Lookahead changes cross-fade (raised cosine, 5 ms) from the old to the new
//!   delay tap. During the cross-fade the sliding-minimum window is widened to
//!   cover both taps (`[m − Lhi − H, m − Llo + A]`, base part
//!   `[m − Lhi, m − Llo + A]`), so the gain is at most the required gain of
//!   *both* samples being mixed and the guarantee still holds: for
//!   `0 ≤ k ≤ A` the base window of `h[n − k]` contains both `n − Lhi` and
//!   `n − Llo`. A further change during a cross-fade is queued until it
//!   completes.
//! - Between `configure`/`reset` and the first processed frame, `set_params`
//!   applies instantly (no cross-fade from the previous lookahead), so the
//!   result does not depend on whether parameters were set before or after
//!   `configure`.
//!
//! # Peak control (crate-internal; off unless set)
//!
//! A slow gain on top of the fast one, for material whose peaks grow over
//! several periods (speech: each glottal pulse a little higher than the
//! last), where the fast ramp would step down at every new peak. Per frame
//! `j`, from the same frame peak `p[j]` and in the same peak-stream time:
//!
//! 1. Static reduction from the compressor curve ([`GainComputer`]; Giannoulis,
//!    Massberg & Reiss, JAES 60(6), 2012) with its threshold
//!    `headroom_db` under the ceiling, `ratio`, a soft knee, and its slope
//!    scaled by a **drive** (0 … 1, per frame or held, set by the caller).
//! 2. A hold of [`PEAK_CONTROL_HOLD_MS`] on that static reduction, longer
//!    than a pitch period, so the smoothed reduction rises continuously
//!    instead of in steps at each waveform peak (a step synchronous with
//!    the waveform puts energy below the voice's fundamental, which nothing
//!    there masks).
//! 3. A branching one-pole in dB (attack, release).
//! 4. The required gain becomes `min(r[j], 10^(−y/20))`.
//!
//! Step 4 keeps every argument above: the pipeline sees a required gain
//! that is at most `r[j]`, so the ceiling still holds exactly, and the
//! peak control gets the lookahead too (its gain enters `L` frames ahead of
//! the audio). Taking the smaller gain, not the product of two stages in
//! series, matters on a step: a peak control in series would add its
//! reduction to the fast one while the fast one holds and releases (up to
//! the sum of both); here the reduction is the larger of the two.
//!
//! # Safety clip
//!
//! After the gain, each sample is checked against the ceiling of the frame it
//! came from. Mathematically it never triggers (see above); it is a guard
//! against bugs, and every intervention is counted ([`Limiter::safety_clip_count`]).
//! A NaN output sample (only possible from NaN/Inf input) is replaced by 0 and
//! counted.
//!
//! # Reuse as a gain core
//!
//! The [true-peak limiter](crate::truepeak) reuses this whole gain machinery
//! through two crate-internal hooks: it supplies the per-frame peaks from its
//! own (oversampled) detector instead of `max_ch |x_ch[j]|`, and it adds an
//! extra audio delay so the audio lines up with its detector's later peak
//! stream. With the defaults (no external peaks, no extra delay) the
//! `Limiter` is unchanged.
//!
//! # Efficiency
//!
//! The limiter is dominated by its serial per-frame gain pipeline, so unlike the
//! gate and compressor it runs one fused loop per frame (peak, gain pipeline,
//! delay, gain, clip); this measured faster than a block split for 1–8
//! channels. The cross-fade path is kept out of line.
//!
//! # Denormals
//!
//! The only recursive state is the release gain, which approaches its target
//! from below; once the gap is under 1e-15 it snaps to the target, so it never
//! decays through the subnormal range. The box stage is integer arithmetic.

use crate::dynamics::frame::{frame_peaks_simd, BLOCK_FRAMES};
use crate::dynamics::{DelayLine, GainComputer, RampSmoother, SegmentedMin};
use crate::simd::{detect, Level};
use crate::util::{db_to_lin, one_pole_coef, DENORMAL_FLUSH};
use crate::Processor;
use fearless_simd::dispatch;
use fearless_simd::prelude::*;

/// Largest supported lookahead (ms). The delay line is sized for this at
/// `configure` time so lookahead changes never allocate.
pub const MAX_LOOKAHEAD_MS: f32 = 20.0;

/// Longest gain hold (ms): the hold is `H = min(GAIN_HOLD_MS, release)`
/// frames, and the release starts `H` frames after the last frame that
/// required the reduction. 20 ms is half a period of 25 Hz, so the gain on a
/// steady sine at or above 25 Hz does not ripple (see the module docs).
pub const GAIN_HOLD_MS: f64 = 20.0;

/// Length of the cross-fade between delay taps on a lookahead change (ms).
const XFADE_MS: f64 = 5.0;
/// Time for the gain to ramp from 0 to 1 when the limiter is disabled (ms).
const BYPASS_RAMP_MS: f64 = 10.0;
/// Frame counter start, so `now − delay` never underflows.
const TIME_ORIGIN: u64 = 1 << 32;
/// Sample-rate clamp for `configure` (keeps fixed-point sums far from overflow).
const MAX_SAMPLE_RATE: f64 = 768_000.0;

/// Slow peak control in the limiter core (see the module docs, "Peak
/// control"). Off (the default for every limiter) when `ratio` is 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeakControlParams {
    /// Threshold below the ceiling (dB, ≥ 0).
    pub headroom_db: f32,
    /// Ratio above the threshold (1 = off).
    pub ratio: f32,
    /// Soft-knee width (dB).
    pub knee_db: f32,
    /// Attack time constant of the reduction (ms).
    pub attack_ms: f32,
    /// Release time constant of the reduction (ms).
    pub release_ms: f32,
}

impl Default for PeakControlParams {
    fn default() -> Self {
        Self {
            headroom_db: 0.0,
            ratio: 1.0,
            knee_db: 0.0,
            attack_ms: 5.0,
            release_ms: 100.0,
        }
    }
}

/// State of the peak control: its curve relative to the ceiling, the
/// one-pole coefficients and the smoothed reduction (dB).
#[derive(Debug, Clone, Copy)]
struct PeakControl {
    params: PeakControlParams,
    /// Curve with the threshold in dB re the ceiling.
    curve: GainComputer,
    att: f64,
    rel: f64,
    reduction_db: f64,
    /// Share (0 … 1) of the ratio's slope in force.
    drive: f64,
    /// Held static reduction (dB) and frames left in its hold.
    held_db: f64,
    hold_left: usize,
    hold_frames: usize,
    /// Smoothed reduction (dB) of each frame of the last block applied.
    block_db: [f32; BLOCK_FRAMES],
}

impl PeakControl {
    fn new() -> Self {
        Self {
            params: PeakControlParams::default(),
            curve: GainComputer {
                threshold_db: 0.0,
                knee_db: 0.0,
                slope: 0.0,
            },
            att: 0.0,
            rel: 0.0,
            reduction_db: 0.0,
            drive: 1.0,
            held_db: 0.0,
            hold_left: 0,
            hold_frames: 0,
            block_db: [0.0; BLOCK_FRAMES],
        }
    }

    /// Back to the state after construction: no reduction, no held static
    /// reduction, full drive. Keeps the parameters and coefficients.
    fn clear(&mut self) {
        self.reduction_db = 0.0;
        self.held_db = 0.0;
        self.hold_left = 0;
        self.drive = 1.0;
    }

    fn set(&mut self, p: &PeakControlParams, fs: f64) {
        let f = |v: f32, lo: f32, hi: f32, d: f32| if v.is_finite() { v.clamp(lo, hi) } else { d };
        let p = PeakControlParams {
            headroom_db: f(p.headroom_db, 0.0, 24.0, 0.0),
            ratio: f(p.ratio, 1.0, 100.0, 1.0),
            knee_db: f(p.knee_db, 0.0, 24.0, 0.0),
            attack_ms: f(p.attack_ms, 0.01, 1000.0, 5.0),
            release_ms: f(p.release_ms, 1.0, 5000.0, 100.0),
        };
        self.params = p;
        self.curve = GainComputer {
            threshold_db: -(p.headroom_db as f64),
            knee_db: p.knee_db as f64,
            slope: 1.0 - 1.0 / p.ratio as f64,
        };
        if fs > 0.0 {
            self.hold_frames = (PEAK_CONTROL_HOLD_MS * 0.001 * fs).round() as usize;
            self.att = 1.0 - one_pole_coef(p.attack_ms as f64, fs);
            self.rel = 1.0 - one_pole_coef(p.release_ms as f64, fs);
        }
        if !self.is_on() {
            self.reduction_db = 0.0;
        }
    }

    /// Whether it can act: a ratio over 1, or a reduction still releasing.
    fn is_on(&self) -> bool {
        self.curve.slope > 0.0 || self.reduction_db > 0.0
    }

    /// Lower the required gains `req` of frames with linked peaks `peaks`
    /// to the peak control's gain where that is lower.
    #[inline(always)]
    fn apply(&mut self, ceiling: f32, peaks: &[f32], drive: Option<&[f32]>, req: &mut [f64]) {
        let c = ceiling as f64;
        let start = c * 10f64.powf(self.curve.knee_start_db() / 20.0);
        for (i, ((r, &p), out)) in req
            .iter_mut()
            .zip(peaks)
            .zip(self.block_db.iter_mut())
            .enumerate()
        {
            if let Some(d) = drive {
                let d = d[i];
                self.drive = if d.is_finite() {
                    d.clamp(0.0, 1.0) as f64
                } else {
                    1.0
                };
            }
            let p = p as f64;
            let slope = self.curve.slope * self.drive;
            let target = if p > start && p.is_finite() && slope > 0.0 {
                GainComputer {
                    slope,
                    ..self.curve
                }
                .reduction_db(20.0 * (p / c).log10())
            } else {
                0.0
            };
            // Hold the static reduction for at least a pitch period, so the
            // smoothed reduction rises continuously instead of in steps at
            // each waveform peak.
            let target = if target >= self.held_db {
                self.hold_left = self.hold_frames;
                target
            } else if self.hold_left > 0 {
                self.hold_left -= 1;
                self.held_db
            } else {
                target
            };
            self.held_db = target;
            let y = self.reduction_db;
            let y = if target > y {
                y + self.att * (target - y)
            } else {
                y + self.rel * (target - y)
            };
            self.reduction_db = if y < 1e-9 { 0.0 } else { y };
            *out = self.reduction_db as f32;
            if self.reduction_db > 0.0 {
                *r = r.min(10f64.powf(-self.reduction_db / 20.0));
            }
        }
    }
}

/// Hold of the peak control's static reduction (ms): longer than a pitch
/// period of speech (2.5 … 12.5 ms).
pub const PEAK_CONTROL_HOLD_MS: f64 = 20.0;

/// Limiter parameters (units as in the GStreamer interface).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LimiterParams {
    /// When false: unity gain, but the lookahead delay is kept (constant latency).
    pub enabled: bool,
    /// Ceiling (dBFS, −60 … 0). No output sample exceeds it.
    pub threshold_db: f32,
    /// Attack time (ms, 0.01 … 50).
    pub attack_ms: f32,
    /// Release time (ms, 1 … 2000): one-pole time constant of the gain's rise.
    /// The rise starts after a hold of `min(`[`GAIN_HOLD_MS`]`, release)`.
    pub release_ms: f32,
    /// Lookahead (ms, 0 … 20). Latency = round(lookahead · fs / 1000) frames.
    pub lookahead_ms: f32,
}

impl Default for LimiterParams {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold_db: -1.0,
            attack_ms: 1.0,
            release_ms: 100.0,
            lookahead_ms: 5.0,
        }
    }
}

/// Clamp into `[lo, hi]`; non-finite values become `def`.
fn finite_clamp(v: f32, def: f32, lo: f32, hi: f32) -> f32 {
    if v.is_finite() {
        v.clamp(lo, hi)
    } else {
        def
    }
}

impl LimiterParams {
    /// Clamp every field into its documented range; non-finite values fall
    /// back to the default.
    fn sanitized(&self) -> Self {
        let d = Self::default();
        Self {
            enabled: self.enabled,
            threshold_db: finite_clamp(self.threshold_db, d.threshold_db, -60.0, 0.0),
            attack_ms: finite_clamp(self.attack_ms, d.attack_ms, 0.01, 50.0),
            release_ms: finite_clamp(self.release_ms, d.release_ms, 1.0, 2000.0),
            lookahead_ms: finite_clamp(self.lookahead_ms, d.lookahead_ms, 0.0, MAX_LOOKAHEAD_MS),
        }
    }
}

/// Sample rate as the limiter uses it after `configure(sample_rate, _)`.
fn effective_rate(sample_rate: f32) -> f64 {
    if sample_rate.is_finite() && sample_rate > 0.0 {
        (sample_rate as f64).min(MAX_SAMPLE_RATE)
    } else {
        48_000.0
    }
}

/// Latency in frames of a limiter configured at `sample_rate` (Hz) with
/// `lookahead_ms`: exactly the value [`Processor::latency_frames`] reports,
/// i.e. `round(lookahead_ms · fs / 1000)` after the same clamping of both
/// arguments (lookahead to `0 … MAX_LOOKAHEAD_MS`, non-finite → default).
pub fn lookahead_frames(lookahead_ms: f32, sample_rate: f32) -> usize {
    let d = LimiterParams::default();
    let ms = finite_clamp(lookahead_ms, d.lookahead_ms, 0.0, MAX_LOOKAHEAD_MS);
    ms_to_frames(ms as f64, effective_rate(sample_rate))
}

/// Cross-fade state between two delay taps.
#[derive(Debug, Clone, Copy)]
struct Crossfade {
    to: usize,
    pos: usize,
}

/// Linked-channel lookahead limiter. See [`LimiterParams`] and the module
/// documentation for the algorithm.
#[derive(Debug)]
pub struct Limiter {
    params: LimiterParams,

    // Configuration.
    sample_rate: f64,
    channels: usize,
    max_frames: usize,

    // Derived from parameters.
    ceiling: f32,
    release_coef: f64,
    bypass_step: f64,
    target_lookahead: usize,
    attack_frames: usize,
    hold_frames: usize,
    max_hold_frames: usize,
    xfade_frames: usize,
    /// Attack or hold changed: the min/ramp pipeline needs a rebuild.
    shape_dirty: bool,

    delay: DelayLine,
    // Per-input-frame history: required gain and ceiling, indexed by frame time.
    req_hist: Vec<f64>,
    ceil_hist: Vec<f32>,
    /// Peak control's smoothed reduction (dB) per input frame, so it can be
    /// read for the frame being output (see
    /// [`Limiter::peak_control_reduction_db`]).
    pc_hist: Vec<f32>,
    hist_mask: usize,

    // Gain pipeline.
    window_min: SegmentedMin,
    win_push_delay: u64,
    /// Oldest frame time (exclusive bound) the hold part of the window may
    /// reach back to after a rebuild; see [`Limiter::rebuild`].
    hold_from: u64,
    /// Window reach `hi + hold` of the last rebuild.
    built_reach: u64,
    ramp: RampSmoother,
    gain: f64,

    // Running state.
    now: u64,
    lookahead: usize,
    xfade: Option<Crossfade>,
    safety_clips: u64,
    /// True from configure/reset until the first frame: parameters apply instantly.
    fresh: bool,
    /// Instruction set of the hot loops (detected in `configure`).
    level: Level,
    /// Set by `set_simd_level`: `configure` keeps `level` instead of detecting.
    level_forced: bool,
    /// Extra audio delay (frames) on top of the lookahead, for a detector
    /// whose peak stream lags the audio (see [`Limiter::set_extra_delay`]).
    /// Always 0 for a plain `Limiter`.
    extra_delay: usize,
    /// Slow peak control (off unless set through
    /// [`Limiter::set_peak_control`]).
    peak_control: PeakControl,
}

impl Default for Limiter {
    fn default() -> Self {
        Self::new()
    }
}

impl Limiter {
    /// Create an unconfigured limiter with default parameters.
    pub fn new() -> Self {
        let mut l = Self {
            params: LimiterParams::default(),
            sample_rate: 0.0,
            channels: 0,
            max_frames: 0,
            ceiling: 1.0,
            release_coef: 0.0,
            bypass_step: 1.0,
            target_lookahead: 0,
            attack_frames: 0,
            hold_frames: 0,
            max_hold_frames: 0,
            xfade_frames: 1,
            shape_dirty: false,
            delay: DelayLine::default(),
            req_hist: Vec::new(),
            ceil_hist: Vec::new(),
            pc_hist: Vec::new(),
            hist_mask: 0,
            window_min: SegmentedMin::default(),
            win_push_delay: 0,
            hold_from: 0,
            built_reach: 0,
            ramp: RampSmoother::default(),
            gain: 1.0,
            now: TIME_ORIGIN,
            lookahead: 0,
            xfade: None,
            safety_clips: 0,
            fresh: true,
            level: Level::baseline(),
            level_forced: false,
            extra_delay: 0,
            peak_control: PeakControl::new(),
        };
        l.set_params(&LimiterParams::default());
        l
    }

    /// Update parameters. Never allocates. Out-of-range values are clamped.
    ///
    /// Between `configure`/`reset` and the first processed frame the new
    /// lookahead and attack apply instantly (no cross-fade).
    pub fn set_params(&mut self, params: &LimiterParams) {
        let p = params.sanitized();
        self.params = p;
        self.ceiling = ceiling_f32(p.threshold_db);
        self.update_derived();
        if self.fresh && self.is_configured() {
            self.lookahead = self.target_lookahead;
            self.xfade = None;
            self.rebuild();
        }
    }

    /// Current (sanitised) parameters.
    pub fn params(&self) -> &LimiterParams {
        &self.params
    }

    /// Number of times the final safety clip changed a sample since the last
    /// `reset`/`configure`. Expected to stay 0 on normal material.
    pub fn safety_clip_count(&self) -> u64 {
        self.safety_clips
    }

    /// The linear ceiling actually enforced: the largest `f32` not above
    /// `10^(threshold_db/20)`.
    pub fn ceiling_linear(&self) -> f32 {
        self.ceiling
    }

    /// Gain (linear) applied to the most recently output frame.
    pub fn current_gain(&self) -> f64 {
        self.gain
    }

    /// Force the SIMD instruction set used by `process` (for hosts and tests),
    /// e.g. `Level::fallback()` (needs fearless_simd's `force_support_fallback`
    /// feature) to run the portable scalar path. By default
    /// `configure` detects the best level ([`crate::simd::detect`]); a level
    /// set here is kept across later `configure` calls. Every level produces
    /// bit-identical output. Never allocates.
    pub fn set_simd_level(&mut self, level: Level) {
        self.level = level;
        self.level_forced = true;
    }

    /// The SIMD instruction set `process` currently dispatches to.
    pub fn simd_level(&self) -> Level {
        self.level
    }

    /// Crate-internal core hook: delay the audio by `frames` more than the
    /// lookahead, for an external detector whose peak stream (see
    /// [`process_with_peaks`](Self::process_with_peaks)) describes the audio
    /// `frames` frames late. The gain pipeline is unchanged: it works in
    /// *peak-stream time*, and the output frame `n` is the input frame
    /// `n − L − frames` times the gain the pipeline computed for peak frame
    /// `n − L`. Takes effect at the next `configure` (it sizes the delay
    /// line); [`Processor::latency_frames`] includes it.
    pub(crate) fn set_extra_delay(&mut self, frames: usize) {
        self.extra_delay = frames;
    }

    /// Crate-internal core hook: slow peak control (see the module docs).
    /// Applies at once, without allocation.
    pub(crate) fn set_peak_control(&mut self, params: &PeakControlParams) {
        self.peak_control.set(params, self.sample_rate);
    }

    /// Crate-internal core hook: the peak control's reduction (dB, ≥ 0) in
    /// force for the most recently output frame, i.e. computed for the peak
    /// frame `L` frames before it (the gain pipeline works in peak-stream
    /// time; the detector delay is on the audio side). It is the same point
    /// in time as [`Limiter::current_gain`], and never exceeds that gain's
    /// reduction: the peak control only lowers the required gain the
    /// pipeline stays under.
    pub(crate) fn peak_control_reduction_db(&self) -> f64 {
        if self.pc_hist.is_empty() {
            return 0.0;
        }
        let t = self.now - 1 - self.lookahead as u64;
        self.pc_hist[t as usize & self.hist_mask] as f64
    }

    /// Crate-internal core hook: share (0 … 1) of the peak control's slope in
    /// force (see the module docs). Applies at once.
    pub(crate) fn set_peak_control_drive(&mut self, drive: f32) {
        self.peak_control.drive = if drive.is_finite() {
            drive.clamp(0.0, 1.0) as f64
        } else {
            1.0
        };
    }

    /// Crate-internal core hook: process `samples` like
    /// [`Processor::process`], but take the linked peak of each frame from
    /// `peaks` (one value per frame, `≥ 0`) instead of measuring the frames.
    /// `peaks.len()` must be at least the frame count. Everything else
    /// (required gain, sliding minimum, ramp, hold, release, delay, safety
    /// clip at the ceiling) is the plain limiter's.
    /// `drive`, when given, holds the peak control's drive per frame (see
    /// [`Limiter::set_peak_control_drive`]).
    pub(crate) fn process_with_peaks(
        &mut self,
        samples: &mut [f32],
        peaks: &[f32],
        drive: Option<&[f32]>,
    ) {
        let ch = self.channels;
        if !self.is_configured() || samples.len() < ch || peaks.len() < samples.len() / ch {
            return;
        }
        self.fresh = false;
        if self.shape_dirty {
            self.rebuild();
        }
        self.start_crossfade_if_needed();

        dispatch!(self.level, simd => self.process_blocks(simd, samples, Some(peaks), drive));
    }

    fn is_configured(&self) -> bool {
        self.channels != 0 && !self.req_hist.is_empty()
    }

    fn update_derived(&mut self) {
        let fs = self.sample_rate;
        let p = self.params;
        self.release_coef = one_pole_coef(p.release_ms as f64, fs.max(1.0));
        self.bypass_step = 1.0 / (BYPASS_RAMP_MS * 0.001 * fs).max(1.0);
        self.target_lookahead = ms_to_frames(p.lookahead_ms as f64, fs).min(self.max_frames);
        let attack = ms_to_frames(p.attack_ms as f64, fs).min(self.max_frames);
        if attack != self.attack_frames {
            self.attack_frames = attack;
            self.shape_dirty = true;
        }
        let hold =
            ms_to_frames((p.release_ms as f64).min(GAIN_HOLD_MS), fs).min(self.max_hold_frames);
        if hold != self.hold_frames {
            self.hold_frames = hold;
            self.shape_dirty = true;
        }
    }

    /// Current window geometry: (shortest tap, longest tap).
    fn taps(&self) -> (usize, usize) {
        match self.xfade {
            Some(xf) => (self.lookahead.min(xf.to), self.lookahead.max(xf.to)),
            None => (self.lookahead, self.lookahead),
        }
    }

    /// Rebuild the min/box pipeline for the current geometry (taps, attack,
    /// hold) from the stored required-gain history. O(lookahead + hold), no
    /// allocation.
    ///
    /// The *base* part of each window, `[t − hi, t − d]`, is always replayed
    /// from the history exactly: the ceiling proof rests on it. The *hold*
    /// part, `[t − hi − H, t − hi)`, never reaches further back than the
    /// previous pipeline did (frames before `hold_from` count as 1, the
    /// neutral value), so a longer hold or a longer window does not pull in
    /// old, lower required gains at once and drop the gain: the hold grows
    /// one frame per frame instead. On a fresh start the history is all 1
    /// and nothing is masked.
    fn rebuild(&mut self) {
        self.shape_dirty = false;
        let (lo, hi) = self.taps();
        let a = self.attack_frames.min(lo);
        let d = (lo - a) as u64;
        let hi = hi as u64;
        let hold = self.hold_frames as u64;
        let mask = self.hist_mask;
        let m = self.now;
        self.win_push_delay = d;

        // Replay h for t in [m-1-a, m-1]: the ramp's output depends only on
        // its last a+1 inputs, so this restores its exact state. The window
        // of h[t] is [t − hi − hold, t − d] (hi + hold − d + 1 values);
        // pre-fill the minimum with the w − 1 values before the first
        // window's end.
        let start = m - 1 - a as u64;
        self.hold_from = if self.fresh {
            0
        } else {
            self.hold_from.max(start - self.built_reach)
        };
        self.built_reach = hi + hold;
        self.window_min.reset((hi + hold - d + 1) as usize);
        self.ramp.reset(a, 1.0);
        // Only values before `start − hi` can be masked: they lie in the hold
        // part of every window replayed or pushed from here on, never in a
        // base part (base windows start at t − hi ≥ start − hi).
        let mask_end = self.hold_from.min(start - hi);
        for e in start - hi - hold..start - d {
            let r = if e < mask_end {
                1.0
            } else {
                self.req_hist[e as usize & mask]
            };
            self.window_min.push(r);
        }
        for t in start..m {
            let h = self.window_min.push(self.req_hist[(t - d) as usize & mask]);
            self.ramp.step(h);
        }
    }

    fn start_crossfade_if_needed(&mut self) {
        if self.xfade.is_none() && self.target_lookahead != self.lookahead {
            self.xfade = Some(Crossfade {
                to: self.target_lookahead,
                pos: 0,
            });
            self.rebuild();
        }
    }

    /// Serial gain pipeline for input frame `now` with required gain `req`
    /// (from [`required_gains`]): stores it, advances the
    /// window minimum, ramp and release, and returns the gain for the output
    /// frame `now`.
    #[inline(always)]
    fn gain_step(&mut self, req: f64, enabled: bool, rise: f64) -> f64 {
        let m = self.now;
        let hmask = self.hist_mask;
        let ceil = if enabled { self.ceiling } else { f32::INFINITY };
        self.req_hist[m as usize & hmask] = req;
        self.ceil_hist[m as usize & hmask] = ceil;

        // Sliding minimum of required gain, then the two-box ramp.
        let pushed = m - self.win_push_delay;
        let h = self.window_min.push(self.req_hist[pushed as usize & hmask]);
        let target = self.ramp.step(h);

        // Release.
        let g = self.gain;
        self.gain = if target <= g {
            target
        } else {
            let up = if enabled {
                g + rise * (target - g)
            } else {
                g + self.bypass_step
            };
            if target - up < DENORMAL_FLUSH {
                target
            } else {
                up
            }
        };
        self.gain
    }

    /// Frames with a single delay tap (no cross-fade running): the serial
    /// gain pipeline for every frame, then one bulk delay-line write/read,
    /// then gain and safety clip. `req` holds the frames' required gains.
    /// `N` is the channel count, or 0 for "use `self.channels`".
    #[inline(always)]
    fn run_block<const N: usize>(
        &mut self,
        frames: &mut [f32],
        req: &[f64],
        enabled: bool,
        rise: f64,
    ) {
        let ch = if N == 0 { self.channels } else { N };
        let nf = req.len();
        let frames = &mut frames[..nf * ch];
        let la = self.lookahead as u64;
        let hmask = self.hist_mask;
        let mut gains = [0.0f64; BLOCK_FRAMES];
        let mut lims = [0.0f32; BLOCK_FRAMES];
        for ((g, lim), &r) in gains.iter_mut().zip(&mut lims).zip(req) {
            *g = self.gain_step(r, enabled, rise);
            *lim = self.ceil_hist[(self.now - la) as usize & hmask];
            self.now += 1;
        }
        self.delay.write_frames(frames);
        self.delay
            .read_frames(la as usize + self.extra_delay, frames);
        let mut clips = 0;
        for ((f, &g), &lim) in frames.chunks_exact_mut(ch).zip(&gains).zip(&lims) {
            for y in f {
                *y = safety((g * *y as f64) as f32, lim, &mut clips);
            }
        }
        self.safety_clips += clips;
    }

    /// One frame during a lookahead cross-fade (two delay taps).
    #[inline(never)]
    fn process_frame_xfade(&mut self, frame: &mut [f32], req: f64, enabled: bool, rise: f64) {
        let g = self.gain_step(req, enabled, rise);
        self.delay.write(frame);
        let m = self.now;
        let hmask = self.hist_mask;
        let mut clips = 0;
        let Some(xf) = self.xfade.as_mut() else {
            return;
        };
        xf.pos += 1;
        let (to, pos) = (xf.to, xf.pos);
        let w =
            0.5 - 0.5 * (std::f64::consts::PI * pos as f64 / (self.xfade_frames + 1) as f64).cos();
        let tap_a = m - self.lookahead as u64;
        let tap_b = m - to as u64;
        let lim =
            self.ceil_hist[tap_a as usize & hmask].max(self.ceil_hist[tap_b as usize & hmask]);
        for ((y, &xa), &xb) in frame
            .iter_mut()
            .zip(self.delay.read(self.lookahead + self.extra_delay))
            .zip(self.delay.read(to + self.extra_delay))
        {
            let v = (g * ((1.0 - w) * xa as f64 + w * xb as f64)) as f32;
            *y = safety(v, lim, &mut clips);
        }
        self.safety_clips += clips;
        self.now += 1;
        if pos >= self.xfade_frames {
            // Cross-fade done: the new tap is now the only one.
            self.lookahead = to;
            self.xfade = None;
            self.rebuild();
            self.start_crossfade_if_needed();
        }
    }
}

/// Final guard: clip `v` to `±lim`, counting interventions (NaN → 0).
#[inline]
fn safety(v: f32, lim: f32, clips: &mut u64) -> f32 {
    if v.abs() <= lim {
        v
    } else {
        *clips += 1;
        if v.is_nan() {
            0.0
        } else {
            lim.copysign(v)
        }
    }
}

/// Largest `f32` that is not above `10^(db/20)`.
fn ceiling_f32(db: f32) -> f32 {
    let c64 = db_to_lin(db as f64);
    let mut c = c64 as f32;
    if c as f64 > c64 && c > 0.0 {
        c = f32::from_bits(c.to_bits() - 1);
    }
    c
}

fn ms_to_frames(ms: f64, fs: f64) -> usize {
    let f = (ms * 0.001 * fs).round();
    if f.is_finite() && f > 0.0 {
        f as usize
    } else {
        0
    }
}

impl Processor for Limiter {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        let fs = effective_rate(sample_rate);
        self.sample_rate = fs;
        let pc = self.peak_control.params;
        self.peak_control.set(&pc, fs);
        self.channels = channels;
        if !self.level_forced {
            self.level = detect();
        }
        if channels == 0 {
            self.delay = DelayLine::default();
            self.req_hist = Vec::new();
            return;
        }
        self.max_frames = ms_to_frames(MAX_LOOKAHEAD_MS as f64, fs);
        self.max_hold_frames = ms_to_frames(GAIN_HOLD_MS, fs);
        self.xfade_frames = ms_to_frames(XFADE_MS, fs).max(1);

        // Room for a whole block on top of the longest delay: a block is
        // written before its delayed frames are read back.
        self.delay
            .allocate(self.max_frames + self.extra_delay + BLOCK_FRAMES, channels);

        // History must reach back a + hi + H + 1 ≤ 2·Lmax + Hmax + 1 frames
        // for a rebuild, besides the frame being written. The window minimum
        // (at most Lmax + Hmax + 1 values) fits in the same size.
        let hist = (2 * self.max_frames + self.max_hold_frames + 2).next_power_of_two();
        self.req_hist = vec![1.0; hist];
        self.ceil_hist = vec![f32::INFINITY; hist];
        self.pc_hist = vec![0.0; hist];
        self.hist_mask = hist - 1;
        self.window_min.allocate(hist);
        self.ramp.allocate(self.max_frames);

        self.attack_frames = usize::MAX; // force recomputation
        let p = self.params;
        self.set_params(&p);
        self.reset();
    }

    fn reset(&mut self) {
        if !self.is_configured() {
            return;
        }
        self.delay.clear();
        self.req_hist.fill(1.0);
        self.ceil_hist.fill(f32::INFINITY);
        self.pc_hist.fill(0.0);
        self.gain = 1.0;
        self.now = TIME_ORIGIN;
        self.lookahead = self.target_lookahead;
        self.xfade = None;
        self.safety_clips = 0;
        self.fresh = true;
        self.peak_control.clear();
        self.rebuild();
    }

    fn latency_frames(&self) -> usize {
        self.target_lookahead + self.extra_delay
    }

    fn process(&mut self, samples: &mut [f32]) {
        let ch = self.channels;
        if !self.is_configured() || samples.len() < ch {
            return;
        }
        self.fresh = false;
        if self.shape_dirty {
            self.rebuild();
        }
        self.start_crossfade_if_needed();

        dispatch!(self.level, simd => self.process_blocks(simd, samples, None, None));
    }
}

/// Required gains `min(1, c / p)` for a block of linked peaks (all 1 when
/// disabled), at native `f64` vector width. IEEE division and compare/select
/// only, so every lane equals [`Limiter::required_gain`] bit for bit.
#[inline(always)]
fn required_gains<S: Simd>(simd: S, ceiling: f32, enabled: bool, peaks: &[f32], out: &mut [f64]) {
    let n = S::f64s::LEN;
    if !enabled {
        out.fill(1.0);
        return;
    }
    let c = S::f64s::splat(simd, ceiling as f64);
    let one = S::f64s::splat(simd, 1.0);
    let mut dst = out.chunks_exact_mut(n);
    let mut src = peaks.chunks_exact(n);
    for (o, p) in (&mut dst).zip(&mut src) {
        let p = S::f64s::from_fn(simd, |i| p[i] as f64);
        p.simd_gt(c).select(c / p, one).store_slice(o);
    }
    let c = ceiling as f64;
    for (o, &p) in dst.into_remainder().iter_mut().zip(src.remainder()) {
        let p = p as f64;
        *o = if p > c { c / p } else { 1.0 };
    }
}

impl Limiter {
    /// Block loop, monomorphised per instruction set: per-frame linked peaks
    /// and required gains (SIMD), then frame by frame through the serial gain
    /// pipeline, with bulk delay-line access and gain application.
    /// `external` replaces the measured frame peaks (see
    /// [`process_with_peaks`](Self::process_with_peaks)).
    #[inline(always)]
    fn process_blocks<S: Simd>(
        &mut self,
        simd: S,
        samples: &mut [f32],
        external: Option<&[f32]>,
        drive: Option<&[f32]>,
    ) {
        let ch = self.channels;
        let enabled = self.params.enabled;
        let rise = 1.0 - self.release_coef;
        let mut peaks = [0.0f32; BLOCK_FRAMES];
        let mut req = [0.0f64; BLOCK_FRAMES];
        for (b, block) in samples.chunks_mut(BLOCK_FRAMES * ch).enumerate() {
            let nf = block.len() / ch;
            match external {
                Some(ext) => {
                    let at = b * BLOCK_FRAMES;
                    peaks[..nf].copy_from_slice(&ext[at..at + nf]);
                }
                None => frame_peaks_simd(simd, block, ch, &mut peaks[..nf]),
            }
            required_gains(simd, self.ceiling, enabled, &peaks[..nf], &mut req[..nf]);
            let t0 = self.now as usize;
            let hmask = self.hist_mask;
            if self.peak_control.is_on() && enabled {
                let at = b * BLOCK_FRAMES;
                let drive = drive.map(|d| &d[at..at + nf]);
                self.peak_control
                    .apply(self.ceiling, &peaks[..nf], drive, &mut req[..nf]);
                for (i, &y) in self.peak_control.block_db[..nf].iter().enumerate() {
                    self.pc_hist[(t0 + i) & hmask] = y;
                }
            } else {
                for i in 0..nf {
                    self.pc_hist[(t0 + i) & hmask] = 0.0;
                }
            }
            let mut i = 0;
            // A cross-fade only starts at the top of `process` or right after
            // another one ends, so once none is running the rest of the call
            // takes the block path.
            while i < nf && self.xfade.is_some() {
                let f = &mut block[i * ch..(i + 1) * ch];
                self.process_frame_xfade(f, req[i], enabled, rise);
                i += 1;
            }
            if i < nf {
                let (frames, req) = (&mut block[i * ch..], &req[i..nf]);
                // Mono/stereo get their own monomorphised loop (frame length
                // known at compile time).
                match ch {
                    1 => self.run_block::<1>(frames, req, enabled, rise),
                    2 => self.run_block::<2>(frames, req, enabled, rise),
                    _ => self.run_block::<0>(frames, req, enabled, rise),
                }
            }
        }
    }
}
