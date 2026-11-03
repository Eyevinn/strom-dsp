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
//! With attack length `A ≤ L` frames, split as `A = A1 + A2`:
//!
//! 1. **Sliding minimum** ([`SegmentedMin`])
//!    `h[m] = min { r[j] : m − L ≤ j ≤ m − L + A }`
//!    (a window of `A + 1` required gains, ending `L − A` frames in the past, so
//!    it is causal). Computed with the van Herk / Gil–Werman segment algorithm
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
//! of `h[n − k]` is `[n − k − L, n − k − L + A]`, which contains `n − L` for every
//! such `k`. Since each `h` is the exact window minimum (including right after
//! a rebuild, which pre-fills the segment minimum with the full window from the
//! stored history), each `h[n − k] ≤ r[n − L]`, so the mean is `≤ r[n − L]`, and the
//! release stage never exceeds the box output. The gain reaches its target
//! exactly when the peak is output, starting `A` frames earlier.
//!
//! To make this hold *exactly* in floating point the box stage works on
//! integers: `h` is quantised downwards to a fixed-point value with
//! [`RAMP_FRAC_BITS`](crate::dynamics::RAMP_FRAC_BITS) fractional bits, the running sums are exact, and each mean
//! is an integer division (rounding down). Rounding down only ever lowers the
//! gain. The ceiling itself is the largest `f32` not above `10^(dB/20)`, so the
//! final `f64 → f32` rounding of `g · x` cannot cross it.
//!
//! With lookahead 0 the windows degenerate to `A = 0`: the gain drops instantly
//! to `r[n]` (a hard limiter with smooth release).
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
//! - Attack changes rebuild the min/box pipeline from the stored history of
//!   required gains *as if* the new attack had always been in force (O(lookahead),
//!   no allocation), so the guarantee holds immediately.
//! - Lookahead changes cross-fade (raised cosine, 5 ms) from the old to the new
//!   delay tap. During the cross-fade the sliding-minimum window is widened to
//!   cover both taps (`[m − Lhi, m − Llo + A]`), so the gain is at most the required
//!   gain of *both* samples being mixed and the guarantee still holds. A further
//!   change during a cross-fade is queued until it completes.
//! - Between `configure`/`reset` and the first processed frame, `set_params`
//!   applies instantly (no cross-fade from the previous lookahead), so the
//!   result does not depend on whether parameters were set before or after
//!   `configure`.
//!
//! # Safety clip
//!
//! After the gain, each sample is checked against the ceiling of the frame it
//! came from. Mathematically it never triggers (see above); it is a guard
//! against bugs, and every intervention is counted ([`Limiter::safety_clip_count`]).
//! A NaN output sample (only possible from NaN/Inf input) is replaced by 0 and
//! counted.
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
use crate::dynamics::{DelayLine, RampSmoother, SegmentedMin};
use crate::simd::{detect, Level};
use crate::util::{db_to_lin, one_pole_coef, DENORMAL_FLUSH};
use crate::Processor;
use fearless_simd::dispatch;
use fearless_simd::prelude::*;

/// Largest supported lookahead (ms). The delay line is sized for this at
/// `configure` time so lookahead changes never allocate.
pub const MAX_LOOKAHEAD_MS: f32 = 20.0;

/// Length of the cross-fade between delay taps on a lookahead change (ms).
const XFADE_MS: f64 = 5.0;
/// Time for the gain to ramp from 0 to 1 when the limiter is disabled (ms).
const BYPASS_RAMP_MS: f64 = 10.0;
/// Frame counter start, so `now − delay` never underflows.
const TIME_ORIGIN: u64 = 1 << 32;
/// Sample-rate clamp for `configure` (keeps fixed-point sums far from overflow).
const MAX_SAMPLE_RATE: f64 = 768_000.0;

/// Limiter parameters (units as in the GStreamer interface).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LimiterParams {
    /// When false: unity gain, but the lookahead delay is kept (constant latency).
    pub enabled: bool,
    /// Ceiling (dBFS, −60 … 0). No output sample exceeds it.
    pub threshold_db: f32,
    /// Attack time (ms, 0.01 … 50).
    pub attack_ms: f32,
    /// Release time (ms, 1 … 2000).
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
    xfade_frames: usize,
    attack_dirty: bool,

    delay: DelayLine,
    // Per-input-frame history: required gain and ceiling, indexed by frame time.
    req_hist: Vec<f64>,
    ceil_hist: Vec<f32>,
    hist_mask: usize,

    // Gain pipeline.
    window_min: SegmentedMin,
    win_push_delay: u64,
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
            xfade_frames: 1,
            attack_dirty: false,
            delay: DelayLine::default(),
            req_hist: Vec::new(),
            ceil_hist: Vec::new(),
            hist_mask: 0,
            window_min: SegmentedMin::default(),
            win_push_delay: 0,
            ramp: RampSmoother::default(),
            gain: 1.0,
            now: TIME_ORIGIN,
            lookahead: 0,
            xfade: None,
            safety_clips: 0,
            fresh: true,
            level: Level::baseline(),
            level_forced: false,
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
            self.attack_dirty = true;
        }
    }

    /// Current window geometry: (shortest tap, longest tap).
    fn taps(&self) -> (usize, usize) {
        match self.xfade {
            Some(xf) => (self.lookahead.min(xf.to), self.lookahead.max(xf.to)),
            None => (self.lookahead, self.lookahead),
        }
    }

    /// Rebuild the min/box pipeline for the current geometry from the stored
    /// required-gain history, as if the geometry had always been in force.
    /// O(lookahead), no allocation.
    fn rebuild(&mut self) {
        self.attack_dirty = false;
        let (lo, hi) = self.taps();
        let a = self.attack_frames.min(lo);
        let d = (lo - a) as u64;
        let hi = hi as u64;
        let mask = self.hist_mask;
        let m = self.now;
        self.win_push_delay = d;

        // Replay h for t in [m-1-a, m-1]: the ramp's output depends only on
        // its last a+1 inputs, so this restores its exact state. The window
        // of h[t] is [t − hi, t − d] (hi − d + 1 values); pre-fill the
        // minimum with the w − 1 values before the first window's end.
        let start = m - 1 - a as u64;
        self.window_min.reset((hi - d + 1) as usize);
        self.ramp.reset(a, 1.0);
        for e in start - hi..start - d {
            self.window_min.push(self.req_hist[e as usize & mask]);
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
        self.delay.read_frames(la as usize, frames);
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
            .zip(self.delay.read(self.lookahead))
            .zip(self.delay.read(to))
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
        self.xfade_frames = ms_to_frames(XFADE_MS, fs).max(1);

        // Room for a whole block on top of the longest delay: a block is
        // written before its delayed frames are read back.
        self.delay
            .allocate(self.max_frames + BLOCK_FRAMES, channels);

        // History must reach back 2·Lmax + 1 frames for a rebuild.
        let hist = (2 * self.max_frames + 2).next_power_of_two();
        self.req_hist = vec![1.0; hist];
        self.ceil_hist = vec![f32::INFINITY; hist];
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
        self.gain = 1.0;
        self.now = TIME_ORIGIN;
        self.lookahead = self.target_lookahead;
        self.xfade = None;
        self.safety_clips = 0;
        self.fresh = true;
        self.rebuild();
    }

    fn latency_frames(&self) -> usize {
        self.target_lookahead
    }

    fn process(&mut self, samples: &mut [f32]) {
        let ch = self.channels;
        if !self.is_configured() || samples.len() < ch {
            return;
        }
        self.fresh = false;
        if self.attack_dirty {
            self.rebuild();
        }
        self.start_crossfade_if_needed();

        dispatch!(self.level, simd => self.process_blocks(simd, samples));
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
    #[inline(always)]
    fn process_blocks<S: Simd>(&mut self, simd: S, samples: &mut [f32]) {
        let ch = self.channels;
        let enabled = self.params.enabled;
        let rise = 1.0 - self.release_coef;
        let mut peaks = [0.0f32; BLOCK_FRAMES];
        let mut req = [0.0f64; BLOCK_FRAMES];
        for block in samples.chunks_mut(BLOCK_FRAMES * ch) {
            let nf = block.len() / ch;
            frame_peaks_simd(simd, block, ch, &mut peaks[..nf]);
            required_gains(simd, self.ceiling, enabled, &peaks[..nf], &mut req[..nf]);
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
