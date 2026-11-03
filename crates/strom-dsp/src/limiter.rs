//! Lookahead sample-peak limiter.
//!
//! # Signal flow
//!
//! The audio is delayed by the lookahead `L` frames. The gain is computed from
//! the *undelayed* input, so it is already down when a peak reaches the output.
//! All channels share one gain (linked), computed from the frame peak
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
//! 1. **Sliding minimum** `h[m] = min { r[j] : m − L ≤ j ≤ m − L + A }`
//!    (a window of `A + 1` required gains, ending `L − A` frames in the past, so
//!    it is causal). Computed with a monotonic deque: O(1) amortised per frame.
//! 2. **Two cascaded moving averages** (boxes) of lengths `A1 + 1` and `A2 + 1`
//!    over `h`. Their combined kernel is a trapezoid/triangle spanning offsets
//!    `0 … A`, so a step in `h` becomes a smooth S-shaped ramp of `A` frames.
//! 3. **Release**: `g[n] = t[n]` when the box output `t[n]` is below the previous
//!    gain (follow downwards instantly; the ramp is already smooth), otherwise a
//!    one-pole rise towards `t[n]` with the release time constant, capped at `t[n]`.
//!
//! *Why the ceiling is guaranteed*: the box output at `n` is a weighted mean
//! (non-negative weights summing to 1) of `h[n − k]` for `0 ≤ k ≤ A`. The window
//! of `h[n − k]` is `[n − k − L, n − k − L + A]`, which contains `n − L` for every
//! such `k`. Hence each `h[n − k] ≤ r[n − L]`, so the mean is `≤ r[n − L]`, and the
//! release stage never exceeds the box output. The gain reaches its target
//! exactly when the peak is output, starting `A` frames earlier.
//!
//! To make this hold *exactly* in floating point the box stage works on
//! integers: `h` is quantised downwards to a fixed-point value with 44
//! fractional bits, the running sums are exact, and each mean is an integer
//! division (rounding down). Rounding down only ever lowers the gain. The ceiling
//! itself is the largest `f32` not above `10^(dB/20)`, so the final `f64 → f32`
//! rounding of `g · x` cannot cross it.
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
//!
//! # Safety clip
//!
//! After the gain, each sample is checked against the ceiling of the frame it
//! came from. Mathematically it never triggers (see above); it is a guard
//! against bugs, and every intervention is counted ([`Limiter::safety_clip_count`]).
//! A NaN output sample (only possible from NaN/Inf input) is replaced by 0 and
//! counted.
//!
//! # Denormals
//!
//! The only recursive state is the release gain, which approaches its target
//! from below; once the gap is under 1e-15 it snaps to the target, so it never
//! decays through the subnormal range. The box stage is integer arithmetic.

use crate::util::{db_to_lin, one_pole_coef, DENORMAL_FLUSH};
use crate::Processor;

/// Largest supported lookahead (ms). The delay line is sized for this at
/// `configure` time so lookahead changes never allocate.
pub const MAX_LOOKAHEAD_MS: f32 = 20.0;

/// Length of the cross-fade between delay taps on a lookahead change (ms).
const XFADE_MS: f64 = 5.0;
/// Time for the gain to ramp from 0 to 1 when the limiter is disabled (ms).
const BYPASS_RAMP_MS: f64 = 10.0;
/// Fixed-point fractional bits of the box stage.
const GAIN_FRAC_BITS: u32 = 44;
const GAIN_ONE: u64 = 1 << GAIN_FRAC_BITS;
const GAIN_SCALE: f64 = GAIN_ONE as f64;
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

impl LimiterParams {
    /// Clamp every field into its documented range; non-finite values fall
    /// back to the default.
    fn sanitized(&self) -> Self {
        let d = Self::default();
        let fix = |v: f32, def: f32, lo: f32, hi: f32| {
            if v.is_finite() {
                v.clamp(lo, hi)
            } else {
                def
            }
        };
        Self {
            enabled: self.enabled,
            threshold_db: fix(self.threshold_db, d.threshold_db, -60.0, 0.0),
            attack_ms: fix(self.attack_ms, d.attack_ms, 0.01, 50.0),
            release_ms: fix(self.release_ms, d.release_ms, 1.0, 2000.0),
            lookahead_ms: fix(self.lookahead_ms, d.lookahead_ms, 0.0, MAX_LOOKAHEAD_MS),
        }
    }
}

/// Monotonic deque for a sliding-window minimum, in a preallocated ring.
/// Values are non-decreasing from front to back; the front is the minimum.
#[derive(Debug, Default)]
struct MinDeque {
    time: Vec<u64>,
    value: Vec<f64>,
    mask: usize,
    head: usize,
    len: usize,
}

impl MinDeque {
    fn allocate(&mut self, capacity_pow2: usize) {
        self.time = vec![0; capacity_pow2];
        self.value = vec![1.0; capacity_pow2];
        self.mask = capacity_pow2 - 1;
        self.clear();
    }

    fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    #[inline]
    fn push(&mut self, t: u64, v: f64) {
        // Drop entries that can never be the minimum again.
        while self.len > 0 {
            let back = (self.head + self.len - 1) & self.mask;
            if self.value[back] >= v {
                self.len -= 1;
            } else {
                break;
            }
        }
        let slot = (self.head + self.len) & self.mask;
        self.time[slot] = t;
        self.value[slot] = v;
        self.len += 1;
    }

    /// Remove entries older than `oldest`.
    #[inline]
    fn expire(&mut self, oldest: u64) {
        while self.len > 0 && self.time[self.head] < oldest {
            self.head = (self.head + 1) & self.mask;
            self.len -= 1;
        }
    }

    #[inline]
    fn min(&self) -> f64 {
        if self.len > 0 {
            self.value[self.head]
        } else {
            1.0
        }
    }
}

/// Exact integer moving average over the last `len` values, ring indexed by
/// absolute frame time.
#[derive(Debug, Default)]
struct BoxStage {
    ring: Vec<u64>,
    len: u64,
    sum: u64,
}

impl BoxStage {
    /// Add the value for frame `now` and return the floor of the window mean.
    #[inline]
    fn step(&mut self, now: u64, mask: usize, v: u64) -> u64 {
        let old = self.ring[(now - self.len) as usize & mask];
        self.sum = self.sum + v - old;
        self.ring[now as usize & mask] = v;
        self.sum / self.len
    }
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

    // Delay line (interleaved frames, power-of-two frame count).
    delay: Vec<f32>,
    delay_mask: usize,
    // Per-input-frame history: required gain and ceiling.
    req_hist: Vec<f64>,
    ceil_hist: Vec<f32>,
    hist_mask: usize,

    // Gain pipeline.
    deque: MinDeque,
    win_hi: u64,
    win_push_delay: u64,
    box1: BoxStage,
    box2: BoxStage,
    gain: f64,

    // Running state.
    now: u64,
    lookahead: usize,
    xfade: Option<Crossfade>,
    safety_clips: u64,
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
            delay: Vec::new(),
            delay_mask: 0,
            req_hist: Vec::new(),
            ceil_hist: Vec::new(),
            hist_mask: 0,
            deque: MinDeque::default(),
            win_hi: 0,
            win_push_delay: 0,
            box1: BoxStage::default(),
            box2: BoxStage::default(),
            gain: 1.0,
            now: TIME_ORIGIN,
            lookahead: 0,
            xfade: None,
            safety_clips: 0,
        };
        l.set_params(&LimiterParams::default());
        l
    }

    /// Update parameters. Never allocates. Out-of-range values are clamped.
    pub fn set_params(&mut self, params: &LimiterParams) {
        let p = params.sanitized();
        self.params = p;
        self.ceiling = ceiling_f32(p.threshold_db);
        self.update_derived();
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
        let a1 = a.div_ceil(2);
        let a2 = a / 2;
        let d = (lo - a) as u64;
        let hi = hi as u64;
        let mask = self.hist_mask;
        let m = self.now;
        self.win_hi = hi;
        self.win_push_delay = d;

        // h (quantised) for t in [m-1-a, m-1].
        let start = m - 1 - a as u64;
        self.deque.clear();
        let mut next = start - hi;
        for t in start..m {
            while next + d <= t {
                self.deque.push(next, self.req_hist[next as usize & mask]);
                next += 1;
            }
            self.deque.expire(t - hi);
            self.box1.ring[t as usize & mask] = quantize(self.deque.min());
        }

        // First box means for t in [m-1-a2, m-1].
        let len1 = a1 as u64 + 1;
        let len2 = a2 as u64 + 1;
        let first = m - 1 - a2 as u64;
        let mut s: u64 = (first - a1 as u64..=first)
            .map(|t| self.box1.ring[t as usize & mask])
            .sum();
        self.box2.ring[first as usize & mask] = s / len1;
        for t in first + 1..m {
            s = s + self.box1.ring[t as usize & mask] - self.box1.ring[(t - len1) as usize & mask];
            self.box2.ring[t as usize & mask] = s / len1;
        }
        self.box1.len = len1;
        self.box1.sum = s;
        self.box2.len = len2;
        self.box2.sum = (first..m).map(|t| self.box2.ring[t as usize & mask]).sum();
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

/// Fixed-point gain, rounded down (so it never exceeds the real value).
#[inline]
fn quantize(g: f64) -> u64 {
    // `as` truncates towards zero (floor for g ≥ 0) and saturates; g ∈ [0, 1].
    (g * GAIN_SCALE) as u64
}

impl Processor for Limiter {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        let fs = if sample_rate.is_finite() && sample_rate > 0.0 {
            (sample_rate as f64).min(MAX_SAMPLE_RATE)
        } else {
            48_000.0
        };
        self.sample_rate = fs;
        self.channels = channels;
        if channels == 0 {
            self.delay = Vec::new();
            return;
        }
        self.max_frames = ms_to_frames(MAX_LOOKAHEAD_MS as f64, fs);
        self.xfade_frames = ms_to_frames(XFADE_MS, fs).max(1);

        let delay_frames = (self.max_frames + 1).next_power_of_two();
        self.delay = vec![0.0; delay_frames * channels];
        self.delay_mask = delay_frames - 1;

        // History must reach back 2·Lmax + 1 frames for a rebuild.
        let hist = (2 * self.max_frames + 2).next_power_of_two();
        self.req_hist = vec![1.0; hist];
        self.ceil_hist = vec![f32::INFINITY; hist];
        self.hist_mask = hist - 1;
        self.deque.allocate(hist);
        self.box1.ring = vec![GAIN_ONE; hist];
        self.box2.ring = vec![GAIN_ONE; hist];

        self.attack_frames = usize::MAX; // force recomputation
        let p = self.params;
        self.set_params(&p);
        self.reset();
    }

    fn reset(&mut self) {
        if self.channels == 0 || self.delay.is_empty() {
            return;
        }
        self.delay.fill(0.0);
        self.req_hist.fill(1.0);
        self.ceil_hist.fill(f32::INFINITY);
        self.gain = 1.0;
        self.now = TIME_ORIGIN;
        self.lookahead = self.target_lookahead;
        self.xfade = None;
        self.safety_clips = 0;
        self.rebuild();
    }

    fn latency_frames(&self) -> usize {
        self.target_lookahead
    }

    fn process(&mut self, samples: &mut [f32]) {
        let ch = self.channels;
        if ch == 0 || self.delay.is_empty() {
            return;
        }
        if self.attack_dirty {
            self.rebuild();
        }
        self.start_crossfade_if_needed();

        let enabled = self.params.enabled;
        let ceiling = self.ceiling;
        let c = ceiling as f64;
        let hmask = self.hist_mask;
        let dmask = self.delay_mask;
        let rise = 1.0 - self.release_coef;

        for frame in samples.chunks_exact_mut(ch) {
            let m = self.now;

            // Detector on the undelayed input.
            let mut peak = 0.0f32;
            for &x in frame.iter() {
                peak = peak.max(x.abs());
            }
            let peak = peak as f64;
            let (req, ceil) = if enabled {
                (if peak > c { c / peak } else { 1.0 }, ceiling)
            } else {
                (1.0, f32::INFINITY)
            };
            self.req_hist[m as usize & hmask] = req;
            self.ceil_hist[m as usize & hmask] = ceil;
            let dpos = (m as usize & dmask) * ch;
            self.delay[dpos..dpos + ch].copy_from_slice(frame);

            // Sliding minimum of required gain.
            let pushed = m - self.win_push_delay;
            self.deque
                .push(pushed, self.req_hist[pushed as usize & hmask]);
            self.deque.expire(m - self.win_hi);
            let h = quantize(self.deque.min());

            // Two exact moving averages.
            let s1 = self.box1.step(m, hmask, h);
            let s2 = self.box2.step(m, hmask, s1);
            let target = s2 as f64 / GAIN_SCALE;

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
            let g = self.gain;

            // Delayed output, possibly cross-fading two taps.
            let tap_a = m - self.lookahead as u64;
            let a_pos = (tap_a as usize & dmask) * ch;
            match self.xfade {
                None => {
                    let lim = self.ceil_hist[tap_a as usize & hmask];
                    for (i, y) in frame.iter_mut().enumerate() {
                        let v = (g * self.delay[a_pos + i] as f64) as f32;
                        *y = self.safety(v, lim);
                    }
                }
                Some(ref mut xf) => {
                    xf.pos += 1;
                    let (to, pos) = (xf.to, xf.pos);
                    let w = 0.5
                        - 0.5
                            * (std::f64::consts::PI * pos as f64 / (self.xfade_frames + 1) as f64)
                                .cos();
                    let tap_b = m - to as u64;
                    let b_pos = (tap_b as usize & dmask) * ch;
                    let lim = self.ceil_hist[tap_a as usize & hmask]
                        .max(self.ceil_hist[tap_b as usize & hmask]);
                    for (i, y) in frame.iter_mut().enumerate() {
                        let xa = self.delay[a_pos + i] as f64;
                        let xb = self.delay[b_pos + i] as f64;
                        let v = (g * ((1.0 - w) * xa + w * xb)) as f32;
                        *y = self.safety(v, lim);
                    }
                    if pos >= self.xfade_frames {
                        // Cross-fade done: the new tap is now the only one.
                        self.lookahead = to;
                        self.xfade = None;
                        self.now += 1;
                        self.rebuild();
                        self.start_crossfade_if_needed();
                        continue;
                    }
                }
            }
            self.now += 1;
        }
    }
}

impl Limiter {
    /// Final guard: clip to the ceiling and count interventions.
    #[inline]
    fn safety(&mut self, v: f32, lim: f32) -> f32 {
        if v.abs() <= lim {
            v
        } else {
            self.safety_clips += 1;
            if v.is_nan() {
                0.0
            } else {
                lim.copysign(v)
            }
        }
    }
}
