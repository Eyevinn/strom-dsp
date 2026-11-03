//! True-peak limiter (phase 2): the phase 1 lookahead limiter with its
//! detector on a 4× oversampled signal, so the ceiling is a *true-peak*
//! ceiling in dBTP in the sense of ITU-R BS.1770 Annex 2.
//!
//! # Signal flow
//!
//! ```text
//!         ┌─ 4× up (Oversampler) ─ max|·| per frame ─ max over ±W ─┐
//!         │                                                       (+)─ peak p ─┐
//!         ├─ |Δ⁶x|/64 (high-frequency term) ── max over ±W ── × κ ─┘           │
//! input ──┤                                                                    ▼
//!         └─ delay L + U + W ──────────────── × gain (phase 1 limiter gain core) ── output
//! ```
//!
//! - **Interpolated peak.** [`Oversampler::upsample_peaks`] interpolates the
//!   input 4× (two half-band stages; BS.1770 Annex 2: oversample by 4, take
//!   the absolute maximum) and gives for each base frame the largest `|x|`
//!   over its 4 sub-samples and all channels (linked). The value for frame
//!   `m` covers the input interval `[m − U, m − U + 1)`, where `U`
//!   ([`DETECTOR_DELAY_FRAMES`], 24) is the interpolator's delay.
//! - **High-frequency term.** `κ · max_ch |Δ⁶x| / 64` ([`HF_WEIGHT`]), the
//!   sixth difference, gain `sin⁶(πf/fs)`: 1 at Nyquist, 0.125 at fs/4,
//!   0.047 at 10 kHz/48 kHz, below 0.0013 under fs/10. It covers what a
//!   BS.1770 meter can read *above* our interpolator near Nyquist (see
//!   below). Delayed to line up with the interpolated peak.
//! - **Guard window.** Sliding maxima over `2W + 1` frames
//!   (`W` = [`GUARD_FRAMES`]) of both terms, added:
//!   `p = max_{±W} |x̂| + κ · max_{±W} hf`. It costs `W` frames of latency.
//! - **Gain.** `p` feeds the phase 1 limiter's gain pipeline unchanged
//!   ([`crate::limiter`]: required gain `min(1, c / p)`, sliding minimum with
//!   hold, two-box S-ramp, one-pole release, per-frame ceiling history,
//!   cross-fades, safety clip) through its crate-internal core hooks. The
//!   attack ramp is always the whole lookahead (`A = L`). The audio is delayed
//!   by `L + U + W`, so it lines up with the detector.
//!
//! # Latency
//!
//! `latency = round(lookahead · fs / 1000) + U + W` frames, with `U = 24`
//! (the 4× interpolator) and `W = 8` (the guard window): the true-peak
//! detector's filter delay is `U + W = 32` frames at every rate. It is
//! constant for a configuration and the same when disabled
//! ([`latency_frames_for`]). With the default 5 ms lookahead: 44.1 kHz
//! 221 + 32 = 253, 48 kHz 240 + 32 = 272, 96 kHz 480 + 32 = 512 frames.
//!
//! # The guarantee, and what it rests on
//!
//! Write `c` for the linear ceiling, `s = n − L − U − W` for the input frame
//! that output frame `n` carries, and `x̂(t)` for our 4× interpolation of
//! the input at sub-sample time `t`.
//!
//! **Proven (exact, by construction):**
//!
//! 1. *Per-frame bound.* The gain core's proof ([`crate::limiter`] module
//!    docs) gives `g[n] ≤ c / p[n − L]` exactly, and `p[n − L]` is at least
//!    `max |x̂(t)|` over `t ∈ [s − W, s + W + 1)`. So for every interpolated
//!    time `t`, **every** output frame within `±W` frames of `t` has a gain
//!    `≤ c / |x̂(t)|`: the output's interpolation near `t` only draws on
//!    samples whose gain respects `t`'s peak.
//! 2. *Sample peak.* The half-band stages keep the input samples exactly at
//!    sub-sample 0 (they are Nyquist filters), so `p ≥ |x|` and the output's
//!    *sample* peak never exceeds `c`. The safety clip at `c` (counted,
//!    [`TruePeakLimiter::safety_clip_count`]) therefore never acts on finite
//!    input; the tests confirm it.
//! 3. *Smooth gain.* The attack is the whole lookahead, `A = L ≥ 0.5 ms` (at
//!    least 22 frames), so a gain drop is the two-box S-ramp with slope at
//!    most `2·Δ/A`; the release is a one-pole of at least 1 ms that starts
//!    after the hold. The gain is a slow signal compared with the audio band.
//!
//! **Bounded by design and verified by tests, not proven in closed form.**
//! The meter's reading of the output, `ŷ(t) = Σ_k m(t − k)·g[k]·x[k]` with
//! the meter's interpolator `m`, splits into three parts:
//!
//! - `G·x̂(t)` with `G` the gain at `t`'s frame. This is at most `c` by (1).
//! - *Gain modulation*, `Σ_k m(t − k)(g[k] − G)·x[k]`. By (1) the gains near
//!   `t` are all held down by `t`'s peak, and by (3) they differ only
//!   smoothly. The worst case is a fast drop next to loud high-frequency
//!   material. The tests drive it hardest (shortest lookahead, 1 ms
//!   release, 15 kHz to Nyquist at the ceiling, then +30 dBFS spikes) and
//!   measure at most −0.04 dB relative to the ceiling.
//! - *Interpolator mismatch*, `G·Σ_k (m − h)(t − k)·x[k]`, where `h` is
//!   ours. Both interpolators reproduce low frequencies, so `m − h` acts only
//!   near Nyquist plus a small passband ripple. This is where content above
//!   our passband edge ([`crate::oversample::PASSBAND_EDGE`] = 20/48 of fs,
//!   18.4 kHz at 44.1 kHz) matters. Our half-band interpolator is accurate
//!   below that edge and falls to −6 dB at Nyquist. A short meter
//!   interpolator (the `ebur128` crate: a 48-tap Hann-windowed sinc at 4×
//!   below 96 kHz, 2× at 96 kHz) has a wide transition band across Nyquist,
//!   so images leak and it can read near-Nyquist material *higher* than ours.
//!   Measured with `ebur128`:
//!   - Steady sines: the meter's own ripple reads up to +0.11 dB above
//!     amplitude (around 0.31·fs). This is absorbed by the 0.1 dB tolerance;
//!     ours reads ≤ 0 dB.
//!   - Short full-band noise bursts (clicks): up to +0.86 dB more than ours.
//!     Without the high-frequency term the limiter output then reached
//!     +0.78 dB over the ceiling. The mismatch is linear in `x` and vanishes
//!     at low frequencies, so it is bounded by `κ · max |Δ⁶x| / 64` over the
//!     meter's support. An adversarial search (hill-climbing snippets to
//!     maximise the excess over our reading beyond 0.1 dB) needed
//!     `κ ≤ 0.34` (`Δ²`: 0.28, `Δ⁴`: 0.33, `Δ⁶`: 0.34). `Δ⁶` was chosen
//!     because it costs the least on programme material. `κ = 0.45` leaves
//!     about 30 % headroom.
//!
//! **Margins.** There is no ceiling margin. The 0.1 dB tolerance absorbs the
//! meter's passband ripple; `κ` covers the near-Nyquist mismatch. The cost of
//! `κ` is extra gain reduction only where the peak-setting content is high in
//! frequency: about 0.2 dB for a 10 kHz tone at the ceiling (48 kHz), 1 dB at
//! 15 kHz, 0 for typical programme below ~5 kHz. At 96 kHz the meter needs
//! only `κ ≈ 0.08`, but `κ` is kept rate-independent for simplicity.
//!
//! **Tested** (`tests/truepeak.rs`, `ebur128` in true-peak mode, at 44.1, 48
//! and 96 kHz):
//! - full-scale and +20 dBFS squares;
//! - impulses and impulse trains;
//! - the fs/4 sine at 45°;
//! - sines from 15 kHz to Nyquist at several phases;
//! - full-band and high-passed noise at +20 dBFS;
//! - steps into huge spikes;
//! - random and adversarially optimised clicks;
//! - a property test over signals and parameters.
//!
//! The worst measured output is ceiling + 0.004 dB, on programme-like
//! multitone material: that is the meter's own ripple. On the nasty and
//! adversarial cases it is ceiling − 0.02 dB.
//!
//! # Bypass and parameter changes
//!
//! These come from the gain core:
//! - `enabled = false` ramps the gain to exactly 1 and leaves a bit-exact
//!   delay of the same latency.
//! - A ceiling change takes effect with the audio's latency, through the
//!   normal ramp.
//! - A lookahead change cross-fades the delay tap (5 ms), with the minimum
//!   window covering both taps.
//!
//! The detector always runs, so re-enabling starts from a warm detector.
//!
//! # Real-time safety, robustness
//!
//! `configure` allocates; `process`, `set_params` and `reset` never do. NaN
//! or infinite input reads as an infinite peak (gain 0); the safety stage
//! turns a resulting NaN into 0 and counts it. The detector is FIR only, so
//! it forgets a NaN after its length. No recursive state can go subnormal
//! (see the limiter docs).
use crate::dynamics::SegmentedMin;
use crate::limiter::{lookahead_frames, Limiter, LimiterParams, MAX_LOOKAHEAD_MS};
use crate::oversample::{OversampleFactor, Oversampler};
use crate::simd::Level;
use crate::Processor;

/// Oversampling factor of the detector (BS.1770 Annex 2: 4×).
pub const OVERSAMPLE: OversampleFactor = OversampleFactor::X4;

/// Delay `U` of the detector's 4× interpolator, in base-rate frames (the
/// oversampler's up-path latency for 4×).
pub const DETECTOR_DELAY_FRAMES: usize = 24;

/// Half-width `W` of the guard window, in base-rate frames: each frame's gain
/// is bounded by the largest interpolated peak within `±W` frames. Covers the
/// main lobes of the interpolation kernel, see the module docs.
pub const GUARD_FRAMES: usize = 8;

/// Weight `κ` of the high-frequency term added to the detected peak:
/// `p = max_{±W} |x̂| + κ · max_{±W} |Δ⁶x| / 64`. It bounds how much more a
/// BS.1770 meter's short interpolator can read than ours on content near
/// Nyquist (module docs, "Interpolator mismatch").
pub const HF_WEIGHT: f64 = 0.45;

/// Shortest lookahead (ms): it is also the attack ramp, which must stay
/// smooth for the true-peak bound (module docs, point 3).
pub const MIN_LOOKAHEAD_MS: f32 = 0.5;

/// Base frames per detector chunk (scratch size).
const CHUNK_FRAMES: usize = 256;

/// True-peak limiter parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TruePeakLimiterParams {
    /// When false: unity gain, same delay (constant latency).
    pub enabled: bool,
    /// True-peak ceiling (dBTP, −60 … 0).
    pub ceiling_dbtp: f32,
    /// Release time (ms, 1 … 2000): one-pole time constant of the gain's
    /// rise, after a hold of `min(20 ms, release)`.
    pub release_ms: f32,
    /// Lookahead (ms, 0.5 … 20); also the attack ramp length.
    pub lookahead_ms: f32,
}

impl Default for TruePeakLimiterParams {
    fn default() -> Self {
        Self {
            enabled: true,
            ceiling_dbtp: -1.0,
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

impl TruePeakLimiterParams {
    /// Clamp every field into its documented range; non-finite values fall
    /// back to the default.
    fn sanitized(&self) -> Self {
        let d = Self::default();
        Self {
            enabled: self.enabled,
            ceiling_dbtp: finite_clamp(self.ceiling_dbtp, d.ceiling_dbtp, -60.0, 0.0),
            release_ms: finite_clamp(self.release_ms, d.release_ms, 1.0, 2000.0),
            lookahead_ms: finite_clamp(
                self.lookahead_ms,
                d.lookahead_ms,
                MIN_LOOKAHEAD_MS,
                MAX_LOOKAHEAD_MS,
            ),
        }
    }

    /// The gain core's parameters: attack = lookahead, threshold = ceiling.
    fn core(&self) -> LimiterParams {
        LimiterParams {
            enabled: self.enabled,
            threshold_db: self.ceiling_dbtp,
            attack_ms: self.lookahead_ms,
            release_ms: self.release_ms,
            lookahead_ms: self.lookahead_ms,
        }
    }
}

/// Latency in frames of a [`TruePeakLimiter`] configured at `sample_rate`
/// with `lookahead_ms`: `round(lookahead · fs / 1000) + U + W`, after the same
/// clamping as [`TruePeakLimiterParams`]. Exactly what
/// [`Processor::latency_frames`] reports.
pub fn latency_frames_for(lookahead_ms: f32, sample_rate: f32) -> usize {
    let d = TruePeakLimiterParams::default();
    let la = finite_clamp(
        lookahead_ms,
        d.lookahead_ms,
        MIN_LOOKAHEAD_MS,
        MAX_LOOKAHEAD_MS,
    );
    lookahead_frames(la, sample_rate) + DETECTOR_DELAY_FRAMES + GUARD_FRAMES
}

/// Linked-channel lookahead limiter with a 4× oversampled (true-peak)
/// detector. See the module documentation.
#[derive(Debug)]
pub struct TruePeakLimiter {
    params: TruePeakLimiterParams,
    core: Limiter,
    detector: Oversampler,
    guard: SegmentedMin,
    guard_hf: SegmentedMin,
    peaks: Vec<f32>,
    /// Previous `HF_ORDER` input samples per channel, newest first.
    prev: Vec<f32>,
    /// High-frequency term, delayed to line up with the interpolator.
    hf_ring: [f64; HF_RING_MASK + 1],
    /// Input frame counter (ring index).
    t: u64,
    channels: usize,
}

/// Order of the difference in the high-frequency term (`Δ⁶`).
const HF_ORDER: usize = 6;
/// Binomial taps of `Δ⁶ / 2⁶`, newest sample first.
const HF_TAPS: [f64; HF_ORDER + 1] = [
    1.0 / 64.0,
    -6.0 / 64.0,
    15.0 / 64.0,
    -20.0 / 64.0,
    15.0 / 64.0,
    -6.0 / 64.0,
    1.0 / 64.0,
];
/// Delay of the high-frequency term so its stencil (centred `HF_ORDER / 2`
/// frames back) lines up with the interpolator's output (`U` frames back).
const HF_DELAY: usize = DETECTOR_DELAY_FRAMES - HF_ORDER / 2;
/// Ring for that delay.
const HF_RING_MASK: usize = 31;
const _: () = assert!(HF_DELAY <= HF_RING_MASK);

/// High-frequency term of one input frame, linked over channels:
/// `max_ch |Δ⁶x[n]| / 64`, the sixth difference, whose gain is
/// `sin⁶(π f / fs)`: 1 at Nyquist, 0.125 at fs/4, 0.047 at 0.21·fs (10 kHz at
/// 48 kHz), below 0.0013 under fs/10. Updates the per-channel history
/// (`HF_ORDER` samples each). Non-finite input reads as infinite.
#[inline(always)]
fn hf_term(frame: &[f32], prev: &mut [f32]) -> f64 {
    let mut m = 0.0f64;
    for (&x, h) in frame.iter().zip(prev.chunks_exact_mut(HF_ORDER)) {
        let mut acc = HF_TAPS[0] * x as f64;
        for (c, &v) in HF_TAPS[1..].iter().zip(h.iter()) {
            acc += c * v as f64;
        }
        let d = acc.abs();
        m = if d > m || d.is_nan() { d } else { m };
        h.copy_within(0..HF_ORDER - 1, 1);
        h[0] = x;
    }
    if m.is_finite() {
        m
    } else {
        f64::INFINITY
    }
}

impl Default for TruePeakLimiter {
    fn default() -> Self {
        Self::new()
    }
}

impl TruePeakLimiter {
    /// Create an unconfigured limiter with default parameters.
    pub fn new() -> Self {
        let mut core = Limiter::new();
        core.set_extra_delay(DETECTOR_DELAY_FRAMES + GUARD_FRAMES);
        let mut l = Self {
            params: TruePeakLimiterParams::default(),
            core,
            detector: Oversampler::new(),
            guard: SegmentedMin::default(),
            guard_hf: SegmentedMin::default(),
            peaks: Vec::new(),
            prev: Vec::new(),
            hf_ring: [0.0; HF_RING_MASK + 1],
            t: 0,
            channels: 0,
        };
        l.set_params(&TruePeakLimiterParams::default());
        l
    }

    /// Update parameters. Never allocates. Out-of-range values are clamped.
    pub fn set_params(&mut self, params: &TruePeakLimiterParams) {
        let p = params.sanitized();
        self.params = p;
        self.core.set_params(&p.core());
    }

    /// Current (sanitised) parameters.
    pub fn params(&self) -> &TruePeakLimiterParams {
        &self.params
    }

    /// Number of times the final safety stage changed a sample since the
    /// last `reset`/`configure`. Expected to stay 0 on any finite input.
    pub fn safety_clip_count(&self) -> u64 {
        self.core.safety_clip_count()
    }

    /// The linear ceiling the detector limits to (largest `f32` not above
    /// `10^(ceiling/20)`); the sample-peak safety stage uses it too.
    pub fn ceiling_linear(&self) -> f32 {
        self.core.ceiling_linear()
    }

    /// Gain (linear) applied to the most recently output frame.
    pub fn current_gain(&self) -> f64 {
        self.core.current_gain()
    }

    /// Gain reduction (dB, ≥ 0) of the most recently output frame.
    pub fn gain_reduction_db(&self) -> f64 {
        let g = self.core.current_gain();
        if g >= 1.0 {
            0.0
        } else if g > 1e-12 {
            -20.0 * g.log10()
        } else {
            240.0
        }
    }

    /// Force the SIMD instruction set of the gain core and the detector (see
    /// [`Limiter::set_simd_level`]). Never allocates.
    pub fn set_simd_level(&mut self, level: Level) {
        self.core.set_simd_level(level);
        self.detector.set_simd_level(level);
    }

    /// The SIMD instruction set `process` currently dispatches to.
    pub fn simd_level(&self) -> Level {
        self.core.simd_level()
    }

    fn is_configured(&self) -> bool {
        self.channels != 0
    }
}

impl Processor for TruePeakLimiter {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        self.channels = channels;
        self.core.configure(sample_rate, channels);
        if channels == 0 {
            return;
        }
        self.detector.configure(OVERSAMPLE, channels);
        debug_assert_eq!(self.detector.up_latency_frames(), DETECTOR_DELAY_FRAMES);
        self.detector.set_simd_level(self.core.simd_level());
        self.guard.allocate(2 * GUARD_FRAMES + 1);
        self.guard_hf.allocate(2 * GUARD_FRAMES + 1);
        self.peaks = vec![0.0; CHUNK_FRAMES];
        self.prev = vec![0.0; HF_ORDER * channels];
        self.reset();
    }

    fn reset(&mut self) {
        if !self.is_configured() {
            return;
        }
        self.detector.reset();
        self.guard.reset(2 * GUARD_FRAMES + 1);
        self.guard_hf.reset(2 * GUARD_FRAMES + 1);
        self.prev.fill(0.0);
        self.hf_ring.fill(0.0);
        self.t = 0;
        self.core.reset();
    }

    fn latency_frames(&self) -> usize {
        self.core.latency_frames()
    }

    fn process(&mut self, samples: &mut [f32]) {
        let ch = self.channels;
        if !self.is_configured() || samples.len() < ch {
            return;
        }
        let frames = samples.len() / ch;
        for chunk in samples[..frames * ch].chunks_mut(CHUNK_FRAMES * ch) {
            let nf = chunk.len() / ch;
            let peaks = &mut self.peaks[..nf];
            self.detector.upsample_peaks(chunk, peaks);
            for (p, frame) in peaks.iter_mut().zip(chunk.chunks_exact(ch)) {
                // High-frequency term of this input frame, delayed by U to
                // line up with the interpolator's output.
                let hf = hf_term(frame, &mut self.prev);
                let at = self.t as usize & HF_RING_MASK;
                let hf_aligned = self.hf_ring[at];
                self.hf_ring[(self.t as usize + HF_DELAY) & HF_RING_MASK] = hf;
                self.t = self.t.wrapping_add(1);
                // Sliding maxima over ±W frames (as minima of negated values:
                // exact). NaN reads as an infinite peak.
                let tp = if p.is_nan() { f32::INFINITY } else { *p };
                let tp = -self.guard.push(-(tp as f64));
                let hf = -self.guard_hf.push(-hf_aligned);
                *p = (tp + HF_WEIGHT * hf) as f32;
            }
            self.core.process_with_peaks(chunk, peaks);
        }
    }
}
