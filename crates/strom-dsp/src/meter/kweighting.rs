//! Real-time BS.1770 loudness detector: K-weighting designed for any sample
//! rate, and momentary / short-term loudness from running sums over 10 ms
//! sub-blocks, cheap enough to read after every audio block.
//!
//! # Why not [`LoudnessMeter`](super::LoudnessMeter)
//!
//! The `ebur128` wrapper is the reference measurement, but it does not suit a
//! detector that drives a gain every block (the broadcast chain's AGC):
//!
//! - **Read cost.** It stores the filtered audio and sums the whole window on
//!   every read: about 20 µs for momentary and 150 µs for short-term at 48 kHz
//!   stereo (`benches/meter.rs`). Read every 10 ms block, the short-term read
//!   alone costs 1.5 % of a core, half the 3 % budget of the whole broadcast chain,
//!   and the cost grows with the sample rate.
//! - **No per-block energies.** It only returns whole-window values. The AGC
//!   gate (see [`crate::agc`]) needs the energy of each short block to decide
//!   which blocks count, which the crate does not expose.
//! - **Fixed windows.** Only 400 ms and 3 s exist; the AGC window is
//!   configurable.
//!
//! It is allocation-free in histogram mode, so the problem is cost and
//! interface, not real-time safety. [`LoudnessDetector`] keeps one energy per
//! 10 ms sub-block in a ring and running sums over it, so every reading is
//! O(1). The tests check it against `ebur128` (within 0.2 LU on noise).
//!
//! # K-weighting for any sample rate
//!
//! ITU-R BS.1770-4/-5, Annex 1, defines K-weighting as two biquads in series
//! (section 2.1, figure 2 and tables 1 and 2): a "pre-filter" (a high shelf
//! of about +4 dB modelling the acoustic effect of the head) and the "RLB"
//! revised low-frequency B-curve high-pass. It publishes the coefficients for
//! 48 kHz only ([`BS1770_PRE_FILTER_48K`], [`BS1770_RLB_48K`]). For other
//! rates we recover an analog prototype from the 48 kHz table and transform
//! it back with the bilinear transform, pre-warped at the prototype's pole
//! frequency (Oppenheim & Schafer, *Discrete-Time Signal Processing*, bilinear
//! transform; J. O. Smith III, *Introduction to Digital Filters*).
//!
//! **Prototype.** Every analog second-order section can be written in the
//! frequency `p = s/Ω0`, normalised to its pole frequency `Ω0`, as
//!
//! ```text
//! H(p) = (V_H·p² + V_B·p/Q + V_L) / (p² + p/Q + 1)
//! ```
//!
//! with `V_L = H(0)` (DC gain), `V_H = H(∞)` (high-frequency gain) and `V_B`
//! the gain of the band term. See [`AnalogBiquad`].
//!
//! **Bilinear transform with pre-warping.** `p = (1/K)·(1 − z⁻¹)/(1 + z⁻¹)`
//! with `K = tan(π·f0/fs)` maps `p = j` exactly to `f0`. Multiplying numerator
//! and denominator by `K²·(1 + z⁻¹)²` gives the un-normalised coefficients
//!
//! ```text
//! a0 = 1 + K/Q + K²            b0 = V_H + V_B·K/Q + V_L·K²
//! a1 = 2·(K² − 1)              b1 = 2·(V_L·K² − V_H)
//! a2 = 1 − K/Q + K²            b2 = V_H − V_B·K/Q + V_L·K²
//! ```
//!
//! (then divided by `a0`; [`AnalogBiquad::to_digital`]).
//!
//! **Inverse** ([`AnalogBiquad::from_digital`]). For normalised `a1, a2`:
//! `1 + a1 + a2 = 4K²/a0`, `1 − a1 + a2 = 4/a0` and `1 − a2 = 2(K/Q)/a0`, so
//!
//! ```text
//! K² = (1 + a1 + a2)/(1 − a1 + a2)       f0 = (fs/π)·atan K
//! Q  = K·(1 − a1 + a2) / (2·(1 − a2))
//! V_L = (b0 + b1 + b2)/(1 + a1 + a2)     (the digital gain at DC)
//! V_H = (b0 − b1 + b2)/(1 − a1 + a2)     (the digital gain at Nyquist)
//! V_B = (b0 − b2)/(1 − a2)
//! ```
//!
//! Applied to the BS.1770 tables at 48 kHz this gives
//!
//! | Stage | f0 | Q | V_H | V_B | V_L |
//! |---|---|---|---|---|---|
//! | pre-filter | 1681.97 Hz | 0.7072 | 1.5849 (+4.00 dB) | 1.2587 | 1.0000 |
//! | RLB | 38.135 Hz | 0.5003 | 1.0050 | 0 | 0 |
//!
//! The transform is a bijection for a fixed `K`, so [`k_weighting`] at
//! 48 kHz reproduces the table to rounding (tested to 1e-12), and at any rate
//! the digital magnitude equals the prototype at the warped frequency,
//! `|H_d(e^{jω})| = |H(j·tan(ω/2)/K)|` (tested). Because pre-warping pins the
//! pole frequency, the response at other rates stays within 0.1 dB of the
//! 48 kHz reference below 16 kHz for 44.1 kHz and above (tested; see
//! `tests/loudness_detector.rs` for the measured deviations at each rate).
//! The prototype values are derived at run time from the table constants,
//! not rounded. `f0` is limited to `0.45·fs` so the pre-filter stays
//! designable below about 3.7 kHz sample rate (8 kHz and up is supported).
//!
//! # Loudness (BS.1770 eq. 2, EBU Tech 3341 windows)
//!
//! `L = −0.691 + 10·log10(Σ_i G_i·z_i)` where `z_i` is the mean square of the
//! K-weighted channel `i` over the window and `G_i` its weight
//! ([`ChannelRole::weight`]: 1.0 for L/R/C and unknown positions, 1.41 for
//! surrounds when the layout is known, 0 for LFE). The constant −0.691
//! compensates the K-weighting gain at 1 kHz, so a 0 dBFS 1 kHz sine in one
//! front channel reads −3.01 LUFS (BS.1770 Annex 1).
//!
//! # Running windows
//!
//! - Each **sub-block** is `round(fs/100)` frames (10 ms at 48 kHz, 44.1 kHz
//!   and every rate that is a multiple of 100 Hz). Its energy is the weighted
//!   channel sum of the K-weighted squares, accumulated in `f64`, divided by
//!   the sub-block length.
//! - A ring holds the last [`SHORT_TERM_SUB_BLOCKS`] + 1 sub-block energies.
//!   Momentary is the mean of the last [`MOMENTARY_SUB_BLOCKS`] (400 ms),
//!   short-term of the last [`SHORT_TERM_SUB_BLOCKS`] (3 s). Sub-blocks
//!   before the first `process` count as silence, as in EBU Tech 3341.
//! - Each window keeps a **running sum**: add the new sub-block, subtract the
//!   one that leaves. To stop rounding drift, each sum is recomputed exactly
//!   from the ring every 128 sub-blocks (1.28 s): O(window) work amortised to
//!   a few additions per sub-block, so drift never accumulates beyond 128
//!   updates. Each sum also carries a bound on its accumulated rounding error
//!   (`ε·(|sum| + new + leaving)` per update); when a loud passage leaves the
//!   window and the bound exceeds 10⁻¹² of the remaining sum, the sum is
//!   recomputed at once, so the cancellation residue never shows in the quiet
//!   reading that follows. Sums are clamped to ≥ 0.
//! - Readings reflect completed sub-blocks only, so they lag by at most one
//!   sub-block (10 ms) and are exact (up to rounding) at sub-block boundaries.
//!
//! # Real-time behaviour
//!
//! [`LoudnessDetector::configure`] allocates; everything else is
//! allocation-free (counting-allocator and RTSan tests). The filters run on
//! the [`crate::biquad`] Direct Form I kernels (`f64` state, channels as SIMD lanes,
//! both stages in series per sample, denormal flush) over a 256-frame
//! scratch copy of the input; non-finite input samples are replaced by 0 in
//! the copy.

use fearless_simd::{Level, Simd};

use super::{ChannelLayout, ChannelRole, MeterError};
use crate::biquad::{run_chain, sanitize_bank, BiquadCoefficients, SteadyStage, STATE_PER_CHANNEL};

/// ITU-R BS.1770 stage 1 ("pre-filter", high shelf) at 48 kHz, as published
/// in the standard (Annex 1, table 1).
pub const BS1770_PRE_FILTER_48K: BiquadCoefficients = BiquadCoefficients {
    b0: 1.535_124_859_586_97,
    b1: -2.691_696_189_406_38,
    b2: 1.198_392_810_852_85,
    a1: -1.690_659_293_182_41,
    a2: 0.732_480_774_215_85,
};

/// ITU-R BS.1770 stage 2 ("RLB" high-pass) at 48 kHz, as published in the
/// standard (Annex 1, table 2).
pub const BS1770_RLB_48K: BiquadCoefficients = BiquadCoefficients {
    b0: 1.0,
    b1: -2.0,
    b2: 1.0,
    a1: -1.990_047_454_833_98,
    a2: 0.990_072_250_366_21,
};

/// The constant of BS.1770 eq. 2: `L = −0.691 + 10·log10(Σ G·z)`.
pub const LOUDNESS_OFFSET_DB: f64 = -0.691;

/// Loudness reported for digital silence (and the floor of every reading), in
/// LUFS. Finite, so readings never carry −∞ into arithmetic.
pub const SILENCE_LUFS: f64 = -200.0;

/// Nominal sub-block length in ms (`round(fs/100)` frames).
pub const SUB_BLOCK_MS: f64 = 10.0;

/// Sub-blocks in the momentary window (400 ms, EBU Tech 3341).
pub const MOMENTARY_SUB_BLOCKS: usize = 40;

/// Sub-blocks in the short-term window (3 s, EBU Tech 3341).
pub const SHORT_TERM_SUB_BLOCKS: usize = 300;

/// Sample-rate range the detector clamps to (Hz).
pub const DETECTOR_RATE_RANGE: (f32, f32) = (8_000.0, 384_000.0);

/// Frames filtered per kernel call (size of the scratch copy).
const CHUNK_FRAMES: usize = 256;

/// Running sums are recomputed exactly every this many pushes.
const RESUM_INTERVAL: u32 = 128;

/// A running sum is also recomputed as soon as its accumulated rounding-error
/// bound exceeds this fraction of the sum (when a loud passage leaves the
/// window, the residue of the subtractions would otherwise dominate).
const MAX_RELATIVE_ERROR: f64 = 1e-12;

/// Convert a weighted mean square (`Σ G·z`) to loudness in LUFS (BS.1770
/// eq. 2). Zero, negative or non-finite input gives [`SILENCE_LUFS`].
#[inline]
pub fn lufs_from_mean_square(mean_square: f64) -> f64 {
    if mean_square > 0.0 && mean_square.is_finite() {
        (LOUDNESS_OFFSET_DB + 10.0 * mean_square.log10()).max(SILENCE_LUFS)
    } else {
        SILENCE_LUFS
    }
}

/// Inverse of [`lufs_from_mean_square`]: the weighted mean square that reads
/// as `lufs`.
#[inline]
pub fn mean_square_from_lufs(lufs: f64) -> f64 {
    10f64.powf((lufs - LOUDNESS_OFFSET_DB) / 10.0)
}

/// An analog second-order section in the normalised frequency `p = s/Ω0`:
/// `H(p) = (V_H·p² + V_B·p/Q + V_L) / (p² + p/Q + 1)`. See the module docs for
/// the bilinear transform in both directions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnalogBiquad {
    /// Pole frequency `f0` in Hz.
    pub f0_hz: f64,
    /// Pole quality factor.
    pub q: f64,
    /// `V_H`: gain as `p → ∞`.
    pub high_gain: f64,
    /// `V_B`: weight of the band (`p/Q`) term.
    pub band_gain: f64,
    /// `V_L`: gain at DC.
    pub low_gain: f64,
}

impl AnalogBiquad {
    /// Recover the prototype from digital coefficients at `sample_rate` (Hz),
    /// inverting the pre-warped bilinear transform. Expects a stable section
    /// with complex or real poles strictly inside the unit circle and
    /// `0 < f0 < fs/2`.
    pub fn from_digital(c: &BiquadCoefficients, sample_rate: f64) -> Self {
        let sum = 1.0 + c.a1 + c.a2;
        let alt = 1.0 - c.a1 + c.a2;
        let k = (sum / alt).sqrt();
        Self {
            f0_hz: sample_rate / core::f64::consts::PI * k.atan(),
            q: k * alt / (2.0 * (1.0 - c.a2)),
            high_gain: (c.b0 - c.b1 + c.b2) / alt,
            band_gain: (c.b0 - c.b2) / (1.0 - c.a2),
            low_gain: (c.b0 + c.b1 + c.b2) / sum,
        }
    }

    /// The pre-warping constant `K = tan(π·f0/fs)`, with `f0` limited to
    /// `0.45·fs`.
    pub fn warp(&self, sample_rate: f64) -> f64 {
        let f0 = self.f0_hz.min(0.45 * sample_rate);
        (core::f64::consts::PI * f0 / sample_rate).tan()
    }

    /// Digital coefficients at `sample_rate` (Hz): bilinear transform
    /// pre-warped at `f0`.
    pub fn to_digital(&self, sample_rate: f64) -> BiquadCoefficients {
        let k = self.warp(sample_rate);
        let k2 = k * k;
        let kq = k / self.q;
        let (vh, vb, vl) = (self.high_gain, self.band_gain, self.low_gain);
        BiquadCoefficients::from_unnormalized(
            vh + vb * kq + vl * k2,
            2.0 * (vl * k2 - vh),
            vh - vb * kq + vl * k2,
            1.0 + kq + k2,
            2.0 * (k2 - 1.0),
            1.0 - kq + k2,
        )
    }

    /// Magnitude `|H(j·w)|` at the normalised analog frequency `w = Ω/Ω0`.
    pub fn magnitude_normalized(&self, w: f64) -> f64 {
        let (nr, ni) = (
            self.low_gain - self.high_gain * w * w,
            self.band_gain * w / self.q,
        );
        let (dr, di) = (1.0 - w * w, w / self.q);
        ((nr * nr + ni * ni) / (dr * dr + di * di)).sqrt()
    }

    /// Magnitude of the prototype at the frequency that the pre-warped
    /// bilinear transform at `sample_rate` maps to `freq_hz`:
    /// `|H(j·tan(π f/fs)/K)|`. Equals the digital section's magnitude.
    pub fn magnitude_warped(&self, freq_hz: f64, sample_rate: f64) -> f64 {
        let w = (core::f64::consts::PI * freq_hz / sample_rate).tan() / self.warp(sample_rate);
        self.magnitude_normalized(w)
    }
}

/// Analog prototypes of the two K-weighting stages (pre-filter, RLB),
/// recovered from the BS.1770 48 kHz tables.
pub fn k_weighting_prototypes() -> [AnalogBiquad; 2] {
    [
        AnalogBiquad::from_digital(&BS1770_PRE_FILTER_48K, 48_000.0),
        AnalogBiquad::from_digital(&BS1770_RLB_48K, 48_000.0),
    ]
}

/// K-weighting coefficients (pre-filter, RLB) at `sample_rate` (Hz).
pub fn k_weighting(sample_rate: f64) -> [BiquadCoefficients; 2] {
    let [pre, rlb] = k_weighting_prototypes();
    [pre.to_digital(sample_rate), rlb.to_digital(sample_rate)]
}

/// Ring of per-sub-block values (newest last). Unwritten slots read as 0.
#[derive(Debug, Clone, Default)]
pub(crate) struct History {
    values: Vec<f64>,
    next: usize,
}

impl History {
    /// Allocate `capacity` slots (≥ 1), all 0.
    pub(crate) fn allocate(&mut self, capacity: usize) {
        self.values = vec![0.0; capacity.max(1)];
        self.next = 0;
    }

    pub(crate) fn capacity(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn clear(&mut self) {
        self.values.fill(0.0);
        self.next = 0;
    }

    pub(crate) fn push(&mut self, v: f64) {
        let cap = self.values.len();
        if cap == 0 {
            return;
        }
        self.values[self.next] = v;
        self.next = if self.next + 1 == cap {
            0
        } else {
            self.next + 1
        };
    }

    /// The value pushed `age` sub-blocks ago (0 = newest); 0 beyond capacity.
    pub(crate) fn back(&self, age: usize) -> f64 {
        let cap = self.values.len();
        if age >= cap {
            return 0.0;
        }
        self.values[(self.next + cap - 1 - age) % cap]
    }

    /// Exact sum of the newest `len` values.
    pub(crate) fn sum_newest(&self, len: usize) -> f64 {
        (0..len.min(self.values.len())).map(|i| self.back(i)).sum()
    }
}

/// Running sum of the newest `len` values of a [`History`] whose capacity is
/// larger than `len`, recomputed exactly every [`RESUM_INTERVAL`] pushes.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct WindowSum {
    len: usize,
    sum: f64,
    /// Bound on the rounding error accumulated since the last exact sum.
    err: f64,
    since_resum: u32,
}

impl WindowSum {
    pub(crate) fn new(len: usize) -> Self {
        Self {
            len,
            sum: 0.0,
            err: 0.0,
            since_resum: 0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Change the window length and recompute the sum (O(len)).
    pub(crate) fn set_len(&mut self, len: usize, h: &History) {
        self.len = len.min(h.capacity().saturating_sub(1));
        self.resum(h);
    }

    pub(crate) fn resum(&mut self, h: &History) {
        self.sum = h.sum_newest(self.len);
        self.err = 0.0;
        self.since_resum = 0;
    }

    /// Update after `h.push(v)`.
    pub(crate) fn pushed(&mut self, h: &History) {
        self.since_resum += 1;
        let new = h.back(0);
        let leaving = h.back(self.len);
        let next = self.sum + new - leaving;
        // Each addition and subtraction rounds by at most ε of its operands.
        self.err += f64::EPSILON * (self.sum.abs() + new + leaving);
        // Resum periodically, and at once when the accumulated error could
        // exceed the tolerance relative to what remains (a loud passage
        // leaving the window).
        if self.since_resum >= RESUM_INTERVAL || self.err > MAX_RELATIVE_ERROR * next {
            self.resum(h);
        } else {
            self.sum = next;
        }
    }

    /// The sum, clamped to ≥ 0.
    pub(crate) fn sum(&self) -> f64 {
        self.sum.max(0.0)
    }
}

/// Real-time BS.1770 loudness detector: K-weighting for any sample rate,
/// BS.1770 channel weights, and momentary (400 ms) and short-term (3 s)
/// loudness readable in O(1) after every block. See the module docs.
///
/// Usage: [`new`](Self::new), [`configure`](Self::configure) (allocates),
/// then [`process`](Self::process) and the readings, all allocation-free.
#[derive(Debug, Clone)]
pub struct LoudnessDetector {
    configured: bool,
    sample_rate: f64,
    channels: usize,
    level: Level,
    coefs: [BiquadCoefficients; 2],
    /// Two DF-I state banks (pre-filter, RLB).
    states: Vec<f64>,
    scratch: Vec<f32>,
    weights: Vec<f64>,
    chan_acc: Vec<f64>,
    sub_len: usize,
    sub_pos: usize,
    sub_acc: f64,
    sub_blocks: u64,
    history: History,
    momentary: WindowSum,
    short_term: WindowSum,
}

impl Default for LoudnessDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl LoudnessDetector {
    /// An unconfigured detector (`process` is a no-op; readings are silence).
    pub fn new() -> Self {
        Self {
            configured: false,
            sample_rate: 48_000.0,
            channels: 1,
            level: crate::simd::detect(),
            coefs: [BiquadCoefficients::IDENTITY; 2],
            states: Vec::new(),
            scratch: Vec::new(),
            weights: Vec::new(),
            chan_acc: Vec::new(),
            sub_len: 480,
            sub_pos: 0,
            sub_acc: 0.0,
            sub_blocks: 0,
            history: History::default(),
            momentary: WindowSum::new(MOMENTARY_SUB_BLOCKS),
            short_term: WindowSum::new(SHORT_TERM_SUB_BLOCKS),
        }
    }

    /// Set the sample rate (Hz, clamped to [`DETECTOR_RATE_RANGE`]; NaN means
    /// 48 kHz), the channel count (≥ 1) and the layout (BS.1770 weights; see
    /// [`ChannelLayout`]). Allocates and resets.
    pub fn configure(&mut self, sample_rate: f32, channels: usize, layout: ChannelLayout) {
        let fs = if sample_rate.is_nan() {
            48_000.0
        } else {
            sample_rate.clamp(DETECTOR_RATE_RANGE.0, DETECTOR_RATE_RANGE.1)
        };
        let ch = channels.max(1);
        self.sample_rate = f64::from(fs);
        self.channels = ch;
        self.coefs = k_weighting(self.sample_rate);
        self.states = vec![0.0; 2 * STATE_PER_CHANNEL * ch];
        self.scratch = vec![0.0; CHUNK_FRAMES * ch];
        self.weights = vec![1.0; ch];
        self.chan_acc = vec![0.0; ch];
        self.sub_len = ((self.sample_rate * SUB_BLOCK_MS * 0.001).round() as usize).max(1);
        self.history.allocate(SHORT_TERM_SUB_BLOCKS + 1);
        self.configured = true;
        self.set_layout(layout);
        self.reset();
    }

    /// Sample rate in Hz after clamping.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Configured channel count.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Override the SIMD level of the filter kernels (hosts, tests).
    pub fn set_simd_level(&mut self, level: Level) {
        self.level = level;
    }

    /// The K-weighting coefficients in use (pre-filter, RLB).
    pub fn coefficients(&self) -> [BiquadCoefficients; 2] {
        self.coefs
    }

    /// Use the BS.1770 weights of `layout` (a layout whose channel count does
    /// not match counts as `Unknown`: every channel 1.0). Allocation-free; the
    /// measurement continues with the new weights from the next sample.
    pub fn set_layout(&mut self, layout: ChannelLayout) {
        let ch = self.weights.len();
        for (c, w) in self.weights.iter_mut().enumerate() {
            *w = layout.role(c, ch).weight();
        }
    }

    /// Override the per-channel roles (one per channel; `DualMono` only on a
    /// 1-channel detector). Allocation-free.
    pub fn set_channel_roles(&mut self, roles: &[ChannelRole]) -> Result<(), MeterError> {
        if roles.len() != self.weights.len()
            || (roles.len() > 1 && roles.contains(&ChannelRole::DualMono))
        {
            return Err(MeterError::InvalidChannelRoles);
        }
        for (w, r) in self.weights.iter_mut().zip(roles) {
            *w = r.weight();
        }
        Ok(())
    }

    /// Per-channel BS.1770 weights in use.
    pub fn channel_weights(&self) -> &[f64] {
        &self.weights
    }

    /// Clear the filter state and all windows (readings become silence).
    pub fn reset(&mut self) {
        self.states.fill(0.0);
        self.chan_acc.fill(0.0);
        self.sub_pos = 0;
        self.sub_acc = 0.0;
        self.sub_blocks = 0;
        self.history.clear();
        self.momentary.resum(&self.history);
        self.short_term.resum(&self.history);
    }

    /// Frames per sub-block (`round(fs/100)`).
    pub fn sub_block_frames(&self) -> usize {
        self.sub_len
    }

    /// Frames still needed to complete the current sub-block (1 … sub-block
    /// length). Callers that act on every sub-block split their buffers here.
    pub fn frames_to_sub_block_end(&self) -> usize {
        self.sub_len - self.sub_pos
    }

    /// Sub-blocks completed since configure/reset.
    pub fn sub_blocks_completed(&self) -> u64 {
        self.sub_blocks
    }

    /// Measure interleaved frames (a trailing partial frame is ignored).
    /// Returns the number of sub-blocks completed during this call.
    /// Allocation-free.
    pub fn process(&mut self, samples: &[f32]) -> usize {
        if !self.configured {
            return 0;
        }
        let level = self.level;
        fearless_simd::dispatch!(level, simd => self.process_with(simd, samples))
    }

    fn process_with<S: Simd>(&mut self, simd: S, samples: &[f32]) -> usize {
        let ch = self.channels;
        let frames = samples.len() / ch;
        let bank = STATE_PER_CHANNEL * ch;
        let stages = [
            SteadyStage {
                coefs: self.coefs[0],
                state: 0,
            },
            SteadyStage {
                coefs: self.coefs[1],
                state: bank,
            },
        ];
        let mut completed = 0;
        let mut pos = 0;
        while pos < frames {
            let n = (frames - pos)
                .min(CHUNK_FRAMES)
                .min(self.sub_len - self.sub_pos);
            let buf = &mut self.scratch[..n * ch];
            for (d, &s) in buf.iter_mut().zip(&samples[pos * ch..(pos + n) * ch]) {
                *d = if s.is_finite() { s } else { 0.0 };
            }
            run_chain(simd, buf, ch, &stages, &[], &mut self.states);
            self.sub_acc += weighted_energy(buf, &self.weights, &mut self.chan_acc);
            self.sub_pos += n;
            pos += n;
            if self.sub_pos == self.sub_len {
                self.finish_sub_block();
                completed += 1;
            }
        }
        sanitize_bank(&mut self.states);
        completed
    }

    fn finish_sub_block(&mut self) {
        let ms = self.sub_acc / self.sub_len as f64;
        self.history
            .push(if ms.is_finite() { ms } else { f64::MAX });
        self.momentary.pushed(&self.history);
        self.short_term.pushed(&self.history);
        self.sub_acc = 0.0;
        self.sub_pos = 0;
        self.sub_blocks += 1;
    }

    /// Weighted mean square (`Σ G·z`) of the momentary window (400 ms).
    pub fn momentary_mean_square(&self) -> f64 {
        self.momentary.sum() / self.momentary.len() as f64
    }

    /// Weighted mean square (`Σ G·z`) of the short-term window (3 s).
    pub fn short_term_mean_square(&self) -> f64 {
        self.short_term.sum() / self.short_term.len() as f64
    }

    /// Momentary loudness in LUFS (last 400 ms of completed sub-blocks).
    pub fn momentary_lufs(&self) -> f64 {
        lufs_from_mean_square(self.momentary_mean_square())
    }

    /// Short-term loudness in LUFS (last 3 s of completed sub-blocks).
    pub fn short_term_lufs(&self) -> f64 {
        lufs_from_mean_square(self.short_term_mean_square())
    }

    /// Weighted mean square of the newest completed sub-block.
    pub fn last_sub_block_mean_square(&self) -> f64 {
        self.history.back(0)
    }

    /// Weighted mean square of the newest `sub_blocks` completed sub-blocks
    /// (clamped to 1 … [`SHORT_TERM_SUB_BLOCKS`]). Exact sum, O(`sub_blocks`).
    pub fn recent_mean_square(&self, sub_blocks: usize) -> f64 {
        let n = sub_blocks.clamp(1, SHORT_TERM_SUB_BLOCKS);
        self.history.sum_newest(n).max(0.0) / n as f64
    }
}

/// `Σ_c G_c · Σ_n y_c[n]²` over interleaved `buf`, accumulated in `f64`.
/// `acc` (one slot per channel) is used as scratch and left at zero.
#[inline]
fn weighted_energy(buf: &[f32], weights: &[f64], acc: &mut [f64]) -> f64 {
    let ch = weights.len();
    match ch {
        1 => {
            buf.iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
                * weights[0]
        }
        2 => {
            let (mut l, mut r) = (0.0f64, 0.0f64);
            for f in buf.as_chunks::<2>().0 {
                l += f64::from(f[0]) * f64::from(f[0]);
                r += f64::from(f[1]) * f64::from(f[1]);
            }
            l * weights[0] + r * weights[1]
        }
        _ => {
            for f in buf.chunks_exact(ch) {
                for (a, &v) in acc.iter_mut().zip(f) {
                    *a += f64::from(v) * f64::from(v);
                }
            }
            let mut e = 0.0;
            for (a, w) in acc.iter_mut().zip(weights) {
                e += *a * w;
                *a = 0.0;
            }
            e
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_sum_tracks_exact_sum() {
        let mut h = History::default();
        h.allocate(11);
        let mut w = WindowSum::new(10);
        let mut x = 1u64;
        for _ in 0..1000 {
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            h.push((x >> 40) as f64 * 1e-3);
            w.pushed(&h);
            let exact = h.sum_newest(10);
            assert!((w.sum() - exact).abs() <= 1e-9 * exact.max(1.0));
        }
    }

    #[test]
    fn history_back_and_capacity() {
        let mut h = History::default();
        h.allocate(3);
        for v in [1.0, 2.0, 3.0, 4.0] {
            h.push(v);
        }
        assert_eq!(
            (h.back(0), h.back(1), h.back(2), h.back(3)),
            (4.0, 3.0, 2.0, 0.0)
        );
    }
}
