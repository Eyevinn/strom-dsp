//! Higher-order filters and crossovers: Butterworth low-/high-pass of order
//! 1–8 as cascaded second-order sections for any sample rate, and
//! Linkwitz-Riley crossovers (LR2/LR4/LR8, 1–5 bands, all-pass compensated).
//!
//! # Butterworth design
//!
//! References: A. V. Oppenheim & R. W. Schafer, *Discrete-Time Signal
//! Processing* (Butterworth prototype, bilinear transform with pre-warping);
//! J. O. Smith III, *Introduction to Digital Filters* (bilinear transform,
//! biquad sections).
//!
//! - **Analog prototype** (cutoff 1 rad/s): `|H(jΩ)|² = 1/(1 + Ω^(2n))`. Its
//!   `n` poles lie on the unit circle in the left half plane at angles
//!   `ψ_k = (2k+1)·π/(2n)` from the imaginary axis, `p = −sin ψ_k ± j cos ψ_k`.
//!   Each conjugate pair is a factor `s² + s/Q_k + 1` with
//!   `Q_k = 1/(2·sin ψ_k)`; an odd order leaves one real pole at `s = −1`.
//!   Low-pass sections are `1/(s² + s/Q_k + 1)` (and `1/(s + 1)`), high-pass
//!   sections `s²/(…)` (and `s/(s + 1)`).
//! - **Bilinear transform with pre-warping**: `s = (1/K)·(1 − z⁻¹)/(1 + z⁻¹)`
//!   with `K = tan(π·fc/fs)`, so the analog cutoff lands exactly on `fc` and
//!   the digital magnitude is `1/√(1 + (tan(π f/fs)/tan(π fc/fs))^(2n))`
//!   (low-pass; the high-pass has the ratio inverted). Substituting into a
//!   pair section and dividing by `cos²(w0/2)` gives exactly the RBJ cookbook
//!   LPF/HPF at `Q_k`, so those designs ([`crate::biquad::BiquadCoefficients`])
//!   are reused; the first-order section is derived directly
//!   (`b = (K, K)/(K+1)`, `a1 = (K−1)/(K+1)` for the low-pass,
//!   `b = (1, −1)/(K+1)` for the high-pass).
//! - **Structure**: the sections run in series through the phase 1 Direct
//!   Form I kernels (`f64` state and coefficients, channels as SIMD lanes,
//!   runtime dispatch, denormal flush every 32 frames; see [`crate::biquad`]).
//!   The order of the sections is the order of `k` (lowest Q first).
//! - **Clamping**: order to 1 … 8, cutoff to `[1 Hz, 0.499·fs]`; NaN/Inf
//!   cutoffs fall back to 100 Hz.
//!
//! # Parameter changes (same approach as the [`crate::Equalizer`])
//!
//! - **Cutoff**: a one-pole smoother with a 10 ms time constant in `ln f`.
//!   Every 32 frames the sections are redesigned at the smoothed frequency and
//!   the coefficients are interpolated linearly per sample in between. Each
//!   end point is a stable design and the biquad stability triangle is convex,
//!   so every intermediate filter is stable.
//! - **Kind or order**: a 10 ms linear crossfade from the old cascade (with its
//!   state) to the new one (starting from zero state), in a second state bank;
//!   the old output is computed into a preallocated scratch buffer.
//! - After `configure`/`reset`, the first `process` call takes the parameters
//!   immediately.
//!
//! # Linkwitz-Riley crossover ([`Crossover`])
//!
//! References: S. H. Linkwitz, "Active Crossover Networks for Noncoincident
//! Drivers", JAES 24(1), 1976; the textbook Linkwitz-Riley property that an
//! LR filter of order `2n` is a Butterworth filter of order `n` applied twice.
//!
//! - **One split.** With the Butterworth prototype `L(s) = 1/B(s)`,
//!   `H(s) = sⁿ/B(s)` and the Butterworth identity
//!   `B(s)·B(−s) = 1 + (−1)ⁿ·s²ⁿ`:
//!   `L² + H² = (1 + s²ⁿ)/B(s)² = B(−s)/B(s)` for even `n` (LR4: n = 2,
//!   LR8: n = 4), and `L² − H² = (1 − s²)/B(s)² = B(−s)/B(s)` for `n = 1`
//!   (LR2). `B(−s)/B(s)` is an all-pass (each pole pair gives the cookbook APF
//!   at `Q_k`, the real pole `(1 − s)/(1 + s)`). The bilinear transform is a
//!   substitution, so the identity holds exactly for the digital filters too
//!   (same pre-warp at the crossover frequency).
//! - **LR2 polarity (our convention).** Without inversion, LR2's `L² + H²`
//!   has a zero at the crossover frequency (a deep notch). The LR2 high-pass
//!   is therefore defined as `−H²` (polarity inverted), so for **every** order
//!   the invariant is the same: the plain sum of the bands is an all-pass.
//!   Consequence: with LR2, band `k` has polarity `(−1)ᵏ` relative to the
//!   textbook `L²`/`H²` cascade. LR4 and LR8 bands are not inverted.
//! - **Band edges.** `|L(f_c)| = |H(f_c)| = 1/√2`, so both bands of a split
//!   are at −6.02 dB at its frequency, for LR2, LR4 and LR8 alike. (For LR2
//!   the two bands are in phase there only because of the inversion above;
//!   textbook LR2 bands are in phase opposition at `f_c`.)
//! - **Topology (N bands, splits `f_0 ≤ … ≤ f_{N−2}`).** A cascade of
//!   high-passes: the remainder starts as the input; split `s` takes band `s`
//!   as `LP_s(remainder)` and replaces the remainder by `HP_s(remainder)`; the
//!   last band is the final remainder. Band `s` then passes through the
//!   all-passes `A_j = LP_j + HP_j` of every *higher* split `j > s` (phase
//!   compensation). By induction from the top split,
//!   `Σ bands = A_0·A_1·…·A_{N−2}`: an all-pass, so the summed bands have a
//!   flat magnitude. The top band needs no compensation, band 0 the most
//!   (LR8 with 5 bands: 4 + 3·2 biquads).
//! - **Latency**: none (IIR only). The sum has the all-pass cascade's phase.
//! - **Parameters**: frequencies are clamped to `[1 Hz, 0.499·fs]` and to be
//!   non-decreasing. Frequency changes glide like the Butterworth cutoff (all
//!   sections redesigned every 32 frames and interpolated per sample; during
//!   a glide the sum deviates slightly from all-pass, because interpolated LP
//!   and HP sections are not exactly complementary). Order and band-count
//!   changes crossfade every band over 10 ms from the old structure (second
//!   state bank, preallocated scratch bands) to the new one; bands that the
//!   old structure did not have fade in from silence, and old bands beyond a
//!   smaller new count are folded into the new top band, so the band sum
//!   crossfades continuously. A structural change takes effect at the start
//!   of the next `process` call (so the realised band count always matches
//!   [`Crossover::num_bands`]); if one arrives while a crossfade is still
//!   running, that crossfade ends at once (a small step, only for structural
//!   changes less than 10 ms apart).
//! - **Processing**: the same DF-I SIMD kernels; each band runs through all
//!   its sections per sample (in groups of at most 8). Outputs are `f32`
//!   buffers owned by the caller.

mod butterworth;
mod crossover;
mod design;

pub use butterworth::{Butterworth, ButterworthParams};
pub use crossover::{Crossover, CrossoverParams, LrOrder, MAX_CROSSOVER_BANDS, MAX_SPLITS};
pub use design::{
    butterworth, butterworth_all_pass, butterworth_pair_q, clamp_cutoff, FilterKind, Sections,
    MAX_ORDER, MAX_SECTIONS, MIN_FREQUENCY_HZ, NYQUIST_FRACTION,
};

use crate::util::one_pole_coef;

/// Time constant (ms) of the frequency smoother.
pub const SMOOTHING_MS: f64 = 10.0;
/// Length (ms) of the crossfade used for structural changes.
pub const CROSSFADE_MS: f64 = 10.0;

/// Coefficient update interval (frames) while a frequency glides.
const RAMP_BLOCK: usize = 32;
/// Frames processed per step during a crossfade (scratch buffer size).
const FADE_CHUNK: usize = 64;
/// The smoother snaps to its target below this distance (in `ln f`).
const SNAP_LN_F: f64 = 1e-7;

/// One-pole smoother of `N` log-frequencies, advanced in blocks.
#[derive(Debug, Clone, Copy)]
struct Glide<const N: usize> {
    cur: [f64; N],
    target: [f64; N],
    settled: bool,
    /// Per-sample pole and its `RAMP_BLOCK`-th power.
    coef: f64,
    block_coef: f64,
}

impl<const N: usize> Default for Glide<N> {
    fn default() -> Self {
        Self {
            cur: [0.0; N],
            target: [0.0; N],
            settled: true,
            coef: 0.0,
            block_coef: 0.0,
        }
    }
}

impl<const N: usize> Glide<N> {
    fn new(sample_rate: f64) -> Self {
        let coef = one_pole_coef(SMOOTHING_MS, sample_rate);
        Self {
            coef,
            block_coef: coef.powi(RAMP_BLOCK as i32),
            ..Self::default()
        }
    }

    fn settled(&self) -> bool {
        self.settled
    }

    fn retarget(&mut self, target: [f64; N]) {
        self.target = target;
        if self.cur != target {
            self.settled = false;
        }
    }

    fn snap(&mut self, target: [f64; N]) {
        self.target = target;
        self.cur = target;
        self.settled = true;
    }

    /// Advance by `n` frames and return the new values.
    fn advance(&mut self, n: usize) -> [f64; N] {
        let k = if n == RAMP_BLOCK {
            self.block_coef
        } else {
            self.coef.powi(n as i32)
        };
        let mut close = true;
        for (c, t) in self.cur.iter_mut().zip(self.target) {
            *c = t + (*c - t) * k;
            close &= (*c - t).abs() < SNAP_LN_F;
        }
        if close {
            self.cur = self.target;
            self.settled = true;
        }
        self.cur
    }
}

/// Crossfade `data` (the new output) with `old`, frame by frame:
/// `old + (new − old)·g`, `g = (done + i + 1)/len` capped at 1.
fn blend_into(data: &mut [f32], old: &[f32], channels: usize, done: usize, len: usize) {
    let inv = 1.0 / len as f64;
    for (i, (d, o)) in data
        .chunks_exact_mut(channels)
        .zip(old.chunks_exact(channels))
        .enumerate()
    {
        let g = ((done + i + 1) as f64 * inv).min(1.0);
        for (d, &o) in d.iter_mut().zip(o) {
            let o = o as f64;
            *d = (o + (*d as f64 - o) * g) as f32;
        }
    }
}
