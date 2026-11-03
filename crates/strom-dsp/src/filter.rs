//! Higher-order filters: Butterworth low-/high-pass of order 1–8 as cascaded
//! second-order sections, designed for any sample rate.
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

mod butterworth;
mod design;

pub use butterworth::{Butterworth, ButterworthParams};
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
