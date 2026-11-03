//! All-pass phase-difference pair ("Hilbert pair") for an analytic-signal
//! envelope in a control path.
//!
//! References: S. Darlington, "Realization of a constant phase difference",
//! Bell System Technical Journal 29, 1950, and the general literature on
//! all-pass phase-difference networks (as used for Weaver single-sideband
//! modulation): two chains of first-order all-pass sections whose phase
//! responses differ by 90° over a wide band. The bilinear transform with
//! pre-warping (Oppenheim & Schafer) maps them to the digital domain.
//!
//! # Design (own numerical design)
//!
//! An analog first-order all-pass `(p − s)/(p + s)` has the phase
//! `−2·atan(ω/p)`: in `u = ln ω` a sigmoid step of `−π` centred on `ln p`.
//! Give the two chains the poles of one increasing sequence alternately
//! (even index to chain A, odd to chain B): the phase difference
//! `φ_A − φ_B` is then an alternating sum of these steps, which oscillates
//! around `−π/2` inside the span of the poles. With equal spacing the
//! interior ripple is tiny (the step derivative is `sech`, whose Fourier
//! transform `π·sech(πν/2)` gives a first ripple harmonic of about
//! `4·e^{−π²/(2d)}` rad for spacing `d`), but the error near both ends of a
//! finite sequence decays only like `e^{−distance}`, so equal spacing needs
//! many extra poles. Instead the pole positions [`POLES`] (16 poles, 8 per
//! chain, in `ln` of the pre-warped frequency relative to the band centre)
//! were found by our own minimax optimisation (Gauss–Newton with
//! Lawson-style reweighting, i.e. iteratively reweighted least squares, on
//! the phase error over the band `[−ln(600)/2, ln(600)/2]`): the result is
//! near-equiripple with a maximum error of 0.0091° over a 600:1 band
//! (bound in [`MAX_PHASE_ERROR_DEG`]). Darlington's elliptic-function solution is the exact
//! optimum of this problem; the numerical one is close enough for an
//! envelope (the test checks the bound).
//!
//! The digital sections come from the bilinear transform with `K = 2·fs`
//! and the poles at pre-warped frequencies, `p_i = K·tan(π·f_i/fs)`, so the
//! digital phase at `f` equals the analog phase at the warped frequency and
//! the design holds exactly in warped frequency. Each section is
//! `H(z) = (c + z⁻¹)/(1 + c·z⁻¹)`, `c = (t − 1)/(t + 1)`,
//! `t = tan(π·f_i/fs)`, computed as `y = c·(x − y₁) + x₁` (state in `f64`).
//!
//! # Envelope
//!
//! For a component at `f` in the design band the two outputs have equal
//! magnitude and a 90° phase difference, so `√(a² + b²)` is its amplitude:
//! flat for a steady sine (no ripple at `2f`, unlike a rectified waveform).
//! The phase of the pair relative to the input does not matter, since only
//! the magnitude is used, in a control path.

use crate::util::DENORMAL_FLUSH;

/// Pole positions (ln of pre-warped frequency relative to the band
/// centre), ascending; even index → chain A, odd → chain B. Own minimax
/// design for a 600:1 band (see the module docs).
pub const POLES: [f64; 16] = [
    -4.603084147294613,
    -3.427079668317532,
    -2.768011373308395,
    -2.2231919654060976,
    -1.7153304387314199,
    -1.2207403232395782,
    -0.7311276981036816,
    -0.24335917139262964,
    0.2439314156653103,
    0.7316756702000554,
    1.2212425050347722,
    1.715771041116759,
    2.223563363005921,
    2.768315762575118,
    3.4273300125180204,
    4.603303968858651,
];

/// Band ratio (pre-warped) the [`POLES`] were designed for.
pub const BAND_RATIO: f64 = 600.0;

/// Bound on the phase error (degrees) of the design over its band (the
/// optimisation reached 0.0091°).
pub const MAX_PHASE_ERROR_DEG: f64 = 0.02;

/// A pair of first-order all-pass chains with a 90° phase difference over
/// a design band, for interleaved multichannel input. Allocates only in
/// [`configure`](Self::configure).
#[derive(Debug, Clone, Default)]
pub struct HilbertPair {
    /// Coefficients of chain A (`a.len()` sections) and chain B.
    a: Vec<f64>,
    b: Vec<f64>,
    /// Per channel and section: (x₁, y₁) of chain A, then of chain B.
    state_a: Vec<(f64, f64)>,
    state_b: Vec<(f64, f64)>,
    channels: usize,
}

impl HilbertPair {
    /// An unconfigured pair (outputs zero).
    pub fn new() -> Self {
        Self::default()
    }

    /// Design for 90° (within [`MAX_PHASE_ERROR_DEG`]) from `f_lo` Hz
    /// upward over a [`BAND_RATIO`]:1 band in pre-warped frequency, at
    /// `sample_rate`, for `channels` channels, and clear the state. The top
    /// of the band is limited to just below Nyquist.
    pub fn configure(&mut self, sample_rate: f64, channels: usize, f_lo: f64) {
        let fs = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate
        } else {
            48_000.0
        };
        let warp = |f: f64| (core::f64::consts::PI * f / fs).tan();
        let top = warp(0.45 * fs);
        let lo = warp(f_lo.clamp(0.1, 0.45 * fs)).min(top / BAND_RATIO);
        let centre = lo * BAND_RATIO.sqrt();
        self.a.clear();
        self.b.clear();
        for (i, &v) in POLES.iter().enumerate() {
            let t = centre * v.exp();
            let c = (t - 1.0) / (t + 1.0);
            if i % 2 == 0 {
                self.a.push(c);
            } else {
                self.b.push(c);
            }
        }
        self.channels = channels.max(1);
        self.state_a = vec![(0.0, 0.0); self.a.len() * self.channels];
        self.state_b = vec![(0.0, 0.0); self.b.len() * self.channels];
    }

    /// Top of the accurate band (Hz) for a design from `f_lo` at `sample_rate`.
    pub fn band_top_hz(sample_rate: f64, f_lo: f64) -> f64 {
        let w = (core::f64::consts::PI * f_lo / sample_rate).tan() * BAND_RATIO;
        w.atan() * sample_rate / core::f64::consts::PI
    }

    /// Clear the filter state.
    pub fn reset(&mut self) {
        self.state_a.fill((0.0, 0.0));
        self.state_b.fill((0.0, 0.0));
    }

    /// Sections per chain (A, B).
    pub fn sections(&self) -> (usize, usize) {
        (self.a.len(), self.b.len())
    }

    /// Phase difference `φ_A − φ_B` (rad) at `f` Hz for the designed
    /// coefficients, from their transfer functions (for tests).
    pub fn phase_difference(&self, f: f64, sample_rate: f64) -> f64 {
        let w = 2.0 * std::f64::consts::PI * f / sample_rate;
        // Phase of (c + e^{-jw})/(1 + c·e^{-jw}).
        let ph = |c: f64| {
            let num = (c + w.cos(), -w.sin());
            let den = (1.0 + c * w.cos(), -c * w.sin());
            num.1.atan2(num.0) - den.1.atan2(den.0)
        };
        let sa: f64 = self.a.iter().map(|&c| ph(c)).sum();
        let sb: f64 = self.b.iter().map(|&c| ph(c)).sum();
        let mut d = sa - sb;
        // Wrap into (−π, π].
        while d > std::f64::consts::PI {
            d -= 2.0 * std::f64::consts::PI;
        }
        while d <= -std::f64::consts::PI {
            d += 2.0 * std::f64::consts::PI;
        }
        d
    }

    /// Process one interleaved frame and return the largest analytic
    /// magnitude `√(a² + b²)` over its channels (linked envelope).
    #[inline]
    pub fn linked_magnitude(&mut self, frame: &[f32]) -> f64 {
        let na = self.a.len();
        let nb = self.b.len();
        let mut m2 = 0.0f64;
        for (c, &x) in frame.iter().enumerate().take(self.channels) {
            let x = x as f64;
            let ya = Self::chain(&self.a, &mut self.state_a[c * na..(c + 1) * na], x);
            let yb = Self::chain(&self.b, &mut self.state_b[c * nb..(c + 1) * nb], x);
            let p = ya * ya + yb * yb;
            if p > m2 {
                m2 = p;
            }
        }
        m2.sqrt()
    }

    #[inline(always)]
    fn chain(coefs: &[f64], state: &mut [(f64, f64)], x: f64) -> f64 {
        let mut v = x;
        for (&c, s) in coefs.iter().zip(state.iter_mut()) {
            let y = c * (v - s.1) + s.0;
            s.0 = v;
            s.1 = if y.abs() < DENORMAL_FLUSH { 0.0 } else { y };
            v = y;
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The designed pair is within [`MAX_PHASE_ERROR_DEG`] of 90° over
    /// its band at common rates.
    #[test]
    fn phase_difference_is_ninety_degrees() {
        for fs in [44_100.0, 48_000.0, 96_000.0] {
            let mut h = HilbertPair::new();
            h.configure(fs, 1, 8.0);
            let top = HilbertPair::band_top_hz(fs, 8.0);
            let mut worst = 0.0f64;
            let mut f = 8.0;
            while f <= top {
                let d = h.phase_difference(f, fs).abs().to_degrees();
                worst = worst.max((d - 90.0).abs());
                f *= 1.005;
            }
            assert!(worst < MAX_PHASE_ERROR_DEG, "fs {fs}: worst {worst}°");
            assert!(top > 4000.0, "fs {fs}: top {top} Hz");
        }
    }

    /// A steady sine gives a flat envelope equal to its amplitude.
    #[test]
    fn sine_envelope_is_flat() {
        let fs = 48_000.0;
        let mut h = HilbertPair::new();
        h.configure(fs, 2, 8.0);
        for f in [30.0, 60.0, 200.0, 1000.0] {
            h.reset();
            let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
            for n in 0..96_000 {
                let x = (0.5 * (2.0 * std::f64::consts::PI * f * n as f64 / fs).sin()) as f32;
                let m = h.linked_magnitude(&[x, 0.5 * x]);
                if n > 48_000 {
                    lo = lo.min(m);
                    hi = hi.max(m);
                }
            }
            assert!(
                (hi - 0.5).abs() < 5e-4 && (lo - 0.5).abs() < 5e-4,
                "{f} Hz: {lo} … {hi}"
            );
        }
    }
}
