//! Butterworth design as cascaded second-order sections (bilinear transform
//! with pre-warping). See the [`crate::filter`] module documentation for the
//! derivation.

use core::f64::consts::PI;

use crate::biquad::BiquadCoefficients;

/// Highest supported Butterworth order.
pub const MAX_ORDER: usize = 8;
/// Most second-order sections any design in this module needs (order 8).
pub const MAX_SECTIONS: usize = MAX_ORDER / 2;
/// Lowest cutoff / crossover frequency (Hz).
pub const MIN_FREQUENCY_HZ: f32 = 1.0;
/// Cutoffs are clamped to this fraction of the sample rate (just below
/// Nyquist, where the bilinear pre-warp `tan(π·f/fs)` diverges).
pub const NYQUIST_FRACTION: f64 = 0.499;

/// Low-pass or high-pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterKind {
    /// Passes frequencies below the cutoff.
    LowPass,
    /// Passes frequencies above the cutoff.
    #[default]
    HighPass,
}

/// A cascade of at most [`MAX_SECTIONS`] biquads, stored inline (never
/// allocates). First-order sections are biquads with `b2 = a2 = 0`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Sections {
    coefs: [BiquadCoefficients; MAX_SECTIONS],
    len: usize,
}

impl Sections {
    /// The sections in processing order.
    pub fn as_slice(&self) -> &[BiquadCoefficients] {
        &self.coefs[..self.len]
    }

    /// Number of sections.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if there are no sections (the identity).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn push(&mut self, c: BiquadCoefficients) {
        self.coefs[self.len] = c;
        self.len += 1;
    }

    /// Analytic magnitude of the whole cascade at `freq_hz` (Hz) for
    /// `sample_rate_hz` (Hz): the product of the section magnitudes.
    pub fn magnitude_at(&self, freq_hz: f64, sample_rate_hz: f64) -> f64 {
        self.as_slice()
            .iter()
            .map(|c| c.magnitude_at(freq_hz, sample_rate_hz))
            .product()
    }

    /// True if every section is stable.
    pub fn is_stable(&self) -> bool {
        self.as_slice().iter().all(|c| c.is_stable())
    }
}

/// Clamp a cutoff (Hz) to `[MIN_FREQUENCY_HZ, NYQUIST_FRACTION·fs]`;
/// non-finite values fall back to `fallback`.
pub fn clamp_cutoff(freq_hz: f64, sample_rate: f64, fallback: f64) -> f64 {
    let f = if freq_hz.is_finite() {
        freq_hz
    } else {
        fallback
    };
    f.min(sample_rate * NYQUIST_FRACTION)
        .max(MIN_FREQUENCY_HZ as f64)
}

/// Q of conjugate pole pair `k` (`0 ≤ k < order/2`) of the order-`order`
/// analog Butterworth prototype.
///
/// The prototype's poles lie on the unit circle in the left half plane at
/// `p = −sin ψ_k ± j·cos ψ_k`, `ψ_k = (2k+1)·π/(2·order)`; a pair gives the
/// factor `s² + 2·sin ψ_k·s + 1 = s² + s/Q_k + 1`, so `Q_k = 1/(2·sin ψ_k)`.
/// For odd orders the remaining pole (`ψ = π/2`) is real, at `s = −1`.
pub fn butterworth_pair_q(order: usize, k: usize) -> f64 {
    let psi = (2 * k + 1) as f64 * PI / (2 * order) as f64;
    1.0 / (2.0 * psi.sin())
}

/// Pre-warped bilinear constant `K = tan(π·f/fs)` (the analog prototype is
/// evaluated at `s = (1/K)·(1 − z⁻¹)/(1 + z⁻¹)`).
fn warp(freq_hz: f64, sample_rate: f64) -> f64 {
    (PI * freq_hz / sample_rate).tan()
}

/// First-order section from the bilinear transform of
/// `(n0 + n1·s)/(1 + s)`, `s = (1/K)(1 − z⁻¹)/(1 + z⁻¹)`: multiplying through
/// by `K(1 + z⁻¹)` gives numerator `(K·n0 + n1) + (K·n0 − n1)·z⁻¹` and
/// denominator `(K + 1) + (K − 1)·z⁻¹`.
fn first_order(k: f64, n0: f64, n1: f64) -> BiquadCoefficients {
    BiquadCoefficients::from_unnormalized(k * n0 + n1, k * n0 - n1, 0.0, k + 1.0, k - 1.0, 0.0)
}

/// Butterworth low-pass or high-pass of `order` (clamped to 1 … 8) with its
/// −3.01 dB point at `cutoff_hz`, as `⌈order/2⌉` sections. The cutoff is
/// expected inside `(0, fs/2)` (see [`clamp_cutoff`]).
///
/// Second-order sections are the bilinear transforms (pre-warped at the
/// cutoff) of `1/(s² + s/Q_k + 1)` and `s²/(s² + s/Q_k + 1)`; these are exactly
/// the RBJ cookbook LPF/HPF at `Q_k`, so [`BiquadCoefficients::low_pass`] and
/// [`BiquadCoefficients::high_pass`] are reused. The first-order section of an
/// odd order is the transform of `1/(1 + s)` or `s/(1 + s)`.
pub fn butterworth(kind: FilterKind, order: usize, cutoff_hz: f64, sample_rate: f64) -> Sections {
    let order = order.clamp(1, MAX_ORDER);
    let mut out = Sections::default();
    for k in 0..order / 2 {
        let q = butterworth_pair_q(order, k);
        out.push(match kind {
            FilterKind::LowPass => BiquadCoefficients::low_pass(cutoff_hz, q, sample_rate),
            FilterKind::HighPass => BiquadCoefficients::high_pass(cutoff_hz, q, sample_rate),
        });
    }
    if order % 2 == 1 {
        let k = warp(cutoff_hz, sample_rate);
        out.push(match kind {
            FilterKind::LowPass => first_order(k, 1.0, 0.0),
            FilterKind::HighPass => first_order(k, 0.0, 1.0),
        });
    }
    out
}

/// The all-pass `B(−s)/B(s)` of an order-`order` Butterworth denominator
/// `B(s)` (bilinear, pre-warped at `freq_hz`). Pair sections are the
/// transforms of `(s² − s/Q_k + 1)/(s² + s/Q_k + 1)`, i.e. the cookbook APF at
/// `Q_k` ([`BiquadCoefficients::all_pass`]); the real pole gives `(1 − s)/(1 + s)`.
pub fn butterworth_all_pass(order: usize, freq_hz: f64, sample_rate: f64) -> Sections {
    let order = order.clamp(1, MAX_ORDER);
    let mut out = Sections::default();
    for k in 0..order / 2 {
        let q = butterworth_pair_q(order, k);
        out.push(BiquadCoefficients::all_pass(freq_hz, q, sample_rate));
    }
    if order % 2 == 1 {
        out.push(first_order(warp(freq_hz, sample_rate), 1.0, -1.0));
    }
    out
}

/// Product of two first-order sections (`b2 = a2 = 0`) as one biquad.
fn multiply_first_order(p: &BiquadCoefficients, q: &BiquadCoefficients) -> BiquadCoefficients {
    BiquadCoefficients {
        b0: p.b0 * q.b0,
        b1: p.b0 * q.b1 + p.b1 * q.b0,
        b2: p.b1 * q.b1,
        a1: p.a1 + q.a1,
        a2: p.a1 * q.a1,
    }
}

/// Linkwitz-Riley filter of order `2·bw_order` (`bw_order` ∈ {1, 2, 4}): the
/// Butterworth filter of order `bw_order` applied twice. For `bw_order = 1`
/// (LR2) the two first-order sections are merged into one biquad, and the
/// high-pass is **polarity-inverted** (`−HP²`), so that `LP² + (−HP²)` is
/// all-pass (see the module documentation).
pub(crate) fn linkwitz_riley(
    kind: FilterKind,
    bw_order: usize,
    freq_hz: f64,
    sample_rate: f64,
) -> Sections {
    let bw = butterworth(kind, bw_order, freq_hz, sample_rate);
    let mut out = Sections::default();
    if bw_order == 1 {
        let s = bw.as_slice()[0];
        let mut c = multiply_first_order(&s, &s);
        if kind == FilterKind::HighPass {
            c.b0 = -c.b0;
            c.b1 = -c.b1;
            c.b2 = -c.b2;
        }
        out.push(c);
    } else {
        for _ in 0..2 {
            for &c in bw.as_slice() {
                out.push(c);
            }
        }
    }
    out
}
