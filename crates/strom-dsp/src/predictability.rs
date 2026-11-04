//! Linear-prediction gain: how predictable the programme is, as one control
//! value per block.
//!
//! The mono sum of the channels is low-passed at [`BAND_HZ`] (4th-order
//! Butterworth) and decimated to about [`ANALYSIS_RATE_HZ`], the classic
//! narrowband rate of speech LPC, so the measure describes the programme
//! below 4 kHz and does not depend on where its upper band edge lies (a
//! band-limited signal is predictable whatever it carries). Pre-emphasised
//! with `1 − μ·z⁻¹`, it feeds an
//! exponentially windowed autocorrelation of lags `0 … P` (each lag a
//! one-pole sum of `y[n]·y[n−k]`, time constant [`WINDOW_MS`]; Barnwell,
//! "Recursive windowing for generating autocorrelation coefficients for LPC
//! analysis", IEEE Trans. ASSP 29(5), 1981). Once per block the
//! Levinson–Durbin recursion (Makhoul, "Linear prediction: a tutorial
//! review", Proc. IEEE 63(4), 1975) gives the order-`P` prediction error
//! power `E_P`, and the prediction gain is `G = 10·log10(r₀ / E_P)` dB.
//!
//! Speech and tones are predictable (`G` well over 10 dB in voiced speech);
//! noise, crowd and applause are not (`G` near 0 dB after the
//! pre-emphasis). Using it to allow more clipping on less predictable
//! material follows the expired US 4,241,266 (expired; status not legally
//! reviewed).
//!
//! # Real-time safety
//!
//! No allocation; fixed-size state. Per input frame two biquads; per
//! analysis sample the recursion (`P + 1` multiply-adds); Levinson–Durbin
//! once per [`Predictability::gain_db`] call. Quiet input (`r₀` under
//! [`QUIET`]) reads 0 dB.

use crate::biquad::BiquadCoefficients;

/// Upper edge of the analysed band (Hz).
pub const BAND_HZ: f64 = 3_500.0;
/// Rate of the analysis (Hz); the decimation factor is
/// `max(1, round(fs / ANALYSIS_RATE_HZ))`.
pub const ANALYSIS_RATE_HZ: f64 = 8_000.0;
/// Prediction order `P`.
pub const ORDER: usize = 10;
/// Time constant of the autocorrelation window (ms).
pub const WINDOW_MS: f64 = 20.0;
/// Pre-emphasis coefficient `μ`.
pub const PRE_EMPHASIS: f64 = 0.9;
/// Autocorrelation `r₀` (mean square of the pre-emphasised mono sum) under
/// which the input counts as quiet.
const QUIET: f64 = 1e-10;
/// Largest reported gain (dB).
pub const MAX_GAIN_DB: f64 = 40.0;

/// Running linear-prediction gain of the mono sum of a signal.
#[derive(Debug, Clone)]
pub struct Predictability {
    lp: [BiquadCoefficients; 2],
    /// Direct-form II transposed state of the two low-pass sections.
    z: [[f64; 2]; 2],
    decim: usize,
    phase: usize,
    lambda: f64,
    r: [f64; ORDER + 1],
    hist: [f64; ORDER],
    pre_prev: f64,
}

impl Default for Predictability {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl Predictability {
    /// A fresh estimator for `sample_rate`.
    pub fn new(sample_rate: f64) -> Self {
        let fs = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate
        } else {
            48_000.0
        };
        let decim = ((fs / ANALYSIS_RATE_HZ).round() as usize).max(1);
        let band = BAND_HZ.min(0.45 * fs);
        Self {
            lp: [
                BiquadCoefficients::low_pass(band, 0.541_196, fs),
                BiquadCoefficients::low_pass(band, 1.306_563, fs),
            ],
            z: [[0.0; 2]; 2],
            decim,
            phase: 0,
            lambda: (-1.0 / (WINDOW_MS * 0.001 * fs / decim as f64)).exp(),
            r: [0.0; ORDER + 1],
            hist: [0.0; ORDER],
            pre_prev: 0.0,
        }
    }

    /// Clear the state.
    pub fn reset(&mut self) {
        self.z = [[0.0; 2]; 2];
        self.phase = 0;
        self.r = [0.0; ORDER + 1];
        self.hist = [0.0; ORDER];
        self.pre_prev = 0.0;
    }

    /// Feed `x` (interleaved, `ch` channels).
    pub fn push(&mut self, x: &[f32], ch: usize) {
        if ch == 0 {
            return;
        }
        let l = self.lambda;
        let w = 1.0 - l;
        let inv = 1.0 / ch as f64;
        for frame in x.chunks_exact(ch) {
            let m = frame.iter().map(|&v| v as f64).sum::<f64>() * inv;
            let mut m = if m.is_finite() { m } else { 0.0 };
            for (c, z) in self.lp.iter().zip(self.z.iter_mut()) {
                let y = c.b0 * m + z[0];
                z[0] = c.b1 * m - c.a1 * y + z[1];
                z[1] = c.b2 * m - c.a2 * y;
                m = y;
            }
            self.phase += 1;
            if self.phase < self.decim {
                continue;
            }
            self.phase = 0;
            let y = m - PRE_EMPHASIS * self.pre_prev;
            self.pre_prev = m;
            self.r[0] = l * self.r[0] + w * y * y;
            for k in 1..=ORDER {
                self.r[k] = l * self.r[k] + w * y * self.hist[k - 1];
            }
            self.hist.copy_within(0..ORDER - 1, 1);
            self.hist[0] = y;
        }
    }

    /// Keep the sums and the filter state out of the subnormal range in
    /// silence. Call it at fixed points of the stream (e.g. once per control
    /// block), so the result does not depend on how the input was split.
    pub fn flush_denormals(&mut self) {
        if self.r[0] < 1e-30 {
            self.r = [0.0; ORDER + 1];
            self.hist = [0.0; ORDER];
            self.pre_prev = 0.0;
        }
        for z in self.z.iter_mut().flatten() {
            if z.abs() < 1e-30 {
                *z = 0.0;
            }
        }
    }

    /// The prediction gain (dB, 0 … [`MAX_GAIN_DB`]) of the current window.
    pub fn gain_db(&self) -> f64 {
        let r = &self.r;
        if r[0] < QUIET {
            return 0.0;
        }
        // Levinson–Durbin; a slight lag window on r₀ (white-noise
        // correction, 40 dB) keeps the recursion well conditioned. The
        // recursive window does not guarantee a positive definite estimate:
        // a reflection coefficient at or over 1 means the window is (close
        // to) a sum of pure tones, which reads as fully predictable.
        let mut e = r[0] * (1.0 + 1e-4);
        let mut a = [0.0f64; ORDER + 1];
        let mut tmp = [0.0f64; ORDER + 1];
        a[0] = 1.0;
        for i in 1..=ORDER {
            let mut acc = r[i];
            for j in 1..i {
                acc += a[j] * r[i - j];
            }
            let k = -acc / e;
            if !k.is_finite() || k.abs() >= 1.0 {
                return MAX_GAIN_DB;
            }
            tmp[..=i].copy_from_slice(&a[..=i]);
            for j in 1..i {
                a[j] = tmp[j] + k * tmp[i - j];
            }
            a[i] = k;
            e *= 1.0 - k * k;
        }
        let g = 10.0 * (r[0] / e.max(1e-300)).log10();
        if g.is_finite() {
            g.clamp(0.0, MAX_GAIN_DB)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gain_of(x: &[f32]) -> f64 {
        let mut p = Predictability::new(48_000.0);
        p.push(x, 1);
        p.gain_db()
    }

    #[test]
    fn tone_is_predictable_noise_is_not() {
        let n = 9_600;
        let tone: Vec<f32> = (0..n)
            .map(|i| {
                (0.3 * (2.0 * core::f64::consts::PI * 440.0 * i as f64 / 48_000.0).sin()) as f32
            })
            .collect();
        // Deterministic white noise (xorshift).
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let noise: Vec<f32> = (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32 * 0.5
            })
            .collect();
        let gt = gain_of(&tone);
        let gn = gain_of(&noise);
        assert!(gt > 30.0, "tone {gt:.1} dB");
        // White noise pre-emphasised is coloured; most of that is predicted.
        assert!(gn < 8.0, "noise {gn:.1} dB");
        assert_eq!(gain_of(&vec![0.0; n]), 0.0);
    }
}
