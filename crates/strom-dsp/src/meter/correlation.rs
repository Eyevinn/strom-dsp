//! Stereo phase correlation and side/mid energy ratio.

use crate::util::{flush_denormal, one_pole_coef};

/// Default integration time of [`CorrelationMeter`] in ms.
pub const DEFAULT_CORRELATION_MS: f32 = 300.0;

/// Mean square below which a channel counts as silent (−100 dBFS RMS).
const SILENT_MEAN_SQUARE: f64 = 1e-10;

/// Phase-correlation meter for one L/R pair.
///
/// Three exponentially weighted averages are kept per sample:
/// `⟨L²⟩`, `⟨R²⟩` and `⟨L·R⟩`, each `m[n] = a·m[n−1] + (1−a)·x[n]` with
/// `a = exp(−1/(τ·fs))` and τ the integration time (63.2 % of a step after τ).
/// The correlation is the normalised cross-correlation at lag 0 (Pearson's
/// coefficient without mean removal, as audio has no DC):
///
/// `r = ⟨L·R⟩ / √(⟨L²⟩·⟨R²⟩)`, in −1 … +1.
///
/// From the same averages, the side/mid energy ratio follows without extra
/// state, with `M = (L+R)/2`, `S = (L−R)/2`:
/// `⟨S²⟩/⟨M²⟩ = (⟨L²⟩ + ⟨R²⟩ − 2⟨LR⟩) / (⟨L²⟩ + ⟨R²⟩ + 2⟨LR⟩)`.
///
/// **Silence:** if either channel's mean square is below −100 dBFS (one
/// channel silent, or both), the correlation is defined as **0.0** ("no
/// common signal"). A signal in only one channel therefore reads 0, as on a
/// broadcast correlation meter.
///
/// Band-agnostic: feed it any L/R pair (full band, or one crossover band).
/// Real-time safe: no allocation; O(1) state; `set_integration_ms` can be
/// called at any time.
#[derive(Debug, Clone)]
pub struct CorrelationMeter {
    sample_rate: f32,
    integration_ms: f32,
    a: f64,
    b: f64,
    ll: f64,
    rr: f64,
    lr: f64,
}

impl CorrelationMeter {
    /// Create a meter at `sample_rate` Hz with `integration_ms` (clamped to
    /// 1 ms … 60 s).
    pub fn new(sample_rate: f32, integration_ms: f32) -> Self {
        let mut m = Self {
            sample_rate: sample_rate.max(1.0),
            integration_ms: DEFAULT_CORRELATION_MS,
            a: 0.0,
            b: 1.0,
            ll: 0.0,
            rr: 0.0,
            lr: 0.0,
        };
        m.set_integration_ms(integration_ms);
        m
    }

    /// Change the integration time (ms, clamped to 1 … 60 000). Keeps state.
    pub fn set_integration_ms(&mut self, integration_ms: f32) {
        let t = if integration_ms.is_finite() {
            integration_ms.clamp(1.0, 60_000.0)
        } else {
            DEFAULT_CORRELATION_MS
        };
        self.integration_ms = t;
        self.a = one_pole_coef(t as f64, self.sample_rate as f64);
        self.b = 1.0 - self.a;
    }

    /// Integration time in ms.
    pub fn integration_ms(&self) -> f32 {
        self.integration_ms
    }

    /// Clear the averages (reads 0 correlation until signal arrives).
    pub fn reset(&mut self) {
        self.ll = 0.0;
        self.rr = 0.0;
        self.lr = 0.0;
    }

    /// Add one L/R sample pair.
    #[inline]
    pub fn push(&mut self, l: f32, r: f32) {
        let (l, r) = (l as f64, r as f64);
        self.ll = self.a * self.ll + self.b * (l * l);
        self.rr = self.a * self.rr + self.b * (r * r);
        self.lr = self.a * self.lr + self.b * (l * r);
    }

    /// Add two planar channel slices (the shorter length is used).
    pub fn process_pair(&mut self, left: &[f32], right: &[f32]) {
        for (&l, &r) in left.iter().zip(right) {
            self.push(l, r);
        }
        self.flush();
    }

    /// Add interleaved frames of `channels` (≥ 2) channels, measuring channel
    /// 0 against channel 1. A trailing partial frame is ignored.
    pub fn process_interleaved(&mut self, samples: &[f32], channels: usize) {
        if channels < 2 {
            return;
        }
        for f in samples.chunks_exact(channels) {
            self.push(f[0], f[1]);
        }
        self.flush();
    }

    fn flush(&mut self) {
        self.ll = flush_denormal(self.ll);
        self.rr = flush_denormal(self.rr);
        self.lr = flush_denormal(self.lr);
        if !(self.ll.is_finite() && self.rr.is_finite() && self.lr.is_finite()) {
            self.reset();
        }
    }

    /// Phase correlation, −1 … +1; 0 when either channel is silent.
    pub fn correlation(&self) -> f64 {
        if self.ll < SILENT_MEAN_SQUARE || self.rr < SILENT_MEAN_SQUARE {
            return 0.0;
        }
        (self.lr / (self.ll * self.rr).sqrt()).clamp(-1.0, 1.0)
    }

    /// Mean square of the mid signal plus side signal, `(⟨L²⟩ + ⟨R²⟩)/2`
    /// (linear power; 1.0 = a full-scale square wave in both channels).
    pub fn mean_square(&self) -> f64 {
        0.5 * (self.ll + self.rr)
    }

    /// Side/mid energy ratio `10·log10(⟨S²⟩/⟨M²⟩)` in dB, clamped to
    /// −100 … +100 dB. Very negative = mono, very positive = polarity-inverted
    /// mono, around 0 dB = wide/uncorrelated. 0 dB for silence.
    pub fn side_mid_ratio_db(&self) -> f64 {
        let sum = self.ll + self.rr;
        if sum < 2.0 * SILENT_MEAN_SQUARE {
            return 0.0;
        }
        let side = (sum - 2.0 * self.lr).max(0.0);
        let mid = (sum + 2.0 * self.lr).max(0.0);
        let floor = sum * 1e-10;
        (10.0 * (side.max(floor) / mid.max(floor)).log10()).clamp(-100.0, 100.0)
    }
}
