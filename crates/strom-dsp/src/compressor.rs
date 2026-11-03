//! Feed-forward downward compressor with soft knee.
//!
//! Structure after Giannoulis, Massberg & Reiss, "Digital Dynamic Range Compressor
//! Design — A Tutorial and Analysis" (JAES 60(6), 2012): the gain computer works on
//! the instantaneous level in dB and the resulting gain reduction is smoothed in the
//! log domain by a branching peak detector.
//!
//! # Signal flow (per frame of interleaved samples)
//!
//! 1. **Level detector**: `x_dB = 20·log10(max_c |x_c|)` over all channels (linked;
//!    floored at −200 dB). For a steady sine this is a peak detector, so levels and
//!    the threshold refer to **peak amplitude**.
//! 2. **Gain computer** (static curve), with threshold `T = 20·log10(threshold)`,
//!    knee width `W = −40·log10(k)` dB (`k` = `knee` clamped to `[0.001, 1]`) and
//!    slope `S = 1 − 1/ratio`, the gain reduction `r ≥ 0` (dB) is
//!    - `r = 0` for `x < T − W/2`,
//!    - `r = S·(x − T + W/2)² / (2W)` for `T − W/2 ≤ x ≤ T + W/2` (quadratic soft knee),
//!    - `r = S·(x − T)` for `x > T + W/2`.
//!
//!    The output level is `x − r`, i.e. above the knee `T + (x − T)/ratio`.
//!    The knee region in linear terms spans `threshold·k … threshold/k`. `W = 0`
//!    (`k = 1`) is a hard knee.
//! 3. **Smoothing** (branching one-pole in the dB domain):
//!    `y[n] = a·y[n−1] + (1−a)·r[n]` with `a = a_A` when `r[n] > y[n−1]` (gain
//!    reduction increasing) and `a = a_R` otherwise, where
//!    `a_A = exp(−1/(τ_A·fs))`, `τ_A = attack_ms/1000` (and likewise for release).
//!    **Time constant definition**: after a step of the static gain reduction, the
//!    smoothed reduction (in dB) covers `1 − 1/e` (≈ 63.2 %) of the step after `τ`.
//! 4. **Output gain**: `g = 10^(−y/20) · makeup`, applied to all channels.
//!
//! # Parameter smoothing
//!
//! Threshold (dB), slope `S`, knee width `W` and makeup gain (linear) glide toward
//! new values with a one-pole smoother of time constant [`PARAM_SMOOTHING_MS`], so
//! parameter changes do not click even at the shortest attack time. Ratio and knee
//! glide through `S` and `W`. Before the first `process` call after
//! `configure`/`reset`, parameters are applied immediately (no glide).
//!
//! # Denormals
//!
//! The smoothed reduction `y` is passed through
//! [`flush_denormal`](crate::util::flush_denormal); parameter smoothers snap to
//! their target when within `1e-9`.
//!
//! Ratio 1 gives `S = 0`, hence `r = 0` and exactly unity gain apart from makeup.

use crate::util::{db_to_lin, flush_denormal, lin_to_db, one_pole_coef};
use crate::Processor;

/// Time constant (ms) of the one-pole glide applied to threshold, ratio, knee and makeup.
pub const PARAM_SMOOTHING_MS: f64 = 10.0;

/// Smallest effective knee factor; smaller values (including 0) are treated as this.
pub const MIN_KNEE: f64 = 0.001;

/// Compressor parameters (units as in the GStreamer interface).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressorParams {
    /// Threshold as linear amplitude (0 … 1).
    pub threshold: f32,
    /// Compression ratio (1 … 100; 4 means 4:1).
    pub ratio: f32,
    /// Attack time (ms, 0.01 … 10000).
    pub attack_ms: f32,
    /// Release time (ms, 0.01 … 10000).
    pub release_ms: f32,
    /// Makeup gain as linear multiplier (0 … 100).
    pub makeup_gain: f32,
    /// Knee factor `k` (0 … 1): soft-knee region spans `threshold·k … threshold/k`,
    /// width `−40·log10(k)` dB. 1 = hard knee; values below 0.001 are treated as 0.001.
    pub knee: f32,
}

impl Default for CompressorParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            ratio: 1.0,
            attack_ms: 20.0,
            release_ms: 100.0,
            makeup_gain: 1.0,
            knee: 0.0625,
        }
    }
}

/// Clamp `v` to `[lo, hi]`, replacing NaN with `fallback`.
fn clamp_or(v: f32, lo: f32, hi: f32, fallback: f32) -> f32 {
    if v.is_nan() {
        fallback
    } else {
        v.clamp(lo, hi)
    }
}

impl CompressorParams {
    /// Copy with every field clamped to its documented range (NaN becomes the default).
    /// `knee` keeps values in `[0, 0.001)` as given; they act as 0.001.
    pub fn clamped(&self) -> Self {
        let d = Self::default();
        Self {
            threshold: clamp_or(self.threshold, 0.0, 1.0, d.threshold),
            ratio: clamp_or(self.ratio, 1.0, 100.0, d.ratio),
            attack_ms: clamp_or(self.attack_ms, 0.01, 10_000.0, d.attack_ms),
            release_ms: clamp_or(self.release_ms, 0.01, 10_000.0, d.release_ms),
            makeup_gain: clamp_or(self.makeup_gain, 0.0, 100.0, d.makeup_gain),
            knee: clamp_or(self.knee, 0.0, 1.0, d.knee),
        }
    }
}

/// Static gain reduction in dB (≥ 0) of the gain computer for input level `x_db`,
/// threshold `t_db`, knee width `w_db` and slope `s = 1 − 1/ratio`.
#[inline]
fn gain_reduction(x_db: f64, t_db: f64, w_db: f64, s: f64) -> f64 {
    let over = x_db - t_db;
    let half = 0.5 * w_db;
    if over <= -half {
        0.0
    } else if over < half {
        let d = over + half;
        s * d * d / (2.0 * w_db)
    } else {
        s * over
    }
}

/// One-pole glide toward `target`, snapping when close.
#[inline]
fn glide(cur: &mut f64, target: f64, k: f64) {
    let d = target - *cur;
    if d.abs() < 1e-9 {
        *cur = target;
    } else {
        *cur += k * d;
    }
}

/// The gain-computer quantities that glide on parameter changes.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Curve {
    threshold_db: f64,
    slope: f64,
    knee_db: f64,
    makeup: f64,
}

impl Curve {
    fn from_params(p: &CompressorParams) -> Self {
        let k = (p.knee as f64).clamp(MIN_KNEE, 1.0);
        Self {
            threshold_db: lin_to_db(p.threshold as f64),
            slope: 1.0 - 1.0 / p.ratio as f64,
            knee_db: (-40.0 * k.log10()).max(0.0),
            makeup: p.makeup_gain as f64,
        }
    }
}

/// Linked-channel feed-forward compressor. See [`CompressorParams`] and the module
/// documentation.
#[derive(Debug)]
pub struct Compressor {
    params: CompressorParams,
    sample_rate: f64,
    /// 0 = not configured.
    channels: usize,

    coef_attack: f64,
    coef_release: f64,
    /// `1 − a` of the parameter glide.
    glide_k: f64,
    target: Curve,
    current: Curve,

    /// Smoothed gain reduction (dB, ≥ 0).
    reduction_db: f64,
    /// True until the first frame is processed after configure/reset: parameters snap.
    fresh: bool,
}

impl Default for Compressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor {
    /// Create an unconfigured compressor with default parameters.
    pub fn new() -> Self {
        let params = CompressorParams::default();
        let curve = Curve::from_params(&params);
        let mut c = Self {
            params,
            sample_rate: 48_000.0,
            channels: 0,
            coef_attack: 0.0,
            coef_release: 0.0,
            glide_k: 1.0,
            target: curve,
            current: curve,
            reduction_db: 0.0,
            fresh: true,
        };
        c.update_coefficients();
        c
    }

    /// Update parameters (clamped to their ranges). Never allocates.
    pub fn set_params(&mut self, params: &CompressorParams) {
        self.params = params.clamped();
        self.update_coefficients();
    }

    /// Current (clamped) parameters.
    pub fn params(&self) -> &CompressorParams {
        &self.params
    }

    /// Smoothed gain reduction (dB, ≥ 0) after the most recently processed frame,
    /// excluding makeup gain.
    pub fn gain_reduction_db(&self) -> f64 {
        self.reduction_db
    }

    fn update_coefficients(&mut self) {
        let fs = self.sample_rate;
        self.coef_attack = one_pole_coef(self.params.attack_ms as f64, fs);
        self.coef_release = one_pole_coef(self.params.release_ms as f64, fs);
        self.glide_k = 1.0 - one_pole_coef(PARAM_SMOOTHING_MS, fs);
        self.target = Curve::from_params(&self.params);
        if self.fresh {
            self.current = self.target;
        }
    }
}

impl Processor for Compressor {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        self.sample_rate = (sample_rate as f64).clamp(1.0, 1.0e6);
        self.channels = channels;
        self.reset();
    }

    fn process(&mut self, samples: &mut [f32]) {
        if self.channels == 0 || samples.len() < self.channels {
            return;
        }
        self.fresh = false;
        let k = self.glide_k;
        for frame in samples.chunks_exact_mut(self.channels) {
            let c = &mut self.current;
            glide(&mut c.threshold_db, self.target.threshold_db, k);
            glide(&mut c.slope, self.target.slope, k);
            glide(&mut c.knee_db, self.target.knee_db, k);
            glide(&mut c.makeup, self.target.makeup, k);

            let mut peak = 0.0f32;
            for s in frame.iter() {
                peak = peak.max(s.abs());
            }
            let x_db = lin_to_db(peak as f64);
            let r = gain_reduction(x_db, c.threshold_db, c.knee_db, c.slope);

            let a = if r > self.reduction_db {
                self.coef_attack
            } else {
                self.coef_release
            };
            self.reduction_db = flush_denormal(a * self.reduction_db + (1.0 - a) * r);

            let g = if self.reduction_db > 0.0 {
                db_to_lin(-self.reduction_db) * c.makeup
            } else {
                c.makeup
            };
            for s in frame.iter_mut() {
                *s = (*s as f64 * g) as f32;
            }
        }
    }

    fn reset(&mut self) {
        self.reduction_db = 0.0;
        self.fresh = true;
        self.update_coefficients();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gain_computer_is_continuous_at_knee_edges() {
        let (t, w, s) = (-20.0, 12.0, 0.75);
        for edge in [t - w / 2.0, t + w / 2.0] {
            let lo = gain_reduction(edge - 1e-9, t, w, s);
            let hi = gain_reduction(edge + 1e-9, t, w, s);
            assert!((lo - hi).abs() < 1e-6);
        }
        // Hard knee.
        assert_eq!(gain_reduction(-21.0, t, 0.0, s), 0.0);
        assert!((gain_reduction(-10.0, t, 0.0, s) - 7.5).abs() < 1e-12);
    }

    #[test]
    fn knee_width_matches_definition() {
        let p = CompressorParams {
            knee: 0.1,
            ..Default::default()
        };
        assert!((Curve::from_params(&p).knee_db - 40.0).abs() < 1e-5);
        let p0 = CompressorParams {
            knee: 0.0,
            ..Default::default()
        };
        assert!((Curve::from_params(&p0).knee_db - 120.0).abs() < 1e-9);
    }
}
