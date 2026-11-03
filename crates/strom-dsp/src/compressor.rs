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
//! 2. **Gain computer** ([`GainComputer`], static curve), with threshold
//!    `T = 20·log10(threshold)`, knee width `W = −40·log10(k)` dB (`k` = `knee`
//!    clamped to `[0.001, 1]`) and slope `S = 1 − 1/ratio`, the gain reduction
//!    `r ≥ 0` (dB) is
//!    - `r = 0` for `x < T − W/2`,
//!    - `r = S·(x − T + W/2)² / (2W)` for `T − W/2 ≤ x ≤ T + W/2` (quadratic soft knee),
//!    - `r = S·(x − T)` for `x > T + W/2`.
//!
//!    The output level is `x − r`, i.e. above the knee `T + (x − T)/ratio`.
//!    The knee region in linear terms spans `threshold·k … threshold/k`. `W = 0`
//!    (`k = 1`) is a hard knee.
//! 3. **Smoothing** ([`AttackRelease`], branching one-pole in the dB domain):
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
//! # Efficiency
//!
//! Processing runs in blocks of [`BLOCK_FRAMES`]: per-frame peaks, the serial
//! detector/smoother chain, then gain application. While the parameters are not
//! gliding, a frame whose peak is below the knee's lower edge (compared in the
//! linear domain) skips the `log10`, and a frame with zero smoothed reduction
//! skips the `exp`; the result is the same as evaluating the full formula.
//!
//! # Denormals
//!
//! The smoothed reduction `y` snaps to its target when within
//! [`DENORMAL_FLUSH`](crate::util::DENORMAL_FLUSH) (so it decays to exactly 0);
//! parameter smoothers snap to their target when within `1e-9`.
//!
//! Ratio 1 gives `S = 0`, hence `r = 0` and exactly unity gain apart from makeup.

use crate::dynamics::frame::{apply_gains_simd, frame_peaks_simd, BLOCK_FRAMES};
use crate::dynamics::{vmath, AttackRelease, GainComputer, OnePole};
use crate::simd::{detect, Level};
use crate::Processor;
use fearless_simd::dispatch;
use fearless_simd::prelude::*;

pub use crate::dynamics::MIN_KNEE;

/// Time constant (ms) of the one-pole glide applied to threshold, ratio, knee and makeup.
pub const PARAM_SMOOTHING_MS: f64 = 10.0;

/// Snap distance of the parameter glides.
const GLIDE_SNAP: f64 = 1e-9;

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
#[cfg(test)]
fn gain_reduction(x_db: f64, t_db: f64, w_db: f64, s: f64) -> f64 {
    GainComputer {
        threshold_db: t_db,
        knee_db: w_db,
        slope: s,
    }
    .reduction_db(x_db)
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
        let gc = GainComputer::from_linear(p.threshold as f64, p.ratio as f64, p.knee as f64);
        Self {
            threshold_db: gc.threshold_db,
            slope: gc.slope,
            knee_db: gc.knee_db,
            makeup: p.makeup_gain as f64,
        }
    }

    fn computer(&self) -> GainComputer {
        GainComputer {
            threshold_db: self.threshold_db,
            knee_db: self.knee_db,
            slope: self.slope,
        }
    }
}

/// One-pole glides for each [`Curve`] field.
#[derive(Debug, Clone, Copy)]
struct CurveGlide {
    threshold_db: OnePole,
    slope: OnePole,
    knee_db: OnePole,
    makeup: OnePole,
}

impl CurveGlide {
    fn set_time(&mut self, fs: f64) {
        for g in [
            &mut self.threshold_db,
            &mut self.slope,
            &mut self.knee_db,
            &mut self.makeup,
        ] {
            g.set_time(PARAM_SMOOTHING_MS, fs);
        }
    }

    fn jump(&mut self, c: &Curve) {
        self.threshold_db.set_value(c.threshold_db);
        self.slope.set_value(c.slope);
        self.knee_db.set_value(c.knee_db);
        self.makeup.set_value(c.makeup);
    }

    #[inline]
    fn step(&mut self, c: &Curve) -> Curve {
        Curve {
            threshold_db: self.threshold_db.step(c.threshold_db),
            slope: self.slope.step(c.slope),
            knee_db: self.knee_db.step(c.knee_db),
            makeup: self.makeup.step(c.makeup),
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

    target: Curve,
    /// Curve in force (equal to `target` when not gliding).
    current: Curve,
    glide: CurveGlide,
    gliding: bool,
    /// Linear level at or below which the settled curve gives exactly 0 dB reduction.
    knee_start_lin: f64,

    /// Smoothed gain reduction (dB, ≥ 0).
    reduction: AttackRelease,
    /// True until the first frame is processed after configure/reset: parameters snap.
    fresh: bool,
    /// Instruction set of the hot loops (detected in `configure`).
    level: Level,
    /// Set by `set_simd_level`: `configure` keeps `level` instead of detecting.
    level_forced: bool,
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
        let glide = OnePole::default().with_snap(GLIDE_SNAP);
        let mut c = Self {
            params,
            sample_rate: 48_000.0,
            channels: 0,
            target: curve,
            current: curve,
            glide: CurveGlide {
                threshold_db: glide,
                slope: glide,
                knee_db: glide,
                makeup: glide,
            },
            gliding: false,
            knee_start_lin: curve.computer().knee_start_lin(),
            reduction: AttackRelease::default(),
            fresh: true,
            level: Level::baseline(),
            level_forced: false,
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
        self.reduction.value()
    }

    /// Force the SIMD instruction set used by `process` (for hosts and tests),
    /// e.g. [`Level::fallback`] to run the portable scalar path. By default
    /// `configure` detects the best level ([`crate::simd::detect`]); a level
    /// set here is kept across later `configure` calls. Every level produces
    /// bit-identical output (the dB conversions use the same approximation on
    /// every level, see the module docs). Never allocates.
    pub fn set_simd_level(&mut self, level: Level) {
        self.level = level;
        self.level_forced = true;
    }

    /// The SIMD instruction set `process` currently dispatches to.
    pub fn simd_level(&self) -> Level {
        self.level
    }

    /// Block loop, monomorphised per instruction set (see "Efficiency" in
    /// the module docs).
    #[inline(always)]
    fn process_blocks<S: Simd>(&mut self, simd: S, samples: &mut [f32]) {
        let ch = self.channels;
        let n = S::f64s::LEN;
        // Per-frame scratch, reused through the stages of a block: peak →
        // level (dB) → static reduction → smoothed reduction → gain.
        let mut peaks = [0.0f32; BLOCK_FRAMES];
        let mut level = [0.0f64; BLOCK_FRAMES];
        let mut makeup = [0.0f64; BLOCK_FRAMES];
        for block in samples.chunks_mut(BLOCK_FRAMES * ch) {
            let frames = block.len() / ch;
            // Vector loops cover whole vectors; lanes past `frames` hold
            // finite leftovers and are ignored. BLOCK_FRAMES is a multiple of
            // every native width.
            let lanes = frames.div_ceil(n) * n;
            frame_peaks_simd(simd, block, ch, &mut peaks[..frames]);
            // Settled curve and the whole block below the knee: the static
            // reduction is exactly 0, no need for the logarithm.
            let quiet = !self.gliding
                && peaks[..frames].iter().fold(0.0f32, |m, &p| m.max(p)) as f64
                    <= self.knee_start_lin;
            if quiet {
                level[..frames].fill(0.0);
            } else {
                for (l, &p) in level.iter_mut().zip(&peaks) {
                    *l = p as f64;
                }
                for v in level[..lanes].chunks_exact_mut(n) {
                    vmath::lin_to_db(simd, S::f64s::from_slice(simd, v)).store_slice(v);
                }
            }

            if quiet {
                makeup.fill(self.current.makeup);
            } else if self.gliding {
                for (l, mk) in level[..frames].iter_mut().zip(&mut makeup) {
                    self.current = self.glide.step(&self.target);
                    if self.current == self.target {
                        self.gliding = false;
                        self.knee_start_lin = self.current.computer().knee_start_lin();
                    }
                    *l = self.current.computer().reduction_db(*l);
                    *mk = self.current.makeup;
                }
            } else {
                let gc = self.current.computer();
                for v in level[..lanes].chunks_exact_mut(n) {
                    reduction_db_simd(simd, &gc, S::f64s::from_slice(simd, v)).store_slice(v);
                }
                makeup.fill(self.current.makeup);
            }
            // Serial attack/release smoothing of the reduction (dB).
            self.reduction.step_block(&mut level[..frames]);

            // Gain 10^(−y/20)·makeup; exactly `makeup` at zero reduction
            // (y ≥ 0 always, so a block without reduction skips the exp).
            if level[..frames].iter().any(|&y| y > 0.0) {
                let zero = S::f64s::splat(simd, 0.0);
                for (v, mk) in level[..lanes]
                    .chunks_exact_mut(n)
                    .zip(makeup.chunks_exact(n))
                {
                    let y = S::f64s::from_slice(simd, v);
                    let mk = S::f64s::from_slice(simd, mk);
                    let g = vmath::db_to_gain(simd, y) * mk;
                    y.simd_gt(zero).select(g, mk).store_slice(v);
                }
            } else {
                level[..frames].copy_from_slice(&makeup[..frames]);
            }
            for (p, &g) in peaks.iter_mut().zip(&level[..frames]) {
                *p = g as f32;
            }
            apply_gains_simd(simd, block, ch, &peaks[..frames]);
        }
    }

    fn update_coefficients(&mut self) {
        let fs = self.sample_rate;
        self.reduction.set_times(
            self.params.attack_ms as f64,
            self.params.release_ms as f64,
            fs,
        );
        self.glide.set_time(fs);
        self.target = Curve::from_params(&self.params);
        if self.fresh {
            self.glide.jump(&self.target);
            self.current = self.target;
        }
        self.gliding = self.current != self.target;
        self.knee_start_lin = self.current.computer().knee_start_lin();
    }
}

/// [`GainComputer::reduction_db`] per lane: the same operations in the same
/// order, so every lane equals the scalar result bit for bit.
#[inline(always)]
fn reduction_db_simd<S: Simd>(simd: S, gc: &GainComputer, x: S::f64s) -> S::f64s {
    let over = x - gc.threshold_db;
    let half = 0.5 * gc.knee_db;
    let d = over + half;
    let in_knee = d * gc.slope * d / (2.0 * gc.knee_db);
    let above = over * gc.slope;
    let r = over
        .simd_lt(S::f64s::splat(simd, half))
        .select(in_knee, above);
    over.simd_le(S::f64s::splat(simd, -half))
        .select(S::f64s::splat(simd, 0.0), r)
}

impl Processor for Compressor {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        self.sample_rate = (sample_rate as f64).clamp(1.0, 1.0e6);
        self.channels = channels;
        if !self.level_forced {
            self.level = detect();
        }
        self.reset();
    }

    fn process(&mut self, samples: &mut [f32]) {
        let ch = self.channels;
        if ch == 0 || samples.len() < ch {
            return;
        }
        self.fresh = false;
        dispatch!(self.level, simd => self.process_blocks(simd, samples));
    }

    fn reset(&mut self) {
        self.reduction.set_value(0.0);
        self.fresh = true;
        self.update_coefficients();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::lin_to_db;

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

    #[test]
    fn fast_path_matches_full_formula() {
        // Settled curve: skipping log10 below the knee must not change the gain.
        let p = CompressorParams {
            threshold: 0.25,
            ratio: 4.0,
            knee: 0.5,
            attack_ms: 0.01,
            release_ms: 0.01,
            ..Default::default()
        };
        let gc = Curve::from_params(&p.clamped()).computer();
        let edge = gc.knee_start_lin();
        for x in [0.0, 1e-6, edge * 0.999, edge, edge * 1.000_001, 0.2, 0.9] {
            let mut c = Compressor::new();
            c.set_params(&p);
            c.configure(48_000.0, 1);
            let mut buf = [x as f32; 64];
            c.process(&mut buf);
            let want = gc.reduction_db(lin_to_db(x as f32 as f64));
            assert!((c.gain_reduction_db() - want).abs() < 1e-9, "x={x}");
        }
    }
}
