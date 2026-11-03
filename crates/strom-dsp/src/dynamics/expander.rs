//! Downward expansion below a threshold: the static curve and the dynamic
//! stage that adds it to a compressor's gain computer.
//!
//! References: Giannoulis, Massberg & Reiss, "Digital Dynamic Range
//! Compressor Design — A Tutorial and Analysis", JAES 60(6), 2012 (the
//! gain computer in the dB domain with a quadratic soft knee; an expander is
//! the same curve mirrored below its threshold); Zölzer (ed.), *DAFX*,
//! dynamics chapter (noise gate and expander: hold time, slower release).
//!
//! # Static curve ([`ExpanderCurve`])
//!
//! With expander threshold `E` (dB), knee width `W` (dB), ratio `R ≥ 1`
//! (1 dB below `E` in gives `R` dB below `E` out), slope `S = R − 1` and
//! floor `F ≥ 0` (largest attenuation, dB), the attenuation `e ≥ 0` (dB) for
//! the detector level `x` (dB) is `min(F, e₀)` with
//!
//! - `e₀ = 0` for `x ≥ E + W/2`,
//! - `e₀ = S·(E + W/2 − x)² / (2W)` for `E − W/2 < x < E + W/2` (quadratic soft knee),
//! - `e₀ = S·(E − x)` for `x ≤ E − W/2`.
//!
//! It is continuous with a continuous first derivative except at the floor.
//! `R = 1` gives `e = 0` everywhere.
//!
//! # Combined with a compressor
//!
//! [`Compressor::set_expander`](crate::Compressor::set_expander) adds `e` to
//! the compressor's static reduction `r` of the same detector level: the
//! total static curve is compression above the compressor threshold, unity
//! between the two thresholds and expansion below the expander threshold,
//! with a soft knee at each. The two parts are smoothed separately, because
//! their good time constants differ: an expander must open fast (a word
//! start) and close slowly (a word ending), the opposite of a compressor's
//! attack and release.
//!
//! # Dynamics ([`ExpanderStage`])
//!
//! Per frame, with the static attenuation `e[n]` and the smoothed value `y`:
//!
//! 1. **Hold.** When `e[n] ≤ y` (the expander wants to open, or stay) the hold
//!    counter restarts at `H` frames and the target is `e[n]`. Otherwise the
//!    target stays `y` while the counter runs down, and becomes `e[n]` once it
//!    has run out. So the expander closes only after the level has stayed
//!    below its opening point for `hold_ms`: word endings and short gaps are
//!    not chopped.
//! 2. **Smoothing.** A branching one-pole ([`AttackRelease`]) in dB: the
//!    *open* time when the attenuation falls, the slower *close* time when it
//!    rises.
//!
//! **Against chatter** this stage uses the **hold plus the slower close
//! time**, not hysteresis: a level hovering at the threshold keeps restarting
//! the hold at its loudest moments, so the gain stays near its most open
//! value, and any closing that does happen is spread over the close time.
//! The soft knee means there is no switching point to chatter around in the
//! first place: the static attenuation near the threshold is a few tenths of
//! a dB.
//!
//! # Parameter changes
//!
//! Threshold, slope, knee and floor glide with one-poles of
//! [`EXPANDER_SMOOTHING_MS`] (10 ms, as the compressor's curve), so no
//! change steps the gain. Times change only coefficients. With ratio 1 the
//! slope glides to 0, the attenuation opens to exactly 0 and the stage turns
//! itself off ([`ExpanderStage::active`]): the host then skips it, which
//! keeps a compressor without expansion bit-exact.

use crate::dynamics::{AttackRelease, OnePole};

/// Time constant (ms) of the glides of threshold, slope, knee and floor.
pub const EXPANDER_SMOOTHING_MS: f64 = 10.0;

/// Range of the expander threshold (dB, detector level).
pub const EXPANDER_THRESHOLD_RANGE_DB: (f32, f32) = (-80.0, 0.0);
/// Range of the expander ratio (1 = off).
pub const EXPANDER_RATIO_RANGE: (f32, f32) = (1.0, 10.0);
/// Range of the floor: the largest attenuation (dB, ≥ 0).
pub const EXPANDER_FLOOR_RANGE_DB: (f32, f32) = (0.0, 40.0);
/// Range of the knee width (dB).
pub const EXPANDER_KNEE_RANGE_DB: (f32, f32) = (0.0, 24.0);
/// Range of the open time (ms).
pub const EXPANDER_OPEN_RANGE_MS: (f32, f32) = (0.1, 100.0);
/// Range of the close time (ms).
pub const EXPANDER_CLOSE_RANGE_MS: (f32, f32) = (1.0, 5000.0);
/// Range of the hold time (ms).
pub const EXPANDER_HOLD_RANGE_MS: (f32, f32) = (0.0, 1000.0);

/// Snap distance (dB) of the glides and of the smoothed attenuation.
const SNAP: f64 = 1e-9;

/// Static downward-expander curve (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpanderCurve {
    /// Expander threshold `E` (dB).
    pub threshold_db: f64,
    /// Knee width `W` (dB, ≥ 0; 0 = hard knee).
    pub knee_db: f64,
    /// Slope `S = ratio − 1` (≥ 0; 0 = no expansion).
    pub slope: f64,
    /// Floor `F`: the largest attenuation (dB, ≥ 0).
    pub floor_db: f64,
}

impl ExpanderCurve {
    /// Attenuation (dB, `0 … floor`) for the detector level `x_db`.
    #[inline]
    pub fn reduction_db(&self, x_db: f64) -> f64 {
        let under = self.threshold_db - x_db;
        let half = 0.5 * self.knee_db;
        let e = if under <= -half {
            0.0
        } else if under < half {
            let d = under + half;
            self.slope * d * d / (2.0 * self.knee_db)
        } else {
            self.slope * under
        };
        if e < self.floor_db {
            e
        } else {
            self.floor_db
        }
    }

    /// Upper edge of the knee, `E + W/2` (dB): at or above it the
    /// attenuation is exactly 0.
    #[inline]
    pub fn knee_end_db(&self) -> f64 {
        self.threshold_db + 0.5 * self.knee_db
    }
}

/// Parameters of the expander part of a gain computer. Units: dB, ms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpanderParams {
    /// Expander threshold (dB, −80 … 0), against the same detector level
    /// as the compressor threshold.
    pub threshold_db: f32,
    /// Ratio (1 … 10; 1 = off): 1 dB below the threshold gives `ratio` dB
    /// below it at the output.
    pub ratio: f32,
    /// Largest attenuation (dB, 0 … 40).
    pub floor_db: f32,
    /// Soft-knee width (dB, 0 … 24; 0 = hard knee).
    pub knee_db: f32,
    /// Time constant of opening (ms, 0.1 … 100).
    pub open_ms: f32,
    /// Time constant of closing (ms, 1 … 5000).
    pub close_ms: f32,
    /// Hold before closing (ms, 0 … 1000).
    pub hold_ms: f32,
}

impl Default for ExpanderParams {
    /// Off (ratio 1), with gentle settings ready for a ratio: threshold
    /// −60 dB, floor 12 dB, knee 6 dB, open 3 ms (faster than a consonant
    /// onset), close 150 ms (a natural decay, not a cut), hold 100 ms (bridges
    /// gaps inside words).
    fn default() -> Self {
        Self {
            threshold_db: -60.0,
            ratio: 1.0,
            floor_db: 12.0,
            knee_db: 6.0,
            open_ms: 3.0,
            close_ms: 150.0,
            hold_ms: 100.0,
        }
    }
}

impl ExpanderParams {
    /// Copy with every field clamped to its range (NaN becomes the default).
    pub fn clamped(&self) -> Self {
        let d = Self::default();
        let c =
            |v: f32, (lo, hi): (f32, f32), fb: f32| if v.is_nan() { fb } else { v.clamp(lo, hi) };
        Self {
            threshold_db: c(
                self.threshold_db,
                EXPANDER_THRESHOLD_RANGE_DB,
                d.threshold_db,
            ),
            ratio: c(self.ratio, EXPANDER_RATIO_RANGE, d.ratio),
            floor_db: c(self.floor_db, EXPANDER_FLOOR_RANGE_DB, d.floor_db),
            knee_db: c(self.knee_db, EXPANDER_KNEE_RANGE_DB, d.knee_db),
            open_ms: c(self.open_ms, EXPANDER_OPEN_RANGE_MS, d.open_ms),
            close_ms: c(self.close_ms, EXPANDER_CLOSE_RANGE_MS, d.close_ms),
            hold_ms: c(self.hold_ms, EXPANDER_HOLD_RANGE_MS, d.hold_ms),
        }
    }

    /// The static curve of these settings (after clamping).
    pub fn curve(&self) -> ExpanderCurve {
        let p = self.clamped();
        ExpanderCurve {
            threshold_db: p.threshold_db as f64,
            knee_db: p.knee_db as f64,
            slope: p.ratio as f64 - 1.0,
            floor_db: p.floor_db as f64,
        }
    }
}

/// The dynamic expander stage: curve glides, hold and open/close smoothing.
/// See the module docs. Never allocates.
#[derive(Debug, Clone)]
pub struct ExpanderStage {
    params: ExpanderParams,
    sample_rate: f64,
    target: ExpanderCurve,
    current: ExpanderCurve,
    glide: [OnePole; 4],
    gliding: bool,
    smooth: AttackRelease,
    hold_frames: u32,
    counter: u32,
}

impl Default for ExpanderStage {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpanderStage {
    /// An inactive stage (ratio 1) at 48 kHz.
    pub fn new() -> Self {
        let params = ExpanderParams::default();
        let curve = params.curve();
        let mut s = Self {
            params,
            sample_rate: 48_000.0,
            target: curve,
            current: curve,
            glide: [OnePole::default().with_snap(SNAP); 4],
            gliding: false,
            smooth: AttackRelease::default().with_snap(SNAP),
            hold_frames: 0,
            counter: 0,
        };
        s.set_rate(48_000.0);
        s
    }

    /// Set the sample rate (Hz); keeps the state.
    pub fn set_rate(&mut self, sample_rate: f64) {
        self.sample_rate = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate
        } else {
            48_000.0
        };
        for g in &mut self.glide {
            g.set_time(EXPANDER_SMOOTHING_MS, self.sample_rate);
        }
        self.update_times();
    }

    /// New parameters (clamped). With `snap` (before the first frame) the
    /// curve jumps to them; otherwise it glides.
    pub fn set_params(&mut self, params: &ExpanderParams, snap: bool) {
        self.params = params.clamped();
        self.target = self.params.curve();
        if snap {
            self.jump();
        }
        self.gliding = self.current != self.target;
        self.update_times();
    }

    /// Current (clamped) parameters.
    pub fn params(&self) -> &ExpanderParams {
        &self.params
    }

    /// Forget the signal: attenuation 0, hold cleared, curve at its target.
    pub fn reset(&mut self) {
        self.jump();
        self.gliding = false;
        self.smooth.set_value(0.0);
        self.counter = 0;
    }

    /// Whether the stage can attenuate: a slope above 0 (set or still
    /// gliding) or attenuation still applied. When false, [`step`](Self::step)
    /// would return exactly 0 and the host may skip it.
    #[inline]
    pub fn active(&self) -> bool {
        self.target.slope > 0.0 || self.current.slope > 0.0 || self.smooth.value() > 0.0
    }

    /// Smoothed attenuation (dB, ≥ 0) after the most recent step.
    #[inline]
    pub fn reduction_db(&self) -> f64 {
        self.smooth.value()
    }

    /// The curve in force (equal to the target when not gliding).
    pub fn curve(&self) -> ExpanderCurve {
        self.current
    }

    /// Advance one frame with detector level `x_db` (dB); returns the
    /// smoothed attenuation (dB, ≥ 0).
    #[inline]
    pub fn step(&mut self, x_db: f64) -> f64 {
        if self.gliding {
            self.current = ExpanderCurve {
                threshold_db: self.glide[0].step(self.target.threshold_db),
                knee_db: self.glide[1].step(self.target.knee_db),
                slope: self.glide[2].step(self.target.slope),
                floor_db: self.glide[3].step(self.target.floor_db),
            };
            if self.current == self.target {
                self.gliding = false;
            }
        }
        let e = if x_db >= self.current.knee_end_db() {
            0.0
        } else {
            self.current.reduction_db(x_db)
        };
        let y = self.smooth.value();
        let target = if e <= y {
            self.counter = self.hold_frames;
            e
        } else if self.counter > 0 {
            self.counter -= 1;
            y
        } else {
            e
        };
        self.smooth.step(target)
    }

    fn jump(&mut self) {
        self.current = self.target;
        self.glide[0].set_value(self.target.threshold_db);
        self.glide[1].set_value(self.target.knee_db);
        self.glide[2].set_value(self.target.slope);
        self.glide[3].set_value(self.target.floor_db);
    }

    fn update_times(&mut self) {
        let p = self.params;
        let fs = self.sample_rate;
        // Rising attenuation is closing; falling is opening.
        self.smooth
            .set_times(p.close_ms as f64, p.open_ms as f64, fs);
        let h = (p.hold_ms as f64 * 0.001 * fs).round();
        self.hold_frames = if h.is_finite() && h > 0.0 {
            h.min(u32::MAX as f64) as u32
        } else {
            0
        };
        self.counter = self.counter.min(self.hold_frames);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(t: f64, w: f64, r: f64, f: f64) -> ExpanderCurve {
        ExpanderCurve {
            threshold_db: t,
            knee_db: w,
            slope: r - 1.0,
            floor_db: f,
        }
    }

    #[test]
    fn hard_knee_is_piecewise_linear_with_floor() {
        let c = curve(-40.0, 0.0, 3.0, 12.0);
        assert_eq!(c.reduction_db(-40.0), 0.0);
        assert_eq!(c.reduction_db(-30.0), 0.0);
        assert!((c.reduction_db(-43.0) - 6.0).abs() < 1e-12);
        assert_eq!(c.reduction_db(-60.0), 12.0);
    }

    #[test]
    fn soft_knee_is_smooth() {
        let c = curve(-40.0, 8.0, 2.0, 60.0);
        for edge in [-44.0, -36.0] {
            let (lo, hi) = (c.reduction_db(edge - 1e-7), c.reduction_db(edge + 1e-7));
            assert!((lo - hi).abs() < 1e-6);
            let h = 1e-4;
            let dl = (c.reduction_db(edge) - c.reduction_db(edge - h)) / h;
            let dr = (c.reduction_db(edge + h) - c.reduction_db(edge)) / h;
            assert!((dl - dr).abs() < 1e-3, "slope jump at {edge}");
        }
        assert!((c.reduction_db(-40.0) - 8.0 / 8.0).abs() < 1e-12);
        assert_eq!(c.reduction_db(c.knee_end_db()), 0.0);
    }

    #[test]
    fn ratio_one_is_inactive() {
        let mut s = ExpanderStage::new();
        s.set_rate(48_000.0);
        assert!(!s.active());
        for _ in 0..100 {
            assert_eq!(s.step(-120.0), 0.0);
        }
    }

    #[test]
    fn turns_itself_off_after_ratio_one() {
        let mut s = ExpanderStage::new();
        let on = ExpanderParams {
            ratio: 4.0,
            threshold_db: -30.0,
            ..Default::default()
        };
        s.set_params(&on, true);
        for _ in 0..240_000 {
            s.step(-80.0);
        }
        assert!((s.reduction_db() - on.floor_db as f64).abs() < 1e-6);
        s.set_params(&ExpanderParams { ratio: 1.0, ..on }, false);
        for _ in 0..48_000 {
            s.step(-80.0);
        }
        assert!(!s.active());
        assert_eq!(s.reduction_db(), 0.0);
    }

    #[test]
    fn hold_delays_closing() {
        let fs = 48_000.0;
        let mut s = ExpanderStage::new();
        s.set_rate(fs);
        let p = ExpanderParams {
            ratio: 3.0,
            threshold_db: -40.0,
            knee_db: 0.0,
            hold_ms: 50.0,
            ..Default::default()
        };
        s.set_params(&p, true);
        for _ in 0..4800 {
            s.step(-20.0);
        }
        // Level drops: nothing happens for the hold time.
        let hold = (0.05 * fs) as usize;
        for _ in 0..hold {
            assert_eq!(s.step(-80.0), 0.0);
        }
        s.step(-80.0);
        assert!(s.reduction_db() > 0.0);
    }
}
