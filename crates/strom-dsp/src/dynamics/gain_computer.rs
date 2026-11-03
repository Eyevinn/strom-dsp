//! Static gain computer of a downward compressor with quadratic soft knee.
//!
//! Reference: D. Giannoulis, M. Massberg, J. D. Reiss, "Digital Dynamic Range
//! Compressor Design — A Tutorial and Analysis", JAES 60(6), 2012 (gain computer
//! with soft knee). Knee parameterisation per spec section 3.3.

use crate::util::{db_to_lin, lin_to_db};

/// Smallest effective linear knee factor `k`; smaller values (including 0) act as this.
pub const MIN_KNEE: f64 = 0.001;

/// Static compressor curve in the dB domain.
///
/// With threshold `T` (dB), knee width `W` (dB) and slope `S = 1 − 1/ratio`, the
/// gain reduction `r ≥ 0` (dB) for an input level `x` (dB) is
///
/// - `r = 0` for `x ≤ T − W/2`,
/// - `r = S·(x − T + W/2)² / (2W)` for `T − W/2 < x < T + W/2`,
/// - `r = S·(x − T)` for `x ≥ T + W/2`,
///
/// which is continuous with a continuous first derivative. `W = 0` is a hard knee.
/// The output level is `x − r`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GainComputer {
    /// Threshold `T` (dB).
    pub threshold_db: f64,
    /// Knee width `W` (dB, ≥ 0).
    pub knee_db: f64,
    /// Slope `S = 1 − 1/ratio` (0 = no compression, 1 = limiting).
    pub slope: f64,
}

impl GainComputer {
    /// Curve from the interface units: `threshold` as linear amplitude, `ratio`
    /// (≥ 1) and the linear knee factor `knee` (spec 3.3): the knee spans
    /// `threshold·k … threshold/k`, i.e. `W = −40·log10(k)` dB, with `k` clamped
    /// to `[MIN_KNEE, 1]`.
    pub fn from_linear(threshold: f64, ratio: f64, knee: f64) -> Self {
        let k = if knee.is_nan() {
            1.0
        } else {
            knee.clamp(MIN_KNEE, 1.0)
        };
        Self {
            threshold_db: lin_to_db(threshold),
            knee_db: (-40.0 * k.log10()).max(0.0),
            slope: 1.0 - 1.0 / ratio.max(1.0),
        }
    }

    /// Gain reduction (dB, ≥ 0) for input level `x_db`.
    #[inline]
    pub fn reduction_db(&self, x_db: f64) -> f64 {
        let over = x_db - self.threshold_db;
        let half = 0.5 * self.knee_db;
        if over <= -half {
            0.0
        } else if over < half {
            let d = over + half;
            self.slope * d * d / (2.0 * self.knee_db)
        } else {
            self.slope * over
        }
    }

    /// Lower edge of the knee, `T − W/2` (dB). Below it the reduction is 0.
    #[inline]
    pub fn knee_start_db(&self) -> f64 {
        self.threshold_db - 0.5 * self.knee_db
    }

    /// Linear amplitude at or below which [`reduction_db`](Self::reduction_db) is
    /// exactly 0 (0 if the curve never compresses, i.e. `S = 0` gives `+∞`).
    ///
    /// Callers can compare a linear level against this and skip the `log10`.
    /// It is rounded slightly down, so a level above it may still give 0.
    pub fn knee_start_lin(&self) -> f64 {
        if self.slope <= 0.0 {
            f64::INFINITY
        } else {
            db_to_lin(self.knee_start_db()) * (1.0 - 1e-9)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_knee_is_piecewise_linear() {
        let g = GainComputer {
            threshold_db: -20.0,
            knee_db: 0.0,
            slope: 0.75,
        };
        assert_eq!(g.reduction_db(-21.0), 0.0);
        assert!((g.reduction_db(-10.0) - 7.5).abs() < 1e-12);
    }

    #[test]
    fn soft_knee_continuous_value_and_slope() {
        let g = GainComputer {
            threshold_db: -20.0,
            knee_db: 12.0,
            slope: 0.75,
        };
        for edge in [-26.0, -14.0] {
            let (lo, hi) = (g.reduction_db(edge - 1e-7), g.reduction_db(edge + 1e-7));
            assert!((lo - hi).abs() < 1e-6);
            let h = 1e-4;
            let dl = (g.reduction_db(edge) - g.reduction_db(edge - h)) / h;
            let dr = (g.reduction_db(edge + h) - g.reduction_db(edge)) / h;
            assert!((dl - dr).abs() < 1e-3, "slope jump at {edge}");
        }
        // At the threshold the knee gives S·W/8.
        assert!((g.reduction_db(-20.0) - 0.75 * 12.0 / 8.0).abs() < 1e-12);
    }

    #[test]
    fn knee_definition_spec_3_3() {
        let g = GainComputer::from_linear(0.5, 4.0, 0.1);
        assert!((g.knee_db - 40.0).abs() < 1e-9);
        assert!((g.slope - 0.75).abs() < 1e-15);
        // Region spans threshold·k … threshold/k.
        assert!((g.knee_start_db() - lin_to_db(0.05)).abs() < 1e-9);
        assert!((g.threshold_db + 0.5 * g.knee_db - lin_to_db(5.0)).abs() < 1e-9);
        assert!((GainComputer::from_linear(0.5, 4.0, 0.0).knee_db - 120.0).abs() < 1e-9);
        assert_eq!(GainComputer::from_linear(0.5, 4.0, 1.0).knee_db, 0.0);
        assert_eq!(GainComputer::from_linear(0.5, 1.0, 0.1).slope, 0.0);
    }

    #[test]
    fn knee_start_lin_bounds_zero_region() {
        let g = GainComputer::from_linear(0.25, 4.0, 0.0625);
        let l = g.knee_start_lin();
        assert_eq!(g.reduction_db(lin_to_db(l)), 0.0);
        assert!(g.reduction_db(lin_to_db(l * 1.001)) > 0.0);
        assert_eq!(
            GainComputer::from_linear(0.25, 1.0, 0.5).knee_start_lin(),
            f64::INFINITY
        );
    }
}
