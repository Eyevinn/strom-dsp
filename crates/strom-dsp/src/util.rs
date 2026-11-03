//! Small shared helpers: unit conversion, one-pole time constants and denormal handling.

/// Convert decibels to a linear amplitude factor: `10^(db/20)`.
#[inline]
pub fn db_to_lin(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

/// Convert a linear amplitude factor to decibels: `20·log10(x)`.
/// Values at or below [`MIN_LIN`] map to the floor `20·log10(MIN_LIN)` (−200 dB)
/// so the result is always finite.
#[inline]
pub fn lin_to_db(x: f64) -> f64 {
    20.0 * x.max(MIN_LIN).log10()
}

/// Smallest linear amplitude treated as non-zero (−200 dB).
pub const MIN_LIN: f64 = 1e-10;

/// Absolute values below this are flushed to zero in recursive states
/// (about −300 dB, far above the f64 denormal range but far below audibility).
pub const DENORMAL_FLUSH: f64 = 1e-15;

/// Flush tiny magnitudes to exactly zero so decaying recursive states never
/// enter the subnormal range.
#[inline]
pub fn flush_denormal(x: f64) -> f64 {
    if x.abs() < DENORMAL_FLUSH {
        0.0
    } else {
        x
    }
}

/// Coefficient `a` of the one-pole smoother `y[n] = a·y[n-1] + (1-a)·x[n]` whose
/// step response reaches `1 - 1/e` (≈ 63.2 %) after `time_ms` milliseconds:
/// `a = exp(-1 / (τ·fs))` with `τ = time_ms / 1000`. A time of 0 (or less) gives
/// `a = 0`, i.e. instantaneous.
#[inline]
pub fn one_pole_coef(time_ms: f64, sample_rate: f64) -> f64 {
    let samples = time_ms * 0.001 * sample_rate;
    if samples <= 0.0 {
        0.0
    } else {
        (-1.0 / samples).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn db_roundtrip() {
        for db in [-120.0, -60.0, -6.0, 0.0, 6.0, 40.0] {
            assert!((lin_to_db(db_to_lin(db)) - db).abs() < 1e-9);
        }
        assert!(lin_to_db(0.0).is_finite());
    }

    #[test]
    fn one_pole_time_constant() {
        let fs = 48_000.0;
        let a = one_pole_coef(10.0, fs);
        let n = (0.010 * fs) as i32;
        let y = 1.0 - a.powi(n);
        assert!((y - (1.0 - (-1.0f64).exp())).abs() < 1e-3);
        assert_eq!(one_pole_coef(0.0, fs), 0.0);
    }
}
