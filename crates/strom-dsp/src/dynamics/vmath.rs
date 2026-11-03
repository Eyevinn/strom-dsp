//! Vectorised dB conversions for the compressor's block path, written with
//! `fearless_simd` and generic over the instruction set.
//!
//! Both functions work on native-width `f64` vectors and use only operations
//! that are exactly specified by IEEE 754 (add, subtract, multiply, divide,
//! compare/select, rounding to an integer, bit manipulation). No fused
//! multiply-add is used, because whether `mul_add` fuses differs between
//! backends. Every lane is therefore computed bit-identically on every SIMD
//! level, including the scalar fallback, which runs the same code.
//!
//! # Algorithms (textbook range reduction + series)
//!
//! - [`lin_to_db`]: `x = 2^e · m` is split by its IEEE bit fields, with `m`
//!   moved into `[√½, √2)` (`e` adjusted). Then
//!   `ln m = 2·atanh(s) = 2·(s + s³/3 + s⁵/5 + …)`, `s = (m − 1)/(m + 1)`,
//!   `|s| ≤ 0.1716`; the series is summed through `s¹⁵/15` (Abramowitz &
//!   Stegun, *Handbook of Mathematical Functions*, 4.1.27). The first omitted
//!   term is below 1e-14 relative, so the result is within about 1e-12 dB of
//!   `20·log10(x)`.
//! - [`db_to_gain`]: `10^(−y/20) = 2^t`, `t = −y·log2(10)/20`, split as
//!   `t = n + f` with `n = round(t)`, `|f| ≤ ½`. `2^f = e^u`, `u = f·ln 2`,
//!   `|u| ≤ 0.347`, by its Taylor series through `u¹¹/11!` (remainder below
//!   1e-14 relative), and `2^n` is built directly in the exponent field.
//!
//! Both are tested against `f64::log10` / `f64::exp` to well below the 0.001 dB
//! the compressor needs (unit tests below: < 1e-9 dB).

use fearless_simd::prelude::*;

use crate::util::MIN_LIN;

/// Largest input of [`lin_to_db`] (larger values and +∞ are clamped to it);
/// `f32::MAX`, so every finite `f32` peak is represented exactly.
pub(crate) const MAX_LIN: f64 = f32::MAX as f64;

/// `20·log10(2)`: dB per octave of amplitude.
const DB_PER_OCTAVE: f64 = 20.0 * std::f64::consts::LOG10_2;
/// `20/ln(10)`: dB per neper of amplitude.
const DB_PER_NEPER: f64 = 20.0 / std::f64::consts::LN_10;
/// `−log2(10)/20`: `10^(−y/20) = 2^(y·NEG_OCTAVES_PER_DB)`.
const NEG_OCTAVES_PER_DB: f64 = -std::f64::consts::LOG2_10 / 20.0;
/// Most negative exponent used by [`db_to_gain`] (`2^−1000`, far below the
/// smallest `f32`), so `2^n` stays a normal `f64`.
const MIN_EXP2: f64 = -1000.0;
/// `2^52`: adding it to a small non-negative integer-valued `f64` puts that
/// integer in the low mantissa bits.
const TWO_POW_52: f64 = 4_503_599_627_370_496.0;
const EXP_MASK_SHIFT: u32 = 52;
const MANTISSA_MASK: u64 = (1 << 52) - 1;
const ONE_BITS: u64 = 0x3FF0_0000_0000_0000;
/// Bits of `2^52` (`0x4330…`), the integer-in-mantissa trick's exponent.
const TWO_POW_52_BITS: u64 = 0x4330_0000_0000_0000;

/// `20·log10(x)` per lane, with `x` clamped to `[MIN_LIN, MAX_LIN]` (NaN
/// becomes `MIN_LIN`), so the result is always finite: −200 dB … ≈ 770.6 dB.
/// Error below 1e-12 dB (see the module docs).
#[inline(always)]
pub(crate) fn lin_to_db<S: Simd>(simd: S, x: S::f64s) -> S::f64s {
    let lo = S::f64s::splat(simd, MIN_LIN);
    let hi = S::f64s::splat(simd, MAX_LIN);
    // Compare-select clamp: NaN fails `x > lo` and becomes MIN_LIN.
    let x = x.simd_gt(lo).select(x, lo);
    let x = x.simd_lt(hi).select(x, hi);

    // x is a positive normal f64: split exponent and mantissa.
    let bits = x.to_bits();
    let biased_exp = bits >> EXP_MASK_SHIFT;
    let m = S::f64s::from_bits((bits & MANTISSA_MASK) | ONE_BITS); // [1, 2)
                                                                   // Exact integer → f64: put the exponent in the mantissa of 2^52.
    let e = S::f64s::from_bits(biased_exp | TWO_POW_52_BITS) - (TWO_POW_52 + 1023.0);

    // Centre the mantissa: m ∈ [√½, √2), so |s| ≤ 0.1716.
    let big = m.simd_ge(S::f64s::splat(simd, std::f64::consts::SQRT_2));
    let m = big.select(m * 0.5, m);
    let e = big.select(e + 1.0, e);

    let s = (m - 1.0) / (m + 1.0);
    let z = s * s;
    // atanh series: 1 + z/3 + z²/5 + … + z⁷/15 (Horner, no FMA).
    let mut p = S::f64s::splat(simd, 1.0 / 15.0);
    for c in [
        1.0 / 13.0,
        1.0 / 11.0,
        1.0 / 9.0,
        1.0 / 7.0,
        1.0 / 5.0,
        1.0 / 3.0,
        1.0,
    ] {
        p = p * z + c;
    }
    let ln_m = s * p * 2.0;
    e * DB_PER_OCTAVE + ln_m * DB_PER_NEPER
}

/// `10^(−y/20)` per lane, i.e. the linear gain of a reduction of `y` dB, for
/// `y ≥ 0` (finite). Results below `2^−1000` are returned as `2^−1000`, which
/// still rounds to 0 in `f32`. Relative error below 1e-13 (see the module docs).
#[inline(always)]
pub(crate) fn db_to_gain<S: Simd>(simd: S, y: S::f64s) -> S::f64s {
    let t = y * NEG_OCTAVES_PER_DB;
    let floor = S::f64s::splat(simd, MIN_EXP2);
    let t = t.simd_gt(floor).select(t, floor);
    let n = t.round_ties_even();
    let u = (t - n) * std::f64::consts::LN_2;

    // e^u = Σ u^k/k!, k = 0 … 11 (Horner, no FMA).
    let mut p = S::f64s::splat(simd, 1.0 / 39_916_800.0);
    for c in [
        1.0 / 3_628_800.0,
        1.0 / 362_880.0,
        1.0 / 40_320.0,
        1.0 / 5_040.0,
        1.0 / 720.0,
        1.0 / 120.0,
        1.0 / 24.0,
        1.0 / 6.0,
        0.5,
        1.0,
        1.0,
    ] {
        p = p * u + c;
    }

    // 2^n for integer n ∈ [−1000, 0]: n + 1023 lands in the low mantissa
    // bits of (n + 1023 + 2^52); shift it into the exponent field.
    let biased = (n + (TWO_POW_52 + 1023.0)).to_bits() << EXP_MASK_SHIFT;
    p * S::f64s::from_bits(biased)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fearless_simd::{dispatch, Level};

    /// Evaluate `f` lane by lane (vectors of `LEN` inputs, last one padded).
    #[inline(always)]
    fn eval<S: Simd>(simd: S, xs: &[f64], f: fn(S, S::f64s) -> S::f64s) -> Vec<f64> {
        let n = S::f64s::LEN;
        let mut out = Vec::with_capacity(xs.len());
        for c in xs.chunks(n) {
            let v = S::f64s::from_fn(simd, |i| c.get(i).copied().unwrap_or(1.0));
            out.extend_from_slice(&f(simd, v).as_slice()[..c.len()]);
        }
        out
    }

    fn db_all(level: Level, xs: &[f64]) -> Vec<f64> {
        dispatch!(level, simd => eval(simd, xs, lin_to_db))
    }

    fn gain_all(level: Level, ys: &[f64]) -> Vec<f64> {
        dispatch!(level, simd => eval(simd, ys, db_to_gain))
    }

    /// Log-spaced test points plus edge cases.
    fn amplitudes() -> Vec<f64> {
        let mut xs: Vec<f64> = (0..=40_000)
            .map(|i| 10f64.powf(-10.0 + i as f64 * 48.0 / 40_000.0))
            .collect();
        let mut seed = 0x1234_5678_9abc_def0u64;
        for _ in 0..20_000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            xs.push(f32::from_bits((seed >> 32) as u32 & 0x7fff_ffff) as f64);
        }
        xs.extend([
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::SQRT_2,
            1.0,
            2.0,
            0.5,
            1.0 - 1e-16,
            MIN_LIN,
            MAX_LIN,
        ]);
        xs.retain(|x| x.is_finite() && *x >= MIN_LIN && *x <= MAX_LIN);
        xs
    }

    #[test]
    fn lin_to_db_matches_log10() {
        let xs = amplitudes();
        let want: Vec<f64> = xs.iter().map(|x| 20.0 * x.log10()).collect();
        for level in [Level::new(), Level::fallback()] {
            let got = db_all(level, &xs);
            let worst = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs())
                .fold(0.0, f64::max);
            eprintln!("max error {worst:e} dB at {level:?}");
            assert!(worst < 1e-9, "max error {worst:e} dB at {level:?}");
        }
    }

    #[test]
    fn lin_to_db_clamps_and_handles_specials() {
        let xs = [0.0, -1.0, 1e-300, f64::NAN, f64::INFINITY, 1e300];
        let got = db_all(Level::new(), &xs);
        let floor = 20.0 * MIN_LIN.log10();
        let top = 20.0 * MAX_LIN.log10();
        for (g, w) in got.iter().zip([floor, floor, floor, floor, top, top]) {
            assert!((g - w).abs() < 1e-9, "{g} vs {w}");
        }
        assert_eq!(db_all(Level::new(), &[1.0]), [0.0]);
    }

    #[test]
    fn db_to_gain_matches_exp() {
        let ys: Vec<f64> = (0..=200_000).map(|i| i as f64 * 0.005).collect();
        for level in [Level::new(), Level::fallback()] {
            let got = gain_all(level, &ys);
            let mut worst = 0.0f64;
            for (g, y) in got.iter().zip(&ys) {
                let want = (-y * std::f64::consts::LN_10 / 20.0).exp();
                // Error expressed in dB: 20·log10(g/want).
                worst = worst.max((20.0 * (g / want).log10()).abs());
            }
            eprintln!("max error {worst:e} dB at {level:?}");
            assert!(worst < 1e-9, "max error {worst:e} dB at {level:?}");
        }
        assert_eq!(gain_all(Level::new(), &[0.0]), [1.0]);
        // Huge reductions stay finite, positive and round to 0 in f32.
        for g in gain_all(Level::new(), &[1e4, 1e6, 1e300]) {
            assert!(g > 0.0 && g.is_finite() && g as f32 == 0.0);
        }
    }

    #[test]
    fn every_level_is_bit_identical() {
        let xs = amplitudes();
        let ys: Vec<f64> = xs.iter().map(|x| x.abs().min(2000.0)).collect();
        let bits = |v: Vec<f64>| v.into_iter().map(f64::to_bits).collect::<Vec<_>>();
        assert_eq!(
            bits(db_all(Level::new(), &xs)),
            bits(db_all(Level::fallback(), &xs))
        );
        assert_eq!(
            bits(gain_all(Level::new(), &ys)),
            bits(gain_all(Level::fallback(), &ys))
        );
    }
}
