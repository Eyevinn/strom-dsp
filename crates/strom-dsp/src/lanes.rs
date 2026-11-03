//! Lane helpers for "one channel per SIMD lane" processing, on top of the
//! explicit vector types of `fearless_simd` (`f64x2`, `f64x4`, `f64x8`).
//!
//! Recursive filters cannot be vectorised along time, but the channels of a
//! frame are independent, so they can be processed as SIMD lanes. Every helper
//! here is `#[inline(always)]`, so it inherits the target features of the
//! kernel it is inlined into (see `Simd::vectorize` in `fearless_simd`).
//!
//! # FMA policy
//!
//! [`madd`] computes `a·b + c` either fused (one rounding) or as a separate
//! multiply and add (two roundings), chosen per SIMD level by [`fma_mode`]:
//! fused on levels with a hardware vector FMA (NEON, AVX2, AVX-512), unfused
//! on SSE2/SSE4.2. The scalar fallback level follows the platform's baseline
//! vector level: fused (per lane with `f64::mul_add`, a single instruction
//! there) on aarch64 and on x86 builds with the `fma` target feature, unfused
//! otherwise. So on a given build the fallback is bit-identical to the
//! baseline vector path.

use fearless_simd::{Level, Select, Simd, SimdFloat};

use crate::util::DENORMAL_FLUSH;

/// How [`madd`] evaluates `a·b + c` at a given level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fma {
    /// Multiply, round, add, round.
    Unfused,
    /// One fused vector instruction (NEON, AVX2, AVX-512).
    Vector,
    /// Fused per lane with `f64::mul_add` (fallback level on a platform whose
    /// baseline has a hardware FMA, where that is one instruction).
    Scalar,
}

/// The FMA mode of the kernels at `level` (see the module documentation).
pub(crate) fn fma_mode(level: Level) -> Fma {
    #[cfg(target_arch = "aarch64")]
    if level.as_neon().is_some() {
        return Fma::Vector;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // `as_avx2` is also `Some` for AVX-512.
        if level.as_avx2().is_some() {
            return Fma::Vector;
        }
        if level.as_sse2().is_some() {
            return Fma::Unfused;
        }
    }
    let _ = level;
    // Scalar fallback: match the baseline vector level of this build.
    if cfg!(any(target_arch = "aarch64", target_feature = "fma")) {
        Fma::Scalar
    } else {
        Fma::Unfused
    }
}

/// Whether the filter kernels fuse their multiply-adds (one rounding) at
/// `level`: `true` for NEON, AVX2 and AVX-512, `false` for SSE2/SSE4.2, and for
/// the scalar fallback the same as the platform's baseline vector level
/// (`true` on aarch64, otherwise only if the build enables the `fma` target
/// feature).
///
/// Two levels with the same answer produce bit-identical filter output; levels
/// with different answers differ only in the rounding of each product-sum.
pub fn uses_fma(level: Level) -> bool {
    fma_mode(level) != Fma::Unfused
}

/// `a·b + c` per [`Fma`] mode. The mode is a constant after inlining, so the
/// match disappears.
#[inline(always)]
pub(crate) fn madd<S: Simd, V: SimdFloat<S, Element = f64>>(fma: Fma, a: V, b: V, c: V) -> V {
    match fma {
        Fma::Vector => a.mul_add(b, c),
        Fma::Scalar => V::from_fn(a.token(), |i| a[i].mul_add(b[i], c[i])),
        Fma::Unfused => a * b + c,
    }
}

/// Read `src.len() ≤ V::LEN` consecutive `f32` samples into the low lanes;
/// the remaining lanes are zero.
#[inline(always)]
pub(crate) fn load_f32<S: Simd, V: SimdFloat<S, Element = f64>>(simd: S, src: &[f32]) -> V {
    V::from_fn(simd, |i| src.get(i).map_or(0.0, |&v| v as f64))
}

/// Write the low `dst.len() ≤ V::LEN` lanes as `f32`, clamped to the finite
/// `f32` range.
#[inline(always)]
pub(crate) fn store_f32<S: Simd, V: SimdFloat<S, Element = f64>>(v: V, dst: &mut [f32]) {
    const MAX: f64 = f32::MAX as f64;
    let v = v.max(-MAX).min(MAX);
    for (d, s) in dst.iter_mut().zip(v.as_slice()) {
        *d = *s as f32;
    }
}

/// Read `src.len() ≤ V::LEN` consecutive `f64` values into the low lanes;
/// the remaining lanes are zero.
#[inline(always)]
pub(crate) fn load<S: Simd, V: SimdFloat<S, Element = f64>>(simd: S, src: &[f64]) -> V {
    V::from_fn(simd, |i| src.get(i).copied().unwrap_or(0.0))
}

/// Write the low `dst.len() ≤ V::LEN` lanes.
#[inline(always)]
pub(crate) fn store<S: Simd, V: SimdFloat<S, Element = f64>>(v: V, dst: &mut [f64]) {
    for (d, s) in dst.iter_mut().zip(v.as_slice()) {
        *d = *s;
    }
}

/// Replace magnitudes below [`DENORMAL_FLUSH`] with exactly zero (branch-free select).
#[inline(always)]
pub(crate) fn flush_denormal<S: Simd, V: SimdFloat<S, Element = f64>>(simd: S, v: V) -> V {
    v.abs()
        .simd_lt(DENORMAL_FLUSH)
        .select(V::splat(simd, 0.0), v)
}

/// Linear blend `a + (b − a)·g` with a scalar weight (one [`madd`]).
#[inline(always)]
pub(crate) fn blend<S: Simd, V: SimdFloat<S, Element = f64>>(
    simd: S,
    fma: Fma,
    a: V,
    b: V,
    g: f64,
) -> V {
    madd(fma, b - a, V::splat(simd, g), a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fearless_simd::{dispatch, f64x2, f64x4, SimdBase};

    #[inline(always)]
    fn check<S: Simd>(simd: S) {
        let fma = fma_mode(simd.level());
        let a = f64x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
        let b = f64x4::from_slice(simd, &[0.5, -1.0, 4.0, 0.0]);
        assert_eq!(*blend(simd, fma, a, b, 0.5), [0.75, 0.5, 3.5, 2.0]);
        assert_eq!(*madd(fma, a, b, a), [1.5, 0.0, 15.0, 4.0]);
        let v = flush_denormal(simd, f64x4::from_slice(simd, &[1e-16, -1e-16, 1e-14, 0.25]));
        assert_eq!(*v, [0.0, 0.0, 1e-14, 0.25]);
        let mut out = [0.0f32; 3];
        store_f32(
            f64x4::from_slice(simd, &[1e300, -1e300, 0.5, 7.0]),
            &mut out,
        );
        assert_eq!(out, [f32::MAX, -f32::MAX, 0.5]);
        let v: f64x2<S> = load_f32(simd, &[0.5]);
        assert_eq!(*v, [0.5, 0.0]);
        let v: f64x4<S> = load(simd, &[1.0, 2.0, 3.0]);
        assert_eq!(*v, [1.0, 2.0, 3.0, 0.0]);
        let mut out = [0.0f64; 3];
        store(v, &mut out);
        assert_eq!(out, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn helpers_are_elementwise() {
        dispatch!(crate::simd::detect(), simd => check(simd));
        dispatch!(Level::fallback(), simd => check(simd));
    }

    /// `(1 + 2⁻³⁰)² − 1` keeps the `2⁻⁶⁰` term only when fused.
    #[inline(always)]
    fn square_minus_one<S: Simd>(simd: S) -> f64 {
        let a = f64x2::splat(simd, 1.0 + 2f64.powi(-30));
        madd(fma_mode(simd.level()), a, a, f64x2::splat(simd, -1.0))[0]
    }

    /// The FMA policy is what [`uses_fma`] says, for the detected level and
    /// for the fallback.
    #[test]
    fn fma_policy_is_observable() {
        let e = 2f64.powi(-30);
        for level in [crate::simd::detect(), Level::fallback()] {
            let r = dispatch!(level, simd => square_minus_one(simd));
            let expect = if uses_fma(level) {
                2.0 * e + e * e
            } else {
                2.0 * e
            };
            assert_eq!(r, expect, "{level:?}");
        }
        assert_eq!(
            uses_fma(Level::fallback()),
            cfg!(any(target_arch = "aarch64", target_feature = "fma"))
        );
    }
}
