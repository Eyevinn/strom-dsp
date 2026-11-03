//! A tiny fixed-width lane type over `[f64; L]` for "one channel per lane"
//! processing.
//!
//! Recursive filters cannot be vectorised along time, but the channels of a
//! frame are independent, so they can be processed as SIMD lanes. Every
//! operation here is a plain element-wise loop over a fixed-size array, which
//! LLVM turns into vector instructions on stable Rust without `unsafe`. The
//! type is deliberately minimal so it can later be swapped for an explicit SIMD
//! type without touching the filter code.

use core::ops::{Add, Mul, Sub};

use crate::util::DENORMAL_FLUSH;

/// `L` values of `f64`, one per channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Lanes<const L: usize>(pub [f64; L]);

impl<const L: usize> Default for Lanes<L> {
    #[inline(always)]
    fn default() -> Self {
        Self([0.0; L])
    }
}

impl<const L: usize> Lanes<L> {
    /// Read `L` consecutive `f32` samples (`src.len() ≥ L`).
    #[inline(always)]
    pub fn load_f32(src: &[f32]) -> Self {
        let src: &[f32; L] = src[..L].try_into().expect("slice of length L");
        Self(core::array::from_fn(|i| src[i] as f64))
    }

    /// Write the lanes as `f32`, clamped to the finite `f32` range
    /// (`dst.len() ≥ L`).
    #[inline(always)]
    pub fn store_f32(self, dst: &mut [f32]) {
        const MAX: f64 = f32::MAX as f64;
        let dst: &mut [f32; L] = (&mut dst[..L]).try_into().expect("slice of length L");
        for (d, v) in dst.iter_mut().zip(self.0) {
            *d = v.clamp(-MAX, MAX) as f32;
        }
    }

    /// Read `L` consecutive `f64` values (`src.len() ≥ L`).
    #[inline(always)]
    pub fn load(src: &[f64]) -> Self {
        Self(src[..L].try_into().expect("slice of length L"))
    }

    /// Write `L` consecutive `f64` values (`dst.len() ≥ L`).
    #[inline(always)]
    pub fn store(self, dst: &mut [f64]) {
        dst[..L].copy_from_slice(&self.0);
    }

    /// Replace magnitudes below [`DENORMAL_FLUSH`] with exactly zero (branch-free select).
    #[inline(always)]
    pub fn flush_denormal(self) -> Self {
        Self(
            self.0
                .map(|v| if v.abs() < DENORMAL_FLUSH { 0.0 } else { v }),
        )
    }

    /// `self·k + acc`, as a fused multiply-add where the target has a hardware
    /// FMA (aarch64 always; x86 only when built with the `fma` target
    /// feature). Elsewhere it is a separate multiply and add, because
    /// `f64::mul_add` would fall back to a slow software routine.
    #[inline(always)]
    pub fn mul_add(self, k: f64, acc: Self) -> Self {
        #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
        {
            Self(core::array::from_fn(|i| self.0[i].mul_add(k, acc.0[i])))
        }
        #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
        {
            self * k + acc
        }
    }

    /// Linear blend `a + (b − a)·g` with a scalar weight.
    #[inline(always)]
    pub fn blend(a: Self, b: Self, g: f64) -> Self {
        a + (b - a) * g
    }
}

impl<const L: usize> Add for Lanes<L> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i] + o.0[i]))
    }
}

impl<const L: usize> Sub for Lanes<L> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i] - o.0[i]))
    }
}

/// Multiply every lane by a scalar (filter coefficients are shared by all channels).
impl<const L: usize> Mul<f64> for Lanes<L> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, k: f64) -> Self {
        Self(self.0.map(|v| v * k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_is_elementwise() {
        let a = Lanes([1.0, 2.0, 3.0]);
        let b = Lanes([0.5, -1.0, 4.0]);
        assert_eq!((a + b).0, [1.5, 1.0, 7.0]);
        assert_eq!((a - b).0, [0.5, 3.0, -1.0]);
        assert_eq!((a * 2.0).0, [2.0, 4.0, 6.0]);
        assert_eq!(Lanes::blend(a, b, 0.5).0, [0.75, 0.5, 3.5]);
        assert_eq!(a.mul_add(2.0, b).0, [2.5, 3.0, 10.0]);
    }

    #[test]
    fn flush_and_f32_io() {
        let v = Lanes([1e-16, -1e-16, 1e-14, 0.25]).flush_denormal();
        assert_eq!(v.0, [0.0, 0.0, 1e-14, 0.25]);
        let mut out = [0.0f32; 3];
        Lanes([1e300, -1e300, 0.5]).store_f32(&mut out);
        assert_eq!(out, [f32::MAX, -f32::MAX, 0.5]);
        assert_eq!(Lanes::<2>::load_f32(&[0.5, -2.0, 9.0]).0, [0.5, -2.0]);
    }
}
