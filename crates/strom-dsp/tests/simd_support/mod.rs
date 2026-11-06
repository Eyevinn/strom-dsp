//! SIMD levels for the cross-level tests: the scalar fallback plus every
//! vector level the running CPU supports, and the comparison rule between
//! them.
//!
//! Every lane of a vector kernel executes the fallback's operations in the
//! same order. The only permitted difference is the FMA policy
//! (`strom_dsp::biquad::uses_fma`): levels with the same policy are
//! bit-identical; levels with different policies (on x86-64: AVX2/AVX-512
//! fuse, the fallback and SSE do not) may differ by the rounding of the
//! product-sums, within `|a − b| ≤ 1e-5·max(|a|, |b|) + 1e-9` per sample.
//!
//! `Level::fallback()` needs fearless_simd's `force_support_fallback`
//! feature, which the dev-dependency enables for tests only.

#![allow(dead_code)]

use fearless_simd::Simd;
use strom_dsp::biquad::uses_fma;
use strom_dsp::simd::{detect, Level};

/// The fallback (first) plus every vector level the running CPU supports.
pub fn levels() -> Vec<(&'static str, Level)> {
    let d = detect();
    #[allow(unused_mut)]
    let mut v = vec![("fallback", Level::fallback())];
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = d.as_neon() {
        v.push(("neon", t.level()));
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if let Some(t) = d.as_sse2() {
            v.push(("sse2", t.level()));
        }
        if let Some(t) = d.as_sse4_2() {
            v.push(("sse4.2", t.level()));
        }
        if let Some(t) = d.as_avx2() {
            v.push(("avx2", t.level()));
        }
        if let Some(t) = d.as_avx512() {
            v.push(("avx512", t.level()));
        }
    }
    let _ = d;
    v
}

/// The vector levels only (every level but the fallback).
pub fn vector_levels() -> Vec<(&'static str, Level)> {
    levels().into_iter().skip(1).collect()
}

/// Every output sample of `got` (run at `level`) equals `reference` (run at
/// `ref_level`) bit for bit when the two levels share an FMA policy, and is
/// within the bound in the module docs otherwise. Returns the worst ratio of
/// difference to bound (0 when bit-identical).
pub fn compare(ctx: &str, level: Level, got: &[f32], ref_level: Level, reference: &[f32]) -> f64 {
    assert_eq!(got.len(), reference.len(), "{ctx}");
    let exact = uses_fma(level) == uses_fma(ref_level);
    let mut worst = 0.0f64;
    for (i, (&a, &b)) in got.iter().zip(reference).enumerate() {
        assert_eq!(a.is_finite(), b.is_finite(), "{ctx}: finiteness at {i}");
        if exact {
            assert_eq!(a.to_bits(), b.to_bits(), "{ctx}: sample {i}: {a} vs {b}");
        } else {
            let (a, b) = (a as f64, b as f64);
            let d = (a - b).abs();
            let bound = 1e-5 * a.abs().max(b.abs()) + 1e-9;
            assert!(d <= bound, "{ctx}: sample {i}: {a} vs {b}");
            worst = worst.max(d / bound);
        }
    }
    worst
}

/// Check every level's output against the fallback's and, for levels that
/// fuse, against the first fusing level's bit for bit (so e.g. AVX2 and
/// AVX-512 must agree exactly). `run(level)` returns all output samples.
pub fn check_all_levels(what: &str, mut run: impl FnMut(Level) -> Vec<f32>) {
    let mut runs: Vec<(&str, Level, Vec<f32>)> = Vec::new();
    for (name, level) in levels() {
        runs.push((name, level, run(level)));
    }
    let (_, fb_level, fb) = &runs[0];
    for (name, level, y) in &runs[1..] {
        compare(
            &format!("{what} {name} vs fallback"),
            *level,
            y,
            *fb_level,
            fb,
        );
    }
    if let Some((_, fused_level, fused)) = runs.iter().find(|r| uses_fma(r.1)) {
        for (name, level, y) in runs.iter().filter(|r| uses_fma(r.1)) {
            compare(
                &format!("{what} {name} vs first fused level"),
                *level,
                y,
                *fused_level,
                fused,
            );
        }
    }
}
