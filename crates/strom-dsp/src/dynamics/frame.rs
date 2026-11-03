//! Linked-channel helpers on interleaved blocks: per-frame peak detection and
//! per-frame gain application.
//!
//! These are the data-parallel parts of a linked dynamics processor. The
//! serial parts (envelopes, smoothers) run between them, one value per frame,
//! on stack arrays of at most [`BLOCK_FRAMES`] frames.
//!
//! Two flavours: plain slice loops ([`frame_peaks`], [`apply_gains`]) and
//! explicit SIMD versions generic over a `fearless_simd` instruction set
//! ([`frame_peaks_simd`], [`apply_gains_simd`]), which the processors call
//! from inside one `dispatch!` per `process` call.

use fearless_simd::prelude::*;

/// Frames per processing block used by the dynamics processors (stack arrays
/// of this size hold per-frame peaks and gains).
pub const BLOCK_FRAMES: usize = 64;

/// Largest absolute sample of one frame (NaN samples are ignored).
#[inline]
pub fn frame_peak(frame: &[f32]) -> f32 {
    frame.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
}

/// Write the linked peak `max_ch |x_ch|` of each interleaved frame of
/// `samples` into `out` (one value per frame). `samples.len()` must equal
/// `out.len() · channels`; extra samples or outputs are ignored.
#[inline]
pub fn frame_peaks(samples: &[f32], channels: usize, out: &mut [f32]) {
    match channels {
        0 => {}
        1 => {
            for (o, &x) in out.iter_mut().zip(samples) {
                *o = x.abs().max(0.0);
            }
        }
        2 => {
            for (o, f) in out.iter_mut().zip(samples.chunks_exact(2)) {
                *o = f[0].abs().max(f[1].abs()).max(0.0);
            }
        }
        _ => {
            for (o, f) in out.iter_mut().zip(samples.chunks_exact(channels)) {
                *o = frame_peak(f);
            }
        }
    }
}

/// Multiply every sample of interleaved frame `i` by `gains[i]`.
/// `samples.len()` must equal `gains.len() · channels`; extra values are ignored.
#[inline]
pub fn apply_gains(samples: &mut [f32], channels: usize, gains: &[f32]) {
    match channels {
        0 => {}
        1 => {
            for (x, &g) in samples.iter_mut().zip(gains) {
                *x *= g;
            }
        }
        2 => {
            for (f, &g) in samples.chunks_exact_mut(2).zip(gains) {
                f[0] *= g;
                f[1] *= g;
            }
        }
        _ => {
            for (f, &g) in samples.chunks_exact_mut(channels).zip(gains) {
                for x in f {
                    *x *= g;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Explicit SIMD versions (fearless_simd, generic over the instruction set).
//
// Every operation used here is exact in IEEE arithmetic (abs, compare, select,
// multiply, lane shuffles, an exact horizontal max of non-NaN lanes), so the
// result is bit-identical for every SIMD level, including the scalar
// fallback, and equal to the scalar functions above.

/// `max(m, a)` that ignores a NaN `a` (comparisons with NaN are false).
/// Deterministic on every backend, unlike a hardware `max` with NaN operands.
#[inline(always)]
fn max_ignore_nan<S: Simd>(m: S::f32s, a: S::f32s) -> S::f32s {
    a.simd_gt(m).select(a, m)
}

/// Scalar twin of [`max_ignore_nan`].
#[inline(always)]
fn max_ignore_nan_scalar(m: f32, a: f32) -> f32 {
    if a > m {
        a
    } else {
        m
    }
}

/// SIMD version of [`frame_peaks`] for the instruction set `simd`: the linked
/// peak `max_ch |x_ch|` of each interleaved frame (NaN samples ignored, so a
/// peak is never NaN). Writes `min(out.len(), samples.len() / channels)` values.
///
/// Mono and stereo run at the native vector width (stereo is de-interleaved in
/// registers); a channel count of at least one vector reduces each frame
/// across vectors; other counts use a scalar loop. The result is identical for
/// every SIMD level. It is `#[inline(always)]`: call it from code running
/// under `fearless_simd::dispatch!` (or `Simd::vectorize`).
#[inline(always)]
pub fn frame_peaks_simd<S: Simd>(simd: S, samples: &[f32], channels: usize, out: &mut [f32]) {
    let n = S::f32s::LEN;
    let frames = out.len().min(samples.len() / channels.max(1));
    let (samples, out) = (&samples[..frames * channels], &mut out[..frames]);
    let zero = S::f32s::splat(simd, 0.0);
    match channels {
        0 => {}
        1 => {
            let mut src = samples.chunks_exact(n);
            let mut dst = out.chunks_exact_mut(n);
            for (s, o) in (&mut src).zip(&mut dst) {
                let a = S::f32s::from_slice(simd, s).abs();
                max_ignore_nan::<S>(zero, a).store_slice(o);
            }
            for (o, &x) in dst.into_remainder().iter_mut().zip(src.remainder()) {
                *o = max_ignore_nan_scalar(0.0, x.abs());
            }
        }
        2 => {
            let mut src = samples.chunks_exact(2 * n);
            let mut dst = out.chunks_exact_mut(n);
            for (s, o) in (&mut src).zip(&mut dst) {
                let a = S::f32s::from_slice(simd, &s[..n]).abs();
                let b = S::f32s::from_slice(simd, &s[n..]).abs();
                let (left, right) = a.deinterleave(b);
                max_ignore_nan::<S>(max_ignore_nan::<S>(zero, left), right).store_slice(o);
            }
            for (o, f) in dst
                .into_remainder()
                .iter_mut()
                .zip(src.remainder().chunks_exact(2))
            {
                *o = max_ignore_nan_scalar(max_ignore_nan_scalar(0.0, f[0].abs()), f[1].abs());
            }
        }
        ch if ch >= n => {
            for (o, f) in out.iter_mut().zip(samples.chunks_exact(ch)) {
                let mut vecs = f.chunks_exact(n);
                let mut m = zero;
                for v in &mut vecs {
                    m = max_ignore_nan::<S>(m, S::f32s::from_slice(simd, v).abs());
                }
                // No lane is NaN here, so the horizontal max is exact.
                let mut p = m.reduce_max();
                for &x in vecs.remainder() {
                    p = max_ignore_nan_scalar(p, x.abs());
                }
                *o = p;
            }
        }
        ch => {
            for (o, f) in out.iter_mut().zip(samples.chunks_exact(ch)) {
                *o = f
                    .iter()
                    .fold(0.0, |m, &x| max_ignore_nan_scalar(m, x.abs()));
            }
        }
    }
}

/// SIMD version of [`apply_gains`] for the instruction set `simd`: multiply
/// every sample of interleaved frame `i` by `gains[i]`, for
/// `min(gains.len(), samples.len() / channels)` frames.
///
/// Mono and stereo run at the native vector width (stereo gains are
/// duplicated with an in-register zip); a channel count of at least one vector
/// splats each frame's gain; other counts use a scalar loop. Bit-identical for
/// every SIMD level (one IEEE multiply per sample). `#[inline(always)]`, like
/// [`frame_peaks_simd`].
#[inline(always)]
pub fn apply_gains_simd<S: Simd>(simd: S, samples: &mut [f32], channels: usize, gains: &[f32]) {
    let n = S::f32s::LEN;
    let frames = gains.len().min(samples.len() / channels.max(1));
    let (samples, gains) = (&mut samples[..frames * channels], &gains[..frames]);
    match channels {
        0 => {}
        1 => {
            let mut dst = samples.chunks_exact_mut(n);
            let mut src = gains.chunks_exact(n);
            for (x, g) in (&mut dst).zip(&mut src) {
                (S::f32s::from_slice(simd, x) * S::f32s::from_slice(simd, g)).store_slice(x);
            }
            for (x, &g) in dst.into_remainder().iter_mut().zip(src.remainder()) {
                *x *= g;
            }
        }
        2 => {
            let mut dst = samples.chunks_exact_mut(2 * n);
            let mut src = gains.chunks_exact(n);
            for (x, g) in (&mut dst).zip(&mut src) {
                let g = S::f32s::from_slice(simd, g);
                let (g_lo, g_hi) = g.interleave(g);
                let (lo, hi) = x.split_at_mut(n);
                (S::f32s::from_slice(simd, lo) * g_lo).store_slice(lo);
                (S::f32s::from_slice(simd, hi) * g_hi).store_slice(hi);
            }
            for (f, &g) in dst
                .into_remainder()
                .chunks_exact_mut(2)
                .zip(src.remainder())
            {
                f[0] *= g;
                f[1] *= g;
            }
        }
        ch if ch >= n => {
            for (f, &g) in samples.chunks_exact_mut(ch).zip(gains) {
                let gv = S::f32s::splat(simd, g);
                let mut vecs = f.chunks_exact_mut(n);
                for v in &mut vecs {
                    (S::f32s::from_slice(simd, v) * gv).store_slice(v);
                }
                for x in vecs.into_remainder() {
                    *x *= g;
                }
            }
        }
        ch => {
            for (f, &g) in samples.chunks_exact_mut(ch).zip(gains) {
                for x in f {
                    *x *= g;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peaks_and_gains_for_all_channel_counts() {
        for ch in [1usize, 2, 3, 8] {
            let frames = 5;
            let s: Vec<f32> = (0..frames * ch)
                .map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) })
                .collect();
            let mut p = vec![0.0; frames];
            frame_peaks(&s, ch, &mut p);
            for (i, &v) in p.iter().enumerate() {
                assert_eq!(v, ((i + 1) * ch - 1) as f32, "ch={ch}");
            }
            let mut y = s.clone();
            let g: Vec<f32> = (0..frames).map(|i| i as f32 * 0.5).collect();
            apply_gains(&mut y, ch, &g);
            for (i, (&a, &b)) in y.iter().zip(&s).enumerate() {
                assert_eq!(a, b * g[i / ch]);
            }
        }
    }

    #[test]
    fn peak_ignores_nan() {
        let mut p = [1.0f32; 2];
        frame_peaks(&[f32::NAN, f32::NAN], 1, &mut p);
        assert_eq!(p, [0.0, 0.0]);
        frame_peaks(&[f32::NAN, f32::NAN], 2, &mut p[..1]);
        assert_eq!(p[0], 0.0);
        assert_eq!(frame_peak(&[f32::NAN, -0.5]), 0.5);
    }
}
