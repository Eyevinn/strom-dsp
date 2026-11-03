//! Linked-channel helpers on interleaved blocks: per-frame peak detection and
//! per-frame gain application.
//!
//! These are the data-parallel parts of a linked dynamics processor. They are
//! written as plain slice loops with mono/stereo specialisations so the
//! compiler can vectorise them; they are the place to swap in explicit SIMD.
//! The serial parts (envelopes, smoothers) run between them, one value per
//! frame, on stack arrays of at most [`BLOCK_FRAMES`] frames.

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
