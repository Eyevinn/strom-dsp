//! Rate conversion between a key (sidechain) signal and the programme, for
//! detectors.
//!
//! # Decision: convert the detector's envelope, not the key audio
//!
//! The ducker needs the *level* of a key that may run at a different sample
//! rate (and channel count) than the programme. Two ways to get there:
//!
//! 1. Resample the key audio to the programme rate (with `rubato`), then run
//!    the detector at the programme rate.
//! 2. Run the detector (key filter, linked peak, RMS envelope) at the key's
//!    own rate, and convert only the resulting **envelope** to the programme
//!    rate.
//!
//! `strom-dsp` uses (2). Reasons:
//!
//! - **A detector does not need band-limited audio.** The key's level goes
//!   through a 10 ms one-pole RMS average, so the envelope's bandwidth is about
//!   16 Hz. Sampling it at a control rate and interpolating linearly is exact
//!   for any practical purpose. A band-limited audio resampler would spend its
//!   whole cost on a precision the detector throws away.
//! - **The key filter runs at the key's true rate**, so its corner frequency
//!   is designed for the rate the key really has, with no resampler in front.
//! - **No added latency or chunking constraints.** A sinc resampler has a
//!   group delay that would have to be compensated, and wants fixed input or
//!   output chunk sizes, while the element receives key buffers of arbitrary
//!   size and timing. Option (2) consumes whatever key frames cover the
//!   programme's time span, in one pass.
//! - **Allocation-free by construction.** The helpers here are plain integer
//!   and float arithmetic.
//!
//! `rubato` (already a dependency for this purpose) documents
//! `Resampler::process_into_buffer` as its allocation-free entry point for
//! real-time use, with buffers sized up front. It is not needed for (2), so the
//! ducker does not use it. If a later element needs the key **audio** at
//! another rate (not just its level), wrap `rubato` here behind an own type,
//! and verify it with the counting-allocator and RTSan tests like everything
//! else on the processing path.
//!
//! # What is here
//!
//! - [`split_point`]: maps a position in a span of programme frames to the
//!   matching position in the key frames that cover the same time span, so a
//!   processor can walk both in step (programme block by programme block).
//! - [`FrameRatio`]: counts how many key frames a run of programme frames
//!   spans, carrying the fraction from call to call (used for the "no key"
//!   path, where the detector decays over silence of the right duration).

/// Position in a span of `key_frames` key frames that corresponds to position
/// `pos` (`0 ..= prog_frames`) in a span of `prog_frames` programme frames
/// covering the same time: `⌊pos · key_frames / prog_frames⌋`.
///
/// `split_point(0, …) = 0` and `split_point(prog_frames, …) = key_frames`, and
/// it never decreases, so consecutive programme blocks get consecutive,
/// non-overlapping key ranges that together cover the whole key span. Returns
/// 0 when `prog_frames` is 0.
#[inline]
pub fn split_point(pos: usize, prog_frames: usize, key_frames: usize) -> usize {
    if prog_frames == 0 {
        return 0;
    }
    ((pos.min(prog_frames) as u128 * key_frames as u128) / prog_frames as u128) as usize
}

/// Converts frame counts at one rate into frame counts at another, carrying
/// the fractional remainder so that long runs never drift.
///
/// After any sequence of [`advance`](Self::advance) calls totalling `n` input
/// frames, the outputs total `⌊n · ratio⌋` (up to floating-point rounding of
/// the carried fraction).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameRatio {
    ratio: f64,
    frac: f64,
}

impl Default for FrameRatio {
    fn default() -> Self {
        Self::new(1.0, 1.0)
    }
}

impl FrameRatio {
    /// Ratio `to_rate / from_rate` (both in Hz, clamped to ≥ 1).
    pub fn new(from_rate: f64, to_rate: f64) -> Self {
        let from = if from_rate.is_finite() {
            from_rate.max(1.0)
        } else {
            1.0
        };
        let to = if to_rate.is_finite() {
            to_rate.max(1.0)
        } else {
            1.0
        };
        Self {
            ratio: to / from,
            frac: 0.0,
        }
    }

    /// The conversion ratio `to_rate / from_rate`.
    pub fn ratio(&self) -> f64 {
        self.ratio
    }

    /// Forget the carried fraction.
    pub fn reset(&mut self) {
        self.frac = 0.0;
    }

    /// Number of output-rate frames spanned by the next `frames` input-rate
    /// frames.
    #[inline]
    pub fn advance(&mut self, frames: usize) -> usize {
        let exact = self.frac + frames as f64 * self.ratio;
        let whole = exact.floor();
        self.frac = exact - whole;
        whole as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_points_cover_the_span_monotonically() {
        for (n, k) in [(480, 160), (64, 1536), (1, 0), (7, 7), (1000, 1)] {
            assert_eq!(split_point(0, n, k), 0);
            assert_eq!(split_point(n, n, k), k);
            let mut prev = 0;
            for pos in 0..=n {
                let s = split_point(pos, n, k);
                assert!(s >= prev && s <= k);
                prev = s;
            }
        }
        assert_eq!(split_point(5, 0, 10), 0);
    }

    #[test]
    fn frame_ratio_does_not_drift() {
        let mut r = FrameRatio::new(48_000.0, 44_100.0);
        let total: usize = (0..10_000).map(|i| r.advance(1 + i % 7)).sum();
        let input: usize = (0..10_000).map(|i| 1 + i % 7).sum();
        let exact = (input as f64 * 44_100.0 / 48_000.0).floor() as usize;
        assert!(total.abs_diff(exact) <= 1, "{total} vs {exact}");
        let mut r = FrameRatio::new(f64::NAN, 0.0);
        assert_eq!(r.advance(10), 10);
    }
}
