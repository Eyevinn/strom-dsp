//! Sidechain key detection: an optional key filter followed by the linked
//! peak, shared by every keyed dynamics processor (gate, compressor, and the
//! ducker / de-esser built on top of these blocks).
//!
//! # Signal flow
//!
//! 1. **Key filter** ([`KeyFilter`]): off, or one second-order section from
//!    the existing [`biquad`](crate::biquad) building blocks (RBJ cookbook
//!    high-pass, or band-pass with constant 0 dB peak gain), run per key
//!    channel with its own state. The key buffer itself is never modified: each
//!    block of at most [`BLOCK_FRAMES`] frames is copied into a scratch buffer
//!    allocated in [`KeyDetector::configure`] and filtered there.
//! 2. **Linked peak**: `max_c |k_c|` over the key's channels for every frame
//!    ([`frame_peaks_simd`]), which is what the processors' detectors consume
//!    in place of the peak of the programme.
//!
//! The key may have a different channel count than the programme; it must
//! have the same frame count and sample rate (resampling is the host's job).
//!
//! # Channel-count contract
//!
//! The key filter keeps state per key channel, so the channel count is fixed
//! at configuration time ([`KeyDetector::configure`], the only allocating
//! call). A call whose `key_channels` differs from the configured count (or
//! is 0) is detected as **silence** (all peaks 0), and so are frames for which
//! the key slice is too short. Silence is the fail-safe choice: a keyed
//! compressor, ducker or de-esser then does nothing, and a keyed gate behaves
//! as if the key had stopped.
//!
//! # Filter changes
//!
//! A filter change only affects the detector path, never the audio directly,
//! and the processors smooth their gain afterwards, so new coefficients apply
//! at the next sample ([`Biquad::set_coefficients`]; DF-I state stays valid).
//! Switching the filter on clears its state.
//!
//! Exactness: with the filter off, the result is bit-identical on every SIMD
//! level. With the filter on, the biquad follows its FMA policy
//! ([`uses_fma`](crate::biquad::uses_fma)): levels with the same policy are
//! bit-identical.

use fearless_simd::{dispatch, Simd};

use crate::biquad::{Biquad, BiquadCoefficients};
use crate::dynamics::frame::{frame_peaks_simd, BLOCK_FRAMES};
use crate::simd::{detect, Level};

/// Lowest key-filter frequency (Hz).
pub const KEY_FILTER_MIN_HZ: f32 = 10.0;
/// Highest key-filter frequency (Hz); also limited to `0.45·fs`.
pub const KEY_FILTER_MAX_HZ: f32 = 20_000.0;
/// Q range of the key filter.
pub const KEY_FILTER_Q_RANGE: (f32, f32) = (0.1, 10.0);

/// Optional filter on the detector (key) path.
///
/// Frequencies in Hz, clamped to `[KEY_FILTER_MIN_HZ, min(KEY_FILTER_MAX_HZ,
/// 0.45·fs)]`; `q` is the cookbook Q, clamped to [`KEY_FILTER_Q_RANGE`]. A NaN
/// frequency or Q turns the filter off.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum KeyFilter {
    /// No filter: the detector sees the key as is.
    #[default]
    Off,
    /// Second-order high-pass at `frequency_hz` (RBJ cookbook HPF;
    /// `q = 0.707` is Butterworth, −3 dB at the corner).
    HighPass {
        /// Corner frequency (Hz).
        frequency_hz: f32,
        /// Cookbook Q (0.707 = Butterworth).
        q: f32,
    },
    /// Second-order band-pass centred on `frequency_hz` with 0 dB gain at the
    /// centre (RBJ cookbook BPF, constant 0 dB peak gain).
    BandPass {
        /// Centre frequency (Hz).
        frequency_hz: f32,
        /// Cookbook Q (bandwidth = centre / Q).
        q: f32,
    },
}

impl KeyFilter {
    /// Biquad coefficients at `sample_rate` (Hz), or `None` for [`KeyFilter::Off`]
    /// (and for NaN parameters).
    pub fn coefficients(&self, sample_rate: f64) -> Option<BiquadCoefficients> {
        let (f, q, high) = match *self {
            KeyFilter::Off => return None,
            KeyFilter::HighPass { frequency_hz, q } => (frequency_hz, q, true),
            KeyFilter::BandPass { frequency_hz, q } => (frequency_hz, q, false),
        };
        if f.is_nan() || q.is_nan() {
            return None;
        }
        let f_max = (KEY_FILTER_MAX_HZ as f64).min(0.45 * sample_rate);
        let f = (f as f64).clamp(
            KEY_FILTER_MIN_HZ as f64,
            f_max.max(KEY_FILTER_MIN_HZ as f64),
        );
        let q = (q as f64).clamp(KEY_FILTER_Q_RANGE.0 as f64, KEY_FILTER_Q_RANGE.1 as f64);
        Some(if high {
            BiquadCoefficients::high_pass(f, q, sample_rate)
        } else {
            BiquadCoefficients::band_pass_peak(f, q, sample_rate)
        })
    }
}

/// Sidechain detector front end: optional [`KeyFilter`] plus the linked peak
/// over the key's channels. See the module documentation.
///
/// Real-time safe after [`configure`](Self::configure): every other method is
/// allocation-free.
#[derive(Debug)]
pub struct KeyDetector {
    filter: KeyFilter,
    /// Coefficients of `filter` at the current rate (`None` = off).
    coefs: Option<BiquadCoefficients>,
    sample_rate: f64,
    /// Configured key channel count (0 = not configured).
    channels: usize,
    /// Per-channel filter state (allocated in `configure`).
    biquad: Option<Biquad>,
    /// One block of key samples to filter (`BLOCK_FRAMES · channels`).
    scratch: Vec<f32>,
    level: Level,
    level_forced: bool,
}

impl Default for KeyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyDetector {
    /// Unconfigured detector (filter off). Does not allocate; every call
    /// reports silence until [`configure`](Self::configure).
    pub fn new() -> Self {
        Self {
            filter: KeyFilter::Off,
            coefs: None,
            sample_rate: 48_000.0,
            channels: 0,
            biquad: None,
            scratch: Vec::new(),
            level: Level::baseline(),
            level_forced: false,
        }
    }

    /// Set the sample rate (Hz) and the key's channel count (≥ 1), allocate the
    /// filter state and scratch buffer, and clear the state. The only
    /// allocating call.
    pub fn configure(&mut self, sample_rate: f32, key_channels: usize) {
        self.sample_rate = (sample_rate as f64).clamp(1.0, 1.0e6);
        self.channels = key_channels;
        if !self.level_forced {
            self.level = detect();
        }
        if key_channels == 0 {
            self.biquad = None;
            self.scratch = Vec::new();
        } else {
            let mut bq = Biquad::new(BiquadCoefficients::IDENTITY, key_channels);
            bq.set_simd_level(self.level);
            self.biquad = Some(bq);
            self.scratch = vec![0.0; BLOCK_FRAMES * key_channels];
        }
        self.coefs = self.filter.coefficients(self.sample_rate);
        self.apply_coefs();
    }

    /// Configured key channel count (0 = not configured).
    pub fn key_channels(&self) -> usize {
        self.channels
    }

    /// Set the key filter. Never allocates; takes effect at the next frame.
    pub fn set_filter(&mut self, filter: KeyFilter) {
        let was_off = self.coefs.is_none();
        self.filter = filter;
        self.coefs = filter.coefficients(self.sample_rate);
        if was_off {
            if let Some(bq) = self.biquad.as_mut() {
                bq.reset();
            }
        }
        self.apply_coefs();
    }

    /// The key filter as last set (unclamped).
    pub fn filter(&self) -> KeyFilter {
        self.filter
    }

    /// Clear the filter state.
    pub fn reset(&mut self) {
        if let Some(bq) = self.biquad.as_mut() {
            bq.reset();
        }
    }

    /// Force the SIMD level of the key filter (kept across `configure`), as
    /// `set_simd_level` on the processors. Never allocates.
    pub fn set_simd_level(&mut self, level: Level) {
        self.level = level;
        self.level_forced = true;
        if let Some(bq) = self.biquad.as_mut() {
            bq.set_simd_level(level);
        }
    }

    fn apply_coefs(&mut self) {
        if let (Some(bq), Some(c)) = (self.biquad.as_mut(), self.coefs) {
            bq.set_coefficients(c);
        }
    }

    /// Linked key peaks, dispatching on the detected SIMD level: writes
    /// `max_c |k_c|` (after the key filter) of each frame of the interleaved
    /// `key` (`key_channels` channels) into `out`, one value per frame, for
    /// `out.len()` frames. See the module docs for the channel-count contract
    /// (mismatch or missing frames read as silence).
    pub fn linked_peaks(&mut self, key: &[f32], key_channels: usize, out: &mut [f32]) {
        dispatch!(self.level, simd => self.linked_peaks_simd(simd, key, key_channels, out));
    }

    /// [`linked_peaks`](Self::linked_peaks) for code already running under
    /// `fearless_simd::dispatch!` with instruction set `simd`.
    #[inline(always)]
    pub fn linked_peaks_simd<S: Simd>(
        &mut self,
        simd: S,
        key: &[f32],
        key_channels: usize,
        out: &mut [f32],
    ) {
        let ch = self.channels;
        if ch == 0 || key_channels != ch {
            out.fill(0.0);
            return;
        }
        let avail = (key.len() / ch).min(out.len());
        let (live, missing) = out.split_at_mut(avail);
        missing.fill(0.0);
        let key = &key[..avail * ch];
        match (self.coefs.is_some(), self.biquad.as_mut()) {
            (true, Some(bq)) => {
                for (k, o) in key
                    .chunks(BLOCK_FRAMES * ch)
                    .zip(live.chunks_mut(BLOCK_FRAMES))
                {
                    let s = &mut self.scratch[..k.len()];
                    s.copy_from_slice(k);
                    bq.process(s);
                    frame_peaks_simd(simd, s, ch, o);
                }
            }
            _ => frame_peaks_simd(simd, key, ch, live),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unfiltered_peak_is_max_over_key_channels() {
        let mut d = KeyDetector::new();
        d.configure(48_000.0, 3);
        let key = [0.1, -0.5, 0.2, 0.0, 0.0, -0.0, 0.3, 0.3, -0.9];
        let mut out = [1.0f32; 3];
        d.linked_peaks(&key, 3, &mut out);
        assert_eq!(out, [0.5, 0.0, 0.9]);
    }

    #[test]
    fn mismatch_and_short_key_read_as_silence() {
        let mut d = KeyDetector::new();
        d.configure(48_000.0, 2);
        let mut out = [1.0f32; 4];
        d.linked_peaks(&[0.5; 8], 1, &mut out);
        assert_eq!(out, [0.0; 4]);
        d.linked_peaks(&[0.5; 4], 2, &mut out);
        assert_eq!(out, [0.5, 0.5, 0.0, 0.0]);
        let mut u = KeyDetector::new();
        u.linked_peaks(&[0.5; 4], 2, &mut out);
        assert_eq!(out, [0.0; 4]);
    }

    #[test]
    fn high_pass_rejects_dc() {
        let mut d = KeyDetector::new();
        d.configure(48_000.0, 1);
        d.set_filter(KeyFilter::HighPass {
            frequency_hz: 1000.0,
            q: 0.707,
        });
        let key = vec![0.5f32; 48_000];
        let mut out = vec![0.0f32; 48_000];
        d.linked_peaks(&key, 1, &mut out);
        assert!(out[40_000..].iter().all(|&p| p < 1e-6));
    }

    #[test]
    fn nan_parameters_turn_the_filter_off() {
        let f = KeyFilter::BandPass {
            frequency_hz: f32::NAN,
            q: 1.0,
        };
        assert!(f.coefficients(48_000.0).is_none());
        let f = KeyFilter::HighPass {
            frequency_hz: 1e9,
            q: 1e9,
        };
        assert!(f.coefficients(8_000.0).unwrap().is_stable());
    }
}
