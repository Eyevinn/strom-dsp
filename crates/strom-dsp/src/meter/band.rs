//! Phase correlation in three bands (low / mid / high).

use super::correlation::{CorrelationMeter, DEFAULT_CORRELATION_MS};
use crate::filter::{Crossover, CrossoverParams, LrOrder, MAX_SPLITS};

/// Number of bands of a [`BandCorrelation`] (low, mid, high).
pub const CORRELATION_BANDS: usize = 3;

/// Default split frequencies (Hz) of [`BandCorrelation`]: low < 250 Hz ≤ mid
/// < 4 kHz ≤ high.
///
/// Own choice: below 250 Hz is the bass region that is normally mono (and
/// where polarity errors cost the most when summed to mono); 250 Hz … 4 kHz
/// holds voice and most instrument fundamentals; above 4 kHz is the
/// "air"/sibilance region, where decorrelation from reverb and wide mixes is
/// expected.
pub const DEFAULT_BAND_SPLITS_HZ: [f32; CORRELATION_BANDS - 1] = [250.0, 4000.0];

/// Frames split per crossover call (size of the preallocated buffers).
const BLOCK_FRAMES: usize = 256;

/// Stereo phase correlation in a low, a mid and a high band.
///
/// L and R pass through the **same** 3-band Linkwitz-Riley LR4
/// [`Crossover`] (one crossover configured for two channels: identical
/// coefficients and independent state per channel, which is exactly two
/// identical crossovers on L and R, with the channels as SIMD lanes). Band
/// `k`'s L/R pair then feeds its own [`CorrelationMeter`]. Because L and R are
/// filtered identically, the filter phase cancels in the correlation: an
/// identical pair reads +1 and an inverted pair −1 in every band.
///
/// Splits default to [`DEFAULT_BAND_SPLITS_HZ`] (250 Hz and 4 kHz); each band
/// meter integrates over `integration_ms` (default
/// [`DEFAULT_CORRELATION_MS`]) and reads 0 when the band is silent in either
/// channel (see [`CorrelationMeter`]).
///
/// Real-time behaviour: [`BandCorrelation::new`] allocates (crossover state
/// and three `BLOCK_FRAMES`-frame band buffers). Everything else, including
/// `set_split_frequencies`, `set_integration_ms` and `reset`, never
/// allocates, locks or blocks. Cost: the crossover (2 LR4 splits with
/// all-pass compensation, about 10 biquads per channel) plus three
/// correlation updates per frame.
#[derive(Debug)]
pub struct BandCorrelation {
    crossover: Crossover,
    meters: [CorrelationMeter; CORRELATION_BANDS],
    /// Interleaved L/R input block.
    input: Vec<f32>,
    /// Interleaved L/R band outputs.
    bands: [Vec<f32>; CORRELATION_BANDS],
}

impl BandCorrelation {
    /// Create a band correlation meter at `sample_rate` Hz with the default
    /// splits and `integration_ms` per band. Allocates (setup only).
    pub fn new(sample_rate: f32, integration_ms: f32) -> Self {
        let mut crossover = Crossover::new();
        crossover.configure(sample_rate, 2);
        let mut b = Self {
            crossover,
            meters: core::array::from_fn(|_| CorrelationMeter::new(sample_rate, integration_ms)),
            input: vec![0.0; 2 * BLOCK_FRAMES],
            bands: core::array::from_fn(|_| vec![0.0; 2 * BLOCK_FRAMES]),
        };
        b.set_split_frequencies(DEFAULT_BAND_SPLITS_HZ);
        b.crossover.reset();
        b
    }

    /// Set the two split frequencies (Hz, low/mid then mid/high). Clamped to
    /// `[1 Hz, 0.499·fs]` and to be non-decreasing; a change glides over about
    /// 10 ms (see [`crate::filter`]). Never allocates.
    pub fn set_split_frequencies(&mut self, splits_hz: [f32; CORRELATION_BANDS - 1]) {
        let mut frequencies_hz = [0.0f32; MAX_SPLITS];
        frequencies_hz[..CORRELATION_BANDS - 1].copy_from_slice(&splits_hz);
        // Unused entries: keep them valid and non-decreasing.
        for f in &mut frequencies_hz[CORRELATION_BANDS - 1..] {
            *f = splits_hz[CORRELATION_BANDS - 2];
        }
        self.crossover.set_params(&CrossoverParams {
            order: LrOrder::Lr4,
            num_bands: CORRELATION_BANDS,
            frequencies_hz,
        });
    }

    /// The split frequencies as realised (Hz, after clamping).
    pub fn split_frequencies_hz(&self) -> [f64; CORRELATION_BANDS - 1] {
        let f = self.crossover.frequencies_hz();
        [f[0], f[1]]
    }

    /// Change every band's integration time (ms, see
    /// [`CorrelationMeter::set_integration_ms`]). Keeps state.
    pub fn set_integration_ms(&mut self, integration_ms: f32) {
        for m in &mut self.meters {
            m.set_integration_ms(integration_ms);
        }
    }

    /// Integration time in ms.
    pub fn integration_ms(&self) -> f32 {
        self.meters[0].integration_ms()
    }

    /// Clear the crossover state and the averages.
    pub fn reset(&mut self) {
        self.crossover.reset();
        for m in &mut self.meters {
            m.reset();
        }
    }

    /// Add interleaved frames of `channels` (≥ 2) channels, measuring channel
    /// 0 (L) against channel 1 (R). A trailing partial frame is ignored.
    pub fn process_interleaved(&mut self, samples: &[f32], channels: usize) {
        if channels < 2 {
            return;
        }
        for block in samples.chunks(BLOCK_FRAMES * channels) {
            let frames = block.len() / channels;
            for (d, f) in self
                .input
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(block.chunks_exact(channels))
            {
                d[0] = f[0];
                d[1] = f[1];
            }
            self.run(frames);
        }
    }

    /// Add two planar channel slices (the shorter length is used).
    pub fn process_pair(&mut self, left: &[f32], right: &[f32]) {
        let n = left.len().min(right.len());
        for start in (0..n).step_by(BLOCK_FRAMES) {
            let end = (start + BLOCK_FRAMES).min(n);
            for (d, (&l, &r)) in self
                .input
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(left[start..end].iter().zip(&right[start..end]))
            {
                d[0] = l;
                d[1] = r;
            }
            self.run(end - start);
        }
    }

    /// Split the first `frames` frames of `input` and update the band meters.
    fn run(&mut self, frames: usize) {
        if frames == 0 {
            return;
        }
        let len = 2 * frames;
        self.crossover.process(&self.input[..len], &mut self.bands);
        for (m, b) in self.meters.iter_mut().zip(&self.bands) {
            m.process_interleaved(&b[..len], 2);
        }
    }

    /// Correlation of band `band` (0 = low, 1 = mid, 2 = high), −1 … +1; 0 for
    /// a silent band or an out-of-range index.
    pub fn correlation(&self, band: usize) -> f64 {
        self.meters
            .get(band)
            .map_or(0.0, CorrelationMeter::correlation)
    }

    /// Correlations of all bands, `[low, mid, high]`.
    pub fn correlations(&self) -> [f64; CORRELATION_BANDS] {
        core::array::from_fn(|k| self.meters[k].correlation())
    }

    /// The meter of band `band` (side/mid ratio, mean square), if it exists.
    pub fn meter(&self, band: usize) -> Option<&CorrelationMeter> {
        self.meters.get(band)
    }
}

impl Default for BandCorrelation {
    /// 48 kHz, default splits and integration time.
    fn default() -> Self {
        Self::new(48_000.0, DEFAULT_CORRELATION_MS)
    }
}
