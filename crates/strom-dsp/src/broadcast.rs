//! Broadcast final processor chain (phase 2 spec §2.6):
//! input gain → wideband AGC → multiband compressor → true-peak limiter.
//!
//! STUB: the public API below is fixed (the GStreamer element
//! `stromdsp-broadcastprocessor` is written against it in parallel); the
//! processing is a passthrough with the reported latency until the chain is
//! implemented.
//!
//! Units follow the element interface (phase 2 spec §3.1): dB, LUFS, dBTP,
//! dB/s, ms, Hz. Meter conventions follow the agreed element message:
//! gain reductions are **positive dB of reduction**, the AGC gain is
//! **signed** (+ = boost), and silence reads [`SILENCE_DB`].

use crate::multiband::{MultibandBandParams, MAX_BANDS};
use crate::Processor;

/// Value reported for silence / no measurement yet (never NaN or −∞).
pub const SILENCE_DB: f64 = -200.0;

/// Number of crossover frequencies (`MAX_BANDS − 1`).
pub const MAX_CROSSOVERS: usize = MAX_BANDS - 1;

/// All parameters of the chain (phase 2 spec §3.1, minus `preset` and
/// `meter-interval`, which belong to the element).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BroadcastParams {
    /// Unity gain with the same delay when false (true bypass).
    pub enabled: bool,
    /// Input gain (dB, −24 … 24).
    pub input_gain_db: f32,
    /// AGC target (LUFS, −40 … −5).
    pub target_loudness_lufs: f32,
    /// True-peak ceiling (dBTP, −12 … 0).
    pub true_peak_ceiling_dbtp: f32,
    /// AGC stage active (bypassed stages keep their latency).
    pub agc_enabled: bool,
    /// Largest AGC boost (dB, 0 … 30).
    pub agc_max_gain_db: f32,
    /// Largest AGC cut (dB, 0 … 30).
    pub agc_max_cut_db: f32,
    /// AGC gain rise rate (dB/s, 0.1 … 20).
    pub agc_rate_up_db_s: f32,
    /// AGC gain fall rate (dB/s, 0.1 … 40).
    pub agc_rate_down_db_s: f32,
    /// AGC freeze level (LUFS, −80 … −20): below it the gain holds.
    pub agc_gate_lufs: f32,
    /// Multiband stage active.
    pub multiband_enabled: bool,
    /// Number of bands (1 … 5). Structural (READY-only on the element).
    pub num_bands: usize,
    /// Gain into all bands (dB, −12 … 12).
    pub multiband_drive_db: f32,
    /// Crossover frequencies (Hz, 20 … 20000), ascending; the first
    /// `num_bands − 1` are used.
    pub crossover_hz: [f32; MAX_CROSSOVERS],
    /// Per-band compressor settings; band 0 is the lowest.
    pub bands: [MultibandBandParams; MAX_BANDS],
    /// Limiter stage active.
    pub limiter_enabled: bool,
    /// Limiter release (ms, 1 … 2000).
    pub limiter_release_ms: f32,
    /// Limiter lookahead (ms, 0.5 … 20). Structural (READY-only on the element).
    pub limiter_lookahead_ms: f32,
}

impl Default for BroadcastParams {
    /// Neutral placeholder until presets exist; the element applies the
    /// `ebu-r128` preset as its default.
    fn default() -> Self {
        Self {
            enabled: true,
            input_gain_db: 0.0,
            target_loudness_lufs: -23.0,
            true_peak_ceiling_dbtp: -1.0,
            agc_enabled: true,
            agc_max_gain_db: 12.0,
            agc_max_cut_db: 12.0,
            agc_rate_up_db_s: 1.0,
            agc_rate_down_db_s: 3.0,
            agc_gate_lufs: -50.0,
            multiband_enabled: true,
            num_bands: 4,
            multiband_drive_db: 0.0,
            crossover_hz: [120.0, 800.0, 4000.0, 10_000.0],
            bands: core::array::from_fn(|k| MultibandBandParams::default_for(k, 4)),
            limiter_enabled: true,
            limiter_release_ms: 100.0,
            limiter_lookahead_ms: 5.0,
        }
    }
}

/// Meter values of the chain (phase 2 spec §2.7). Read once per block by
/// the element; dB/LUFS values are [`SILENCE_DB`] when there is nothing to
/// measure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BroadcastMeters {
    /// Input momentary loudness (LUFS, 400 ms).
    pub input_momentary_lufs: f64,
    /// Input short-term loudness (LUFS, 3 s).
    pub input_short_term_lufs: f64,
    /// Output momentary loudness (LUFS).
    pub output_momentary_lufs: f64,
    /// Output short-term loudness (LUFS).
    pub output_short_term_lufs: f64,
    /// Output gated integrated loudness since configure/reset/`reset_integrated` (LUFS).
    pub output_integrated_lufs: f64,
    /// Input true peak (dBTP), max since the last [`BroadcastProcessor::take_meters`].
    pub input_true_peak_dbtp: f64,
    /// Output true peak (dBTP), max since the last [`BroadcastProcessor::take_meters`].
    pub output_true_peak_dbtp: f64,
    /// AGC gain (dB, signed: + = boost).
    pub agc_gain_db: f64,
    /// Gain reduction per band (positive dB); only the first `num_bands` are meaningful.
    pub band_gain_reduction_db: [f64; MAX_BANDS],
    /// Limiter gain reduction (positive dB).
    pub limiter_gain_reduction_db: f64,
}

impl Default for BroadcastMeters {
    fn default() -> Self {
        Self {
            input_momentary_lufs: SILENCE_DB,
            input_short_term_lufs: SILENCE_DB,
            output_momentary_lufs: SILENCE_DB,
            output_short_term_lufs: SILENCE_DB,
            output_integrated_lufs: SILENCE_DB,
            input_true_peak_dbtp: SILENCE_DB,
            output_true_peak_dbtp: SILENCE_DB,
            agc_gain_db: 0.0,
            band_gain_reduction_db: [0.0; MAX_BANDS],
            limiter_gain_reduction_db: 0.0,
        }
    }
}

/// The broadcast final processor. See the module documentation.
#[derive(Debug, Default)]
pub struct BroadcastProcessor {
    params: BroadcastParams,
    sample_rate: f32,
    channels: usize,
}

impl BroadcastProcessor {
    /// Create an unconfigured processor with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update parameters (clamped). Never allocates. Structural fields
    /// (`num_bands`, `limiter_lookahead_ms`) take effect at the next
    /// `configure`.
    pub fn set_params(&mut self, params: &BroadcastParams) {
        self.params = *params;
    }

    /// Current parameters.
    pub fn params(&self) -> &BroadcastParams {
        &self.params
    }

    /// Meter values since the last call; resets the true-peak maxima.
    /// Never allocates. Cheap enough to call once per buffer.
    pub fn take_meters(&mut self) -> BroadcastMeters {
        BroadcastMeters::default()
    }

    /// Restart the output integrated loudness measurement.
    pub fn reset_integrated(&mut self) {}

    /// Total latency in frames for a configuration (lookahead + oversampling
    /// filter delay). Constant for a given rate and lookahead, independent of
    /// stage bypass. Usable before `configure`.
    pub fn latency_frames_for(_sample_rate: f32, lookahead_ms: f32) -> usize {
        (lookahead_ms.max(0.0) as f64 * 0.001 * 48_000.0).round() as usize
    }
}

impl Processor for BroadcastProcessor {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        self.sample_rate = sample_rate;
        self.channels = channels;
    }

    fn process(&mut self, _samples: &mut [f32]) {}

    fn reset(&mut self) {}

    fn latency_frames(&self) -> usize {
        Self::latency_frames_for(self.sample_rate, self.params.limiter_lookahead_ms)
    }
}
