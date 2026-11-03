//! Feed-forward downward compressor with soft knee.
//!
//! STUB: API is fixed; implementation to be written.

use crate::Processor;

/// Compressor parameters (units as in the GStreamer interface).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressorParams {
    /// Threshold as linear amplitude (0 … 1).
    pub threshold: f32,
    /// Compression ratio (1 … 100; 4 means 4:1).
    pub ratio: f32,
    /// Attack time (ms, 0.01 … 10000).
    pub attack_ms: f32,
    /// Release time (ms, 0.01 … 10000).
    pub release_ms: f32,
    /// Makeup gain as linear multiplier (0 … 100).
    pub makeup_gain: f32,
    /// Knee factor `k` (0 … 1): soft-knee region spans `threshold·k … threshold/k`,
    /// width `−40·log10(k)` dB. 1 = hard knee; values below 0.001 are treated as 0.001.
    pub knee: f32,
}

impl Default for CompressorParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            ratio: 1.0,
            attack_ms: 20.0,
            release_ms: 100.0,
            makeup_gain: 1.0,
            knee: 0.0625,
        }
    }
}

/// Linked-channel feed-forward compressor. See [`CompressorParams`].
#[derive(Debug, Default)]
pub struct Compressor {
    params: CompressorParams,
}

impl Compressor {
    /// Create an unconfigured compressor with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update parameters. Never allocates.
    pub fn set_params(&mut self, params: &CompressorParams) {
        self.params = *params;
    }

    /// Current parameters.
    pub fn params(&self) -> &CompressorParams {
        &self.params
    }
}

impl Processor for Compressor {
    fn configure(&mut self, _sample_rate: f32, _channels: usize) {}
    fn process(&mut self, _samples: &mut [f32]) {}
    fn reset(&mut self) {}
}
