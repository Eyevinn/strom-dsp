//! Lookahead sample-peak limiter.
//!
//! STUB: API is fixed; implementation to be written.

use crate::Processor;

/// Largest supported lookahead (ms). The delay line is sized for this at
/// `configure` time so lookahead changes never allocate.
pub const MAX_LOOKAHEAD_MS: f32 = 20.0;

/// Limiter parameters (units as in the GStreamer interface).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LimiterParams {
    /// When false: unity gain, but the lookahead delay is kept (constant latency).
    pub enabled: bool,
    /// Ceiling (dBFS, −60 … 0). No output sample exceeds it.
    pub threshold_db: f32,
    /// Attack time (ms, 0.01 … 50).
    pub attack_ms: f32,
    /// Release time (ms, 1 … 2000).
    pub release_ms: f32,
    /// Lookahead (ms, 0 … 20). Latency = round(lookahead · fs / 1000) frames.
    pub lookahead_ms: f32,
}

impl Default for LimiterParams {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold_db: -1.0,
            attack_ms: 1.0,
            release_ms: 100.0,
            lookahead_ms: 5.0,
        }
    }
}

/// Linked-channel lookahead limiter. See [`LimiterParams`].
#[derive(Debug, Default)]
pub struct Limiter {
    params: LimiterParams,
}

impl Limiter {
    /// Create an unconfigured limiter with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update parameters. Never allocates.
    pub fn set_params(&mut self, params: &LimiterParams) {
        self.params = *params;
    }

    /// Current parameters.
    pub fn params(&self) -> &LimiterParams {
        &self.params
    }

    /// Number of times the final safety clip changed a sample since the last
    /// `reset`/`configure`. Expected to stay 0 on normal material.
    pub fn safety_clip_count(&self) -> u64 {
        0
    }
}

impl Processor for Limiter {
    fn configure(&mut self, _sample_rate: f32, _channels: usize) {}
    fn process(&mut self, _samples: &mut [f32]) {}
    fn reset(&mut self) {}
}
