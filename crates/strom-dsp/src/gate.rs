//! Downward noise gate with hysteresis, hold and attack/release fades.
//!
//! STUB: API is fixed; implementation to be written.

use crate::Processor;

/// Gate parameters (units as in the GStreamer interface).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateParams {
    /// Level (dBFS, −80 … 0) above which the gate opens.
    pub open_threshold_db: f32,
    /// Level (dBFS, −80 … 0) below which the gate starts closing. Treated as
    /// equal to `open_threshold_db` if larger.
    pub close_threshold_db: f32,
    /// Opening fade time (ms, 0.01 … 200).
    pub attack_ms: f32,
    /// Time the gate stays open after the level drops below the close threshold (ms, 0 … 1000).
    pub hold_ms: f32,
    /// Closing fade time (ms, 1 … 2000).
    pub release_ms: f32,
    /// Attenuation when closed (dB, −80 … 0).
    pub reduction_db: f32,
}

impl Default for GateParams {
    fn default() -> Self {
        Self {
            open_threshold_db: -20.0,
            close_threshold_db: -30.0,
            attack_ms: 5.0,
            hold_ms: 50.0,
            release_ms: 100.0,
            reduction_db: -60.0,
        }
    }
}

/// Linked-channel noise gate. See [`GateParams`].
#[derive(Debug, Default)]
pub struct Gate {
    params: GateParams,
}

impl Gate {
    /// Create an unconfigured gate with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update parameters. Never allocates.
    pub fn set_params(&mut self, params: &GateParams) {
        self.params = *params;
    }

    /// Current parameters.
    pub fn params(&self) -> &GateParams {
        &self.params
    }
}

impl Processor for Gate {
    fn configure(&mut self, _sample_rate: f32, _channels: usize) {}
    fn process(&mut self, _samples: &mut [f32]) {}
    fn reset(&mut self) {}
}
