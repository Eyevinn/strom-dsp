//! Parametric equalizer: up to 8 RBJ-cookbook biquad bands in series.
//!
//! STUB: API is fixed; implementation to be written.

use crate::Processor;

/// Maximum number of bands.
pub const MAX_BANDS: usize = 8;

/// Filter type of one band. The integer values are part of the plugin interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum BandType {
    /// 0: band bypassed.
    #[default]
    Off = 0,
    /// 1: second-order low-pass.
    LowPass = 1,
    /// 2: second-order high-pass.
    HighPass = 2,
    /// 3: band-pass, constant skirt gain (peak gain = Q).
    BandPassSkirt = 3,
    /// 4: band-pass, constant 0 dB peak gain.
    BandPassPeak = 4,
    /// 5: notch.
    Notch = 5,
    /// 6: all-pass.
    AllPass = 6,
    /// 7: peaking (bell).
    Peaking = 7,
    /// 8: low shelf.
    LowShelf = 8,
    /// 9: high shelf.
    HighShelf = 9,
}

impl BandType {
    /// Map the interface integer to a type; anything outside 0 … 9 maps to [`BandType::Off`].
    pub fn from_i32(v: i32) -> Self {
        match v {
            1 => Self::LowPass,
            2 => Self::HighPass,
            3 => Self::BandPassSkirt,
            4 => Self::BandPassPeak,
            5 => Self::Notch,
            6 => Self::AllPass,
            7 => Self::Peaking,
            8 => Self::LowShelf,
            9 => Self::HighShelf,
            _ => Self::Off,
        }
    }
}

/// Parameters of one band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandParams {
    /// Band active. A disabled band costs nothing and is bit-exact passthrough.
    pub enabled: bool,
    /// Filter type.
    pub band_type: BandType,
    /// Centre / corner frequency (Hz, 1 … 24000; clamped just below Nyquist).
    pub frequency: f32,
    /// Gain (dB, −60 … 60); used by peaking and shelves only.
    pub gain_db: f32,
    /// Cookbook Q (0.01 … 100), also for shelves.
    pub q: f32,
}

impl Default for BandParams {
    fn default() -> Self {
        Self {
            enabled: true,
            band_type: BandType::Off,
            frequency: 1000.0,
            gain_db: 0.0,
            q: 0.707,
        }
    }
}

/// Equalizer parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EqualizerParams {
    /// Number of active bands (1 … 8). Bands at index ≥ `num_bands` are inactive.
    pub num_bands: usize,
    /// All eight band settings, always present.
    pub bands: [BandParams; MAX_BANDS],
}

impl Default for EqualizerParams {
    fn default() -> Self {
        Self {
            num_bands: 4,
            bands: [BandParams::default(); MAX_BANDS],
        }
    }
}

/// Multi-band parametric equalizer. See [`EqualizerParams`].
#[derive(Debug, Default)]
pub struct Equalizer {
    params: EqualizerParams,
}

impl Equalizer {
    /// Create an unconfigured equalizer with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update parameters. Never allocates.
    pub fn set_params(&mut self, params: &EqualizerParams) {
        self.params = *params;
    }

    /// Current parameters.
    pub fn params(&self) -> &EqualizerParams {
        &self.params
    }
}

impl Processor for Equalizer {
    fn configure(&mut self, _sample_rate: f32, _channels: usize) {}
    fn process(&mut self, _samples: &mut [f32]) {}
    fn reset(&mut self) {}
}
