//! `strom-dsp`: a small clean-room audio DSP library.
//!
//! The channel-strip processors, re-exported at the crate root, all work in
//! place on interleaved `f32` frames:
//!
//! - [`Gate`]: downward noise gate with hysteresis and hold.
//! - [`Compressor`]: feed-forward downward compressor with soft knee.
//! - [`Equalizer`]: up to 8 RBJ-cookbook biquad bands in series.
//! - [`Limiter`]: lookahead sample-peak limiter.
//!
//! Further processors live in their modules: [`deesser`], [`ducker`],
//! [`delay`], [`stereo`], [`cleanup`], [`agc`], [`multiband`],
//! [`truepeak`] (true-peak limiter), [`clipper`] (final clipper) and
//! [`broadcast`] (the broadcast chain,
//! with its [`presets`]). Measurement is in [`meter`]; the building blocks
//! ([`biquad`], [`filter`], [`oversample`], [`resample`], [`stft`],
//! [`dynamics`]) are public too.
//!
//! # Usage contract (shared by every processor)
//!
//! 1. Construct with `new()` (cheap, unconfigured: `process` is a no-op passthrough).
//! 2. Call [`Processor::configure`] with sample rate and channel count. This is the
//!    only place that allocates.
//! 3. Call `set_params(&Params)` whenever parameters change. Never allocates;
//!    safe to call between any two `process` calls. Out-of-range values are clamped.
//! 4. Call [`Processor::process`] on interleaved buffers whose length is a multiple
//!    of the channel count. Real-time safe: no allocation, locking, logging or I/O.
//! 5. Call [`Processor::reset`] on flush/discontinuity.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod agc;
pub mod biquad;
pub mod broadcast;
pub mod cleanup;
pub mod clipper;
pub mod compressor;
pub mod deesser;
pub mod delay;
pub mod ducker;
pub mod dynamics;
pub mod equalizer;
pub mod filter;
pub mod gate;
pub mod limiter;
pub mod meter;
pub mod multiband;
pub mod oversample;
pub mod presets;
pub mod resample;
pub mod simd;
pub mod stereo;
pub mod stft;
pub mod truepeak;
pub mod util;

mod lanes;

pub use compressor::{Compressor, CompressorParams};
pub use equalizer::{BandParams, BandType, Equalizer, EqualizerParams, MAX_BANDS};
pub use gate::{Gate, GateParams};
pub use limiter::{Limiter, LimiterParams, MAX_LOOKAHEAD_MS};

/// Common interface of all processors.
pub trait Processor {
    /// Set sample rate (Hz, 8 000 … 192 000 supported) and channel count (≥ 1).
    /// Allocates all internal buffers and resets state.
    fn configure(&mut self, sample_rate: f32, channels: usize);

    /// Process interleaved frames in place. `samples.len()` must be a multiple
    /// of the configured channel count. A no-op if not configured.
    fn process(&mut self, samples: &mut [f32]);

    /// Clear all signal-dependent state (filters, envelopes, delay lines).
    fn reset(&mut self);

    /// Processing latency in frames (0 unless the processor looks ahead).
    fn latency_frames(&self) -> usize {
        0
    }
}
