//! Measurement harness for strom-dsp: a fixed set of synthetic signals, run
//! through a processor configuration, reduced to a fixed set of numbers.
//!
//! Test and benchmark support only. The plugin never depends on this crate.
//!
//! - [`signals`]: the signals (synthetic, fixed seeds, generated in code).
//! - [`spectrum`]: windowed FFT magnitude with sine-calibrated levels.
//! - [`metrics`]: generic metrics over signals: loudness, true peak,
//!   gain-reduction split, clip depth, aliasing, IMD and NMR.
//! - [`chain`]: the broadcast chain cases and the numbers per case.
//! - [`crackle`]: crackle on speech: the chain on corpus clips, worst-frame
//!   NMR and clicks, and the acceptance check over presets and latency
//!   modes.
//! - [`baseline`]: the checked-in baseline (`baseline.json`) and the
//!   regression check against it.
//! - [`declip`]: the declipper on the test corpus, and its acceptance
//!   check (never worse than the clipped input).
//! - [`listen`]: the listening kit: processor configurations over the
//!   test corpus, before/after audio, numbers and a static listening page
//!   (binary `strom-dsp-listen`).
//!
//! Every metric is a plain function over interleaved `f32` signals, so a
//! new stage only needs to hand its before/after signals (or its meter
//! series) to the functions here.

pub mod baseline;
pub mod chain;
pub mod crackle;
pub mod declip;
pub mod listen;
pub mod metrics;
pub mod signals;
pub mod spectrum;

/// Shared with the strom-dsp integration tests (same files, so the same
/// generators and seeds).
#[path = "../../strom-dsp/tests/agc_support/mod.rs"]
pub mod agc_support;
#[path = "../../strom-dsp/tests/programme_support/mod.rs"]
pub mod programme_support;
