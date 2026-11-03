//! Measurement: BS.1770 loudness and true peak (wrapping `ebur128`), stereo
//! phase correlation and mono/stereo/inverted detection.
//!
//! - [`LoudnessMeter`]: momentary, short-term and integrated loudness, loudness
//!   range (EBU Tech 3342), sample peak and true peak, for 1–64 channels with
//!   BS.1770 channel weights from a [`ChannelLayout`] or explicit
//!   [`ChannelRole`]s.
//! - [`CorrelationMeter`]: phase correlation of any L/R pair (−1 … +1) with a
//!   configurable integration time, plus the side/mid energy ratio.
//! - [`MonoDetector`]: classifies a stereo pair as [`StereoState::Mono`],
//!   [`StereoState::Stereo`] or [`StereoState::Inverted`], with hysteresis.
//!
//! Meters never modify the audio: their `process` methods take `&[f32]`.
//! Every `process`/`push`/reading method is real-time safe after construction
//! (no allocation, locking or I/O); see the type docs for read costs.
//!
//! Per-band correlation (low/mid/high, phase 1b spec 2.2) is deliberately not
//! implemented here: it needs the Linkwitz-Riley crossover from
//! [`crate::filter`]. [`CorrelationMeter`] is band-agnostic, so a band meter is
//! one crossover plus one `CorrelationMeter` per band.
// TODO(phase 1b, 2.2): add `BandCorrelation` once `filter::Crossover` exists:
// split L and R with two identical crossovers and feed each band's L/R pair to
// its own `CorrelationMeter`.

mod correlation;
mod loudness;
mod mono;

pub use correlation::{CorrelationMeter, DEFAULT_CORRELATION_MS};
pub use loudness::{
    ChannelLayout, ChannelRole, LoudnessConfig, LoudnessMeter, LoudnessReading, MeterError,
    MAX_LOUDNESS_CHANNELS,
};
pub use mono::{MonoDetector, MonoDetectorParams, StereoState};
