//! Measurement: BS.1770 loudness and true peak (wrapping `ebur128`), stereo
//! phase correlation and mono/stereo/inverted detection.
//!
//! - [`LoudnessMeter`]: momentary, short-term and integrated loudness, loudness
//!   range (EBU Tech 3342), sample peak and true peak, for 1–64 channels with
//!   BS.1770 channel weights from a [`ChannelLayout`] or explicit
//!   [`ChannelRole`]s.
//! - [`CorrelationMeter`]: phase correlation of any L/R pair (−1 … +1) with a
//!   configurable integration time, plus the side/mid energy ratio.
//! - [`BandCorrelation`]: the same correlation in a low, a mid and a high
//!   band, split by identical Linkwitz-Riley LR4 crossovers
//!   ([`crate::filter::Crossover`]) on L and R.
//! - [`LoudnessDetector`]: our own real-time BS.1770 detector (K-weighting
//!   designed for any sample rate, O(1) momentary and short-term readings
//!   from 10 ms sub-block running sums), for driving a gain every block.
//! - [`MonoDetector`]: classifies a stereo pair as [`StereoState::Mono`],
//!   [`StereoState::Stereo`] or [`StereoState::Inverted`], with hysteresis.
//!
//! Meters never modify the audio: their `process` methods take `&[f32]`.
//! Every `process`/`push`/reading method is real-time safe after construction
//! (no allocation, locking or I/O); see the type docs for read costs.

mod band;
mod correlation;
mod kweighting;
mod loudness;
mod mono;

pub use kweighting::{
    k_weighting, k_weighting_prototypes, lufs_from_mean_square, mean_square_from_lufs,
    AnalogBiquad, LoudnessDetector, BS1770_PRE_FILTER_48K, BS1770_RLB_48K, DETECTOR_RATE_RANGE,
    LOUDNESS_OFFSET_DB, MOMENTARY_SUB_BLOCKS, SHORT_TERM_SUB_BLOCKS, SILENCE_LUFS, SUB_BLOCK_MS,
};
pub(crate) use kweighting::{History, WindowSum};

pub use band::{BandCorrelation, CORRELATION_BANDS, DEFAULT_BAND_SPLITS_HZ};
pub use correlation::{CorrelationMeter, DEFAULT_CORRELATION_MS};
pub use loudness::{
    ChannelLayout, ChannelRole, LoudnessConfig, LoudnessMeter, LoudnessReading, MeterError,
    MAX_LOUDNESS_CHANNELS,
};
pub use mono::{MonoDetector, MonoDetectorParams, StereoState};
