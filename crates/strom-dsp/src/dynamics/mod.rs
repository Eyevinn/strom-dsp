//! Reusable building blocks for dynamics processors.
//!
//! The [`Gate`](crate::Gate), [`Compressor`](crate::Compressor) and
//! [`Limiter`](crate::Limiter) are assembled from these parts; new elements
//! (expanders, de-essers, duckers, meters, …) can be built from them too.
//!
//! All blocks follow the crate's real-time contract: anything that allocates is
//! named `allocate`/`new` and happens at configuration time; every per-sample
//! method is allocation-free, lock-free and panic-free for valid inputs.
//!
//! | Block | What it does |
//! |---|---|
//! | [`OnePole`] | one-pole smoother toward a target (parameter glides, fades) |
//! | [`AttackRelease`] | one-pole with separate rising/falling time constants |
//! | [`PeakEnvelope`] | peak follower: instant attack, exponential release |
//! | [`GainComputer`] | static compressor curve with quadratic soft knee (dB in, dB out) |
//! | [`GateStateMachine`] | open/close hysteresis with hold counter |
//! | [`DelayLine`] | multichannel interleaved ring buffer with frame taps |
//! | [`SlidingMin`] | O(1) amortised sliding-window minimum |
//! | [`BoxAverage`] | exact integer moving average, rounding down |
//! | [`RampSmoother`] | two cascaded boxes: smooth S-ramp that never exceeds its inputs' bound |
//! | [`frame`] | linked-channel helpers: per-frame peaks and gain application |
//!
//! **Time-constant convention** for every exponential block here: a time `τ`
//! (ms) gives the coefficient `a = exp(−1/(τ·fs))`
//! ([`one_pole_coef`](crate::util::one_pole_coef)), and the step response covers
//! `1 − 1/e` (≈ 63.2 %) of the step after `τ`.

pub mod frame;
mod gain_computer;
mod gate_state;
mod ramp;
mod ring;
mod smoother;

pub use gain_computer::{GainComputer, MIN_KNEE};
pub use gate_state::GateStateMachine;
pub use ramp::{BoxAverage, RampSmoother, RAMP_FRAC_BITS};
pub use ring::{DelayLine, SlidingMin};
pub use smoother::{AttackRelease, OnePole, PeakEnvelope};
