//! Downward noise gate with hysteresis, hold and attack/release fades.
//!
//! # Signal flow (per frame of interleaved samples)
//!
//! 1. **Key level**: the largest absolute sample value over all channels of the
//!    frame (channels are linked, so one gain is shared by all of them).
//! 2. **Detector**: a peak envelope follower ([`PeakEnvelope`]) with
//!    instantaneous attack and a fixed exponential release of [`DETECTOR_RELEASE_MS`]:
//!    `e[n] = p[n]` if `p[n] > e[n-1]`, otherwise `e[n] = a_d·e[n-1] + (1-a_d)·p[n]`
//!    with `a_d = exp(-1 / (τ_d·fs))`, `τ_d = DETECTOR_RELEASE_MS / 1000`. For a steady
//!    sine the envelope sits at the sine's peak amplitude (ripple well below 1 dB above
//!    ~200 Hz), so the thresholds compare against the **peak level in dBFS**.
//! 3. **State machine with hysteresis and hold** ([`GateStateMachine`]): a closed
//!    gate opens when `e > open_threshold`. An open gate re-arms its hold counter
//!    (`round(hold_ms·fs/1000)` frames) every frame where `e ≥ close_threshold`;
//!    once `e` is below the close threshold the counter runs down, and when it is
//!    exhausted the gate closes. If `close_threshold_db > open_threshold_db` the
//!    close threshold is treated as equal to the open threshold.
//! 4. **Gain fade** ([`AttackRelease`] on the linear gain): the gain `g` follows its
//!    target (1 when open, the linear `reduction` when closed) with a one-pole smoother,
//!    `g[n] = g[n-1] + (1 - a)·(target - g[n-1])`, where `a = exp(-1 / (τ·fs))`,
//!    `τ = attack_ms/1000` while the gain rises and `τ = release_ms/1000` while it
//!    falls. So **attack/release is the time for the linear gain to cover
//!    `1 - 1/e` (≈ 63.2 %) of the distance to its target** after a step.
//!
//! Parameter changes are click-free: a changed `reduction` (or a state change from
//! a changed threshold) only moves the gain target, and the gain always reaches it
//! through the attack/release smoother. The exception is the time between
//! `configure`/`reset` and the first processed frame: there `set_params` applies
//! instantly, so the result does not depend on whether parameters were set before
//! or after `configure`.
//!
//! Processing runs in blocks of [`BLOCK_FRAMES`]: per-frame peaks for the block,
//! then the serial detector/state/fade chain, then the gains are applied.
//!
//! Denormals: the detector envelope is passed through
//! [`flush_denormal`](crate::util::flush_denormal); the gain is snapped to its
//! target once it is within `1e-12` (targets are ≥ 1e-4, so it never decays toward 0).

use crate::dynamics::frame::{apply_gains, frame_peaks, BLOCK_FRAMES};
use crate::dynamics::{AttackRelease, GateStateMachine, PeakEnvelope};
use crate::util::db_to_lin;
use crate::Processor;

/// Release time constant (ms) of the gate's internal peak detector (attack is instantaneous).
pub const DETECTOR_RELEASE_MS: f64 = 20.0;

/// The gain snaps to its target within this distance (targets are ≥ 1e-4).
const GAIN_SNAP: f64 = 1e-12;

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

/// Clamp `v` to `[lo, hi]`, replacing NaN with `fallback`.
fn clamp_or(v: f32, lo: f32, hi: f32, fallback: f32) -> f32 {
    if v.is_nan() {
        fallback
    } else {
        v.clamp(lo, hi)
    }
}

impl GateParams {
    /// Copy with every field clamped to its documented range (NaN becomes the default).
    pub fn clamped(&self) -> Self {
        let d = Self::default();
        Self {
            open_threshold_db: clamp_or(self.open_threshold_db, -80.0, 0.0, d.open_threshold_db),
            close_threshold_db: clamp_or(self.close_threshold_db, -80.0, 0.0, d.close_threshold_db),
            attack_ms: clamp_or(self.attack_ms, 0.01, 200.0, d.attack_ms),
            hold_ms: clamp_or(self.hold_ms, 0.0, 1000.0, d.hold_ms),
            release_ms: clamp_or(self.release_ms, 1.0, 2000.0, d.release_ms),
            reduction_db: clamp_or(self.reduction_db, -80.0, 0.0, d.reduction_db),
        }
    }
}

/// Linked-channel noise gate. See [`GateParams`] and the module documentation.
#[derive(Debug)]
pub struct Gate {
    params: GateParams,
    sample_rate: f64,
    /// 0 = not configured.
    channels: usize,

    reduction_lin: f64,
    detector: PeakEnvelope,
    state: GateStateMachine,
    /// Linear gain: attack while rising (opening), release while falling.
    fade: AttackRelease,
    /// True from configure/reset until the first frame is processed: parameter
    /// changes apply instantly, including the initial (closed) gain.
    fresh: bool,
}

impl Default for Gate {
    fn default() -> Self {
        Self::new()
    }
}

impl Gate {
    /// Create an unconfigured gate with default parameters.
    pub fn new() -> Self {
        let mut g = Self {
            params: GateParams::default(),
            sample_rate: 48_000.0,
            channels: 0,
            reduction_lin: 0.0,
            detector: PeakEnvelope::default(),
            state: GateStateMachine::default(),
            fade: AttackRelease::default().with_snap(GAIN_SNAP),
            fresh: true,
        };
        g.update_coefficients();
        g.reset();
        g
    }

    /// Update parameters (clamped to their ranges). Never allocates.
    ///
    /// Between `configure`/`reset` and the first processed frame the new
    /// parameters apply instantly (the gate starts closed at the new
    /// `reduction`); afterwards every change is faded.
    pub fn set_params(&mut self, params: &GateParams) {
        self.params = params.clamped();
        self.update_coefficients();
    }

    /// Current (clamped) parameters.
    pub fn params(&self) -> &GateParams {
        &self.params
    }

    /// Linear gain applied to the most recently processed frame (1 = fully open).
    pub fn current_gain(&self) -> f64 {
        self.fade.value()
    }

    /// Whether the gate state machine is currently open (including during hold).
    pub fn is_open(&self) -> bool {
        self.state.is_open()
    }

    fn update_coefficients(&mut self) {
        let p = &self.params;
        let fs = self.sample_rate;
        self.state.set_thresholds(
            db_to_lin(p.open_threshold_db as f64),
            db_to_lin(p.close_threshold_db as f64),
        );
        self.state
            .set_hold_frames((p.hold_ms as f64 * 0.001 * fs).round() as u64);
        self.reduction_lin = db_to_lin(p.reduction_db as f64);
        self.fade
            .set_times(p.attack_ms as f64, p.release_ms as f64, fs);
        self.detector.set_release(DETECTOR_RELEASE_MS, fs);
        if self.fresh {
            self.fade.set_value(self.reduction_lin);
        }
    }
}

impl Processor for Gate {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        self.sample_rate = (sample_rate as f64).clamp(1.0, 1.0e6);
        self.channels = channels;
        self.update_coefficients();
        self.reset();
    }

    fn process(&mut self, samples: &mut [f32]) {
        let ch = self.channels;
        if ch == 0 || samples.len() < ch {
            return;
        }
        self.fresh = false;
        let closed = self.reduction_lin;
        let mut gains = [0.0f32; BLOCK_FRAMES];
        for block in samples.chunks_mut(BLOCK_FRAMES * ch) {
            let gains = &mut gains[..block.len() / ch];
            // Per-frame linked peaks, then the serial detector → state → fade
            // chain turns each frame's peak into that frame's gain.
            frame_peaks(block, ch, gains);
            for v in gains.iter_mut() {
                let env = self.detector.step(*v as f64);
                let target = if self.state.step(env) { 1.0 } else { closed };
                *v = self.fade.step(target) as f32;
            }
            apply_gains(block, ch, gains);
        }
    }

    fn reset(&mut self) {
        self.detector.reset();
        self.state.reset();
        self.fade.set_value(self.reduction_lin);
        self.fresh = true;
    }
}
