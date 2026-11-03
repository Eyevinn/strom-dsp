//! Downward noise gate with hysteresis, hold and attack/release fades.
//!
//! # Signal flow (per frame of interleaved samples)
//!
//! 1. **Key level**: the largest absolute sample value over all channels of the
//!    frame (channels are linked, so one gain is shared by all of them).
//! 2. **Detector**: a peak envelope follower with instantaneous attack and a fixed
//!    exponential release of [`DETECTOR_RELEASE_MS`]:
//!    `e[n] = p[n]` if `p[n] > e[n-1]`, otherwise `e[n] = a_d·e[n-1] + (1-a_d)·p[n]`
//!    with `a_d = exp(-1 / (τ_d·fs))`, `τ_d = DETECTOR_RELEASE_MS / 1000`. For a steady
//!    sine the envelope sits at the sine's peak amplitude (ripple well below 1 dB above
//!    ~200 Hz), so the thresholds compare against the **peak level in dBFS**.
//! 3. **State machine with hysteresis and hold**: a closed gate opens when
//!    `e > open_threshold`. An open gate re-arms its hold counter
//!    (`round(hold_ms·fs/1000)` frames) every frame where `e ≥ close_threshold`;
//!    once `e` is below the close threshold the counter runs down, and when it is
//!    exhausted the gate closes. If `close_threshold_db > open_threshold_db` the
//!    close threshold is treated as equal to the open threshold.
//! 4. **Gain fade**: the linear gain `g` follows its target (1 when open, the
//!    linear `reduction` when closed) with a one-pole smoother,
//!    `g[n] = g[n-1] + (1 - a)·(target - g[n-1])`, where `a = exp(-1 / (τ·fs))`,
//!    `τ = attack_ms/1000` while the gain rises and `τ = release_ms/1000` while it
//!    falls. So **attack/release is the time for the linear gain to cover
//!    `1 - 1/e` (≈ 63.2 %) of the distance to its target** after a step.
//!
//! Parameter changes are click-free: a changed `reduction` (or a state change from
//! a changed threshold) only moves the gain target, and the gain always reaches it
//! through the attack/release smoother.
//!
//! Denormals: the detector envelope is passed through
//! [`flush_denormal`](crate::util::flush_denormal); the gain is snapped to its
//! target once it is within `1e-12` (targets are ≥ 1e-4, so it never decays toward 0).

use crate::util::{db_to_lin, flush_denormal, one_pole_coef};
use crate::Processor;

/// Release time constant (ms) of the gate's internal peak detector (attack is instantaneous).
pub const DETECTOR_RELEASE_MS: f64 = 20.0;

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

    // Derived coefficients.
    open_lin: f64,
    close_lin: f64,
    reduction_lin: f64,
    coef_attack: f64,
    coef_release: f64,
    coef_detector: f64,
    hold_frames: u64,

    // Signal state.
    env: f64,
    gain: f64,
    open: bool,
    hold_left: u64,
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
            open_lin: 0.0,
            close_lin: 0.0,
            reduction_lin: 0.0,
            coef_attack: 0.0,
            coef_release: 0.0,
            coef_detector: 0.0,
            hold_frames: 0,
            env: 0.0,
            gain: 0.0,
            open: false,
            hold_left: 0,
        };
        g.update_coefficients();
        g.reset();
        g
    }

    /// Update parameters (clamped to their ranges). Never allocates.
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
        self.gain
    }

    /// Whether the gate state machine is currently open (including during hold).
    pub fn is_open(&self) -> bool {
        self.open
    }

    fn update_coefficients(&mut self) {
        let p = &self.params;
        let open_db = p.open_threshold_db as f64;
        let close_db = (p.close_threshold_db as f64).min(open_db);
        self.open_lin = db_to_lin(open_db);
        self.close_lin = db_to_lin(close_db);
        self.reduction_lin = db_to_lin(p.reduction_db as f64);
        let fs = self.sample_rate;
        self.coef_attack = one_pole_coef(p.attack_ms as f64, fs);
        self.coef_release = one_pole_coef(p.release_ms as f64, fs);
        self.coef_detector = one_pole_coef(DETECTOR_RELEASE_MS, fs);
        self.hold_frames = (p.hold_ms as f64 * 0.001 * fs).round() as u64;
        self.hold_left = self.hold_left.min(self.hold_frames);
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
        if self.channels == 0 {
            return;
        }
        for frame in samples.chunks_exact_mut(self.channels) {
            // Linked key level: max |x| over channels.
            let mut peak = 0.0f32;
            for s in frame.iter() {
                peak = peak.max(s.abs());
            }
            let peak = peak as f64;

            // Peak detector, instantaneous attack, exponential release.
            self.env = if peak > self.env {
                peak
            } else {
                flush_denormal(self.coef_detector * self.env + (1.0 - self.coef_detector) * peak)
            };

            // Hysteresis + hold state machine.
            if self.open {
                if self.env >= self.close_lin {
                    self.hold_left = self.hold_frames;
                } else if self.hold_left > 0 {
                    self.hold_left -= 1;
                } else {
                    self.open = false;
                }
            } else if self.env > self.open_lin {
                self.open = true;
                self.hold_left = self.hold_frames;
            }

            // Gain fade.
            let target = if self.open { 1.0 } else { self.reduction_lin };
            let diff = target - self.gain;
            if diff.abs() < 1e-12 {
                self.gain = target;
            } else {
                let a = if diff > 0.0 {
                    self.coef_attack
                } else {
                    self.coef_release
                };
                self.gain += (1.0 - a) * diff;
            }

            let g = self.gain;
            for s in frame.iter_mut() {
                *s = (*s as f64 * g) as f32;
            }
        }
    }

    fn reset(&mut self) {
        self.env = 0.0;
        self.open = false;
        self.hold_left = 0;
        self.gain = self.reduction_lin;
    }
}
