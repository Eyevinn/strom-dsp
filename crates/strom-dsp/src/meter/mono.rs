//! Mono / stereo / polarity-inverted detection with hysteresis.

use super::correlation::CorrelationMeter;

/// Frames between two decisions (the averages are still updated per sample).
const DECISION_FRAMES: usize = 32;

/// What a [`MonoDetector`] reports for a stereo pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StereoState {
    /// L ≈ R: the two channels carry the same signal.
    Mono,
    /// Anything that is neither mono nor inverted.
    Stereo,
    /// L ≈ −R: mono with one channel's polarity inverted.
    Inverted,
}

/// Thresholds and timing of a [`MonoDetector`]. Ratios in dB, times in ms.
///
/// A state is **entered** when both its *enter* conditions hold for `hold_ms`
/// without interruption, and **kept** while both its (looser) *exit*
/// conditions hold. For `Mono` the conditions are on the side/mid ratio
/// `10·log10(⟨S²⟩/⟨M²⟩)` and the correlation `r`; `Inverted` uses the same
/// thresholds on the mid/side ratio and `−r`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonoDetectorParams {
    /// Averaging time of the correlation/energy estimates (ms).
    pub integration_ms: f32,
    /// Side/mid ratio at or below which mono can be entered (dB).
    pub enter_ratio_db: f32,
    /// Side/mid ratio above which mono is left (dB, ≥ `enter_ratio_db`).
    pub exit_ratio_db: f32,
    /// Correlation at or above which mono can be entered.
    pub enter_correlation: f32,
    /// Correlation below which mono is left (≤ `enter_correlation`).
    pub exit_correlation: f32,
    /// How long a new state must be indicated before it is reported (ms).
    pub hold_ms: f32,
    /// Below this level (dBFS, mean square of L and R), the state is frozen:
    /// silence and low noise never change the decision.
    pub silence_db: f32,
}

impl Default for MonoDetectorParams {
    fn default() -> Self {
        Self {
            integration_ms: 300.0,
            enter_ratio_db: -25.0,
            exit_ratio_db: -15.0,
            enter_correlation: 0.95,
            exit_correlation: 0.85,
            hold_ms: 500.0,
            silence_db: -60.0,
        }
    }
}

/// Classifies a stereo pair as mono, stereo or polarity-inverted, from the
/// correlation and the side/mid energy ratio of a [`CorrelationMeter`], with
/// two-threshold hysteresis and a hold time so that signals near a threshold
/// do not make the state flap. Starts in [`StereoState::Stereo`].
///
/// Requiring *both* the correlation and the energy ratio means that a mono
/// source panned off-centre (r = 1, but unequal levels, so the side energy is
/// not small) is reported as stereo: only `L ≈ R` counts as mono.
///
/// Real-time safe: no allocation after `new`.
#[derive(Debug, Clone)]
pub struct MonoDetector {
    meter: CorrelationMeter,
    params: MonoDetectorParams,
    sample_rate: f32,
    hold_frames: usize,
    silence_power: f64,
    state: StereoState,
    pending: StereoState,
    pending_frames: usize,
    since_decision: usize,
}

impl MonoDetector {
    /// Create a detector at `sample_rate` Hz.
    pub fn new(sample_rate: f32, params: &MonoDetectorParams) -> Self {
        let mut d = Self {
            meter: CorrelationMeter::new(sample_rate, params.integration_ms),
            params: *params,
            sample_rate: sample_rate.max(1.0),
            hold_frames: 0,
            silence_power: 0.0,
            state: StereoState::Stereo,
            pending: StereoState::Stereo,
            pending_frames: 0,
            since_decision: 0,
        };
        d.set_params(params);
        d
    }

    /// Change the parameters (keeps the current state). Exit thresholds are
    /// clamped so they are never stricter than the enter thresholds.
    pub fn set_params(&mut self, params: &MonoDetectorParams) {
        let mut p = *params;
        p.exit_ratio_db = p.exit_ratio_db.max(p.enter_ratio_db);
        p.exit_correlation = p.exit_correlation.min(p.enter_correlation);
        p.hold_ms = p.hold_ms.clamp(0.0, 60_000.0);
        self.params = p;
        self.meter.set_integration_ms(p.integration_ms);
        self.hold_frames = (p.hold_ms as f64 * 0.001 * self.sample_rate as f64).round() as usize;
        self.silence_power = 10f64.powf(p.silence_db as f64 / 10.0);
    }

    /// Current parameters.
    pub fn params(&self) -> &MonoDetectorParams {
        &self.params
    }

    /// The reported state.
    pub fn state(&self) -> StereoState {
        self.state
    }

    /// The underlying correlation meter (correlation, side/mid ratio).
    pub fn meter(&self) -> &CorrelationMeter {
        &self.meter
    }

    /// Back to [`StereoState::Stereo`] with cleared averages.
    pub fn reset(&mut self) {
        self.meter.reset();
        self.state = StereoState::Stereo;
        self.pending = StereoState::Stereo;
        self.pending_frames = 0;
        self.since_decision = 0;
    }

    /// Add interleaved frames (`channels` ≥ 2; channel 0 = L, channel 1 = R).
    pub fn process_interleaved(&mut self, samples: &[f32], channels: usize) {
        if channels < 2 {
            return;
        }
        for f in samples.chunks_exact(channels) {
            self.meter.push(f[0], f[1]);
            self.tick();
        }
    }

    /// Add two planar channel slices (the shorter length is used).
    pub fn process_pair(&mut self, left: &[f32], right: &[f32]) {
        for (&l, &r) in left.iter().zip(right) {
            self.meter.push(l, r);
            self.tick();
        }
    }

    #[inline]
    fn tick(&mut self) {
        self.since_decision += 1;
        if self.since_decision >= DECISION_FRAMES {
            let n = self.since_decision;
            self.since_decision = 0;
            self.decide(n);
        }
    }

    fn qualifies(&self, state: StereoState, strict: bool) -> bool {
        let p = &self.params;
        let (ratio, corr) = if strict {
            (p.enter_ratio_db as f64, p.enter_correlation as f64)
        } else {
            (p.exit_ratio_db as f64, p.exit_correlation as f64)
        };
        let sm = self.meter.side_mid_ratio_db();
        let r = self.meter.correlation();
        match state {
            StereoState::Mono => sm <= ratio && r >= corr,
            StereoState::Inverted => -sm <= ratio && -r >= corr,
            StereoState::Stereo => true,
        }
    }

    fn decide(&mut self, frames: usize) {
        if self.meter.mean_square() < self.silence_power {
            // Frozen during silence; a pending change has to start over.
            self.pending = self.state;
            self.pending_frames = 0;
            return;
        }
        let target = if self.state != StereoState::Stereo && self.qualifies(self.state, false) {
            self.state
        } else if self.qualifies(StereoState::Mono, true) {
            StereoState::Mono
        } else if self.qualifies(StereoState::Inverted, true) {
            StereoState::Inverted
        } else {
            StereoState::Stereo
        };
        if target == self.state {
            self.pending = target;
            self.pending_frames = 0;
            return;
        }
        if target != self.pending {
            self.pending = target;
            self.pending_frames = 0;
        }
        self.pending_frames += frames;
        if self.pending_frames >= self.hold_frames {
            self.state = target;
            self.pending_frames = 0;
        }
    }
}
