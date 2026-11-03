//! Bass clipper: lookahead gain control of the lowest multiband band from
//! its analytic-signal envelope, so a loud bass is held down without
//! generating harmonics.
//!
//! References: expired US 4,495,643 (gain control of a low band from its
//! envelope, without generating harmonics); expired US 5,737,434
//! (look-ahead, band-limited gain control signal); expired US 5,737,432
//! (lowering the bass clip level when bass and the upper clipper coincide)
//! — all "expired; status not legally reviewed". S. Darlington (1950) and
//! the all-pass phase-difference literature for the envelope
//! ([`crate::filter::hilbert`]). The gain pipeline reuses the limiter's
//! gain core (sliding minimum, cascaded-box ramp, one-pole release; see
//! [`crate::limiter`]).
//!
//! # Signal flow (per frame of the band signal)
//!
//! ```text
//! band ──┬──────────────── delay L ───────────────── × g ──→ out
//!        └→ analytic magnitude → required gain → min over L+1 → ramp L → release ─┘
//! ```
//!
//! 1. **Envelope.** `env = max_c √(a_c² + b_c²)`, the analytic magnitude of
//!    the band from the all-pass pair ([`HilbertPair`]), linked over the
//!    channels. For a steady sine it is flat (its amplitude), so the gain on
//!    it is constant and adds no harmonics. The pair is in the control path
//!    only: its phase never reaches the audio.
//! 2. **Required gain** `r = min(1, T/env)` with the clip threshold `T`
//!    (linear) lowered by the IM feedback below.
//! 3. **Lookahead** (the limiter's construction with attack = lookahead
//!    `L`, no hold): the audio is delayed by `L` frames; the minimum of `r`
//!    over the last `L + 1` frames ([`SegmentedMin`]) goes through the
//!    cascaded-box ramp of length `L` ([`RampSmoother`]). The gain then
//!    reaches the required gain of a frame exactly when that frame leaves
//!    the delay, starting `L` frames earlier, along a smooth S-shaped ramp.
//!    With a lookahead shorter than [`BASS_MIN_RAMP_MS`] (down to none) the
//!    ramp keeps that length and the gain arrives late by the difference.
//! 4. **Release**: the gain follows a falling ramp at once and rises toward
//!    it with a one-pole of [`BASS_CLIP_RELEASE_MS`].
//!
//! **Band-limited gain.** The gain only moves along the cascaded-box ramp
//! (a low-pass whose impulse response is the `L`-frame triangle-like
//! kernel: magnitude `|sinc|²`, first zero at `2/L`, 2 kHz for 1 ms) and the
//! release one-pole (cut-off `1/(2π·τ)`, 3.2 Hz for 50 ms). On a steady
//! bass the gain is constant. See the element page for the limits of this at
//! the short lookaheads the broadcast chain uses (1 ms, or none).
//!
//! # IM feedback
//!
//! A slow control value `v` (dB, `0 … `[`BASS_IM_MAX_DB`]) lowers the
//! threshold: `T_eff = T − v`. Per frame, while the band envelope is within
//! [`BASS_PEAK_MARGIN_DB`] of the threshold (a bass peak) and the upper
//! (final) clipper reports activity `a` (`0 … 1`, see
//! [`BassClipper::set_upper_activity`]), `v` moves toward
//! `BASS_IM_MAX_DB · min(1, a / `[`BASS_IM_FULL_ACTIVITY`]`)` with
//! [`BASS_IM_ATTACK_MS`]; otherwise it returns toward 0 with
//! [`BASS_IM_RELEASE_MS`]. So the bass is clipped a little harder only while
//! it coincides with the upper clipper working, and the threshold comes
//! back when the coincidence stops. With no activity input (the default,
//! 0) `v` stays 0.
//!
//! # Enable and bypass
//!
//! `enabled` crossfades (raised cosine, [`BASS_CLIP_FADE_MS`]) between the
//! controlled and the plainly delayed band. The delay is always there, so
//! the latency does not change. Once faded out, the control path stops and
//! the output is the band delayed by `L`, bit for bit; it restarts from a
//! clean state when enabled again.
//!
//! # Real-time safety
//!
//! [`BassClipper::configure`] allocates (delay, window, ramp, all-pass
//! state). `set_params`, `set_upper_activity`, `process` and `reset` never
//! allocate.

use crate::dynamics::{DelayLine, RampSmoother, SegmentedMin};
use crate::filter::HilbertPair;
use crate::util::{db_to_lin, lin_to_db, one_pole_coef};

/// Range of the clip threshold (dB, analytic envelope of the band, which
/// reads a steady sine's peak).
pub const BASS_CLIP_THRESHOLD_RANGE_DB: (f32, f32) = (-40.0, 0.0);
/// Release time constant of the gain (ms).
pub const BASS_CLIP_RELEASE_MS: f64 = 50.0;
/// Time constant (ms) of each of the two cascaded one-poles through which
/// a threshold change glides (critically damped, starts with zero slope).
/// It keeps a threshold change click-free with any lookahead, including
/// none, where the gain would otherwise follow the new threshold at once.
pub const BASS_THRESHOLD_GLIDE_MS: f64 = 5.0;
/// Shortest gain ramp (ms). With a lookahead at least this long the ramp
/// equals the lookahead (the gain arrives exactly when a frame leaves the
/// delay); with a shorter one, down to none, the ramp keeps this length and
/// the gain arrives late by the difference (an attack time) instead of
/// stepping: a gain step on a bass tone is a click.
pub const BASS_MIN_RAMP_MS: f64 = 0.5;
/// Crossfade time of `enabled` (ms, raised cosine).
pub const BASS_CLIP_FADE_MS: f64 = 10.0;
/// Largest threshold reduction of the IM feedback (dB): the documented
/// floor of the bass threshold is `T − 3 dB`.
pub const BASS_IM_MAX_DB: f64 = 3.0;
/// Time constant of the IM feedback lowering the threshold (ms).
pub const BASS_IM_ATTACK_MS: f64 = 30.0;
/// Time constant of the IM feedback releasing the threshold (ms).
pub const BASS_IM_RELEASE_MS: f64 = 1000.0;
/// A frame counts as a bass peak while the envelope is within this many dB
/// of the (nominal) threshold.
pub const BASS_PEAK_MARGIN_DB: f64 = 6.0;
/// Upper-clipper activity at which the IM feedback asks for its full
/// threshold reduction (1 % of samples touched). A high gain: the threshold
/// goes down until the coincidence nearly stops, bounded by
/// [`BASS_IM_MAX_DB`].
pub const BASS_IM_FULL_ACTIVITY: f64 = 0.01;
/// [`BASS_PEAK_MARGIN_DB`] as a linear factor.
const PEAK_MARGIN: f64 = 0.501_187_233_627_272_2;
/// Lowest frequency (Hz) of the envelope's design band; the band spans 600:1
/// in pre-warped frequency above it (8 Hz … about 4.6 kHz at 48 kHz).
pub const ENVELOPE_LOW_HZ: f64 = 8.0;

/// What the gain control reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BassEnvelope {
    /// The analytic-signal magnitude (the design).
    #[default]
    Analytic,
    /// The rectified waveform `max_c |x_c|`. Only a reference for tests: it
    /// ripples at twice the bass frequency, so the gain does too and adds
    /// harmonics.
    Rectified,
}

/// Bass-clipper parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BassClipperParams {
    /// Control on (crossfaded; the delay stays either way).
    pub enabled: bool,
    /// Clip threshold (dB, −40 … 0) against the band's analytic envelope.
    pub threshold_db: f32,
}

impl Default for BassClipperParams {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold_db: -6.0,
        }
    }
}

impl BassClipperParams {
    /// Copy with the threshold clamped (NaN becomes the default).
    pub fn clamped(&self) -> Self {
        let (lo, hi) = BASS_CLIP_THRESHOLD_RANGE_DB;
        Self {
            enabled: self.enabled,
            threshold_db: if self.threshold_db.is_nan() {
                Self::default().threshold_db
            } else {
                self.threshold_db.clamp(lo, hi)
            },
        }
    }
}

/// The bass clipper. See the module docs.
#[derive(Debug, Clone, Default)]
pub struct BassClipper {
    params: BassClipperParams,
    /// Enabled by the host's structure (e.g. at least two bands).
    allowed: bool,
    channels: usize,
    lookahead: usize,
    /// Length of the gain ramp: `max(lookahead, BASS_MIN_RAMP_MS)`.
    ramp_len: usize,
    envelope: HilbertPair,
    window: SegmentedMin,
    ramp: RampSmoother,
    delay: DelayLine,
    /// Gain after the release stage.
    gain: f64,
    k_release: f64,
    /// Fade position, 0 (off) … `fade_len` (on).
    fade_pos: usize,
    fade_len: usize,
    /// Control path running (on, or fading).
    running: bool,
    /// IM feedback value (dB, ≥ 0).
    im_db: f64,
    k_im_up: f64,
    k_im_down: f64,
    activity: f64,
    /// Nominal threshold (linear): its target, the first glide stage and
    /// the value in force.
    threshold_target: f64,
    threshold_mid: f64,
    threshold: f64,
    k_glide: f64,
    /// Smallest applied gain since the last `take_gain_reduction_db`.
    min_gain: f64,
    envelope_kind: BassEnvelope,
}

impl BassClipper {
    /// An unconfigured clipper (passes audio through until configured).
    pub fn new() -> Self {
        let mut b = Self {
            gain: 1.0,
            min_gain: 1.0,
            ..Self::default()
        };
        b.set_params(&BassClipperParams::default());
        b
    }

    /// Configure for `sample_rate`, `channels` and a lookahead of
    /// `lookahead` frames (the latency). Allocates; resets the state.
    pub fn configure(&mut self, sample_rate: f32, channels: usize, lookahead: usize) {
        let fs = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate as f64
        } else {
            48_000.0
        };
        self.channels = channels.max(1);
        self.lookahead = lookahead;
        self.envelope.configure(fs, self.channels, ENVELOPE_LOW_HZ);
        self.window.allocate(lookahead + 1);
        self.ramp_len = lookahead.max((BASS_MIN_RAMP_MS * 0.001 * fs).round() as usize);
        self.ramp.allocate(self.ramp_len);
        self.delay.allocate(lookahead, self.channels);
        self.k_release = 1.0 - one_pole_coef(BASS_CLIP_RELEASE_MS, fs);
        self.k_glide = 1.0 - one_pole_coef(BASS_THRESHOLD_GLIDE_MS, fs);
        self.k_im_up = 1.0 - one_pole_coef(BASS_IM_ATTACK_MS, fs);
        self.k_im_down = 1.0 - one_pole_coef(BASS_IM_RELEASE_MS, fs);
        self.fade_len = ((BASS_CLIP_FADE_MS * 0.001 * fs).round() as usize).max(1);
        self.reset();
    }

    /// Latency (frames): the lookahead.
    pub fn latency_frames(&self) -> usize {
        self.lookahead
    }

    /// New parameters (clamped). Never allocates. A threshold change glides
    /// ([`BASS_THRESHOLD_GLIDE_MS`]) while the control runs, and applies to
    /// frames as they enter the lookahead; while the control is stopped it
    /// applies at once.
    pub fn set_params(&mut self, params: &BassClipperParams) {
        self.params = params.clamped();
        self.threshold_target = db_to_lin(self.params.threshold_db as f64);
        if !self.running {
            self.snap_threshold();
        }
    }

    fn snap_threshold(&mut self) {
        self.threshold_mid = self.threshold_target;
        self.threshold = self.threshold_target;
    }

    /// Current (clamped) parameters.
    pub fn params(&self) -> &BassClipperParams {
        &self.params
    }

    /// Select what the gain control reads (default
    /// [`BassEnvelope::Analytic`]; the other choice exists for tests).
    pub fn set_envelope(&mut self, kind: BassEnvelope) {
        self.envelope_kind = kind;
    }

    /// Allow or forbid the control by structure (the multiband allows it
    /// with two or more bands). Crossfaded like `enabled`.
    pub fn set_allowed(&mut self, allowed: bool) {
        self.allowed = allowed;
    }

    /// Activity of the upper (final) clipper in its most recent block, as
    /// the share of samples it touched (0 … 1; 0 = none, the default).
    /// Feeds the IM feedback (see the module docs). Never allocates.
    pub fn set_upper_activity(&mut self, activity: f32) {
        let a = activity as f64;
        self.activity = if a.is_finite() {
            a.clamp(0.0, 1.0)
        } else {
            0.0
        };
    }

    /// Current IM feedback (dB of threshold reduction, 0 … 3).
    pub fn im_reduction_db(&self) -> f64 {
        self.im_db
    }

    /// Gain reduction (dB, ≥ 0) of the most recent frame.
    pub fn gain_reduction_db(&self) -> f64 {
        -lin_to_db(self.applied(self.gain)).min(0.0)
    }

    /// Largest gain reduction (dB, ≥ 0) since the previous call.
    pub fn take_gain_reduction_db(&mut self) -> f64 {
        let g = self.min_gain;
        self.min_gain = self.applied(self.gain);
        -lin_to_db(g).min(0.0)
    }

    /// Forget the signal: silence in the delay, gain 1, no IM feedback;
    /// the enable fade jumps to its target.
    pub fn reset(&mut self) {
        self.envelope.reset();
        self.window.reset(self.lookahead + 1);
        self.ramp.reset(self.ramp_len, 1.0);
        self.delay.clear();
        self.gain = 1.0;
        self.im_db = 0.0;
        self.min_gain = 1.0;
        self.snap_threshold();
        let on = self.on();
        self.fade_pos = if on { self.fade_len } else { 0 };
        self.running = on;
    }

    fn on(&self) -> bool {
        self.params.enabled && self.allowed
    }

    /// The applied gain for control gain `g` at the current fade position.
    #[inline]
    fn applied(&self, g: f64) -> f64 {
        if self.fade_pos >= self.fade_len {
            g
        } else if self.fade_pos == 0 {
            1.0
        } else {
            let w = 0.5
                - 0.5 * (core::f64::consts::PI * self.fade_pos as f64 / self.fade_len as f64).cos();
            1.0 - w * (1.0 - g)
        }
    }

    /// Process interleaved band frames in place: delayed by the lookahead
    /// and multiplied by the control gain.
    pub fn process(&mut self, band: &mut [f32]) {
        let ch = self.channels;
        if ch == 0 || self.window.window() == 0 {
            return;
        }
        let on = self.on();
        if on && !self.running {
            // Restart the control path from a clean state.
            self.envelope.reset();
            self.window.reset(self.lookahead + 1);
            self.ramp.reset(self.ramp_len, 1.0);
            self.gain = 1.0;
            self.im_db = 0.0;
            self.snap_threshold();
            self.running = true;
        }
        let l = self.lookahead;
        if !self.running && l == 0 {
            return; // off and no delay: exactly the input
        }
        for frame in band.chunks_exact_mut(ch) {
            if self.running {
                let env = match self.envelope_kind {
                    BassEnvelope::Analytic => self.envelope.linked_magnitude(frame),
                    BassEnvelope::Rectified => {
                        frame.iter().fold(0.0f64, |m, &v| m.max((v as f64).abs()))
                    }
                };
                if self.threshold != self.threshold_target {
                    let k = self.k_glide;
                    self.threshold_mid += k * (self.threshold_target - self.threshold_mid);
                    self.threshold += k * (self.threshold_mid - self.threshold);
                    if (self.threshold - self.threshold_target).abs()
                        <= 1e-9 * self.threshold_target
                    {
                        self.snap_threshold();
                    }
                }
                // IM feedback: lower the threshold while bass peaks and
                // upper-clipper activity coincide.
                let target = if env >= self.threshold * PEAK_MARGIN {
                    BASS_IM_MAX_DB * (self.activity / BASS_IM_FULL_ACTIVITY).min(1.0)
                } else {
                    0.0
                };
                let k = if target > self.im_db {
                    self.k_im_up
                } else {
                    self.k_im_down
                };
                self.im_db += k * (target - self.im_db);
                if self.im_db < 1e-9 {
                    self.im_db = 0.0;
                }
                let thr = if self.im_db > 0.0 {
                    self.threshold * db_to_lin(-self.im_db)
                } else {
                    self.threshold
                };
                let r = if env > thr { thr / env } else { 1.0 };
                let h = self.window.push(r);
                let t = self.ramp.step(h);
                self.gain = if t <= self.gain {
                    t
                } else {
                    let g = self.gain + self.k_release * (t - self.gain);
                    if t - g < 1e-12 {
                        t
                    } else {
                        g
                    }
                };
                // Advance the fade.
                if on {
                    self.fade_pos = (self.fade_pos + 1).min(self.fade_len);
                } else {
                    self.fade_pos = self.fade_pos.saturating_sub(1);
                }
            }
            self.delay.write(frame);
            frame.copy_from_slice(self.delay.read(l));
            if self.running {
                let g = self.applied(self.gain);
                if g < self.min_gain {
                    self.min_gain = g;
                }
                if g != 1.0 {
                    for v in frame.iter_mut() {
                        *v = (*v as f64 * g) as f32;
                    }
                }
                if !on && self.fade_pos == 0 {
                    self.running = false;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_margin_matches_its_db_value() {
        assert!((PEAK_MARGIN - db_to_lin(-BASS_PEAK_MARGIN_DB)).abs() < 1e-15);
    }

    #[test]
    fn disabled_is_a_plain_delay() {
        let mut b = BassClipper::new();
        b.set_allowed(true);
        b.configure(48_000.0, 2, 10);
        let mut x: Vec<f32> = (0..200).map(|i| i as f32 * 0.01).collect();
        let orig = x.clone();
        b.process(&mut x);
        for n in 0..100 {
            let want = if n >= 10 {
                [orig[2 * (n - 10)], orig[2 * (n - 10) + 1]]
            } else {
                [0.0, 0.0]
            };
            assert_eq!([x[2 * n], x[2 * n + 1]], want);
        }
    }
}
