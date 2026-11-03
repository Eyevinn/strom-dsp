//! Recursive one-pole smoothers and the peak envelope follower.
//!
//! Reference: J. O. Smith III, *Introduction to Digital Filters*, one-pole
//! low-pass; Giannoulis, Massberg & Reiss (JAES 2012), level detection and
//! attack/release smoothing ("branching" one-pole).

use crate::util::{flush_denormal, one_pole_coef, DENORMAL_FLUSH};

/// One-pole smoother `y[n] = y[n−1] + (1 − a)·(x[n] − y[n−1])`, the impulse-invariant
/// discretisation of an RC low-pass, `a = exp(−1/(τ·fs))`.
///
/// Units are whatever the caller feeds (linear gain, dB, Hz, …); the smoother is
/// linear in its input. The step response covers `1 − 1/e` of a step after `τ`.
///
/// When the distance to the target falls below the *snap* distance the value
/// jumps exactly onto the target, so it settles in finite time and a decaying
/// state never enters the subnormal range (the default snap is
/// [`DENORMAL_FLUSH`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnePole {
    /// `1 − a`.
    k: f64,
    value: f64,
    snap: f64,
}

impl Default for OnePole {
    fn default() -> Self {
        Self {
            k: 1.0,
            value: 0.0,
            snap: DENORMAL_FLUSH,
        }
    }
}

impl OnePole {
    /// Smoother with time constant `time_ms` at `sample_rate` Hz, starting at `initial`.
    /// `time_ms ≤ 0` is instantaneous.
    pub fn new(time_ms: f64, sample_rate: f64, initial: f64) -> Self {
        let mut s = Self {
            value: initial,
            ..Self::default()
        };
        s.set_time(time_ms, sample_rate);
        s
    }

    /// Builder: set the snap distance (≥ 0).
    pub fn with_snap(mut self, snap: f64) -> Self {
        self.snap = snap.max(0.0);
        self
    }

    /// Change the time constant (ms). The current value is kept.
    pub fn set_time(&mut self, time_ms: f64, sample_rate: f64) {
        self.k = 1.0 - one_pole_coef(time_ms, sample_rate);
    }

    /// Current value.
    #[inline]
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Jump to `value` immediately.
    #[inline]
    pub fn set_value(&mut self, value: f64) {
        self.value = value;
    }

    /// Advance one sample toward `target` and return the new value.
    #[inline]
    pub fn step(&mut self, target: f64) -> f64 {
        let d = target - self.value;
        if d.abs() < self.snap {
            self.value = target;
        } else {
            self.value += self.k * d;
        }
        self.value
    }
}

/// One-pole smoother with separate time constants for rising and falling input
/// (the "branching" smoother of Giannoulis et al.): the attack coefficient is
/// used while the target is above the current value, the release coefficient
/// otherwise.
///
/// Feed it gain reduction in dB for a compressor (rising reduction = attack),
/// or linear gain for a gate (rising gain = opening = attack). Snapping as in
/// [`OnePole`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttackRelease {
    k_attack: f64,
    k_release: f64,
    value: f64,
    snap: f64,
}

impl Default for AttackRelease {
    fn default() -> Self {
        Self {
            k_attack: 1.0,
            k_release: 1.0,
            value: 0.0,
            snap: DENORMAL_FLUSH,
        }
    }
}

impl AttackRelease {
    /// Smoother with attack/release time constants (ms) at `sample_rate` Hz, starting at 0.
    pub fn new(attack_ms: f64, release_ms: f64, sample_rate: f64) -> Self {
        let mut s = Self::default();
        s.set_times(attack_ms, release_ms, sample_rate);
        s
    }

    /// Builder: set the snap distance (≥ 0).
    pub fn with_snap(mut self, snap: f64) -> Self {
        self.snap = snap.max(0.0);
        self
    }

    /// Change both time constants (ms). The current value is kept.
    pub fn set_times(&mut self, attack_ms: f64, release_ms: f64, sample_rate: f64) {
        self.k_attack = 1.0 - one_pole_coef(attack_ms, sample_rate);
        self.k_release = 1.0 - one_pole_coef(release_ms, sample_rate);
    }

    /// Current value.
    #[inline]
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Jump to `value` immediately.
    #[inline]
    pub fn set_value(&mut self, value: f64) {
        self.value = value;
    }

    /// Advance one sample toward `target` and return the new value.
    #[inline]
    pub fn step(&mut self, target: f64) -> f64 {
        // Branchy on purpose: once settled, the predicted snap branch makes
        // the next value independent of the previous one (fast for a gate
        // that sits at its target most of the time).
        let d = target - self.value;
        if d.abs() < self.snap {
            self.value = target;
        } else {
            let k = if d > 0.0 {
                self.k_attack
            } else {
                self.k_release
            };
            self.value += k * d;
        }
        self.value
    }

    /// The same smoother over a block, for signals that keep moving (a
    /// compressor's gain reduction): each element of `buf` is a target on
    /// input and the smoothed value after that step on output.
    ///
    /// It evaluates the one-pole as `a·y + k·x` (`a = 1 − k`) instead of
    /// `y + k·(x − y)`, which shortens the loop-carried dependency, so results
    /// may differ from [`step`](Self::step) in the last bits. Deterministic:
    /// plain IEEE multiply/add/compare, no fused multiply-add.
    #[inline]
    pub fn step_block(&mut self, buf: &mut [f64]) {
        let k = [self.k_attack, self.k_release];
        let a = [1.0 - k[0], 1.0 - k[1]];
        let snap = self.snap;
        let mut y = self.value;
        for v in buf {
            y = Self::next(y, *v, k, a, snap);
            *v = y;
        }
        self.value = y;
    }

    /// One step, `y ← a·y + k·x` with `(a, k)` = attack while `x > y`,
    /// release otherwise (`a = 1 − k`), snapping onto `x` when within `snap`.
    ///
    /// This form keeps the loop-carried dependency short (multiply, add,
    /// select): `k·x` and the direction test do not wait on the previous
    /// step's arithmetic, and both candidates are computed before selecting,
    /// so there is no data-dependent branch on noisy material.
    #[inline(always)]
    fn next(y: f64, x: f64, k: [f64; 2], a: [f64; 2], snap: f64) -> f64 {
        let up = y * a[0] + k[0] * x;
        let down = y * a[1] + k[1] * x;
        let stepped = if x > y { up } else { down };
        if (x - y).abs() < snap {
            x
        } else {
            stepped
        }
    }
}

/// Peak envelope follower with instantaneous attack and exponential release:
/// `e[n] = p[n]` if `p[n] > e[n−1]`, else `e[n] = a·e[n−1] + (1−a)·p[n]`.
///
/// Input is a non-negative level (typically linear peak amplitude). For a steady
/// sine it sits at the peak amplitude with a small ripple that shrinks as the
/// release grows relative to the period. The state is denormal-flushed.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PeakEnvelope {
    a: f64,
    value: f64,
}

impl PeakEnvelope {
    /// Follower with release time constant `release_ms` at `sample_rate` Hz, starting at 0.
    pub fn new(release_ms: f64, sample_rate: f64) -> Self {
        Self {
            a: one_pole_coef(release_ms, sample_rate),
            value: 0.0,
        }
    }

    /// Change the release time constant (ms). The current value is kept.
    pub fn set_release(&mut self, release_ms: f64, sample_rate: f64) {
        self.a = one_pole_coef(release_ms, sample_rate);
    }

    /// Current envelope value.
    #[inline]
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Clear to 0.
    pub fn reset(&mut self) {
        self.value = 0.0;
    }

    /// Feed one level sample and return the new envelope.
    #[inline]
    pub fn step(&mut self, level: f64) -> f64 {
        self.value = if level > self.value {
            level
        } else {
            flush_denormal(self.a * self.value + (1.0 - self.a) * level)
        };
        self.value
    }
}

/// Mean-square (RMS) envelope: a one-pole average of the squared input,
/// `m[n] = a·m[n−1] + (1−a)·x[n]²`, `a = exp(−1/(τ·fs))`.
///
/// [`step`](Self::step) returns the **sine-calibrated power** `2·m`, so that
/// `sqrt(2·m)` of a steady sine equals its peak amplitude (+3.01 dB over the
/// plain RMS); a constant `|x| = c` reads `√2·c`. Compare it with squared
/// linear thresholds. After the input stops, `sqrt(2·m)` decays with time
/// constant `2τ`. The state is denormal-flushed.
///
/// Unlike [`PeakEnvelope`], single peaks do not move it much: a level must be
/// sustained over about `τ` to register (J. O. Smith III, one-pole averager;
/// Zölzer, *DAFX*, RMS measurement in the dynamics chapter).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MeanSquareEnvelope {
    a: f64,
    value: f64,
}

impl MeanSquareEnvelope {
    /// Envelope with averaging time constant `time_ms` at `sample_rate` Hz, starting at 0.
    pub fn new(time_ms: f64, sample_rate: f64) -> Self {
        Self {
            a: one_pole_coef(time_ms, sample_rate),
            value: 0.0,
        }
    }

    /// Change the averaging time constant (ms). The current value is kept.
    pub fn set_time(&mut self, time_ms: f64, sample_rate: f64) {
        self.a = one_pole_coef(time_ms, sample_rate);
    }

    /// Current sine-calibrated power `2·m`.
    #[inline]
    pub fn value(&self) -> f64 {
        2.0 * self.value
    }

    /// Clear to 0.
    pub fn reset(&mut self) {
        self.value = 0.0;
    }

    /// Feed one amplitude sample (any sign) and return the new sine-calibrated
    /// power `2·m`. NaN input is treated as 0; ±∞ saturates at `f64::MAX`.
    #[inline]
    pub fn step(&mut self, x: f64) -> f64 {
        let sq = if x.is_nan() {
            0.0
        } else {
            (x * x).min(f64::MAX)
        };
        self.value = flush_denormal(self.a * self.value + (1.0 - self.a) * sq);
        2.0 * self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 48_000.0;

    #[test]
    fn mean_square_reads_sine_peak_and_decays_with_two_tau() {
        let mut e = MeanSquareEnvelope::new(10.0, FS);
        let mut v = 0.0;
        for n in 0..48_000 {
            v = e.step(0.5 * (2.0 * std::f64::consts::PI * 1000.0 * n as f64 / FS).sin());
        }
        assert!((v.sqrt() / 0.5 - 1.0).abs() < 0.01, "{}", v.sqrt());
        let start = e.value();
        for _ in 0..960 {
            e.step(0.0);
        }
        // Amplitude falls by 1/e after 2τ = 20 ms.
        assert!(((e.value() / start).sqrt() - (-1.0f64).exp()).abs() < 1e-9);
        assert!(e.step(f64::NAN).is_finite());
        assert!(e.step(f64::INFINITY).is_finite());
    }

    #[test]
    fn one_pole_reaches_63_percent_after_tau() {
        let mut s = OnePole::new(10.0, FS, 0.0);
        for _ in 0..480 {
            s.step(1.0);
        }
        assert!((s.value() - (1.0 - (-1.0f64).exp())).abs() < 1e-3);
    }

    #[test]
    fn one_pole_zero_time_is_instant_and_snaps() {
        let mut s = OnePole::new(0.0, FS, 3.0);
        assert_eq!(s.step(-2.0), -2.0);
        let mut s = OnePole::new(1.0, FS, 1.0).with_snap(1e-6);
        for _ in 0..10_000 {
            s.step(0.0);
        }
        assert_eq!(s.value(), 0.0, "decay must snap to exactly 0");
    }

    #[test]
    fn attack_release_branches_on_direction() {
        // Up with 1 ms, down with 100 ms: after 1 ms, up covers 63 %, down ≈ 1 %.
        let mut up = AttackRelease::new(1.0, 100.0, FS);
        let mut down = AttackRelease::new(1.0, 100.0, FS);
        down.set_value(1.0);
        for _ in 0..48 {
            up.step(1.0);
            down.step(0.0);
        }
        let e = 1.0 - (-1.0f64).exp();
        assert!((up.value() - e).abs() < 1e-2);
        let expected_down = (-48.0 / 4800.0f64).exp();
        assert!((down.value() - expected_down).abs() < 1e-3);
    }

    #[test]
    fn peak_envelope_instant_attack_exponential_release() {
        let mut e = PeakEnvelope::new(20.0, FS);
        assert_eq!(e.step(0.5), 0.5);
        assert_eq!(e.step(0.8), 0.8);
        // Release: after τ = 20 ms of silence, e = 0.8/e.
        for _ in 0..960 {
            e.step(0.0);
        }
        assert!((e.value() - 0.8 * (-1.0f64).exp()).abs() < 1e-3);
        // Eventually flushes to exactly zero.
        for _ in 0..200_000 {
            e.step(0.0);
        }
        assert_eq!(e.value(), 0.0);
    }

    #[test]
    fn peak_envelope_holds_sine_peak() {
        let mut e = PeakEnvelope::new(20.0, FS);
        let mut lo = f64::INFINITY;
        for n in 0..48_000 {
            let x = (2.0 * std::f64::consts::PI * 1000.0 * n as f64 / FS)
                .sin()
                .abs();
            let v = e.step(x);
            if n > 24_000 {
                lo = lo.min(v);
            }
        }
        // Ripple over one half period (24 samples) of a 1 kHz sine.
        assert!(lo > (-24.0 / 960.0f64).exp() - 1e-3);
    }
}
