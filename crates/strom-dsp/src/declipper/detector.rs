//! Waveform clip detector: finds runs of consecutive samples on a flat plateau.
//!
//! Hard clipping replaces the top (or bottom) of the waveform by a constant:
//! consecutive samples sit at the same value, and the waveform enters and
//! leaves that value with a slope it could not have at a smooth peak. The
//! detector looks for exactly that shape, one sample at a time, on the
//! waveform itself. It never looks at how often a value occurs.
//!
//! # Run rule
//!
//! A *candidate* is a run of consecutive samples `x[s] … x[e]` that all stay
//! within `tol = tolerance · |x[s]|` of the first sample `x[s]`. When the
//! next sample leaves that band, the candidate closes and is a *clipped run*
//! when all of these hold:
//!
//! 1. **Length** `L = e − s + 1 ≥ 2`.
//! 2. **Level** `|x[s]| ≥ min_level`. Clipping happens at high levels; the
//!    floor sits low enough to cover gain applied after the clip.
//! 3. **Sign-consistent plateau with corners.** With `d = sign(x[s])`, the
//!    samples just before and after the run are both *inside* the plateau,
//!    by a clear edge that is steeper than the run's own slope at that end:
//!    `d·(x[s] − x[s−1]) ≥ edge + CORNER_RATIO·|x[s+1] − x[s]|` and
//!    `d·(x[e] − x[e+1]) ≥ edge + CORNER_RATIO·|x[e] − x[e−1]|`, with
//!    `edge = max(EDGE_RATIO · |x[s]|, 4·tol)`. A step (flat between a rise
//!    and a further rise) is not a clip. The edge also rejects the broad,
//!    quantised peak of a loud low-frequency tone: a smooth peak that stays
//!    flat to within one quantisation step over `L` samples leaves it by
//!    about one step.
//! 4. **Two-sample runs.** A flat pair is also what an unclipped band-limited
//!    peak looks like when it falls half way between two samples. A pair is
//!    accepted when
//!    - a run of length ≥ 3 with the same direction was accepted at the same
//!      level (within `tol`) during the last [`CEILING_HOLD_S`] seconds
//!      (clipping hits the same ceiling over and over), or
//!    - the waveform bends sharply at both ends of the pair: the second
//!      difference at the edge sample is at least [`CURVATURE_RATIO`] times
//!      the second difference one sample further out, on both sides. At a
//!      smooth peak the bend is about the same at both places; at a clip the
//!      slope stops abruptly. This needs one sample after the closing
//!      sample, so such a pair is reported one sample later.
//!
//! The clip level is the plateau's own value, so it adapts per run: a clip
//! below 0 dBFS (gain applied after clipping), a DC offset after clipping
//! (different levels for the top and bottom plateaus) and requantisation
//! after the gain all leave the plateau flat, and are found at their level.
//!
//! A run longer than `max_run` is reported with `too_long = true`; the
//! declipper passes it through unchanged and counts it.
//!
//! Single clipped samples (a peak that only just exceeds the clip level) have
//! no plateau and are not detected.
//!
//! # Tilted plateaus
//!
//! A high-pass after the clip (a DC blocker, AC coupling) makes the plateau
//! droop towards zero: a first-order high-pass with corner `f_c` turns a
//! flat plateau into a decaying exponential whose slope is about
//! `ω_c·|level|` per sample and whose second difference is about
//! `ω_c²·|level|`, `ω_c = 2π·f_c/fs`. With `max_tilt = ω_max > 0` a sample
//! continues a candidate when it has the candidate's sign, its step from
//! the previous sample is at most `tol + ω_max·|x[s]|`, and (from the third
//! sample on) the second difference is at most `tol + ω_max²·|x[s]|`. The
//! corner rule (3) is what separates such a run from the smooth, slowly
//! curving top of a low-frequency tone. A pair must still be flat. When a
//! tilted candidate is rejected it is re-anchored at its last sample, because
//! it can start one sample early, on the last sample of the rising edge.

/// Smallest edge into and out of a plateau, relative to its level (−60 dB).
pub const EDGE_RATIO: f32 = 1e-3;

/// How much steeper the edge into (out of) a run must be than the run's own
/// slope at that end (see the corner rule).
pub const CORNER_RATIO: f32 = 4.0;

/// For a two-sample run without a known ceiling: how much more the
/// waveform must bend at each edge of the run than one sample further out.
pub const CURVATURE_RATIO: f32 = 4.0;

/// How long an accepted run of length ≥ 3 vouches for two-sample runs at the
/// same level (seconds).
pub const CEILING_HOLD_S: f32 = 2.0;

/// Detector settings (all linear).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectorParams {
    /// Lowest plateau magnitude that counts as clipping (linear, > 0).
    pub min_level: f32,
    /// Plateau flatness, relative to the plateau level (≥ 0).
    pub tolerance: f32,
    /// Longest run that is reported as repairable (frames, ≥ 2).
    pub max_run: usize,
    /// Ceiling hold time in frames (see [`CEILING_HOLD_S`]).
    pub ceiling_hold: u64,
    /// Tilted plateaus: largest slope inside a run, relative to its level
    /// per sample (0 = flat plateaus only). See "Tilted plateaus" in the
    /// module docs.
    pub max_tilt: f32,
}

impl Default for DetectorParams {
    fn default() -> Self {
        Self {
            min_level: 0.1,
            tolerance: 1e-5,
            max_run: 96,
            ceiling_hold: (CEILING_HOLD_S * 48_000.0) as u64,
            max_tilt: 0.0,
        }
    }
}

/// One detected clipped run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipRun {
    /// Stream index of the first clipped sample.
    pub start: u64,
    /// Number of clipped samples (≥ 2).
    pub len: usize,
    /// Plateau value (the first sample of the run).
    pub level: f32,
    /// `+1` for a top plateau (the true signal was higher), `−1` for a bottom one.
    pub dir: i8,
    /// Longer than `max_run`: reported but not to be repaired.
    pub too_long: bool,
}

impl ClipRun {
    /// Stream index of the last clipped sample.
    pub fn end(&self) -> u64 {
        self.start + self.len as u64 - 1
    }
}

#[derive(Debug, Clone, Copy)]
struct Deferred {
    run: ClipRun,
    /// |second difference| at the last run sample.
    exit_bend: f32,
    /// Last run sample and the sample after it.
    last: f32,
    after: f32,
}

/// Streaming clip detector for one channel. Allocation-free.
#[derive(Debug, Clone)]
pub struct ClipDetector {
    params: DetectorParams,
    /// Index of the next sample.
    index: u64,
    /// Previous sample, and the one before it.
    prev: f32,
    prev2: f32,
    cand_start: u64,
    cand_len: usize,
    cand_ref: f32,
    /// Sample before the candidate, and the one before that.
    cand_before: f32,
    cand_before2: f32,
    /// A two-sample run whose entry corner passed, waiting for one more
    /// sample to test its exit corner.
    deferred: Option<Deferred>,
    /// Second sample of the candidate.
    cand_second: f32,
    /// Last and second-to-last sample of the candidate.
    cand_last: f32,
    cand_last2: f32,
    /// Last accepted level of a run of length ≥ 3, per direction (0: top, 1: bottom).
    ceiling: [f32; 2],
    /// Stream index at which `ceiling` was last confirmed; `None` = never.
    ceiling_seen: [Option<u64>; 2],
}

impl Default for ClipDetector {
    fn default() -> Self {
        Self::new(DetectorParams::default())
    }
}

impl ClipDetector {
    /// New detector, at stream index 0, preceded by silence.
    pub fn new(params: DetectorParams) -> Self {
        Self {
            params: sanitize(params),
            index: 0,
            prev: 0.0,
            prev2: 0.0,
            cand_start: 0,
            cand_len: 0,
            cand_ref: 0.0,
            cand_before: 0.0,
            cand_before2: 0.0,
            deferred: None,
            cand_second: 0.0,
            cand_last: 0.0,
            cand_last2: 0.0,
            ceiling: [0.0; 2],
            ceiling_seen: [None; 2],
        }
    }

    /// Change the settings. Applies from the next candidate on.
    pub fn set_params(&mut self, params: DetectorParams) {
        self.params = sanitize(params);
    }

    /// The current settings.
    pub fn params(&self) -> DetectorParams {
        self.params
    }

    /// Back to stream index 0, preceded by silence.
    pub fn reset(&mut self) {
        *self = Self::new(self.params);
    }

    /// Index of the next sample to be pushed.
    pub fn index(&self) -> u64 {
        self.index
    }

    #[inline]
    fn tol(&self, level: f32) -> f32 {
        self.params.tolerance * level.abs()
    }

    /// Push the next sample. Returns a run when this sample closes one.
    #[inline]
    pub fn push(&mut self, x: f32) -> Option<ClipRun> {
        let n = self.index;
        self.index += 1;
        let deferred = self.deferred.take().and_then(|d| {
            // Exit corner: the bend at the last run sample against the bend
            // one sample later.
            let bend_after = (d.last - 2.0 * d.after + x).abs();
            (d.exit_bend >= CURVATURE_RATIO * bend_after).then_some(d.run)
        });
        let run = self.step(x, n);
        run.or(deferred)
    }

    #[inline]
    fn step(&mut self, x: f32, n: u64) -> Option<ClipRun> {
        if self.cand_len > 0 && self.member(x) {
            self.cand_len += 1;
            if self.cand_len == 2 {
                self.cand_second = x;
            }
            self.cand_last2 = self.cand_last;
            self.cand_last = x;
            self.shift(x);
            return None;
        }
        let run = if self.cand_len >= 2 {
            self.evaluate(x, n)
        } else {
            None
        };
        if run.is_none() && self.params.max_tilt > 0.0 && self.cand_len >= 2 && x.is_finite() {
            // Tilted mode: a candidate can start one sample early, on the
            // last sample of the edge. Re-anchor at its last sample and test
            // `x` against that.
            self.cand_start = n - 1;
            self.cand_len = 1;
            self.cand_ref = self.cand_last;
            self.cand_before = self.cand_last2;
            self.cand_before2 = self.prev2;
            if self.member(x) {
                self.cand_len = 2;
                self.cand_second = x;
                self.cand_last2 = self.cand_last;
                self.cand_last = x;
                self.shift(x);
                return None;
            }
        }
        if x.is_finite() {
            self.cand_start = n;
            self.cand_len = 1;
            self.cand_ref = x;
            self.cand_last = x;
            self.cand_before = self.prev;
            self.cand_before2 = self.prev2;
            self.shift(x);
        } else {
            // A non-finite sample never starts a run, and the next run does
            // not use it as its entry edge.
            self.cand_len = 0;
            self.shift(0.0);
        }
        run
    }

    #[inline]
    fn shift(&mut self, x: f32) {
        self.prev2 = self.prev;
        self.prev = x;
    }

    /// The candidate still open at the newest sample, if it already looks
    /// like the start of a clipped run (length ≥ 2, level and entry edge
    /// pass): `(start, len, dir)`.
    pub fn open_candidate(&self) -> Option<(u64, usize, i8)> {
        if self.cand_len < 2 {
            return None;
        }
        let level = self.cand_ref;
        // The candidate's first sample is always finite.
        if level.abs() < self.params.min_level {
            return None;
        }
        let dir: i8 = if level > 0.0 { 1 } else { -1 };
        let edge = self.edge(level);
        if f32::from(dir) * (level - self.cand_before) < edge {
            return None;
        }
        Some((self.cand_start, self.cand_len, dir))
    }

    /// Does `x` continue the open candidate?
    #[inline]
    fn member(&self, x: f32) -> bool {
        let r = self.cand_ref;
        let tol = self.tol(r);
        let tilt = self.params.max_tilt;
        if tilt == 0.0 {
            return (x - r).abs() <= tol;
        }
        let mag = r.abs();
        // Both tests are false for NaN.
        let same_sign = x * r > 0.0;
        let small_step = (x - self.cand_last).abs() <= tol + tilt * mag;
        if !same_sign || !small_step {
            return false;
        }
        self.cand_len < 2
            || (x - 2.0 * self.cand_last + self.cand_last2).abs() <= tol + tilt * tilt * mag
    }

    #[inline]
    fn edge(&self, level: f32) -> f32 {
        (EDGE_RATIO * level.abs()).max(4.0 * self.tol(level))
    }

    /// Decide on the candidate that `after` (at index `n`) just closed.
    fn evaluate(&mut self, after: f32, n: u64) -> Option<ClipRun> {
        let level = self.cand_ref;
        let mag = level.abs();
        // The candidate's first sample is always finite.
        if mag < self.params.min_level || !after.is_finite() {
            return None;
        }
        let dir: i8 = if level > 0.0 { 1 } else { -1 };
        let d = f32::from(dir);
        let edge = self.edge(level);
        let last = self.cand_last;
        // Corner rule: the edge into (out of) the run is steeper than the
        // run's own slope at that end by `CORNER_RATIO`.
        let slope_in = (self.cand_second - level).abs();
        let slope_out = (last - self.cand_last2).abs();
        if !(d * (level - self.cand_before) >= edge + CORNER_RATIO * slope_in
            && d * (last - after) >= edge + CORNER_RATIO * slope_out)
        {
            return None;
        }
        let slot = usize::from(dir < 0);
        let len = self.cand_len;
        let flat = (last - level).abs() <= self.tol(level);
        if len >= 3 {
            if !flat {
                return Some(self.run(level, dir));
            }
            self.ceiling[slot] = level;
            self.ceiling_seen[slot] = Some(n);
        } else {
            let known = flat
                && match self.ceiling_seen[slot] {
                    Some(seen) => {
                        n - seen <= self.params.ceiling_hold
                            && (level - self.ceiling[slot]).abs() <= self.tol(level)
                    }
                    None => false,
                };
            if !known {
                // No known ceiling: accept a flat pair only on its corners.
                // At a smooth peak that falls between two samples the
                // waveform bends about as much one sample before the pair
                // as at its edge; at a clip it bends far more at the edge.
                let bend_edge = (self.cand_before - level).abs();
                let bend_before = (self.cand_before2 - 2.0 * self.cand_before + level).abs();
                if flat && bend_edge >= CURVATURE_RATIO * bend_before {
                    self.deferred = Some(Deferred {
                        run: self.run(level, dir),
                        exit_bend: (after - last).abs(),
                        last,
                        after,
                    });
                }
                return None;
            }
            self.ceiling_seen[slot] = Some(n);
        }
        Some(self.run(level, dir))
    }

    fn run(&self, level: f32, dir: i8) -> ClipRun {
        ClipRun {
            start: self.cand_start,
            len: self.cand_len,
            level,
            dir,
            too_long: self.cand_len > self.params.max_run,
        }
    }
}

fn sanitize(mut p: DetectorParams) -> DetectorParams {
    p.min_level = if p.min_level.is_finite() {
        p.min_level.clamp(1e-6, 1e6)
    } else {
        0.1
    };
    p.tolerance = if p.tolerance.is_finite() {
        p.tolerance.clamp(0.0, 0.1)
    } else {
        1e-5
    };
    p.max_run = p.max_run.max(2);
    p.max_tilt = if p.max_tilt.is_finite() {
        p.max_tilt.clamp(0.0, 0.1)
    } else {
        0.0
    };
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs(x: &[f32], p: DetectorParams) -> Vec<ClipRun> {
        let mut d = ClipDetector::new(p);
        let mut out = Vec::new();
        for &v in x {
            if let Some(r) = d.push(v) {
                out.push(r);
            }
        }
        out
    }

    #[test]
    fn finds_flat_top_and_bottom() {
        let x = [
            0.0, 0.5, 0.8, 0.8, 0.8, 0.5, 0.0, -0.5, -0.8, -0.8, -0.8, -0.3,
        ];
        let r = runs(&x, DetectorParams::default());
        assert_eq!(r.len(), 2);
        assert_eq!((r[0].start, r[0].len, r[0].dir), (2, 3, 1));
        assert_eq!((r[1].start, r[1].len, r[1].dir), (8, 3, -1));
    }

    #[test]
    fn step_is_not_a_clip() {
        let x = [0.0, 0.5, 0.8, 0.8, 0.8, 0.9, 1.0, 0.2];
        assert!(runs(&x, DetectorParams::default()).is_empty());
    }

    #[test]
    fn pair_needs_known_ceiling() {
        let lone = [0.0, 0.5, 0.8, 0.8, 0.5, 0.0];
        assert!(runs(&lone, DetectorParams::default()).is_empty());
        let after_long = [0.0, 0.5, 0.8, 0.8, 0.8, 0.5, 0.0, 0.5, 0.8, 0.8, 0.5];
        let r = runs(&after_long, DetectorParams::default());
        assert_eq!(r.len(), 2);
        assert_eq!(r[1].len, 2);
    }

    #[test]
    fn pair_with_sharp_corners_accepted_one_sample_late() {
        let x = [0.0, 0.2, 0.4, 0.6, 0.8, 0.8, 0.6, 0.4, 0.2, 0.0];
        let mut d = ClipDetector::default();
        let at: Vec<(usize, ClipRun)> = x
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| d.push(v).map(|r| (i, r)))
            .collect();
        assert_eq!(at.len(), 1);
        assert_eq!((at[0].0, at[0].1.start, at[0].1.len), (7, 4, 2));
    }

    #[test]
    fn smooth_pair_rejected() {
        // A tone whose peak falls half way between two samples.
        let x: Vec<f32> = (-6..6)
            .map(|k| 0.9 * (0.3 * (k as f32 + 0.5)).cos())
            .collect();
        assert_eq!(x[5], x[6]);
        assert!(runs(&x, DetectorParams::default()).is_empty());
    }

    #[test]
    fn tilted_run_needs_tilt_mode() {
        let x = [0.0, 0.4, 0.8, 0.79, 0.78, 0.77, 0.4, 0.0];
        assert!(runs(&x, DetectorParams::default()).is_empty());
        let p = DetectorParams {
            max_tilt: 0.02,
            ..Default::default()
        };
        let r = runs(&x, p);
        assert_eq!(r.len(), 1);
        assert_eq!((r[0].start, r[0].len, r[0].dir), (2, 4, 1));
    }

    #[test]
    fn low_level_and_nan_ignored() {
        let x = [
            0.0,
            0.05,
            0.05,
            0.05,
            0.0,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            0.0,
        ];
        assert!(runs(&x, DetectorParams::default()).is_empty());
    }
}
