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
//!
//!    **Soft corners.** The sample next to a clipped run can lie just inside
//!    the level (the waveform crossed the level between two samples). A
//!    corner is also accepted when that sample lies inside the level by more
//!    than `2·tol` and the step one sample further out meets the edge. At a
//!    smooth peak the slope dies away towards the plateau, so the step
//!    further out is small too. Soft corners count only for runs that need
//!    no confirmation (flat runs of [`MIN_LONG_RUN`] or more, or at the
//!    known level, see below); a soft exit is decided one sample late.
//! 4. **Length and level.** A flat run of [`MIN_LONG_RUN`] or more samples
//!    is a clipped run on its own. A shorter one (two or three samples) is
//!    also what an unclipped peak can look like (a band-limited peak half way
//!    between two samples, a quantised peak), so it is accepted only
//!    - at the known clip level (next section), or
//!    - when it repeats an earlier short run: its level equals (within
//!      `tol`) the level of the last short run at no known level, in the
//!      same direction or mirrored in the other (symmetric clipping), seen
//!      within the last [`CEILING_HOLD_S`] seconds. Clipping hits the same
//!      level over and over; the earlier run is not repaired, it only
//!      vouches for the later ones.
//!
//!    A pair takes part in this only when it lies inside the cubic through
//!    the two samples on each side of it (Lagrange interpolation through
//!    `x[s−2], x[s−1], x[e+1], x[e+2]`, evaluated at `s` and `e`): the mean
//!    of the two interpolated values exceeds the level by at least
//!    `max(tol, SINGLE_DEFICIT·|level|)`. A smooth peak between two samples
//!    lies outside that cubic (its interpolation error has the sign of the
//!    fourth derivative, which at a peak points outwards), a clipped one
//!    inside. This needs the second sample after the pair, so a pair at no
//!    known level is decided one sample later. A steady tone whose peaks
//!    fall between samples repeats the same pair level, so without this test
//!    it would confirm itself.
//!
//! The clip level is the plateau's own value, so it adapts per run: a clip
//! below 0 dBFS (gain applied after clipping), a DC offset after clipping
//! (different levels for the top and bottom plateaus) and requantisation
//! after the gain all leave the plateau flat, and are found at their level.
//!
//! A run longer than `max_run` is reported with `too_long = true`; the
//! declipper passes it through unchanged and counts it.
//!
//! # Known clip level
//!
//! A peak that only just exceeds the clip level is clipped on one sample (or
//! two): it has no plateau of its own. Clipping hits the same level over and
//! over, though, so once the source has shown its clip level the detector
//! remembers it, per direction (top and bottom):
//!
//! - the level is **confirmed** by every flat run of three or more samples
//!   that passes the rules above (a new level replaces the old one), and by
//!   every flat pair accepted at that level;
//! - it **expires** [`CEILING_HOLD_S`] seconds after its last confirmation;
//!   single samples never confirm it, so it cannot keep itself alive;
//! - a **single sample** is a clipped run of length 1 when it sits within
//!   `tol` of the known level of its direction and both of its neighbours
//!   lie inside the level by at least
//!   `max(edge, tol + max_tilt·|x|)` (a sharp corner on each side; the
//!   `max_tilt` term keeps it apart from a tilted plateau). It is reported
//!   one sample late, when its right neighbour arrives.
//!
//! A change of gain after the clip moves every plateau to a new level, which
//! a long run confirms before single samples are taken at it; the old level
//! expires. Unclipped material never confirms a level, so it never gets a
//! single-sample detection.
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
//! curving top of a low-frequency tone. The droop only goes one way: a
//! high-passed plateau decays towards zero (the high-pass removes the
//! plateau's local mean, which lies inside the clip level), so a sample that
//! is further from zero than the one before it (by more than `tol`) does not
//! continue a tilted candidate. A pair must still be flat. When a
//! tilted candidate is rejected it is re-anchored at its last sample, because
//! it can start one sample early, on the last sample of the rising edge.

/// Smallest edge into and out of a plateau, relative to its level (−60 dB).
pub const EDGE_RATIO: f32 = 1e-3;

/// How much steeper the edge into (out of) a run must be than the run's own
/// slope at that end (see the corner rule).
pub const CORNER_RATIO: f32 = 4.0;

/// How long a confirmed clip level stays known after its last confirmation
/// (seconds): it vouches for two-sample and single-sample runs at that level.
pub const CEILING_HOLD_S: f32 = 5.0;

/// Shortest flat run that is a clipped run on its own (and confirms its
/// level as the known clip level). Shorter flat runs need a known level, or
/// an earlier short run at the same level.
pub const MIN_LONG_RUN: usize = 4;

/// Shortest tilted run (see "Tilted plateaus").
pub const MIN_TILTED_RUN: usize = 4;

/// How far (relative to its level) a single sample must lie inside the
/// cubic through its four neighbours (−80 dB): above the rounding noise of
/// 16-bit audio at full scale, which can put a smooth peak a little inside
/// the cubic.
pub const SINGLE_DEFICIT: f32 = 1e-4;

/// Most runs one [`ClipDetector::push`] can produce (the rest wait for the
/// next calls).
const OUT_QUEUE: usize = 4;

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
    /// Detect single clipped samples at the known clip level (see "Known
    /// clip level" in the module docs).
    pub singles: bool,
}

impl Default for DetectorParams {
    fn default() -> Self {
        Self {
            min_level: 0.1,
            tolerance: 1e-5,
            max_run: 96,
            ceiling_hold: (CEILING_HOLD_S * 48_000.0) as u64,
            max_tilt: 0.0,
            singles: true,
        }
    }
}

/// One detected clipped run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipRun {
    /// Stream index of the first clipped sample.
    pub start: u64,
    /// Number of clipped samples (≥ 1; 1 only at the known clip level).
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

/// A candidate that passed the corner rule, waiting for its decision. The
/// decision needs the second sample after the run when the exit corner is
/// soft, and for a flat pair (cubic test); it is then taken one sample late.
#[derive(Debug, Clone, Copy)]
struct Closed {
    run: ClipRun,
    flat: bool,
    /// A corner was soft (allowed only for runs that need no confirmation).
    soft: bool,
    /// Soft exit: the edge the next step out must reach.
    need_out: Option<f32>,
    /// The two samples before the run, and the one after it.
    before2: f32,
    before: f32,
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
    /// A closed candidate whose decision waits for one more sample.
    deferred: Option<Closed>,
    /// Second sample of the candidate.
    cand_second: f32,
    /// Last and second-to-last sample of the candidate.
    cand_last: f32,
    cand_last2: f32,
    /// Last accepted level of a run of length ≥ 3, per direction (0: top, 1: bottom).
    ceiling: [f32; 2],
    /// Stream index at which `ceiling` was last confirmed; `None` = never.
    ceiling_seen: [Option<u64>; 2],
    /// Level of the last short flat run at no known level, per direction,
    /// and when it was seen.
    candidate: [f32; 2],
    candidate_seen: [Option<u64>; 2],
    /// Index of the last single-sample run.
    last_single: Option<u64>,
    /// The last four samples, newest first.
    hist: [f32; 4],
    /// Runs found but not yet returned (ring).
    out: [ClipRun; OUT_QUEUE],
    out_head: usize,
    out_len: usize,
}

const NO_RUN: ClipRun = ClipRun {
    start: 0,
    len: 1,
    level: 0.0,
    dir: 1,
    too_long: false,
};

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
            candidate: [0.0; 2],
            candidate_seen: [None; 2],
            hist: [0.0; 4],
            last_single: None,
            out: [NO_RUN; OUT_QUEUE],
            out_head: 0,
            out_len: 0,
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
    ///
    /// When one sample closes more than one run (rare: a single sample and a
    /// pair that both wait for this sample), the runs come out in order of
    /// their start, one per call, so a run can come out a call or two after
    /// the sample that closed it.
    #[inline]
    pub fn push(&mut self, x: f32) -> Option<ClipRun> {
        let n = self.index;
        self.index += 1;
        let single = self.single(x, n);
        if let Some(r) = single {
            self.last_single = Some(r.start);
        }
        self.hist = [x, self.hist[0], self.hist[1], self.hist[2]];
        let deferred = self
            .deferred
            .take()
            .and_then(|c| self.decide(c, Some(x), n));
        let run = self.step(x, n);
        // Ordered by start: a deferred pair ends before the sample that
        // closed it, a single is the sample before this one, and a run
        // closed by this sample ends there too, so at most one of `run` and
        // `single` exists.
        for r in [deferred, single, run].into_iter().flatten() {
            if self.out_len < OUT_QUEUE {
                self.out[(self.out_head + self.out_len) % OUT_QUEUE] = r;
                self.out_len += 1;
            }
        }
        if self.out_len == 0 {
            return None;
        }
        let r = self.out[self.out_head];
        self.out_head = (self.out_head + 1) % OUT_QUEUE;
        self.out_len -= 1;
        Some(r)
    }

    /// Is the sample two back (index `n − 2`) a single clipped sample at
    /// the known clip level, with `x` the second sample after it?
    #[inline]
    fn single(&self, x: f32, n: u64) -> Option<ClipRun> {
        if !self.params.singles || n < 4 || self.ceiling_seen == [None, None] {
            return None;
        }
        let [b1, v, a1, a2] = self.hist;
        let mag = v.abs();
        if mag < self.params.min_level || !(x + b1 + v + a1 + a2).is_finite() {
            return None;
        }
        let dir: i8 = if v > 0.0 { 1 } else { -1 };
        let m = n - 2;
        if !self.is_known(usize::from(dir < 0), v, m) {
            return None;
        }
        let d = f32::from(dir);
        let edge = self.edge(v);
        // Corners on both sides, and the sample lies inside the cubic
        // through its four neighbours (a smooth peak lies outside it).
        let cubic = (4.0 * (a1 + b1) - (a2 + x)) / 6.0;
        (d * (v - a1) >= edge
            && d * (v - b1) >= edge
            && d * (cubic - v) >= self.tol(v).max(SINGLE_DEFICIT * mag))
        .then_some(ClipRun {
            start: m,
            len: 1,
            level: v,
            dir,
            too_long: false,
        })
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
        if run.is_none()
            && self.deferred.is_none()
            && self.params.max_tilt > 0.0
            && self.cand_len >= 2
            && x.is_finite()
        {
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
        let step = x - self.cand_last;
        let small_step = step.abs() <= tol + tilt * mag;
        // A high-passed plateau droops towards zero, never away from it.
        let droops = step * r.signum() <= tol;
        if !same_sign || !small_step || !droops {
            return false;
        }
        self.cand_len < 2
            || (x - 2.0 * self.cand_last + self.cand_last2).abs() <= tol + tilt * tilt * mag
    }

    #[inline]
    fn edge(&self, level: f32) -> f32 {
        (EDGE_RATIO * level.abs()).max(4.0 * self.tol(level))
    }

    /// Decide on the candidate that `after` (at index `n`) just closed, or
    /// keep it for the next sample (see [`Closed`]).
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
        let tol = self.tol(level);
        // Corner rule: the edge into (out of) the run is steeper than the
        // run's own slope at that end by `CORNER_RATIO`. A soft corner: the
        // sample next to the run lies only just inside it, and the edge is
        // one sample further out.
        let slope_in = (self.cand_second - level).abs();
        let slope_out = (last - self.cand_last2).abs();
        let need_in = edge + CORNER_RATIO * slope_in;
        let need_out = edge + CORNER_RATIO * slope_out;
        let step_in = d * (level - self.cand_before);
        let soft_in = step_in < need_in;
        if soft_in
            && !(step_in > 2.0 * tol && d * (self.cand_before - self.cand_before2) >= need_in)
        {
            return None;
        }
        let step_out = d * (last - after);
        let soft_out = step_out < need_out;
        if soft_out && step_out <= 2.0 * tol {
            return None;
        }
        let flat = (last - level).abs() <= tol;
        let c = Closed {
            run: self.run(level, dir),
            flat,
            soft: soft_in || soft_out,
            need_out: soft_out.then_some(need_out),
            before2: self.cand_before2,
            before: self.cand_before,
            after,
        };
        if soft_out || (flat && self.cand_len == 2) {
            self.deferred = Some(c);
            return None;
        }
        self.decide(c, None, n)
    }

    /// Decide on a closed candidate; `next` is the second sample after it
    /// (for a decision taken one sample late).
    fn decide(&mut self, c: Closed, next: Option<f32>, n: u64) -> Option<ClipRun> {
        let ClipRun {
            level, dir, len, ..
        } = c.run;
        let d = f32::from(dir);
        if let Some(need_out) = c.need_out {
            let x = next?;
            if !(x.is_finite() && d * (c.after - x) >= need_out) {
                return None;
            }
        }
        let slot = usize::from(dir < 0);
        if !c.flat {
            // Tilted: only long runs with sharp corners (a short straight
            // piece of waveform at a peak looks the same).
            if len < MIN_TILTED_RUN || c.soft {
                return None;
            }
            let mut run = c.run;
            if self.last_single == Some(run.start) {
                // Its first sample was already reported as a single (it
                // droops from there): report the rest.
                run.start += 1;
                run.len -= 1;
            }
            return Some(run);
        }
        if len >= MIN_LONG_RUN || self.is_known(slot, level, n) {
            self.confirm(slot, level, n);
            return Some(c.run);
        }
        if c.soft {
            // Soft corners only for runs that need no confirmation.
            return None;
        }
        if len == 2 {
            // Cubic through x[s−2], x[s−1], x[e+1], x[e+2] (nodes −2, −1, 2,
            // 3), evaluated at s (0) and e (1): the pair must lie inside it.
            let x = next?;
            let p0 = -0.3 * c.before2 + c.before + 0.5 * c.after - 0.2 * x;
            let p1 = -0.3 * x + c.after + 0.5 * c.before - 0.2 * c.before2;
            let margin = self.tol(level).max(SINGLE_DEFICIT * level.abs());
            if !(x.is_finite() && d * (0.5 * (p0 + p1) - level) >= margin) {
                return None;
            }
        }
        self.short_run(c.run, n)
    }

    /// Is `level` the known clip level of direction `slot` at index `n`?
    #[inline]
    fn is_known(&self, slot: usize, level: f32, n: u64) -> bool {
        match self.ceiling_seen[slot] {
            Some(seen) => {
                n.saturating_sub(seen) <= self.params.ceiling_hold
                    && (level - self.ceiling[slot]).abs() <= self.tol(self.ceiling[slot])
            }
            None => false,
        }
    }

    /// Does `level` (direction `slot`) repeat the last short run at no known
    /// level, in the same direction or mirrored in the other (symmetric
    /// clipping)?
    #[inline]
    fn repeats_candidate(&self, slot: usize, level: f32, n: u64) -> bool {
        [(slot, level), (1 - slot, -level)].iter().any(|&(k, l)| {
            self.candidate_seen[k].is_some_and(|seen| {
                n.saturating_sub(seen) <= self.params.ceiling_hold
                    && (l - self.candidate[k]).abs() <= self.tol(self.candidate[k])
            })
        })
    }

    /// Make `level` the known clip level of direction `slot`, confirmed at `n`.
    #[inline]
    fn confirm(&mut self, slot: usize, level: f32, n: u64) {
        if !self.is_known(slot, level, n) {
            self.ceiling[slot] = level;
        }
        self.ceiling_seen[slot] = Some(n);
        self.candidate_seen[slot] = None;
    }

    /// A short flat run (2 or 3 samples, corners passed) at no known level:
    /// accepted when an earlier short run reached the same level within the
    /// hold time (the level is then known); otherwise it becomes that
    /// earlier run for the next one.
    fn short_run(&mut self, run: ClipRun, n: u64) -> Option<ClipRun> {
        let slot = usize::from(run.dir < 0);
        if self.repeats_candidate(slot, run.level, n) {
            self.confirm(slot, run.level, n);
            Some(run)
        } else {
            self.candidate[slot] = run.level;
            self.candidate_seen[slot] = Some(n);
            None
        }
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
            0.0, 0.5, 0.8, 0.8, 0.8, 0.8, 0.5, 0.0, -0.5, -0.8, -0.8, -0.8, -0.8, -0.3,
        ];
        let r = runs(&x, DetectorParams::default());
        assert_eq!(r.len(), 2);
        assert_eq!((r[0].start, r[0].len, r[0].dir), (2, 4, 1));
        assert_eq!((r[1].start, r[1].len, r[1].dir), (9, 4, -1));
    }

    #[test]
    fn step_is_not_a_clip() {
        let x = [0.0, 0.5, 0.8, 0.8, 0.8, 0.8, 0.9, 1.0, 0.2];
        assert!(runs(&x, DetectorParams::default()).is_empty());
    }

    #[test]
    fn short_run_needs_a_repeat_or_known_level() {
        // A lone three-sample run is only a candidate.
        let lone = [0.0, 0.5, 0.8, 0.8, 0.8, 0.5, 0.0];
        assert!(runs(&lone, DetectorParams::default()).is_empty());
        // The mirrored repeat confirms the level: the second run counts.
        let mirrored = [
            0.0, 0.5, 0.8, 0.8, 0.8, 0.5, 0.0, -0.5, -0.8, -0.8, -0.8, -0.5, 0.0,
        ];
        let r = runs(&mirrored, DetectorParams::default());
        assert_eq!(r.len(), 1);
        assert_eq!((r[0].start, r[0].len, r[0].dir), (8, 3, -1));
        // After a long run the level is known: a pair at it counts at once.
        let after_long = [
            0.0, 0.5, 0.8, 0.8, 0.8, 0.8, 0.5, 0.0, 0.5, 0.8, 0.8, 0.5, 0.0,
        ];
        let r = runs(&after_long, DetectorParams::default());
        assert_eq!(r.len(), 2);
        assert_eq!((r[1].start, r[1].len), (9, 2));
    }

    #[test]
    fn pair_inside_its_cubic_is_a_candidate_one_sample_late() {
        // Two clipped pairs at the same level: the second is accepted, on
        // the second sample after it.
        let x = [
            0.0, 0.0, 0.6, 0.8, 0.8, 0.6, 0.0, 0.0, 0.0, 0.6, 0.8, 0.8, 0.6, 0.0, 0.0,
        ];
        let mut d = ClipDetector::default();
        let at: Vec<(usize, ClipRun)> = x
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| d.push(v).map(|r| (i, r)))
            .collect();
        assert_eq!(at.len(), 1);
        assert_eq!((at[0].0, at[0].1.start, at[0].1.len), (13, 10, 2));
    }

    #[test]
    fn single_at_known_level() {
        // A long run makes 0.8 known; a single sample at 0.8 that lies
        // inside the cubic through its neighbours is then a run.
        let x = [
            0.0, 0.5, 0.8, 0.8, 0.8, 0.8, 0.5, 0.0, 0.2, 0.7, 0.8, 0.7, 0.2, 0.0, 0.0,
        ];
        let r = runs(&x, DetectorParams::default());
        assert_eq!(r.len(), 2);
        assert_eq!((r[1].start, r[1].len), (10, 1));
        // A smooth peak at the same level (outside its cubic) is not.
        let smooth = [
            0.0, 0.5, 0.8, 0.8, 0.8, 0.8, 0.5, 0.0, 0.606, 0.75, 0.8, 0.75, 0.606, 0.0, 0.0,
        ];
        assert_eq!(runs(&smooth, DetectorParams::default()).len(), 1);
        let off = DetectorParams {
            singles: false,
            ..Default::default()
        };
        assert_eq!(runs(&x, off).len(), 1);
    }

    #[test]
    fn soft_corner_for_long_runs() {
        // The sample before the plateau lies just inside it; the step one
        // further out is the edge.
        let x = [0.0, 0.3, 0.7995, 0.8, 0.8, 0.8, 0.8, 0.5, 0.0, 0.0];
        let r = runs(&x, DetectorParams::default());
        assert_eq!(r.len(), 1);
        assert_eq!((r[0].start, r[0].len), (3, 4));
        // A plateau the waveform approaches slowly is not a clip.
        let slow = [0.0, 0.799, 0.7995, 0.8, 0.8, 0.8, 0.8, 0.5, 0.0, 0.0];
        assert!(runs(&slow, DetectorParams::default()).is_empty());
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
