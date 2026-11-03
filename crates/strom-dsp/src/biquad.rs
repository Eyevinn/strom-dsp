//! Second-order IIR ("biquad") building blocks: RBJ cookbook coefficient
//! design, analytic magnitude, and a multichannel Direct Form I filter with
//! optional per-sample coefficient interpolation.
//!
//! # Coefficient design
//!
//! The design functions follow the equations of R. Bristow-Johnson's
//! "Audio EQ Cookbook" (W3C Note, public domain): `w0 = 2π·f0/fs`,
//! `α = sin(w0)/(2Q)`, `A = 10^(gain/40)`; the shelves use the cookbook's Q form.
//! `1 − cos w0` and `1 + cos w0` are evaluated as `2·sin²(w0/2)` and
//! `2·cos²(w0/2)` (same value, no cancellation at very low frequencies), and the
//! shelf terms are rewritten in terms of `1 − cos w0`. All coefficients are
//! normalised by `a0`.
//!
//! The functions do **not** clamp: they expect `0 < f0 < fs/2`, `Q > 0` and a
//! finite gain. Within that domain every design is stable (poles strictly inside
//! the unit circle).
//!
//! # Filter structure
//!
//! [`Biquad`] is Direct Form I with `f64` coefficients and `f64` state
//! (J. O. Smith III, *Introduction to Digital Filters*):
//! `y[n] = b0·x[n] + b1·x[n−1] + b2·x[n−2] − a1·y[n−1] − a2·y[n−2]`.
//! DF-I keeps only past inputs and outputs as state, so coefficients may change
//! at any sample without leaving the state inconsistent. Each output (the only
//! recursive quantity) is flushed to zero below `1e-15` so decays never reach
//! the subnormal range.
//!
//! # Coefficient interpolation
//!
//! [`Biquad::ramp_to`] moves the coefficients linearly, per sample, to a new
//! set over a given number of frames. If both end points are stable, every
//! intermediate filter is stable too: the stability region of the denominator
//! `1 + a1·z⁻¹ + a2·z⁻²` is the triangle `|a2| < 1, |a1| < 1 + a2`, which is
//! convex, so it contains the segment between any two of its points.
//!
//! # Multichannel processing
//!
//! A biquad is recursive in time, so it cannot be vectorised along time. The
//! channels of a frame are independent, though, so they are processed as SIMD
//! lanes: channels are split into pairs (plus one single channel if the count
//! is odd) and each pair's state is stored structure-of-arrays
//! (`x1[2], x2[2], y1[2], y2[2]`). Mono and stereo are single groups and get
//! dedicated, bounds-check-free loops. The same kernels run several biquads in
//! series sample by sample (see [`crate::Equalizer`]), so the independent
//! recursions of different stages overlap in the CPU pipeline; this measured
//! about 1.9× faster than running each biquad over the whole buffer in turn
//! (same kernels, Apple M5 Pro, 4 bands).
//!
//! Denormals: the steady-state kernel flushes the filter state to zero below
//! `1e-15` every 32 frames rather than every sample (a
//! per-sample flush lengthens the recursive dependency chain and cost about
//! 30 %). Subnormal values can therefore exist for at most that many frames
//! before they are cleared, so there is no sustained slow path. Kernels that
//! interpolate or crossfade (only while parameters change) flush every sample.

use core::f64::consts::PI;

use crate::lanes::Lanes;

/// Normalised biquad coefficients (`a0 = 1`):
/// `H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BiquadCoefficients {
    /// Feed-forward coefficient of `x[n]`.
    pub b0: f64,
    /// Feed-forward coefficient of `x[n−1]`.
    pub b1: f64,
    /// Feed-forward coefficient of `x[n−2]`.
    pub b2: f64,
    /// Feedback coefficient of `y[n−1]` (sign as in the denominator above).
    pub a1: f64,
    /// Feedback coefficient of `y[n−2]` (sign as in the denominator above).
    pub a2: f64,
}

/// Quantities shared by all cookbook designs at one frequency and Q.
struct Warp {
    sin_w: f64,
    cos_w: f64,
    /// `1 − cos w0`, computed without cancellation.
    one_minus_cos: f64,
    /// `1 + cos w0`, computed without cancellation.
    one_plus_cos: f64,
    alpha: f64,
}

impl Warp {
    fn new(f0_hz: f64, q: f64, sample_rate_hz: f64) -> Self {
        let w0 = 2.0 * PI * f0_hz / sample_rate_hz;
        let (sin_w, cos_w) = w0.sin_cos();
        let (half_sin, half_cos) = (0.5 * w0).sin_cos();
        Self {
            sin_w,
            cos_w,
            one_minus_cos: 2.0 * half_sin * half_sin,
            one_plus_cos: 2.0 * half_cos * half_cos,
            alpha: sin_w / (2.0 * q),
        }
    }

    /// Denominator `(1 + α, −2 cos w0, 1 − α)` shared by the non-gain types.
    fn plain_den(&self) -> (f64, f64, f64) {
        (1.0 + self.alpha, -2.0 * self.cos_w, 1.0 - self.alpha)
    }
}

impl BiquadCoefficients {
    /// The identity filter `H(z) = 1`.
    pub const IDENTITY: Self = Self {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    };

    /// Normalise un-normalised coefficients `(b0, b1, b2, a0, a1, a2)` by `a0`.
    pub fn from_unnormalized(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        let inv = 1.0 / a0;
        Self {
            b0: b0 * inv,
            b1: b1 * inv,
            b2: b2 * inv,
            a1: a1 * inv,
            a2: a2 * inv,
        }
    }

    fn from_parts(b: (f64, f64, f64), a: (f64, f64, f64)) -> Self {
        Self::from_unnormalized(b.0, b.1, b.2, a.0, a.1, a.2)
    }

    /// Second-order low-pass at corner `f0_hz` (Hz) with quality `q`, for
    /// `sample_rate_hz` (Hz). `|H(f0)| = Q`, unity at DC.
    pub fn low_pass(f0_hz: f64, q: f64, sample_rate_hz: f64) -> Self {
        let w = Warp::new(f0_hz, q, sample_rate_hz);
        let k = w.one_minus_cos;
        Self::from_parts((0.5 * k, k, 0.5 * k), w.plain_den())
    }

    /// Second-order high-pass at corner `f0_hz` (Hz) with quality `q`.
    /// `|H(f0)| = Q`, unity at Nyquist.
    pub fn high_pass(f0_hz: f64, q: f64, sample_rate_hz: f64) -> Self {
        let w = Warp::new(f0_hz, q, sample_rate_hz);
        let k = w.one_plus_cos;
        Self::from_parts((0.5 * k, -k, 0.5 * k), w.plain_den())
    }

    /// Band-pass with constant skirt gain centred on `f0_hz` (Hz): peak gain = Q.
    pub fn band_pass_skirt(f0_hz: f64, q: f64, sample_rate_hz: f64) -> Self {
        let w = Warp::new(f0_hz, q, sample_rate_hz);
        Self::from_parts((0.5 * w.sin_w, 0.0, -0.5 * w.sin_w), w.plain_den())
    }

    /// Band-pass with constant 0 dB peak gain centred on `f0_hz` (Hz).
    pub fn band_pass_peak(f0_hz: f64, q: f64, sample_rate_hz: f64) -> Self {
        let w = Warp::new(f0_hz, q, sample_rate_hz);
        Self::from_parts((w.alpha, 0.0, -w.alpha), w.plain_den())
    }

    /// Notch at `f0_hz` (Hz) (zero gain at `f0`, unity at DC and Nyquist).
    pub fn notch(f0_hz: f64, q: f64, sample_rate_hz: f64) -> Self {
        let w = Warp::new(f0_hz, q, sample_rate_hz);
        Self::from_parts((1.0, -2.0 * w.cos_w, 1.0), w.plain_den())
    }

    /// All-pass with its phase centre (−180°) at `f0_hz` (Hz); unity magnitude.
    pub fn all_pass(f0_hz: f64, q: f64, sample_rate_hz: f64) -> Self {
        let w = Warp::new(f0_hz, q, sample_rate_hz);
        Self::from_parts(
            (1.0 - w.alpha, -2.0 * w.cos_w, 1.0 + w.alpha),
            w.plain_den(),
        )
    }

    /// Peaking (bell) at `f0_hz` (Hz) with `gain_db` (dB) at the centre and
    /// unity far away.
    pub fn peaking(f0_hz: f64, gain_db: f64, q: f64, sample_rate_hz: f64) -> Self {
        let w = Warp::new(f0_hz, q, sample_rate_hz);
        let a = 10f64.powf(gain_db / 40.0);
        let c = -2.0 * w.cos_w;
        Self::from_parts(
            (1.0 + w.alpha * a, c, 1.0 - w.alpha * a),
            (1.0 + w.alpha / a, c, 1.0 - w.alpha / a),
        )
    }

    /// Shelf numerator/denominator terms in terms of `1 − cos w0`:
    /// `p = (A+1) − (A−1)cos w0`, `m = (A+1) + (A−1)cos w0`,
    /// `dm = (A−1) − (A+1)cos w0`, `dp = (A−1) + (A+1)cos w0`, `sa = 2√A·α`.
    fn shelf_terms(w: &Warp, a: f64) -> (f64, f64, f64, f64, f64) {
        let k = w.one_minus_cos;
        let p = 2.0 + (a - 1.0) * k;
        let m = 2.0 * a - (a - 1.0) * k;
        let dm = -2.0 + (a + 1.0) * k;
        let dp = 2.0 * a - (a + 1.0) * k;
        let sa = 2.0 * a.sqrt() * w.alpha;
        (p, m, dm, dp, sa)
    }

    /// Low shelf with corner `f0_hz` (Hz), shelf gain `gain_db` (dB) below the
    /// corner and the cookbook Q form (`gain_db/2` at `f0`).
    pub fn low_shelf(f0_hz: f64, gain_db: f64, q: f64, sample_rate_hz: f64) -> Self {
        let w = Warp::new(f0_hz, q, sample_rate_hz);
        let a = 10f64.powf(gain_db / 40.0);
        let (p, m, dm, dp, sa) = Self::shelf_terms(&w, a);
        Self::from_parts(
            (a * (p + sa), 2.0 * a * dm, a * (p - sa)),
            (m + sa, -2.0 * dp, m - sa),
        )
    }

    /// High shelf with corner `f0_hz` (Hz), shelf gain `gain_db` (dB) above the
    /// corner and the cookbook Q form (`gain_db/2` at `f0`).
    pub fn high_shelf(f0_hz: f64, gain_db: f64, q: f64, sample_rate_hz: f64) -> Self {
        let w = Warp::new(f0_hz, q, sample_rate_hz);
        let a = 10f64.powf(gain_db / 40.0);
        let (p, m, dm, dp, sa) = Self::shelf_terms(&w, a);
        Self::from_parts(
            (a * (m + sa), -2.0 * a * dp, a * (m - sa)),
            (p + sa, 2.0 * dm, p - sa),
        )
    }

    /// Magnitude `|H(e^{jω})|` at normalised angular frequency `w` (rad/sample).
    pub fn magnitude(&self, w: f64) -> f64 {
        // e^{-jw} = cos w − j sin w ; e^{-2jw} = cos 2w − j sin 2w
        let (s1, c1) = w.sin_cos();
        let (s2, c2) = (2.0 * w).sin_cos();
        let nr = self.b0 + self.b1 * c1 + self.b2 * c2;
        let ni = -(self.b1 * s1 + self.b2 * s2);
        let dr = 1.0 + self.a1 * c1 + self.a2 * c2;
        let di = -(self.a1 * s1 + self.a2 * s2);
        ((nr * nr + ni * ni) / (dr * dr + di * di)).sqrt()
    }

    /// Magnitude `|H|` at `freq_hz` (Hz) for a sample rate `sample_rate_hz` (Hz).
    pub fn magnitude_at(&self, freq_hz: f64, sample_rate_hz: f64) -> f64 {
        self.magnitude(2.0 * PI * freq_hz / sample_rate_hz)
    }

    /// True if both poles lie strictly inside the unit circle
    /// (stability triangle `|a2| < 1`, `|a1| < 1 + a2`).
    pub fn is_stable(&self) -> bool {
        self.a2.abs() < 1.0 && self.a1.abs() < 1.0 + self.a2
    }

    /// Per-sample increment that moves `self` linearly to `to` in `frames`
    /// steps (`frames ≥ 1`).
    pub fn ramp_step(&self, to: &Self, frames: usize) -> Self {
        let k = 1.0 / frames.max(1) as f64;
        Self {
            b0: (to.b0 - self.b0) * k,
            b1: (to.b1 - self.b1) * k,
            b2: (to.b2 - self.b2) * k,
            a1: (to.a1 - self.a1) * k,
            a2: (to.a2 - self.a2) * k,
        }
    }

    /// Add a per-sample increment (from [`BiquadCoefficients::ramp_step`]).
    #[inline(always)]
    pub fn add_step(&mut self, d: &Self) {
        self.b0 += d.b0;
        self.b1 += d.b1;
        self.b2 += d.b2;
        self.a1 += d.a1;
        self.a2 += d.a2;
    }
}

// ---------------------------------------------------------------------------
// Lane kernels (crate-internal; shared by `Biquad` and `Equalizer`).

/// Maximum number of biquads run in series by one kernel call.
pub(crate) const MAX_STAGES: usize = 8;

/// Number of `f64` values of DF-I state per channel.
pub(crate) const STATE_PER_CHANNEL: usize = 4;

/// DF-I state of `L` channels, structure-of-arrays.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct LaneState<const L: usize> {
    x1: Lanes<L>,
    x2: Lanes<L>,
    y1: Lanes<L>,
    y2: Lanes<L>,
}

impl<const L: usize> LaneState<L> {
    /// Load from `[x1; L][x2; L][y1; L][y2; L]`.
    #[inline(always)]
    fn load(src: &[f64]) -> Self {
        Self {
            x1: Lanes::load(&src[0..]),
            x2: Lanes::load(&src[L..]),
            y1: Lanes::load(&src[2 * L..]),
            y2: Lanes::load(&src[3 * L..]),
        }
    }

    #[inline(always)]
    fn store(&self, dst: &mut [f64]) {
        self.x1.store(&mut dst[0..]);
        self.x2.store(&mut dst[L..]);
        self.y1.store(&mut dst[2 * L..]);
        self.y2.store(&mut dst[3 * L..]);
    }

    /// One DF-I step. The other terms are summed as a shallow tree first and
    /// the `y[n−1]` term last, so the loop-carried dependency is a single
    /// (fused, where available) multiply-add, and the path from the stage
    /// input to its output stays short.
    /// No denormal flush here (it would lengthen that dependency); callers
    /// flush with [`LaneState::flush_denormals`] at least every
    /// [`FLUSH_INTERVAL`] frames.
    #[inline(always)]
    fn tick(&mut self, c: &BiquadCoefficients, x: Lanes<L>) -> Lanes<L> {
        let p = self.x1.mul_add(c.b1, x * c.b0);
        let q = self.y2.mul_add(-c.a2, self.x2 * c.b2);
        let y = self.y1.mul_add(-c.a1, p + q);
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    /// Flush every state value below [`crate::util::DENORMAL_FLUSH`] to zero.
    #[inline(always)]
    fn flush_denormals(&mut self) {
        self.x1 = self.x1.flush_denormal();
        self.x2 = self.x2.flush_denormal();
        self.y1 = self.y1.flush_denormal();
        self.y2 = self.y2.flush_denormal();
    }
}

/// Frames between denormal flushes of the filter state in the kernels.
pub(crate) const FLUSH_INTERVAL: usize = 32;

/// Split `channels` into lane groups `(first channel, width)`: pairs, plus a
/// single last channel if the count is odd.
///
/// Pairs match 128-bit SIMD (NEON, SSE2: two `f64`). Groups of four were
/// measured to be slower on NEON: the state of a 4-band chain no longer fits
/// in registers.
pub(crate) fn channel_groups(channels: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..channels / 2)
        .map(|g| (2 * g, 2))
        .chain((channels % 2 == 1).then_some((channels - 1, 1)))
}

/// One steady (fixed-coefficient) biquad in a series chain.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SteadyStage {
    pub coefs: BiquadCoefficients,
    /// Start of this stage's state bank (`STATE_PER_CHANNEL · channels` values).
    pub state: usize,
}

/// Run `stages` in series over interleaved `data` (whole frames of `channels`),
/// sample by sample. `states` holds every bank referenced by the stages.
///
/// `ramps` is either empty (fixed coefficients: the steady-state loop) or has
/// one entry per stage; a stage with `Some(delta)` adds `delta` to its
/// coefficients before each sample (per-sample linear interpolation), the
/// others keep theirs.
pub(crate) fn run_chain(
    data: &mut [f32],
    channels: usize,
    stages: &[SteadyStage],
    ramps: &[Option<BiquadCoefficients>],
    states: &mut [f64],
) {
    debug_assert!(stages.len() <= MAX_STAGES);
    debug_assert!(ramps.is_empty() || ramps.len() == stages.len());
    if stages.is_empty() {
        return;
    }
    for (c0, width) in channel_groups(channels) {
        match (width, ramps.is_empty()) {
            (2, true) => chain_group::<2, false>(data, channels, c0, stages, ramps, states),
            (2, false) => chain_group::<2, true>(data, channels, c0, stages, ramps, states),
            (_, true) => chain_group::<1, false>(data, channels, c0, stages, ramps, states),
            (_, false) => chain_group::<1, true>(data, channels, c0, stages, ramps, states),
        }
    }
}

fn chain_group<const L: usize, const RAMP: bool>(
    data: &mut [f32],
    channels: usize,
    c0: usize,
    stages: &[SteadyStage],
    ramps: &[Option<BiquadCoefficients>],
    states: &mut [f64],
) {
    // A fixed stage count lets the compiler unroll the chain and keep every
    // state in registers.
    let args = (data, channels, c0, stages, ramps, states);
    match stages.len() {
        1 => chain_n::<L, 1, RAMP>(args),
        2 => chain_n::<L, 2, RAMP>(args),
        3 => chain_n::<L, 3, RAMP>(args),
        4 => chain_n::<L, 4, RAMP>(args),
        5 => chain_n::<L, 5, RAMP>(args),
        6 => chain_n::<L, 6, RAMP>(args),
        7 => chain_n::<L, 7, RAMP>(args),
        _ => chain_n::<L, 8, RAMP>(args),
    }
}

/// One frame through `N` stages.
#[inline(always)]
fn chain_tick<const L: usize, const N: usize, const RAMP: bool>(
    coefs: &mut [BiquadCoefficients; N],
    deltas: &[Option<BiquadCoefficients>; N],
    st: &mut [LaneState<L>; N],
    mut x: Lanes<L>,
) -> Lanes<L> {
    for ((c, d), s) in coefs.iter_mut().zip(deltas).zip(st.iter_mut()) {
        if RAMP {
            if let Some(d) = d {
                c.add_step(d);
            }
        }
        x = s.tick(c, x);
    }
    x
}

type ChainArgs<'a> = (
    &'a mut [f32],
    usize,
    usize,
    &'a [SteadyStage],
    &'a [Option<BiquadCoefficients>],
    &'a mut [f64],
);

fn chain_n<const L: usize, const N: usize, const RAMP: bool>(
    (data, channels, c0, stages, ramps, states): ChainArgs<'_>,
) {
    let off = STATE_PER_CHANNEL * c0;
    let mut coefs: [BiquadCoefficients; N] = core::array::from_fn(|i| stages[i].coefs);
    let deltas: [Option<BiquadCoefficients>; N] =
        core::array::from_fn(|i| if RAMP { ramps[i] } else { None });
    let mut st: [LaneState<L>; N] =
        core::array::from_fn(|i| LaneState::load(&states[stages[i].state + off..]));
    for block in data.chunks_mut(FLUSH_INTERVAL * channels) {
        if channels == L {
            for frame in block.chunks_exact_mut(L) {
                let y =
                    chain_tick::<L, N, RAMP>(&mut coefs, &deltas, &mut st, Lanes::load_f32(frame));
                y.store_f32(frame);
            }
        } else {
            for frame in block.chunks_exact_mut(channels) {
                let lanes = &mut frame[c0..c0 + L];
                let y =
                    chain_tick::<L, N, RAMP>(&mut coefs, &deltas, &mut st, Lanes::load_f32(lanes));
                y.store_f32(lanes);
            }
        }
        st.iter_mut().for_each(LaneState::flush_denormals);
    }
    for (s, stage) in st.iter().zip(stages) {
        s.store(&mut states[stage.state + off..]);
    }
}

/// The realisation being faded out during a crossfade.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FadeOut {
    /// `None`: the old realisation is the dry input.
    pub old: Option<SteadyStage>,
    /// Crossfade frames already done before the first frame of this call.
    pub done: usize,
    /// `1 / crossfade length`.
    pub inv_len: f64,
}

/// One stage of a chain whose coefficients and/or realisation are changing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ChangingStage {
    /// `None`: the new realisation is the dry input (fading out to "off").
    pub new: Option<SteadyStage>,
    /// Per-sample coefficient increment of `new`, added before each sample.
    pub delta: Option<BiquadCoefficients>,
    /// Crossfade from an old realisation, if one is running.
    pub fade: Option<FadeOut>,
}

/// Like [`run_chain`], with crossfades between two realisations per stage
/// (and per-sample coefficient interpolation). Used only while a crossfade
/// is running.
pub(crate) fn run_changing(
    data: &mut [f32],
    channels: usize,
    stages: &[ChangingStage],
    states: &mut [f64],
) {
    debug_assert!(stages.len() <= MAX_STAGES);
    if stages.is_empty() {
        return;
    }
    for (c0, width) in channel_groups(channels) {
        match width {
            2 => changing_group::<2>(data, channels, c0, stages, states),
            _ => changing_group::<1>(data, channels, c0, stages, states),
        }
    }
}

fn changing_group<const L: usize>(
    data: &mut [f32],
    channels: usize,
    c0: usize,
    stages: &[ChangingStage],
    states: &mut [f64],
) {
    let off = STATE_PER_CHANNEL * c0;
    let n = stages.len();
    let load = |s: Option<SteadyStage>, states: &[f64]| {
        s.map(|s| LaneState::<L>::load(&states[s.state + off..]))
            .unwrap_or_default()
    };
    let mut coefs = [BiquadCoefficients::IDENTITY; MAX_STAGES];
    let mut cur = [LaneState::<L>::default(); MAX_STAGES];
    let mut old = [LaneState::<L>::default(); MAX_STAGES];
    for (k, s) in stages.iter().enumerate() {
        coefs[k] = s.new.map(|n| n.coefs).unwrap_or_default();
        cur[k] = load(s.new, states);
        old[k] = load(s.fade.and_then(|f| f.old), states);
    }
    for (i, frame) in data.chunks_exact_mut(channels).enumerate() {
        let lanes = &mut frame[c0..c0 + L];
        let mut x = Lanes::<L>::load_f32(lanes);
        for (k, s) in stages.iter().enumerate() {
            if let Some(d) = &s.delta {
                coefs[k].add_step(d);
            }
            let y_new = match s.new {
                Some(_) => {
                    let y = cur[k].tick(&coefs[k], x);
                    cur[k].flush_denormals();
                    y
                }
                None => x,
            };
            x = match &s.fade {
                None => y_new,
                Some(f) => {
                    let y_old = match &f.old {
                        Some(o) => {
                            let y = old[k].tick(&o.coefs, x);
                            old[k].flush_denormals();
                            y
                        }
                        None => x,
                    };
                    let g = ((f.done + i + 1) as f64 * f.inv_len).min(1.0);
                    Lanes::blend(y_old, y_new, g)
                }
            };
        }
        x.store_f32(lanes);
    }
    for (k, s) in stages[..n].iter().enumerate() {
        if let Some(st) = s.new {
            cur[k].store(&mut states[st.state + off..]);
        }
        if let Some(o) = s.fade.and_then(|f| f.old) {
            old[k].store(&mut states[o.state + off..]);
        }
    }
}

/// Zero the whole state region if any value in it is non-finite (a NaN/Inf
/// input poisoned it).
pub(crate) fn sanitize_bank(states: &mut [f64]) {
    if states.iter().any(|v| !v.is_finite()) {
        states.iter_mut().for_each(|v| *v = 0.0);
    }
}

// ---------------------------------------------------------------------------
// Public filter.

/// A single biquad filter (Direct Form I, `f64` state) for interleaved `f32`
/// audio with any number of channels; every channel uses the same coefficients
/// and has its own state. See the module documentation.
///
/// Real-time safe after construction: [`Biquad::process`] and the coefficient
/// setters never allocate.
#[derive(Debug, Clone)]
pub struct Biquad {
    coefs: BiquadCoefficients,
    /// Coefficients at the end of a running ramp.
    target: BiquadCoefficients,
    delta: BiquadCoefficients,
    ramp_left: usize,
    channels: usize,
    /// DF-I state, `STATE_PER_CHANNEL · channels` values in lane-group layout.
    state: Vec<f64>,
}

impl Biquad {
    /// Create a filter for `channels` interleaved channels (at least 1) with
    /// the given coefficients and zero state. Allocates the state.
    pub fn new(coefs: BiquadCoefficients, channels: usize) -> Self {
        let channels = channels.max(1);
        Self {
            coefs,
            target: coefs,
            delta: BiquadCoefficients::default(),
            ramp_left: 0,
            channels,
            state: vec![0.0; STATE_PER_CHANNEL * channels],
        }
    }

    /// Number of interleaved channels.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Coefficients in use at the next sample (mid-ramp: the interpolated set).
    pub fn coefficients(&self) -> BiquadCoefficients {
        self.coefs
    }

    /// Coefficients the filter is heading to (equal to
    /// [`Biquad::coefficients`] unless a ramp is running).
    pub fn target_coefficients(&self) -> BiquadCoefficients {
        self.target
    }

    /// Switch to new coefficients at the next sample (cancels any ramp).
    pub fn set_coefficients(&mut self, coefs: BiquadCoefficients) {
        self.coefs = coefs;
        self.target = coefs;
        self.ramp_left = 0;
    }

    /// Interpolate the coefficients linearly, per sample, from the current set
    /// to `coefs` over the next `frames` frames (`0` switches immediately). If
    /// both end points are stable, so is every intermediate filter.
    pub fn ramp_to(&mut self, coefs: BiquadCoefficients, frames: usize) {
        if frames == 0 {
            self.set_coefficients(coefs);
            return;
        }
        self.delta = self.coefs.ramp_step(&coefs, frames);
        self.target = coefs;
        self.ramp_left = frames;
    }

    /// Clear the filter state (coefficients are kept).
    pub fn reset(&mut self) {
        self.state.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Filter interleaved frames in place. A trailing partial frame is left
    /// untouched. Outputs are clamped to the finite `f32` range; if a
    /// non-finite input poisons the state, the state is cleared afterwards.
    pub fn process(&mut self, samples: &mut [f32]) {
        let ch = self.channels;
        let frames = samples.len() / ch;
        let mut data = &mut samples[..frames * ch];
        let stage = [SteadyStage {
            coefs: self.coefs,
            state: 0,
        }];
        if self.ramp_left > 0 && frames > 0 {
            let n = self.ramp_left.min(frames);
            let (head, tail) = data.split_at_mut(n * ch);
            run_chain(head, ch, &stage, &[Some(self.delta)], &mut self.state);
            self.ramp_left -= n;
            if self.ramp_left == 0 {
                self.coefs = self.target;
            } else {
                // Same accumulation as the kernel, so the next call continues seamlessly.
                for _ in 0..n {
                    self.coefs.add_step(&self.delta);
                }
            }
            data = tail;
        }
        let stage = [SteadyStage {
            coefs: self.coefs,
            state: 0,
        }];
        run_chain(data, ch, &stage, &[], &mut self.state);
        sanitize_bank(&mut self.state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 48_000.0;

    fn db(x: f64) -> f64 {
        20.0 * x.log10()
    }

    /// Steady-state gain at `f` (Hz, integer number of cycles per second),
    /// measured on interleaved data with `ch` channels, all channels fed the
    /// same sine with a channel-dependent phase.
    fn measured_gain(c: BiquadCoefficients, ch: usize, f: f64) -> Vec<f64> {
        let mut bq = Biquad::new(c, ch);
        let n = FS as usize;
        let sig = |i: usize, k: usize| (2.0 * PI * f * i as f64 / FS + k as f64).sin() * 0.25;
        let mut warm: Vec<f32> = (0..n)
            .flat_map(|i| (0..ch).map(move |k| sig(i, k) as f32))
            .collect();
        bq.process(&mut warm);
        let mut x: Vec<f32> = (n..2 * n)
            .flat_map(|i| (0..ch).map(move |k| sig(i, k) as f32))
            .collect();
        for chunk in x.chunks_mut(ch * 333) {
            bq.process(chunk);
        }
        (0..ch)
            .map(|k| {
                let (mut s, mut co) = (0.0, 0.0);
                for i in 0..n {
                    let ph = 2.0 * PI * f * (n + i) as f64 / FS + k as f64;
                    s += x[i * ch + k] as f64 * ph.sin();
                    co += x[i * ch + k] as f64 * ph.cos();
                }
                2.0 * (s * s + co * co).sqrt() / n as f64 / 0.25
            })
            .collect()
    }

    #[test]
    fn characteristic_points_from_theory() {
        let m = |c: BiquadCoefficients, f: f64| db(c.magnitude_at(f, FS));
        let f0 = 1000.0;
        // LP/HP: unity in the passband, |H(f0)| = Q.
        assert!(m(BiquadCoefficients::low_pass(f0, 2.0, FS), 0.0).abs() < 1e-9);
        assert!((m(BiquadCoefficients::low_pass(f0, 2.0, FS), f0) - db(2.0)).abs() < 1e-9);
        assert!(m(BiquadCoefficients::high_pass(f0, 2.0, FS), FS / 2.0).abs() < 1e-9);
        assert!((m(BiquadCoefficients::high_pass(f0, 2.0, FS), f0) - db(2.0)).abs() < 1e-9);
        // Band-passes: zero at DC, peak Q resp. 1 at f0.
        let bps = BiquadCoefficients::band_pass_skirt(f0, 4.0, FS);
        assert!((m(bps, f0) - db(4.0)).abs() < 1e-9);
        assert!(bps.magnitude(0.0) < 1e-12);
        assert!(m(BiquadCoefficients::band_pass_peak(f0, 4.0, FS), f0).abs() < 1e-9);
        // Notch: zero at f0, unity at DC.
        let n = BiquadCoefficients::notch(f0, 4.0, FS);
        assert!(n.magnitude_at(f0, FS) < 1e-9);
        assert!(m(n, 0.0).abs() < 1e-9);
        // All-pass: unity everywhere.
        for f in [10.0, 1000.0, 20_000.0] {
            assert!(m(BiquadCoefficients::all_pass(f0, 0.7, FS), f).abs() < 1e-9);
        }
        // Peaking: gain at f0, unity at DC and Nyquist.
        let p = BiquadCoefficients::peaking(f0, -9.0, 1.5, FS);
        assert!((m(p, f0) + 9.0).abs() < 1e-9);
        assert!(m(p, 0.0).abs() < 1e-9 && m(p, FS / 2.0).abs() < 1e-9);
        // Shelves: full gain deep in the shelf, unity on the other side, half at f0.
        let ls = BiquadCoefficients::low_shelf(f0, 12.0, 0.707, FS);
        assert!((m(ls, 0.0) - 12.0).abs() < 1e-9);
        assert!(m(ls, FS / 2.0).abs() < 1e-9);
        assert!((m(ls, f0) - 6.0).abs() < 1e-9);
        let hs = BiquadCoefficients::high_shelf(f0, -12.0, 0.707, FS);
        assert!((m(hs, FS / 2.0) + 12.0).abs() < 1e-9);
        assert!(m(hs, 0.0).abs() < 1e-9);
        assert!((m(hs, f0) + 6.0).abs() < 1e-9);
    }

    #[test]
    fn designs_are_stable_across_the_range() {
        for &fs in &[8_000.0, 48_000.0, 192_000.0] {
            for &f in &[1.0, 20.0, 1000.0, 0.499 * fs] {
                for &q in &[0.01, 0.707, 100.0] {
                    for &g in &[-60.0, 0.0, 60.0] {
                        for c in [
                            BiquadCoefficients::low_pass(f, q, fs),
                            BiquadCoefficients::high_pass(f, q, fs),
                            BiquadCoefficients::band_pass_skirt(f, q, fs),
                            BiquadCoefficients::band_pass_peak(f, q, fs),
                            BiquadCoefficients::notch(f, q, fs),
                            BiquadCoefficients::all_pass(f, q, fs),
                            BiquadCoefficients::peaking(f, g, q, fs),
                            BiquadCoefficients::low_shelf(f, g, q, fs),
                            BiquadCoefficients::high_shelf(f, g, q, fs),
                        ] {
                            assert!(c.is_stable(), "{f} {q} {g} {fs}: {c:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn identity_passes_through_exactly() {
        let mut bq = Biquad::new(BiquadCoefficients::IDENTITY, 3);
        let input: Vec<f32> = (0..300).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut x = input.clone();
        bq.process(&mut x);
        assert_eq!(x, input);
    }

    /// The impulse response obeys the difference equation exactly (checked
    /// against a direct scalar evaluation), for every lane-group layout.
    #[test]
    fn impulse_response_matches_difference_equation() {
        let c = BiquadCoefficients::peaking(3000.0, 9.0, 2.0, FS);
        let n = 64;
        let mut reference = vec![0.0f64; n];
        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
        for (i, r) in reference.iter_mut().enumerate() {
            let x = if i == 0 { 1.0 } else { 0.0 };
            let y = c.b0 * x + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
            (x2, x1, y2, y1) = (x1, x, y1, y);
            *r = y;
        }
        for ch in [1usize, 2, 3, 4, 5, 7, 8] {
            for hot in 0..ch {
                let mut bq = Biquad::new(c, ch);
                let mut x = vec![0.0f32; n * ch];
                x[hot] = 1.0;
                bq.process(&mut x);
                for i in 0..n {
                    for k in 0..ch {
                        let v = x[i * ch + k] as f64;
                        if k == hot {
                            assert!((v - reference[i]).abs() < 1e-6, "ch {ch} {i}");
                        } else {
                            assert_eq!(v, 0.0, "crosstalk ch {ch} lane {k}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn measured_gain_matches_magnitude_for_all_layouts() {
        let c = BiquadCoefficients::high_shelf(2000.0, 6.0, 0.707, FS);
        for ch in [1, 2, 3, 6, 8] {
            for f in [200.0, 2000.0, 10_000.0] {
                for g in measured_gain(c, ch, f) {
                    assert!((db(g) - db(c.magnitude_at(f, FS))).abs() < 0.01);
                }
            }
        }
    }

    #[test]
    fn split_buffers_equal_one_buffer() {
        let c = BiquadCoefficients::low_pass(500.0, 0.707, FS);
        let input: Vec<f32> = (0..2 * 1000)
            .map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0)
            .collect();
        let mut a = input.clone();
        let mut b = input;
        Biquad::new(c, 2).process(&mut a);
        let mut bq = Biquad::new(c, 2);
        for chunk in b.chunks_mut(2 * 37) {
            bq.process(chunk);
        }
        assert_eq!(a, b);
    }

    #[test]
    fn ramp_reaches_target_and_interpolates_linearly() {
        let from = BiquadCoefficients::peaking(1000.0, 0.0, 1.0, FS);
        let to = BiquadCoefficients::peaking(1000.0, 12.0, 1.0, FS);
        let mut bq = Biquad::new(from, 1);
        bq.ramp_to(to, 100);
        let mut x = vec![0.0f32; 40];
        bq.process(&mut x);
        // After 40 of 100 steps the coefficients are 40 % of the way.
        let mid = bq.coefficients();
        assert!((mid.b0 - (from.b0 + 0.4 * (to.b0 - from.b0))).abs() < 1e-12);
        assert!((mid.a2 - (from.a2 + 0.4 * (to.a2 - from.a2))).abs() < 1e-12);
        let mut x = vec![0.0f32; 100];
        bq.process(&mut x);
        assert_eq!(bq.coefficients(), to);
        assert_eq!(bq.target_coefficients(), to);
    }

    #[test]
    fn linear_interpolation_between_stable_filters_is_stable() {
        let a = BiquadCoefficients::low_pass(20.0, 0.1, FS);
        let b = BiquadCoefficients::peaking(20_000.0, 60.0, 100.0, FS);
        let d = a.ramp_step(&b, 1000);
        let mut c = a;
        for _ in 0..1000 {
            c.add_step(&d);
            assert!(c.is_stable(), "{c:?}");
        }
    }

    #[test]
    fn non_finite_input_does_not_poison_later_output() {
        let mut bq = Biquad::new(BiquadCoefficients::low_pass(100.0, 0.707, FS), 2);
        let mut x = [f32::NAN, 0.5, f32::INFINITY, 0.5];
        bq.process(&mut x);
        let mut y = [0.25f32; 64];
        bq.process(&mut y);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    /// After the input stops, the decaying state reaches exactly zero (it is
    /// flushed below 1e-15) instead of lingering in the subnormal range.
    #[test]
    fn decaying_state_is_flushed_to_zero() {
        for ch in [1usize, 2, 3] {
            let mut bq = Biquad::new(BiquadCoefficients::low_pass(5000.0, 0.707, FS), ch);
            let mut x = vec![0.0f32; 4096 * ch];
            x.iter_mut().take(ch).for_each(|v| *v = 1.0);
            bq.process(&mut x);
            assert!(bq.state.iter().all(|&v| v == 0.0), "{:?}", bq.state);
            assert!(x[x.len() - ch..].iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn channel_groups_cover_every_channel_once() {
        for ch in 1..=13 {
            let mut seen = vec![0; ch];
            for (c0, w) in channel_groups(ch) {
                assert!(matches!(w, 1 | 2));
                seen[c0..c0 + w].iter_mut().for_each(|s| *s += 1);
            }
            assert!(seen.iter().all(|&s| s == 1), "{ch}");
        }
    }
}
