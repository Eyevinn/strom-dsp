//! Stereo tools: mode matrix, polarity, constant-power balance, M/S width.
//!
//! [`StereoTools`] processes two interleaved channels (L, R). Every setting is
//! linear, so the whole processor is one 2×2 matrix `out = T · in`, built in
//! this order (see [`matrix_for`]):
//!
//! `T = B · W · M · P`
//!
//! 1. **`P`, polarity** (on the inputs): `diag(±1, ±1)` from `invert_left` /
//!    `invert_right`. Applied first, so a polarity-flipped microphone can be
//!    fixed before a mono sum.
//! 2. **`M`, mode**:
//!
//!    | Mode | Matrix rows (L out, R out) |
//!    |---|---|
//!    | `Stereo` | `[1, 0]`, `[0, 1]` |
//!    | `Mono` | `[k, k]`, `[k, k]`, `k = 1/√2` (−3 dB law) or `1/2` (−6 dB law) |
//!    | `LeftToBoth` | `[1, 0]`, `[1, 0]` |
//!    | `RightToBoth` | `[0, 1]`, `[0, 1]` |
//!    | `Swap` | `[0, 1]`, `[1, 0]` |
//!    | `MsEncode` | `[½, ½]`, `[½, −½]`: L out = M = (L+R)/2, R out = S = (L−R)/2 |
//!    | `MsDecode` | `[1, 1]`, `[1, −1]`: input L = M, input R = S; L = M+S, R = M−S |
//!
//!    **M/S scaling:** `M = (L+R)/2`, `S = (L−R)/2` and the inverse
//!    `L = M+S`, `R = M−S`. Decode·Encode is exactly the identity matrix, so
//!    `ms-encode` followed by `ms-decode` is a round trip (up to f32 rounding
//!    of the intermediate signal). Centre content (`L = R = x`) gives `M = x`.
//! 3. **`W`, width** through M/S, `w ∈ [0, 2]`: `S' = w·S`, i.e.
//!    `W = ½·[[1+w, 1−w], [1−w, 1+w]]`. `w = 0` is mono `(L+R)/2` on both
//!    channels, `w = 1` the identity, `w = 2` doubles the side signal. No gain
//!    compensation: at `w = 2` a hard-left signal of 1 becomes (1.5, −0.5).
//! 4. **`B`, balance** `b ∈ [−1, 1]`, constant-power law normalised to unity
//!    at the centre: `θ = (b + 1)·π/4`, `g_L = √2·cos θ`, `g_R = √2·sin θ`,
//!    so `g_L² + g_R² = 2` for every `b` (power of a signal with equal,
//!    uncorrelated L/R is constant). `b = 0` gives exactly (1, 1), `b = −1`
//!    gives (√2, 0) (+3.01 dB on the remaining side), `b = 1` gives (0, √2).
//!
//! # Exactness
//!
//! When the target matrix is reached and is a *signed permutation-like* matrix
//! (each output row takes exactly one input with gain +1 or −1: stereo, swap,
//! left/right-to-both, polarity, combinations thereof, with balance 0 and
//! width 1), samples are copied or negated directly, so the output is
//! **bit-exact**. The identity (the defaults) does not touch the buffer at all.
//! Otherwise each sample is `T₀₀·L + T₀₁·R` (and so on) in f64, clamped to the
//! finite f32 range and rounded once to f32.
//!
//! # Smoothing and mode switches
//!
//! Any parameter change (mode, mono law, polarity, balance, width, enabled)
//! moves the target matrix. The applied matrix then ramps linearly from its
//! current value to the target over [`RAMP_MS`]. Because the processor is
//! linear, this is exactly a linear crossfade between the old and the new
//! processing, so mode switches crossfade too. A change during a ramp starts a
//! new ramp from the matrix reached so far.
//!
//! **Discontinuity bound.** With `D = max |x[n] − x[n−1]|` and `A = max |x|`
//! over both input channels, and `‖·‖` the maximum absolute row sum, each
//! output step satisfies `|y[n] − y[n−1]| ≤ ‖T_n‖·D + ‖T_n − T_{n−1}‖·A`, and
//! during a ramp of `N` frames `‖T_n − T_{n−1}‖ = ‖T_target − T_start‖ / N`.
//! [`ramp_frames`] gives `N`.
//!
//! `enabled = false` ramps to the identity; once there, the buffer is untouched
//! (bit-exact). As everywhere in this crate, between
//! [`configure`](Processor::configure)/[`reset`](Processor::reset) and the
//! first processed frame new parameters apply instantly (no ramp).
//!
//! Configured with a channel count other than 2, `process` is a no-op.

use crate::Processor;

/// Duration of the linear matrix ramp for parameter changes and mode switches (ms).
pub const RAMP_MS: f32 = 10.0;

/// A 2×2 matrix, `m[row][col]`; row 0 produces L, row 1 produces R.
pub type Matrix = [[f64; 2]; 2];

const IDENTITY: Matrix = [[1.0, 0.0], [0.0, 1.0]];

/// Channel routing mode (the `M` stage).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StereoMode {
    /// Unchanged.
    #[default]
    Stereo,
    /// L+R sum on both channels, scaled by the [`MonoLaw`].
    Mono,
    /// Left input on both outputs.
    LeftToBoth,
    /// Right input on both outputs.
    RightToBoth,
    /// Left and right exchanged.
    Swap,
    /// L/R in, M/S out: `M = (L+R)/2` on the left, `S = (L−R)/2` on the right.
    MsEncode,
    /// M/S in (M left, S right), L/R out: `L = M+S`, `R = M−S`.
    MsDecode,
}

/// Gain applied to the L+R sum in [`StereoMode::Mono`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MonoLaw {
    /// `1/√2` (−3.01 dB): keeps the power of uncorrelated L/R.
    #[default]
    Minus3Db,
    /// `1/2` (−6.02 dB): keeps the level of identical (centre) L/R.
    Minus6Db,
}

impl MonoLaw {
    /// The linear gain of this law.
    pub fn gain(self) -> f64 {
        match self {
            MonoLaw::Minus3Db => std::f64::consts::FRAC_1_SQRT_2,
            MonoLaw::Minus6Db => 0.5,
        }
    }
}

/// Parameters of [`StereoTools`]. The defaults are an exact passthrough.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoParams {
    /// `false`: ramp to the identity (then bit-exact passthrough).
    pub enabled: bool,
    /// Routing mode.
    pub mode: StereoMode,
    /// Gain of the mono sum.
    pub mono_law: MonoLaw,
    /// Invert the polarity of the left input.
    pub invert_left: bool,
    /// Invert the polarity of the right input.
    pub invert_right: bool,
    /// Balance, −1 (left) … 0 (centre) … 1 (right), constant power.
    pub balance: f32,
    /// Stereo width, 0 (mono) … 1 (unchanged) … 2 (wide).
    pub width: f32,
}

impl Default for StereoParams {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: StereoMode::Stereo,
            mono_law: MonoLaw::Minus3Db,
            invert_left: false,
            invert_right: false,
            balance: 0.0,
            width: 1.0,
        }
    }
}

/// Constant-power balance gains `(g_L, g_R)` for `balance ∈ [−1, 1]`
/// (clamped; NaN counts as 0): `θ = (b+1)·π/4`, `g_L = √2·cos θ`,
/// `g_R = √2·sin θ`. Exactly `(1, 1)` at 0, `(√2, 0)` at −1, `(0, √2)` at 1.
pub fn balance_gains(balance: f32) -> (f64, f64) {
    let b = if balance.is_nan() {
        0.0
    } else {
        f64::from(balance).clamp(-1.0, 1.0)
    };
    if b == 0.0 {
        return (1.0, 1.0);
    }
    if b == -1.0 {
        return (std::f64::consts::SQRT_2, 0.0);
    }
    if b == 1.0 {
        return (0.0, std::f64::consts::SQRT_2);
    }
    let theta = (b + 1.0) * std::f64::consts::FRAC_PI_4;
    (
        std::f64::consts::SQRT_2 * theta.cos(),
        std::f64::consts::SQRT_2 * theta.sin(),
    )
}

/// The mode matrix `M` (see the module docs).
pub fn mode_matrix(mode: StereoMode, law: MonoLaw) -> Matrix {
    match mode {
        StereoMode::Stereo => IDENTITY,
        StereoMode::Mono => {
            let k = law.gain();
            [[k, k], [k, k]]
        }
        StereoMode::LeftToBoth => [[1.0, 0.0], [1.0, 0.0]],
        StereoMode::RightToBoth => [[0.0, 1.0], [0.0, 1.0]],
        StereoMode::Swap => [[0.0, 1.0], [1.0, 0.0]],
        StereoMode::MsEncode => [[0.5, 0.5], [0.5, -0.5]],
        StereoMode::MsDecode => [[1.0, 1.0], [1.0, -1.0]],
    }
}

/// The width matrix `W = ½·[[1+w, 1−w], [1−w, 1+w]]`, `w` clamped to
/// `[0, 2]` (NaN counts as 1). Exactly the identity at `w = 1`.
pub fn width_matrix(width: f32) -> Matrix {
    let w = if width.is_nan() {
        1.0
    } else {
        f64::from(width).clamp(0.0, 2.0)
    };
    let a = 0.5 * (1.0 + w);
    let b = 0.5 * (1.0 - w);
    [[a, b], [b, a]]
}

fn mul(a: &Matrix, b: &Matrix) -> Matrix {
    let mut m = [[0.0; 2]; 2];
    for (r, row) in m.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = a[r][0] * b[0][c] + a[r][1] * b[1][c];
        }
    }
    m
}

/// The complete matrix `T = B · W · M · P` for `p` (the identity when
/// disabled). Products by exact 0/±1 entries are exact, so pure routing
/// settings give matrices with entries exactly 0 and ±1.
pub fn matrix_for(p: &StereoParams) -> Matrix {
    if !p.enabled {
        return IDENTITY;
    }
    let pol = [
        [if p.invert_left { -1.0 } else { 1.0 }, 0.0],
        [0.0, if p.invert_right { -1.0 } else { 1.0 }],
    ];
    let (gl, gr) = balance_gains(p.balance);
    let bal = [[gl, 0.0], [0.0, gr]];
    let m = mul(&mode_matrix(p.mode, p.mono_law), &pol);
    let m = mul(&width_matrix(p.width), &m);
    let m = mul(&bal, &m);
    // Normalise −0.0 to 0.0 so the routing classification below is simple.
    m.map(|row| row.map(|v| if v == 0.0 { 0.0 } else { v }))
}

/// Ramp length in frames at `sample_rate` (≥ 1).
pub fn ramp_frames(sample_rate: f32) -> usize {
    let f = (f64::from(RAMP_MS) * f64::from(sample_rate.max(0.0)) / 1000.0).round();
    if f.is_finite() && f >= 1.0 {
        f as usize
    } else {
        1
    }
}

/// How a reached matrix is applied.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Route {
    /// Identity: leave the buffer untouched.
    Identity,
    /// Each output takes one input (index), optionally negated: bit-exact.
    Signed([(usize, bool); 2]),
    /// General matrix in f64.
    General,
}

fn classify(m: &Matrix) -> Route {
    if *m == IDENTITY {
        return Route::Identity;
    }
    let mut route = [(0usize, false); 2];
    for (r, row) in m.iter().enumerate() {
        route[r] = match (row[0], row[1]) {
            (1.0, 0.0) => (0, false),
            (-1.0, 0.0) => (0, true),
            (0.0, 1.0) => (1, false),
            (0.0, -1.0) => (1, true),
            _ => return Route::General,
        };
    }
    Route::Signed(route)
}

#[inline]
fn apply_general(frame: &mut [f32], m: &Matrix) {
    let l = f64::from(frame[0]);
    let r = f64::from(frame[1]);
    let max = f64::from(f32::MAX);
    frame[0] = (m[0][0] * l + m[0][1] * r).clamp(-max, max) as f32;
    frame[1] = (m[1][0] * l + m[1][1] * r).clamp(-max, max) as f32;
}

/// Stereo matrix processor (see the module documentation).
#[derive(Debug, Clone)]
pub struct StereoTools {
    params: StereoParams,
    sample_rate: f32,
    channels: usize,
    /// Matrix applied now (end of the last processed frame).
    current: Matrix,
    target: Matrix,
    /// Matrix at the start of the active ramp.
    start: Matrix,
    ramp_frames: usize,
    /// Frames into the active ramp (0 = none).
    ramp_pos: usize,
    route: Route,
    fresh: bool,
}

impl Default for StereoTools {
    fn default() -> Self {
        Self {
            params: StereoParams::default(),
            sample_rate: 0.0,
            channels: 0,
            current: IDENTITY,
            target: IDENTITY,
            start: IDENTITY,
            ramp_frames: 1,
            ramp_pos: 0,
            route: Route::Identity,
            fresh: true,
        }
    }
}

impl StereoTools {
    /// Unconfigured processor (process is a no-op until [`Processor::configure`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// Current parameters.
    pub fn params(&self) -> &StereoParams {
        &self.params
    }

    /// Set parameters (never allocates). A changed matrix ramps over [`RAMP_MS`].
    pub fn set_params(&mut self, p: &StereoParams) {
        self.params = *p;
        let t = matrix_for(p);
        if t == self.target && !self.fresh {
            return;
        }
        self.target = t;
        if self.fresh {
            self.snap();
        } else if self.current != t {
            self.start = self.current;
            self.ramp_pos = 1;
        } else {
            self.ramp_pos = 0;
            self.route = classify(&t);
        }
    }

    /// The matrix applied to the last processed frame.
    pub fn current_matrix(&self) -> Matrix {
        self.current
    }

    /// The matrix the processor is heading for (`matrix_for(params)`).
    pub fn target_matrix(&self) -> Matrix {
        self.target
    }

    /// Whether a ramp is in progress.
    pub fn is_ramping(&self) -> bool {
        self.ramp_pos > 0
    }

    fn snap(&mut self) {
        self.current = self.target;
        self.start = self.target;
        self.ramp_pos = 0;
        self.route = classify(&self.target);
    }
}

impl Processor for StereoTools {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        self.sample_rate = sample_rate;
        self.channels = channels;
        self.ramp_frames = ramp_frames(sample_rate);
        self.reset();
    }

    fn reset(&mut self) {
        self.target = matrix_for(&self.params);
        self.fresh = true;
        self.snap();
    }

    fn process(&mut self, samples: &mut [f32]) {
        if self.channels != 2 {
            return;
        }
        let usable = samples.len() / 2 * 2;
        if usable == 0 {
            return;
        }
        self.fresh = false;
        let mut frames = samples[..usable].chunks_exact_mut(2);
        if self.ramp_pos > 0 {
            let n = self.ramp_frames as f64;
            let (s, t) = (self.start, self.target);
            for frame in frames.by_ref() {
                let k = self.ramp_pos;
                let m = if k >= self.ramp_frames {
                    t
                } else {
                    let g = k as f64 / n;
                    let mut m = [[0.0; 2]; 2];
                    for r in 0..2 {
                        for c in 0..2 {
                            m[r][c] = s[r][c] + g * (t[r][c] - s[r][c]);
                        }
                    }
                    m
                };
                apply_general(frame, &m);
                self.current = m;
                if k >= self.ramp_frames {
                    self.snap();
                    break;
                }
                self.ramp_pos += 1;
            }
        }
        match self.route {
            Route::Identity => {}
            Route::Signed([(il, nl), (ir, nr)]) => {
                for frame in frames {
                    let (l, r) = (frame[il], frame[ir]);
                    frame[0] = if nl { -l } else { l };
                    frame[1] = if nr { -r } else { r };
                }
            }
            Route::General => {
                let m = self.current;
                for frame in frames {
                    apply_general(frame, &m);
                }
            }
        }
    }
}
