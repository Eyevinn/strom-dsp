//! Final clipper: oversampled soft clipping in which only the **clippings**
//! (what the curve removes) are resampled and filtered, and then subtracted
//! from the programme.
//!
//! # Signal flow (one pass; two passes in series)
//!
//! ```text
//!        ┌─ delay (L) ─────────────────────────────────────────────────┐
//! x ─────┤                                                            (−)─→ y
//!        └─ 8× up → c(u) (antialiased) → 8× down → shaping FIR → low cut ┘
//!                    clippings
//! ```
//!
//! - **Differential clipping** (expired US 6,337,999; US 4,208,548): the
//!   input is upsampled 8× with the shared [`Oversampler`], a soft-knee
//!   static curve `f` is applied, and the **clippings** `c(u) = u − f(u)`
//!   are formed at the high rate. Only the clippings are downsampled and
//!   filtered; the programme itself stays at the base rate, delayed to
//!   match, so it never passes through the resampling filters. Where the
//!   curve does not act the clippings are exactly 0, and the output is the
//!   delayed input bit for bit.
//! - **Curve.** Identity up to the knee start `k = T·10^(−knee/20)`, then
//!   `k + w·tanh((|u| − k)/w)` with `w = T − k`, sign-symmetric: continuous
//!   in value, slope and curvature at the knee, approaching the threshold
//!   `T` but never reaching it.
//! - **Antialiasing.** The clippings are formed with first-order
//!   antiderivative antialiasing (Bilbao, Esqueda, Parker & Välimäki, IEEE
//!   SPL 24(7), 2017): the mean of `c` over the line between two successive
//!   high-rate inputs, from the closed-form antiderivative
//!   `C(u) = e²/2 − w²·ln cosh(e/w)`, `e = |u| − k`. It attenuates what
//!   folds back from near multiples of the high rate, and delays the
//!   clippings by half a high-rate sample, which the shaping FIR takes back.
//!   Why 8× and not 4×: a 15 kHz tone 6 dB over the threshold left a
//!   −79.8 dBFS alias at 4× (−52.5 dBFS without the antialiasing), −95.8
//!   dBFS at 8×. 8× costs 2 more frames of latency per pass.
//! - **Shaping** (expired US 4,460,871; Armstrong 2002): one linear-phase
//!   FIR of [`SHAPE_TAPS`] taps ([`design_shaping_fir`]): a **band limit** at
//!   `bandwidth_hz` whose passband rises toward the cut-off as
//!   `1 + a·(f/f_b)²` with `a` = [`EDGE_RISE`], so cutting the out-of-band
//!   part of the clippings brings less of the peak back ("overshoot
//!   compensation"). Then a **low cut**, −3 dB at `low_cut_hz` (the bass
//!   crossover): a delay-complementary filter `H = 1 − L` with zero delay
//!   (`low_cut_sections`). It falls at 12 dB/octave below the cut, and
//!   above about ten times the cut it is the identity to within 0.3 dB and
//!   a few milliradians, so the clippings stay aligned with the peaks they
//!   remove. A linear-phase FIR low cut at 120 Hz would need several ms of
//!   latency per pass. Low-frequency clippings are not subtracted (the
//!   output's bass band stays the input's, as in US 4,208,548): bass peaks
//!   are left to the true-peak limiter after the clipper (or a bass clipper
//!   before it), and the
//!   bass is not modulated by peaks in the upper bands.
//! - **Two passes** (Armstrong 2002, repeated clipping and filtering):
//!   band-limiting the clippings lets peaks grow back a little; a second,
//!   identical pass on the first pass's output takes most of that off.
//! - **Early-out.** Before upsampling a chunk the clipper checks a hard
//!   bound: `B · max|x|` over the chunk and the interpolator's memory, with
//!   `B` the largest absolute sum of the interpolator's polyphase branches
//!   (computed in `configure`). Below the knee no interpolated sample can
//!   reach the curve, so the resampling is skipped and the clippings are 0,
//!   exactly as if it had run. On return the interpolator is primed with
//!   the frames it missed, so its state is exact.
//!
//! Channels are clipped independently.
//!
//! # Latency
//!
//! Per pass, the oversampler's round trip (49 frames at 8×) plus the FIR's
//! `(SHAPE_TAPS − 1)/2` = 20 frames: 69 frames, so 138 frames (2.875 ms at
//! 48 kHz) for two passes, at every sample rate ([`latency_frames_for`]).
//!
//! # Activity (for the clip budget and for upstream stages)
//!
//! Every `process` call accumulates the energy of the input (delayed to the
//! output), the energy of what the clipper removed, and the number of
//! high-rate samples the first pass's curve touched.
//! [`FinalClipper::activity`] returns these for the most recent `process`
//! call, [`FinalClipper::take_stats`] the sums since its last call. The clip
//! depth is `10·log10(removed energy / input energy)` in dB
//! ([`ClipStats::depth_db`], −200 when nothing was removed);
//! [`ClipStats::is_active`] says whether anything was removed.
//!
//! # Real-time safety
//!
//! `configure` allocates. `set_params`, `process` and `reset` never do.
//! A non-finite input sample produces no clipping (it passes through to the
//! limiter after the clipper, which handles it).

use crate::biquad::{Biquad, BiquadCoefficients};
use crate::dynamics::DelayLine;
use crate::filter::MAX_OCTAVES_PER_MS;
use crate::oversample::{bessel_i0, OversampleFactor, Oversampler};
use crate::Processor;

/// Taps of the shaping FIR (odd: linear phase, integer delay).
pub const SHAPE_TAPS: usize = 41;
/// Delay of the shaping FIR (frames).
pub const SHAPE_DELAY: usize = (SHAPE_TAPS - 1) / 2;
/// Kaiser β of the shaping FIR window (about 50 dB side lobes).
const SHAPE_BETA: f64 = 4.5;
/// Rise `a` of the shaping passband toward its cut-off, `1 + a·(f/f_b)²`
/// (+6 dB at the cut-off). Chosen by measurement (`tests/clipper.rs`): of
/// 0, 0.3, 0.6, 1 and 1.5 it is the smallest that holds band-limited
/// signals driven 4 dB into the clipper (two tones at 19 and 20 kHz,
/// speech-like and pink noise) within +0.3 dB of the threshold.
pub const EDGE_RISE: f64 = 1.0;
/// Passes of clip-and-shape.
pub const PASSES: usize = 2;
/// Default oversampling of the curve.
pub const DEFAULT_OVERSAMPLE: OversampleFactor = OversampleFactor::X8;
/// Range of the threshold (dBFS).
pub const THRESHOLD_RANGE_DB: (f32, f32) = (-24.0, 0.0);
/// Range of the knee width (dB).
pub const KNEE_RANGE_DB: (f32, f32) = (0.5, 6.0);
/// Range of the clippings' band limit (Hz); also kept below 0.46·fs.
pub const BANDWIDTH_RANGE_HZ: (f32, f32) = (8_000.0, 22_000.0);
/// Largest band limit as a fraction of the sample rate.
const BANDWIDTH_MAX_FRACTION: f64 = 0.46;
/// Range of the low cut (Hz).
pub const LOW_CUT_RANGE_HZ: (f32, f32) = (20.0, 1_000.0);
/// Frames per internal chunk.
const CHUNK: usize = 256;
/// Glide of the shaping coefficients after a parameter change (ms).
const GLIDE_MS: f64 = 20.0;
/// Frames of zero clippings after which the low cut's state is cleared and
/// the stage idles (its 2nd-order low-pass has decayed below 1e-15 by then
/// for any cut ≥ 20 Hz).
const LOW_CUT_SETTLE_MS: f64 = 500.0;

/// Final clipper parameters. Units: dB, Hz.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FinalClipperParams {
    /// Threshold `T` (dBFS, −24 … 0): the level the curve approaches.
    pub threshold_db: f32,
    /// Knee width (dB, 0.5 … 6): the curve starts this far below `T`.
    pub knee_db: f32,
    /// Band limit of the clippings (Hz, 8 000 … 22 000, at most 0.46·fs).
    pub bandwidth_hz: f32,
    /// Low cut of the clippings (Hz, 20 … 1 000): the bass crossover.
    pub low_cut_hz: f32,
    /// Apply the low cut (tests switch it off as a guard).
    pub low_cut: bool,
    /// Raise the shaping passband toward the cut-off (overshoot
    /// compensation). Tests switch it off as a guard.
    pub overshoot_compensation: bool,
    /// Clip-and-shape passes that act (1 or 2). An inactive pass is a pure
    /// delay, so the latency does not change.
    pub passes: usize,
}

impl Default for FinalClipperParams {
    fn default() -> Self {
        Self {
            threshold_db: -1.5,
            knee_db: 2.0,
            bandwidth_hz: 20_000.0,
            low_cut_hz: 120.0,
            low_cut: true,
            overshoot_compensation: true,
            passes: PASSES,
        }
    }
}

fn finite_clamp(v: f32, def: f32, lo: f32, hi: f32) -> f32 {
    if v.is_finite() {
        v.clamp(lo, hi)
    } else {
        def
    }
}

impl FinalClipperParams {
    /// Copy with every field in range (non-finite values become defaults).
    pub fn clamped(&self) -> Self {
        let d = Self::default();
        Self {
            threshold_db: finite_clamp(
                self.threshold_db,
                d.threshold_db,
                THRESHOLD_RANGE_DB.0,
                THRESHOLD_RANGE_DB.1,
            ),
            knee_db: finite_clamp(self.knee_db, d.knee_db, KNEE_RANGE_DB.0, KNEE_RANGE_DB.1),
            bandwidth_hz: finite_clamp(
                self.bandwidth_hz,
                d.bandwidth_hz,
                BANDWIDTH_RANGE_HZ.0,
                BANDWIDTH_RANGE_HZ.1,
            ),
            low_cut_hz: finite_clamp(
                self.low_cut_hz,
                d.low_cut_hz,
                LOW_CUT_RANGE_HZ.0,
                LOW_CUT_RANGE_HZ.1,
            ),
            low_cut: self.low_cut,
            overshoot_compensation: self.overshoot_compensation,
            passes: self.passes.clamp(1, PASSES),
        }
    }
}

/// Energies and counts of the clipper's work over some stretch of audio.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ClipStats {
    /// Sum of squares of the input, delayed to the output (all channels).
    pub input_energy: f64,
    /// Sum of squares of what the clipper removed (input delayed − output).
    pub removed_energy: f64,
    /// High-rate samples (all channels) on which the first pass's curve acted.
    pub touched: u64,
    /// High-rate samples (all channels) the first pass looked at.
    pub samples: u64,
}

impl ClipStats {
    /// Clip depth (dB): removed energy relative to input energy. −200 when
    /// nothing was removed or there was no input.
    pub fn depth_db(&self) -> f64 {
        if self.removed_energy > 0.0 && self.input_energy > 0.0 {
            10.0 * (self.removed_energy / self.input_energy).log10()
        } else {
            -200.0
        }
    }

    /// Share (0 … 1) of the first pass's high-rate samples the curve touched.
    pub fn touched_share(&self) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            self.touched as f64 / self.samples as f64
        }
    }

    /// Whether the curve acted (on any sample of the first pass).
    pub fn is_active(&self) -> bool {
        self.touched > 0
    }

    /// Add `other` to these sums.
    pub fn accumulate(&mut self, other: &ClipStats) {
        self.input_energy += other.input_energy;
        self.removed_energy += other.removed_energy;
        self.touched += other.touched;
        self.samples += other.samples;
    }
}

/// Latency (frames) of a [`FinalClipper`] with `factor` oversampling (`None`
/// = no oversampling): `PASSES · (round trip + SHAPE_DELAY)`. 138 frames at
/// 8×, at every sample rate.
pub fn latency_frames_for(factor: Option<OversampleFactor>) -> usize {
    let round_trip = match factor {
        None => 0,
        Some(OversampleFactor::X2) => 39,
        Some(OversampleFactor::X4) => 47,
        Some(OversampleFactor::X8) => 49,
    };
    PASSES * (round_trip + SHAPE_DELAY)
}

/// Design the shaping FIR (see the module docs): `SHAPE_TAPS` taps of a
/// Kaiser-windowed ideal low-pass at `bandwidth_hz` whose passband is
/// `1 + a·(f/f_b)²`, centred `advance` frames before tap `SHAPE_DELAY` (the
/// curve's half-sample delay, see the module docs). With `compensate`,
/// `a` = [`EDGE_RISE`]; otherwise `a = 0` (a plain windowed low-pass). Window method: Oppenheim
/// & Schafer, *Discrete-Time Signal Processing*, §7.5–7.6. The ideal
/// response is integrated in closed form:
/// `h(t) = (2/fs)·∫₀^{f_b} (1 + a f²/f_b²) cos(2πft/fs) df`.
pub fn design_shaping_fir(
    bandwidth_hz: f64,
    sample_rate: f64,
    compensate: bool,
    advance: f64,
) -> [f32; SHAPE_TAPS] {
    let fs = sample_rate;
    let fb = bandwidth_hz.clamp(1.0, BANDWIDTH_MAX_FRACTION * fs);
    let a = if compensate { EDGE_RISE } else { 0.0 };
    let i0b = bessel_i0(SHAPE_BETA);
    let half = SHAPE_DELAY as f64 + 0.5;
    let mut h = [0.0f32; SHAPE_TAPS];
    for (i, tap) in h.iter_mut().enumerate() {
        let n = i as f64 - SHAPE_DELAY as f64 + advance;
        let ideal = if n.abs() < 1e-9 {
            2.0 / fs * (fb + a * fb / 3.0)
        } else {
            let th = 2.0 * core::f64::consts::PI * n / fs;
            let (s, c) = (th * fb).sin_cos();
            let i0 = s / th;
            let i2 = fb * fb * s / th + 2.0 * fb * c / (th * th) - 2.0 * s / (th * th * th);
            2.0 / fs * (i0 + a / (fb * fb) * i2)
        };
        let r = n / half;
        let w = bessel_i0(SHAPE_BETA * (1.0 - r * r).max(0.0).sqrt()) / i0b;
        *tap = (ideal * w) as f32;
    }
    h
}

/// Corner of the low cut relative to its −3 dB frequency: `|H|` of
/// [`low_cut_sections`] is −3 dB at 0.61 of its corner.
const LOW_CUT_CORNER: f64 = 1.0 / 0.61;

/// The low cut's two sections, −3 dB at `cut_hz` (bilinear, pre-warped at
/// the corner `ω_c` = [`LOW_CUT_CORNER`]·`cut_hz`):
/// `H(s) = s²(s + 3)/(s + 1)³` in `s/ω_c`, as `s²/(s + 1)²` (the cookbook
/// high-pass at Q = ½) times the shelf `(s + 3)/(s + 1)`.
/// `H = 1 − L` with `L(s) = (1 + 3s)/(1 + s)³`, a low-pass that matches 1
/// to first order at DC, so `H` falls at 12 dB/octave below the cut. At
/// high frequencies `L ≈ −3/s²` is real to within `O(s⁻³)`, so `H` is the
/// identity to within `3(f_c/f)²` in gain and about `(f_c/f)³` rad in
/// phase: the clippings stay aligned with the peaks they remove, at no
/// latency. The price of no latency is a gain rise of up to +2.3 dB around
/// twice the corner (the complement of a low-pass cannot be flat and
/// causal at once). With a cut at 120 Hz: −15 dB at 50 Hz, +2.3 dB near
/// 400 Hz, +0.6 dB at 1.2 kHz, within 0.3 dB above 2 kHz.
fn low_cut_sections(cut_hz: f64, fs: f64) -> [BiquadCoefficients; 2] {
    let fc = (cut_hz * LOW_CUT_CORNER).clamp(1.0, 0.45 * fs);
    let k = (core::f64::consts::PI * fc / fs).tan();
    [
        BiquadCoefficients::high_pass(fc, 0.5, fs),
        BiquadCoefficients::from_unnormalized(
            1.0 + 3.0 * k,
            3.0 * k - 1.0,
            0.0,
            1.0 + k,
            k - 1.0,
            0.0,
        ),
    ]
}

/// The soft-knee curve, as its clipping `c(u) = u − f(u)`: 0 up to the knee
/// start `k`, then `sign(u)·(e − w·tanh(e/w))` with `e = |u| − k`.
#[derive(Debug, Clone, Copy)]
struct Curve {
    k: f64,
    w: f64,
}

/// `ln(cosh(x))` for `x ≥ 0`, without overflow.
#[inline(always)]
fn ln_cosh(x: f64) -> f64 {
    x + (-2.0 * x).exp().ln_1p() - core::f64::consts::LN_2
}

impl Curve {
    fn new(threshold_db: f32, knee_db: f32) -> Self {
        let t = 10f64.powf(threshold_db as f64 / 20.0);
        let k = t * 10f64.powf(-(knee_db as f64) / 20.0);
        Self { k, w: t - k }
    }

    /// The clipping `c(u)`.
    #[inline(always)]
    fn clip(&self, u: f64) -> f64 {
        let e = u.abs() - self.k;
        if e > 0.0 {
            (e - self.w * (e / self.w).tanh()).copysign(u)
        } else {
            0.0
        }
    }

    /// Antiderivative `C(u) = ∫₀ᵘ c` (even): `e²/2 − w²·ln cosh(e/w)`.
    #[inline(always)]
    fn integral(&self, u: f64) -> f64 {
        let e = u.abs() - self.k;
        if e > 0.0 {
            0.5 * e * e - self.w * self.w * ln_cosh(e / self.w)
        } else {
            0.0
        }
    }

    /// First-order antiderivative antialiasing (Bilbao, Esqueda, Parker &
    /// Välimäki, IEEE SPL 24(7), 2017): the mean of `c` over the straight
    /// line from `prev` to `u`, `(C(u) − C(prev))/(u − prev)`, or `c` at the
    /// midpoint when the two are within `1e-6`. Exactly 0 while both are
    /// below the knee. Non-finite results give 0.
    #[inline(always)]
    fn adaa(&self, u: f32, prev: f32) -> f32 {
        let (u, p) = (u as f64, prev as f64);
        if u.abs() <= self.k && p.abs() <= self.k {
            return 0.0;
        }
        let d = u - p;
        let y = if d.abs() > 1e-6 {
            (self.integral(u) - self.integral(p)) / d
        } else {
            self.clip(0.5 * (u + p))
        };
        if y.is_finite() {
            y as f32
        } else {
            0.0
        }
    }
}

/// Replace the interleaved curve inputs in `buf` by their antialiased
/// clippings; `prev` holds each channel's previous input. The curve moves
/// linearly (in amplitude) from `from` to `to` over the block, so a
/// threshold glide has no steps. Returns how many samples have a non-zero
/// clipping.
fn clip_frames(from: &Curve, to: &Curve, buf: &mut [f32], prev: &mut [f32]) -> u64 {
    let ch = prev.len();
    let frames = buf.len() / ch;
    let steady = from.k == to.k && from.w == to.w;
    let mut touched = 0u64;
    for (i, frame) in buf.chunks_exact_mut(ch).enumerate() {
        let curve = if steady {
            *to
        } else {
            let t = (i + 1) as f64 / frames as f64;
            Curve {
                k: from.k + (to.k - from.k) * t,
                w: from.w + (to.w - from.w) * t,
            }
        };
        let k = curve.k as f32;
        for (v, p) in frame.iter_mut().zip(prev.iter_mut()) {
            let u = *v;
            let c = if u.abs() <= k && p.abs() <= k {
                0.0
            } else {
                curve.adaa(u, *p)
            };
            *p = u;
            touched += (c != 0.0) as u64;
            *v = c;
        }
    }
    touched
}

/// One clip-and-shape pass.
#[derive(Debug)]
struct Pass {
    os: Oversampler,
    /// Main path: the pass's input, delayed by `latency`.
    main: DelayLine,
    /// The pass's input, for priming the interpolator after an idle stretch.
    history: DelayLine,
    /// Shaping FIR history plus block, per channel (`ch × fir_stride`).
    fir_buf: Vec<f32>,
    fir_stride: usize,
    /// Low cut: a 2nd-order high-pass and a first-order shelf in series.
    low_cut: [Biquad; 2],
    /// Previous (high-rate) curve input per channel, for the antialiasing.
    prev: Vec<f32>,
    /// The interpolator's state reflects every input frame so far.
    up_live: bool,
    /// Frames since the last non-zero high-rate clipping.
    since_clip: usize,
    /// Frames since the last non-zero clipping into the FIR.
    since_fir: usize,
    /// Frames since the last non-zero sample out of the FIR.
    since_low_cut: usize,
}

/// The final clipper. See the module documentation.
#[derive(Debug)]
pub struct FinalClipper {
    params: FinalClipperParams,
    factor: Option<OversampleFactor>,
    sample_rate: f64,
    channels: usize,
    passes: [Pass; PASSES],
    /// Base frames each pass's input history must hold to prime the
    /// interpolator (its memory).
    prime_frames: usize,
    /// Frames of zero clippings the decimator needs to hold only zeros.
    down_span: usize,
    /// `B`: bound of the interpolator's gain (largest polyphase absolute sum).
    interp_bound: f32,
    /// Latency of one pass (frames).
    pass_latency: usize,
    /// Input delayed by the whole latency, for the removed-energy meter.
    reference: DelayLine,
    /// Current and target shaping coefficients and remaining glide (frames).
    fir: [f32; SHAPE_TAPS],
    fir_target: [f32; SHAPE_TAPS],
    fir_glide: usize,
    low_cut_coefs: [BiquadCoefficients; 2],
    /// Low-cut frequency in force (Hz), gliding toward the parameter.
    low_cut_cur_hz: f64,
    low_cut_settle: usize,
    /// Scratch.
    up: Vec<f32>,
    clip: Vec<f32>,
    delayed: Vec<f32>,
    prime: Vec<f32>,
    /// Stats of the most recent `process` call and since `take_stats`.
    last: ClipStats,
    total: ClipStats,
    fresh: bool,
    /// Threshold and knee in force (dB), gliding toward the parameters.
    cur_threshold_db: f32,
    cur_knee_db: f32,
    /// Remaining glide of the curve (frames).
    curve_glide: usize,
    /// The curve at the start and the end of the current chunk.
    chunk_curves: (Curve, Curve),
    early_out: bool,
}

impl Default for FinalClipper {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass {
    fn new() -> Self {
        Self {
            os: Oversampler::new(),
            main: DelayLine::default(),
            history: DelayLine::default(),
            fir_buf: Vec::new(),
            fir_stride: 0,
            low_cut: [
                Biquad::new(BiquadCoefficients::IDENTITY, 1),
                Biquad::new(BiquadCoefficients::IDENTITY, 1),
            ],
            prev: Vec::new(),
            up_live: false,
            since_clip: usize::MAX / 2,
            since_fir: usize::MAX / 2,
            since_low_cut: usize::MAX / 2,
        }
    }

    fn reset(&mut self) {
        self.os.reset();
        self.main.clear();
        self.history.clear();
        self.fir_buf.fill(0.0);
        self.low_cut.iter_mut().for_each(Biquad::reset);
        self.prev.fill(0.0);
        self.up_live = true;
        self.since_clip = usize::MAX / 2;
        self.since_fir = usize::MAX / 2;
        self.since_low_cut = usize::MAX / 2;
    }
}

impl FinalClipper {
    /// An unconfigured clipper (a no-op until `configure`) with default
    /// parameters and 8× oversampling.
    pub fn new() -> Self {
        let mut s = Self {
            params: FinalClipperParams::default(),
            factor: Some(DEFAULT_OVERSAMPLE),
            sample_rate: 48_000.0,
            channels: 0,
            passes: [Pass::new(), Pass::new()],
            prime_frames: 0,
            down_span: 0,
            interp_bound: 1.0,
            pass_latency: 0,
            reference: DelayLine::default(),
            fir: [0.0; SHAPE_TAPS],
            fir_target: [0.0; SHAPE_TAPS],
            fir_glide: 0,
            low_cut_coefs: [BiquadCoefficients::IDENTITY; 2],
            low_cut_cur_hz: 120.0,
            low_cut_settle: 0,
            up: Vec::new(),
            clip: Vec::new(),
            delayed: Vec::new(),
            prime: Vec::new(),
            last: ClipStats::default(),
            total: ClipStats::default(),
            fresh: true,
            cur_threshold_db: 0.0,
            cur_knee_db: 0.0,
            curve_glide: 0,
            chunk_curves: (Curve { k: 1.0, w: 0.1 }, Curve { k: 1.0, w: 0.1 }),
            early_out: true,
        };
        s.set_params(&FinalClipperParams::default());
        s
    }

    /// Oversampling of the curve (`None` = none, at the base rate; tests
    /// use it as a guard). Structural: takes effect at the next `configure`.
    pub fn set_oversampling(&mut self, factor: Option<OversampleFactor>) {
        self.factor = factor;
    }

    /// Update parameters (clamped). Never allocates. A change of
    /// bandwidth, compensation or low cut glides over 20 ms.
    pub fn set_params(&mut self, params: &FinalClipperParams) {
        let p = params.clamped();
        let old = self.params;
        self.params = p;
        let fs = self.sample_rate;
        if self.fresh {
            self.cur_threshold_db = p.threshold_db;
            self.cur_knee_db = p.knee_db;
            self.curve_glide = 0;
        } else if p.threshold_db != old.threshold_db || p.knee_db != old.knee_db {
            self.curve_glide = (GLIDE_MS * 0.001 * fs).round() as usize;
        }
        if p.bandwidth_hz != old.bandwidth_hz
            || p.overshoot_compensation != old.overshoot_compensation
            || self.fresh
        {
            self.fir_target = design_shaping_fir(
                p.bandwidth_hz as f64,
                fs,
                p.overshoot_compensation,
                self.advance(),
            );
            if self.fresh {
                self.fir = self.fir_target;
                self.fir_glide = 0;
            } else {
                self.fir_glide = (GLIDE_MS * 0.001 * fs).round() as usize;
            }
        }
        if self.fresh {
            // A change while running glides in `process_chunk`.
            self.low_cut_cur_hz = p.low_cut_hz as f64;
            self.low_cut_coefs = low_cut_sections(self.low_cut_cur_hz, fs);
            for pass in &mut self.passes {
                for (b, c) in pass.low_cut.iter_mut().zip(self.low_cut_coefs) {
                    b.set_coefficients(c);
                }
            }
        }
    }

    /// Current (clamped) parameters.
    pub fn params(&self) -> &FinalClipperParams {
        &self.params
    }

    /// Stats of the most recent `process` call: clip depth, activity and
    /// touched share of that block ([`ClipStats::depth_db`],
    /// [`ClipStats::is_active`], [`ClipStats::touched_share`]).
    pub fn activity(&self) -> ClipStats {
        self.last
    }

    /// Sums since the previous call (or `configure`/`reset`), then restart.
    pub fn take_stats(&mut self) -> ClipStats {
        core::mem::take(&mut self.total)
    }

    /// Switch the early-out off (`false`) or on (the default). The output is
    /// the same either way; tests use it to prove that, benchmarks to
    /// measure the worst case.
    pub fn set_early_out(&mut self, on: bool) {
        self.early_out = on;
    }

    /// The interpolator gain bound `B` (see the module docs).
    pub fn interpolator_bound(&self) -> f32 {
        self.interp_bound
    }

    /// The curve's antialiasing delays the clippings by half a high-rate
    /// sample; the shaping FIR takes it back (frames).
    fn advance(&self) -> f64 {
        0.5 / self.factor.map_or(1, OversampleFactor::ratio) as f64
    }

    /// Process one chunk of at most `CHUNK` frames through pass `k`.
    fn run_pass(&mut self, k: usize, x: &mut [f32]) -> u64 {
        let ch = self.channels;
        let n = x.len() / ch;
        let p = self.params;
        let (from, to) = self.chunk_curves;
        let knee = from.k.min(to.k) as f32;
        let ratio = self.factor.map_or(1, OversampleFactor::ratio);
        let lat = self.pass_latency;
        let bound = self.interp_bound;
        let prime_frames = self.prime_frames;
        let down_span = self.down_span;
        let fir = &self.fir;
        let pass = &mut self.passes[k];
        let active = k < p.passes;

        // Main path: x delayed by the pass latency.
        let delayed = &mut self.delayed[..n * ch];
        pass.main.write_frames(x);
        pass.main.read_frames(lat, delayed);
        if !active {
            x.copy_from_slice(delayed);
            return 0;
        }

        // Hard bound on every interpolated sample this chunk can produce.
        let mut peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if self.factor.is_some() {
            let hist = &mut self.prime[..prime_frames * ch];
            pass.history.read_frames(0, hist);
            peak = hist.iter().fold(peak, |m, v| m.max(v.abs()));
        }
        // NaN fails the comparison and runs the full path.
        let quiet = self.early_out && peak * bound < knee;

        let clip = &mut self.clip[..n * ch];
        let mut touched = 0u64;
        if quiet && pass.since_clip >= down_span {
            // No sample can reach the curve, and the decimator holds only
            // zeros: the clippings are exactly 0.
            if self.factor.is_some() {
                pass.up_live = false;
            } else {
                pass.prev.copy_from_slice(&x[(n - 1) * ch..]);
            }
            clip.fill(0.0);
        } else if let Some(_f) = self.factor {
            if !pass.up_live {
                // Replay the frames the interpolator missed (its memory).
                let hist = &mut self.prime[..prime_frames * ch];
                pass.history.read_frames(0, hist);
                let up = &mut self.up[..prime_frames * ratio * ch];
                pass.os.upsample(hist, up);
                pass.prev.copy_from_slice(&up[up.len() - ch..]);
                pass.up_live = true;
            }
            let up = &mut self.up[..n * ratio * ch];
            pass.os.upsample(x, up);
            touched = clip_frames(&from, &to, up, &mut pass.prev);
            pass.os.downsample(up, clip);
        } else {
            clip.copy_from_slice(x);
            touched = clip_frames(&from, &to, clip, &mut pass.prev);
        }
        if touched > 0 {
            pass.since_clip = 0;
        } else {
            pass.since_clip = pass.since_clip.saturating_add(n);
        }
        if self.factor.is_some() {
            pass.history.write_frames(x);
        }

        // Shaping FIR (per channel; history of SHAPE_TAPS − 1 frames).
        let any = clip.iter().any(|&v| v != 0.0);
        if any {
            pass.since_fir = 0;
        } else {
            pass.since_fir = pass.since_fir.saturating_add(n);
        }
        let fir_live = any || pass.since_fir < SHAPE_TAPS + n;
        if fir_live {
            let h = SHAPE_TAPS - 1;
            let stride = pass.fir_stride;
            for c in 0..ch {
                let buf = &mut pass.fir_buf[c * stride..(c + 1) * stride];
                for i in 0..n {
                    buf[h + i] = clip[i * ch + c];
                }
                for i in 0..n {
                    // y[n] = Σ_j h[j]·e[n − j]; the window's last entry is e[n].
                    let win = &buf[i..i + SHAPE_TAPS];
                    let acc = win
                        .iter()
                        .rev()
                        .zip(fir.iter())
                        .fold(0.0f32, |a, (&e, &h)| a + e * h);
                    clip[i * ch + c] = acc;
                }
                buf.copy_within(n..n + h, 0);
            }
        } else {
            clip.fill(0.0);
        }

        // Low cut.
        if p.low_cut {
            let any = fir_live && clip.iter().any(|&v| v != 0.0);
            if any {
                pass.since_low_cut = 0;
            } else {
                pass.since_low_cut = pass.since_low_cut.saturating_add(n);
            }
            if pass.since_low_cut < self.low_cut_settle {
                for b in &mut pass.low_cut {
                    b.process(clip);
                }
            } else if pass.since_low_cut < self.low_cut_settle + n {
                // Settled: clear the decayed state once.
                pass.low_cut.iter_mut().for_each(Biquad::reset);
            }
        }

        for ((o, &d), &c) in x.iter_mut().zip(delayed.iter()).zip(clip.iter()) {
            *o = d - c;
        }
        touched
    }

    fn process_chunk(&mut self, x: &mut [f32]) {
        let ch = self.channels;
        let n = x.len() / ch;
        // Glide the curve (linear in dB per chunk, linear in amplitude
        // within it) and the shaping coefficients.
        let from = Curve::new(self.cur_threshold_db, self.cur_knee_db);
        if self.curve_glide > 0 {
            let step = n.min(self.curve_glide) as f32 / self.curve_glide as f32;
            self.cur_threshold_db += (self.params.threshold_db - self.cur_threshold_db) * step;
            self.cur_knee_db += (self.params.knee_db - self.cur_knee_db) * step;
            self.curve_glide -= n.min(self.curve_glide);
            if self.curve_glide == 0 {
                self.cur_threshold_db = self.params.threshold_db;
                self.cur_knee_db = self.params.knee_db;
            }
        }
        self.chunk_curves = (from, Curve::new(self.cur_threshold_db, self.cur_knee_db));
        // The low cut glides in log frequency at most 0.05 octave/ms (the
        // crossover's rate), its coefficients ramped over the chunk.
        let target = self.params.low_cut_hz as f64;
        if self.low_cut_cur_hz != target {
            let max = MAX_OCTAVES_PER_MS * 1000.0 * n as f64 / self.sample_rate;
            let oct = (target / self.low_cut_cur_hz).log2().clamp(-max, max);
            self.low_cut_cur_hz = if (target / self.low_cut_cur_hz).log2().abs() <= max {
                target
            } else {
                self.low_cut_cur_hz * oct.exp2()
            };
            self.low_cut_coefs = low_cut_sections(self.low_cut_cur_hz, self.sample_rate);
            for pass in &mut self.passes {
                for (b, c) in pass.low_cut.iter_mut().zip(self.low_cut_coefs) {
                    b.ramp_to(c, n);
                }
            }
        }
        if self.fir_glide > 0 {
            let step = n.min(self.fir_glide) as f32 / self.fir_glide as f32;
            for (c, t) in self.fir.iter_mut().zip(self.fir_target.iter()) {
                *c += (t - *c) * step;
            }
            self.fir_glide -= n.min(self.fir_glide);
            if self.fir_glide == 0 {
                self.fir = self.fir_target;
            }
        }
        self.reference.write_frames(x);
        let touched = self.run_pass(0, x);
        self.run_pass(1, x);
        // Meter: input delayed to the output against what was removed.
        let lat = PASSES * self.pass_latency;
        let r = &mut self.delayed[..n * ch];
        self.reference.read_frames(lat, r);
        let (mut e_in, mut e_rm) = (0.0f64, 0.0f64);
        for (&a, &y) in r.iter().zip(x.iter()) {
            let d = (a - y) as f64;
            e_in += (a as f64) * (a as f64);
            e_rm += d * d;
        }
        let ratio = self.factor.map_or(1, OversampleFactor::ratio) as u64;
        let s = ClipStats {
            input_energy: if e_in.is_finite() { e_in } else { 0.0 },
            removed_energy: if e_rm.is_finite() { e_rm } else { 0.0 },
            touched,
            samples: (n * ch) as u64 * ratio,
        };
        self.last.accumulate(&s);
        self.total.accumulate(&s);
    }

    /// Clip depth (dB) of the most recent `process` call (−200 when it
    /// removed nothing).
    pub fn depth_db(&self) -> f64 {
        self.last.depth_db()
    }
}

/// Impulse response of the `factor` interpolator at the high rate
/// and the length (base frames) of the decimator's response.
fn interpolator_response(factor: OversampleFactor) -> (Vec<f32>, usize) {
    let r = factor.ratio();
    let len = 128;
    let mut os = Oversampler::new();
    os.configure(factor, 1);
    let mut x = vec![0.0f32; len];
    x[0] = 1.0;
    let mut up = vec![0.0f32; len * r];
    os.upsample(&x, &mut up);
    // The decimator's memory: the longest response to a high-rate impulse
    // at any phase.
    let mut down_len = 0;
    for phase in 0..r {
        os.reset();
        let mut hi = vec![0.0f32; len * r];
        hi[phase] = 1.0;
        let mut dn = vec![0.0f32; len];
        os.downsample(&hi, &mut dn);
        let l = dn.iter().rposition(|&v| v != 0.0).map_or(0, |i| i + 1);
        down_len = down_len.max(l);
    }
    (up, down_len)
}

impl Processor for FinalClipper {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        self.channels = channels;
        if channels == 0 {
            return;
        }
        self.sample_rate = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate as f64
        } else {
            48_000.0
        };
        let ratio = self.factor.map_or(1, OversampleFactor::ratio);
        let (round_trip, prime, down_span, bound) = match self.factor {
            Some(f) => {
                let (h, down_len) = interpolator_response(f);
                let up_len = h.iter().rposition(|&v| v != 0.0).map_or(0, |i| i + 1);
                let prime = up_len.div_ceil(ratio) + 1;
                let bound = (0..ratio)
                    .map(|p| {
                        h.iter()
                            .skip(p)
                            .step_by(ratio)
                            .map(|v| v.abs())
                            .sum::<f32>()
                    })
                    .fold(0.0f32, f32::max);
                let mut os = Oversampler::new();
                os.configure(f, 1);
                (os.latency_frames(), prime, down_len + 2, bound * 1.0001)
            }
            None => (0, 0, 0, 1.0),
        };
        self.prime_frames = prime;
        self.down_span = down_span;
        self.interp_bound = bound;
        self.pass_latency = round_trip + SHAPE_DELAY;
        debug_assert_eq!(PASSES * self.pass_latency, latency_frames_for(self.factor));
        let stride = SHAPE_TAPS - 1 + CHUNK;
        for pass in &mut self.passes {
            if let Some(f) = self.factor {
                pass.os.configure(f, channels);
            }
            pass.main.allocate(self.pass_latency + CHUNK, channels);
            pass.history.allocate(prime.max(1) + CHUNK, channels);
            pass.fir_buf = vec![0.0; stride * channels];
            pass.fir_stride = stride;
            pass.low_cut = self.low_cut_coefs.map(|c| Biquad::new(c, channels));
            pass.prev = vec![0.0; channels];
        }
        self.reference
            .allocate(PASSES * self.pass_latency + CHUNK, channels);
        self.up = vec![0.0; (CHUNK.max(prime) * ratio).max(CHUNK) * channels];
        self.clip = vec![0.0; CHUNK * channels];
        self.delayed = vec![0.0; CHUNK * channels];
        self.prime = vec![0.0; prime.max(1) * channels];
        self.low_cut_settle = (LOW_CUT_SETTLE_MS * 0.001 * self.sample_rate).round() as usize;
        self.fresh = true;
        let p = self.params;
        self.set_params(&p);
        self.reset();
    }

    fn process(&mut self, samples: &mut [f32]) {
        let ch = self.channels;
        if ch == 0 || self.clip.is_empty() {
            return;
        }
        let usable = samples.len() / ch * ch;
        self.fresh = false;
        self.last = ClipStats::default();
        for chunk in samples[..usable].chunks_mut(CHUNK * ch) {
            self.process_chunk(chunk);
        }
    }

    fn reset(&mut self) {
        if self.channels == 0 {
            return;
        }
        for p in &mut self.passes {
            p.reset();
        }
        self.reference.clear();
        self.fir = self.fir_target;
        self.fir_glide = 0;
        self.cur_threshold_db = self.params.threshold_db;
        self.cur_knee_db = self.params.knee_db;
        self.curve_glide = 0;
        self.last = ClipStats::default();
        self.total = ClipStats::default();
    }

    fn latency_frames(&self) -> usize {
        latency_frames_for(self.factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaping_fir_is_linear_phase_with_unit_dc() {
        for comp in [true, false] {
            let h = design_shaping_fir(20_000.0, 48_000.0, comp, 0.0);
            for i in 0..SHAPE_DELAY {
                assert_eq!(h[i], h[SHAPE_TAPS - 1 - i]);
            }
            let dc: f32 = h.iter().sum();
            assert!((dc - 1.0).abs() < 0.02, "{dc}");
        }
        let h = design_shaping_fir(20_000.0, 48_000.0, true, 0.0);
        let a = 1.0 + EDGE_RISE / 3.0;
        assert!((h[SHAPE_DELAY] as f64 - 2.0 * 20.0 / 48.0 * a).abs() < 1e-6);
    }

    #[test]
    fn latency_is_as_documented() {
        assert_eq!(latency_frames_for(Some(OversampleFactor::X4)), 134);
        assert_eq!(latency_frames_for(Some(OversampleFactor::X8)), 138);
        let mut c = FinalClipper::new();
        c.configure(48_000.0, 2);
        assert_eq!(c.latency_frames(), 138);
        assert!(c.interpolator_bound() > 1.0 && c.interpolator_bound() < 2.5);
    }

    #[test]
    fn adaa_matches_curve_for_slow_input() {
        let c = Curve::new(-1.0, 2.0);
        for i in 0..200 {
            let u = -2.0 + 0.02 * i as f64;
            let y = c.adaa(u as f32, (u - 1e-4) as f32) as f64;
            let m = c.clip(u - 0.5e-4);
            assert!((y - m).abs() < 1e-5, "{u}: {y} vs {m}");
        }
        assert_eq!(c.adaa(0.5, 0.4), 0.0);
        assert_eq!(c.adaa(f32::NAN, 0.4), 0.0);
        assert_eq!(c.adaa(f32::INFINITY, 0.4), 0.0);
    }

    #[test]
    fn low_cut_response() {
        let fs = 48_000.0;
        let [a, b] = low_cut_sections(120.0, fs);
        let db = |f: f64| 20.0 * (a.magnitude_at(f, fs) * b.magnitude_at(f, fs)).log10();
        println!(
            "50 Hz {:.2}, 120 Hz {:.2}, 1.2 kHz {:.2}, 2 kHz {:.2}",
            db(50.0),
            db(120.0),
            db(1200.0),
            db(2000.0)
        );
        assert!((db(120.0) + 3.0).abs() < 0.2, "{}", db(120.0));
        assert!(db(50.0) < -14.0, "{}", db(50.0));
        let peak = (100..4000).map(|f| db(f as f64)).fold(f64::MIN, f64::max);
        assert!(peak < 2.4, "{peak}");
        assert!(db(2000.0).abs() < 0.3 && db(5000.0).abs() < 0.05);
    }
}
