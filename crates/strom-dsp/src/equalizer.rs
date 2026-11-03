//! Parametric equalizer: up to 8 RBJ-cookbook biquad bands in series.
//!
//! # Design
//!
//! - **Coefficients** follow the equations of R. Bristow-Johnson's "Audio EQ
//!   Cookbook" (W3C Note, public domain): `w0 = 2π·f0/fs`, `α = sin(w0)/(2Q)`,
//!   `A = 10^(gain/40)`, with the cookbook Q form for the shelves as well.
//!   `1 − cos w0` and `1 + cos w0` are evaluated as `2·sin²(w0/2)` and
//!   `2·cos²(w0/2)` (same value, no cancellation at very low frequencies).
//! - **Structure**: Direct Form I with `f64` coefficients and `f64` state per
//!   band and channel (J. O. Smith, *Introduction to Digital Filters*). DF-I
//!   keeps only past inputs and outputs as state, so coefficient changes never
//!   leave internal state inconsistent with the new coefficients.
//! - **Frequency clamp**: `1 Hz ≤ f0 ≤ min(24 kHz, 0.499·fs)`. The cookbook
//!   filters are stable for every `0 < w0 < π`, `Q > 0`, `A > 0`; the clamp keeps
//!   a safe margin from `w0 = π` where low-pass/all-pass poles reach `z = −1`.
//! - **Parameter changes** (frequency, gain, Q of a band whose type stays the
//!   same): the parameters are smoothed by a one-pole smoother with a 10 ms
//!   time constant in the domains `ln f`, dB and `ln Q`. Every 32 frames the
//!   coefficients for the smoothed parameters are recomputed and the
//!   coefficients are linearly interpolated per sample across those 32 frames.
//!   Every intermediate coefficient set is stable: each endpoint is a stable
//!   cookbook filter, and the stability triangle `|a2| < 1, |a1| < 1 + a2` is
//!   convex, so a linear interpolation of two stable denominators is stable.
//! - **Type changes / enabling / disabling** (including bands moving across
//!   `num_bands`): the band's output is crossfaded linearly over 10 ms from the
//!   old realisation (old filter with its state, or the dry input for "off")
//!   to the new one (new filter starting from zero state, or the dry input).
//!   A further type change requested during a crossfade starts when the running
//!   crossfade has finished.
//! - **Cost / bit-exactness**: a band whose type is Off, which is disabled or
//!   whose index is `≥ num_bands` is skipped entirely once any fade-out has
//!   completed, so it costs nothing and leaves samples bit-identical.
//! - **Initial parameters**: after [`Processor::configure`] or
//!   [`Processor::reset`], the first `process` call takes the current
//!   parameters immediately (no smoothing or crossfade from stale values).
//! - **Denormals**: each biquad output (the only recursive state) is flushed
//!   to zero below 1e-15 with [`flush_denormal`]. Inputs are `f32` samples
//!   promoted to `f64`, which can never be `f64` subnormals.
//! - **Robustness**: samples written back are clamped to the finite `f32`
//!   range, so even eight +60 dB bands in series cannot produce `Inf`. If a
//!   non-finite input poisons a band state, that state is cleared at the end of
//!   the `process` call.

use crate::util::{flush_denormal, one_pole_coef};
use crate::Processor;

/// Maximum number of bands.
pub const MAX_BANDS: usize = 8;

/// Lowest centre/corner frequency (Hz).
pub const MIN_FREQUENCY_HZ: f32 = 1.0;
/// Highest centre/corner frequency (Hz) before the Nyquist clamp.
pub const MAX_FREQUENCY_HZ: f32 = 24_000.0;
/// The frequency is additionally clamped to this fraction of the sample rate.
pub const NYQUIST_FRACTION: f64 = 0.499;
/// Gain range (dB) for peaking and shelving bands.
pub const MIN_GAIN_DB: f32 = -60.0;
/// Gain range (dB) for peaking and shelving bands.
pub const MAX_GAIN_DB: f32 = 60.0;
/// Lowest Q.
pub const MIN_Q: f32 = 0.01;
/// Highest Q.
pub const MAX_Q: f32 = 100.0;
/// Time constant (ms) of the frequency/gain/Q smoother.
pub const SMOOTHING_MS: f64 = 10.0;
/// Length (ms) of the crossfade used for type changes and enabling/disabling.
pub const CROSSFADE_MS: f64 = 10.0;

/// Coefficient update interval (frames) while parameters are being smoothed.
const RAMP_BLOCK: usize = 32;
/// Snap thresholds for the smoother (ln-Hz, dB, ln-Q).
const SNAP_LN_F: f64 = 1e-7;
const SNAP_DB: f64 = 1e-5;
const SNAP_LN_Q: f64 = 1e-7;

/// Filter type of one band. The integer values are part of the plugin interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum BandType {
    /// 0: band bypassed.
    #[default]
    Off = 0,
    /// 1: second-order low-pass.
    LowPass = 1,
    /// 2: second-order high-pass.
    HighPass = 2,
    /// 3: band-pass, constant skirt gain (peak gain = Q).
    BandPassSkirt = 3,
    /// 4: band-pass, constant 0 dB peak gain.
    BandPassPeak = 4,
    /// 5: notch.
    Notch = 5,
    /// 6: all-pass.
    AllPass = 6,
    /// 7: peaking (bell).
    Peaking = 7,
    /// 8: low shelf.
    LowShelf = 8,
    /// 9: high shelf.
    HighShelf = 9,
}

impl BandType {
    /// Map the interface integer to a type; anything outside 0 … 9 maps to [`BandType::Off`].
    pub fn from_i32(v: i32) -> Self {
        match v {
            1 => Self::LowPass,
            2 => Self::HighPass,
            3 => Self::BandPassSkirt,
            4 => Self::BandPassPeak,
            5 => Self::Notch,
            6 => Self::AllPass,
            7 => Self::Peaking,
            8 => Self::LowShelf,
            9 => Self::HighShelf,
            _ => Self::Off,
        }
    }
}

/// Parameters of one band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandParams {
    /// Band active. A disabled band costs nothing and is bit-exact passthrough.
    pub enabled: bool,
    /// Filter type.
    pub band_type: BandType,
    /// Centre / corner frequency (Hz, 1 … 24000; clamped just below Nyquist).
    pub frequency: f32,
    /// Gain (dB, −60 … 60); used by peaking and shelves only.
    pub gain_db: f32,
    /// Cookbook Q (0.01 … 100), also for shelves.
    pub q: f32,
}

impl Default for BandParams {
    fn default() -> Self {
        Self {
            enabled: true,
            band_type: BandType::Off,
            frequency: 1000.0,
            gain_db: 0.0,
            q: 0.707,
        }
    }
}

/// Equalizer parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EqualizerParams {
    /// Number of active bands (1 … 8). Bands at index ≥ `num_bands` are inactive.
    pub num_bands: usize,
    /// All eight band settings, always present.
    pub bands: [BandParams; MAX_BANDS],
}

impl Default for EqualizerParams {
    fn default() -> Self {
        Self {
            num_bands: 4,
            bands: [BandParams::default(); MAX_BANDS],
        }
    }
}

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

impl BiquadCoefficients {
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

    fn lerp_step(&self, to: &Self, inv_n: f64) -> Self {
        Self {
            b0: (to.b0 - self.b0) * inv_n,
            b1: (to.b1 - self.b1) * inv_n,
            b2: (to.b2 - self.b2) * inv_n,
            a1: (to.a1 - self.a1) * inv_n,
            a2: (to.a2 - self.a2) * inv_n,
        }
    }

    #[inline(always)]
    fn add(&mut self, d: &Self) {
        self.b0 += d.b0;
        self.b1 += d.b1;
        self.b2 += d.b2;
        self.a1 += d.a1;
        self.a2 += d.a2;
    }
}

/// Clamp a frequency (Hz) to the range the equalizer realises at `sample_rate`.
pub fn clamp_frequency(freq_hz: f64, sample_rate: f64) -> f64 {
    let f = if freq_hz.is_finite() { freq_hz } else { 1000.0 };
    let hi = (sample_rate * NYQUIST_FRACTION).min(MAX_FREQUENCY_HZ as f64);
    f.min(hi).max(MIN_FREQUENCY_HZ as f64)
}

fn clamp_gain(gain_db: f64) -> f64 {
    if gain_db.is_finite() {
        gain_db.clamp(MIN_GAIN_DB as f64, MAX_GAIN_DB as f64)
    } else {
        0.0
    }
}

fn clamp_q(q: f64) -> f64 {
    if q.is_finite() {
        q.clamp(MIN_Q as f64, MAX_Q as f64)
    } else {
        0.707
    }
}

/// Cookbook coefficients for an already-clamped frequency/gain/Q.
/// Returns `None` for [`BandType::Off`].
fn cookbook(ty: BandType, f0: f64, gain_db: f64, q: f64, fs: f64) -> Option<BiquadCoefficients> {
    let w0 = 2.0 * std::f64::consts::PI * f0 / fs;
    let (sin_w, cos_w) = w0.sin_cos();
    let half_sin = (0.5 * w0).sin();
    let half_cos = (0.5 * w0).cos();
    // 1 − cos w0 and 1 + cos w0 without cancellation.
    let one_minus_cos = 2.0 * half_sin * half_sin;
    let one_plus_cos = 2.0 * half_cos * half_cos;
    let alpha = sin_w / (2.0 * q);

    // Common denominator of the non-gain types: 1 + α, −2 cos w0, 1 − α.
    let plain_den = (1.0 + alpha, -2.0 * cos_w, 1.0 - alpha);

    let (b0, b1, b2, (a0, a1, a2)) = match ty {
        BandType::Off => return None,
        BandType::LowPass => (
            0.5 * one_minus_cos,
            one_minus_cos,
            0.5 * one_minus_cos,
            plain_den,
        ),
        BandType::HighPass => (
            0.5 * one_plus_cos,
            -one_plus_cos,
            0.5 * one_plus_cos,
            plain_den,
        ),
        BandType::BandPassSkirt => (0.5 * sin_w, 0.0, -0.5 * sin_w, plain_den),
        BandType::BandPassPeak => (alpha, 0.0, -alpha, plain_den),
        BandType::Notch => (1.0, -2.0 * cos_w, 1.0, plain_den),
        BandType::AllPass => (1.0 - alpha, -2.0 * cos_w, 1.0 + alpha, plain_den),
        BandType::Peaking => {
            let a = 10f64.powf(gain_db / 40.0);
            (
                1.0 + alpha * a,
                -2.0 * cos_w,
                1.0 - alpha * a,
                (1.0 + alpha / a, -2.0 * cos_w, 1.0 - alpha / a),
            )
        }
        BandType::LowShelf | BandType::HighShelf => {
            let a = 10f64.powf(gain_db / 40.0);
            let sa = 2.0 * a.sqrt() * alpha;
            // (A+1) − (A−1)cos w0, (A+1) + (A−1)cos w0, and the two b1/a1 terms,
            // rewritten in terms of 1 − cos w0 for low-frequency precision.
            let p = 2.0 + (a - 1.0) * one_minus_cos;
            let m = 2.0 * a - (a - 1.0) * one_minus_cos;
            let am1_minus = -2.0 + (a + 1.0) * one_minus_cos; // (A−1) − (A+1)cos w0
            let am1_plus = 2.0 * a - (a + 1.0) * one_minus_cos; // (A−1) + (A+1)cos w0
            if ty == BandType::LowShelf {
                (
                    a * (p + sa),
                    2.0 * a * am1_minus,
                    a * (p - sa),
                    (m + sa, -2.0 * am1_plus, m - sa),
                )
            } else {
                (
                    a * (m + sa),
                    -2.0 * a * am1_plus,
                    a * (m - sa),
                    (p + sa, 2.0 * am1_minus, p - sa),
                )
            }
        }
    };
    let inv = 1.0 / a0;
    Some(BiquadCoefficients {
        b0: b0 * inv,
        b1: b1 * inv,
        b2: b2 * inv,
        a1: a1 * inv,
        a2: a2 * inv,
    })
}

/// Normalised cookbook coefficients of one band as the equalizer realises it at
/// `sample_rate` (Hz), after clamping frequency, gain and Q. `None` if the band
/// is disabled or of type Off.
pub fn band_coefficients(band: &BandParams, sample_rate: f32) -> Option<BiquadCoefficients> {
    if !band.enabled {
        return None;
    }
    let fs = sample_rate as f64;
    cookbook(
        band.band_type,
        clamp_frequency(band.frequency as f64, fs),
        clamp_gain(band.gain_db as f64),
        clamp_q(band.q as f64),
        fs,
    )
}

/// Analytic linear magnitude `|H(e^{jω})|` of one band (in steady state) at
/// `freq_hz`, for a sample rate `sample_rate` (Hz). 1 for a disabled/Off band.
pub fn band_magnitude(band: &BandParams, sample_rate: f32, freq_hz: f64) -> f64 {
    match band_coefficients(band, sample_rate) {
        Some(c) => c.magnitude(2.0 * std::f64::consts::PI * freq_hz / sample_rate as f64),
        None => 1.0,
    }
}

/// Clamped target of one band, in the smoothing domains.
#[derive(Debug, Clone, Copy, Default)]
struct Target {
    /// `Off` when the band is inactive for any reason.
    kind: BandType,
    ln_f: f64,
    gain_db: f64,
    ln_q: f64,
}

/// Runtime state of one band slot (everything except the per-channel filter state).
#[derive(Debug, Clone, Copy, Default)]
struct BandSlot {
    /// Realisation currently being faded in (or steady).
    kind: BandType,
    ln_f: f64,
    gain_db: f64,
    ln_q: f64,
    coefs: BiquadCoefficients,
    /// Smoothed parameters equal the target and `coefs` are exact.
    settled: bool,
    /// Which of the two state banks belongs to `kind`.
    bank: usize,
    /// Realisation being faded out.
    prev_kind: BandType,
    prev_coefs: BiquadCoefficients,
    /// Remaining crossfade samples (0 = no crossfade running).
    fade_left: usize,
}

/// Multi-band parametric equalizer. See [`EqualizerParams`] and the module
/// documentation for structure, smoothing and clamping.
#[derive(Debug, Default)]
pub struct Equalizer {
    params: EqualizerParams,
    sample_rate: f64,
    channels: usize,
    targets: [Target; MAX_BANDS],
    slots: [BandSlot; MAX_BANDS],
    /// DF-I state `[x1, x2, y1, y2]`, layout `[band][bank][channel]`.
    states: Vec<[f64; 4]>,
    /// Smoother pole for one full ramp block.
    smooth_block_coef: f64,
    fade_len: usize,
    /// Apply targets without smoothing/crossfade at the next `process`.
    snap_pending: bool,
}

impl Equalizer {
    /// Create an unconfigured equalizer with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update parameters. Never allocates. Out-of-range values are clamped;
    /// changes are applied smoothly (see the module documentation).
    pub fn set_params(&mut self, params: &EqualizerParams) {
        self.params = *params;
        self.update_targets();
    }

    /// Current parameters (as last passed to [`Equalizer::set_params`]).
    pub fn params(&self) -> &EqualizerParams {
        &self.params
    }

    fn update_targets(&mut self) {
        let fs = if self.sample_rate > 0.0 {
            self.sample_rate
        } else {
            48_000.0
        };
        let n = self.params.num_bands.clamp(1, MAX_BANDS);
        for (i, (t, b)) in self
            .targets
            .iter_mut()
            .zip(self.params.bands.iter())
            .enumerate()
        {
            t.kind = if i < n && b.enabled {
                b.band_type
            } else {
                BandType::Off
            };
            t.ln_f = clamp_frequency(b.frequency as f64, fs).ln();
            t.gain_db = clamp_gain(b.gain_db as f64);
            t.ln_q = clamp_q(b.q as f64).ln();
        }
    }

    fn coefs_for(&self, kind: BandType, ln_f: f64, gain_db: f64, ln_q: f64) -> BiquadCoefficients {
        cookbook(kind, ln_f.exp(), gain_db, ln_q.exp(), self.sample_rate).unwrap_or_default()
    }

    /// Set slot `b` to its target immediately (no smoothing).
    fn snap_slot_params(&mut self, b: usize) {
        let t = self.targets[b];
        let coefs = self.coefs_for(t.kind, t.ln_f, t.gain_db, t.ln_q);
        let s = &mut self.slots[b];
        s.kind = t.kind;
        s.ln_f = t.ln_f;
        s.gain_db = t.gain_db;
        s.ln_q = t.ln_q;
        s.coefs = coefs;
        s.settled = true;
    }

    fn snap_all(&mut self) {
        for b in 0..MAX_BANDS {
            self.snap_slot_params(b);
            let s = &mut self.slots[b];
            s.fade_left = 0;
            s.prev_kind = BandType::Off;
        }
        self.states.iter_mut().for_each(|s| *s = [0.0; 4]);
    }

    /// Start a crossfade of slot `b` from its current realisation to its target type.
    fn begin_fade(&mut self, b: usize) {
        let ch = self.channels;
        let s = &mut self.slots[b];
        s.prev_kind = s.kind;
        s.prev_coefs = s.coefs;
        s.bank ^= 1;
        s.fade_left = self.fade_len;
        let start = (b * 2 + s.bank) * ch;
        self.states[start..start + ch]
            .iter_mut()
            .for_each(|st| *st = [0.0; 4]);
        self.snap_slot_params(b);
    }

    /// Advance the smoother of slot `b` by `n` frames and return the new coefficients.
    fn advance_smoother(&mut self, b: usize, n: usize) -> BiquadCoefficients {
        let t = self.targets[b];
        let k = if n == RAMP_BLOCK {
            self.smooth_block_coef
        } else {
            one_pole_coef(SMOOTHING_MS, self.sample_rate).powi(n as i32)
        };
        let s = &mut self.slots[b];
        s.ln_f = t.ln_f + (s.ln_f - t.ln_f) * k;
        s.gain_db = t.gain_db + (s.gain_db - t.gain_db) * k;
        s.ln_q = t.ln_q + (s.ln_q - t.ln_q) * k;
        if (s.ln_f - t.ln_f).abs() < SNAP_LN_F
            && (s.gain_db - t.gain_db).abs() < SNAP_DB
            && (s.ln_q - t.ln_q).abs() < SNAP_LN_Q
        {
            s.ln_f = t.ln_f;
            s.gain_db = t.gain_db;
            s.ln_q = t.ln_q;
            s.settled = true;
        }
        let (kind, f, g, q) = (s.kind, s.ln_f, s.gain_db, s.ln_q);
        self.coefs_for(kind, f, g, q)
    }

    fn process_band(&mut self, b: usize, samples: &mut [f32], frames: usize) {
        let ch = self.channels;
        let mut pos = 0;
        while pos < frames {
            let t = self.targets[b];
            let s = &mut self.slots[b];
            if s.fade_left == 0 && t.kind != s.kind {
                self.begin_fade(b);
            } else if s.kind == t.kind
                && s.kind != BandType::Off
                && (s.ln_f != t.ln_f || s.gain_db != t.gain_db || s.ln_q != t.ln_q)
            {
                // Same type, new frequency/gain/Q: (re)start smoothing.
                s.settled = false;
            }

            let slot = self.slots[b];
            if slot.kind == BandType::Off && slot.fade_left == 0 {
                // Idle band: nothing changes for the rest of this buffer.
                return;
            }

            let remaining = frames - pos;
            let ramping = slot.kind != BandType::Off && !slot.settled;
            let mut n = remaining;
            if ramping {
                n = n.min(RAMP_BLOCK);
            }
            if slot.fade_left > 0 {
                n = n.min(slot.fade_left);
            }

            let delta = if ramping {
                let next = self.advance_smoother(b, n);
                let d = slot.coefs.lerp_step(&next, 1.0 / n as f64);
                // Store the exact end point to avoid accumulated drift.
                self.slots[b].coefs = next;
                Some(d)
            } else {
                None
            };

            let chunk = &mut samples[pos * ch..(pos + n) * ch];
            let band_states = &mut self.states[b * 2 * ch..(b + 1) * 2 * ch];
            let (bank0, bank1) = band_states.split_at_mut(ch);
            let (cur_st, prev_st) = if slot.bank == 0 {
                (bank0, bank1)
            } else {
                (bank1, bank0)
            };

            let mut coefs = slot.coefs;
            let mut prev_coefs = slot.prev_coefs;
            if slot.fade_left == 0 {
                run_steady(chunk, ch, &mut coefs, delta.as_ref(), cur_st);
            } else {
                let fade_len = self.fade_len;
                let done = fade_len - slot.fade_left;
                run_fade(
                    chunk,
                    ch,
                    FadeSide {
                        active: slot.kind != BandType::Off,
                        coefs: &mut coefs,
                        delta: delta.as_ref(),
                        states: cur_st,
                    },
                    FadeSide {
                        active: slot.prev_kind != BandType::Off,
                        coefs: &mut prev_coefs,
                        delta: None,
                        states: prev_st,
                    },
                    done,
                    1.0 / fade_len as f64,
                );
                self.slots[b].fade_left -= n;
                if self.slots[b].fade_left == 0 {
                    self.slots[b].prev_kind = BandType::Off;
                }
            }
            pos += n;
        }
    }

    fn sanitize_states(&mut self) {
        let ch = self.channels;
        for b in 0..MAX_BANDS {
            let region = &mut self.states[b * 2 * ch..(b + 1) * 2 * ch];
            if region.iter().any(|s| s.iter().any(|v| !v.is_finite())) {
                region.iter_mut().for_each(|s| *s = [0.0; 4]);
            }
        }
    }
}

/// One DF-I step: `y = b0·x + b1·x1 + b2·x2 − a1·y1 − a2·y2`.
#[inline(always)]
fn df1_tick(c: &BiquadCoefficients, st: &mut [f64; 4], x: f64) -> f64 {
    let y = c.b0 * x + c.b1 * st[0] + c.b2 * st[1] - c.a1 * st[2] - c.a2 * st[3];
    let y = flush_denormal(y);
    st[1] = st[0];
    st[0] = x;
    st[3] = st[2];
    st[2] = y;
    y
}

#[inline(always)]
fn to_sample(y: f64) -> f32 {
    y.clamp(-(f32::MAX as f64), f32::MAX as f64) as f32
}

fn run_steady(
    chunk: &mut [f32],
    ch: usize,
    coefs: &mut BiquadCoefficients,
    delta: Option<&BiquadCoefficients>,
    states: &mut [[f64; 4]],
) {
    for frame in chunk.chunks_exact_mut(ch) {
        if let Some(d) = delta {
            coefs.add(d);
        }
        for (s, st) in frame.iter_mut().zip(states.iter_mut()) {
            *s = to_sample(df1_tick(coefs, st, *s as f64));
        }
    }
}

struct FadeSide<'a> {
    active: bool,
    coefs: &'a mut BiquadCoefficients,
    delta: Option<&'a BiquadCoefficients>,
    states: &'a mut [[f64; 4]],
}

fn run_fade(
    chunk: &mut [f32],
    ch: usize,
    new: FadeSide<'_>,
    old: FadeSide<'_>,
    done: usize,
    inv_len: f64,
) {
    for (i, frame) in chunk.chunks_exact_mut(ch).enumerate() {
        if let Some(d) = new.delta {
            new.coefs.add(d);
        }
        let g = ((done + i + 1) as f64 * inv_len).min(1.0);
        for (c, s) in frame.iter_mut().enumerate() {
            let x = *s as f64;
            let y_new = if new.active {
                df1_tick(new.coefs, &mut new.states[c], x)
            } else {
                x
            };
            let y_old = if old.active {
                df1_tick(old.coefs, &mut old.states[c], x)
            } else {
                x
            };
            *s = to_sample(y_old + (y_new - y_old) * g);
        }
    }
}

impl Processor for Equalizer {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        let fs = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate as f64
        } else {
            48_000.0
        };
        self.sample_rate = fs;
        self.channels = channels;
        self.states = vec![[0.0; 4]; MAX_BANDS * 2 * channels];
        self.smooth_block_coef = one_pole_coef(SMOOTHING_MS, fs).powi(RAMP_BLOCK as i32);
        self.fade_len = ((CROSSFADE_MS * 0.001 * fs).round() as usize).max(1);
        self.slots = [BandSlot::default(); MAX_BANDS];
        self.update_targets();
        self.snap_pending = true;
    }

    fn process(&mut self, samples: &mut [f32]) {
        if self.channels == 0 || self.sample_rate <= 0.0 {
            return;
        }
        if self.snap_pending {
            self.snap_all();
            self.snap_pending = false;
        }
        let frames = samples.len() / self.channels;
        if frames == 0 {
            return;
        }
        let mut touched = false;
        for b in 0..MAX_BANDS {
            let s = &self.slots[b];
            if s.kind == BandType::Off && s.fade_left == 0 && self.targets[b].kind == BandType::Off
            {
                continue;
            }
            touched = true;
            self.process_band(b, samples, frames);
        }
        if touched {
            self.sanitize_states();
        }
    }

    fn reset(&mut self) {
        self.states.iter_mut().for_each(|s| *s = [0.0; 4]);
        self.snap_pending = true;
    }
}
