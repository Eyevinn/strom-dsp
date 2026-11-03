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
//! - **Processing order**: built on the [`crate::biquad`] kernels. All active
//!   bands run in series *sample by sample* (not band by band over the whole
//!   buffer), with the channels of a frame as SIMD lanes, and the signal stays
//!   in `f64` between bands. The steady state (no band changing) is a plain
//!   loop with fixed coefficients; per-sample coefficient interpolation and
//!   crossfading only run in segments where a band is actually changing.
//! - **Denormals**: filter states are flushed to zero below 1e-15 (every 32
//!   frames in the steady state, every sample while changing; see
//!   [`crate::biquad`]). Inputs are `f32` samples promoted to `f64`, which can
//!   never be `f64` subnormals.
//! - **Robustness**: samples written back are clamped to the finite `f32`
//!   range, so even eight +60 dB bands in series cannot produce `Inf`. If a
//!   non-finite input poisons a band state, that state is cleared at the end of
//!   the `process` call.

use crate::biquad::{
    run_changing, run_steady, sanitize_bank, ChangingStage, FadeOut, SteadyStage, MAX_STAGES,
    STATE_PER_CHANNEL,
};
use crate::util::one_pole_coef;
use crate::Processor;

/// Maximum number of bands.
pub const MAX_BANDS: usize = 8;
const _: () = assert!(MAX_BANDS <= MAX_STAGES);

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

pub use crate::biquad::BiquadCoefficients;

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
    type C = BiquadCoefficients;
    Some(match ty {
        BandType::Off => return None,
        BandType::LowPass => C::low_pass(f0, q, fs),
        BandType::HighPass => C::high_pass(f0, q, fs),
        BandType::BandPassSkirt => C::band_pass_skirt(f0, q, fs),
        BandType::BandPassPeak => C::band_pass_peak(f0, q, fs),
        BandType::Notch => C::notch(f0, q, fs),
        BandType::AllPass => C::all_pass(f0, q, fs),
        BandType::Peaking => C::peaking(f0, gain_db, q, fs),
        BandType::LowShelf => C::low_shelf(f0, gain_db, q, fs),
        BandType::HighShelf => C::high_shelf(f0, gain_db, q, fs),
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
    /// DF-I state, `[band][bank]` regions of `STATE_PER_CHANNEL · channels`
    /// values each (lane-group layout, see [`crate::biquad`]).
    states: Vec<f64>,
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
        self.states.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Offset of the state region of band `b`, bank `bank`.
    fn bank_offset(&self, b: usize, bank: usize) -> usize {
        (b * 2 + bank) * STATE_PER_CHANNEL * self.channels
    }

    /// Start a crossfade of slot `b` from its current realisation to its target type.
    fn begin_fade(&mut self, b: usize) {
        let len = STATE_PER_CHANNEL * self.channels;
        let s = &mut self.slots[b];
        s.prev_kind = s.kind;
        s.prev_coefs = s.coefs;
        s.bank ^= 1;
        s.fade_left = self.fade_len;
        let start = self.bank_offset(b, self.slots[b].bank);
        self.states[start..start + len]
            .iter_mut()
            .for_each(|v| *v = 0.0);
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

    /// At a segment boundary: start a pending type change of band `b`, or
    /// (re)start smoothing if its frequency/gain/Q target moved.
    fn update_band(&mut self, b: usize) {
        let t = self.targets[b];
        let s = &mut self.slots[b];
        if s.fade_left == 0 && t.kind != s.kind {
            self.begin_fade(b);
        } else if s.kind == t.kind
            && s.kind != BandType::Off
            && (s.ln_f != t.ln_f || s.gain_db != t.gain_db || s.ln_q != t.ln_q)
        {
            s.settled = false;
        }
    }

    /// Process one segment of `n` frames starting at frame `pos`, during which
    /// no band starts or ends a crossfade and ramping bands advance by at most
    /// one ramp block. Returns false if every band was idle.
    fn process_segment(&mut self, data: &mut [f32], n: usize) -> bool {
        let mut steady = [SteadyStage {
            coefs: BiquadCoefficients::IDENTITY,
            state: 0,
        }; MAX_STAGES];
        let mut changing = [ChangingStage {
            new: None,
            delta: None,
            fade: None,
        }; MAX_STAGES];
        let mut count = 0;
        let mut any_changing = false;
        let inv_fade = 1.0 / self.fade_len as f64;
        for b in 0..MAX_BANDS {
            let slot = self.slots[b];
            if slot.kind == BandType::Off && slot.fade_left == 0 {
                continue;
            }
            let delta = if slot.kind != BandType::Off && !slot.settled {
                let next = self.advance_smoother(b, n);
                // Store the exact end point, so the interpolation never drifts.
                self.slots[b].coefs = next;
                Some(slot.coefs.ramp_step(&next, n))
            } else {
                None
            };
            let cur = SteadyStage {
                coefs: slot.coefs,
                state: self.bank_offset(b, slot.bank),
            };
            let fade = (slot.fade_left > 0).then(|| FadeOut {
                old: (slot.prev_kind != BandType::Off).then(|| SteadyStage {
                    coefs: slot.prev_coefs,
                    state: self.bank_offset(b, slot.bank ^ 1),
                }),
                done: self.fade_len - slot.fade_left,
                inv_len: inv_fade,
            });
            any_changing |= delta.is_some() || fade.is_some();
            steady[count] = cur;
            changing[count] = ChangingStage {
                new: (slot.kind != BandType::Off).then_some(cur),
                delta,
                fade,
            };
            count += 1;
        }
        if count == 0 {
            return false;
        }
        if any_changing {
            run_changing(data, self.channels, &changing[..count], &mut self.states);
        } else {
            run_steady(data, self.channels, &steady[..count], &mut self.states);
        }
        true
    }

    fn sanitize_states(&mut self) {
        let len = 2 * STATE_PER_CHANNEL * self.channels;
        self.states.chunks_exact_mut(len).for_each(sanitize_bank);
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
        self.states = vec![0.0; MAX_BANDS * 2 * STATE_PER_CHANNEL * channels];
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
        let ch = self.channels;
        let frames = samples.len() / ch;
        let idle = (0..MAX_BANDS).all(|b| {
            let s = &self.slots[b];
            s.kind == BandType::Off && s.fade_left == 0 && self.targets[b].kind == BandType::Off
        });
        if frames == 0 || idle {
            return;
        }
        let mut touched = false;
        let mut pos = 0;
        while pos < frames {
            for b in 0..MAX_BANDS {
                self.update_band(b);
            }
            // Segment: up to the next ramp-block or crossfade boundary of any
            // band; the whole rest of the buffer when everything is steady.
            let mut n = frames - pos;
            for s in &self.slots {
                if s.kind != BandType::Off && !s.settled {
                    n = n.min(RAMP_BLOCK);
                }
                if s.fade_left > 0 {
                    n = n.min(s.fade_left);
                }
            }
            touched |= self.process_segment(&mut samples[pos * ch..(pos + n) * ch], n);
            for s in self.slots.iter_mut().filter(|s| s.fade_left > 0) {
                s.fade_left -= n;
                if s.fade_left == 0 {
                    s.prev_kind = BandType::Off;
                }
            }
            pos += n;
        }
        if touched {
            self.sanitize_states();
        }
    }

    fn reset(&mut self) {
        self.states.iter_mut().for_each(|v| *v = 0.0);
        self.snap_pending = true;
    }
}
