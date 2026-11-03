//! Input cleanup: DC blocker, Butterworth high-pass and a mains-hum notch comb
//! (fundamental plus harmonics), in that order.
//!
//! # DC blocker
//!
//! A first-order high-pass with its −3.01 dB corner at [`DC_BLOCK_HZ`] (5 Hz)
//! at every sample rate: the bilinear transform (pre-warped) of `s/(s + 1)`,
//! i.e. the order-1 [`Butterworth`] design (reused, see [`crate::filter`]).
//! With `K = tan(π·5/fs)` it is `H(z) = g·(1 − z⁻¹)/(1 − R·z⁻¹)` with
//! `R = (1 − K)/(1 + K)` and `g = 1/(1 + K) = (1 + R)/2`: the classic
//! one-zero/one-pole DC blocker (J. O. Smith III, *Introduction to Digital
//! Filters*, "DC blocker"), scaled for exactly unity gain at Nyquist. The zero
//! sits exactly at `z = 1`, so DC is removed completely; at 20 Hz the loss is
//! 0.26 dB, and the time constant is `1/(2π·5 Hz)` ≈ 32 ms.
//!
//! # High-pass
//!
//! [`Butterworth`] high-pass of order 1 … 8 (6 dB/octave per order) at
//! `highpass_hz`; 0 = off. Cutoff changes glide and order changes crossfade
//! inside the filter (see [`crate::filter`]).
//!
//! # Hum notches
//!
//! For `f0` = 50 or 60 Hz and `k = 1 … harmonics`, one RBJ cookbook notch
//! (R. Bristow-Johnson, "Audio EQ Cookbook") at `k·f0`; harmonics at or above
//! `0.49·fs` are skipped. The notches run in series.
//!
//! **Constant bandwidth (own design choice).** Every notch has the same −3 dB
//! bandwidth `B = f0 / hum_q` Hz (so `hum_q` is the Q of the fundamental and
//! harmonic `k` has `Q_k ≈ k·hum_q`). With a constant *Q* instead, the
//! higher harmonics would be `k` times wider and would eat into the passband
//! between them; with constant bandwidth the passband between neighbouring
//! notches (spacing `f0`) looks the same everywhere.
//!
//! The bandwidth is set exactly in the digital domain: the cookbook notch
//! `(1 − 2cos w·z⁻¹ + z⁻²)/((1 + α) − 2cos w·z⁻¹ + (1 − α)·z⁻²)` has its
//! −3 dB points `Δω` apart with `tan(Δω/2) = α` (the standard second-order
//! notch, e.g. Regalia & Mitra's all-pass-based form, `(1 + A(z))/2` with
//! `a2 = (1 − α)/(1 + α)`). So `α = tan(π·B/fs)`, i.e. the cookbook
//! `Q_k = sin(w_k)/(2·tan(π·B/fs))`.
//!
//! **Depth and flatness.** Each notch has a zero exactly on the unit circle at
//! `k·f0`, so its depth is unlimited in theory; measured with steady sines it is
//! far below −40 dB. Between two notches a single notch's loss is
//! `10·log10(1 + (B/2δ)²)` dB at a distance `δ`; at the midpoint (`δ = f0/2`)
//! with `hum_q = 10` that is 0.04 dB per neighbour. Summed over all ten
//! notches (analytic, 50/60 Hz, 8–96 kHz) the worst loss is 0.11 dB at the
//! midpoints and 0.15 dB at the third-points (`(k + ⅓)·f0`, `(k + ⅔)·f0`) for
//! `hum_q = 10`, and 0.40 dB at the third-points for `hum_q = 6`. At
//! `hum_q = 5` it would be 0.57 dB, so [`HUM_Q_RANGE`] starts at 6: over the
//! whole range the passband between the notches stays within 0.5 dB.
//!
//! # Switching and parameter changes
//!
//! Each of the three stages is skipped entirely while it is off, so the
//! default (everything off) is a bit-exact passthrough. Switching a stage on
//! or off crossfades linearly between the dry and the filtered signal over
//! [`FADE_MS`]; a stage that has faded out is reset, so it starts from clean
//! state when it is switched on again. A change of the hum frequency, the
//! number of harmonics or `hum_q` crossfades over [`FADE_MS`] from the old
//! notch comb (with its state) to the new one (from zero state). A notch
//! started from zero state passes the signal and only then builds up its
//! rejection, so the new comb has no start-up transient. A further hum change
//! during a crossfade starts when that crossfade ends. Between
//! `configure`/`reset` and the first `process` call, parameters apply
//! immediately (no fades), so the call order does not change the output.
//!
//! Filters run in `f64` (DF-I, phase 1 SIMD kernels) and flush their state
//! below `1e-15`; no latency.

use fearless_simd::Simd;

use crate::biquad::{
    run_chain, sanitize_bank, BiquadCoefficients, SteadyStage, MAX_STAGES, STATE_PER_CHANNEL,
};
use crate::filter::{Butterworth, ButterworthParams, FilterKind, MAX_ORDER};
use crate::simd::Level;
use crate::Processor;

/// Corner frequency (Hz, −3.01 dB) of the DC blocker.
pub const DC_BLOCK_HZ: f32 = 5.0;
/// Range of the high-pass cutoff (Hz); 0 means off.
pub const HIGHPASS_RANGE_HZ: (f32, f32) = (0.0, 1_000.0);
/// Most hum harmonics (the fundamental counts as harmonic 1).
pub const MAX_HUM_HARMONICS: usize = 10;
/// Range of `hum_q` (Q of the fundamental notch; every notch has bandwidth
/// `f0/hum_q`). The lower end keeps the passband within 0.5 dB.
pub const HUM_Q_RANGE: (f32, f32) = (6.0, 100.0);
/// Length (ms) of the on/off and hum-change crossfades.
pub const FADE_MS: f64 = 10.0;

/// Notches above this fraction of the sample rate are skipped.
const MAX_NOTCH_FRACTION: f64 = 0.49;
/// Frames per step while a crossfade runs (scratch size).
const FADE_CHUNK: usize = 64;

/// Mains-hum fundamental of the notch comb.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HumFrequency {
    /// No hum notches.
    #[default]
    Off,
    /// 50 Hz mains (Europe, most of Asia and Africa).
    Hz50,
    /// 60 Hz mains (Americas, parts of Asia).
    Hz60,
}

impl HumFrequency {
    /// Fundamental in Hz (`None` for [`HumFrequency::Off`]).
    pub fn hz(self) -> Option<f64> {
        match self {
            Self::Off => None,
            Self::Hz50 => Some(50.0),
            Self::Hz60 => Some(60.0),
        }
    }
}

/// Cleanup parameters. Out-of-range values are clamped; NaN falls back to the
/// default. The default turns everything off (bit-exact passthrough).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CleanupParams {
    /// DC blocker on/off (first-order high-pass at [`DC_BLOCK_HZ`]).
    pub dc_block: bool,
    /// High-pass cutoff (Hz, −3.01 dB); 0 = off. See [`HIGHPASS_RANGE_HZ`].
    pub highpass_hz: f32,
    /// High-pass Butterworth order, 1 … 8.
    pub highpass_order: usize,
    /// Hum fundamental, or off.
    pub hum: HumFrequency,
    /// Number of notches: the fundamental and harmonics 2 … n (1 … 10).
    pub hum_harmonics: usize,
    /// Q of the fundamental notch; every notch has the bandwidth `f0/hum_q`.
    pub hum_q: f32,
}

impl Default for CleanupParams {
    fn default() -> Self {
        Self {
            dc_block: false,
            highpass_hz: 0.0,
            highpass_order: 2,
            hum: HumFrequency::Off,
            hum_harmonics: 5,
            hum_q: 10.0,
        }
    }
}

impl CleanupParams {
    /// Copy with every field clamped to its range (NaN becomes the default).
    pub fn clamped(&self) -> Self {
        let d = Self::default();
        let clamp =
            |v: f32, r: (f32, f32), fb: f32| if v.is_nan() { fb } else { v.clamp(r.0, r.1) };
        Self {
            dc_block: self.dc_block,
            highpass_hz: clamp(self.highpass_hz, HIGHPASS_RANGE_HZ, d.highpass_hz),
            highpass_order: self.highpass_order.clamp(1, MAX_ORDER),
            hum: self.hum,
            hum_harmonics: self.hum_harmonics.clamp(1, MAX_HUM_HARMONICS),
            hum_q: clamp(self.hum_q, HUM_Q_RANGE, d.hum_q),
        }
    }
}

/// The notch comb of one hum setting: at most [`MAX_HUM_HARMONICS`] biquads,
/// stored inline (never allocates).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HumNotches {
    coefs: [BiquadCoefficients; MAX_HUM_HARMONICS],
    len: usize,
}

impl HumNotches {
    /// Design the comb for `hum`, `harmonics` and `hum_q` at `sample_rate`
    /// (Hz), after clamping (see the module documentation).
    pub fn design(hum: HumFrequency, harmonics: usize, hum_q: f32, sample_rate: f64) -> Self {
        let mut out = Self::default();
        let Some(f0) = hum.hz() else {
            return out;
        };
        let q = if hum_q.is_nan() { 10.0 } else { hum_q }.clamp(HUM_Q_RANGE.0, HUM_Q_RANGE.1);
        let alpha = (core::f64::consts::PI * f0 / q as f64 / sample_rate).tan();
        for k in 1..=harmonics.clamp(1, MAX_HUM_HARMONICS) {
            let f = k as f64 * f0;
            if f >= MAX_NOTCH_FRACTION * sample_rate {
                break;
            }
            let w = 2.0 * core::f64::consts::PI * f / sample_rate;
            let qk = w.sin() / (2.0 * alpha);
            out.coefs[out.len] = BiquadCoefficients::notch(f, qk, sample_rate);
            out.len += 1;
        }
        out
    }

    /// The notches, lowest first.
    pub fn as_slice(&self) -> &[BiquadCoefficients] {
        &self.coefs[..self.len]
    }

    /// Analytic magnitude of the comb at `freq_hz` (Hz) for `sample_rate`.
    pub fn magnitude_at(&self, freq_hz: f64, sample_rate: f64) -> f64 {
        self.as_slice()
            .iter()
            .map(|c| c.magnitude_at(freq_hz, sample_rate))
            .product()
    }
}

/// The hum notch comb as a processor, with a crossfade between combs.
#[derive(Debug)]
struct NotchBank {
    sample_rate: f64,
    channels: usize,
    target: HumNotches,
    current: HumNotches,
    prev: HumNotches,
    /// State bank of `current` (0 or 1); the other belongs to `prev`.
    bank: usize,
    states: Vec<f64>,
    scratch: Vec<f32>,
    fade_left: usize,
    fade_len: usize,
    snap_pending: bool,
    level: Level,
}

impl NotchBank {
    fn new() -> Self {
        Self {
            sample_rate: 48_000.0,
            channels: 0,
            target: HumNotches::default(),
            current: HumNotches::default(),
            prev: HumNotches::default(),
            bank: 0,
            states: Vec::new(),
            scratch: Vec::new(),
            fade_left: 0,
            fade_len: 1,
            snap_pending: true,
            level: Level::baseline(),
        }
    }

    fn bank_len(&self) -> usize {
        MAX_HUM_HARMONICS * STATE_PER_CHANNEL * self.channels
    }

    fn clear_bank(&mut self, bank: usize) {
        let len = self.bank_len();
        self.states[bank * len..(bank + 1) * len].fill(0.0);
    }

    /// Run `notches` over `data` with state bank `bank`, at most
    /// `MAX_STAGES` biquads per kernel call.
    fn run<S: Simd>(
        simd: S,
        notches: &HumNotches,
        bank: usize,
        bank_len: usize,
        data: &mut [f32],
        ch: usize,
        states: &mut [f64],
    ) {
        let step = STATE_PER_CHANNEL * ch;
        let base = bank * bank_len;
        for (g, group) in notches.as_slice().chunks(MAX_STAGES).enumerate() {
            let mut stages = [SteadyStage {
                coefs: BiquadCoefficients::IDENTITY,
                state: 0,
            }; MAX_STAGES];
            for (i, c) in group.iter().enumerate() {
                stages[i] = SteadyStage {
                    coefs: *c,
                    state: base + (g * MAX_STAGES + i) * step,
                };
            }
            run_chain(simd, data, ch, &stages[..group.len()], &[], states);
        }
    }

    fn process_with<S: Simd>(&mut self, simd: S, samples: &mut [f32]) {
        let ch = self.channels;
        if self.snap_pending {
            self.current = self.target;
            self.fade_left = 0;
            self.states.fill(0.0);
            self.snap_pending = false;
        }
        let bank_len = self.bank_len();
        let frames = samples.len() / ch;
        let mut pos = 0;
        while pos < frames {
            if self.fade_left == 0 && self.current != self.target {
                self.prev = self.current;
                self.bank ^= 1;
                self.clear_bank(self.bank);
                self.current = self.target;
                self.fade_left = self.fade_len;
            }
            let mut n = frames - pos;
            if self.fade_left > 0 {
                n = n.min(self.fade_left).min(FADE_CHUNK);
            }
            let data = &mut samples[pos * ch..(pos + n) * ch];
            if self.fade_left > 0 {
                let old = &mut self.scratch[..n * ch];
                old.copy_from_slice(data);
                Self::run(
                    simd,
                    &self.prev,
                    self.bank ^ 1,
                    bank_len,
                    old,
                    ch,
                    &mut self.states,
                );
            }
            Self::run(
                simd,
                &self.current,
                self.bank,
                bank_len,
                data,
                ch,
                &mut self.states,
            );
            if self.fade_left > 0 {
                let done = self.fade_len - self.fade_left;
                crossfade(data, &self.scratch[..n * ch], ch, done, self.fade_len, 1.0);
                self.fade_left -= n;
            }
            pos += n;
        }
        self.states
            .chunks_exact_mut(bank_len)
            .for_each(sanitize_bank);
    }
}

impl Processor for NotchBank {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        self.sample_rate = sample_rate as f64;
        self.channels = channels.max(1);
        self.states = vec![0.0; 2 * self.bank_len()];
        self.scratch = vec![0.0; FADE_CHUNK * self.channels];
        self.fade_len = ((FADE_MS * 0.001 * self.sample_rate).round() as usize).max(1);
        self.bank = 0;
        self.snap_pending = true;
    }

    fn process(&mut self, samples: &mut [f32]) {
        if self.channels == 0 {
            return;
        }
        let level = self.level;
        fearless_simd::dispatch!(level, simd => self.process_with(simd, samples));
    }

    fn reset(&mut self) {
        self.states.fill(0.0);
        self.snap_pending = true;
    }
}

/// Replace `data` (the new signal) by `old + (new − old)·w` frame by frame,
/// with `w` going linearly from `(done + 1)/len` (capped at 1) upward if
/// `dir > 0`, or `1 − (done + 1)/len` downward if `dir < 0`.
fn crossfade(data: &mut [f32], old: &[f32], ch: usize, done: usize, len: usize, dir: f64) {
    let inv = 1.0 / len as f64;
    for (i, (d, o)) in data
        .chunks_exact_mut(ch)
        .zip(old.chunks_exact(ch))
        .enumerate()
    {
        let up = ((done + i + 1) as f64 * inv).min(1.0);
        let w = if dir > 0.0 { up } else { 1.0 - up };
        for (d, &o) in d.iter_mut().zip(o) {
            let o = o as f64;
            *d = (o + (*d as f64 - o) * w) as f32;
        }
    }
}

/// Dry/wet switch of one stage: skipped while off, run while on, linear
/// crossfade of `fade_len` frames in between.
#[derive(Debug, Clone, Copy)]
struct Switch {
    on: bool,
    /// Wet weight in frames: 0 (dry) … `fade_len` (wet).
    pos: usize,
    fade_len: usize,
}

impl Switch {
    fn new() -> Self {
        Self {
            on: false,
            pos: 0,
            fade_len: 1,
        }
    }

    fn configure(&mut self, sample_rate: f64) {
        self.fade_len = ((FADE_MS * 0.001 * sample_rate).round() as usize).max(1);
        self.snap();
    }

    /// Jump to the end state (used after configure/reset).
    fn snap(&mut self) {
        self.pos = if self.on { self.fade_len } else { 0 };
    }

    /// Run `stage` over `samples` according to the switch state; `dry` is
    /// scratch of at least `FADE_CHUNK · ch` samples.
    fn run<P: Processor>(
        &mut self,
        stage: &mut P,
        samples: &mut [f32],
        ch: usize,
        dry: &mut [f32],
    ) {
        let frames = samples.len() / ch;
        let mut f = 0;
        while f < frames {
            if !self.on && self.pos == 0 {
                return;
            }
            if self.on && self.pos == self.fade_len {
                stage.process(&mut samples[f * ch..frames * ch]);
                return;
            }
            let n = (frames - f).min(FADE_CHUNK).min(if self.on {
                self.fade_len - self.pos
            } else {
                self.pos
            });
            let data = &mut samples[f * ch..(f + n) * ch];
            let d = &mut dry[..n * ch];
            d.copy_from_slice(data);
            stage.process(data);
            if self.on {
                crossfade(data, d, ch, self.pos, self.fade_len, 1.0);
                self.pos += n;
            } else {
                crossfade(data, d, ch, self.fade_len - self.pos, self.fade_len, -1.0);
                self.pos -= n;
                if self.pos == 0 {
                    // Faded out: start from clean state when switched on again.
                    stage.reset();
                }
            }
            f += n;
        }
    }
}

/// Input cleanup processor: DC blocker → Butterworth high-pass → hum notch
/// comb, each optional, for interleaved `f32` audio. See the module
/// documentation.
///
/// Implements [`Processor`]: `configure` allocates, `set_params`, `process`
/// and `reset` never do. No latency. With every stage off (the default)
/// `process` leaves the samples untouched.
#[derive(Debug)]
pub struct Cleanup {
    params: CleanupParams,
    sample_rate: f64,
    channels: usize,
    dc: Butterworth,
    highpass: Butterworth,
    hum: NotchBank,
    dc_switch: Switch,
    hp_switch: Switch,
    hum_switch: Switch,
    /// Dry copy of one chunk during a switch crossfade.
    dry: Vec<f32>,
    /// `set_simd_level` was called: `configure` keeps the hum level.
    hum_forced: bool,
    /// No frame processed since configure/reset: switches snap (no fade).
    fresh: bool,
}

impl Default for Cleanup {
    fn default() -> Self {
        Self::new()
    }
}

impl Cleanup {
    /// Create an unconfigured processor with default parameters (all off).
    pub fn new() -> Self {
        let mut dc = Butterworth::new();
        dc.set_params(&ButterworthParams {
            kind: FilterKind::HighPass,
            order: 1,
            cutoff_hz: DC_BLOCK_HZ,
        });
        let mut c = Self {
            params: CleanupParams::default(),
            sample_rate: 48_000.0,
            channels: 0,
            dc,
            highpass: Butterworth::new(),
            hum: NotchBank::new(),
            dc_switch: Switch::new(),
            hp_switch: Switch::new(),
            hum_switch: Switch::new(),
            dry: Vec::new(),
            hum_forced: false,
            fresh: true,
        };
        c.apply_params();
        c
    }

    /// Update parameters (clamped). Never allocates.
    pub fn set_params(&mut self, params: &CleanupParams) {
        self.params = params.clamped();
        self.apply_params();
    }

    /// Current (clamped) parameters.
    pub fn params(&self) -> &CleanupParams {
        &self.params
    }

    /// The notch comb for `params` at `sample_rate` (Hz), as realised.
    pub fn hum_design(params: &CleanupParams, sample_rate: f32) -> HumNotches {
        let p = params.clamped();
        HumNotches::design(p.hum, p.hum_harmonics, p.hum_q, sample_rate as f64)
    }

    /// Force the SIMD level of every filter (kept across `configure`).
    /// Levels with the same [`uses_fma`](crate::biquad::uses_fma) give
    /// bit-identical output.
    pub fn set_simd_level(&mut self, level: Level) {
        self.dc.set_simd_level(level);
        self.highpass.set_simd_level(level);
        self.hum.level = level;
        self.hum_forced = true;
    }

    fn apply_params(&mut self) {
        let p = self.params;
        self.dc_switch.on = p.dc_block;
        self.hp_switch.on = p.highpass_hz > 0.0;
        if self.hp_switch.on {
            self.highpass.set_params(&ButterworthParams {
                kind: FilterKind::HighPass,
                order: p.highpass_order,
                cutoff_hz: p.highpass_hz,
            });
        }
        self.hum_switch.on = p.hum != HumFrequency::Off;
        if self.hum_switch.on {
            self.hum.target = HumNotches::design(p.hum, p.hum_harmonics, p.hum_q, self.sample_rate);
        }
    }
}

impl Processor for Cleanup {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        self.sample_rate = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate as f64
        } else {
            48_000.0
        };
        self.channels = channels.max(1);
        let fs = self.sample_rate as f32;
        self.dc.configure(fs, self.channels);
        self.highpass.configure(fs, self.channels);
        self.hum.configure(fs, self.channels);
        if !self.hum_forced {
            self.hum.level = crate::simd::detect();
        }
        self.dry = vec![0.0; FADE_CHUNK * self.channels];
        self.apply_params();
        self.reset();
    }

    fn process(&mut self, samples: &mut [f32]) {
        let ch = self.channels;
        if ch == 0 {
            return;
        }
        if self.fresh {
            // First call after configure/reset: take the switch states as
            // they are now (no fade), whatever the call order was.
            for s in [
                &mut self.dc_switch,
                &mut self.hp_switch,
                &mut self.hum_switch,
            ] {
                s.snap();
            }
            self.fresh = false;
        }
        let usable = samples.len() / ch * ch;
        let samples = &mut samples[..usable];
        self.dc_switch.run(&mut self.dc, samples, ch, &mut self.dry);
        self.hp_switch
            .run(&mut self.highpass, samples, ch, &mut self.dry);
        self.hum_switch
            .run(&mut self.hum, samples, ch, &mut self.dry);
    }

    fn reset(&mut self) {
        self.dc.reset();
        self.highpass.reset();
        self.hum.reset();
        for s in [
            &mut self.dc_switch,
            &mut self.hp_switch,
            &mut self.hum_switch,
        ] {
            s.configure(self.sample_rate);
        }
        self.fresh = true;
    }
}
