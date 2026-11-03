//! Linkwitz-Riley crossover (LR2/LR4/LR8, 1–5 bands) with all-pass
//! compensation. See the [`crate::filter`] module documentation for the
//! topology and the LR2 polarity convention.

use fearless_simd::Simd;

use super::design::{butterworth_all_pass, clamp_cutoff, linkwitz_riley, FilterKind, Sections};
use super::{blend_into, Glide, CROSSFADE_MS, FADE_CHUNK, RAMP_BLOCK};
use crate::biquad::{
    run_chain, sanitize_bank, BiquadCoefficients, SteadyStage, MAX_STAGES, STATE_PER_CHANNEL,
};
use crate::simd::Level;

/// Most bands a [`Crossover`] splits into.
pub const MAX_CROSSOVER_BANDS: usize = 5;
/// Most crossover frequencies (`MAX_CROSSOVER_BANDS − 1`).
pub const MAX_SPLITS: usize = MAX_CROSSOVER_BANDS - 1;

/// Fallback frequencies (Hz) for non-finite inputs, and the defaults.
const DEFAULT_FREQUENCIES_HZ: [f32; MAX_SPLITS] = [200.0, 2000.0, 6000.0, 12_000.0];

/// Linkwitz-Riley order: the square of a Butterworth filter of half the order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LrOrder {
    /// 2nd order, 12 dB/octave (Butterworth 1 squared). The high band of
    /// every split is polarity-inverted (see the module documentation).
    Lr2,
    /// 4th order, 24 dB/octave (Butterworth 2 squared).
    #[default]
    Lr4,
    /// 8th order, 48 dB/octave (Butterworth 4 squared).
    Lr8,
}

impl LrOrder {
    /// Order of the Butterworth filter that is squared (1, 2 or 4).
    pub fn butterworth_order(self) -> usize {
        match self {
            Self::Lr2 => 1,
            Self::Lr4 => 2,
            Self::Lr8 => 4,
        }
    }
}

/// Parameters of a [`Crossover`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossoverParams {
    /// Linkwitz-Riley order of every split.
    pub order: LrOrder,
    /// Number of bands, 1 … 5 (clamped).
    pub num_bands: usize,
    /// Crossover frequencies (Hz), lowest first; only the first
    /// `num_bands − 1` are used. Each is clamped to `[1 Hz, 0.499·fs]` and to
    /// at least the previous one.
    pub frequencies_hz: [f32; MAX_SPLITS],
}

impl Default for CrossoverParams {
    fn default() -> Self {
        Self {
            order: LrOrder::Lr4,
            num_bands: 3,
            frequencies_hz: DEFAULT_FREQUENCIES_HZ,
        }
    }
}

/// Structural part of the parameters (a change is crossfaded).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Shape {
    order: LrOrder,
    bands: usize,
}

/// Coefficients of every split: LR low-pass, LR high-pass and the Butterworth
/// all-pass `LP + HP` used for compensation.
#[derive(Debug, Clone, Copy, Default)]
struct Design {
    lp: [Sections; MAX_SPLITS],
    hp: [Sections; MAX_SPLITS],
    ap: [Sections; MAX_SPLITS],
}

impl Design {
    fn new(order: LrOrder, ln_f: &[f64; MAX_SPLITS], fs: f64) -> Self {
        let n = order.butterworth_order();
        let mut d = Self::default();
        for (s, &lf) in ln_f.iter().enumerate() {
            let f = lf.exp();
            d.lp[s] = linkwitz_riley(FilterKind::LowPass, n, f, fs);
            d.hp[s] = linkwitz_riley(FilterKind::HighPass, n, f, fs);
            d.ap[s] = butterworth_all_pass(n, f, fs);
        }
        d
    }
}

/// Biquad state slots per bank: LP (split × 4), HP (split × 4), all-pass
/// (band × split × 2).
const LP_BASE: usize = 0;
const HP_BASE: usize = MAX_SPLITS * 4;
const AP_BASE: usize = 2 * MAX_SPLITS * 4;
const SLOTS_PER_BANK: usize = AP_BASE + MAX_CROSSOVER_BANDS * MAX_SPLITS * 2;
/// Most stages one band runs: LR8 low-pass (4) + three LR8 all-passes (2 each).
const MAX_BAND_STAGES: usize = 4 + (MAX_SPLITS - 1) * 2;

/// Linkwitz-Riley crossover for interleaved `f32` audio: splits the input into
/// 1–5 bands whose plain sum is an all-pass (flat magnitude, zero added
/// latency). Band 0 is the lowest.
///
/// Usage: [`Crossover::configure`] (allocates), [`Crossover::set_params`],
/// [`Crossover::process`] with caller-owned band buffers (allocate them at
/// configure time, e.g. `MAX_CROSSOVER_BANDS` `Vec<f32>`s of the largest
/// buffer size), [`Crossover::reset`]. Nothing after `configure` allocates.
#[derive(Debug)]
pub struct Crossover {
    params: CrossoverParams,
    sample_rate: f64,
    channels: usize,
    target: Shape,
    target_ln_f: [f64; MAX_SPLITS],
    shape: Shape,
    glide: Glide<MAX_SPLITS>,
    coefs: Design,
    bank: usize,
    prev_shape: Shape,
    prev_coefs: Design,
    fade_left: usize,
    fade_len: usize,
    /// Two banks of `SLOTS_PER_BANK` DF-I states.
    states: Vec<f64>,
    /// Old-structure band outputs during a crossfade,
    /// `MAX_CROSSOVER_BANDS × FADE_CHUNK × channels`.
    scratch: Vec<f32>,
    snap_pending: bool,
    level: Level,
    level_forced: bool,
}

impl Default for Crossover {
    fn default() -> Self {
        Self::new()
    }
}

impl Crossover {
    /// Create an unconfigured crossover with default parameters. Until
    /// [`Crossover::configure`], `process` copies the input to band 0 and
    /// zeroes the others.
    pub fn new() -> Self {
        Self {
            params: CrossoverParams::default(),
            sample_rate: 0.0,
            channels: 0,
            target: Shape::default(),
            target_ln_f: [0.0; MAX_SPLITS],
            shape: Shape::default(),
            glide: Glide::default(),
            coefs: Design::default(),
            bank: 0,
            prev_shape: Shape::default(),
            prev_coefs: Design::default(),
            fade_left: 0,
            fade_len: 1,
            states: Vec::new(),
            scratch: Vec::new(),
            snap_pending: true,
            level: Level::baseline(),
            level_forced: false,
        }
    }

    /// Set the sample rate (Hz) and channel count (≥ 1). Allocates all
    /// internal state and resets it. The only method that allocates.
    pub fn configure(&mut self, sample_rate: f32, channels: usize) {
        self.sample_rate = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate as f64
        } else {
            48_000.0
        };
        self.channels = channels.max(1);
        self.states = vec![0.0; 2 * self.bank_len()];
        self.scratch = vec![0.0; MAX_CROSSOVER_BANDS * FADE_CHUNK * self.channels];
        self.fade_len = ((CROSSFADE_MS * 0.001 * self.sample_rate).round() as usize).max(1);
        self.glide = Glide::new(self.sample_rate);
        self.bank = 0;
        self.update_target();
        self.snap_pending = true;
        if !self.level_forced {
            self.level = crate::simd::detect();
        }
    }

    /// Override the SIMD level detected in `configure` (persists across
    /// `configure`). Levels with the same [`crate::biquad::uses_fma`] give
    /// bit-identical output.
    pub fn set_simd_level(&mut self, level: Level) {
        self.level = level;
        self.level_forced = true;
    }

    /// The SIMD level the crossover runs at.
    pub fn simd_level(&self) -> Level {
        self.level
    }

    /// Update parameters. Never allocates; out-of-range values are clamped.
    /// Frequency changes glide, order and band-count changes crossfade
    /// (module documentation).
    pub fn set_params(&mut self, params: &CrossoverParams) {
        self.params = *params;
        self.update_target();
    }

    /// Current parameters (as last passed to [`Crossover::set_params`]).
    pub fn params(&self) -> &CrossoverParams {
        &self.params
    }

    /// Number of bands `process` writes (the clamped `num_bands`).
    pub fn num_bands(&self) -> usize {
        self.params.num_bands.clamp(1, MAX_CROSSOVER_BANDS)
    }

    /// Crossover frequencies (Hz) as realised at the configured sample rate
    /// after clamping (entries beyond `num_bands() − 1` are clamped too, but
    /// unused).
    pub fn frequencies_hz(&self) -> [f64; MAX_SPLITS] {
        self.target_ln_f.map(f64::exp)
    }

    /// Clear all filter state; the next `process` takes the current
    /// parameters immediately.
    pub fn reset(&mut self) {
        self.states.iter_mut().for_each(|v| *v = 0.0);
        self.snap_pending = true;
    }

    fn fs(&self) -> f64 {
        if self.sample_rate > 0.0 {
            self.sample_rate
        } else {
            48_000.0
        }
    }

    fn update_target(&mut self) {
        let fs = self.fs();
        self.target = Shape {
            order: self.params.order,
            bands: self.num_bands(),
        };
        let mut prev = 0.0f64;
        for (s, (t, &f)) in self
            .target_ln_f
            .iter_mut()
            .zip(&self.params.frequencies_hz)
            .enumerate()
        {
            let f = clamp_cutoff(f as f64, fs, DEFAULT_FREQUENCIES_HZ[s] as f64).max(prev);
            prev = f;
            *t = f.ln();
        }
        self.glide.retarget(self.target_ln_f);
    }

    fn bank_len(&self) -> usize {
        SLOTS_PER_BANK * STATE_PER_CHANNEL * self.channels
    }

    fn snap(&mut self) {
        self.shape = self.target;
        self.glide.snap(self.target_ln_f);
        self.coefs = Design::new(self.shape.order, &self.target_ln_f, self.sample_rate);
        self.fade_left = 0;
        self.states.iter_mut().for_each(|v| *v = 0.0);
    }

    fn begin_fade(&mut self) {
        self.prev_shape = self.shape;
        self.prev_coefs = self.coefs;
        self.bank ^= 1;
        let len = self.bank_len();
        let start = self.bank * len;
        self.states[start..start + len]
            .iter_mut()
            .for_each(|v| *v = 0.0);
        self.shape = self.target;
        self.glide.snap(self.target_ln_f);
        self.coefs = Design::new(self.shape.order, &self.target_ln_f, self.sample_rate);
        self.fade_left = self.fade_len;
    }

    /// Split interleaved `input` into the bands: `bands[k]` receives band `k`
    /// for `k < num_bands()`; further buffers are left untouched. Each used
    /// buffer must hold at least `input.len()` samples; only the first
    /// `input.len()` are written (a trailing partial frame is left as is).
    /// Real-time safe: never allocates, locks or blocks.
    ///
    /// # Panics
    ///
    /// If fewer than `num_bands()` buffers are passed or one is too short.
    pub fn process<B: AsMut<[f32]>>(&mut self, input: &[f32], bands: &mut [B]) {
        let nb = self.num_bands();
        assert!(bands.len() >= nb, "crossover: {nb} band buffers needed");
        let len = input.len();
        if self.channels == 0 {
            bands[0].as_mut()[..len].copy_from_slice(input);
            for b in &mut bands[1..nb] {
                b.as_mut()[..len].iter_mut().for_each(|v| *v = 0.0);
            }
            return;
        }
        let mut it = bands.iter_mut();
        let mut outs: [&mut [f32]; MAX_CROSSOVER_BANDS] = core::array::from_fn(|k| {
            if k < nb {
                let b = it.next().expect("checked above").as_mut();
                assert!(b.len() >= len, "crossover: band buffer {k} too short");
                b
            } else {
                Default::default()
            }
        });
        let level = self.level;
        fearless_simd::dispatch!(level, simd => self.process_with(simd, input, &mut outs));
    }

    fn process_with<S: Simd>(
        &mut self,
        simd: S,
        input: &[f32],
        outs: &mut [&mut [f32]; MAX_CROSSOVER_BANDS],
    ) {
        if self.snap_pending {
            self.snap();
            self.snap_pending = false;
        }
        let ch = self.channels;
        let frames = input.len() / ch;
        let bank_len = self.bank_len();
        let mut pos = 0;
        while pos < frames {
            // Structural changes start at the beginning of a call, so the
            // realised band count always equals `num_bands()` (the number of
            // buffers the caller passes). A change arriving during a running
            // crossfade ends that crossfade at once (see the module docs).
            if pos == 0 && self.shape != self.target {
                self.begin_fade();
            }
            let mut n = frames - pos;
            if !self.glide.settled() {
                n = n.min(RAMP_BLOCK);
            }
            if self.fade_left > 0 {
                n = n.min(self.fade_left).min(FADE_CHUNK);
            }
            let (a, b) = (pos * ch, (pos + n) * ch);
            let seg_in = &input[a..b];
            if self.fade_left > 0 {
                let mut it = self.scratch.chunks_exact_mut(FADE_CHUNK * ch);
                let mut old: [&mut [f32]; MAX_CROSSOVER_BANDS] =
                    core::array::from_fn(|_| &mut it.next().expect("sized in configure")[..n * ch]);
                let e = Engine {
                    channels: ch,
                    bands: self.prev_shape.bands,
                    cur: &self.prev_coefs,
                    next: None,
                    frames: n,
                    base: (self.bank ^ 1) * bank_len,
                };
                e.run(simd, seg_in, &mut old, &mut self.states);
                // Old bands beyond the new count fold into the new top band,
                // so the band sum crossfades continuously.
                let (nb_old, nb_new) = (self.prev_shape.bands, self.shape.bands);
                if nb_old > nb_new {
                    let (keep, drop) = old.split_at_mut(nb_new);
                    for d in drop.iter().take(nb_old - nb_new) {
                        for (k, &v) in keep[nb_new - 1].iter_mut().zip(d.iter()) {
                            *k += v;
                        }
                    }
                }
                // Bands the old structure did not have fade in from silence.
                for o in &mut old[nb_old..] {
                    o.iter_mut().for_each(|v| *v = 0.0);
                }
            }
            let next = if self.glide.settled() {
                None
            } else {
                let ln_f = self.glide.advance(n);
                Some(Design::new(self.shape.order, &ln_f, self.sample_rate))
            };
            let nb = self.shape.bands;
            let mut it = outs.iter_mut();
            let mut seg: [&mut [f32]; MAX_CROSSOVER_BANDS] = core::array::from_fn(|k| {
                let o = it.next().expect("array length");
                if k < nb {
                    &mut o[a..b]
                } else {
                    Default::default()
                }
            });
            let e = Engine {
                channels: ch,
                bands: nb,
                cur: &self.coefs,
                next: next.as_ref(),
                frames: n,
                base: self.bank * bank_len,
            };
            e.run(simd, seg_in, &mut seg, &mut self.states);
            if let Some(d) = next {
                // Keep the exact end point so the interpolation never drifts.
                self.coefs = d;
            }
            if self.fade_left > 0 {
                let done = self.fade_len - self.fade_left;
                for (k, s) in seg.iter_mut().take(nb).enumerate() {
                    let old = &self.scratch[k * FADE_CHUNK * ch..][..n * ch];
                    blend_into(s, old, ch, done, self.fade_len);
                }
                self.fade_left -= n;
            }
            pos += n;
        }
        self.states
            .chunks_exact_mut(bank_len)
            .for_each(sanitize_bank);
    }
}

/// One pass of a crossover structure over a segment.
struct Engine<'a> {
    channels: usize,
    bands: usize,
    cur: &'a Design,
    /// Coefficients at the end of the segment, if they are being interpolated.
    next: Option<&'a Design>,
    frames: usize,
    /// Start of the state bank.
    base: usize,
}

impl Engine<'_> {
    fn slot(&self, index: usize) -> usize {
        self.base + index * STATE_PER_CHANNEL * self.channels
    }

    /// Topology: the remainder starts as the input; split `s` (lowest first)
    /// takes band `s` as `LP_s(remainder)` followed by the all-passes of every
    /// higher split, and replaces the remainder by `HP_s(remainder)`. The last
    /// band is the final remainder.
    fn run<S: Simd>(&self, simd: S, input: &[f32], outs: &mut [&mut [f32]], states: &mut [f64]) {
        let last = self.bands - 1;
        outs[last].copy_from_slice(input);
        let (lo, hi) = outs.split_at_mut(last);
        let rem = &mut *hi[0];
        for (s, band) in lo.iter_mut().enumerate() {
            band.copy_from_slice(rem);
            let mut st = Stages::default();
            st.push(self, |d| &d.lp[s], LP_BASE + s * 4);
            for j in s + 1..last {
                st.push(self, |d| &d.ap[j], AP_BASE + (s * MAX_SPLITS + j) * 2);
            }
            st.run(simd, band, self.channels, states);
            let mut st = Stages::default();
            st.push(self, |d| &d.hp[s], HP_BASE + s * 4);
            st.run(simd, rem, self.channels, states);
        }
    }
}

/// Stages of one band (or remainder) pass, with optional per-stage ramps.
struct Stages {
    stages: [SteadyStage; MAX_BAND_STAGES],
    ramps: [Option<BiquadCoefficients>; MAX_BAND_STAGES],
    len: usize,
    ramp: bool,
}

impl Default for Stages {
    fn default() -> Self {
        Self {
            stages: [SteadyStage {
                coefs: BiquadCoefficients::IDENTITY,
                state: 0,
            }; MAX_BAND_STAGES],
            ramps: [None; MAX_BAND_STAGES],
            len: 0,
            ramp: false,
        }
    }
}

impl Stages {
    fn push(&mut self, e: &Engine<'_>, pick: impl Fn(&Design) -> &Sections, slot0: usize) {
        let cur = pick(e.cur).as_slice();
        for (i, c) in cur.iter().enumerate() {
            self.stages[self.len] = SteadyStage {
                coefs: *c,
                state: e.slot(slot0 + i),
            };
            if let Some(next) = e.next {
                self.ramps[self.len] = Some(c.ramp_step(&pick(next).as_slice()[i], e.frames));
                self.ramp = true;
            }
            self.len += 1;
        }
    }

    /// Run in chunks of at most `MAX_STAGES` (the kernel limit).
    fn run<S: Simd>(&self, simd: S, data: &mut [f32], channels: usize, states: &mut [f64]) {
        let mut i = 0;
        while i < self.len {
            let j = (i + MAX_STAGES).min(self.len);
            let ramps: &[Option<BiquadCoefficients>] =
                if self.ramp { &self.ramps[i..j] } else { &[] };
            run_chain(simd, data, channels, &self.stages[i..j], ramps, states);
            i = j;
        }
    }
}
