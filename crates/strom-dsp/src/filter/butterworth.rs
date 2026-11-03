//! Multichannel Butterworth low-/high-pass processor (order 1–8).

use fearless_simd::Simd;

use super::design::{butterworth, clamp_cutoff, FilterKind, Sections, MAX_ORDER, MAX_SECTIONS};
use super::{blend_into, Glide, CROSSFADE_MS, FADE_CHUNK, RAMP_BLOCK};
use crate::biquad::{run_chain, sanitize_bank, BiquadCoefficients, SteadyStage, STATE_PER_CHANNEL};
use crate::simd::Level;
use crate::Processor;

/// Parameters of a [`Butterworth`] filter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ButterworthParams {
    /// Low-pass or high-pass.
    pub kind: FilterKind,
    /// Filter order, 1 … 8 (6 dB/octave per order); clamped.
    pub order: usize,
    /// −3.01 dB frequency (Hz); clamped to `[1 Hz, 0.499·fs]`.
    pub cutoff_hz: f32,
}

impl Default for ButterworthParams {
    fn default() -> Self {
        Self {
            kind: FilterKind::HighPass,
            order: 2,
            cutoff_hz: 100.0,
        }
    }
}

/// Default cutoff (Hz) used when a non-finite cutoff is passed.
const DEFAULT_CUTOFF_HZ: f64 = 100.0;

/// Realised structure: kind and order (a change of either is crossfaded).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Shape {
    kind: FilterKind,
    order: usize,
}

/// Butterworth low-pass or high-pass filter of order 1–8 for interleaved
/// `f32` audio, as cascaded biquads (see the [`crate::filter`] module
/// documentation for design, smoothing and SIMD).
///
/// Implements [`Processor`]: `configure` allocates, `set_params`/`process`/
/// `reset` never do.
#[derive(Debug)]
pub struct Butterworth {
    params: ButterworthParams,
    sample_rate: f64,
    channels: usize,
    /// Target structure and `ln(cutoff)`.
    target: Shape,
    target_ln_f: f64,
    /// Structure being faded in (or steady), its smoothed `ln(cutoff)` and
    /// the exact coefficients for that cutoff.
    shape: Shape,
    glide: Glide<1>,
    coefs: Sections,
    /// State bank of `shape` (0 or 1); the other bank belongs to `prev`.
    bank: usize,
    prev_coefs: Sections,
    fade_left: usize,
    fade_len: usize,
    /// Two banks of `MAX_SECTIONS` DF-I states.
    states: Vec<f64>,
    /// Old-structure output during a crossfade, `FADE_CHUNK · channels`.
    scratch: Vec<f32>,
    snap_pending: bool,
    level: Level,
    level_forced: bool,
}

impl Default for Butterworth {
    fn default() -> Self {
        Self::new()
    }
}

impl Butterworth {
    /// Create an unconfigured filter with default parameters
    /// (`process` is a passthrough until [`Processor::configure`]).
    pub fn new() -> Self {
        Self {
            params: ButterworthParams::default(),
            sample_rate: 0.0,
            channels: 0,
            target: Shape::default(),
            target_ln_f: 0.0,
            shape: Shape::default(),
            glide: Glide::default(),
            coefs: Sections::default(),
            bank: 0,
            prev_coefs: Sections::default(),
            fade_left: 0,
            fade_len: 1,
            states: Vec::new(),
            scratch: Vec::new(),
            snap_pending: true,
            level: Level::baseline(),
            level_forced: false,
        }
    }

    /// Override the SIMD level detected in `configure` (persists across
    /// `configure`). Levels with the same [`crate::biquad::uses_fma`] give
    /// bit-identical output.
    pub fn set_simd_level(&mut self, level: Level) {
        self.level = level;
        self.level_forced = true;
    }

    /// The SIMD level the filter runs at.
    pub fn simd_level(&self) -> Level {
        self.level
    }

    /// Current parameters (as last passed to [`Butterworth::set_params`]).
    pub fn params(&self) -> &ButterworthParams {
        &self.params
    }

    /// Update parameters. Never allocates; out-of-range values are clamped.
    /// Cutoff changes glide, kind/order changes crossfade (module docs).
    pub fn set_params(&mut self, params: &ButterworthParams) {
        self.params = *params;
        self.update_target();
    }

    /// Sections the filter realises at `sample_rate` (Hz) for `params`, after
    /// clamping (the steady-state transfer function).
    pub fn design(params: &ButterworthParams, sample_rate: f32) -> Sections {
        let fs = sample_rate as f64;
        butterworth(
            params.kind,
            params.order.clamp(1, MAX_ORDER),
            clamp_cutoff(params.cutoff_hz as f64, fs, DEFAULT_CUTOFF_HZ),
            fs,
        )
    }

    fn fs(&self) -> f64 {
        if self.sample_rate > 0.0 {
            self.sample_rate
        } else {
            48_000.0
        }
    }

    fn update_target(&mut self) {
        let p = &self.params;
        self.target = Shape {
            kind: p.kind,
            order: p.order.clamp(1, MAX_ORDER),
        };
        self.target_ln_f = clamp_cutoff(p.cutoff_hz as f64, self.fs(), DEFAULT_CUTOFF_HZ).ln();
        self.glide.retarget([self.target_ln_f]);
    }

    fn sections_for(&self, shape: Shape, ln_f: f64) -> Sections {
        butterworth(shape.kind, shape.order, ln_f.exp(), self.sample_rate)
    }

    fn bank_len(&self) -> usize {
        MAX_SECTIONS * STATE_PER_CHANNEL * self.channels
    }

    /// Jump to the target (no glide, no crossfade) and clear the state.
    fn snap(&mut self) {
        self.shape = self.target;
        self.glide.snap([self.target_ln_f]);
        self.coefs = self.sections_for(self.shape, self.target_ln_f);
        self.fade_left = 0;
        self.states.iter_mut().for_each(|v| *v = 0.0);
    }

    fn begin_fade(&mut self) {
        self.prev_coefs = self.coefs;
        self.bank ^= 1;
        let len = self.bank_len();
        let start = self.bank * len;
        self.states[start..start + len]
            .iter_mut()
            .for_each(|v| *v = 0.0);
        self.shape = self.target;
        self.glide.snap([self.target_ln_f]);
        self.coefs = self.sections_for(self.shape, self.target_ln_f);
        self.fade_left = self.fade_len;
    }

    fn stages(&self, coefs: &Sections, bank: usize) -> ([SteadyStage; MAX_SECTIONS], usize) {
        let base = bank * self.bank_len();
        let step = STATE_PER_CHANNEL * self.channels;
        let mut out = [SteadyStage {
            coefs: BiquadCoefficients::IDENTITY,
            state: 0,
        }; MAX_SECTIONS];
        for (i, c) in coefs.as_slice().iter().enumerate() {
            out[i] = SteadyStage {
                coefs: *c,
                state: base + i * step,
            };
        }
        (out, coefs.len())
    }

    fn process_with<S: Simd>(&mut self, simd: S, samples: &mut [f32]) {
        if self.snap_pending {
            self.snap();
            self.snap_pending = false;
        }
        let ch = self.channels;
        let frames = samples.len() / ch;
        let mut pos = 0;
        while pos < frames {
            if self.fade_left == 0 && self.shape != self.target {
                self.begin_fade();
            }
            let mut n = frames - pos;
            if !self.glide.settled() {
                n = n.min(RAMP_BLOCK);
            }
            if self.fade_left > 0 {
                n = n.min(self.fade_left).min(FADE_CHUNK);
            }
            let data = &mut samples[pos * ch..(pos + n) * ch];
            if self.fade_left > 0 {
                let (st, len) = self.stages(&self.prev_coefs, self.bank ^ 1);
                let old = &mut self.scratch[..n * ch];
                old.copy_from_slice(data);
                run_chain(simd, old, ch, &st[..len], &[], &mut self.states);
            }
            // The new (or only) structure, gliding if needed.
            let (st, len) = self.stages(&self.coefs, self.bank);
            if self.glide.settled() {
                run_chain(simd, data, ch, &st[..len], &[], &mut self.states);
            } else {
                let ln_f = self.glide.advance(n)[0];
                let next = self.sections_for(self.shape, ln_f);
                let mut ramps = [None; MAX_SECTIONS];
                for (r, (a, b)) in ramps
                    .iter_mut()
                    .zip(self.coefs.as_slice().iter().zip(next.as_slice()))
                {
                    *r = Some(a.ramp_step(b, n));
                }
                run_chain(simd, data, ch, &st[..len], &ramps[..len], &mut self.states);
                // Keep the exact end point so the interpolation never drifts.
                self.coefs = next;
            }
            if self.fade_left > 0 {
                let done = self.fade_len - self.fade_left;
                blend_into(data, &self.scratch[..n * ch], ch, done, self.fade_len);
                self.fade_left -= n;
            }
            pos += n;
        }
        let len = self.bank_len();
        self.states.chunks_exact_mut(len).for_each(sanitize_bank);
    }
}

impl Processor for Butterworth {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        self.sample_rate = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate as f64
        } else {
            48_000.0
        };
        self.channels = channels.max(1);
        self.states = vec![0.0; 2 * self.bank_len()];
        self.scratch = vec![0.0; FADE_CHUNK * self.channels];
        self.fade_len = ((CROSSFADE_MS * 0.001 * self.sample_rate).round() as usize).max(1);
        self.glide = Glide::new(self.sample_rate);
        self.bank = 0;
        self.update_target();
        self.snap_pending = true;
        if !self.level_forced {
            self.level = crate::simd::detect();
        }
    }

    fn process(&mut self, samples: &mut [f32]) {
        if self.channels == 0 {
            return;
        }
        let level = self.level;
        fearless_simd::dispatch!(level, simd => self.process_with(simd, samples));
    }

    fn reset(&mut self) {
        self.states.iter_mut().for_each(|v| *v = 0.0);
        self.snap_pending = true;
    }
}
