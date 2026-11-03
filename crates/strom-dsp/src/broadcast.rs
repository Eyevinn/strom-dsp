//! Broadcast final processor chain:
//! input gain → wideband AGC → multiband compressor → true-peak limiter.
//!
//! Units follow the element interface (`stromdsp-broadcastprocessor`): dB,
//! LUFS, dBTP, dB/s, ms, Hz. Meter conventions follow the element's public
//! contract (its meter message):
//! gain reductions are **positive dB of reduction**, the AGC gain is
//! **signed** (+ = boost), and silence reads [`SILENCE_DB`].
//!
//! # Signal flow
//!
//! ```text
//!         ┌──────────────── dry: delay (total latency) ─────────────────┐
//! input ──┤                                                             ├─ fade ─→ output
//!         └→ gain → [AGC] → [multiband] → [limiter] ────────────────────┘
//!                     each [stage] = fade(stage(x), delay_stage(x))
//! ```
//!
//! - **Input gain**: glides in dB through two cascaded one-poles of 5 ms
//!   each (critically damped, starts with zero slope); a settled 0 dB is
//!   skipped (bit-exact).
//! - **AGC**: [`crate::agc::Agc`] (gated, rate-limited loudness rider; no
//!   latency).
//! - **Multiband**: [`crate::multiband::MultibandCompressor`] (LR4
//!   crossover, compressor per band; no latency, an all-pass at ratio 1).
//! - **Limiter**: [`crate::truepeak::TruePeakLimiter`] with ceiling
//!   `true_peak_ceiling_dbtp` (dBTP, BS.1770 Annex 2 detector); its latency
//!   (lookahead + 32 frames of detector delay) is the whole chain latency.
//!
//! # Bypass and latency
//!
//! Every stage and the whole chain (`enabled`) have a **raised-cosine
//! crossfade** of 10 ms between the processed ("wet") and the
//! unprocessed ("dry") signal. The dry path of a stage with latency `L` is
//! its input delayed by exactly `L` frames, and the dry path of the chain is
//! the raw input delayed by the total latency, so the latency never changes
//! with bypass. A settled fade is exact: fully wet passes the stage output
//! unchanged, fully dry copies the delayed input bit for bit. With
//! `enabled = false` (after the fade) the output is therefore the input
//! delayed by [`Processor::latency_frames`], sample for sample.
//!
//! Bypassed stages, and the whole chain when disabled, **keep running** on
//! their input: their detectors and gains stay current, so switching back
//! is a plain crossfade to a settled signal, and the CPU load does not
//! depend on the bypass state. The multiband stage is an all-pass even at
//! ratio 1, so its crossfade briefly mixes two phase responses (a short
//! comb, for the 10 ms of the fade only).
//!
//! Between `configure`/`reset` and the first processed frame, `set_params`
//! applies bypass states and the input gain instantly (no fade).
//!
//! The latency is `agc latency (0) + limiter latency`, i.e.
//! `round(lookahead · fs / 1000) + 32` frames (272 at 48 kHz with the
//! default 5 ms), and depends only on the sample rate and the lookahead ([`BroadcastProcessor::latency_frames_for`]).
//! `num_bands` and `limiter_lookahead_ms` are structural: they take effect
//! at the next [`Processor::configure`].
//!
//! # Meters
//!
//! | Value | Source | Cost |
//! |---|---|---|
//! | input momentary / short-term | a BS.1770 [`LoudnessDetector`] on the raw input (**before the input gain**) | ≈ 1.3 µs per 480 stereo frames, O(1) read |
//! | output momentary / short-term | a second [`LoudnessDetector`] on the final output | ≈ 1.3 µs per 480 stereo frames, O(1) read |
//! | output integrated | [`LoudnessMeter`] (histogram mode, no true peak) on the final output | ≈ 7 µs per 480 stereo frames, < 1 µs read |
//! | input / output true peak | 4× [`Oversampler`] peaks (max over sub-samples and channels) of the raw input (before the input gain) / of the final output, held until [`BroadcastProcessor::take_meters`] | upsampling per block |
//! | AGC gain | [`crate::agc::Agc::gain_db`] | O(1) |
//! | band gain reduction | [`crate::multiband::MultibandCompressor::gain_reduction_db`] at read time | O(1) |
//! | limiter gain reduction | max of the value at the end of each internal chunk since the last read | O(1) |
//!
//! The input meters show what actually comes in: they are measured before
//! the input gain, so `input_gain_db` does not change them.
//!
//! The AGC gain and the gain reductions read 0 while their stage (or the
//! chain) is switched off, i.e. they show what is being applied.
//! [`BroadcastProcessor::take_meters`] is O(1) and allocation-free: call it
//! once per buffer if needed.

use crate::agc::{Agc, AgcParams};
use crate::dynamics::OnePole;
use crate::meter::{ChannelLayout, LoudnessConfig, LoudnessDetector, LoudnessMeter};
use crate::multiband::{MultibandBandParams, MultibandCompressor, MultibandParams, MAX_BANDS};
use crate::oversample::{OversampleFactor, Oversampler};
use crate::truepeak::{self, TruePeakLimiter, TruePeakLimiterParams};
use crate::util::{db_to_lin, lin_to_db};
use crate::Processor;

/// Length of every bypass / `enabled` crossfade (ms, raised cosine).
const FADE_MS: f64 = 10.0;

/// Time constant of each of the two cascaded one-poles of the input-gain
/// glide (ms).
const GAIN_GLIDE_MS: f64 = 5.0;
/// Snap distance (dB) of the input-gain glide.
const GAIN_SNAP_DB: f64 = 1e-6;
/// AGC detector window (ms): the EBU Tech 3341 short-term length.
const AGC_WINDOW_MS: f32 = 3000.0;
/// Frames per internal chunk (sizes the scratch buffers).
const CHUNK: usize = 512;

/// Value reported for silence / no measurement yet (never NaN or −∞).
pub const SILENCE_DB: f64 = -200.0;

/// Number of crossover frequencies (`MAX_BANDS − 1`).
pub const MAX_CROSSOVERS: usize = MAX_BANDS - 1;

/// All parameters of the chain (the element's properties, minus `preset` and
/// `meter-interval`, which belong to the element).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BroadcastParams {
    /// Unity gain with the same delay when false (true bypass).
    pub enabled: bool,
    /// Input gain (dB, −24 … 24).
    pub input_gain_db: f32,
    /// AGC target (LUFS, −40 … −5).
    pub target_loudness_lufs: f32,
    /// True-peak ceiling (dBTP, −12 … 0).
    pub true_peak_ceiling_dbtp: f32,
    /// AGC stage active (bypassed stages keep their latency).
    pub agc_enabled: bool,
    /// Largest AGC boost (dB, 0 … 30).
    pub agc_max_gain_db: f32,
    /// Largest AGC cut (dB, 0 … 30).
    pub agc_max_cut_db: f32,
    /// AGC gain rise rate (dB/s, 0.1 … 20).
    pub agc_rate_up_db_s: f32,
    /// AGC gain fall rate (dB/s, 0.1 … 40).
    pub agc_rate_down_db_s: f32,
    /// AGC freeze level (LUFS, −80 … −20): below it the gain holds.
    pub agc_gate_lufs: f32,
    /// Multiband stage active.
    pub multiband_enabled: bool,
    /// Number of bands (1 … 5). Structural (READY-only on the element).
    pub num_bands: usize,
    /// Gain into all bands (dB, −12 … 12).
    pub multiband_drive_db: f32,
    /// Crossover frequencies (Hz, 20 … 20000), ascending; the first
    /// `num_bands − 1` are used.
    pub crossover_hz: [f32; MAX_CROSSOVERS],
    /// Per-band compressor settings; band 0 is the lowest.
    pub bands: [MultibandBandParams; MAX_BANDS],
    /// Limiter stage active.
    pub limiter_enabled: bool,
    /// Limiter release (ms, 1 … 2000).
    pub limiter_release_ms: f32,
    /// Limiter lookahead (ms, 0.5 … 20). Structural (READY-only on the element).
    pub limiter_lookahead_ms: f32,
}

impl Default for BroadcastParams {
    /// Neutral library defaults. They share the top-level values of the
    /// `ebu-r128` preset (target, ceiling, AGC range and rates, crossovers)
    /// but not its band settings or limiter release. The element applies
    /// the full `ebu-r128` preset ([`BroadcastParams::preset`]) as its
    /// default.
    fn default() -> Self {
        Self {
            enabled: true,
            input_gain_db: 0.0,
            target_loudness_lufs: -23.0,
            true_peak_ceiling_dbtp: -1.0,
            agc_enabled: true,
            agc_max_gain_db: 12.0,
            agc_max_cut_db: 12.0,
            agc_rate_up_db_s: 1.0,
            agc_rate_down_db_s: 3.0,
            agc_gate_lufs: -50.0,
            multiband_enabled: true,
            num_bands: 4,
            multiband_drive_db: 0.0,
            crossover_hz: [120.0, 800.0, 4000.0, 10_000.0],
            bands: core::array::from_fn(|k| MultibandBandParams::default_for(k, 4)),
            limiter_enabled: true,
            limiter_release_ms: 100.0,
            limiter_lookahead_ms: 5.0,
        }
    }
}

/// Meter values of the chain. Read once per block by
/// the element; dB/LUFS values are [`SILENCE_DB`] when there is nothing to
/// measure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BroadcastMeters {
    /// Input momentary loudness (LUFS, 400 ms), measured before the input gain.
    pub input_momentary_lufs: f64,
    /// Input short-term loudness (LUFS, 3 s), measured before the input gain.
    pub input_short_term_lufs: f64,
    /// Output momentary loudness (LUFS).
    pub output_momentary_lufs: f64,
    /// Output short-term loudness (LUFS).
    pub output_short_term_lufs: f64,
    /// Output gated integrated loudness since configure/reset/`reset_integrated` (LUFS).
    pub output_integrated_lufs: f64,
    /// Input true peak (dBTP) before the input gain, max since the last
    /// [`BroadcastProcessor::take_meters`].
    pub input_true_peak_dbtp: f64,
    /// Output true peak (dBTP), max since the last [`BroadcastProcessor::take_meters`].
    pub output_true_peak_dbtp: f64,
    /// AGC gain (dB, signed: + = boost).
    pub agc_gain_db: f64,
    /// Gain reduction per band (positive dB); only the first `num_bands` are meaningful.
    pub band_gain_reduction_db: [f64; MAX_BANDS],
    /// Limiter gain reduction (positive dB).
    pub limiter_gain_reduction_db: f64,
}

impl Default for BroadcastMeters {
    fn default() -> Self {
        Self {
            input_momentary_lufs: SILENCE_DB,
            input_short_term_lufs: SILENCE_DB,
            output_momentary_lufs: SILENCE_DB,
            output_short_term_lufs: SILENCE_DB,
            output_integrated_lufs: SILENCE_DB,
            input_true_peak_dbtp: SILENCE_DB,
            output_true_peak_dbtp: SILENCE_DB,
            agc_gain_db: 0.0,
            band_gain_reduction_db: [0.0; MAX_BANDS],
            limiter_gain_reduction_db: 0.0,
        }
    }
}

/// Clamp `v` into `[lo, hi]`; NaN becomes `fallback`.
fn clamp_or(v: f32, lo: f32, hi: f32, fallback: f32) -> f32 {
    if v.is_nan() {
        fallback
    } else {
        v.clamp(lo, hi)
    }
}

impl BroadcastParams {
    /// Copy with every field clamped to its range (NaN becomes the default).
    /// Crossover frequencies are clamped to 20 … 20 000 Hz and made
    /// non-decreasing (a frequency below its predecessor is raised to it).
    fn clamped(&self) -> Self {
        let d = Self::default();
        let mut crossover_hz = self.crossover_hz;
        let mut prev = 20.0f32;
        for (f, df) in crossover_hz.iter_mut().zip(d.crossover_hz) {
            *f = clamp_or(*f, 20.0, 20_000.0, df).max(prev);
            prev = *f;
        }
        Self {
            enabled: self.enabled,
            input_gain_db: clamp_or(self.input_gain_db, -24.0, 24.0, d.input_gain_db),
            target_loudness_lufs: clamp_or(
                self.target_loudness_lufs,
                -40.0,
                -5.0,
                d.target_loudness_lufs,
            ),
            true_peak_ceiling_dbtp: clamp_or(
                self.true_peak_ceiling_dbtp,
                -12.0,
                0.0,
                d.true_peak_ceiling_dbtp,
            ),
            agc_enabled: self.agc_enabled,
            agc_max_gain_db: clamp_or(self.agc_max_gain_db, 0.0, 30.0, d.agc_max_gain_db),
            agc_max_cut_db: clamp_or(self.agc_max_cut_db, 0.0, 30.0, d.agc_max_cut_db),
            agc_rate_up_db_s: clamp_or(self.agc_rate_up_db_s, 0.1, 20.0, d.agc_rate_up_db_s),
            agc_rate_down_db_s: clamp_or(self.agc_rate_down_db_s, 0.1, 40.0, d.agc_rate_down_db_s),
            agc_gate_lufs: clamp_or(self.agc_gate_lufs, -80.0, -20.0, d.agc_gate_lufs),
            multiband_enabled: self.multiband_enabled,
            num_bands: self.num_bands.clamp(1, MAX_BANDS),
            multiband_drive_db: clamp_or(self.multiband_drive_db, -12.0, 12.0, 0.0),
            crossover_hz,
            bands: core::array::from_fn(|k| self.bands[k].clamped()),
            limiter_enabled: self.limiter_enabled,
            limiter_release_ms: clamp_or(
                self.limiter_release_ms,
                1.0,
                2000.0,
                d.limiter_release_ms,
            ),
            limiter_lookahead_ms: clamp_lookahead(self.limiter_lookahead_ms),
        }
    }
}

/// The lookahead range of the element (ms, 0.5 … 20; NaN/Inf → 5).
fn clamp_lookahead(ms: f32) -> f32 {
    if ms.is_finite() {
        ms.clamp(0.5, 20.0)
    } else {
        5.0
    }
}

/// `x` (dB/LUFS) with NaN and everything below [`SILENCE_DB`] mapped to it.
fn or_silence(x: f64) -> f64 {
    if x >= SILENCE_DB {
        x
    } else {
        SILENCE_DB
    }
}

// ---------------------------------------------------------------------------
// Stage adapters. Each wraps one stage behind the same small interface, so
// swapping an implementation touches only its adapter.

/// AGC stage: [`Agc`] with the chain's parameters. Latency 0.
#[derive(Debug, Default)]
struct AgcStage {
    inner: Agc,
}

impl AgcStage {
    /// Latency of the stage (frames).
    fn latency_frames_for(_sample_rate: f32) -> usize {
        0
    }

    fn configure(&mut self, sample_rate: f32, channels: usize) {
        self.inner
            .configure_with_layout(sample_rate, channels, ChannelLayout::Unknown);
    }

    fn set_params(&mut self, p: &BroadcastParams) {
        self.inner.set_params(&AgcParams {
            target_lufs: p.target_loudness_lufs,
            max_gain_db: p.agc_max_gain_db,
            max_cut_db: p.agc_max_cut_db,
            rate_up_db_per_s: p.agc_rate_up_db_s,
            rate_down_db_per_s: p.agc_rate_down_db_s,
            gate_lufs: p.agc_gate_lufs,
            window_ms: AGC_WINDOW_MS,
        });
    }

    fn process(&mut self, x: &mut [f32]) {
        self.inner.process(x);
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn gain_db(&self) -> f64 {
        self.inner.gain_db()
    }
}

/// Limiter stage: [`TruePeakLimiter`] (BS.1770 Annex 2 detector). Its own
/// `enabled` stays on: the chain does the bypass. Latency = lookahead +
/// detector delay.
#[derive(Debug, Default)]
struct LimiterStage {
    inner: TruePeakLimiter,
}

impl LimiterStage {
    /// Latency of the stage (frames) for a sample rate and a lookahead
    /// (already clamped to the element range).
    fn latency_frames_for(sample_rate: f32, lookahead_ms: f32) -> usize {
        truepeak::latency_frames_for(lookahead_ms, sample_rate)
    }

    fn configure(&mut self, sample_rate: f32, channels: usize) {
        self.inner.configure(sample_rate, channels);
    }

    fn set_params(&mut self, p: &BroadcastParams, lookahead_ms: f32) {
        self.inner.set_params(&TruePeakLimiterParams {
            enabled: true,
            ceiling_dbtp: p.true_peak_ceiling_dbtp,
            release_ms: p.limiter_release_ms,
            lookahead_ms,
        });
    }

    fn process(&mut self, x: &mut [f32]) {
        self.inner.process(x);
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Current gain reduction (positive dB).
    fn gain_reduction_db(&self) -> f64 {
        self.inner.gain_reduction_db()
    }

    fn latency_frames(&self) -> usize {
        self.inner.latency_frames()
    }
}

// ---------------------------------------------------------------------------
// Building blocks of the chain.

/// Fixed delay of whole interleaved frames, processed in place (a ring that
/// swaps its contents with the signal). A delay of 0 is a no-op.
#[derive(Debug, Default)]
struct FixedDelay {
    buf: Vec<f32>,
    pos: usize,
}

impl FixedDelay {
    /// Allocate a delay of `frames` frames of `channels` channels.
    fn allocate(&mut self, frames: usize, channels: usize) {
        self.buf = vec![0.0; frames * channels];
        self.pos = 0;
    }

    fn clear(&mut self) {
        self.buf.fill(0.0);
        self.pos = 0;
    }

    /// Replace `data` by the signal of `buf.len()` samples ago.
    fn process(&mut self, data: &mut [f32]) {
        let n = self.buf.len();
        if n == 0 {
            return;
        }
        let mut rest = data;
        while !rest.is_empty() {
            let k = (n - self.pos).min(rest.len());
            let (head, tail) = rest.split_at_mut(k);
            self.buf[self.pos..self.pos + k].swap_with_slice(head);
            self.pos += k;
            if self.pos == n {
                self.pos = 0;
            }
            rest = tail;
        }
    }
}

/// Raised-cosine crossfade between a wet and a dry signal. `pos` counts
/// frames from 0 (dry) to `len` (wet); the wet weight is
/// `½ − ½·cos(π·pos/len)`.
#[derive(Debug, Clone, Copy)]
struct Fade {
    pos: usize,
    len: usize,
    on: bool,
}

impl Default for Fade {
    fn default() -> Self {
        Self {
            pos: 1,
            len: 1,
            on: true,
        }
    }
}

impl Fade {
    /// New fade length (frames ≥ 1); snaps to the current target.
    fn set_len(&mut self, len: usize) {
        self.len = len.max(1);
        self.snap();
    }

    fn snap(&mut self) {
        self.pos = if self.on { self.len } else { 0 };
    }

    /// Set the target state; `snap` jumps there at once.
    fn set(&mut self, on: bool, snap: bool) {
        self.on = on;
        if snap {
            self.snap();
        }
    }

    fn is_wet(&self) -> bool {
        self.on && self.pos == self.len
    }

    /// Mix `dry` into `wet` (in place) frame by frame, advancing the fade.
    /// A settled fade is exact: wet is left alone, dry is copied.
    fn mix(&mut self, wet: &mut [f32], dry: &[f32], ch: usize) {
        if self.is_wet() {
            return;
        }
        if !self.on && self.pos == 0 {
            wet.copy_from_slice(dry);
            return;
        }
        let inv = core::f64::consts::PI / self.len as f64;
        for (w, d) in wet.chunks_exact_mut(ch).zip(dry.chunks_exact(ch)) {
            if self.on {
                self.pos = (self.pos + 1).min(self.len);
            } else {
                self.pos = self.pos.saturating_sub(1);
            }
            if self.pos == self.len {
                continue;
            }
            if self.pos == 0 {
                w.copy_from_slice(d);
                continue;
            }
            let g = (0.5 - 0.5 * (inv * self.pos as f64).cos()) as f32;
            for (w, &d) in w.iter_mut().zip(d) {
                *w = d + g * (*w - d);
            }
        }
    }
}

/// Run one stage with its bypass: `data` becomes the fade between
/// `stage(data)` and `data` delayed by `delay` (the stage's latency). The
/// delay is fed even while fully wet, so a fade can start at any time.
fn run_stage(
    fade: &mut Fade,
    delay: &mut FixedDelay,
    scratch: &mut [f32],
    data: &mut [f32],
    ch: usize,
    stage: impl FnOnce(&mut [f32]),
) {
    let need_dry = !fade.is_wet() || !delay.buf.is_empty();
    if need_dry {
        scratch.copy_from_slice(data);
        delay.process(scratch);
    }
    stage(data);
    fade.mix(data, scratch, ch);
}

/// Input gain glide: two cascaded one-poles in dB.
#[derive(Debug, Clone, Copy, Default)]
struct GainGlide {
    pre: OnePole,
    cur: OnePole,
    target_db: f64,
    settled: bool,
}

impl GainGlide {
    fn set_rate(&mut self, fs: f64) {
        self.pre = OnePole::new(GAIN_GLIDE_MS, fs, self.target_db).with_snap(GAIN_SNAP_DB);
        self.cur = OnePole::new(GAIN_GLIDE_MS, fs, self.target_db).with_snap(GAIN_SNAP_DB);
        self.settled = true;
    }

    fn snap(&mut self) {
        self.pre.set_value(self.target_db);
        self.cur.set_value(self.target_db);
        self.settled = true;
    }

    fn retarget(&mut self, db: f64) {
        if db != self.target_db {
            self.target_db = db;
            self.settled = false;
        }
    }

    fn apply(&mut self, data: &mut [f32], ch: usize) {
        if self.settled {
            if self.target_db != 0.0 {
                let g = db_to_lin(self.target_db) as f32;
                data.iter_mut().for_each(|v| *v *= g);
            }
            return;
        }
        for frame in data.chunks_exact_mut(ch) {
            let p = self.pre.step(self.target_db);
            let g = db_to_lin(self.cur.step(p));
            for v in frame {
                *v = (*v as f64 * g) as f32;
            }
        }
        if self.cur.value() == self.target_db && self.pre.value() == self.target_db {
            self.settled = true;
        }
    }
}

/// The broadcast final processor. See the module documentation.
#[derive(Debug)]
pub struct BroadcastProcessor {
    /// Clamped parameters as last set.
    params: BroadcastParams,
    sample_rate: f32,
    channels: usize,
    /// Structural values in force since the last `configure`.
    active_lookahead_ms: f32,
    active_bands: usize,
    /// True from configure/reset until the first processed frame.
    fresh: bool,

    input_gain: GainGlide,
    agc: AgcStage,
    multiband: MultibandCompressor,
    limiter: LimiterStage,

    agc_fade: Fade,
    multiband_fade: Fade,
    limiter_fade: Fade,
    chain_fade: Fade,
    agc_delay: FixedDelay,
    multiband_delay: FixedDelay,
    limiter_delay: FixedDelay,
    chain_delay: FixedDelay,
    /// Dry signal of the current stage (`CHUNK × channels`).
    stage_dry: Vec<f32>,
    /// Dry signal of the whole chain (`CHUNK × channels`).
    chain_dry: Vec<f32>,
    /// Per-frame true-peak estimates (`CHUNK`).
    peaks: Vec<f32>,

    /// Loudness of the raw input (before the input gain).
    input_detector: LoudnessDetector,
    output_detector: LoudnessDetector,
    integrated: Option<LoudnessMeter>,
    input_tp: Oversampler,
    output_tp: Oversampler,
    input_peak: f32,
    output_peak: f32,
    limiter_gr_max: f64,
}

impl Default for BroadcastProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl BroadcastProcessor {
    /// Create an unconfigured processor with default parameters.
    pub fn new() -> Self {
        let params = BroadcastParams::default();
        let mut s = Self {
            params,
            sample_rate: 0.0,
            channels: 0,
            active_lookahead_ms: params.limiter_lookahead_ms,
            active_bands: params.num_bands,
            fresh: true,
            input_gain: GainGlide::default(),
            agc: AgcStage::default(),
            multiband: MultibandCompressor::new(),
            limiter: LimiterStage::default(),
            agc_fade: Fade::default(),
            multiband_fade: Fade::default(),
            limiter_fade: Fade::default(),
            chain_fade: Fade::default(),
            agc_delay: FixedDelay::default(),
            multiband_delay: FixedDelay::default(),
            limiter_delay: FixedDelay::default(),
            chain_delay: FixedDelay::default(),
            stage_dry: Vec::new(),
            chain_dry: Vec::new(),
            peaks: Vec::new(),
            input_detector: LoudnessDetector::new(),
            output_detector: LoudnessDetector::new(),
            integrated: None,
            input_tp: Oversampler::new(),
            output_tp: Oversampler::new(),
            input_peak: 0.0,
            output_peak: 0.0,
            limiter_gr_max: 0.0,
        };
        s.set_params(&params);
        s
    }

    /// Update parameters (clamped). Never allocates. Structural fields
    /// (`num_bands`, `limiter_lookahead_ms`) take effect at the next
    /// `configure`.
    pub fn set_params(&mut self, params: &BroadcastParams) {
        let p = params.clamped();
        self.params = p;
        if self.channels == 0 {
            // Unconfigured: the structural values follow until `configure`.
            self.active_lookahead_ms = p.limiter_lookahead_ms;
            self.active_bands = p.num_bands;
        }
        let snap = self.fresh;
        self.input_gain.retarget(p.input_gain_db as f64);
        if snap {
            self.input_gain.snap();
        }
        self.agc.set_params(&p);
        self.multiband.set_params(&MultibandParams {
            num_bands: self.active_bands,
            frequencies_hz: p.crossover_hz,
            drive_db: p.multiband_drive_db,
            bands: p.bands,
            ..MultibandParams::default()
        });
        self.limiter.set_params(&p, self.active_lookahead_ms);
        self.agc_fade.set(p.agc_enabled, snap);
        self.multiband_fade.set(p.multiband_enabled, snap);
        self.limiter_fade.set(p.limiter_enabled, snap);
        self.chain_fade.set(p.enabled, snap);
    }

    /// Current parameters.
    pub fn params(&self) -> &BroadcastParams {
        &self.params
    }

    /// Meter values since the last call; resets the true-peak maxima.
    /// Never allocates. Cheap enough to call once per buffer.
    pub fn take_meters(&mut self) -> BroadcastMeters {
        if self.channels == 0 {
            return BroadcastMeters::default();
        }
        let on = self.params.enabled;
        let input = &self.input_detector;
        let mut band_gain_reduction_db = [0.0; MAX_BANDS];
        if on && self.params.multiband_enabled {
            band_gain_reduction_db = self.multiband.gain_reduction_db();
        }
        let m = BroadcastMeters {
            input_momentary_lufs: or_silence(input.momentary_lufs()),
            input_short_term_lufs: or_silence(input.short_term_lufs()),
            output_momentary_lufs: or_silence(self.output_detector.momentary_lufs()),
            output_short_term_lufs: or_silence(self.output_detector.short_term_lufs()),
            output_integrated_lufs: or_silence(
                self.integrated
                    .as_ref()
                    .map_or(SILENCE_DB, LoudnessMeter::integrated),
            ),
            input_true_peak_dbtp: or_silence(lin_to_db(self.input_peak as f64)),
            output_true_peak_dbtp: or_silence(lin_to_db(self.output_peak as f64)),
            agc_gain_db: if on && self.params.agc_enabled {
                self.agc.gain_db()
            } else {
                0.0
            },
            band_gain_reduction_db,
            limiter_gain_reduction_db: if on && self.params.limiter_enabled {
                self.limiter_gr_max
            } else {
                0.0
            },
        };
        self.input_peak = 0.0;
        self.output_peak = 0.0;
        self.limiter_gr_max = self.limiter.gain_reduction_db();
        m
    }

    /// Restart the output integrated loudness measurement.
    pub fn reset_integrated(&mut self) {
        if let Some(m) = &mut self.integrated {
            m.reset();
        }
    }

    /// Total latency in frames for a configuration (lookahead + oversampling
    /// filter delay). Constant for a given rate and lookahead, independent of
    /// stage bypass. Usable before `configure`.
    pub fn latency_frames_for(sample_rate: f32, lookahead_ms: f32) -> usize {
        AgcStage::latency_frames_for(sample_rate)
            + LimiterStage::latency_frames_for(sample_rate, clamp_lookahead(lookahead_ms))
    }

    /// Process one chunk (at most `CHUNK` frames).
    fn process_chunk(&mut self, chunk: &mut [f32]) {
        let ch = self.channels;
        let len = chunk.len();
        let frames = len / ch;

        // Dry path of the whole chain: the raw input, delayed.
        let chain_dry = &mut self.chain_dry[..len];
        chain_dry.copy_from_slice(chunk);
        self.chain_delay.process(chain_dry);

        // Input meters: the raw input, before the input gain.
        self.input_detector.process(chunk);
        self.input_tp
            .upsample_peaks(chunk, &mut self.peaks[..frames]);
        self.input_peak = self.peaks[..frames]
            .iter()
            .fold(self.input_peak, |a, &b| a.max(b));

        self.input_gain.apply(chunk, ch);

        let dry = &mut self.stage_dry[..len];
        let agc = &mut self.agc;
        run_stage(
            &mut self.agc_fade,
            &mut self.agc_delay,
            dry,
            chunk,
            ch,
            |x| agc.process(x),
        );
        let mb = &mut self.multiband;
        run_stage(
            &mut self.multiband_fade,
            &mut self.multiband_delay,
            dry,
            chunk,
            ch,
            |x| mb.process(x),
        );
        let lim = &mut self.limiter;
        run_stage(
            &mut self.limiter_fade,
            &mut self.limiter_delay,
            dry,
            chunk,
            ch,
            |x| lim.process(x),
        );
        self.limiter_gr_max = self.limiter_gr_max.max(self.limiter.gain_reduction_db());

        self.chain_fade.mix(chunk, &self.chain_dry[..len], ch);

        self.output_detector.process(chunk);
        if let Some(m) = &mut self.integrated {
            m.process(chunk);
        }
        self.output_tp
            .upsample_peaks(chunk, &mut self.peaks[..frames]);
        self.output_peak = self.peaks[..frames]
            .iter()
            .fold(self.output_peak, |a, &b| a.max(b));
    }
}

impl Processor for BroadcastProcessor {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        self.sample_rate = sample_rate;
        self.channels = channels;
        if channels == 0 {
            return;
        }
        let p = self.params;
        self.active_lookahead_ms = p.limiter_lookahead_ms;
        self.active_bands = p.num_bands;

        self.agc.configure(sample_rate, channels);
        self.multiband.configure(sample_rate, channels);
        self.limiter.configure(sample_rate, channels);
        // Apply the parameters (structural values included) before sizing
        // the delays from the limiter's realised latency.
        self.set_params(&p);
        let agc_latency = AgcStage::latency_frames_for(sample_rate);
        let limiter_latency = self.limiter.latency_frames();
        debug_assert_eq!(
            agc_latency + limiter_latency,
            Self::latency_frames_for(sample_rate, self.active_lookahead_ms)
        );
        self.agc_delay.allocate(agc_latency, channels);
        self.multiband_delay.allocate(0, channels);
        self.limiter_delay.allocate(limiter_latency, channels);
        self.chain_delay
            .allocate(agc_latency + limiter_latency, channels);

        self.stage_dry = vec![0.0; CHUNK * channels];
        self.chain_dry = vec![0.0; CHUNK * channels];
        self.peaks = vec![0.0; CHUNK];

        self.input_detector
            .configure(sample_rate, channels, ChannelLayout::Unknown);
        self.output_detector
            .configure(sample_rate, channels, ChannelLayout::Unknown);
        self.integrated = LoudnessMeter::new(&LoudnessConfig {
            sample_rate,
            channels,
            layout: ChannelLayout::Unknown,
            true_peak: false,
        })
        .ok();
        self.input_tp.configure(OversampleFactor::X4, channels);
        self.output_tp.configure(OversampleFactor::X4, channels);

        let fs = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate as f64
        } else {
            48_000.0
        };
        self.input_gain.set_rate(fs);
        let fade = (FADE_MS * 0.001 * fs).round() as usize;
        for f in [
            &mut self.agc_fade,
            &mut self.multiband_fade,
            &mut self.limiter_fade,
            &mut self.chain_fade,
        ] {
            f.set_len(fade);
        }
        self.reset();
    }

    fn process(&mut self, samples: &mut [f32]) {
        let ch = self.channels;
        if ch == 0 || self.stage_dry.is_empty() {
            return;
        }
        let usable = samples.len() / ch * ch;
        if usable == 0 {
            return;
        }
        self.fresh = false;
        for chunk in samples[..usable].chunks_mut(CHUNK * ch) {
            self.process_chunk(chunk);
        }
    }

    fn reset(&mut self) {
        if self.channels == 0 {
            return;
        }
        self.agc.reset();
        self.multiband.reset();
        self.limiter.reset();
        for d in [
            &mut self.agc_delay,
            &mut self.multiband_delay,
            &mut self.limiter_delay,
            &mut self.chain_delay,
        ] {
            d.clear();
        }
        self.input_gain.snap();
        for f in [
            &mut self.agc_fade,
            &mut self.multiband_fade,
            &mut self.limiter_fade,
            &mut self.chain_fade,
        ] {
            f.snap();
        }
        self.input_detector.reset();
        self.output_detector.reset();
        if let Some(m) = &mut self.integrated {
            m.reset();
        }
        self.input_tp.reset();
        self.output_tp.reset();
        self.input_peak = 0.0;
        self.output_peak = 0.0;
        self.limiter_gr_max = 0.0;
        self.fresh = true;
    }

    fn latency_frames(&self) -> usize {
        Self::latency_frames_for(self.sample_rate, self.active_lookahead_ms)
    }
}
