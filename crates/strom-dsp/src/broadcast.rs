//! Broadcast final processor chain:
//! input gain → spike guard → wideband AGC → multiband compressor (per-band
//! downward expander, bass clipper on band 0) → final clipper → true-peak
//! limiter, with a clip budget governor around the clipper and the limiter.
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
//!         └→ gain → [spike guard] → [AGC] → [multiband] → [clipper] → [limiter] ┘
//!                     each [stage] = fade(stage(x), delay_stage(x))
//! ```
//!
//! - **Input gain**: glides in dB through two cascaded one-poles of 5 ms
//!   each (critically damped, starts with zero slope); a settled 0 dB is
//!   skipped (bit-exact).
//! - **Spike guard**: [`crate::spike_guard::SpikeGuard`] removes short,
//!   isolated peaks (clicks) before the AGC and the multiband, so they
//!   neither pump nor count toward the measured loudness. Outside events it
//!   is an exact delay ([`crate::spike_guard`] gives its latency).
//! - **AGC**: [`crate::agc::Agc`] (gated, rate-limited loudness rider; no
//!   latency), with a slow output-referenced trim on its target (below).
//! - **Multiband**: [`crate::multiband::MultibandCompressor`] (LR4
//!   crossover, compressor per band, an all-pass at ratio 1), with a
//!   downward expander per band (`band_expanders`, `expander_hold_ms`;
//!   knee [`EXPANDER_KNEE_DB`], open [`EXPANDER_OPEN_MS`], close
//!   [`EXPANDER_CLOSE_MS`]) and the bass clipper on band 0
//!   ([`crate::bassclip`]). The bass clipper's lookahead
//!   (`bass_clipper_lookahead_ms`, structural: [`STANDARD_BASS_LOOKAHEAD_MS`]
//!   or [`LOW_LATENCY_BASS_LOOKAHEAD_MS`]) is the stage's latency (constant,
//!   also with the clipper off). Its IM-feedback input
//!   ([`crate::multiband::MultibandCompressor::set_upper_clip_activity`])
//!   stays at "no activity" until a clipper after the multiband feeds it.
//! - **Final clipper**: [`crate::clipper::FinalClipper`] (oversampled
//!   soft clipping of shaped clippings; `clipper_latency` sets 69 or 48
//!   frames), threshold [`CLIPPER_MARGIN_DB`] under the ceiling, low cut at
//!   the lowest crossover, band limit `clipper_bandwidth_hz`.
//! - **Limiter**: [`crate::truepeak::TruePeakLimiter`] with ceiling
//!   `true_peak_ceiling_dbtp` (dBTP, BS.1770 Annex 2 detector; lookahead +
//!   32 frames of detector delay). It stays the guaranteed ceiling.
//! - **Clip budget governor**: [`crate::governor::ClipBudgetGovernor`],
//!   below.
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
//! The latency is `spike guard + agc (0) + multiband + clipper + limiter`,
//! i.e. `24 + round(spike lookahead · fs / 1000) + round(bass lookahead ·
//! fs / 1000) + C + round(limiter lookahead · fs / 1000) + 32` frames with
//! `C` = 69 (standard clipper latency) or 48 (low): 48 + 48 + 69 + 72 + 32
//! = 269 frames (5.60 ms) at 48 kHz with the defaults (0.5 ms, 1 ms,
//! standard, 1.5 ms). It depends only on the sample rate and the
//! structural fields ([`BroadcastProcessor::latency_frames_for_params`]).
//! `num_bands`, `spike_guard_lookahead_ms`, `bass_clipper_lookahead_ms`,
//! `clipper_latency` and `limiter_lookahead_ms` are structural: they take
//! effect at the next [`Processor::configure`].
//!
//! # Output-referenced trim
//!
//! The AGC measures its *input*. The multiband and the limiter after it
//! take level away, more so on peaky programme (dense hits, clicks), and
//! the AGC alone never makes that up. A slow outer loop does: an integral
//! controller (Åström & Hägglund, *PID Controllers*, ch. 3) on the
//! loudness at the chain output, whose output `trim` is added to the AGC
//! target ([`crate::agc::Agc::set_target_offset_db`]).
//!
//! - **Measurement:** the output [`LoudnessDetector`]'s 10 ms sub-blocks,
//!   gated exactly like the AGC's own measurement (400 ms blocks below
//!   `agc_gate_lufs` are left out) and averaged over 3 s.
//! - **Law:** every 10 ms, `trim += (target − measured_out) · 10 ms / τ`,
//!   τ = 8 s, with the step limited to `min(0.5 dB/s, ½ · slower AGC rate)`
//!   and `|trim| ≤ 6 dB`.
//! - **Freeze** (the trim keeps its value) when: the newest 100 ms of the
//!   output is below the gate, or the window holds less than 400 ms of
//!   gated output (pauses, silence: the same rule as the AGC); the AGC or
//!   the chain is bypassed or fading; the AGC gain has not stayed within
//!   1 dB of its target for a whole window (the window would still hold
//!   output from an AGC ramp); or the AGC target is at a gain bound in the
//!   direction the error pushes (anti-windup by conditional integration).
//! - **Bounds and rates:** the trim only moves the AGC target, so the AGC's
//!   gain bounds and rate limits apply to the total. The trim is reset to
//!   0 by `configure`/`reset`.
//!
//! **Stability.** Let `s ∈ [0, 1]` be the incremental gain of the stages
//! after the AGC (dB out per dB in: 1 when they are idle, `1/ratio` while
//! compressing, near 0 when the limiter holds every peak). The AGC follows
//! the trim without lag, since the trim moves at most half as fast as the
//! AGC may. The loop is then an integrator `s/(τ·p)` with the measurement
//! window, a moving average of `W` = 3 s: a linear-phase filter with a
//! group delay of `T = W/2 = 1.5 s` and a gain ≤ 1. The crossover is at
//! `ω_c = s/τ ≤ 0.125 rad/s`, so the phase margin is
//! `90° − ω_c·T ≥ 79°`, and the gain at the −180° frequency
//! (`π/(2T)` ≈ 1.05 rad/s) is at most `1/(τ·1.05)`, a gain margin of at
//! least 18 dB. `s·T/τ ≤ 0.19 < 1/e`, the bound below which an integrator
//! with a pure delay `T` settles without oscillating. The closed-loop time
//! constant is `τ/s` ≥ 8 s, against detector windows of 0.1, 0.4 and 3 s.
//! The rate and range limits only slow the loop. When the later stages
//! cannot deliver more loudness (`s` near 0) the trim stops at its range
//! or at the AGC bound; it does not oscillate.
//!
//! **Pauses.** During a pause the trim is frozen (a holding AGC counts as
//! settled: its gain does not move). After it, the gated output window
//! leaves the pause out, the trim waits for 400 ms of returning
//! programme, and if the AGC gain has to move by more than 1 dB, for a
//! whole window after it settles. The trim never runs up on silence, so
//! there is no overshoot when the programme returns.
//!
//! # Clip budget governor
//!
//! Every output sub-block the chain hands the governor what the clipper and
//! the limiter did (see [`crate::governor`] for the measures, the budget
//! mapping and the law). Its outputs act here:
//!
//! - **Trim hold:** while the governor holds, the trim takes no positive
//!   steps.
//! - **Cut:** the AGC target offset is `trim − cut`.
//! - **Density:** every band's threshold is lowered by `density` and its
//!   makeup raised by `density·(1 − 1/ratio)` before the parameters reach
//!   the multiband (passed on in steps of at least 0.01 dB; the band
//!   compressors glide thresholds and makeup).
//!
//! The governor is frozen while the AGC, the clipper or the chain is
//! bypassed or fading, and in pauses and below the gate (the trim's rule).
//! Switching it off drops density and cut at once; the AGC's rates and the
//! compressors' glides make that smooth.
//!
//! # Meters
//!
//! | Value | Source | Cost |
//! |---|---|---|
//! | input momentary / short-term | a BS.1770 [`LoudnessDetector`] on the raw input (**before the input gain**) | ≈ 1.3 µs per 480 stereo frames, O(1) read |
//! | output momentary / short-term | a second [`LoudnessDetector`] on the final output | ≈ 1.3 µs per 480 stereo frames, O(1) read |
//! | output integrated | [`LoudnessMeter`] (histogram mode, no true peak) on the final output | ≈ 7 µs per 480 stereo frames, < 1 µs read |
//! | input / output true peak | 4× [`Oversampler`] peaks (max over sub-samples and channels) of the raw input (before the input gain) / of the final output, held until [`BroadcastProcessor::take_meters`] | upsampling per block |
//! | spike events | events of the spike guard since the last read | O(1) |
//! | AGC gain | [`crate::agc::Agc::gain_db`] (the trim included: it is the applied gain) | O(1) |
//! | band gain reduction | [`crate::multiband::MultibandCompressor::gain_reduction_db`] at read time | O(1) |
//! | band expander gain reduction | [`crate::multiband::MultibandCompressor::expander_reduction_db`] at read time | O(1) |
//! | bass-clip gain reduction | largest since the last read ([`crate::multiband::MultibandCompressor::take_bass_clip_gain_reduction_db`]) | O(1) |
//! | limiter gain reduction | max of the value at the end of each internal chunk since the last read | O(1) |
//! | clip depth | the clipper's removed / input energy since the last read | O(1) |
//! | loudness deficit, governor state | [`crate::governor::ClipBudgetGovernor`] | O(1) |
//!
//! The input meters show what actually comes in: they are measured before
//! the input gain, so `input_gain_db` does not change them.
//!
//! The AGC gain and the gain reductions read 0 while their stage (or the
//! chain) is switched off, i.e. they show what is being applied.
//! [`BroadcastProcessor::take_meters`] is O(1) and allocation-free: call it
//! once per buffer if needed.

use crate::agc::{Agc, AgcParams, HOLD_BLOCK_SUB_BLOCKS, MIN_ACTIVE_MS};
use crate::bassclip::{BassClipperParams, BASS_CLIP_THRESHOLD_RANGE_DB};
use crate::clipper::{ClipStats, ClipperLatency, FinalClipper, FinalClipperParams};
use crate::dynamics::expander::{
    EXPANDER_FLOOR_RANGE_DB, EXPANDER_HOLD_RANGE_MS, EXPANDER_RATIO_RANGE,
    EXPANDER_THRESHOLD_RANGE_DB,
};
use crate::dynamics::ExpanderParams;
use crate::dynamics::OnePole;
use crate::governor::{ClipBudgetGovernor, GovernorInputs, GovernorState};
use crate::meter::{
    lufs_from_mean_square, ChannelLayout, History, LoudnessConfig, LoudnessDetector, LoudnessMeter,
    WindowSum, SUB_BLOCK_MS,
};
use crate::multiband::{
    MultibandBandParams, MultibandCompressor, MultibandParams, MAX_BANDS, MAX_BASS_LOOKAHEAD_MS,
};
use crate::oversample::{OversampleFactor, Oversampler};
use crate::spike_guard::{self, SpikeGuard, SpikeGuardParams};
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

/// Output trim: window of the gated output loudness (ms), the same 3 s as
/// the AGC detector.
const TRIM_WINDOW_MS: f64 = 3000.0;
/// Output trim: integral time constant (s). The correction moves by
/// `error / TRIM_TIME_CONSTANT_S` dB per second. See "Output-referenced
/// trim" in the module docs for the stability bound.
const TRIM_TIME_CONSTANT_S: f64 = 8.0;
/// Output trim: largest slew (dB/s), and at most this fraction of the
/// slower AGC rate, so the AGC always keeps up with the trim.
const TRIM_MAX_RATE_DB_S: f64 = 0.5;
const TRIM_RATE_FRACTION: f64 = 0.5;
/// Output trim: range (dB, ±). The AGC's own gain bounds apply on top.
const TRIM_MAX_DB: f64 = 6.0;
/// Output trim: the AGC counts as settled while its applied gain is within
/// this distance of its target gain (dB).
const TRIM_SETTLED_DB: f64 = 1.0;

/// Final clipper: threshold below the true-peak ceiling (dB). The clipper
/// holds sample peaks within +0.3 dB of its threshold on band-limited
/// material, so the limiter after it normally catches only inter-sample
/// residue and bass peaks (the clipper leaves bass to the limiter).
pub const CLIPPER_MARGIN_DB: f32 = 0.5;
/// Default limiter lookahead (ms). With the final clipper taking the peaks,
/// the limiter handles only residue, so 1.5 ms is enough (see
/// `docs/elements/broadcastprocessor.md`); 0.75 ms for low latency.
pub const DEFAULT_LOOKAHEAD_MS: f32 = 1.5;
/// Limiter lookahead (ms) of the low-latency configuration.
pub const LOW_LATENCY_LOOKAHEAD_MS: f32 = 0.75;

/// Final clipper: knee width (dB) of its curve.
pub const CLIPPER_KNEE_DB: f32 = 2.0;
/// Range of `clipper_bandwidth_hz` (Hz).
pub const CLIPPER_BANDWIDTH_RANGE_HZ: (f32, f32) = crate::clipper::BANDWIDTH_RANGE_HZ;
/// Range of `clip_budget_db` (dB).
pub const CLIP_BUDGET_RANGE_DB: (f32, f32) = crate::governor::BUDGET_RANGE_DB;
/// Largest change of the limiter's ceiling per internal chunk (dB). A
/// ceiling jump becomes steps of at most this size, each ramped over the
/// limiter's lookahead, so a short lookahead does not turn a 6 dB jump into
/// an audible step (6 dB takes about 64 ms at 48 kHz).
const CEILING_STEP_DB: f32 = 1.0;
/// Smallest density change (dB) passed on to the multiband thresholds.
const DENSITY_STEP_DB: f64 = 0.01;

/// Value reported for silence / no measurement yet (never NaN or −∞).
pub const SILENCE_DB: f64 = -200.0;

/// Number of crossover frequencies (`MAX_BANDS − 1`).
pub const MAX_CROSSOVERS: usize = MAX_BANDS - 1;

/// Bass clipper lookahead (ms) of the standard configuration: the
/// multiband stage's latency. The analytic envelope reaches a bass onset
/// later than any lookahead of a few ms could cover, so a longer one buys
/// little: on a 60 Hz onset 12 dB over the threshold 3 ms let 11.3 dB
/// through, 1 ms 12.0 dB, none 12.0 dB, and steady bass stays clean (2nd
/// and 3rd harmonic under −125 dBc) at every setting. What gets through
/// goes on to the final clipper (which leaves bass alone) and the limiter
/// (see `tests/broadcast_latency.rs` for what that costs there).
pub const STANDARD_BASS_LOOKAHEAD_MS: f32 = 1.0;
/// Bass clipper lookahead (ms) of the low-latency configuration.
pub const LOW_LATENCY_BASS_LOOKAHEAD_MS: f32 = 0.0;
/// Soft-knee width of every band's expander (dB).
pub const EXPANDER_KNEE_DB: f32 = 6.0;
/// Open time of every band's expander (ms): faster than a consonant onset.
pub const EXPANDER_OPEN_MS: f32 = 3.0;
/// Close time of every band's expander (ms): the slower release in the
/// expansion region, a natural decay rather than a cut.
pub const EXPANDER_CLOSE_MS: f32 = 150.0;

/// The downward expander of one band (see [`crate::dynamics::expander`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandExpanderParams {
    /// Expander threshold (dB, −80 … 0), against the band's sine-calibrated
    /// level like the band threshold.
    pub threshold_db: f32,
    /// Ratio (1 … 10; 1 = off).
    pub ratio: f32,
    /// Largest attenuation (dB, 0 … 40).
    pub floor_db: f32,
}

impl Default for BandExpanderParams {
    /// Off (ratio 1), threshold −45 dB, floor 6 dB.
    fn default() -> Self {
        Self {
            threshold_db: -45.0,
            ratio: 1.0,
            floor_db: 6.0,
        }
    }
}

impl BandExpanderParams {
    /// Copy clamped to the ranges (NaN becomes the default).
    pub fn clamped(&self) -> Self {
        let d = Self::default();
        Self {
            threshold_db: clamp_or(
                self.threshold_db,
                EXPANDER_THRESHOLD_RANGE_DB.0,
                EXPANDER_THRESHOLD_RANGE_DB.1,
                d.threshold_db,
            ),
            ratio: clamp_or(
                self.ratio,
                EXPANDER_RATIO_RANGE.0,
                EXPANDER_RATIO_RANGE.1,
                d.ratio,
            ),
            floor_db: clamp_or(
                self.floor_db,
                EXPANDER_FLOOR_RANGE_DB.0,
                EXPANDER_FLOOR_RANGE_DB.1,
                d.floor_db,
            ),
        }
    }
}

/// All parameters of the chain (the element's properties, minus `preset` and
/// `meter-interval`, which belong to the element).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BroadcastParams {
    /// Unity gain with the same delay when false (true bypass).
    pub enabled: bool,
    /// Input gain (dB, −24 … 24).
    pub input_gain_db: f32,
    /// Spike guard stage active.
    pub spike_guard_enabled: bool,
    /// Spike guard cap above the programme envelope (dB, 6 … 30).
    pub spike_threshold_db: f32,
    /// Spike guard lookahead (ms, 0 … 2; standard 0.5, low latency 0).
    /// Structural: part of the latency, applied at the next configure.
    pub spike_guard_lookahead_ms: f32,
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
    /// Final clipper stage active.
    pub clipper_enabled: bool,
    /// Band limit of the clipper's clippings (Hz, 8 000 … 22 000; at most
    /// 0.46·fs).
    pub clipper_bandwidth_hz: f32,
    /// Clip budget (dB, 0 … 6): how much clipping and limiting the governor
    /// allows (see [`crate::governor`]).
    pub clip_budget_db: f32,
    /// Clip budget governor active.
    pub governor_enabled: bool,
    /// Limiter stage active.
    pub limiter_enabled: bool,
    /// Limiter release (ms, 1 … 2000).
    pub limiter_release_ms: f32,
    /// Limiter lookahead (ms, 0.5 … 20). Structural (READY-only on the element).
    pub limiter_lookahead_ms: f32,
    /// Per-band downward expander; band 0 is the lowest.
    pub band_expanders: [BandExpanderParams; MAX_BANDS],
    /// Hold of every band's expander before it closes (ms, 0 … 1000).
    pub expander_hold_ms: f32,
    /// Bass clipper on band 0 active (crossfaded; its latency stays).
    pub bass_clipper_enabled: bool,
    /// Bass clip threshold (dB, −40 … 0), against band 0's analytic envelope
    /// (a steady sine reads its peak).
    pub bass_clip_threshold_db: f32,
    /// Bass clipper lookahead (ms, 0 … 10; standard 1, low latency 0).
    /// Structural: part of the latency, applied at the next configure.
    pub bass_clipper_lookahead_ms: f32,
    /// Latency setting of the final clipper. Structural (takes effect at
    /// the next `configure`).
    pub clipper_latency: ClipperLatency,
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
            spike_guard_enabled: true,
            spike_threshold_db: spike_guard::DEFAULT_THRESHOLD_DB,
            spike_guard_lookahead_ms: spike_guard::STANDARD_LOOKAHEAD_MS,
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
            clipper_enabled: true,
            clipper_bandwidth_hz: 20_000.0,
            clip_budget_db: 2.0,
            governor_enabled: true,
            limiter_enabled: true,
            limiter_release_ms: 100.0,
            limiter_lookahead_ms: DEFAULT_LOOKAHEAD_MS,
            band_expanders: [BandExpanderParams::default(); MAX_BANDS],
            expander_hold_ms: 100.0,
            bass_clipper_enabled: false,
            bass_clip_threshold_db: -12.0,
            bass_clipper_lookahead_ms: STANDARD_BASS_LOOKAHEAD_MS,
            clipper_latency: ClipperLatency::Standard,
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
    /// Spike guard events since the last [`BroadcastProcessor::take_meters`]
    /// (0 while the stage or the chain is off).
    pub spike_events: u64,
    /// AGC gain (dB, signed: + = boost).
    pub agc_gain_db: f64,
    /// Gain reduction per band (positive dB); only the first `num_bands` are meaningful.
    pub band_gain_reduction_db: [f64; MAX_BANDS],
    /// Limiter gain reduction (positive dB).
    pub limiter_gain_reduction_db: f64,
    /// Expander gain reduction per band (positive dB); only the first
    /// `num_bands` are meaningful.
    pub band_expander_gain_reduction_db: [f64; MAX_BANDS],
    /// Bass-clipper gain reduction (positive dB), largest since the last
    /// [`BroadcastProcessor::take_meters`].
    pub bass_clip_gain_reduction_db: f64,
    /// Final clipper clip depth since the last read (dB): energy removed
    /// relative to the clipper's input; [`SILENCE_DB`] when nothing was
    /// removed.
    pub clip_depth_db: f64,
    /// Loudness deficit the governor accepts (LU, ≥ 0).
    pub loudness_deficit_lu: f64,
    /// What the governor is doing.
    pub governor_state: GovernorState,
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
            spike_events: 0,
            agc_gain_db: 0.0,
            band_gain_reduction_db: [0.0; MAX_BANDS],
            limiter_gain_reduction_db: 0.0,
            band_expander_gain_reduction_db: [0.0; MAX_BANDS],
            bass_clip_gain_reduction_db: 0.0,
            clip_depth_db: SILENCE_DB,
            loudness_deficit_lu: 0.0,
            governor_state: GovernorState::Off,
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
            spike_guard_enabled: self.spike_guard_enabled,
            spike_threshold_db: SpikeGuardParams {
                enabled: true,
                threshold_db: self.spike_threshold_db,
            }
            .clamped()
            .threshold_db,
            spike_guard_lookahead_ms: spike_guard::clamp_lookahead_ms(
                self.spike_guard_lookahead_ms,
            ),
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
            clipper_enabled: self.clipper_enabled,
            clipper_bandwidth_hz: clamp_or(
                self.clipper_bandwidth_hz,
                CLIPPER_BANDWIDTH_RANGE_HZ.0,
                CLIPPER_BANDWIDTH_RANGE_HZ.1,
                d.clipper_bandwidth_hz,
            ),
            clip_budget_db: clamp_or(
                self.clip_budget_db,
                CLIP_BUDGET_RANGE_DB.0,
                CLIP_BUDGET_RANGE_DB.1,
                d.clip_budget_db,
            ),
            governor_enabled: self.governor_enabled,
            limiter_enabled: self.limiter_enabled,
            limiter_release_ms: clamp_or(
                self.limiter_release_ms,
                1.0,
                2000.0,
                d.limiter_release_ms,
            ),
            limiter_lookahead_ms: clamp_lookahead(self.limiter_lookahead_ms),
            band_expanders: core::array::from_fn(|k| self.band_expanders[k].clamped()),
            expander_hold_ms: clamp_or(
                self.expander_hold_ms,
                EXPANDER_HOLD_RANGE_MS.0,
                EXPANDER_HOLD_RANGE_MS.1,
                d.expander_hold_ms,
            ),
            bass_clipper_enabled: self.bass_clipper_enabled,
            bass_clip_threshold_db: clamp_or(
                self.bass_clip_threshold_db,
                BASS_CLIP_THRESHOLD_RANGE_DB.0,
                BASS_CLIP_THRESHOLD_RANGE_DB.1,
                d.bass_clip_threshold_db,
            ),
            bass_clipper_lookahead_ms: clamp_or(
                self.bass_clipper_lookahead_ms,
                0.0,
                MAX_BASS_LOOKAHEAD_MS,
                d.bass_clipper_lookahead_ms,
            ),
            clipper_latency: self.clipper_latency,
        }
    }
}

/// The lookahead range of the element (ms, 0.5 … 20; NaN/Inf → the
/// default).
fn clamp_lookahead(ms: f32) -> f32 {
    if ms.is_finite() {
        ms.clamp(0.5, 20.0)
    } else {
        DEFAULT_LOOKAHEAD_MS
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

/// Spike guard stage: [`SpikeGuard`]. Its own `enabled` stays on: the chain
/// does the bypass.
#[derive(Debug, Default)]
struct SpikeStage {
    inner: SpikeGuard,
}

impl SpikeStage {
    /// Latency of the stage (frames).
    fn latency_frames_for(sample_rate: f32, lookahead_ms: f32) -> usize {
        spike_guard::latency_frames_for(sample_rate, lookahead_ms)
    }

    fn configure(&mut self, sample_rate: f32, channels: usize, lookahead_ms: f32) {
        self.inner.set_lookahead_ms(lookahead_ms);
        self.inner.configure(sample_rate, channels);
    }

    fn set_params(&mut self, p: &BroadcastParams) {
        self.inner.set_params(&SpikeGuardParams {
            enabled: true,
            threshold_db: p.spike_threshold_db,
        });
    }

    fn process(&mut self, x: &mut [f32]) {
        self.inner.process(x);
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

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
        self.inner.set_target_offset_db(0.0);
    }

    fn gain_db(&self) -> f64 {
        self.inner.gain_db()
    }
}

/// Output-referenced trim of the AGC target: a slow, gated integral loop on
/// the loudness at the chain output. See "Output-referenced trim" in the
/// module docs.
#[derive(Debug, Default)]
struct OutputTrim {
    /// Energy of each output sub-block if it was active, else 0.
    active_energy: History,
    /// 1 for each active output sub-block, else 0.
    active_flag: History,
    energy_sum: WindowSum,
    count_sum: WindowSum,
    /// Consecutive sub-blocks with the AGC settled and in use.
    settled: usize,
    /// Window length (sub-blocks).
    window: usize,
    /// The correction (dB), handed to the AGC as its target offset.
    trim_db: f64,
    /// The newest sub-block passed the gating (not a pause, enough active
    /// output in the window).
    gated: bool,
    /// Gated output loudness over the window (LUFS), once measurable.
    measured: Option<f64>,
}

/// Everything [`OutputTrim::on_sub_block`] needs from the chain.
struct TrimInputs {
    /// The AGC's output reaches the chain output (AGC and chain fully wet).
    agc_in_use: bool,
    target_lufs: f64,
    gate_lufs: f64,
    max_gain_db: f64,
    max_cut_db: f64,
    /// Slower of the AGC's up and down rates (dB/s).
    slower_rate_db_s: f64,
    /// The governor's cut of the AGC target (dB, ≥ 0).
    cut_db: f64,
    /// The governor forbids adding gain.
    hold_up: bool,
}

impl OutputTrim {
    fn allocate(&mut self) {
        self.window = (TRIM_WINDOW_MS / SUB_BLOCK_MS).round() as usize;
        self.active_energy.allocate(self.window + 1);
        self.active_flag.allocate(self.window + 1);
        self.reset();
    }

    fn reset(&mut self) {
        self.active_energy.clear();
        self.active_flag.clear();
        self.energy_sum.set_len(self.window, &self.active_energy);
        self.count_sum.set_len(self.window, &self.active_flag);
        self.settled = 0;
        self.trim_db = 0.0;
        self.gated = false;
        self.measured = None;
    }

    /// Record the output sub-block `out` just completed and integrate.
    /// Returns the new trim (dB).
    fn on_sub_block(&mut self, out: &LoudnessDetector, agc: &Agc, p: &TrimInputs) -> f64 {
        // Gated output measurement, as in the AGC: 400 ms blocks below the
        // gate are left out, and the newest 100 ms below it freezes.
        let active = out.momentary_lufs() >= p.gate_lufs;
        let present =
            lufs_from_mean_square(out.recent_mean_square(HOLD_BLOCK_SUB_BLOCKS)) >= p.gate_lufs;
        let e = out.last_sub_block_mean_square();
        self.active_energy.push(if active { e } else { 0.0 });
        self.active_flag.push(if active { 1.0 } else { 0.0 });
        self.energy_sum.pushed(&self.active_energy);
        self.count_sum.pushed(&self.active_flag);
        let count = self.count_sum.sum().round();

        // The window must hold only output made with a settled AGC gain (a
        // holding AGC is settled: its gain does not move).
        let agc_settled =
            p.agc_in_use && (agc.target_gain_db() - agc.gain_db()).abs() <= TRIM_SETTLED_DB;
        self.settled = if agc_settled { self.settled + 1 } else { 0 };

        let min_active = MIN_ACTIVE_MS / SUB_BLOCK_MS;
        self.gated = present && count >= min_active;
        if count >= min_active {
            self.measured = Some(lufs_from_mean_square(self.energy_sum.sum() / count));
        }
        if !self.gated || self.settled < self.window {
            return self.trim_db; // freeze
        }
        let measured = lufs_from_mean_square(self.energy_sum.sum() / count);
        let error = p.target_lufs - measured;
        // Anti-windup: stop integrating into an AGC gain bound.
        let want = p.target_lufs + self.trim_db - p.cut_db - agc.measured_lufs();
        if (error > 0.0 && want >= p.max_gain_db) || (error < 0.0 && want <= -p.max_cut_db) {
            return self.trim_db;
        }
        let dt = SUB_BLOCK_MS * 0.001;
        let max_step = TRIM_MAX_RATE_DB_S.min(TRIM_RATE_FRACTION * p.slower_rate_db_s) * dt;
        let mut step = (error * dt / TRIM_TIME_CONSTANT_S).clamp(-max_step, max_step);
        if p.hold_up {
            step = step.min(0.0);
        }
        if step.is_finite() {
            self.trim_db = (self.trim_db + step).clamp(-TRIM_MAX_DB, TRIM_MAX_DB);
        }
        self.trim_db
    }
}

/// Final clipper stage: [`FinalClipper`] with the configured
/// [`ClipperLatency`], threshold [`CLIPPER_MARGIN_DB`] under the true-peak
/// ceiling, low cut at the lowest crossover. Latency 69 frames (standard)
/// or 48 (low) at every rate.
#[derive(Debug, Default)]
struct ClipperStage {
    inner: FinalClipper,
}

impl ClipperStage {
    fn latency_frames(latency: ClipperLatency) -> usize {
        latency.latency_frames()
    }

    fn configure(&mut self, sample_rate: f32, channels: usize, latency: ClipperLatency) {
        self.inner.set_latency(latency);
        self.inner.configure(sample_rate, channels);
    }

    fn set_params(&mut self, p: &BroadcastParams) {
        self.inner.set_params(&FinalClipperParams {
            threshold_db: p.true_peak_ceiling_dbtp - CLIPPER_MARGIN_DB,
            knee_db: CLIPPER_KNEE_DB,
            bandwidth_hz: p.clipper_bandwidth_hz,
            low_cut_hz: p.crossover_hz[0],
            ..FinalClipperParams::default()
        });
    }

    fn process(&mut self, x: &mut [f32]) {
        self.inner.process(x);
    }

    fn reset(&mut self) {
        self.inner.reset();
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

    fn set_params(&mut self, p: &BroadcastParams, ceiling_dbtp: f32, lookahead_ms: f32) {
        self.inner.set_params(&TruePeakLimiterParams {
            enabled: true,
            ceiling_dbtp,
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
    active_spike_lookahead_ms: f32,
    active_bass_lookahead_ms: f32,
    active_clipper_latency: ClipperLatency,
    /// Limiter ceiling in force (dBTP), gliding toward the parameter.
    ceiling_db: f32,
    active_bands: usize,
    /// True from configure/reset until the first processed frame.
    fresh: bool,

    input_gain: GainGlide,
    spike: SpikeStage,
    agc: AgcStage,
    trim: OutputTrim,
    multiband: MultibandCompressor,
    clipper: ClipperStage,
    governor: ClipBudgetGovernor,
    limiter: LimiterStage,
    /// Density applied to the multiband thresholds (dB).
    applied_density_db: f64,
    /// Clipper stats and limiter gain reduction (dB · frames) since the
    /// last output sub-block, and since the last `take_meters`.
    sub_clip: ClipStats,
    sub_gr: f64,
    sub_frames: usize,
    meter_clip: ClipStats,

    spike_fade: Fade,
    agc_fade: Fade,
    multiband_fade: Fade,
    clipper_fade: Fade,
    limiter_fade: Fade,
    chain_fade: Fade,
    spike_delay: FixedDelay,
    agc_delay: FixedDelay,
    multiband_delay: FixedDelay,
    clipper_delay: FixedDelay,
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
    /// Spike guard events since the last `take_meters`.
    spike_events: u64,
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
            active_spike_lookahead_ms: params.spike_guard_lookahead_ms,
            active_bass_lookahead_ms: params.bass_clipper_lookahead_ms,
            active_clipper_latency: params.clipper_latency,
            ceiling_db: params.true_peak_ceiling_dbtp,
            active_bands: params.num_bands,
            fresh: true,
            input_gain: GainGlide::default(),
            spike: SpikeStage::default(),
            agc: AgcStage::default(),
            trim: OutputTrim::default(),
            multiband: MultibandCompressor::new(),
            clipper: ClipperStage::default(),
            governor: ClipBudgetGovernor::default(),
            limiter: LimiterStage::default(),
            spike_fade: Fade::default(),
            applied_density_db: 0.0,
            sub_clip: ClipStats::default(),
            sub_gr: 0.0,
            sub_frames: 0,
            meter_clip: ClipStats::default(),
            agc_fade: Fade::default(),
            multiband_fade: Fade::default(),
            clipper_fade: Fade::default(),
            limiter_fade: Fade::default(),
            chain_fade: Fade::default(),
            spike_delay: FixedDelay::default(),
            agc_delay: FixedDelay::default(),
            multiband_delay: FixedDelay::default(),
            clipper_delay: FixedDelay::default(),
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
            spike_events: 0,
        };
        s.set_params(&params);
        s
    }

    /// Update parameters (clamped). Never allocates. Structural fields
    /// (`num_bands`, `spike_guard_lookahead_ms`, `limiter_lookahead_ms`)
    /// take effect at the next `configure`.
    pub fn set_params(&mut self, params: &BroadcastParams) {
        let p = params.clamped();
        self.params = p;
        if self.channels == 0 {
            // Unconfigured: the structural values follow until `configure`.
            self.active_lookahead_ms = p.limiter_lookahead_ms;
            self.active_spike_lookahead_ms = p.spike_guard_lookahead_ms;
            self.active_bass_lookahead_ms = p.bass_clipper_lookahead_ms;
            self.active_clipper_latency = p.clipper_latency;
            self.active_bands = p.num_bands;
        }
        let snap = self.fresh;
        self.input_gain.retarget(p.input_gain_db as f64);
        if snap {
            self.input_gain.snap();
        }
        self.spike.set_params(&p);
        self.agc.set_params(&p);
        self.governor
            .set_params(p.governor_enabled, p.clip_budget_db);
        self.applied_density_db = self.governor.density_db();
        self.apply_multiband_params();
        self.multiband
            .set_expanders(&core::array::from_fn(|k| ExpanderParams {
                threshold_db: p.band_expanders[k].threshold_db,
                ratio: p.band_expanders[k].ratio,
                floor_db: p.band_expanders[k].floor_db,
                knee_db: EXPANDER_KNEE_DB,
                open_ms: EXPANDER_OPEN_MS,
                close_ms: EXPANDER_CLOSE_MS,
                hold_ms: p.expander_hold_ms,
            }));
        self.multiband.set_bass_clipper(&BassClipperParams {
            enabled: p.bass_clipper_enabled,
            threshold_db: p.bass_clip_threshold_db,
        });
        self.clipper.set_params(&p);
        if snap {
            self.ceiling_db = p.true_peak_ceiling_dbtp;
        }
        self.limiter
            .set_params(&p, self.ceiling_db, self.active_lookahead_ms);
        self.spike_fade.set(p.spike_guard_enabled, snap);
        self.agc_fade.set(p.agc_enabled, snap);
        self.multiband_fade.set(p.multiband_enabled, snap);
        self.clipper_fade.set(p.clipper_enabled, snap);
        self.limiter_fade.set(p.limiter_enabled, snap);
        self.chain_fade.set(p.enabled, snap);
    }

    /// Hand the band settings to the multiband, with the governor's
    /// density: every threshold lowered by `density` dB and the makeup
    /// raised by `density·(1 − 1/ratio)`, so passages well above the
    /// thresholds keep their level (see [`crate::governor`]).
    fn apply_multiband_params(&mut self) {
        let p = &self.params;
        let d = self.applied_density_db as f32;
        let mut bands = p.bands;
        if d > 0.0 {
            for b in &mut bands {
                b.threshold_db -= d;
                b.makeup_db += d * (1.0 - 1.0 / b.ratio.max(1.0));
            }
        }
        self.multiband.set_params(&MultibandParams {
            num_bands: self.active_bands,
            frequencies_hz: p.crossover_hz,
            drive_db: p.multiband_drive_db,
            bands,
            ..MultibandParams::default()
        });
    }

    /// The clip budget governor (for inspection: density, cut, state).
    pub fn governor(&self) -> &ClipBudgetGovernor {
        &self.governor
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
        let mut band_expander_gain_reduction_db = [0.0; MAX_BANDS];
        let mut bass_clip_gain_reduction_db = self.multiband.take_bass_clip_gain_reduction_db();
        if on && self.params.multiband_enabled {
            band_gain_reduction_db = self.multiband.gain_reduction_db();
            band_expander_gain_reduction_db = self.multiband.expander_reduction_db();
        } else {
            bass_clip_gain_reduction_db = 0.0;
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
            spike_events: if on && self.params.spike_guard_enabled {
                self.spike_events
            } else {
                0
            },
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
            band_expander_gain_reduction_db,
            bass_clip_gain_reduction_db,
            clip_depth_db: if on && self.params.clipper_enabled {
                or_silence(self.meter_clip.depth_db())
            } else {
                SILENCE_DB
            },
            loudness_deficit_lu: if on { self.governor.deficit_lu() } else { 0.0 },
            governor_state: self.governor.state(),
        };
        self.meter_clip = ClipStats::default();
        self.input_peak = 0.0;
        self.output_peak = 0.0;
        self.limiter_gr_max = self.limiter.gain_reduction_db();
        self.spike_events = 0;
        m
    }

    /// Restart the output integrated loudness measurement.
    pub fn reset_integrated(&mut self) {
        if let Some(m) = &mut self.integrated {
            m.reset();
        }
    }

    /// Total latency in frames for a configuration with the standard spike
    /// guard lookahead, the standard clipper latency and this limiter
    /// lookahead. Constant for a given rate and lookahead, independent of
    /// stage bypass. Usable before `configure`.
    pub fn latency_frames_for(sample_rate: f32, lookahead_ms: f32) -> usize {
        Self::latency_frames_for_params(
            sample_rate,
            &BroadcastParams {
                limiter_lookahead_ms: lookahead_ms,
                ..BroadcastParams::default()
            },
        )
    }

    /// Total latency in frames for a rate, a limiter lookahead and a
    /// clipper latency setting (standard spike guard lookahead).
    pub fn latency_frames_for_config(
        sample_rate: f32,
        lookahead_ms: f32,
        clipper: ClipperLatency,
    ) -> usize {
        Self::latency_frames_for_params(
            sample_rate,
            &BroadcastParams {
                limiter_lookahead_ms: lookahead_ms,
                clipper_latency: clipper,
                ..BroadcastParams::default()
            },
        )
    }

    /// Total latency in frames for the structural fields of `p` (spike
    /// guard lookahead, bass clipper lookahead, clipper latency, limiter
    /// lookahead). Usable before `configure`.
    pub fn latency_frames_for_params(sample_rate: f32, p: &BroadcastParams) -> usize {
        SpikeStage::latency_frames_for(
            sample_rate,
            spike_guard::clamp_lookahead_ms(p.spike_guard_lookahead_ms),
        ) + AgcStage::latency_frames_for(sample_rate)
            + MultibandCompressor::bass_lookahead_frames(p.bass_clipper_lookahead_ms, sample_rate)
            + ClipperStage::latency_frames(p.clipper_latency)
            + LimiterStage::latency_frames_for(sample_rate, clamp_lookahead(p.limiter_lookahead_ms))
    }

    /// Process one chunk (at most `CHUNK` frames).
    fn process_chunk(&mut self, chunk: &mut [f32]) {
        // The limiter's ceiling moves at most CEILING_STEP_DB per chunk.
        let target = self.params.true_peak_ceiling_dbtp;
        if self.ceiling_db != target {
            let d = (target - self.ceiling_db).clamp(-CEILING_STEP_DB, CEILING_STEP_DB);
            self.ceiling_db = if (target - self.ceiling_db).abs() <= CEILING_STEP_DB {
                target
            } else {
                self.ceiling_db + d
            };
            let p = self.params;
            self.limiter
                .set_params(&p, self.ceiling_db, self.active_lookahead_ms);
        }
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
        let spike = &mut self.spike;
        run_stage(
            &mut self.spike_fade,
            &mut self.spike_delay,
            dry,
            chunk,
            ch,
            |x| spike.process(x),
        );
        self.spike_events += self.spike.inner.take_events();
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
        let clip = &mut self.clipper;
        run_stage(
            &mut self.clipper_fade,
            &mut self.clipper_delay,
            dry,
            chunk,
            ch,
            |x| clip.process(x),
        );
        let clip_stats = self.clipper.inner.activity();
        self.sub_clip.accumulate(&clip_stats);
        self.meter_clip.accumulate(&clip_stats);
        let lim = &mut self.limiter;
        run_stage(
            &mut self.limiter_fade,
            &mut self.limiter_delay,
            dry,
            chunk,
            ch,
            |x| lim.process(x),
        );
        let gr = self.limiter.gain_reduction_db();
        self.limiter_gr_max = self.limiter_gr_max.max(gr);
        self.sub_gr += gr * frames as f64;
        self.sub_frames += frames;

        self.chain_fade.mix(chunk, &self.chain_dry[..len], ch);

        // Output loudness, sub-block by sub-block, feeding the trim.
        let trim_in = TrimInputs {
            agc_in_use: self.agc_fade.is_wet() && self.chain_fade.is_wet(),
            target_lufs: self.params.target_loudness_lufs as f64,
            gate_lufs: self.params.agc_gate_lufs as f64,
            max_gain_db: self.params.agc_max_gain_db as f64,
            max_cut_db: self.params.agc_max_cut_db as f64,
            slower_rate_db_s: self
                .params
                .agc_rate_up_db_s
                .min(self.params.agc_rate_down_db_s) as f64,
            cut_db: self.governor.cut_db(),
            hold_up: self.governor.holds_trim(),
        };
        let mut rest: &[f32] = chunk;
        while !rest.is_empty() {
            let n = (rest.len() / ch).min(self.output_detector.frames_to_sub_block_end());
            let (seg, tail) = rest.split_at(n * ch);
            if self.output_detector.process(seg) > 0 {
                let trim = self
                    .trim
                    .on_sub_block(&self.output_detector, &self.agc.inner, &trim_in);
                self.on_governor_sub_block(&trim_in);
                self.agc
                    .inner
                    .set_target_offset_db(trim - self.governor.cut_db());
            }
            rest = tail;
        }
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

impl BroadcastProcessor {
    /// One output sub-block for the governor: hand over what the clipper
    /// and the limiter did since the last one, then pass a changed density
    /// on to the multiband.
    fn on_governor_sub_block(&mut self, trim_in: &TrimInputs) {
        let in_use = trim_in.agc_in_use && self.clipper_fade.is_wet();
        let inputs = GovernorInputs {
            active: self.output_detector.momentary_lufs() >= trim_in.gate_lufs,
            frozen: !in_use || !self.trim.gated,
            clip: self.sub_clip,
            limiter_gr_db: if self.sub_frames > 0 && self.limiter_fade.is_wet() {
                self.sub_gr / self.sub_frames as f64
            } else {
                0.0
            },
            target_lufs: trim_in.target_lufs,
            output_lufs: self.trim.measured,
            density_available: self.multiband_fade.is_wet(),
            slower_agc_rate_db_s: trim_in.slower_rate_db_s,
        };
        self.governor.on_sub_block(&inputs);
        self.sub_clip = ClipStats::default();
        self.sub_gr = 0.0;
        self.sub_frames = 0;
        let d = self.governor.density_db();
        if (d - self.applied_density_db).abs() >= DENSITY_STEP_DB
            || (d == 0.0 && self.applied_density_db != 0.0)
        {
            self.applied_density_db = d;
            self.apply_multiband_params();
        }
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
        self.active_spike_lookahead_ms = p.spike_guard_lookahead_ms;
        self.active_bass_lookahead_ms = p.bass_clipper_lookahead_ms;
        self.active_clipper_latency = p.clipper_latency;
        self.active_bands = p.num_bands;

        self.spike
            .configure(sample_rate, channels, self.active_spike_lookahead_ms);
        self.agc.configure(sample_rate, channels);
        self.trim.allocate();
        self.multiband
            .set_bass_lookahead_ms(self.active_bass_lookahead_ms);
        self.multiband.configure(sample_rate, channels);
        self.clipper
            .configure(sample_rate, channels, self.active_clipper_latency);
        self.governor.allocate();
        self.limiter.configure(sample_rate, channels);
        // Apply the parameters (structural values included) before sizing
        // the delays from the limiter's realised latency.
        self.set_params(&p);
        let spike_latency = self.spike.inner.latency_frames();
        let agc_latency = AgcStage::latency_frames_for(sample_rate);
        let multiband_latency = self.multiband.latency_frames();
        let clipper_latency = self.clipper.inner.latency_frames();
        let limiter_latency = self.limiter.latency_frames();
        let total =
            spike_latency + agc_latency + multiband_latency + clipper_latency + limiter_latency;
        debug_assert_eq!(total, self.latency_frames());
        self.spike_delay.allocate(spike_latency, channels);
        self.agc_delay.allocate(agc_latency, channels);
        self.multiband_delay.allocate(multiband_latency, channels);
        self.clipper_delay.allocate(clipper_latency, channels);
        self.limiter_delay.allocate(limiter_latency, channels);
        self.chain_delay.allocate(total, channels);

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
            &mut self.spike_fade,
            &mut self.agc_fade,
            &mut self.multiband_fade,
            &mut self.clipper_fade,
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
        self.spike.reset();
        self.agc.reset();
        self.multiband.reset();
        self.trim.reset();
        self.ceiling_db = self.params.true_peak_ceiling_dbtp;
        let p = self.params;
        self.limiter
            .set_params(&p, self.ceiling_db, self.active_lookahead_ms);
        self.clipper.reset();
        self.governor.reset();
        if self.applied_density_db != 0.0 {
            self.applied_density_db = 0.0;
            self.apply_multiband_params();
        }
        self.sub_clip = ClipStats::default();
        self.sub_gr = 0.0;
        self.sub_frames = 0;
        self.meter_clip = ClipStats::default();
        self.limiter.reset();
        for d in [
            &mut self.spike_delay,
            &mut self.agc_delay,
            &mut self.multiband_delay,
            &mut self.clipper_delay,
            &mut self.limiter_delay,
            &mut self.chain_delay,
        ] {
            d.clear();
        }
        self.input_gain.snap();
        for f in [
            &mut self.spike_fade,
            &mut self.agc_fade,
            &mut self.multiband_fade,
            &mut self.clipper_fade,
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
        self.spike_events = 0;
        self.fresh = true;
    }

    fn latency_frames(&self) -> usize {
        Self::latency_frames_for_params(
            self.sample_rate,
            &BroadcastParams {
                spike_guard_lookahead_ms: self.active_spike_lookahead_ms,
                bass_clipper_lookahead_ms: self.active_bass_lookahead_ms,
                limiter_lookahead_ms: self.active_lookahead_ms,
                clipper_latency: self.active_clipper_latency,
                ..self.params
            },
        )
    }
}
