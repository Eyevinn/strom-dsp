//! Feed-forward downward compressor with soft knee.
//!
//! Structure after Giannoulis, Massberg & Reiss, "Digital Dynamic Range Compressor
//! Design — A Tutorial and Analysis" (JAES 60(6), 2012): the gain computer works on
//! the instantaneous level in dB and the resulting gain reduction is smoothed in the
//! log domain by a branching peak detector.
//!
//! # Signal flow (per frame of interleaved samples)
//!
//! 1. **Level detector**: the linked peak `p[n] = max_c |x_c[n]|` over all
//!    channels, then a **peak hold**: the detector level is the maximum of `p`
//!    over the last `H` frames (sliding window, van Herk / Gil–Werman
//!    [`SegmentedMin`] on `−p`), `H = min(`[`DETECTOR_HOLD_MS`]`, release)`, and
//!    `x_dB = 20·log10(level)` (floored at −200 dB). For a steady sine this is
//!    its **peak amplitude**, so levels and the threshold refer to peak level.
//!
//!    Why the hold (a fix for distortion found by ear on bass-heavy programme): without it the detector followed each
//!    half-cycle of the waveform, so with fast times the gain rippled at twice
//!    the signal frequency and distorted bass (2.5 % THD at 60 Hz with attack
//!    1 ms / release 50 ms; Giannoulis et al. 2012, section on artefacts of
//!    fast time constants: the detector release must be long compared with the
//!    period of the lowest frequency of interest). A window that is at least
//!    half a period long always contains a waveform peak, so the level of a
//!    steady sine above `1/(2H)` (25 Hz) is constant and the gain does not
//!    ripple at all, while an attack is still seen instantly. The cost is that
//!    a release starts `H` later. Tying `H` to the release keeps that delay at
//!    most one release time: an operator who sets a release below 20 ms gets
//!    a proportionally shorter hold (and more low-frequency distortion, which
//!    is then their choice).
//! 2. **Gain computer** ([`GainComputer`], static curve), with threshold
//!    `T = 20·log10(threshold)`, knee width `W = −40·log10(k)` dB (`k` = `knee`
//!    clamped to `[0.001, 1]`) and slope `S = 1 − 1/ratio`, the gain reduction
//!    `r ≥ 0` (dB) is
//!    - `r = 0` for `x < T − W/2`,
//!    - `r = S·(x − T + W/2)² / (2W)` for `T − W/2 ≤ x ≤ T + W/2` (quadratic soft knee),
//!    - `r = S·(x − T)` for `x > T + W/2`.
//!
//!    The output level is `x − r`, i.e. above the knee `T + (x − T)/ratio`.
//!    The knee region in linear terms spans `threshold·k … threshold/k`. `W = 0`
//!    (`k = 1`) is a hard knee.
//! 3. **Smoothing** ([`AttackRelease`], branching one-pole in the dB domain):
//!    `y[n] = a·y[n−1] + (1−a)·r[n]` with `a = a_A` when `r[n] > y[n−1]` (gain
//!    reduction increasing) and `a = a_R` otherwise, where
//!    `a_A = exp(−1/(τ_A·fs))`, `τ_A = attack_ms/1000` (and likewise for release).
//!    **Time constant definition**: after a step of the static gain reduction, the
//!    smoothed reduction (in dB) covers `1 − 1/e` (≈ 63.2 %) of the step after `τ`.
//!    A rising input level steps the static reduction at once (attack: 63.2 %
//!    after `τ_A`); a falling level steps it after the hold, i.e. `H` frames
//!    later (release: 63.2 % after `H + τ_R`).
//! 4. **Output gain**: `g = 10^(−y/20) · makeup`, applied to all channels.
//!
//! # Parameter smoothing
//!
//! Threshold (dB), slope `S`, knee width `W` and makeup gain (linear) glide toward
//! new values with a one-pole smoother of time constant [`PARAM_SMOOTHING_MS`], so
//! parameter changes do not click even at the shortest attack time. Ratio and knee
//! glide through `S` and `W`. Before the first `process` call after
//! `configure`/`reset`, parameters are applied immediately (no glide).
//!
//! # Efficiency
//!
//! Processing runs in blocks of [`BLOCK_FRAMES`]: per-frame peaks, the serial
//! detector/smoother chain, then gain application. While the parameters are not
//! gliding, a block whose held levels are all below the knee's lower edge
//! (compared in the linear domain) skips the `log10`, and a block with zero smoothed reduction
//! skips the `exp`; the result is the same as evaluating the full formula.
//!
//! # Denormals
//!
//! The smoothed reduction `y` snaps to its target when within
//! [`DENORMAL_FLUSH`](crate::util::DENORMAL_FLUSH) (so it decays to exactly 0);
//! parameter smoothers snap to their target when within `1e-9`.
//!
//! Ratio 1 gives `S = 0`, hence `r = 0` and exactly unity gain apart from makeup.
//!
//! # Sidechain (key input)
//!
//! [`Compressor::process_with_key`] feeds the level detector (step 1) from a
//! separate key signal through a [`KeyDetector`]: optional key filter
//! ([`Compressor::set_key_filter`], high-pass or band-pass) and the linked
//! peak over the key's channels, which may differ in number from the
//! programme's ([`Compressor::set_key_channels`]). The hold of step 1 and
//! steps 2–4 are unchanged.
//! [`Processor::process`] never touches the key path and is bit-identical to
//! the compressor without sidechain support.
//!
//! # Detector modes and calibration
//!
//! [`Compressor::set_detector`] selects what step 1 measures before the hold.
//! Both readings are **sine-calibrated**: a steady sine reads its peak level,
//! so the threshold means the same in both modes (and the same as the gate's
//! thresholds).
//!
//! - [`CompressorDetector::Peak`] (default, the original behaviour plus the hold):
//!   the level is the linked peak `max_c |x_c|`. A sine reads its peak, a
//!   constant `|x| = c` reads `c`, noise reads its largest peaks (RMS + crest
//!   factor).
//! - [`CompressorDetector::Rms`] (the classic choice for programme and
//!   speech; Zölzer, *DAFX*, RMS level measurement): a one-pole mean square of
//!   the linked peak, `m[n] = a·m[n−1] + (1−a)·p[n]²`, `a = exp(−1/(τ·fs))`,
//!   `τ =` rms-window ([`Compressor::set_rms_window`], 1 … 100 ms, default
//!   [`DEFAULT_RMS_WINDOW_MS`]), read as `sqrt(2·m)` (the gate's
//!   [`MeanSquareEnvelope`]). The factor 2 (+3.01 dB) is the sine
//!   calibration: a sine reads its peak, a constant `c` reads `√2·c`, noise
//!   reads its RMS + 3.01 dB. The hold then takes the maximum of `2·m`, so
//!   the remaining ripple of the mean square at `2f` does not modulate the
//!   gain; the price is that a low-frequency sine reads its ripple maximum,
//!   `10·log10(1 + r)` dB above its peak with `r = 1/sqrt(1 + (4π·f·τ)²)`:
//!   at τ = 10 ms +0.63 dB at 50 Hz, +0.33 dB at 100 Hz, +0.17 dB at
//!   200 Hz, +0.03 dB at 1 kHz. A level step from silence to `L` dB above a
//!   threshold crosses it after `−τ·ln(1 − 10^(−L/10))`; after the key stops
//!   the reading decays with `2τ` (after the hold).
//!
//! Switching modes while running seeds the new detector with the level the
//! old one held (a peak `p` becomes the power `p²` and vice versa), so the
//! detector level is continuous at the switch; differences between the two
//! readings afterwards are followed with the attack/release times.
//!
//! **Peak mode reads speech harder than an unheld peak detector.** The hold
//! makes the peak detector read the true waveform peak of every cycle, which
//! is what the static curve documents (a steady sine of any frequency above
//! 25 Hz gets exactly `r(peak)`). Without it, the reading dipped between
//! half-cycles and, with attack times longer than a half-period, settled
//! below the peak. On speech this means more gain reduction than before the
//! hold: on the speech-like test programme of
//! `tests/compressor_auto_makeup.rs` (−20.8 LUFS, peaks −3 dBFS; threshold
//! −20 dB, 4:1, knee 0.5012, 10 ms / 100 ms) the output is 3.1 LU quieter
//! than without the hold (−6.3 LU instead of −3.2 LU below the input; RMS
//! mode: −3.9 LU, almost unchanged). The hold length hardly matters: 10 ms
//! gives −6.2 LU, 5 ms −6.0 LU, so a shorter hold would not make peak mode
//! gentler, but at 5 ms the THD at 50 Hz rises from below 0.00001 % to
//! 0.32 %. Operators who want the gentler behaviour on speech should use the
//! `rms` detector or a higher threshold (about 3 dB).
//!
//! # Auto-makeup
//!
//! [`Compressor::set_auto_makeup`] (default off) adds a makeup gain `M` that
//! keeps the programme's perceived level roughly where it was without
//! compression, while threshold, ratio, knee and times change. `M` is
//! programme-dependent: a slow, gated estimate of the gain reduction the
//! programme actually gets, weighted by energy, as a loudness meter weights
//! it (ITU-R BS.1770 loudness is a mean square). The idea of deriving the
//! makeup from a long-term average of the gain reduction, instead of the
//! static curve at a fixed level, is from Giannoulis, Massberg & Reiss,
//! "Parameter Automation in a Dynamic Range Compressor" (JAES 61(10), 2013).
//! The weighting, the split below, the gate and the bounds are our own.
//!
//! Why not a fixed reference level: the static reduction at a fixed level
//! (Giannoulis et al. 2012 use 0 dBFS; an earlier version used −18 dB) is right only
//! for programme that sits at that level. Speech peaking near −3 dBFS reads
//! well above −18 dB, so the old reference under-compensated by 2 … 8 LU;
//! 0 dBFS over-compensates quiet programme by as much. No fixed level fits
//! programme at −16 and at −24 LUFS.
//!
//! **Estimate.** Per frame `n` whose held detector level `x[n]` (dB, as fed
//! to the gain computer) passes the gate [`AUTO_MAKEUP_GATE_DB`], with
//! programme energy `E[n] = p[n]²` (linked peak of the programme, also with
//! a key), static reduction `r[n]` and smoothed reduction `y[n]`:
//!
//! - `H(x)`: an energy histogram of `x` (1 dB bins from the gate to +10 dB,
//!   each frame split linearly between its two nearest bins);
//! - `A = Σ E·10^(−r/10)` and `B = Σ E·10^(−y/10)`.
//!
//! All three age exponentially with [`AUTO_MAKEUP_TIME_S`] (10 s) of gated
//! frames, updated every [`AUTO_MAKEUP_UPDATE_MS`]. Then, for the current
//! curve `r(·)`,
//!
//! `M = clamp(M_s + C, 0, `[`AUTO_MAKEUP_MAX_DB`]`)`,
//! `M_s = −10·log10(Σ H(x)·10^(−r(x)/10) / Σ H(x))`,
//! `C = clamp(10·log10(A/B), ±`[`AUTO_MAKEUP_CORRECTION_DB`]`)`.
//!
//! `M_s` is the makeup that gives the measured level distribution its energy
//! back through the static curve. It does not depend on the settings, so
//! when threshold, ratio or knee change, `M_s` for the new curve is known at
//! once. `C` corrects for attack and release (the smoothed reduction is not
//! the static one; on speech `|C|` is typically below 1 dB) and is
//! re-measured slowly after a change. With an estimate that is right, the
//! output energy over the last ~10 s equals the input's: `10^(M/10)·B ≈ ΣE`.
//!
//! - **Prior (start):** before any programme, `M` is the static reduction at
//!   [`AUTO_MAKEUP_PRIOR_DB`] (−12 dB, [`static_makeup_db`]), held as a mass
//!   worth [`AUTO_MAKEUP_PRIOR_S`] (0.5 s) of a sine at that level, which
//!   fades with the same time in gated frames. The first syllables move the
//!   estimate smoothly instead of in jumps.
//! - **Gate:** frames below −50 dB (detector level) neither enter nor age
//!   the statistic, so it freezes in silence and in pauses. Above the gate,
//!   noise or quiet programme that is not compressed can only lower `M`
//!   (`M ≥ 0`, and a frame without reduction pulls `M` toward 0); `M` never
//!   rises on noise.
//! - **Bounds:** `0 ≤ M ≤ 24 dB`; the correction is at most ±6 dB. The bound
//!   limits how far noise in pauses is raised after heavy compression
//!   (like any makeup, `M` also applies in pauses).
//! - **Gain:** `10^((M − y)/20) · makeup`. The explicit `makeup_gain` applies
//!   on top, unchanged. `M` glides to each new target with
//!   [`PARAM_SMOOTHING_MS`] (10 ms); targets change every 10 ms by at most
//!   a small step from the 10 s statistic, or once on a setting change, so
//!   the gain never steps.
//! - **Switching:** on → off glides `M` to 0 dB, off → on glides to the
//!   estimate. The statistic is kept while off but not updated (no cost
//!   while off); `reset`/`configure` returns it to the prior.
//! - **Determinism:** the statistic is updated per frame, so the result
//!   does not depend on how the caller splits buffers; no allocation (a
//!   fixed 61-bin array).
//!
//! Trade-off: `M` follows the programme with a 10 s memory, so level
//! differences that last much longer than that (a quiet speaker for a
//! minute) are evened out less than shorter ones; the energy weighting
//! keeps louder passages dominant, which slows this down. Measured
//! (`tests/compressor_auto_makeup.rs`): integrated loudness within ±1.1 LU
//! of the input over thresholds −30 … −10 dB, ratios 2 … 10, both detectors,
//! programmes at −16, −21 and −24 LUFS, and within ±0.6 LU for attack/release
//! from 0.1/20 ms to 200/2000 ms.

use crate::dynamics::frame::{apply_gains_simd, frame_peaks_simd, BLOCK_FRAMES};
use crate::dynamics::{
    vmath, AttackRelease, GainComputer, KeyDetector, KeyFilter, MeanSquareEnvelope, OnePole,
    SegmentedMin,
};
use crate::simd::{detect, Level};
use crate::util::db_to_lin;
use crate::Processor;
use fearless_simd::dispatch;
use fearless_simd::prelude::*;

pub use crate::dynamics::MIN_KNEE;

/// Time constant (ms) of the one-pole glide applied to threshold, ratio, knee and makeup.
pub const PARAM_SMOOTHING_MS: f64 = 10.0;

/// Longest peak hold of the level detector (ms): the hold is
/// `min(DETECTOR_HOLD_MS, release)`. 20 ms is half a period of 25 Hz, so a
/// steady sine at or above 25 Hz gives a ripple-free level (see the module docs).
pub const DETECTOR_HOLD_MS: f64 = 20.0;

/// Level detector of the compressor (see "Detector modes" in the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressorDetector {
    /// Linked sample peak (the original behaviour), with the peak hold.
    #[default]
    Peak,
    /// Sine-calibrated RMS over the rms-window, with the peak hold.
    Rms,
}

/// Default RMS averaging time (ms), the same as the gate's detector.
pub const DEFAULT_RMS_WINDOW_MS: f32 = 10.0;
/// Range of the RMS averaging time (ms).
pub const RMS_WINDOW_RANGE: (f32, f32) = (1.0, 100.0);

/// Time constant (s) of the programme statistic behind the automatic
/// makeup: how long the level distribution remembers programme (counted in
/// frames that pass [`AUTO_MAKEUP_GATE_DB`]). See "Auto-makeup" in the
/// module docs.
pub const AUTO_MAKEUP_TIME_S: f64 = 10.0;

/// Gate of the automatic makeup (dB, detector level, sine-calibrated):
/// frames whose held detector level is below it neither enter nor age the
/// programme statistic, so the estimate freezes in silence and pauses.
pub const AUTO_MAKEUP_GATE_DB: f64 = -50.0;

/// Largest automatic makeup (dB). It bounds how far quiet parts and noise
/// in pauses can be raised after strong compression.
pub const AUTO_MAKEUP_MAX_DB: f64 = 24.0;

/// Largest dynamic correction (dB, either sign) of the automatic makeup:
/// the measured difference between the static curve's prediction and the
/// gain actually applied (attack/release), see "Auto-makeup".
pub const AUTO_MAKEUP_CORRECTION_DB: f64 = 6.0;

/// Level (dB, sine-calibrated detector level) of the prior: before any
/// programme has been measured, the automatic makeup is the static gain
/// reduction at this level ([`static_makeup_db`]), a level typical of the
/// loud parts of speech and programme peaking near −6 … 0 dBFS.
pub const AUTO_MAKEUP_PRIOR_DB: f64 = -12.0;

/// Weight and lifetime of the prior (s): it counts as this many seconds of
/// a steady sine at [`AUTO_MAKEUP_PRIOR_DB`] and fades with this time
/// constant (in gated frames), so measured programme takes over within a
/// few times this.
pub const AUTO_MAKEUP_PRIOR_S: f64 = 0.5;

/// Interval (ms of gated frames) at which the statistic is aged and the
/// estimate recomputed; the result glides with [`PARAM_SMOOTHING_MS`].
pub const AUTO_MAKEUP_UPDATE_MS: f64 = 10.0;

/// Bin width (dB) of the level distribution. Each frame is split linearly
/// between the two nearest bins, so the estimate of a steady level is exact
/// to well under 0.05 dB.
const AUTO_BIN_DB: f64 = 1.0;
/// Highest binned level (dB); louder frames count as this level.
const AUTO_TOP_DB: f64 = 10.0;
/// Number of bins from [`AUTO_MAKEUP_GATE_DB`] to [`AUTO_TOP_DB`].
const AUTO_BINS: usize = ((AUTO_TOP_DB - AUTO_MAKEUP_GATE_DB) / AUTO_BIN_DB) as usize + 1;

/// Static automatic makeup (dB, 0 … [`AUTO_MAKEUP_MAX_DB`]) of the curve
/// `gc`: its gain reduction at [`AUTO_MAKEUP_PRIOR_DB`]. This is the
/// estimate before any programme has been measured.
pub fn static_makeup_db(gc: &GainComputer) -> f64 {
    gc.reduction_db(AUTO_MAKEUP_PRIOR_DB)
        .clamp(0.0, AUTO_MAKEUP_MAX_DB)
}

/// Per-block scratch of the automatic makeup (on the stack, no allocation).
struct AutoScratch {
    /// Held detector level (dB).
    level_db: [f64; BLOCK_FRAMES],
    /// Programme frame energy.
    energy: [f64; BLOCK_FRAMES],
    /// Static reduction (dB), then its power gain.
    static_power: [f64; BLOCK_FRAMES],
    /// Smoothed reduction (dB), then its power gain.
    applied_power: [f64; BLOCK_FRAMES],
}

impl AutoScratch {
    fn new() -> Self {
        Self {
            level_db: [1.0; BLOCK_FRAMES],
            energy: [0.0; BLOCK_FRAMES],
            static_power: [0.0; BLOCK_FRAMES],
            applied_power: [0.0; BLOCK_FRAMES],
        }
    }
}

/// One gated frame for [`MakeupEstimator::push`].
#[derive(Debug, Clone, Copy)]
struct MakeupFrame {
    /// Held detector level (dB, sine-calibrated).
    level_db: f64,
    /// Programme energy of the frame (linked peak squared).
    energy: f64,
    /// Static power gain `10^(−r/10)` of the frame's static reduction `r`.
    static_power: f64,
    /// Applied power gain `10^(−y/10)` of the smoothed reduction `y`.
    applied_power: f64,
}

/// Programme statistic of the automatic makeup (see "Auto-makeup" in the
/// module docs):
///
/// - an energy-weighted, exponentially ageing distribution of the detector
///   level (dB). It does not depend on threshold, ratio or knee, so the
///   makeup for new settings is known at once;
/// - two energy sums over the same frames, through the static gain and
///   through the applied (smoothed) gain, whose ratio is the slow
///   correction for attack/release;
/// - the prior, a mass at [`AUTO_MAKEUP_PRIOR_DB`] that fades quickly.
///
/// Frames collect in a pending part that is merged at each update, so the
/// estimate depends only on the state at the last update, not on when a
/// setter recomputes it.
#[derive(Debug, Clone)]
struct MakeupEstimator {
    /// Energy per level bin (bin `i` = `AUTO_MAKEUP_GATE_DB + i·AUTO_BIN_DB`).
    bins: [f64; AUTO_BINS],
    /// Prior mass at [`AUTO_MAKEUP_PRIOR_DB`].
    prior: f64,
    /// `Σ E·10^(−r/10)` (static curve).
    static_energy: f64,
    /// `Σ E·10^(−y/10)` (applied gain).
    applied_energy: f64,
    /// Frames since the last update, merged into the above at the next.
    pending_bins: [f64; AUTO_BINS],
    pending_static: f64,
    pending_applied: f64,
    /// Gated frames since the last update.
    count: usize,
    /// Gated frames per update (≥ 1).
    update_frames: usize,
    /// Ageing factor per update, `exp(−update_frames/(τ·fs))`.
    decay: f64,
    /// Fading factor of the prior per update.
    prior_decay: f64,
}

impl MakeupEstimator {
    fn new() -> Self {
        Self {
            bins: [0.0; AUTO_BINS],
            prior: 0.0,
            static_energy: 0.0,
            applied_energy: 0.0,
            pending_bins: [0.0; AUTO_BINS],
            pending_static: 0.0,
            pending_applied: 0.0,
            count: 0,
            update_frames: 1,
            decay: 1.0,
            prior_decay: 1.0,
        }
    }

    fn set_rate(&mut self, fs: f64) {
        let frames = (AUTO_MAKEUP_UPDATE_MS * 0.001 * fs).round();
        self.update_frames = if frames >= 1.0 { frames as usize } else { 1 };
        let n = self.update_frames as f64;
        self.decay = (-n / (AUTO_MAKEUP_TIME_S * fs)).exp();
        self.prior_decay = (-n / (AUTO_MAKEUP_PRIOR_S * fs)).exp();
    }

    /// Forget the programme: only the prior remains.
    fn reset(&mut self, fs: f64) {
        *self = Self {
            update_frames: self.update_frames,
            decay: self.decay,
            prior_decay: self.prior_decay,
            ..Self::new()
        };
        // A sine of peak A has mean frame energy A²/2.
        self.prior = 0.5 * db_to_lin(2.0 * AUTO_MAKEUP_PRIOR_DB) * AUTO_MAKEUP_PRIOR_S * fs;
    }

    /// Count one frame. Frames below the gate are ignored (the statistic
    /// freezes). Returns true at an update: the statistic was aged and
    /// merged, and the estimate should be recomputed.
    #[inline]
    fn push(&mut self, f: MakeupFrame) -> bool {
        // Gate (also rejects NaN levels); non-finite energy is ignored.
        let open = f.level_db >= AUTO_MAKEUP_GATE_DB;
        if !open || !f.energy.is_finite() {
            return false;
        }
        let pos = (f.level_db.min(AUTO_TOP_DB) - AUTO_MAKEUP_GATE_DB) / AUTO_BIN_DB;
        let i = pos as usize;
        if i + 1 >= AUTO_BINS {
            self.pending_bins[AUTO_BINS - 1] += f.energy;
        } else {
            let frac = pos - i as f64;
            self.pending_bins[i] += f.energy * (1.0 - frac);
            self.pending_bins[i + 1] += f.energy * frac;
        }
        self.pending_static += f.energy * f.static_power;
        self.pending_applied += f.energy * f.applied_power;
        self.count += 1;
        if self.count < self.update_frames {
            return false;
        }
        self.count = 0;
        self.update();
        true
    }

    /// Age the committed statistic by one update and merge the pending part.
    fn update(&mut self) {
        for (b, p) in self.bins.iter_mut().zip(self.pending_bins.iter_mut()) {
            *b = *b * self.decay + *p;
            *p = 0.0;
        }
        let total: f64 = self.bins.iter().sum();
        // Bins that have aged far below the rest no longer matter: flush
        // them so they never become subnormal.
        let floor = total * 1e-30;
        for b in self.bins.iter_mut() {
            if *b < floor {
                *b = 0.0;
            }
        }
        self.static_energy = self.static_energy * self.decay + self.pending_static;
        self.applied_energy = self.applied_energy * self.decay + self.pending_applied;
        self.pending_static = 0.0;
        self.pending_applied = 0.0;
        self.prior *= self.prior_decay;
        if self.prior < floor {
            self.prior = 0.0;
        }
    }

    /// Makeup (dB, 0 … [`AUTO_MAKEUP_MAX_DB`]) for the static curve `gc`:
    /// the static part `M_s = −10·log10(Σ E_i·10^(−r(x_i)/10) / Σ E_i)`
    /// over the bins and the prior, plus the correction
    /// `C = 10·log10(Σ E·10^(−r/10) / Σ E·10^(−y/10))` (±
    /// [`AUTO_MAKEUP_CORRECTION_DB`]).
    fn makeup_db(&self, gc: &GainComputer) -> f64 {
        if gc.slope <= 0.0 {
            return 0.0;
        }
        // Power gain 10^(−r/10) at level x.
        let power = |x: f64| {
            let g = db_to_lin(-gc.reduction_db(x));
            g * g
        };
        let mut kept = self.prior * power(AUTO_MAKEUP_PRIOR_DB);
        let mut total = self.prior;
        for (i, &e) in self.bins.iter().enumerate() {
            if e > 0.0 {
                kept += e * power(AUTO_MAKEUP_GATE_DB + i as f64 * AUTO_BIN_DB);
                total += e;
            }
        }
        // Also catches NaN.
        let valid = total > 0.0 && kept > 0.0;
        if !valid {
            return static_makeup_db(gc);
        }
        let correction = if self.static_energy > 0.0 && self.applied_energy > 0.0 {
            (10.0 * (self.static_energy / self.applied_energy).log10())
                .clamp(-AUTO_MAKEUP_CORRECTION_DB, AUTO_MAKEUP_CORRECTION_DB)
        } else {
            0.0
        };
        (-10.0 * (kept / total).log10() + correction).clamp(0.0, AUTO_MAKEUP_MAX_DB)
    }
}

/// Largest sample rate `configure` accepts (Hz); also sizes the hold window.
const MAX_SAMPLE_RATE: f64 = 1.0e6;

/// Hold length in frames for `release_ms` at `fs` (≥ 1; 1 = no hold).
fn hold_frames(release_ms: f64, fs: f64) -> usize {
    let f = (release_ms.min(DETECTOR_HOLD_MS) * 0.001 * fs).round();
    if f.is_finite() && f >= 1.0 {
        f as usize
    } else {
        1
    }
}

/// Snap distance of the parameter glides.
const GLIDE_SNAP: f64 = 1e-9;

/// Compressor parameters (units as in the GStreamer interface).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressorParams {
    /// Threshold as linear amplitude (0 … 1).
    pub threshold: f32,
    /// Compression ratio (1 … 100; 4 means 4:1).
    pub ratio: f32,
    /// Attack time (ms, 0.01 … 10000).
    pub attack_ms: f32,
    /// Release time (ms, 0.01 … 10000).
    pub release_ms: f32,
    /// Makeup gain as linear multiplier (0 … 100).
    pub makeup_gain: f32,
    /// Knee factor `k` (0 … 1): soft-knee region spans `threshold·k … threshold/k`,
    /// width `−40·log10(k)` dB. 1 = hard knee; values below 0.001 are treated as 0.001.
    pub knee: f32,
}

impl Default for CompressorParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            ratio: 1.0,
            attack_ms: 20.0,
            release_ms: 100.0,
            makeup_gain: 1.0,
            knee: 0.0625,
        }
    }
}

/// Clamp `v` to `[lo, hi]`, replacing NaN with `fallback`.
fn clamp_or(v: f32, lo: f32, hi: f32, fallback: f32) -> f32 {
    if v.is_nan() {
        fallback
    } else {
        v.clamp(lo, hi)
    }
}

impl CompressorParams {
    /// Copy with every field clamped to its documented range (NaN becomes the default).
    /// `knee` keeps values in `[0, 0.001)` as given; they act as 0.001.
    pub fn clamped(&self) -> Self {
        let d = Self::default();
        Self {
            threshold: clamp_or(self.threshold, 0.0, 1.0, d.threshold),
            ratio: clamp_or(self.ratio, 1.0, 100.0, d.ratio),
            attack_ms: clamp_or(self.attack_ms, 0.01, 10_000.0, d.attack_ms),
            release_ms: clamp_or(self.release_ms, 0.01, 10_000.0, d.release_ms),
            makeup_gain: clamp_or(self.makeup_gain, 0.0, 100.0, d.makeup_gain),
            knee: clamp_or(self.knee, 0.0, 1.0, d.knee),
        }
    }
}

/// Static gain reduction in dB (≥ 0) of the gain computer for input level `x_db`,
/// threshold `t_db`, knee width `w_db` and slope `s = 1 − 1/ratio`.
#[cfg(test)]
fn gain_reduction(x_db: f64, t_db: f64, w_db: f64, s: f64) -> f64 {
    GainComputer {
        threshold_db: t_db,
        knee_db: w_db,
        slope: s,
    }
    .reduction_db(x_db)
}

/// The gain-computer quantities that glide on parameter changes.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Curve {
    threshold_db: f64,
    slope: f64,
    knee_db: f64,
    makeup: f64,
}

impl Curve {
    /// Curve with the explicit makeup gain (the automatic makeup is a
    /// separate factor, see `Compressor::auto_gain`).
    fn from_params(p: &CompressorParams) -> Self {
        let gc = GainComputer::from_linear(p.threshold as f64, p.ratio as f64, p.knee as f64);
        Self {
            threshold_db: gc.threshold_db,
            slope: gc.slope,
            knee_db: gc.knee_db,
            makeup: p.makeup_gain as f64,
        }
    }

    fn computer(&self) -> GainComputer {
        GainComputer {
            threshold_db: self.threshold_db,
            knee_db: self.knee_db,
            slope: self.slope,
        }
    }
}

/// One-pole glides for each [`Curve`] field.
#[derive(Debug, Clone, Copy)]
struct CurveGlide {
    threshold_db: OnePole,
    slope: OnePole,
    knee_db: OnePole,
    makeup: OnePole,
}

impl CurveGlide {
    fn set_time(&mut self, fs: f64) {
        for g in [
            &mut self.threshold_db,
            &mut self.slope,
            &mut self.knee_db,
            &mut self.makeup,
        ] {
            g.set_time(PARAM_SMOOTHING_MS, fs);
        }
    }

    fn jump(&mut self, c: &Curve) {
        self.threshold_db.set_value(c.threshold_db);
        self.slope.set_value(c.slope);
        self.knee_db.set_value(c.knee_db);
        self.makeup.set_value(c.makeup);
    }

    #[inline]
    fn step(&mut self, c: &Curve) -> Curve {
        Curve {
            threshold_db: self.threshold_db.step(c.threshold_db),
            slope: self.slope.step(c.slope),
            knee_db: self.knee_db.step(c.knee_db),
            makeup: self.makeup.step(c.makeup),
        }
    }
}

/// Linked-channel feed-forward compressor. See [`CompressorParams`] and the module
/// documentation.
#[derive(Debug)]
pub struct Compressor {
    params: CompressorParams,
    sample_rate: f64,
    /// 0 = not configured.
    channels: usize,

    target: Curve,
    /// Curve in force (equal to `target` when not gliding).
    current: Curve,
    glide: CurveGlide,
    gliding: bool,
    /// Linear level at or below which the settled curve gives exactly 0 dB reduction.
    knee_start_lin: f64,

    /// Sliding maximum of the detector level (as a minimum of its negation).
    hold: SegmentedMin,
    /// Most recent held detector level (amplitude for peak, power for RMS).
    held: f64,
    detector: CompressorDetector,
    /// RMS averaging time (ms, clamped).
    rms_window_ms: f32,
    /// Mean-square envelope of the RMS detector.
    mean_square: MeanSquareEnvelope,
    /// Automatic makeup on (see "Auto-makeup" in the module docs).
    auto_makeup: bool,
    /// Programme statistic of the automatic makeup.
    estimator: MakeupEstimator,
    /// Automatic makeup the gain glides toward (linear; 1 when off).
    auto_target: f64,
    /// Glide of the automatic makeup (linear), [`PARAM_SMOOTHING_MS`].
    auto_gain: OnePole,
    /// The automatic makeup factor is in use (on, or gliding back to 1).
    auto_active: bool,
    /// Smoothed gain reduction (dB, ≥ 0).
    reduction: AttackRelease,
    /// True until the first frame is processed after configure/reset: parameters snap.
    fresh: bool,
    /// Instruction set of the hot loops (detected in `configure`).
    level: Level,
    /// Set by `set_simd_level`: `configure` keeps `level` instead of detecting.
    level_forced: bool,
    /// Sidechain front end, used only by `process_with_key`.
    key: KeyDetector,
    /// Key channel count set by `set_key_channels` (0 = follow `channels`).
    key_channels: usize,
}

impl Default for Compressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor {
    /// Create an unconfigured compressor with default parameters.
    pub fn new() -> Self {
        let params = CompressorParams::default();
        let curve = Curve::from_params(&params);
        let glide = OnePole::default().with_snap(GLIDE_SNAP);
        let mut c = Self {
            params,
            sample_rate: 48_000.0,
            channels: 0,
            target: curve,
            current: curve,
            glide: CurveGlide {
                threshold_db: glide,
                slope: glide,
                knee_db: glide,
                makeup: glide,
            },
            gliding: false,
            knee_start_lin: curve.computer().knee_start_lin(),
            hold: SegmentedMin::default(),
            held: 0.0,
            detector: CompressorDetector::Peak,
            rms_window_ms: DEFAULT_RMS_WINDOW_MS,
            mean_square: MeanSquareEnvelope::default(),
            auto_makeup: false,
            estimator: MakeupEstimator::new(),
            auto_target: 1.0,
            auto_gain: OnePole::new(0.0, 48_000.0, 1.0).with_snap(GLIDE_SNAP),
            auto_active: false,
            reduction: AttackRelease::default(),
            fresh: true,
            level: Level::baseline(),
            level_forced: false,
            key: KeyDetector::new(),
            key_channels: 0,
        };
        c.estimator.reset(c.sample_rate);
        c.update_coefficients();
        c
    }

    /// Set the key's channel count for
    /// [`process_with_key`](Self::process_with_key) (≥ 1). **Allocates**
    /// (key-filter state): call it at setup time, like `configure`. Without
    /// this call the key has the programme's channel count.
    pub fn set_key_channels(&mut self, key_channels: usize) {
        self.key_channels = key_channels.max(1);
        self.key
            .configure(self.sample_rate as f32, self.key_channels);
    }

    /// Key channel count expected by `process_with_key`.
    pub fn key_channels(&self) -> usize {
        self.key.key_channels()
    }

    /// Set the key filter of the sidechain detector path. Never allocates.
    pub fn set_key_filter(&mut self, filter: KeyFilter) {
        self.key.set_filter(filter);
    }

    /// Current key filter.
    pub fn key_filter(&self) -> KeyFilter {
        self.key.filter()
    }

    /// Process interleaved `samples` in place, with the level detector fed
    /// from the interleaved `key` (`key_channels` channels, same rate and
    /// frame count) instead of `samples`. Real-time safe. `key_channels` must
    /// equal [`key_channels`](Self::key_channels); a mismatching or too short
    /// key is detected as silence (see [`KeyDetector`]).
    pub fn process_with_key(&mut self, samples: &mut [f32], key: &[f32], key_channels: usize) {
        let ch = self.channels;
        if ch == 0 || samples.len() < ch {
            return;
        }
        self.fresh = false;
        let key = Some((key, key_channels));
        dispatch!(self.level, simd => self.process_blocks(simd, samples, key));
    }

    /// Update parameters (clamped to their ranges). Never allocates.
    pub fn set_params(&mut self, params: &CompressorParams) {
        self.params = params.clamped();
        self.update_coefficients();
    }

    /// Current (clamped) parameters.
    pub fn params(&self) -> &CompressorParams {
        &self.params
    }

    /// Select the level detector (see "Detector modes" in the module docs).
    /// Never allocates. While running, the new detector is seeded with the
    /// level the old one held, so the detector level is continuous.
    pub fn set_detector(&mut self, detector: CompressorDetector) {
        if detector == self.detector {
            return;
        }
        self.detector = detector;
        if self.fresh {
            self.held = 0.0;
            self.mean_square.reset();
            self.hold.reset(self.hold.window());
        } else {
            // Peak amplitude p ↔ sine-calibrated power p².
            self.held = match detector {
                CompressorDetector::Rms => {
                    let power = self.held * self.held;
                    self.mean_square.set_value(power);
                    power
                }
                CompressorDetector::Peak => self.held.sqrt(),
            };
            self.hold.reset(self.hold.window());
            self.hold.push(-self.held);
        }
    }

    /// Current level detector.
    pub fn detector(&self) -> CompressorDetector {
        self.detector
    }

    /// Set the RMS averaging time (ms, 1 … 100, default
    /// [`DEFAULT_RMS_WINDOW_MS`]; NaN becomes the default). Used only by
    /// [`CompressorDetector::Rms`]. Never allocates; the envelope keeps its
    /// value, so a change does not jump.
    pub fn set_rms_window(&mut self, window_ms: f32) {
        let (lo, hi) = RMS_WINDOW_RANGE;
        self.rms_window_ms = clamp_or(window_ms, lo, hi, DEFAULT_RMS_WINDOW_MS);
        self.mean_square
            .set_time(self.rms_window_ms as f64, self.sample_rate);
    }

    /// Current (clamped) RMS averaging time (ms).
    pub fn rms_window(&self) -> f32 {
        self.rms_window_ms
    }

    /// Switch the automatic makeup gain on or off (see "Auto-makeup" in the
    /// module docs). The explicit `makeup_gain` still applies on top. The
    /// change glides over [`PARAM_SMOOTHING_MS`]. The programme statistic is
    /// kept while off (but not updated), so switching back on resumes from
    /// it. Never allocates.
    pub fn set_auto_makeup(&mut self, on: bool) {
        self.auto_makeup = on;
        self.update_auto_target();
    }

    /// Whether the automatic makeup gain is on.
    pub fn auto_makeup(&self) -> bool {
        self.auto_makeup
    }

    /// Makeup gain (dB) the automatic makeup currently targets for the
    /// current parameters and the programme measured so far (0 when it is
    /// off), excluding `makeup_gain`. The applied value glides toward it.
    pub fn auto_makeup_gain_db(&self) -> f64 {
        if self.auto_makeup {
            crate::util::lin_to_db(self.auto_target)
        } else {
            0.0
        }
    }

    /// Recompute the automatic makeup target from the statistic and the
    /// target curve. Never allocates.
    fn update_auto_target(&mut self) {
        self.auto_target = if self.auto_makeup {
            db_to_lin(self.estimator.makeup_db(&self.target.computer()))
        } else {
            1.0
        };
        if self.fresh {
            self.auto_gain.set_value(self.auto_target);
        }
        self.auto_active = self.auto_makeup || self.auto_gain.value() != 1.0;
    }

    /// Smoothed gain reduction (dB, ≥ 0) after the most recently processed frame,
    /// excluding makeup gain.
    pub fn gain_reduction_db(&self) -> f64 {
        self.reduction.value()
    }

    /// Force the SIMD instruction set used by `process` (for hosts and tests),
    /// e.g. `Level::fallback()` (needs fearless_simd's `force_support_fallback`
    /// feature) to run the portable scalar path. By default
    /// `configure` detects the best level ([`crate::simd::detect`]); a level
    /// set here is kept across later `configure` calls. Every level produces
    /// bit-identical output (the dB conversions use the same approximation on
    /// every level, see the module docs). Never allocates.
    pub fn set_simd_level(&mut self, level: Level) {
        self.level = level;
        self.level_forced = true;
        self.key.set_simd_level(level);
    }

    /// The SIMD instruction set `process` currently dispatches to.
    pub fn simd_level(&self) -> Level {
        self.level
    }

    /// Block loop, monomorphised per instruction set (see "Efficiency" in
    /// the module docs).
    #[inline(always)]
    fn process_blocks<S: Simd>(
        &mut self,
        simd: S,
        samples: &mut [f32],
        key: Option<(&[f32], usize)>,
    ) {
        let ch = self.channels;
        let n = S::f64s::LEN;
        // Per-frame scratch, reused through the stages of a block: peak →
        // level (dB) → static reduction → smoothed reduction → gain.
        let mut peaks = [0.0f32; BLOCK_FRAMES];
        let mut level = [0.0f64; BLOCK_FRAMES];
        let mut makeup = [0.0f64; BLOCK_FRAMES];
        let mut auto_scratch = AutoScratch::new();
        for (b, block) in samples.chunks_mut(BLOCK_FRAMES * ch).enumerate() {
            let frames = block.len() / ch;
            if frames == 0 {
                continue;
            }
            // Vector loops cover whole vectors; lanes past `frames` hold
            // finite leftovers and are ignored. BLOCK_FRAMES is a multiple of
            // every native width.
            let lanes = frames.div_ceil(n) * n;
            match key {
                None => frame_peaks_simd(simd, block, ch, &mut peaks[..frames]),
                Some((k, kc)) => {
                    let start = (b * BLOCK_FRAMES * kc).min(k.len());
                    self.key
                        .linked_peaks_simd(simd, &k[start..], kc, &mut peaks[..frames]);
                }
            }
            // Detector, then the peak hold: sliding maximum over the last
            // `H` frames (serial). Peak: amplitude; RMS: sine-calibrated power.
            let rms = self.detector == CompressorDetector::Rms;
            let mut block_max = 0.0f64;
            for (l, &p) in level[..frames].iter_mut().zip(&peaks) {
                let d = if rms {
                    self.mean_square.step(p as f64)
                } else {
                    p as f64
                };
                let h = -self.hold.push(-d);
                block_max = if h > block_max { h } else { block_max };
                *l = h;
            }
            self.held = level[frames - 1];
            let rms_scale = if rms { 0.5 } else { 1.0 };
            // Automatic makeup, measuring: level (dB) and programme energy.
            let auto = self.auto_active;
            let measure = auto && self.auto_makeup;
            if measure {
                auto_scratch.level_db[..frames].copy_from_slice(&level[..frames]);
                auto_scratch.level_db[frames..lanes].fill(1.0);
                for v in auto_scratch.level_db[..lanes].chunks_exact_mut(n) {
                    (vmath::lin_to_db(simd, S::f64s::from_slice(simd, v)) * rms_scale)
                        .store_slice(v);
                }
                // Programme peaks: the detector's, unless a key feeds it.
                let mut own = [0.0f32; BLOCK_FRAMES];
                let prog = if key.is_some() {
                    frame_peaks_simd(simd, block, ch, &mut own[..frames]);
                    &own[..frames]
                } else {
                    &peaks[..frames]
                };
                for (e, &p) in auto_scratch.energy.iter_mut().zip(prog) {
                    *e = p as f64 * p as f64;
                }
            }
            // Settled curve and the whole block below the knee: the static
            // reduction is exactly 0, no need for the logarithm.
            let knee_start = if rms {
                self.knee_start_lin * self.knee_start_lin
            } else {
                self.knee_start_lin
            };
            let quiet = !self.gliding && block_max <= knee_start;
            if quiet {
                level[..frames].fill(0.0);
            } else {
                // Lanes past `frames` must hold finite values.
                level[frames..lanes].fill(1.0);
                // RMS: 20·log10(sqrt(P)) = 0.5 · 20·log10(P) (exact scaling).
                for v in level[..lanes].chunks_exact_mut(n) {
                    (vmath::lin_to_db(simd, S::f64s::from_slice(simd, v)) * rms_scale)
                        .store_slice(v);
                }
            }

            if quiet {
                makeup.fill(self.current.makeup);
            } else if self.gliding {
                for (l, mk) in level[..frames].iter_mut().zip(&mut makeup) {
                    self.current = self.glide.step(&self.target);
                    if self.current == self.target {
                        self.gliding = false;
                        self.knee_start_lin = self.current.computer().knee_start_lin();
                    }
                    *l = self.current.computer().reduction_db(*l);
                    *mk = self.current.makeup;
                }
            } else {
                let gc = self.current.computer();
                for v in level[..lanes].chunks_exact_mut(n) {
                    reduction_db_simd(simd, &gc, S::f64s::from_slice(simd, v)).store_slice(v);
                }
                makeup.fill(self.current.makeup);
            }
            if measure {
                auto_scratch.static_power[..lanes].copy_from_slice(&level[..lanes]);
            }
            // Serial attack/release smoothing of the reduction (dB).
            self.reduction.step_block(&mut level[..frames]);
            if measure {
                auto_scratch.applied_power[..lanes].copy_from_slice(&level[..lanes]);
            }

            // Gain 10^(−y/20)·makeup; exactly `makeup` at zero reduction
            // (y ≥ 0 always, so a block without reduction skips the exp).
            if level[..frames].iter().any(|&y| y > 0.0) {
                let zero = S::f64s::splat(simd, 0.0);
                for (v, mk) in level[..lanes]
                    .chunks_exact_mut(n)
                    .zip(makeup.chunks_exact(n))
                {
                    let y = S::f64s::from_slice(simd, v);
                    let mk = S::f64s::from_slice(simd, mk);
                    let g = vmath::db_to_gain(simd, y) * mk;
                    y.simd_gt(zero).select(g, mk).store_slice(v);
                }
            } else {
                level[..frames].copy_from_slice(&makeup[..frames]);
            }
            if auto {
                self.auto_makeup_block(simd, frames, &mut auto_scratch, &mut level);
            }
            for (p, &g) in peaks.iter_mut().zip(&level[..frames]) {
                *p = g as f32;
            }
            apply_gains_simd(simd, block, ch, &peaks[..frames]);
        }
    }

    /// Automatic makeup for one block (see "Auto-makeup" in the module
    /// docs). While on, feeds each frame to the statistic (static reduction
    /// and smoothed reduction in `s`, both dB, become power gains here) and
    /// recomputes the target at each update; always multiplies the frame
    /// gains `gains` by the gliding makeup factor.
    #[inline(always)]
    fn auto_makeup_block<S: Simd>(
        &mut self,
        simd: S,
        frames: usize,
        s: &mut AutoScratch,
        gains: &mut [f64; BLOCK_FRAMES],
    ) {
        if !self.auto_makeup {
            for g in gains[..frames].iter_mut() {
                *g *= self.auto_gain.step(self.auto_target);
            }
            if self.auto_gain.value() == self.auto_target {
                self.auto_active = false;
            }
            return;
        }
        let n = S::f64s::LEN;
        let lanes = frames.div_ceil(n) * n;
        for buf in [&mut s.static_power, &mut s.applied_power] {
            for v in buf[..lanes].chunks_exact_mut(n) {
                let g = vmath::db_to_gain(simd, S::f64s::from_slice(simd, v));
                (g * g).store_slice(v);
            }
        }
        let gc = self.target.computer();
        for (i, g) in gains[..frames].iter_mut().enumerate() {
            let frame = MakeupFrame {
                level_db: s.level_db[i],
                energy: s.energy[i],
                static_power: s.static_power[i],
                applied_power: s.applied_power[i],
            };
            if self.estimator.push(frame) {
                self.auto_target = db_to_lin(self.estimator.makeup_db(&gc));
            }
            *g *= self.auto_gain.step(self.auto_target);
        }
    }

    fn update_coefficients(&mut self) {
        let fs = self.sample_rate;
        self.reduction.set_times(
            self.params.attack_ms as f64,
            self.params.release_ms as f64,
            fs,
        );
        self.glide.set_time(fs);
        self.mean_square.set_time(self.rms_window_ms as f64, fs);
        let h = hold_frames(self.params.release_ms as f64, fs);
        if h != self.hold.window() {
            // Restart the window with the level held so far, so a release
            // change never drops the detector level abruptly.
            self.hold.reset(h);
            if !self.fresh {
                self.hold.push(-self.held);
            }
        }
        self.target = Curve::from_params(&self.params);
        if self.fresh {
            self.glide.jump(&self.target);
            self.current = self.target;
        }
        self.gliding = self.current != self.target;
        self.knee_start_lin = self.current.computer().knee_start_lin();
        self.auto_gain.set_time(PARAM_SMOOTHING_MS, fs);
        self.estimator.set_rate(fs);
        self.update_auto_target();
    }
}

/// [`GainComputer::reduction_db`] per lane: the same operations in the same
/// order, so every lane equals the scalar result bit for bit.
#[inline(always)]
fn reduction_db_simd<S: Simd>(simd: S, gc: &GainComputer, x: S::f64s) -> S::f64s {
    let over = x - gc.threshold_db;
    let half = 0.5 * gc.knee_db;
    let d = over + half;
    let in_knee = d * gc.slope * d / (2.0 * gc.knee_db);
    let above = over * gc.slope;
    let r = over
        .simd_lt(S::f64s::splat(simd, half))
        .select(in_knee, above);
    over.simd_le(S::f64s::splat(simd, -half))
        .select(S::f64s::splat(simd, 0.0), r)
}

impl Processor for Compressor {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        self.sample_rate = (sample_rate as f64).clamp(1.0, MAX_SAMPLE_RATE);
        self.channels = channels;
        // The only allocation: the hold window for the longest hold.
        self.hold
            .allocate(hold_frames(DETECTOR_HOLD_MS, self.sample_rate));
        if !self.level_forced {
            self.level = detect();
        }
        let kc = if self.key_channels > 0 {
            self.key_channels
        } else {
            channels
        };
        self.key.configure(sample_rate, kc);
        self.reset();
    }

    fn process(&mut self, samples: &mut [f32]) {
        let ch = self.channels;
        if ch == 0 || samples.len() < ch {
            return;
        }
        self.fresh = false;
        dispatch!(self.level, simd => self.process_blocks(simd, samples, None));
    }

    fn reset(&mut self) {
        self.key.reset();
        self.reduction.set_value(0.0);
        self.fresh = true;
        self.held = 0.0;
        self.mean_square.reset();
        let h = hold_frames(self.params.release_ms as f64, self.sample_rate);
        self.hold.reset(h);
        self.estimator.reset(self.sample_rate);
        self.update_coefficients();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::lin_to_db;

    #[test]
    fn gain_computer_is_continuous_at_knee_edges() {
        let (t, w, s) = (-20.0, 12.0, 0.75);
        for edge in [t - w / 2.0, t + w / 2.0] {
            let lo = gain_reduction(edge - 1e-9, t, w, s);
            let hi = gain_reduction(edge + 1e-9, t, w, s);
            assert!((lo - hi).abs() < 1e-6);
        }
        // Hard knee.
        assert_eq!(gain_reduction(-21.0, t, 0.0, s), 0.0);
        assert!((gain_reduction(-10.0, t, 0.0, s) - 7.5).abs() < 1e-12);
    }

    #[test]
    fn knee_width_matches_definition() {
        let p = CompressorParams {
            knee: 0.1,
            ..Default::default()
        };
        assert!((Curve::from_params(&p).knee_db - 40.0).abs() < 1e-5);
        let p0 = CompressorParams {
            knee: 0.0,
            ..Default::default()
        };
        assert!((Curve::from_params(&p0).knee_db - 120.0).abs() < 1e-9);
    }

    #[test]
    fn fast_path_matches_full_formula() {
        // Settled curve: skipping log10 below the knee must not change the gain.
        let p = CompressorParams {
            threshold: 0.25,
            ratio: 4.0,
            knee: 0.5,
            attack_ms: 0.01,
            release_ms: 0.01,
            ..Default::default()
        };
        let gc = Curve::from_params(&p.clamped()).computer();
        let edge = gc.knee_start_lin();
        for x in [0.0, 1e-6, edge * 0.999, edge, edge * 1.000_001, 0.2, 0.9] {
            let mut c = Compressor::new();
            c.set_params(&p);
            c.configure(48_000.0, 1);
            let mut buf = [x as f32; 64];
            c.process(&mut buf);
            let want = gc.reduction_db(lin_to_db(x as f32 as f64));
            assert!((c.gain_reduction_db() - want).abs() < 1e-9, "x={x}");
        }
    }
}
