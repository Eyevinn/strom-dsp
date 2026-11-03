//! Downward noise gate with hysteresis, hold and attack/release fades.
//!
//! # Signal flow (per frame of interleaved samples)
//!
//! 1. **Key level**: the largest absolute sample value over all channels of the
//!    frame (channels are linked, so one gain is shared by all of them).
//! 2. **Detector**: a peak envelope follower ([`PeakEnvelope`]) with
//!    instantaneous attack and a fixed exponential release of [`DETECTOR_RELEASE_MS`]:
//!    `e[n] = p[n]` if `p[n] > e[n-1]`, otherwise `e[n] = a_d·e[n-1] + (1-a_d)·p[n]`
//!    with `a_d = exp(-1 / (τ_d·fs))`, `τ_d = DETECTOR_RELEASE_MS / 1000`. For a steady
//!    sine the envelope sits at the sine's peak amplitude (ripple well below 1 dB above
//!    ~200 Hz), so the thresholds compare against the **peak level in dBFS**.
//! 3. **State machine with hysteresis and hold** ([`GateStateMachine`]): a closed
//!    gate opens when `e > open_threshold`. An open gate re-arms its hold counter
//!    (`round(hold_ms·fs/1000)` frames) every frame where `e ≥ close_threshold`;
//!    once `e` is below the close threshold the counter runs down, and when it is
//!    exhausted the gate closes. If `close_threshold_db > open_threshold_db` the
//!    close threshold is treated as equal to the open threshold.
//! 4. **Gain fade** ([`AttackRelease`] on the linear gain): the gain `g` follows its
//!    target (1 when open, the linear `reduction` when closed) with a one-pole smoother,
//!    `g[n] = g[n-1] + (1 - a)·(target - g[n-1])`, where `a = exp(-1 / (τ·fs))`,
//!    `τ = attack_ms/1000` while the gain rises and `τ = release_ms/1000` while it
//!    falls. So **attack/release is the time for the linear gain to cover
//!    `1 - 1/e` (≈ 63.2 %) of the distance to its target** after a step.
//!
//! Parameter changes are click-free: a changed `reduction` (or a state change from
//! a changed threshold) only moves the gain target, and the gain always reaches it
//! through the attack/release smoother. The exception is the time between
//! `configure`/`reset` and the first processed frame: there `set_params` applies
//! instantly, so the result does not depend on whether parameters were set before
//! or after `configure`.
//!
//! Processing runs in blocks of [`BLOCK_FRAMES`]: per-frame peaks for the block,
//! then the serial detector/state/fade chain, then the gains are applied.
//!
//! Denormals: the detector envelope is passed through
//! [`flush_denormal`](crate::util::flush_denormal); the gain is snapped to its
//! target once it is within `1e-12` (targets are ≥ 1e-4, so it never decays toward 0).
//!
//! # Sidechain (key input)
//!
//! [`Gate::process_with_key`] takes the detector input (step 1) from a
//! separate key signal instead of the processed audio, through a
//! [`KeyDetector`]: an optional key filter ([`Gate::set_key_filter`]) and the
//! linked peak over the key's channels. The key may have a different channel
//! count ([`Gate::set_key_channels`], default: the programme's); everything
//! after step 1 is unchanged. [`Processor::process`] never touches the key
//! path and is bit-identical to the gate without sidechain support.
//!
//! # Expander mode
//!
//! [`Gate::set_mode`] with [`GateMode::Expander`] replaces the fixed closed
//! gain by a downward-expansion curve (Zölzer, *DAFX*, dynamics chapter: an
//! expander is a static curve with slope `ratio` below its threshold). With
//! the detector level `x = 20·log10(e)` dB, `T = open_threshold_db`,
//! `R` = [`Gate::set_expander_ratio`] (1 … 20) and `D = reduction_db`:
//!
//! - below `T` the output level is `T + (x − T)·R`, i.e. the static gain is
//!   `G(x) = max(D, (R − 1)·(x − T))` dB; at and above `T`, `G = 0` dB;
//! - `reduction_db` is the floor (maximum depth) of the expansion.
//!
//! Hysteresis and hold work exactly as in gate mode; only the *closed* gain
//! changes. While the state machine is open (level above `open_threshold`,
//! then held down to `close_threshold` and for `hold` ms) the target gain is 1;
//! while it is closed the target is `10^(G(x)/20)`, evaluated per frame. A
//! rising level therefore follows the curve continuously up to `T` (where
//! `G = 0`), and a falling level keeps unity gain through the hysteresis band
//! and the hold time before the curve takes over. Gate mode is the limit
//! `R → ∞` of this definition. The target goes through the same linear
//! attack/release fade as in gate mode (step 4), so switching modes or ratios
//! while running is click-free. Ratio 1 never attenuates.
//!
//! The curve is evaluated per block with the vectorised, IEEE-exact dB
//! conversions of the compressor, so every SIMD level is bit-identical.

use crate::dynamics::frame::{apply_gains_simd, frame_peaks_simd, BLOCK_FRAMES};
use crate::dynamics::{
    vmath, AttackRelease, GateStateMachine, KeyDetector, KeyFilter, PeakEnvelope,
};
use crate::simd::{detect, Level};
use crate::util::db_to_lin;
use crate::Processor;
use fearless_simd::dispatch;
use fearless_simd::prelude::*;

/// Gate operating mode (see "Expander mode" in the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GateMode {
    /// Classic gate: fixed attenuation `reduction_db` when closed (phase 1 behaviour).
    #[default]
    Gate,
    /// Downward expander below `open_threshold_db` with slope `ratio`, floored
    /// at `reduction_db`.
    Expander,
}

/// Default expander ratio.
pub const DEFAULT_EXPANDER_RATIO: f32 = 2.0;
/// Expander ratio range.
pub const EXPANDER_RATIO_RANGE: (f32, f32) = (1.0, 20.0);

/// Release time constant (ms) of the gate's internal peak detector (attack is instantaneous).
pub const DETECTOR_RELEASE_MS: f64 = 20.0;

/// The gain snaps to its target within this distance (targets are ≥ 1e-4).
const GAIN_SNAP: f64 = 1e-12;

/// Gate parameters (units as in the GStreamer interface).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateParams {
    /// Level (dBFS, −80 … 0) above which the gate opens.
    pub open_threshold_db: f32,
    /// Level (dBFS, −80 … 0) below which the gate starts closing. Treated as
    /// equal to `open_threshold_db` if larger.
    pub close_threshold_db: f32,
    /// Opening fade time (ms, 0.01 … 200).
    pub attack_ms: f32,
    /// Time the gate stays open after the level drops below the close threshold (ms, 0 … 1000).
    pub hold_ms: f32,
    /// Closing fade time (ms, 1 … 2000).
    pub release_ms: f32,
    /// Attenuation when closed (dB, −80 … 0).
    pub reduction_db: f32,
}

impl Default for GateParams {
    fn default() -> Self {
        Self {
            open_threshold_db: -20.0,
            close_threshold_db: -30.0,
            attack_ms: 5.0,
            hold_ms: 50.0,
            release_ms: 100.0,
            reduction_db: -60.0,
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

impl GateParams {
    /// Copy with every field clamped to its documented range (NaN becomes the default).
    pub fn clamped(&self) -> Self {
        let d = Self::default();
        Self {
            open_threshold_db: clamp_or(self.open_threshold_db, -80.0, 0.0, d.open_threshold_db),
            close_threshold_db: clamp_or(self.close_threshold_db, -80.0, 0.0, d.close_threshold_db),
            attack_ms: clamp_or(self.attack_ms, 0.01, 200.0, d.attack_ms),
            hold_ms: clamp_or(self.hold_ms, 0.0, 1000.0, d.hold_ms),
            release_ms: clamp_or(self.release_ms, 1.0, 2000.0, d.release_ms),
            reduction_db: clamp_or(self.reduction_db, -80.0, 0.0, d.reduction_db),
        }
    }
}

/// Linked-channel noise gate. See [`GateParams`] and the module documentation.
#[derive(Debug)]
pub struct Gate {
    params: GateParams,
    sample_rate: f64,
    /// 0 = not configured.
    channels: usize,

    reduction_lin: f64,
    detector: PeakEnvelope,
    state: GateStateMachine,
    /// Linear gain: attack while rising (opening), release while falling.
    fade: AttackRelease,
    /// True from configure/reset until the first frame is processed: parameter
    /// changes apply instantly, including the initial (closed) gain.
    fresh: bool,
    /// Instruction set of the hot loops (detected in `configure`).
    level: Level,
    /// Set by `set_simd_level`: `configure` keeps `level` instead of detecting.
    level_forced: bool,

    /// Sidechain front end, used only by `process_with_key`.
    key: KeyDetector,
    /// Key channel count set by `set_key_channels` (0 = follow `channels`).
    key_channels: usize,
    mode: GateMode,
    /// Expander ratio (clamped).
    ratio: f32,
    /// Expander curve: threshold (dB), slope `R − 1`, depth `−reduction_db` (≥ 0).
    expander: ExpanderCurve,
}

/// The expander's static curve as a reduction in dB (≥ 0):
/// `min(depth, slope·(T − x))` below `T`, 0 at and above.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ExpanderCurve {
    threshold_db: f64,
    slope: f64,
    depth_db: f64,
}

impl Default for Gate {
    fn default() -> Self {
        Self::new()
    }
}

impl Gate {
    /// Create an unconfigured gate with default parameters.
    pub fn new() -> Self {
        let mut g = Self {
            params: GateParams::default(),
            sample_rate: 48_000.0,
            channels: 0,
            reduction_lin: 0.0,
            detector: PeakEnvelope::default(),
            state: GateStateMachine::default(),
            fade: AttackRelease::default().with_snap(GAIN_SNAP),
            fresh: true,
            level: Level::baseline(),
            level_forced: false,
            key: KeyDetector::new(),
            key_channels: 0,
            mode: GateMode::Gate,
            ratio: DEFAULT_EXPANDER_RATIO,
            expander: ExpanderCurve {
                threshold_db: 0.0,
                slope: 0.0,
                depth_db: 0.0,
            },
        };
        g.update_coefficients();
        g.reset();
        g
    }

    /// Update parameters (clamped to their ranges). Never allocates.
    ///
    /// Between `configure`/`reset` and the first processed frame the new
    /// parameters apply instantly (the gate starts closed at the new
    /// `reduction`); afterwards every change is faded.
    pub fn set_params(&mut self, params: &GateParams) {
        self.params = params.clamped();
        self.update_coefficients();
    }

    /// Current (clamped) parameters.
    pub fn params(&self) -> &GateParams {
        &self.params
    }

    /// Linear gain applied to the most recently processed frame (1 = fully open).
    pub fn current_gain(&self) -> f64 {
        self.fade.value()
    }

    /// Whether the gate state machine is currently open (including during hold).
    pub fn is_open(&self) -> bool {
        self.state.is_open()
    }

    /// Force the SIMD instruction set used by `process` (for hosts and tests),
    /// e.g. `Level::fallback()` (needs fearless_simd's `force_support_fallback`
    /// feature) to run the portable scalar path. By default
    /// `configure` detects the best level ([`crate::simd::detect`]); a level
    /// set here is kept across later `configure` calls. Every level produces
    /// bit-identical output. Never allocates.
    pub fn set_simd_level(&mut self, level: Level) {
        self.level = level;
        self.level_forced = true;
        self.key.set_simd_level(level);
    }

    /// The SIMD instruction set `process` currently dispatches to.
    pub fn simd_level(&self) -> Level {
        self.level
    }

    /// Select gate or expander mode (see the module docs). Never allocates;
    /// the change is faded like any other parameter change.
    pub fn set_mode(&mut self, mode: GateMode) {
        self.mode = mode;
    }

    /// Current mode.
    pub fn mode(&self) -> GateMode {
        self.mode
    }

    /// Set the expander ratio (1 … 20, default 2; NaN becomes the default).
    /// Only used in [`GateMode::Expander`]. Never allocates.
    pub fn set_expander_ratio(&mut self, ratio: f32) {
        let (lo, hi) = EXPANDER_RATIO_RANGE;
        self.ratio = clamp_or(ratio, lo, hi, DEFAULT_EXPANDER_RATIO);
        self.update_coefficients();
    }

    /// Current (clamped) expander ratio.
    pub fn expander_ratio(&self) -> f32 {
        self.ratio
    }

    /// Set the key's channel count for [`process_with_key`](Self::process_with_key)
    /// (≥ 1). **Allocates** (key-filter state): call it at setup time, like
    /// `configure`. Without this call the key has the programme's channel count.
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

    /// Process interleaved `samples` in place, with the detector fed from the
    /// interleaved `key` (`key_channels` channels, same rate and frame count)
    /// instead of `samples`. Real-time safe. `key_channels` must equal
    /// [`key_channels`](Self::key_channels); a mismatching or too short key is
    /// detected as silence (see [`KeyDetector`]).
    pub fn process_with_key(&mut self, samples: &mut [f32], key: &[f32], key_channels: usize) {
        let ch = self.channels;
        if ch == 0 || samples.len() < ch {
            return;
        }
        self.fresh = false;
        let key = Some((key, key_channels));
        dispatch!(self.level, simd => self.process_blocks(simd, samples, key));
    }

    fn update_coefficients(&mut self) {
        let p = &self.params;
        let fs = self.sample_rate;
        self.state.set_thresholds(
            db_to_lin(p.open_threshold_db as f64),
            db_to_lin(p.close_threshold_db as f64),
        );
        self.state
            .set_hold_frames((p.hold_ms as f64 * 0.001 * fs).round() as u64);
        self.reduction_lin = db_to_lin(p.reduction_db as f64);
        self.fade
            .set_times(p.attack_ms as f64, p.release_ms as f64, fs);
        self.detector.set_release(DETECTOR_RELEASE_MS, fs);
        self.expander = ExpanderCurve {
            threshold_db: p.open_threshold_db as f64,
            slope: self.ratio as f64 - 1.0,
            depth_db: -(p.reduction_db as f64),
        };
        if self.fresh {
            self.fade.set_value(self.reduction_lin);
        }
    }

    /// Block loop, monomorphised per instruction set: per-frame linked peaks
    /// (SIMD; of the block itself, or of the key), then the serial detector →
    /// state → fade chain turns each frame's peak into its gain (scalar), then
    /// the gains are applied (SIMD).
    #[inline(always)]
    fn process_blocks<S: Simd>(
        &mut self,
        simd: S,
        samples: &mut [f32],
        key: Option<(&[f32], usize)>,
    ) {
        let ch = self.channels;
        let closed = self.reduction_lin;
        let mut gains = [0.0f32; BLOCK_FRAMES];
        for (b, block) in samples.chunks_mut(BLOCK_FRAMES * ch).enumerate() {
            let gains = &mut gains[..block.len() / ch];
            match key {
                None => frame_peaks_simd(simd, block, ch, gains),
                Some((k, kc)) => {
                    let start = (b * BLOCK_FRAMES * kc).min(k.len());
                    self.key.linked_peaks_simd(simd, &k[start..], kc, gains);
                }
            }
            match self.mode {
                GateMode::Gate => {
                    for v in gains.iter_mut() {
                        let env = self.detector.step(*v as f64);
                        let target = if self.state.step(env) { 1.0 } else { closed };
                        *v = self.fade.step(target) as f32;
                    }
                }
                GateMode::Expander => self.expander_block(simd, gains),
            }
            apply_gains_simd(simd, block, ch, gains);
        }
    }

    /// Expander mode for one block: `gains` holds the frames' peaks on input
    /// and their gains on output. Serial detector and state machine, then the
    /// static curve for every frame (SIMD, IEEE-exact), then the serial fade
    /// toward 1 (open) or the curve (closed).
    #[inline(always)]
    fn expander_block<S: Simd>(&mut self, simd: S, gains: &mut [f32]) {
        let frames = gains.len();
        let n = S::f64s::LEN;
        let lanes = frames.div_ceil(n) * n;
        let mut env = [0.0f64; BLOCK_FRAMES];
        let mut open = [false; BLOCK_FRAMES];
        for ((e, o), &p) in env.iter_mut().zip(&mut open).zip(gains.iter()) {
            *e = self.detector.step(p as f64);
            *o = self.state.step(*e);
        }
        let c = self.expander;
        let zero = S::f64s::splat(simd, 0.0);
        let depth = S::f64s::splat(simd, c.depth_db);
        let floor = S::f64s::splat(simd, self.reduction_lin);
        // Open throughout the block: the curve is not needed (the result is
        // the same, unity targets only).
        let curve_lanes = if open[..frames].iter().all(|&o| o) {
            0
        } else {
            lanes
        };
        for v in env[..curve_lanes].chunks_exact_mut(n) {
            let x = vmath::lin_to_db(simd, S::f64s::from_slice(simd, v));
            // Reduction (dB, ≥ 0): slope·(T − x) below T, capped at the depth.
            let y = (S::f64s::splat(simd, c.threshold_db) - x) * c.slope;
            let y = y.simd_gt(zero).select(y, zero);
            let g = vmath::db_to_gain(simd, y);
            y.simd_lt(depth).select(g, floor).store_slice(v);
        }
        for ((v, &g), &o) in gains.iter_mut().zip(&env).zip(&open) {
            let target = if o { 1.0 } else { g };
            *v = self.fade.step(target) as f32;
        }
    }
}

impl Processor for Gate {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        self.sample_rate = (sample_rate as f64).clamp(1.0, 1.0e6);
        self.channels = channels;
        if !self.level_forced {
            self.level = detect();
        }
        let kc = if self.key_channels > 0 {
            self.key_channels
        } else {
            channels
        };
        self.key.configure(sample_rate, kc);
        self.update_coefficients();
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
        self.detector.reset();
        self.state.reset();
        self.fade.set_value(self.reduction_lin);
        self.fresh = true;
    }
}
