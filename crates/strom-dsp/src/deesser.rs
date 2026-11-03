//! De-esser: a linked sibilance detector that lowers either the sibilance band
//! (split-band) or the whole signal (wideband).
//!
//! Built entirely from existing blocks: [`KeyDetector`] (detector filter and
//! linked peak), [`GainComputer`] (static curve), [`AttackRelease`] (gain
//! smoothing in dB) and the Linkwitz-Riley [`Crossover`] (band split).
//!
//! # Signal flow (per frame of interleaved samples)
//!
//! 1. **Detector**: the input is filtered by a second-order RBJ high-pass at
//!    `frequency` (Q = 0.707, Butterworth: −3 dB at `frequency`, 12 dB/octave
//!    below) and the linked peak `p = max_c |hp_c|` is taken over the channels
//!    ([`KeyDetector`] with [`KeyFilter::HighPass`]). Level `x = 20·log10(p)`.
//! 2. **Static curve**: [`GainComputer`] with threshold `T = threshold_db`,
//!    slope 1 (∞:1, the detected band is held at the threshold) and a fixed
//!    soft knee of [`KNEE_DB`] dB, then limited to `max_reduction_db`:
//!    `r = min(max_reduction, curve(x))`.
//! 3. **Smoothing**: branching one-pole on `r` in dB ([`AttackRelease`]):
//!    attack while the reduction rises, release while it falls. After a step,
//!    63.2 % of the step is covered after the attack/release time.
//! 4. **Gain** `g = 10^(−r_s/20)` (exactly 1 when `r_s = 0`).
//! 5. **Apply**: the input is split by a 2-band LR4 [`Crossover`] at
//!    `frequency` into a low band `L` and a high band `H` with `L + H` an
//!    all-pass (flat magnitude, ±0.05 dB tested). The output is
//!    `y = g_L·L + g·H` with
//!    - split-band: `g_L = 1` (only the band above `frequency` is reduced),
//!    - wideband: `g_L = g` (everything is reduced).
//!
//!    Both modes use the same split, so they have the same phase response and
//!    a mode change can be crossfaded: `g_L = 1 + m·(g − 1)`, where the mix
//!    `m` glides between 0 (split-band) and 1 (wideband) with a one-pole of
//!    [`PARAM_SMOOTHING_MS`]. With no reduction the output is the all-pass
//!    `L + H` of the input in either mode.
//!
//! The detector filter (12 dB/octave high-pass) and the reduced band (LR4
//! high band, 24 dB/octave) share the corner frequency; the detector being a
//! little wider is deliberate: the slope of a sibilant just below the corner
//! still counts.
//!
//! # Parameter changes
//!
//! - `threshold_db` and `max_reduction_db` glide with a 10 ms one-pole
//!   ([`PARAM_SMOOTHING_MS`]); attack/release take effect at once (they only
//!   change the smoother's coefficients).
//! - `frequency_hz`: the crossover glides it (10 ms in `ln f`, see
//!   [`crate::filter`]); the detector filter switches at the next frame (it is
//!   on the detector path only, and its output is smoothed afterwards).
//! - `mode`: the mix `m` glides (see above).
//! - After `configure`/`reset`, the first `process` call takes all parameters
//!   immediately.
//!
//! # Denormals
//!
//! The smoothed reduction snaps to its target within
//! [`DENORMAL_FLUSH`](crate::util::DENORMAL_FLUSH); the glides snap within
//! `1e-9`; the filters flush their state (see [`crate::biquad`]).

use crate::dynamics::frame::BLOCK_FRAMES;
use crate::dynamics::{AttackRelease, GainComputer, KeyDetector, KeyFilter, OnePole};
use crate::filter::{Crossover, CrossoverParams, LrOrder, MAX_SPLITS};
use crate::simd::Level;
use crate::util::{db_to_lin, lin_to_db};
use crate::Processor;

/// Time constant (ms) of the threshold, max-reduction and mode glides.
pub const PARAM_SMOOTHING_MS: f64 = 10.0;
/// Width (dB) of the fixed soft knee of the static curve.
pub const KNEE_DB: f64 = 6.0;
/// Range of the detector/split frequency (Hz). It is further limited to
/// `0.45·fs` at the configured rate.
pub const FREQUENCY_RANGE_HZ: (f32, f32) = (1_000.0, 20_000.0);
/// Range of the threshold (dBFS, detector peak level).
pub const THRESHOLD_RANGE_DB: (f32, f32) = (-60.0, 0.0);
/// Range of the maximum reduction (dB, positive = attenuation).
pub const MAX_REDUCTION_RANGE_DB: (f32, f32) = (0.0, 40.0);
/// Range of the attack time (ms).
pub const ATTACK_RANGE_MS: (f32, f32) = (0.01, 200.0);
/// Range of the release time (ms).
pub const RELEASE_RANGE_MS: (f32, f32) = (1.0, 2000.0);

/// Highest frequency as a fraction of the sample rate.
const MAX_FREQUENCY_FRACTION: f64 = 0.45;
/// Snap distance of the parameter glides.
const GLIDE_SNAP: f64 = 1e-9;
/// Butterworth Q of the detector high-pass.
const DETECTOR_Q: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// What the de-esser reduces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeesserMode {
    /// Only the band above `frequency` (LR4 high band) is reduced.
    #[default]
    SplitBand,
    /// The whole signal is reduced.
    Wideband,
}

/// De-esser parameters. Out-of-range values are clamped; NaN falls back to
/// the default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeesserParams {
    /// Detector high-pass corner and split frequency (Hz), see
    /// [`FREQUENCY_RANGE_HZ`].
    pub frequency_hz: f32,
    /// Threshold (dBFS) on the detector's linked peak level.
    pub threshold_db: f32,
    /// Largest gain reduction (dB, ≥ 0).
    pub max_reduction_db: f32,
    /// Attack time constant (ms): reduction rising.
    pub attack_ms: f32,
    /// Release time constant (ms): reduction falling.
    pub release_ms: f32,
    /// Split-band or wideband.
    pub mode: DeesserMode,
}

impl Default for DeesserParams {
    /// Live-speech defaults: 6 kHz, −30 dBFS, at most 10 dB, 1 ms attack,
    /// 50 ms release, split-band.
    fn default() -> Self {
        Self {
            frequency_hz: 6_000.0,
            threshold_db: -30.0,
            max_reduction_db: 10.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            mode: DeesserMode::SplitBand,
        }
    }
}

fn clamp_or(v: f32, range: (f32, f32), fallback: f32) -> f32 {
    if v.is_nan() {
        fallback
    } else {
        v.clamp(range.0, range.1)
    }
}

impl DeesserParams {
    /// Copy with every field clamped to its range (NaN becomes the default).
    pub fn clamped(&self) -> Self {
        let d = Self::default();
        Self {
            frequency_hz: clamp_or(self.frequency_hz, FREQUENCY_RANGE_HZ, d.frequency_hz),
            threshold_db: clamp_or(self.threshold_db, THRESHOLD_RANGE_DB, d.threshold_db),
            max_reduction_db: clamp_or(
                self.max_reduction_db,
                MAX_REDUCTION_RANGE_DB,
                d.max_reduction_db,
            ),
            attack_ms: clamp_or(self.attack_ms, ATTACK_RANGE_MS, d.attack_ms),
            release_ms: clamp_or(self.release_ms, RELEASE_RANGE_MS, d.release_ms),
            mode: self.mode,
        }
    }
}

/// Linked-channel de-esser for interleaved `f32` audio. See the module
/// documentation for the signal flow.
///
/// Implements [`Processor`]: `configure` allocates, `set_params`, `process`
/// and `reset` never do. No latency.
#[derive(Debug)]
pub struct Deesser {
    params: DeesserParams,
    sample_rate: f64,
    /// 0 = not configured.
    channels: usize,
    key: KeyDetector,
    crossover: Crossover,
    /// Low and high band of one block (`BLOCK_FRAMES · channels` each).
    bands: [Vec<f32>; 2],
    reduction: AttackRelease,
    threshold: OnePole,
    max_reduction: OnePole,
    /// 0 = split-band, 1 = wideband.
    mix: OnePole,
    /// Parameters snap on the next `process` (after configure/reset).
    fresh: bool,
}

impl Default for Deesser {
    fn default() -> Self {
        Self::new()
    }
}

impl Deesser {
    /// Create an unconfigured de-esser with default parameters (`process` is
    /// a passthrough until [`Processor::configure`]).
    pub fn new() -> Self {
        let glide = OnePole::default().with_snap(GLIDE_SNAP);
        Self {
            params: DeesserParams::default(),
            sample_rate: 48_000.0,
            channels: 0,
            key: KeyDetector::new(),
            crossover: Crossover::new(),
            bands: [Vec::new(), Vec::new()],
            reduction: AttackRelease::default(),
            threshold: glide,
            max_reduction: glide,
            mix: glide,
            fresh: true,
        }
    }

    /// Update parameters (clamped). Never allocates.
    pub fn set_params(&mut self, params: &DeesserParams) {
        self.params = params.clamped();
        self.apply_params();
    }

    /// Current (clamped) parameters.
    pub fn params(&self) -> &DeesserParams {
        &self.params
    }

    /// Smoothed gain reduction (dB, ≥ 0) after the last processed frame.
    pub fn gain_reduction_db(&self) -> f64 {
        self.reduction.value()
    }

    /// Detector/split frequency (Hz) as realised at the configured rate.
    pub fn effective_frequency_hz(&self) -> f64 {
        (self.params.frequency_hz as f64).min(MAX_FREQUENCY_FRACTION * self.sample_rate)
    }

    /// Force the SIMD level of the detector filter and the crossover (kept
    /// across `configure`). Levels with the same
    /// [`uses_fma`](crate::biquad::uses_fma) give bit-identical output.
    pub fn set_simd_level(&mut self, level: Level) {
        self.key.set_simd_level(level);
        self.crossover.set_simd_level(level);
    }

    fn apply_params(&mut self) {
        let p = self.params;
        let f = self.effective_frequency_hz() as f32;
        self.key.set_filter(KeyFilter::HighPass {
            frequency_hz: f,
            q: DETECTOR_Q,
        });
        self.crossover.set_params(&CrossoverParams {
            order: LrOrder::Lr4,
            num_bands: 2,
            frequencies_hz: [f; MAX_SPLITS],
        });
        self.reduction
            .set_times(p.attack_ms as f64, p.release_ms as f64, self.sample_rate);
    }

    fn mix_target(&self) -> f64 {
        match self.params.mode {
            DeesserMode::SplitBand => 0.0,
            DeesserMode::Wideband => 1.0,
        }
    }
}

impl Processor for Deesser {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        self.sample_rate = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate as f64
        } else {
            48_000.0
        };
        self.channels = channels.max(1);
        self.key.configure(self.sample_rate as f32, self.channels);
        self.crossover
            .configure(self.sample_rate as f32, self.channels);
        self.bands = [
            vec![0.0; BLOCK_FRAMES * self.channels],
            vec![0.0; BLOCK_FRAMES * self.channels],
        ];
        for g in [&mut self.threshold, &mut self.max_reduction, &mut self.mix] {
            g.set_time(PARAM_SMOOTHING_MS, self.sample_rate);
        }
        self.apply_params();
        self.reset();
    }

    fn process(&mut self, samples: &mut [f32]) {
        let ch = self.channels;
        if ch == 0 {
            return;
        }
        if self.fresh {
            self.threshold.set_value(self.params.threshold_db as f64);
            self.max_reduction
                .set_value(self.params.max_reduction_db as f64);
            self.mix.set_value(self.mix_target());
            self.fresh = false;
        }
        let t_target = self.params.threshold_db as f64;
        let m_target = self.params.max_reduction_db as f64;
        let mix_target = self.mix_target();
        let frames = samples.len() / ch;
        let mut peaks = [0.0f32; BLOCK_FRAMES];
        for block in samples[..frames * ch].chunks_mut(BLOCK_FRAMES * ch) {
            let n = block.len() / ch;
            self.key.linked_peaks(block, ch, &mut peaks[..n]);
            self.crossover.process(block, &mut self.bands);
            let [low, high] = &self.bands;
            let settled = self.threshold.value() == t_target
                && self.max_reduction.value() == m_target
                && self.mix.value() == mix_target;
            let mut gc = GainComputer {
                threshold_db: t_target,
                knee_db: KNEE_DB,
                slope: 1.0,
            };
            // Below this linear level the static reduction is exactly 0.
            let quiet_below = if settled { gc.knee_start_lin() } else { -1.0 };
            for (i, &p) in peaks[..n].iter().enumerate() {
                let (max_red, mix) = if settled {
                    (m_target, mix_target)
                } else {
                    gc.threshold_db = self.threshold.step(t_target);
                    (self.max_reduction.step(m_target), self.mix.step(mix_target))
                };
                let target = if (p as f64) <= quiet_below {
                    0.0
                } else {
                    gc.reduction_db(lin_to_db(p as f64)).min(max_red)
                };
                let r = self.reduction.step(target);
                let g = if r == 0.0 { 1.0 } else { db_to_lin(-r) };
                let g_low = 1.0 + mix * (g - 1.0);
                let o = i * ch;
                for c in o..o + ch {
                    block[c] = (g_low * low[c] as f64 + g * high[c] as f64) as f32;
                }
            }
        }
    }

    fn reset(&mut self) {
        self.key.reset();
        self.crossover.reset();
        self.reduction.set_value(0.0);
        self.fresh = true;
    }
}
