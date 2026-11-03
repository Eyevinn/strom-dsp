//! Wideband loudness-targeting AGC: a slow, gated,
//! rate-limited gain rider driven by the BS.1770 loudness detector
//! ([`LoudnessDetector`]).
//!
//! # Design
//!
//! The AGC is **feed-forward**: it measures the loudness of its *input* and
//! moves one linked gain (all channels) toward `target − measured`. The
//! output loudness is `measured + gain`, and since the measurement does not
//! depend on the gain there is no loop that could oscillate.
//!
//! **Detector.** The K-weighted, channel-weighted energy of every 10 ms
//! sub-block comes from [`LoudnessDetector`] (BS.1770 eq. 2). The AGC
//! averages it over its own window (`window_ms`, default 3 s, the EBU Tech
//! 3341 short-term length), **gated** as below.
//!
//! **Gate (freeze).** Each sub-block is *active* when the momentary loudness
//! (the 400 ms ending with it) is at or above `gate_lufs`. This is the
//! absolute gating of BS.1770-4 (Annex 1, section 5.1: 400 ms blocks below a
//! fixed threshold are left out of the mean), with the AGC gate as the
//! threshold and evaluated every 10 ms. The measured loudness is the mean
//! energy of the **active** sub-blocks in the window only, so long pauses and
//! room tone neither dilute it nor count as programme, while short gaps
//! inside programme (shorter than a 400 ms block) count as BS.1770 would
//! count them, so the AGC lands on the BS.1770 integrated loudness of gappy
//! material such as speech. The gain **holds** (keeps its value; it does not
//! run up) when
//!
//! - the newest 100 ms ([`HOLD_BLOCK_SUB_BLOCKS`]) is below the gate (a
//!   pause or gap has started: freeze at once, before the silent sub-blocks
//!   that are still active can lower the measurement), or
//! - the window holds less than [`MIN_ACTIVE_MS`] (400 ms, one momentary
//!   window) of active sub-blocks (too little programme to judge).
//!
//! **Recovery after a pause.** During the pause the gain is frozen. When the
//! programme returns, the gain still holds until 400 ms of it has been
//! measured; from then on the measurement is the gated mean of the returning
//! programme (the pause is not in it, apart from at most the 400 ms tail at
//! its start), so the gain moves straight to the correct value,
//! monotonically, at the rate limit. It never runs past its target (rate
//! limiting stops exactly at the target), so there is no overshoot.
//! Programme from before a pause that is still in the window (a pause
//! shorter than the window) counts with its real level, which is what an
//! operator expects.
//!
//! **Gain law.** On every sub-block the target gain is
//! `clamp(target_lufs − measured, −max_cut_db, +max_gain_db)` (while not
//! holding). The applied gain in dB moves toward it linearly at most at
//! `rate_up_db_per_s` (gain rising) or `rate_down_db_per_s` (gain falling),
//! per sample; inside a ramp the linear gain is advanced by a constant
//! per-sample factor `10^(±step/20)` and re-anchored to `10^(gain_db/20)` at
//! the end of every segment, so it cannot drift. Down is faster than up by
//! default (3 vs 1 dB/s): a loud change is corrected sooner than a quiet one
//! is lifted.
//!
//! **Convergence time.** After a reset or a pause, programme at loudness `L`
//! that needs a gain change `Δ` is within 1 dB of target after at most
//!
//! ```text
//! T = 0.4 s · 10^((gate − L)/10)  +  0.41 s  +  (|Δ| − 1 dB) / rate
//! ```
//!
//! (`rate` = up or down rate for the direction): the first term is the time
//! until the momentary loudness of the programme (400 ms window, still
//! partly silent) passes the gate, at most 400 ms; then 400 ms of gated
//! programme plus one 10 ms sub-block; then the rate-limited ramp. With the
//! defaults (gate −50 LUFS, 1 / 3 dB/s), programme at −35 LUFS (+12 dB) takes
//! ≤ 11.4 s and at −11 LUFS (−12 dB) ≤ 4.1 s. A level change *within* continuous programme is also
//! delayed by the window: the gated mean crosses the midpoint after about
//! half the window. `tests/agc.rs` checks the bound for pink noise from −40
//! to −6 LUFS.
//!
//! **Parameter changes** are click-free by construction: target, bounds and
//! gate only change the target gain, and the applied gain always moves at the
//! rate limit. After a bound is lowered the gain glides into the new range at
//! the rate limit (also while holding). Changing the window recomputes the
//! window sums (O(window), allocation-free).
//!
//! **Cost** (`benches/agc.rs`, 48 kHz stereo, 480-frame blocks, Apple M5
//! Pro): the whole AGC takes about 1.7 µs per block (0.017 % of one core),
//! of which the detector is 1.3 µs. Feeding and reading the `ebur128`
//! wrapper every block instead costs about 230 µs.
//!
//! **Numerics.** `f64` state; a gain of exactly 0 dB is bit-exact
//! passthrough; output samples are limited to the finite `f32` range;
//! non-finite input is ignored by the detector (it reads as silence) and
//! cannot reach the gain.

use crate::meter::{
    lufs_from_mean_square, ChannelLayout, ChannelRole, History, LoudnessDetector, MeterError,
    WindowSum, SILENCE_LUFS, SUB_BLOCK_MS,
};
use crate::util::db_to_lin;
use crate::Processor;

/// Sub-blocks (of 10 ms) whose ungated loudness must be above the gate for
/// the gain to move: 100 ms. Below it the gain freezes at once.
pub const HOLD_BLOCK_SUB_BLOCKS: usize = 10;

/// Minimum active (above-gate) programme in the window before the gain may
/// move, in ms.
pub const MIN_ACTIVE_MS: f64 = 400.0;

/// Range of [`AgcParams::window_ms`] (ms).
pub const AGC_WINDOW_RANGE_MS: (f32, f32) = (400.0, 10_000.0);

/// Parameters of the [`Agc`]. Units follow the broadcast element interface
/// (LUFS, dB, dB/s, ms). Out-of-range values are clamped; NaN means default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgcParams {
    /// Target loudness (LUFS, −40 … −5). Default −23 (EBU R 128).
    pub target_lufs: f32,
    /// Largest gain the AGC may apply (dB, 0 … 30). Default 12.
    pub max_gain_db: f32,
    /// Largest cut the AGC may apply (dB, 0 … 30, a positive number).
    /// Default 12.
    pub max_cut_db: f32,
    /// Gain slew rate while rising (dB/s, 0.1 … 20). Default 1.
    pub rate_up_db_per_s: f32,
    /// Gain slew rate while falling (dB/s, 0.1 … 40). Default 3.
    pub rate_down_db_per_s: f32,
    /// Gate (LUFS, −80 … −20): below it the gain holds. Default −50.
    pub gate_lufs: f32,
    /// Detector window (ms, 400 … 10 000). Default 3000 (short-term).
    pub window_ms: f32,
}

impl Default for AgcParams {
    fn default() -> Self {
        Self {
            target_lufs: -23.0,
            max_gain_db: 12.0,
            max_cut_db: 12.0,
            rate_up_db_per_s: 1.0,
            rate_down_db_per_s: 3.0,
            gate_lufs: -50.0,
            window_ms: 3000.0,
        }
    }
}

fn clamp_or(v: f32, lo: f32, hi: f32, fallback: f32) -> f32 {
    if v.is_nan() {
        fallback
    } else {
        v.clamp(lo, hi)
    }
}

impl AgcParams {
    /// Copy with every field clamped to its documented range (NaN becomes the
    /// default).
    pub fn clamped(&self) -> Self {
        let d = Self::default();
        Self {
            target_lufs: clamp_or(self.target_lufs, -40.0, -5.0, d.target_lufs),
            max_gain_db: clamp_or(self.max_gain_db, 0.0, 30.0, d.max_gain_db),
            max_cut_db: clamp_or(self.max_cut_db, 0.0, 30.0, d.max_cut_db),
            rate_up_db_per_s: clamp_or(self.rate_up_db_per_s, 0.1, 20.0, d.rate_up_db_per_s),
            rate_down_db_per_s: clamp_or(self.rate_down_db_per_s, 0.1, 40.0, d.rate_down_db_per_s),
            gate_lufs: clamp_or(self.gate_lufs, -80.0, -20.0, d.gate_lufs),
            window_ms: clamp_or(
                self.window_ms,
                AGC_WINDOW_RANGE_MS.0,
                AGC_WINDOW_RANGE_MS.1,
                d.window_ms,
            ),
        }
    }
}

/// Number of 10 ms sub-blocks in `window_ms`.
fn window_sub_blocks(window_ms: f32) -> usize {
    (f64::from(window_ms) / SUB_BLOCK_MS).round().max(1.0) as usize
}

/// Wideband loudness-targeting AGC. See the module docs for the design.
///
/// Follows the crate's [`Processor`] contract: `new`, `configure` (the only
/// allocation), `set_params` (any time, allocation-free), `process` in place
/// on interleaved frames, `reset`. Latency 0.
#[derive(Debug, Clone)]
pub struct Agc {
    params: AgcParams,
    detector: LoudnessDetector,
    configured: bool,
    sample_rate: f64,
    channels: usize,
    /// Energy of each sub-block if it was active, else 0.
    active_energy: History,
    /// 1 for each active sub-block, else 0 (sums of these are exact).
    active_flag: History,
    energy_sum: WindowSum,
    count_sum: WindowSum,
    min_active: f64,
    gain_db: f64,
    gain_lin: f64,
    target_gain_db: f64,
    measured_lufs: f64,
    holding: bool,
}

impl Default for Agc {
    fn default() -> Self {
        Self::new()
    }
}

impl Agc {
    /// An unconfigured AGC with default parameters (`process` is a no-op).
    pub fn new() -> Self {
        let params = AgcParams::default();
        let n = window_sub_blocks(params.window_ms);
        Self {
            params,
            detector: LoudnessDetector::new(),
            configured: false,
            sample_rate: 48_000.0,
            channels: 1,
            active_energy: History::default(),
            active_flag: History::default(),
            energy_sum: WindowSum::new(n),
            count_sum: WindowSum::new(n),
            min_active: MIN_ACTIVE_MS / SUB_BLOCK_MS,
            gain_db: 0.0,
            gain_lin: 1.0,
            target_gain_db: 0.0,
            measured_lufs: SILENCE_LUFS,
            holding: true,
        }
    }

    /// Like [`Processor::configure`], with the channel layout that sets the
    /// BS.1770 channel weights of the detector.
    pub fn configure_with_layout(
        &mut self,
        sample_rate: f32,
        channels: usize,
        layout: ChannelLayout,
    ) {
        self.detector.configure(sample_rate, channels, layout);
        self.sample_rate = self.detector.sample_rate();
        self.channels = self.detector.channels();
        let cap = window_sub_blocks(AGC_WINDOW_RANGE_MS.1) + 1;
        self.active_energy.allocate(cap);
        self.active_flag.allocate(cap);
        self.configured = true;
        self.reset();
    }

    /// Change the detector's channel weights (allocation-free).
    pub fn set_channel_layout(&mut self, layout: ChannelLayout) {
        self.detector.set_layout(layout);
    }

    /// Override the detector's per-channel roles (allocation-free).
    pub fn set_channel_roles(&mut self, roles: &[ChannelRole]) -> Result<(), MeterError> {
        self.detector.set_channel_roles(roles)
    }

    /// Set the parameters (clamped). Allocation-free, click-free, any time.
    pub fn set_params(&mut self, params: &AgcParams) {
        let p = params.clamped();
        let window_changed = window_sub_blocks(p.window_ms) != self.energy_sum.len();
        self.params = p;
        if window_changed {
            let n = window_sub_blocks(p.window_ms);
            self.energy_sum.set_len(n, &self.active_energy);
            self.count_sum.set_len(n, &self.active_flag);
        }
        if self.configured {
            self.update_target();
        }
    }

    /// Current (clamped) parameters.
    pub fn params(&self) -> &AgcParams {
        &self.params
    }

    /// The input loudness detector (momentary and short-term readings of the
    /// AGC input, for metering).
    pub fn detector(&self) -> &LoudnessDetector {
        &self.detector
    }

    /// Gain currently applied, in dB (positive = boost).
    pub fn gain_db(&self) -> f64 {
        self.gain_db
    }

    /// Gain the AGC is moving toward, in dB.
    pub fn target_gain_db(&self) -> f64 {
        self.target_gain_db
    }

    /// Gated input loudness over the AGC window (LUFS; [`SILENCE_LUFS`] when
    /// no sub-block in the window is above the gate).
    pub fn measured_lufs(&self) -> f64 {
        self.measured_lufs
    }

    /// Whether the gain is currently frozen by the gate.
    pub fn is_holding(&self) -> bool {
        self.holding
    }

    fn rate_step(&self, rising: bool) -> f64 {
        let rate = if rising {
            self.params.rate_up_db_per_s
        } else {
            self.params.rate_down_db_per_s
        };
        f64::from(rate) / self.sample_rate
    }

    /// Record the sub-block the detector just completed and recompute the
    /// target gain.
    fn on_sub_block(&mut self) {
        let gate = f64::from(self.params.gate_lufs);
        // Measurement gating on the 400 ms block (BS.1770 block length).
        let active = self.detector.momentary_lufs() >= gate;
        // Freeze as soon as the newest 100 ms falls below the gate.
        let present =
            lufs_from_mean_square(self.detector.recent_mean_square(HOLD_BLOCK_SUB_BLOCKS)) >= gate;
        let e = self.detector.last_sub_block_mean_square();
        self.active_energy.push(if active { e } else { 0.0 });
        self.active_flag.push(if active { 1.0 } else { 0.0 });
        self.energy_sum.pushed(&self.active_energy);
        self.count_sum.pushed(&self.active_flag);
        let count = self.count_sum.sum().round();
        self.measured_lufs = if count > 0.0 {
            lufs_from_mean_square(self.energy_sum.sum() / count)
        } else {
            SILENCE_LUFS
        };
        self.holding = !present || count < self.min_active;
        self.update_target();
    }

    fn update_target(&mut self) {
        let max_gain = f64::from(self.params.max_gain_db);
        let max_cut = f64::from(self.params.max_cut_db);
        let want = if self.holding {
            self.gain_db
        } else {
            f64::from(self.params.target_lufs) - self.measured_lufs
        };
        self.target_gain_db = want.clamp(-max_cut, max_gain);
    }

    /// Apply the gain (ramping toward the target) to whole frames in place.
    fn apply_gain(&mut self, seg: &mut [f32]) {
        let ch = self.channels;
        let frames = seg.len() / ch;
        let diff = self.target_gain_db - self.gain_db;
        let mut start = 0;
        if diff != 0.0 {
            let rising = diff > 0.0;
            let step = self.rate_step(rising);
            let need = (diff.abs() / step).ceil();
            let ramp = if need >= frames as f64 {
                frames
            } else {
                need as usize
            };
            let factor = db_to_lin(if rising { step } else { -step });
            let reaches = ramp as f64 >= need;
            let mut g = self.gain_lin;
            let end_lin = db_to_lin(self.target_gain_db);
            for (i, frame) in seg.chunks_exact_mut(ch).take(ramp).enumerate() {
                g = if reaches && i + 1 == ramp {
                    end_lin
                } else {
                    g * factor
                };
                scale(frame, g);
            }
            self.gain_db = if reaches {
                self.target_gain_db
            } else {
                self.gain_db + if rising { step } else { -step } * ramp as f64
            };
            self.gain_lin = db_to_lin(self.gain_db);
            start = ramp;
        }
        if self.gain_db != 0.0 && start < frames {
            let g = self.gain_lin;
            scale(&mut seg[start * ch..frames * ch], g);
        }
    }
}

/// Multiply samples by `g`, limited to the finite `f32` range.
#[inline]
fn scale(samples: &mut [f32], g: f64) {
    for v in samples {
        *v = (f64::from(*v) * g).clamp(-f64::from(f32::MAX), f64::from(f32::MAX)) as f32;
    }
}

impl Processor for Agc {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        self.configure_with_layout(sample_rate, channels, ChannelLayout::Unknown);
    }

    fn process(&mut self, samples: &mut [f32]) {
        if !self.configured {
            return;
        }
        let ch = self.channels;
        let frames = samples.len() / ch;
        let mut pos = 0;
        while pos < frames {
            let n = (frames - pos).min(self.detector.frames_to_sub_block_end());
            let seg = &mut samples[pos * ch..(pos + n) * ch];
            // Measure the input, then apply the gain decided so far.
            let done = self.detector.process(seg) > 0;
            self.apply_gain(seg);
            if done {
                self.on_sub_block();
            }
            pos += n;
        }
    }

    /// Clear the detector and the gated window, and return the gain to 0 dB.
    fn reset(&mut self) {
        self.detector.reset();
        self.active_energy.clear();
        self.active_flag.clear();
        let n = window_sub_blocks(self.params.window_ms);
        self.energy_sum.set_len(n, &self.active_energy);
        self.count_sum.set_len(n, &self.active_flag);
        self.gain_db = 0.0;
        self.gain_lin = 1.0;
        self.target_gain_db = 0.0;
        self.measured_lufs = SILENCE_LUFS;
        self.holding = true;
    }
}
