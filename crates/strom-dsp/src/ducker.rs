//! Ducker: lowers the programme (e.g. a music bed) while a key (e.g. speech)
//! is present.
//!
//! The programme is processed in place; the key is a separate, read-only
//! signal that may have its own sample rate and channel count
//! ([`Ducker::set_key_format`]).
//!
//! # Signal flow
//!
//! 1. **Key detector, at the key's own rate.** A [`KeyDetector`] (optional
//!    key high-pass, RBJ cookbook HPF with Q 0.707, then the linked peak over
//!    the key's channels) feeds a sine-calibrated RMS envelope
//!    ([`MeanSquareEnvelope`], τ = [`DETECTOR_RMS_MS`], the gate's detector):
//!    a steady sine reads its peak level in dBFS, and single peaks do not
//!    trigger it. The key level is `L = 10·log10(P)` dB, `P` the envelope's
//!    sine-calibrated power.
//! 2. **Envelope rate conversion** (see [`crate::resample`] for why the
//!    envelope is converted rather than the key audio). The programme is
//!    processed in blocks of up to [`BLOCK_FRAMES`] frames. The key frames
//!    covering the same time span as each block are run through the detector,
//!    and `P` is interpolated linearly across the block's frames, from its
//!    value at the end of the previous block to its value at the end of this
//!    one.
//! 3. **Static curve** (reduction `c ≥ 0` dB, with `T` = threshold and
//!    `D = −depth ≥ 0`), evaluated per programme frame:
//!    - [`RangeMode::Fixed`]: `c = D` while `L > T`, else 0.
//!    - [`RangeMode::Proportional`]: `c = min(D, L − T)` while `L > T`, else 0.
//!      One dB of reduction per dB the key is above the threshold, capped at
//!      the depth. At `L = T` the curve is continuous (0).
//! 4. **Hold** (peak hold with a timer): the target `h` follows `c` upward
//!    instantly and re-arms a timer of `hold` ms whenever `c ≥ h`. While
//!    `c < h` the timer runs down, and when it has run out `h` drops to `c`.
//!    In fixed mode this is exactly "stay ducked for `hold` ms after the key
//!    falls below the threshold". In proportional mode it holds the deepest
//!    recent reduction for `hold` ms, so the bed does not pump between words.
//!    `h` is also capped at `D`, so a smaller depth applies at once.
//! 5. **Fades** (linear in dB): the reduction `r` moves toward `h` at a
//!    constant rate, `S / attack` while ducking and `S / release` while
//!    recovering, with `S = max(D, 1 dB)`. So **attack is the time to go from
//!    unity to full depth, release the time to come back**, and smaller
//!    (proportional) targets are reached sooner. The 1 dB floor of `S` makes
//!    sure a residual reduction always recovers, even at depth 0.
//! 6. The programme is multiplied by `10^(−r/20)`, the same gain on every
//!    channel.
//!
//! Why linear-in-dB fades and not the one-pole of the other dynamics blocks:
//! a one-pole never *reaches* its target, while a ducker is specified (and
//! operated) in fade times: "duck by 12 dB in 20 ms, recover in 500 ms". A
//! constant dB rate is the fader move an operator would make.
//!
//! Metering: [`Ducker::reduction_db`] (≤ 0, the gain applied to the last
//! frame), [`Ducker::key_active`] (key above threshold, or within `hold` ms of
//! it) and [`Ducker::key_level_db`].
//!
//! # Missing key
//!
//! [`Processor::process`] (no key) and [`Ducker::process_with_key`] with a
//! zero channel count treat the key as **silence** for the programme's
//! duration: the detector decays, the hold runs out and the programme
//! recovers to unity with the release fade. Missing frames at the end of a
//! key slice are not silence: the key slice always spans the programme's
//! time span, whatever its length (see `process_with_key`). A host that knows
//! a key stalled pads the missing part with zeros.
//!
//! # Real-time safety, denormals, SIMD
//!
//! [`Processor::configure`] and [`Ducker::set_key_format`] allocate; every
//! other method is allocation-free. The envelope is denormal-flushed, and the
//! reduction reaches its target exactly (it is clamped, not approached).
//! Key peaks and gain application use the `fearless_simd` kernels of
//! [`crate::dynamics::frame`]; everything else is scalar. With the key filter
//! off, every SIMD level is bit-identical; with it on, levels with the same
//! FMA policy are (see [`KeyDetector`]).

use fearless_simd::{dispatch, Simd};

use crate::dynamics::frame::{apply_gains_simd, BLOCK_FRAMES};
use crate::dynamics::{KeyDetector, KeyFilter, MeanSquareEnvelope};
use crate::gate::DETECTOR_RMS_MS;
use crate::resample::{split_point, FrameRatio};
use crate::simd::{detect, Level};
use crate::util::{db_to_lin, lin_to_db};
use crate::Processor;

/// How the reduction depends on the key level (see the module docs, step 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RangeMode {
    /// Full `depth` whenever the key is above the threshold.
    #[default]
    Fixed,
    /// 1 dB of reduction per dB the key is above the threshold, up to `depth`.
    Proportional,
}

/// Threshold range (dB, key level).
pub const THRESHOLD_RANGE_DB: (f32, f32) = (-80.0, 0.0);
/// Depth range (dB, gain at full ducking).
pub const DEPTH_RANGE_DB: (f32, f32) = (-80.0, 0.0);
/// Attack range (ms, unity to full depth).
pub const ATTACK_RANGE_MS: (f32, f32) = (0.1, 2000.0);
/// Hold range (ms).
pub const HOLD_RANGE_MS: (f32, f32) = (0.0, 5000.0);
/// Release range (ms, full depth back to unity).
pub const RELEASE_RANGE_MS: (f32, f32) = (1.0, 10_000.0);
/// Key high-pass range (Hz, 0 = off). Values between 0 and
/// [`KEY_FILTER_MIN_HZ`](crate::dynamics::KEY_FILTER_MIN_HZ) act as that
/// minimum.
pub const KEY_HIGHPASS_RANGE_HZ: (f32, f32) = (0.0, 20_000.0);
/// Smallest fade span (dB): the fade rates are `max(depth, this) / time`.
pub const MIN_FADE_SPAN_DB: f64 = 1.0;
/// Largest detector power used (+1000 dB): keeps the interpolation finite.
const MAX_POWER: f64 = 1e100;
/// Q of the key high-pass (Butterworth).
pub const KEY_HIGHPASS_Q: f32 = 0.707;

/// Ducker parameters. Defaults suit a music bed ducked under live speech.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DuckerParams {
    /// Key level (dBFS, sine-calibrated RMS, −80 … 0) above which the
    /// programme is ducked. Default −30.
    pub threshold_db: f32,
    /// Programme gain at full ducking (dB, −80 … 0). Default −12.
    pub depth_db: f32,
    /// Fade time from unity to full depth (ms, 0.1 … 2000). Default 20.
    pub attack_ms: f32,
    /// Time the reduction is held after the key drops (ms, 0 … 5000). Default 300.
    pub hold_ms: f32,
    /// Fade time from full depth back to unity (ms, 1 … 10 000). Default 500.
    pub release_ms: f32,
    /// Key high-pass corner (Hz, 0 = off, up to 20 000). Default 100: keeps
    /// rumble and plosives in a speech key from triggering.
    pub key_highpass_hz: f32,
    /// Fixed or proportional reduction. Default fixed.
    pub range_mode: RangeMode,
}

impl Default for DuckerParams {
    fn default() -> Self {
        Self {
            threshold_db: -30.0,
            depth_db: -12.0,
            attack_ms: 20.0,
            hold_ms: 300.0,
            release_ms: 500.0,
            key_highpass_hz: 100.0,
            range_mode: RangeMode::Fixed,
        }
    }
}

/// Clamp `v` to `range`, replacing NaN with `fallback`.
fn clamp_or(v: f32, range: (f32, f32), fallback: f32) -> f32 {
    if v.is_nan() {
        fallback
    } else {
        v.clamp(range.0, range.1)
    }
}

impl DuckerParams {
    /// Copy with every field clamped to its documented range (NaN becomes the
    /// default).
    pub fn clamped(&self) -> Self {
        let d = Self::default();
        Self {
            threshold_db: clamp_or(self.threshold_db, THRESHOLD_RANGE_DB, d.threshold_db),
            depth_db: clamp_or(self.depth_db, DEPTH_RANGE_DB, d.depth_db),
            attack_ms: clamp_or(self.attack_ms, ATTACK_RANGE_MS, d.attack_ms),
            hold_ms: clamp_or(self.hold_ms, HOLD_RANGE_MS, d.hold_ms),
            release_ms: clamp_or(self.release_ms, RELEASE_RANGE_MS, d.release_ms),
            key_highpass_hz: clamp_or(
                self.key_highpass_hz,
                KEY_HIGHPASS_RANGE_HZ,
                d.key_highpass_hz,
            ),
            range_mode: self.range_mode,
        }
    }

    /// The key filter these parameters select.
    pub fn key_filter(&self) -> KeyFilter {
        if self.key_highpass_hz > 0.0 {
            KeyFilter::HighPass {
                frequency_hz: self.key_highpass_hz,
                q: KEY_HIGHPASS_Q,
            }
        } else {
            KeyFilter::Off
        }
    }
}

/// Keyed ducker with linked channels. See [`DuckerParams`] and the module
/// documentation.
#[derive(Debug)]
pub struct Ducker {
    params: DuckerParams,
    sample_rate: f64,
    /// 0 = not configured.
    channels: usize,
    /// Key format from `set_key_format` (0 = follow the programme).
    key_rate_set: f32,
    key_channels_set: usize,
    /// Effective key rate (Hz).
    key_rate: f64,

    key: KeyDetector,
    env: MeanSquareEnvelope,
    /// Envelope power at the end of the last block.
    power: f64,
    /// Programme frames → key frames, for the no-key (silence) path.
    no_key: FrameRatio,

    // Derived from params.
    threshold_power: f64,
    threshold_db: f64,
    /// `−depth_db` (≥ 0).
    depth: f64,
    /// Fade steps in dB per frame.
    up: f64,
    down: f64,
    hold_frames: u64,

    // State.
    /// Held target reduction (dB, ≥ 0).
    held: f64,
    peak_left: u64,
    hold_left: u64,
    active: bool,
    /// Current reduction (dB, ≥ 0).
    reduction: f64,
    /// Linear gain for `gain_for` (cache, so steady states skip `powf`).
    gain: f64,
    gain_for: f64,

    level: Level,
    level_forced: bool,
}

impl Default for Ducker {
    fn default() -> Self {
        Self::new()
    }
}

impl Ducker {
    /// Unconfigured ducker with default parameters (a passthrough until
    /// [`Processor::configure`]). Does not allocate.
    pub fn new() -> Self {
        let params = DuckerParams::default();
        let mut d = Self {
            params,
            sample_rate: 48_000.0,
            channels: 0,
            key_rate_set: 0.0,
            key_channels_set: 0,
            key_rate: 48_000.0,
            key: KeyDetector::new(),
            env: MeanSquareEnvelope::default(),
            power: 0.0,
            no_key: FrameRatio::default(),
            threshold_power: 0.0,
            threshold_db: 0.0,
            depth: 0.0,
            up: 0.0,
            down: 0.0,
            hold_frames: 0,
            held: 0.0,
            peak_left: 0,
            hold_left: 0,
            active: false,
            reduction: 0.0,
            gain: 1.0,
            gain_for: 0.0,
            level: Level::baseline(),
            level_forced: false,
        };
        d.key.set_filter(params.key_filter());
        d.update_coefficients();
        d
    }

    /// Update parameters (clamped). Never allocates; every change is faded
    /// (the reduction only ever moves at the attack/release rates).
    pub fn set_params(&mut self, params: &DuckerParams) {
        let p = params.clamped();
        if p.key_filter() != self.params.key_filter() {
            self.key.set_filter(p.key_filter());
        }
        self.params = p;
        self.update_coefficients();
    }

    /// Current (clamped) parameters.
    pub fn params(&self) -> &DuckerParams {
        &self.params
    }

    /// Set the key's sample rate (Hz) and channel count (≥ 1). **Allocates**
    /// (key-filter state): call it at setup time, like `configure`. Clears the
    /// key detector, not the reduction (which then recovers smoothly if the
    /// key is gone). Without this call the key has the programme's format.
    pub fn set_key_format(&mut self, key_rate: f32, key_channels: usize) {
        self.key_rate_set = if key_rate.is_finite() {
            key_rate.clamp(1.0, 1.0e6)
        } else {
            0.0
        };
        self.key_channels_set = key_channels.max(1);
        self.configure_key();
    }

    /// Effective key sample rate (Hz).
    pub fn key_rate(&self) -> f64 {
        self.key_rate
    }

    /// Key channel count expected by [`process_with_key`](Self::process_with_key).
    pub fn key_channels(&self) -> usize {
        self.key.key_channels()
    }

    /// Gain applied to the last processed frame, in dB (≤ 0; 0 = unducked).
    pub fn reduction_db(&self) -> f64 {
        0.0 - self.reduction
    }

    /// Linear gain applied to the last processed frame.
    pub fn current_gain(&self) -> f64 {
        db_to_lin(-self.reduction)
    }

    /// Whether the key is above the threshold, or was within the last `hold` ms.
    pub fn key_active(&self) -> bool {
        self.active
    }

    /// Detected key level (dBFS, sine-calibrated RMS after the key filter;
    /// −200 for silence).
    pub fn key_level_db(&self) -> f64 {
        lin_to_db(self.power.sqrt())
    }

    /// Force the SIMD level (kept across `configure`), e.g.
    /// `Level::fallback()` in tests. Never allocates.
    pub fn set_simd_level(&mut self, level: Level) {
        self.level = level;
        self.level_forced = true;
        self.key.set_simd_level(level);
    }

    /// The SIMD level `process` dispatches to.
    pub fn simd_level(&self) -> Level {
        self.level
    }

    /// Duck the interleaved programme `samples` in place, driven by the
    /// interleaved `key` (`key_channels` channels at [`key_rate`](Self::key_rate)).
    ///
    /// The key slice spans **the same time** as `samples`, whatever its
    /// length: with equal rates that is the same frame count, at 16 kHz
    /// against a 48 kHz programme a third of it, and so on. It is split
    /// across the programme proportionally ([`split_point`]). A slice that
    /// is shorter than the nominal span (e.g. a rounding frame) just
    /// stretches slightly; a host that knows part of the key is missing pads
    /// it with zeros. `key_channels` must equal [`key_channels`](Self::key_channels)
    /// (otherwise the key reads as silence, see [`KeyDetector`]); 0 means "no
    /// key", as [`Processor::process`]. Real-time safe.
    pub fn process_with_key(&mut self, samples: &mut [f32], key: &[f32], key_channels: usize) {
        let ch = self.channels;
        if ch == 0 || samples.len() < ch {
            return;
        }
        let key = (key_channels > 0).then_some((key, key_channels));
        dispatch!(self.level, simd => self.process_blocks(simd, samples, key));
    }

    fn configure_key(&mut self) {
        self.key_rate = if self.key_rate_set > 0.0 {
            self.key_rate_set as f64
        } else {
            self.sample_rate
        };
        let kc = if self.key_channels_set > 0 {
            self.key_channels_set
        } else {
            self.channels.max(1)
        };
        self.key.configure(self.key_rate as f32, kc);
        self.env.set_time(DETECTOR_RMS_MS, self.key_rate);
        self.env.reset();
        self.power = 0.0;
        self.no_key = FrameRatio::new(self.sample_rate, self.key_rate);
    }

    fn update_coefficients(&mut self) {
        let p = &self.params;
        let fs = self.sample_rate;
        self.threshold_db = p.threshold_db as f64;
        // The detector delivers sine-calibrated power: compare with 10^(T/10).
        self.threshold_power = db_to_lin(2.0 * self.threshold_db);
        self.depth = -(p.depth_db as f64);
        let span = self.depth.max(MIN_FADE_SPAN_DB);
        self.up = span / (p.attack_ms as f64 * 0.001 * fs).max(1.0);
        self.down = span / (p.release_ms as f64 * 0.001 * fs).max(1.0);
        self.hold_frames = (p.hold_ms as f64 * 0.001 * fs).round() as u64;
        self.hold_left = self.hold_left.min(self.hold_frames);
        self.peak_left = self.peak_left.min(self.hold_frames);
        self.held = self.held.min(self.depth);
    }

    /// One programme frame: detector power `p` in, linear gain out
    /// (steps 3–6 of the module docs).
    #[inline(always)]
    fn step_frame(&mut self, p: f64) -> f64 {
        let above = p > self.threshold_power;
        let curve = if !above {
            0.0
        } else {
            match self.params.range_mode {
                RangeMode::Fixed => self.depth,
                RangeMode::Proportional => {
                    (10.0 * p.log10() - self.threshold_db).clamp(0.0, self.depth)
                }
            }
        };
        // Activity (for metering): above, or within the hold time.
        if above {
            self.hold_left = self.hold_frames;
            self.active = true;
        } else if self.hold_left > 0 {
            self.hold_left -= 1;
        } else {
            self.active = false;
        }
        // Peak hold of the target.
        if curve >= self.held {
            self.held = curve;
            self.peak_left = self.hold_frames;
        } else if self.peak_left > 0 {
            self.peak_left -= 1;
        } else {
            self.held = curve;
        }
        // Linear-in-dB fade toward the target.
        let (t, r) = (self.held, self.reduction);
        self.reduction = if t > r {
            (r + self.up).min(t)
        } else {
            (r - self.down).max(t)
        };
        if self.reduction != self.gain_for {
            self.gain_for = self.reduction;
            self.gain = db_to_lin(-self.reduction);
        }
        self.gain
    }

    #[inline(always)]
    fn process_blocks<S: Simd>(
        &mut self,
        simd: S,
        samples: &mut [f32],
        key: Option<(&[f32], usize)>,
    ) {
        let ch = self.channels;
        let frames = samples.len() / ch;
        let samples = &mut samples[..frames * ch];
        let key_frames = key.map_or(0, |(k, kc)| k.len() / kc);
        let mut gains = [0.0f32; BLOCK_FRAMES];
        let mut peaks = [0.0f32; BLOCK_FRAMES];
        let mut pos = 0usize;
        for block in samples.chunks_mut(BLOCK_FRAMES * ch) {
            let n = block.len() / ch;
            let p0 = self.power;
            match key {
                Some((k, kc)) => {
                    let mut i = split_point(pos, frames, key_frames);
                    let end = split_point(pos + n, frames, key_frames);
                    while i < end {
                        let m = (end - i).min(BLOCK_FRAMES);
                        let peaks = &mut peaks[..m];
                        self.key
                            .linked_peaks_simd(simd, &k[i * kc..(i + m) * kc], kc, peaks);
                        for &pk in peaks.iter() {
                            self.env.step(pk as f64);
                        }
                        i += m;
                    }
                }
                None => {
                    for _ in 0..self.no_key.advance(n) {
                        self.env.step(0.0);
                    }
                }
            }
            // Capped so that interpolation stays finite for an infinite key.
            let p1 = self.env.value().min(MAX_POWER);
            self.power = p1;
            let inv = 1.0 / n as f64;
            for (j, g) in gains[..n].iter_mut().enumerate() {
                let p = p0 + (p1 - p0) * ((j + 1) as f64 * inv);
                *g = self.step_frame(p) as f32;
            }
            apply_gains_simd(simd, block, ch, &gains[..n]);
            pos += n;
        }
    }
}

impl Processor for Ducker {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        self.sample_rate = (sample_rate as f64).clamp(1.0, 1.0e6);
        self.channels = channels;
        if !self.level_forced {
            self.level = detect();
        }
        self.configure_key();
        self.update_coefficients();
        self.reset();
    }

    /// Process without a key: the key reads as silence (see "Missing key").
    fn process(&mut self, samples: &mut [f32]) {
        self.process_with_key(samples, &[], 0);
    }

    fn reset(&mut self) {
        self.key.reset();
        self.env.reset();
        self.power = 0.0;
        self.no_key.reset();
        self.held = 0.0;
        self.peak_left = 0;
        self.hold_left = 0;
        self.active = false;
        self.reduction = 0.0;
        self.gain = 1.0;
        self.gain_for = 0.0;
    }
}
