//! Real-time declipper: finds hard-clipped runs on the waveform and rebuilds
//! them by constrained AR interpolation.
//!
//! [`Declipper`] delays the signal by a fixed latency chosen with
//! [`DeclipperLatency`] (3, 5 or 10 ms; `round(ms · fs)` frames, see
//! [`Declipper::latency_frames_for`]). Inside that delay it:
//!
//! 1. **detects** clipped runs per channel with the [`ClipDetector`]: two or
//!    more consecutive samples on a flat (or, optionally, slightly drooping)
//!    plateau, entered and left by a sharp corner from the same side, at a
//!    level above `min_level`. The clip level is each plateau's own value,
//!    so a clip below 0 dBFS, or with a DC offset after it, is found too;
//!    the detector docs give the exact rules;
//! 2. **rebuilds** each run from a window of [`PAST_CONTEXT_MS`] before it
//!    and up to [`FUTURE_CONTEXT_MS`] after it, by least-squares AR
//!    interpolation ([`lsar`]). The context is asymmetric: the past is
//!    always there, the future costs latency. A run is rebuilt as soon as
//!    the full future context has arrived, or at the latest when its first
//!    sample is about to leave the delay line, with whatever future context
//!    there is by then (`latency − run length − 1` frames or more). The AR model
//!    is first estimated by Burg's method from the known samples of the
//!    window only, then from the whole window with the rebuilt samples in
//!    it (Janssen et al. 1986 alternate estimate and interpolation); the run's samples
//!    are the values that minimise the prediction error, and each one is kept
//!    on the clipped side of its observed value (Abel & Smith 1991) and no
//!    more than `max_overshoot_db` beyond it. AR estimate and interpolation
//!    alternate [`AR_ROUNDS`] times. Later runs inside the window are
//!    unknowns of the same solve (up to [`JOINT_LATER`] samples), so they do
//!    not pull the estimate towards their clipped values; they are rebuilt
//!    again in their own turn.
//!
//! Samples that are not in a repaired run come out bit-exact, delayed by the
//! latency. A run longer than `max_run_ms`, or too long to be found and
//! rebuilt inside the latency ([`Declipper::max_run_frames`]), is passed
//! through unchanged and counted ([`DeclipperStats::runs_too_long`]).
//!
//! # Latency
//!
//! | [`DeclipperLatency`] | latency | longest run at 48 kHz |
//! |---|---|---|
//! | `Low` | 3 ms | `max_run_ms` (default 2 ms = 96 frames) |
//! | `Normal` (default) | 5 ms | likewise |
//! | `High` | 10 ms | likewise |
//!
//! The latency is fixed at [`configure`](Processor::configure); a change of
//! [`DeclipperParams::latency`] takes effect at the next `configure`. What
//! each setting costs in quality is measured in `tests/declipper_quality.rs`.
//!
//! # Headroom
//!
//! The rebuilt peaks lie beyond the clip level, so the output is hotter than
//! the input: by up to `max_overshoot_db` over the clip level in the worst
//! case. In float this is harmless; the stage that converts to a fixed-point
//! format (or that must meet a ceiling) needs a limiter after the declipper.
//! [`DeclipperStats::max_overshoot_db`] reports what was actually used.
//!
//! # Channels
//!
//! Channels are independent: each has its own detector and is rebuilt from
//! its own samples.
//!
//! # Real-time safety
//!
//! [`configure`](Processor::configure) allocates the delay rings, the run
//! queues and one solver sized for the largest problem (window of
//! past and future context plus the longest run, `max run + JOINT_LATER`
//! unknowns). `process`
//! never allocates. The cost per run is `O(m³)` in the number of unknowns `m`,
//! bounded by the run cap.
//!
//! # Enabled
//!
//! `enabled = false` crossfades (raised cosine, [`XFADE_MS`]) to the dry
//! input delayed by the same latency, so toggling never shifts the audio in
//! time. Detection and rebuilding keep running while disabled, so
//! re-enabling fades back without a gap.

pub mod detector;
pub mod lsar;

pub use detector::{ClipDetector, ClipRun, DetectorParams};
pub use lsar::{ArMethod, LsarError, LsarSolver};

use crate::Processor;

/// Context before a run used for the AR model (ms; at least `3 × AR order`
/// frames). The past costs no latency, only computation: 10 ms gave
/// 0.4–0.8 dB more SDR improvement on the clipped samples of recorded
/// speech than 5 ms, for about 40 % more work per run.
pub const PAST_CONTEXT_MS: f32 = 10.0;

/// Most context after a run used for the AR model (ms). A run is rebuilt
/// once this much has arrived, or earlier when the latency runs out.
pub const FUTURE_CONTEXT_MS: f32 = 5.0;

/// Least context after a run (ms, at least 4 frames): longer runs than the
/// latency allows with this much context are passed through.
pub const MIN_FUTURE_CONTEXT_MS: f32 = 0.5;

/// Delay between a run's last sample and the sample that reports it
/// (frames): a pair is decided on the second sample after it, and one
/// detector call can queue a run for the next.
const DETECT_DELAY: usize = 4;

/// Latency setting of the [`Declipper`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeclipperLatency {
    /// 3 ms: less future context for the rebuild.
    Low,
    /// 5 ms.
    #[default]
    Normal,
    /// 10 ms: the full future context for runs up to 5 ms.
    High,
}

impl DeclipperLatency {
    /// The latency in milliseconds.
    pub fn ms(self) -> f32 {
        match self {
            Self::Low => 3.0,
            Self::Normal => 5.0,
            Self::High => 10.0,
        }
    }
}

/// AR model order at 48 kHz; it scales with the sample rate, within
/// [`AR_ORDER_MIN`] … [`AR_ORDER_MAX`].
pub const AR_ORDER_48K: usize = 48;
/// Lowest AR order.
pub const AR_ORDER_MIN: usize = 12;
/// Highest AR order.
pub const AR_ORDER_MAX: usize = 64;

/// Alternations of AR estimate and interpolation per run.
pub const AR_ROUNDS: usize = 2;

/// Most samples of later runs in the window that join a run's solve.
pub const JOINT_LATER: usize = 8;

/// Hard cap on the longest repairable run (frames), whatever the rate.
pub const MAX_RUN_FRAMES_CAP: usize = 256;

/// Range of [`DeclipperParams::max_run_ms`].
pub const MAX_RUN_MS_RANGE: (f32, f32) = (0.05, 4.0);

/// Range of [`DeclipperParams::max_overshoot_db`].
pub const MAX_OVERSHOOT_DB_RANGE: (f32, f32) = (0.0, 24.0);

/// Range of [`DeclipperParams::min_level_db`].
pub const MIN_LEVEL_DB_RANGE: (f32, f32) = (-60.0, 0.0);

/// Range of [`DeclipperParams::tolerance`].
pub const TOLERANCE_RANGE: (f32, f32) = (0.0, 0.01);

/// Range of [`DeclipperParams::tilt_hz`].
pub const TILT_HZ_RANGE: (f32, f32) = (0.0, 200.0);

/// Bypass crossfade (ms).
pub const XFADE_MS: f32 = 5.0;

/// Parameters of [`Declipper`]. All but `enabled` and `latency` take effect
/// for runs that close after the change.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeclipperParams {
    /// `false`: crossfade to the delayed dry input.
    pub enabled: bool,
    /// Latency setting (applied at the next `configure`). Default
    /// [`DeclipperLatency::Normal`] (5 ms).
    pub latency: DeclipperLatency,
    /// Lowest plateau level that counts as clipping (dBFS). Default −20:
    /// covers up to 20 dB of gain applied after the clip.
    pub min_level_db: f32,
    /// Plateau flatness relative to its level. Default 1e-5 (≈ 0.0001 dB):
    /// digital clipping, with or without gain, DC or requantisation after
    /// it, gives exactly equal samples.
    pub tolerance: f32,
    /// Longest run that is rebuilt (ms). Default 2 ms; longer runs pass
    /// through unchanged. AR interpolation of a gap of a few ms is poor, and
    /// the solve cost grows with the cube of the run length.
    pub max_run_ms: f32,
    /// How far beyond the clip level a rebuilt sample may go (dB). Default 12.
    pub max_overshoot_db: f32,
    /// Also find plateaus that droop like the output of a first-order
    /// high-pass with a corner up to this frequency, applied after the clip
    /// (Hz; 0 = flat plateaus only). Default 50 Hz: covers DC blockers and
    /// AC coupling after the clip.
    pub tilt_hz: f32,
}

impl Default for DeclipperParams {
    fn default() -> Self {
        Self {
            enabled: true,
            latency: DeclipperLatency::Normal,
            min_level_db: -20.0,
            tolerance: 1e-5,
            max_run_ms: 2.0,
            max_overshoot_db: 12.0,
            tilt_hz: 50.0,
        }
    }
}

/// Counters since the last [`Declipper::take_stats`] (or configure/reset).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DeclipperStats {
    /// Runs rebuilt.
    pub runs_repaired: u64,
    /// Samples rebuilt.
    pub samples_repaired: u64,
    /// Runs longer than `max_run_ms`, passed through.
    pub runs_too_long: u64,
    /// Runs left as they were because the solve failed (non-finite input,
    /// numerical breakdown) or the run queue was full.
    pub runs_failed: u64,
    /// Largest rebuilt magnitude over its clip level (dB, ≥ 0; 0 if none).
    pub max_overshoot_db: f32,
    /// Largest rebuilt magnitude (linear).
    pub max_repaired_peak: f32,
}

#[derive(Debug, Clone, Default)]
struct Channel {
    det: ClipDetector,
    /// Observed input.
    input: Vec<f32>,
    /// Output: input with rebuilt runs written over it.
    work: Vec<f32>,
    /// Direction of samples in runs queued for rebuilding (0: none).
    dir: Vec<i8>,
    /// Queued runs (ring).
    pending: Vec<ClipRun>,
    p_head: usize,
    p_len: usize,
}

/// The declipper. See the [module docs](self).
#[derive(Debug, Clone, Default)]
pub struct Declipper {
    params: DeclipperParams,
    configured: bool,
    sample_rate: f32,
    channels: usize,
    /// Latency setting in effect (the one at the last `configure`).
    latency_mode: DeclipperLatency,
    latency: usize,
    past: usize,
    future: usize,
    order: usize,
    max_run_limit: usize,
    max_run: usize,
    cap_factor: f64,
    ring_mask: usize,
    /// Index of the next frame.
    n: u64,
    ch: Vec<Channel>,
    solver: LsarSolver,
    w: Vec<f64>,
    unknown: Vec<usize>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    stats: DeclipperStats,
    xfade_len: usize,
    /// Fade position 0 … xfade_len (xfade_len = fully wet).
    fade: usize,
    fade_set: bool,
}

impl Declipper {
    /// Unconfigured declipper (`process` is a passthrough).
    pub fn new() -> Self {
        Self::default()
    }

    /// AR order at `sample_rate`.
    pub fn order_for(sample_rate: f32) -> usize {
        let p = (AR_ORDER_48K as f64 * f64::from(sample_rate) / 48_000.0).round();
        if p.is_finite() {
            (p as usize).clamp(AR_ORDER_MIN, AR_ORDER_MAX)
        } else {
            AR_ORDER_48K
        }
    }

    /// Context frames before a run at `sample_rate`.
    pub fn past_context_frames_for(sample_rate: f32) -> usize {
        ms_frames(PAST_CONTEXT_MS, sample_rate).max(3 * Self::order_for(sample_rate))
    }

    /// Latency (frames) of `mode` at `sample_rate`: `round(ms · fs)`, at
    /// least 16.
    pub fn latency_frames_for(sample_rate: f32, mode: DeclipperLatency) -> usize {
        ms_frames(mode.ms(), sample_rate).max(16)
    }

    /// Context frames before a run.
    pub fn past_context_frames(&self) -> usize {
        self.past
    }

    /// Most context frames after a run.
    pub fn future_context_frames(&self) -> usize {
        self.future
    }

    /// Longest repairable run (frames) at the current settings: `max_run_ms`,
    /// or what the latency allows (`latency − 4 − max(0.5 ms, 4 frames)`).
    pub fn max_run_frames(&self) -> usize {
        self.max_run
    }

    /// The latency setting in effect.
    pub fn latency_mode(&self) -> DeclipperLatency {
        self.latency_mode
    }

    /// Current parameters.
    pub fn params(&self) -> DeclipperParams {
        self.params
    }

    /// Set parameters (clamped to their ranges; never allocates).
    pub fn set_params(&mut self, p: &DeclipperParams) {
        let mut p = *p;
        let fin = |v: f32, d: f32| if v.is_finite() { v } else { d };
        let d = DeclipperParams::default();
        p.min_level_db =
            fin(p.min_level_db, d.min_level_db).clamp(MIN_LEVEL_DB_RANGE.0, MIN_LEVEL_DB_RANGE.1);
        p.tolerance = fin(p.tolerance, d.tolerance).clamp(TOLERANCE_RANGE.0, TOLERANCE_RANGE.1);
        p.max_run_ms =
            fin(p.max_run_ms, d.max_run_ms).clamp(MAX_RUN_MS_RANGE.0, MAX_RUN_MS_RANGE.1);
        p.max_overshoot_db = fin(p.max_overshoot_db, d.max_overshoot_db)
            .clamp(MAX_OVERSHOOT_DB_RANGE.0, MAX_OVERSHOOT_DB_RANGE.1);
        p.tilt_hz = fin(p.tilt_hz, d.tilt_hz).clamp(TILT_HZ_RANGE.0, TILT_HZ_RANGE.1);
        let enabled_changed = p.enabled != self.params.enabled;
        self.params = p;
        if enabled_changed && !self.fade_set {
            // Before the first frame, the new state applies at once.
            self.fade = if p.enabled { self.xfade_len } else { 0 };
        }
        self.apply_params();
    }

    fn apply_params(&mut self) {
        if !self.configured {
            return;
        }
        let p = self.params;
        self.max_run = ms_frames(p.max_run_ms, self.sample_rate).clamp(1, self.max_run_limit);
        self.cap_factor = 10f64.powf(f64::from(p.max_overshoot_db) / 20.0) - 1.0;
        let dp = DetectorParams {
            min_level: 10f32.powf(p.min_level_db / 20.0),
            tolerance: p.tolerance,
            max_run: self.max_run,
            ceiling_hold: (detector::CEILING_HOLD_S * self.sample_rate) as u64,
            max_tilt: 2.0 * std::f32::consts::PI * p.tilt_hz / self.sample_rate,
            singles: true,
        };
        for c in &mut self.ch {
            c.det.set_params(dp);
        }
    }

    /// Counters since the last call (or configure/reset); resets them.
    pub fn take_stats(&mut self) -> DeclipperStats {
        std::mem::take(&mut self.stats)
    }

    /// Counters since the last [`take_stats`](Self::take_stats).
    pub fn stats(&self) -> DeclipperStats {
        self.stats
    }

    #[inline]
    fn gain_at(&self) -> f32 {
        if self.fade >= self.xfade_len {
            1.0
        } else {
            let x = self.fade as f32 / self.xfade_len as f32;
            0.5 - 0.5 * (std::f32::consts::PI * x).cos()
        }
    }

    /// Rebuild `run` of channel `c` (the newest frame is `self.n`).
    fn rebuild(&mut self, c: usize, run: ClipRun) {
        let mask = self.ring_mask as u64;
        let ch = &mut self.ch[c];
        let s = run.start;
        let e = run.end();
        let now = self.n;
        let ws = s.saturating_sub(self.past as u64);
        let mut we = (e + self.future as u64).min(now);
        let open = ch.det.open_candidate();

        self.unknown.clear();
        self.lo.clear();
        self.hi.clear();
        let cap = self.cap_factor;
        let push = |unknown: &mut Vec<usize>,
                    lo: &mut Vec<f64>,
                    hi: &mut Vec<f64>,
                    idx: usize,
                    obs: f32,
                    dir: i8| {
            let obs = f64::from(obs);
            let reach = (cap * obs.abs()).min(f64::from(f32::MAX) - obs.abs());
            unknown.push(idx);
            if dir > 0 {
                lo.push(obs);
                hi.push(obs + reach);
            } else {
                lo.push(obs - reach);
                hi.push(obs);
            }
        };
        for i in s..=e {
            let obs = ch.input[(i & mask) as usize];
            push(
                &mut self.unknown,
                &mut self.lo,
                &mut self.hi,
                (i - ws) as usize,
                obs,
                run.dir,
            );
        }
        let mut later = 0usize;
        let mut i = e + 1;
        while i <= we {
            let slot = (i & mask) as usize;
            let mut d = ch.dir[slot];
            if d == 0 {
                if let Some((os, ol, od)) = open {
                    if i >= os && i < os + ol as u64 {
                        d = od;
                    }
                }
            }
            if d != 0 {
                if later == JOINT_LATER {
                    we = i - 1;
                    break;
                }
                later += 1;
                push(
                    &mut self.unknown,
                    &mut self.lo,
                    &mut self.hi,
                    (i - ws) as usize,
                    ch.input[slot],
                    d,
                );
            }
            i += 1;
        }
        let len = (we - ws + 1) as usize;
        for (k, v) in self.w[..len].iter_mut().enumerate() {
            *v = f64::from(ch.work[((ws + k as u64) & mask) as usize]);
        }
        let order = self.order.min((len - 1) / 2);
        let result = if order >= 2 {
            self.solver.interpolate(
                &mut self.w[..len],
                &self.unknown,
                &self.lo,
                &self.hi,
                order,
                AR_ROUNDS,
            )
        } else {
            Err(LsarError::WindowTooShort)
        };
        for i in s..=e {
            ch.dir[(i & mask) as usize] = 0;
        }
        if result.is_err() {
            self.stats.runs_failed += 1;
            return;
        }
        let level = f64::from(run.level).abs();
        let mut peak = 0.0f64;
        for i in s..=e {
            let y = self.w[(i - ws) as usize];
            peak = peak.max(y.abs());
            ch.work[(i & mask) as usize] = y as f32;
        }
        self.stats.runs_repaired += 1;
        self.stats.samples_repaired += run.len as u64;
        if level > 0.0 {
            let over = (20.0 * (peak / level).log10()).max(0.0) as f32;
            self.stats.max_overshoot_db = self.stats.max_overshoot_db.max(over);
        }
        self.stats.max_repaired_peak = self.stats.max_repaired_peak.max(peak as f32);
    }
}

fn ms_frames(ms: f32, sample_rate: f32) -> usize {
    let f = (f64::from(ms) * f64::from(sample_rate) / 1000.0).round();
    if f.is_finite() && f > 0.0 {
        f as usize
    } else {
        0
    }
}

impl Processor for Declipper {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        let sample_rate = if sample_rate.is_finite() {
            sample_rate.clamp(8_000.0, 192_000.0)
        } else {
            48_000.0
        };
        let channels = channels.max(1);
        self.sample_rate = sample_rate;
        self.channels = channels;
        self.order = Self::order_for(sample_rate);
        self.latency_mode = self.params.latency;
        self.latency = Self::latency_frames_for(sample_rate, self.latency_mode);
        self.past = Self::past_context_frames_for(sample_rate);
        let min_future = ms_frames(MIN_FUTURE_CONTEXT_MS, sample_rate).max(4);
        self.max_run_limit = self
            .latency
            .saturating_sub(DETECT_DELAY + min_future)
            .clamp(1, MAX_RUN_FRAMES_CAP);
        self.future = ms_frames(FUTURE_CONTEXT_MS, sample_rate)
            .min(self.latency - 1)
            .max(1);
        let ring = (self.latency + self.past + 2).next_power_of_two();
        self.ring_mask = ring - 1;
        let pending_cap = (self.latency + self.past) / 2 + 4;
        self.ch = (0..channels)
            .map(|_| Channel {
                det: ClipDetector::default(),
                input: vec![0.0; ring],
                work: vec![0.0; ring],
                dir: vec![0; ring],
                pending: vec![
                    ClipRun {
                        start: 0,
                        len: 2,
                        level: 0.0,
                        dir: 1,
                        too_long: false,
                    };
                    pending_cap
                ],
                p_head: 0,
                p_len: 0,
            })
            .collect();
        let n_max = self.past + self.max_run_limit + self.future + 1;
        let m_max = self.max_run_limit + JOINT_LATER;
        self.solver = LsarSolver::new(n_max, m_max, self.order);
        self.solver.method = ArMethod::Burg;
        self.w = vec![0.0; n_max];
        self.unknown = Vec::with_capacity(m_max);
        self.lo = Vec::with_capacity(m_max);
        self.hi = Vec::with_capacity(m_max);
        self.xfade_len = ms_frames(XFADE_MS, sample_rate).max(1);
        self.configured = true;
        self.apply_params();
        self.reset();
    }

    fn process(&mut self, samples: &mut [f32]) {
        if !self.configured {
            return;
        }
        let nch = self.channels;
        let mask = self.ring_mask as u64;
        let lat = self.latency as u64;
        let fut = self.future as u64;
        for frame in samples.chunks_exact_mut(nch) {
            let n = self.n;
            let slot = (n & mask) as usize;
            for (c, x) in frame.iter().enumerate() {
                let ch = &mut self.ch[c];
                ch.input[slot] = *x;
                ch.work[slot] = *x;
                ch.dir[slot] = 0;
                if let Some(run) = ch.det.push(*x) {
                    if run.too_long || run.start + lat < n {
                        self.stats.runs_too_long += 1;
                    } else if ch.p_len == ch.pending.len() {
                        self.stats.runs_failed += 1;
                    } else {
                        for i in run.start..=run.end() {
                            ch.dir[(i & mask) as usize] = run.dir;
                        }
                        let cap = ch.pending.len();
                        ch.pending[(ch.p_head + ch.p_len) % cap] = run;
                        ch.p_len += 1;
                    }
                }
                loop {
                    let ch = &mut self.ch[c];
                    if ch.p_len == 0 {
                        break;
                    }
                    let run = ch.pending[ch.p_head];
                    // Full future context, or the deadline: the run's first
                    // sample leaves the delay line at this frame.
                    if run.end() + fut > n && run.start + lat > n {
                        break;
                    }
                    ch.p_head = (ch.p_head + 1) % ch.pending.len();
                    ch.p_len -= 1;
                    self.rebuild(c, run);
                }
            }
            // Output frame n − latency.
            self.fade_set = true;
            let target = if self.params.enabled {
                self.xfade_len
            } else {
                0
            };
            if self.fade < target {
                self.fade += 1;
            } else if self.fade > target {
                self.fade -= 1;
            }
            let g = self.gain_at();
            let out_slot = (n.wrapping_sub(lat) & mask) as usize;
            for (c, y) in frame.iter_mut().enumerate() {
                let ch = &self.ch[c];
                let dry = ch.input[out_slot];
                let wet = ch.work[out_slot];
                *y = if self.fade == self.xfade_len {
                    wet
                } else if self.fade == 0 {
                    dry
                } else {
                    dry + g * (wet - dry)
                };
            }
            self.n += 1;
        }
    }

    fn reset(&mut self) {
        for c in &mut self.ch {
            c.det.reset();
            c.input.fill(0.0);
            c.work.fill(0.0);
            c.dir.fill(0);
            c.p_head = 0;
            c.p_len = 0;
        }
        self.n = 0;
        self.stats = DeclipperStats::default();
        self.fade = if self.params.enabled {
            self.xfade_len
        } else {
            0
        };
        self.fade_set = false;
    }

    fn latency_frames(&self) -> usize {
        if self.configured {
            self.latency
        } else {
            0
        }
    }
}
