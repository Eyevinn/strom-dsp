//! Multiband compressor: LR4 crossover, one compressor per band,
//! summed.
//!
//! # Signal flow
//!
//! ```text
//! input → drive (dB) → Crossover (LR4, 1–5 bands, all-pass compensated)
//!       → Compressor per band (linked channels) → Σ bands → output
//! ```
//!
//! - **Crossover**: the [`Crossover`] at Linkwitz-Riley 4th order
//!   (S. Linkwitz, JAES 1976; see [`crate::filter`] for the topology). Its
//!   plain band sum is an all-pass, so with every band at unity gain the
//!   chain has a flat magnitude response and zero added latency.
//! - **Per band**: the [`Compressor`] (Giannoulis, Massberg & Reiss,
//!   JAES 2012), driven through its public API. Each band has its own
//!   threshold, ratio, attack, release, makeup and knee, so the bands are
//!   independent; within a band the compressor's detector links all channels
//!   (the linked peak or sine-calibrated RMS of the band signal).
//! - **Sum**: the band outputs are added in `f32`. With one band the
//!   crossover passes the input through bit for bit, so the chain is exactly
//!   a wideband compressor.
//!
//! # Units (own interface)
//!
//! Everything is in dB/ms/Hz. The wrapper converts to the compressor's
//! linear units: `threshold = 10^(threshold_db/20)`,
//! `makeup_gain = 10^(makeup_db/20)` and the knee factor
//! `k = 10^(−knee_db/40)` (the compressor's knee spans `threshold·k …
//! threshold/k`, i.e. `W = −40·log10(k)` dB). Thresholds compare against the
//! compressor's detector reading of the **band** signal: the peak level of a
//! steady sine (both detector modes are sine-calibrated).
//!
//! # Drive
//!
//! `drive_db` is one gain applied to the input before the crossover. The
//! crossover is linear, so this is the same as feeding every band `drive_db`
//! more level. It is not compensated at the output: below threshold the
//! output rises by `drive_db`, above a band's threshold (hard knee, ratio
//! `R`) that band's output rises by `drive_db / R`. Drive changes glide in dB
//! through two cascaded one-poles of [`DRIVE_SMOOTHING_MS`] each (critically
//! damped: the glide starts with zero speed, so a ±12 dB jump does not
//! click). A settled drive of 0 dB is skipped (bit-exact).
//!
//! # Parameter changes
//!
//! - Threshold, ratio, knee and makeup glide inside each compressor (10 ms,
//!   see [`crate::compressor`]); attack and release change only the
//!   smoothing coefficients, so the gain does not step.
//! - Crossover frequencies glide (see [`crate::filter`]).
//! - **Band count** is structural. It is accepted by
//!   [`MultibandCompressor::set_params`] at any time and handled by the
//!   crossover's 10 ms crossfade (new bands fade in from silence, removed
//!   bands fold into the new top band during the fade). The compressor of a
//!   band that becomes active starts from a reset state. Hosts should still
//!   treat the band count as a setup-time property (`stromdsp-broadcastprocessor` makes
//!   `num-bands` READY-only), because the crossfade is not designed to be
//!   inaudible on programme material.
//!
//! # Real-time safety
//!
//! [`Processor::configure`] allocates the band buffers
//! (`MAX_CROSSOVER_BANDS × CHUNK_FRAMES × channels`) and configures the
//! crossover and the compressors. `set_params`, `process` and `reset` never
//! allocate; `process` works through the input in chunks of
//! [`CHUNK_FRAMES`] frames.

use crate::compressor::{CompressorDetector, CompressorParams};
use crate::dynamics::OnePole;
use crate::filter::{Crossover, CrossoverParams, LrOrder, MAX_CROSSOVER_BANDS, MAX_SPLITS};
use crate::simd::Level;
use crate::util::db_to_lin;
use crate::{Compressor, Processor};

pub use crate::filter::MAX_CROSSOVER_BANDS as MAX_BANDS;

/// Frames processed per internal chunk (sizes the band buffers).
pub const CHUNK_FRAMES: usize = 1024;

/// Time constant (ms) of each of the two cascaded one-poles of the drive glide.
pub const DRIVE_SMOOTHING_MS: f64 = 5.0;

/// Range of `drive_db` (dB).
pub const DRIVE_RANGE_DB: (f32, f32) = (-12.0, 12.0);
/// Range of a band's `threshold_db` (dB).
pub const THRESHOLD_RANGE_DB: (f32, f32) = (-60.0, 0.0);
/// Range of a band's `ratio`.
pub const RATIO_RANGE: (f32, f32) = (1.0, 20.0);
/// Range of a band's `attack_ms`.
pub const ATTACK_RANGE_MS: (f32, f32) = (0.1, 500.0);
/// Range of a band's `release_ms`.
pub const RELEASE_RANGE_MS: (f32, f32) = (5.0, 5000.0);
/// Range of a band's `makeup_db` (dB).
pub const MAKEUP_RANGE_DB: (f32, f32) = (-12.0, 24.0);
/// Range of a band's `knee_db` (dB; 0 = hard knee).
pub const KNEE_RANGE_DB: (f32, f32) = (0.0, 24.0);

/// Default crossover frequencies (Hz).
pub const DEFAULT_FREQUENCIES_HZ: [f32; MAX_SPLITS] = [120.0, 800.0, 4000.0, 10_000.0];

/// Default number of bands (1 … 5).
pub const DEFAULT_NUM_BANDS: usize = 4;

/// Snap distance (dB) of the drive glide.
const DRIVE_SNAP_DB: f64 = 1e-6;

/// Parameters of one band. Units: dB, ms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultibandBandParams {
    /// Threshold (dB, −60 … 0), against the band's sine-calibrated level.
    pub threshold_db: f32,
    /// Ratio (1 … 20; 1 = no compression).
    pub ratio: f32,
    /// Attack time (ms, 0.1 … 500).
    pub attack_ms: f32,
    /// Release time (ms, 5 … 5000).
    pub release_ms: f32,
    /// Makeup gain (dB, −12 … 24).
    pub makeup_db: f32,
    /// Soft-knee width (dB, 0 … 24; 0 = hard knee).
    pub knee_db: f32,
}

impl MultibandBandParams {
    /// Gentle defaults for band `k` of `n` (own choice): threshold −18 dB,
    /// ratio 2, knee 6 dB, makeup 0; time constants shorter toward the top
    /// band (lowest band 30 ms / 300 ms, top band 3 ms / 80 ms), so low
    /// bands do not modulate their own waveform and high bands catch
    /// transients.
    pub fn default_for(k: usize, n: usize) -> Self {
        const ATTACK: [f32; MAX_CROSSOVER_BANDS] = [30.0, 15.0, 8.0, 5.0, 3.0];
        const RELEASE: [f32; MAX_CROSSOVER_BANDS] = [300.0, 200.0, 150.0, 100.0, 80.0];
        let n = n.clamp(1, MAX_CROSSOVER_BANDS);
        // Map band k of n onto the 5-entry table, lowest to lowest and top to top.
        let i = if n == 1 {
            1
        } else {
            (k.min(n - 1) * (MAX_CROSSOVER_BANDS - 1) + (n - 1) / 2) / (n - 1)
        };
        Self {
            threshold_db: -18.0,
            ratio: 2.0,
            attack_ms: ATTACK[i],
            release_ms: RELEASE[i],
            makeup_db: 0.0,
            knee_db: 6.0,
        }
    }

    /// Copy with every field clamped to its range (NaN becomes the default).
    pub fn clamped(&self) -> Self {
        let d = Self::default();
        let c =
            |v: f32, (lo, hi): (f32, f32), fb: f32| if v.is_nan() { fb } else { v.clamp(lo, hi) };
        Self {
            threshold_db: c(self.threshold_db, THRESHOLD_RANGE_DB, d.threshold_db),
            ratio: c(self.ratio, RATIO_RANGE, d.ratio),
            attack_ms: c(self.attack_ms, ATTACK_RANGE_MS, d.attack_ms),
            release_ms: c(self.release_ms, RELEASE_RANGE_MS, d.release_ms),
            makeup_db: c(self.makeup_db, MAKEUP_RANGE_DB, d.makeup_db),
            knee_db: c(self.knee_db, KNEE_RANGE_DB, d.knee_db),
        }
    }

    /// The [`Compressor`] parameters (linear units) for these settings,
    /// after clamping. See "Units" in the module documentation.
    pub fn compressor_params(&self) -> CompressorParams {
        let p = self.clamped();
        CompressorParams {
            threshold: db_to_lin(p.threshold_db as f64) as f32,
            ratio: p.ratio,
            attack_ms: p.attack_ms,
            release_ms: p.release_ms,
            makeup_gain: db_to_lin(p.makeup_db as f64) as f32,
            knee: 10f64.powf(-(p.knee_db as f64) / 40.0) as f32,
        }
    }
}

impl Default for MultibandBandParams {
    fn default() -> Self {
        Self::default_for(1, MAX_CROSSOVER_BANDS)
    }
}

/// Parameters of a [`MultibandCompressor`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultibandParams {
    /// Number of bands, 1 … 5 (clamped). Structural: see "Parameter changes"
    /// in the module documentation.
    pub num_bands: usize,
    /// Crossover frequencies (Hz), lowest first; the first `num_bands − 1`
    /// are used. Clamped by the crossover to `[1 Hz, 0.499·fs]` and to be
    /// non-decreasing (a frequency below its predecessor is raised to it).
    pub frequencies_hz: [f32; MAX_SPLITS],
    /// Gain into all bands (dB, −12 … 12).
    pub drive_db: f32,
    /// Level detector of every band's compressor.
    pub detector: CompressorDetector,
    /// Per-band settings; band 0 is the lowest. Only the first `num_bands`
    /// are used.
    pub bands: [MultibandBandParams; MAX_CROSSOVER_BANDS],
}

impl Default for MultibandParams {
    fn default() -> Self {
        Self {
            num_bands: DEFAULT_NUM_BANDS,
            frequencies_hz: DEFAULT_FREQUENCIES_HZ,
            drive_db: 0.0,
            detector: CompressorDetector::Peak,
            bands: core::array::from_fn(|k| MultibandBandParams::default_for(k, DEFAULT_NUM_BANDS)),
        }
    }
}

/// Drive gain glide: two cascaded one-poles in dB.
#[derive(Debug, Clone, Copy, Default)]
struct DriveGlide {
    pre: OnePole,
    cur: OnePole,
    target_db: f64,
    settled: bool,
}

impl DriveGlide {
    fn set_time(&mut self, fs: f64) {
        self.pre = OnePole::new(DRIVE_SMOOTHING_MS, fs, self.target_db).with_snap(DRIVE_SNAP_DB);
        self.cur = OnePole::new(DRIVE_SMOOTHING_MS, fs, self.target_db).with_snap(DRIVE_SNAP_DB);
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

    /// Apply the (possibly gliding) drive to interleaved `data` in place.
    fn apply(&mut self, data: &mut [f32], channels: usize) {
        if self.settled {
            if self.target_db != 0.0 {
                let g = db_to_lin(self.target_db) as f32;
                data.iter_mut().for_each(|v| *v *= g);
            }
            return;
        }
        for frame in data.chunks_exact_mut(channels) {
            let p = self.pre.step(self.target_db);
            let c = self.cur.step(p);
            let g = db_to_lin(c);
            for v in frame {
                *v = (*v as f64 * g) as f32;
            }
        }
        if self.cur.value() == self.target_db && self.pre.value() == self.target_db {
            self.settled = true;
        }
    }
}

/// Multiband compressor for interleaved `f32` audio. See the module
/// documentation for the signal flow, units and parameter behaviour.
#[derive(Debug)]
pub struct MultibandCompressor {
    params: MultibandParams,
    channels: usize,
    crossover: Crossover,
    compressors: [Compressor; MAX_CROSSOVER_BANDS],
    /// `MAX_CROSSOVER_BANDS` buffers of `CHUNK_FRAMES × channels`.
    bands: Vec<Vec<f32>>,
    drive: DriveGlide,
    /// Drive snaps on the first `process` after configure/reset.
    fresh: bool,
}

impl Default for MultibandCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl MultibandCompressor {
    /// Create an unconfigured multiband compressor with default parameters
    /// (`process` is a passthrough until [`Processor::configure`]).
    pub fn new() -> Self {
        let mut m = Self {
            params: MultibandParams::default(),
            channels: 0,
            crossover: Crossover::new(),
            compressors: core::array::from_fn(|_| Compressor::new()),
            bands: Vec::new(),
            drive: DriveGlide::default(),
            fresh: true,
        };
        let p = m.params;
        m.set_params(&p);
        m
    }

    /// Update parameters. Never allocates; out-of-range values are clamped.
    /// All changes are smoothed (see "Parameter changes" in the module docs).
    pub fn set_params(&mut self, params: &MultibandParams) {
        let old_bands = self.num_bands();
        let mut p = *params;
        p.num_bands = p.num_bands.clamp(1, MAX_CROSSOVER_BANDS);
        p.drive_db = if p.drive_db.is_nan() {
            0.0
        } else {
            p.drive_db.clamp(DRIVE_RANGE_DB.0, DRIVE_RANGE_DB.1)
        };
        for b in &mut p.bands {
            *b = b.clamped();
        }
        self.params = p;
        self.crossover.set_params(&CrossoverParams {
            order: LrOrder::Lr4,
            num_bands: p.num_bands,
            frequencies_hz: p.frequencies_hz,
        });
        for (k, (c, b)) in self.compressors.iter_mut().zip(&p.bands).enumerate() {
            // A band that becomes active starts from a clean state.
            if k >= old_bands && k < p.num_bands {
                c.reset();
            }
            c.set_detector(p.detector);
            c.set_params(&b.compressor_params());
        }
        self.drive.retarget(p.drive_db as f64);
    }

    /// Current (clamped) parameters.
    pub fn params(&self) -> &MultibandParams {
        &self.params
    }

    /// Number of active bands.
    pub fn num_bands(&self) -> usize {
        self.params.num_bands.clamp(1, MAX_CROSSOVER_BANDS)
    }

    /// Crossover frequencies (Hz) as realised after clamping (the first
    /// `num_bands() − 1` are in use).
    pub fn frequencies_hz(&self) -> [f64; MAX_SPLITS] {
        self.crossover.frequencies_hz()
    }

    /// Smoothed gain reduction (dB, ≥ 0, excluding makeup) of band `k` after
    /// the most recently processed frame; 0 for an inactive band.
    pub fn band_gain_reduction_db(&self, k: usize) -> f64 {
        if k < self.num_bands() {
            self.compressors[k].gain_reduction_db()
        } else {
            0.0
        }
    }

    /// [`band_gain_reduction_db`](Self::band_gain_reduction_db) of every
    /// band (inactive bands read 0), for metering.
    pub fn gain_reduction_db(&self) -> [f64; MAX_CROSSOVER_BANDS] {
        core::array::from_fn(|k| self.band_gain_reduction_db(k))
    }

    /// Force the SIMD level of the crossover and of every compressor
    /// (persists across `configure`). Never allocates.
    pub fn set_simd_level(&mut self, level: Level) {
        self.crossover.set_simd_level(level);
        for c in &mut self.compressors {
            c.set_simd_level(level);
        }
    }
}

impl Processor for MultibandCompressor {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        let channels = channels.max(1);
        self.channels = channels;
        self.crossover.configure(sample_rate, channels);
        for c in &mut self.compressors {
            c.configure(sample_rate, channels);
        }
        self.bands = vec![vec![0.0; CHUNK_FRAMES * channels]; MAX_CROSSOVER_BANDS];
        let fs = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate as f64
        } else {
            48_000.0
        };
        self.drive.set_time(fs);
        let p = self.params;
        self.set_params(&p);
        self.reset();
    }

    fn process(&mut self, samples: &mut [f32]) {
        let ch = self.channels;
        if ch == 0 {
            return;
        }
        if self.fresh {
            self.drive.snap();
            self.fresh = false;
        }
        let nb = self.num_bands();
        let usable = samples.len() / ch * ch;
        for chunk in samples[..usable].chunks_mut(CHUNK_FRAMES * ch) {
            let len = chunk.len();
            self.drive.apply(chunk, ch);
            self.crossover.process(chunk, &mut self.bands[..nb]);
            for (c, b) in self.compressors.iter_mut().zip(&mut self.bands[..nb]) {
                c.process(&mut b[..len]);
            }
            chunk.copy_from_slice(&self.bands[0][..len]);
            for b in &self.bands[1..nb] {
                for (o, &v) in chunk.iter_mut().zip(&b[..len]) {
                    *o += v;
                }
            }
        }
    }

    fn reset(&mut self) {
        self.crossover.reset();
        for c in &mut self.compressors {
            c.reset();
        }
        self.fresh = true;
    }
}
