//! Bandwidth guard: an estimate of the programme's upper band edge, for the
//! final clipper to leave band-limited programme unclipped (see
//! [`crate::clipper`], "Depth").
//!
//! A band-limited source (phone speech at about 3.4 kHz, a low-bitrate codec
//! at 8 … 16 kHz) has nothing above its band edge to mask what a clipper
//! adds there. The guard measures, for each candidate edge
//! [`CANDIDATE_EDGES_HZ`], the energy of the mono sum above it
//! (4th-order Butterworth high-pass) against the full-band energy, both as
//! slow mean squares (one-pole, [`WINDOW_MS`]; the energy above a
//! candidate falls with the shorter [`FALL_MS`], so content that stops is
//! noticed within a few seconds), the textbook short-time spectral
//! estimate of a filter bank. The programme has content above a
//! candidate while that ratio is over [`PRESENT_DB`]; it stops having it
//! once the ratio falls under [`ABSENT_DB`] (hysteresis, so programme near
//! a threshold does not flap). The estimated edge is the lowest candidate
//! above which there is no content, or none (full band).
//!
//! The input is fed in blocks ([`BandwidthGuard::push`], then
//! [`BandwidthGuard::end_block`]); the slow energies and the decision are
//! updated once per block. The estimate uses past input only (no latency)
//! and holds while a block is quieter than [`MIN_LEVEL_DB`] (pauses do not
//! open or close anything).
//!
//! # Real-time safety
//!
//! Fixed-size state; no allocation. Per frame: one mono sum and two biquads
//! per candidate; per block, a few divisions and one `exp`.

use crate::biquad::BiquadCoefficients;

/// Candidate band edges (Hz), ascending. Candidates at or over 0.45·fs are
/// not used.
pub const CANDIDATE_EDGES_HZ: [f64; 3] = [4_000.0, 8_000.0, 12_000.0];
/// Number of candidates.
pub const EDGES: usize = CANDIDATE_EDGES_HZ.len();
/// Time constant of the energy estimates (ms).
pub const WINDOW_MS: f64 = 1_000.0;
/// Time constant of the energy above a candidate while it falls (ms): a
/// switch from full-band to band-limited programme (−3 to −50 dB) is
/// noticed within about 3 s.
pub const FALL_MS: f64 = 250.0;
/// Content above a candidate counts as present while the energy above it
/// is over this, relative to the full-band energy (dB). Chosen on the test
/// corpus: above 12 kHz, full-band speech and music read −24 … −43 dB,
/// phone, codec and 16 kHz-sampled speech −52 … −80 dB (the high-pass
/// skirt passes some of the content just under a candidate).
pub const PRESENT_DB: f64 = -40.0;
/// … and as absent once it falls under this (dB).
pub const ABSENT_DB: f64 = -46.0;
/// Frames quieter than this mean square (dBFS) hold the estimate.
pub const MIN_LEVEL_DB: f64 = -60.0;

/// Upper band edge estimator (see the module docs).
#[derive(Debug, Clone)]
pub struct BandwidthGuard {
    hp: [[BiquadCoefficients; 2]; EDGES],
    z: [[[f64; 2]; 2]; EDGES],
    /// Candidates in use (under 0.45·fs).
    used: usize,
    fs: f64,
    /// Slow (one-pole per block) mean squares of the full band and above
    /// each candidate.
    full: f64,
    above: [f64; EDGES],
    /// Content present above each candidate.
    present: [bool; EDGES],
    min_level: f64,
    /// Sums of squares of the current block, and its length.
    block_full: f64,
    block_above: [f64; EDGES],
    block_frames: usize,
}

impl Default for BandwidthGuard {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl BandwidthGuard {
    /// A fresh guard for `sample_rate` (every candidate open: full band).
    pub fn new(sample_rate: f64) -> Self {
        let fs = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate
        } else {
            48_000.0
        };
        let used = CANDIDATE_EDGES_HZ
            .iter()
            .take_while(|&&e| e < 0.45 * fs)
            .count();
        let hp = CANDIDATE_EDGES_HZ.map(|e| {
            let e = e.min(0.45 * fs);
            [
                BiquadCoefficients::high_pass(e, 0.541_196, fs),
                BiquadCoefficients::high_pass(e, 1.306_563, fs),
            ]
        });
        Self {
            hp,
            z: [[[0.0; 2]; 2]; EDGES],
            used,
            fs,
            full: 0.0,
            above: [0.0; EDGES],
            present: [true; EDGES],
            min_level: 10f64.powf(MIN_LEVEL_DB / 10.0),
            block_full: 0.0,
            block_above: [0.0; EDGES],
            block_frames: 0,
        }
    }

    /// Clear the state (full band).
    pub fn reset(&mut self) {
        self.z = [[[0.0; 2]; 2]; EDGES];
        self.full = 0.0;
        self.above = [0.0; EDGES];
        self.present = [true; EDGES];
        self.block_full = 0.0;
        self.block_above = [0.0; EDGES];
        self.block_frames = 0;
    }

    /// Feed `x` (interleaved, `ch` channels): the energies of the current
    /// block. [`BandwidthGuard::end_block`] turns them into the estimate.
    pub fn push(&mut self, x: &[f32], ch: usize) {
        if ch == 0 {
            return;
        }
        let inv = 1.0 / ch as f64;
        for frame in x.chunks_exact(ch) {
            let m = frame.iter().map(|&v| v as f64).sum::<f64>() * inv;
            let m = if m.is_finite() { m } else { 0.0 };
            self.block_full += m * m;
            for k in 0..self.used {
                let y = self.filter(k, m);
                self.block_above[k] += y * y;
            }
            self.block_frames += 1;
        }
    }

    /// End of a block of input: unless the block was quiet, update the slow
    /// energies with its mean squares and decide, with hysteresis, where the
    /// programme has content.
    pub fn end_block(&mut self) {
        let frames = self.block_frames.max(1) as f64;
        let full = self.block_full / frames;
        // A NaN level counts as quiet.
        let loud = full > self.min_level;
        if loud {
            let a = self.block_coef(self.block_frames, WINDOW_MS);
            let a_fall = self.block_coef(self.block_frames, FALL_MS);
            self.full = a * self.full + (1.0 - a) * full;
            for k in 0..self.used {
                let v = self.block_above[k] / frames;
                let a = if v < self.above[k] { a_fall } else { a };
                self.above[k] = a * self.above[k] + (1.0 - a) * v;
            }
            let present = 10f64.powf(PRESENT_DB / 10.0);
            let absent = 10f64.powf(ABSENT_DB / 10.0);
            for k in 0..self.used {
                let r = self.above[k] / self.full.max(1e-30);
                if self.present[k] && r < absent {
                    self.present[k] = false;
                } else if !self.present[k] && r > present {
                    self.present[k] = true;
                }
            }
        }
        self.block_full = 0.0;
        self.block_above = [0.0; EDGES];
        self.block_frames = 0;
        self.flush();
    }

    /// One-pole coefficient of time constant `ms` for a block of `frames`.
    fn block_coef(&self, frames: usize, ms: f64) -> f64 {
        (-(frames as f64) / (ms * 0.001 * self.fs)).exp()
    }

    #[inline(always)]
    fn filter(&mut self, k: usize, x: f64) -> f64 {
        let mut v = x;
        for (c, z) in self.hp[k].iter().zip(self.z[k].iter_mut()) {
            let y = c.b0 * v + z[0];
            z[0] = c.b1 * v - c.a1 * y + z[1];
            z[1] = c.b2 * v - c.a2 * y;
            v = y;
        }
        v
    }

    /// Keep the state out of the subnormal range.
    fn flush(&mut self) {
        for z in self.z.iter_mut().flatten().flatten() {
            if z.abs() < 1e-30 {
                *z = 0.0;
            }
        }
        if self.full < 1e-30 {
            self.full = 0.0;
        }
        for e in &mut self.above {
            if *e < 1e-30 {
                *e = 0.0;
            }
        }
    }

    /// Index of the estimated edge in [`CANDIDATE_EDGES_HZ`]: the lowest
    /// candidate above which there is no content (and none above any
    /// higher one); `None` for full band.
    pub fn edge(&self) -> Option<usize> {
        let mut edge = None;
        for k in (0..self.used).rev() {
            if self.present[k] {
                break;
            }
            edge = Some(k);
        }
        edge
    }

    /// Energy above candidate `k` relative to the full band (dB), from the
    /// slow estimates (−300 when there is none).
    pub fn above_db(&self, k: usize) -> f64 {
        if k >= self.used || self.above[k] <= 0.0 || self.full <= 0.0 {
            return -300.0;
        }
        10.0 * (self.above[k] / self.full).log10()
    }

    /// The estimated edge (Hz), `None` for full band.
    pub fn edge_hz(&self) -> Option<f64> {
        self.edge().map(|k| CANDIDATE_EDGES_HZ[k])
    }
}
