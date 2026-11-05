//! Linear-prediction gain: how predictable the programme is, as one control
//! value per block.
//!
//! The mono sum of the channels is low-passed at [`BAND_HZ`] (8th-order
//! Butterworth, the anti-alias filter: −17 dB at 4.5 kHz, −25 dB at 5 kHz,
//! −37 dB at 6 kHz, whatever the input rate) and resampled to exactly
//! [`ANALYSIS_RATE_HZ`], the classic narrowband rate of speech LPC, so the
//! measure describes the programme below 4 kHz and does not depend on
//! where its upper band edge lies (a band-limited signal is predictable
//! whatever it carries). The resampling picks the analysis samples from the
//! filtered signal by 4-point cubic (Catmull–Rom) interpolation at the
//! fractional positions `k·fs/8000`; at a rate that is a whole multiple of
//! 8 kHz (16, 32, 48, 96, 192 kHz) every position falls on an input sample
//! and it is plain decimation. The analysis, its window and its thresholds
//! are therefore the same at every input rate above 8 kHz; at 8 kHz and
//! below the filtered input is analysed as it is (band edge at most
//! 0.45·fs). Pre-emphasised with `1 − μ·z⁻¹`, it feeds an
//! exponentially windowed autocorrelation of lags `0 … P` (each lag a
//! one-pole sum of `y[n]·y[n−k]`, time constant [`WINDOW_MS`]; Barnwell,
//! "Recursive windowing for generating autocorrelation coefficients for LPC
//! analysis", IEEE Trans. ASSP 29(5), 1981). Once per block the
//! Levinson–Durbin recursion (Makhoul, "Linear prediction: a tutorial
//! review", Proc. IEEE 63(4), 1975) gives the order-`P` prediction error
//! power `E_P`, and the prediction gain is `G = 10·log10(r₀ / E_P)` dB.
//!
//! Speech and tones are predictable (`G` well over 10 dB in voiced speech);
//! noise, crowd and applause are not (`G` near 0 dB after the
//! pre-emphasis). Programme with less than [`BAND_SHARE_MIN`] (−30 dB) of
//! its power in the analysed band (a high tone or twin-tone, nothing
//! below 3.5 kHz) cannot be judged there and reads as predictable
//! ([`MAX_GAIN_DB`]): the clipper then leaves it to the limiter, which
//! keeps the difference tones that clipping two high tones creates out of
//! the audible band. Until the window has seen one time constant of analysis
//! samples after a reset, too few for an order-`P` estimate, the gain
//! reads 0 dB (not predictable: the plain clipper's behaviour).
//! Using it to allow more clipping on less predictable
//! material follows the expired US 4,241,266 (expired; status not legally
//! reviewed).
//!
//! # Real-time safety
//!
//! No allocation; fixed-size state. Per input frame four biquads; per
//! analysis sample a cubic interpolation and the recursion (`P + 1` multiply-adds); Levinson–Durbin
//! once per [`Predictability::gain_db`] call. Quiet input (`r₀` under
//! `QUIET`, 1e-10) reads 0 dB.

use crate::biquad::BiquadCoefficients;

/// Upper edge of the analysed band (Hz).
pub const BAND_HZ: f64 = 3_500.0;
/// Rate of the analysis (Hz), exact at every input rate above it.
pub const ANALYSIS_RATE_HZ: f64 = 8_000.0;
/// Second-order sections of the anti-alias low-pass (8th-order
/// Butterworth).
const LP_SECTIONS: usize = 4;
/// Prediction order `P`.
pub const ORDER: usize = 10;
/// Time constant of the autocorrelation window (ms).
pub const WINDOW_MS: f64 = 20.0;
/// Pre-emphasis coefficient `μ`.
pub const PRE_EMPHASIS: f64 = 0.9;
/// Autocorrelation `r₀` (mean square of the pre-emphasised mono sum) under
/// which the input counts as quiet.
const QUIET: f64 = 1e-10;
/// Largest reported gain (dB).
pub const MAX_GAIN_DB: f64 = 40.0;
/// Share of the programme's power (−30 dB) under which the analysed band
/// counts as empty: the programme lives above it, and reads as
/// predictable ([`MAX_GAIN_DB`]).
pub const BAND_SHARE_MIN: f64 = 1e-3;

/// Running linear-prediction gain of the mono sum of a signal.
#[derive(Debug, Clone)]
pub struct Predictability {
    lp: [BiquadCoefficients; LP_SECTIONS],
    /// Direct-form II transposed state of the low-pass sections.
    z: [[f64; 2]; LP_SECTIONS],
    /// Input frames per analysis sample (`fs / 8000`, at least 1).
    step: f64,
    /// Position of the next analysis sample, in input frames after
    /// `taps[1]` (the interpolation runs between `taps[1]` and `taps[2]`).
    pos: f64,
    /// The last four filtered samples, oldest first.
    taps: [f64; 4],
    lambda: f64,
    r: [f64; ORDER + 1],
    hist: [f64; ORDER],
    pre_prev: f64,
    /// Windowed mean squares of the mono sum: the whole band (per input
    /// frame) and the analysed band (per analysis sample, before the
    /// pre-emphasis), and the full band's window coefficient.
    full_ms: f64,
    band_ms: f64,
    lambda_in: f64,
    /// Analysis samples since the reset (saturating) and the count after
    /// which the window is full enough to read.
    seen: usize,
    warm: usize,
}

impl Default for Predictability {
    fn default() -> Self {
        Self::new(48_000.0)
    }
}

impl Predictability {
    /// A fresh estimator for `sample_rate`.
    pub fn new(sample_rate: f64) -> Self {
        let fs = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate
        } else {
            48_000.0
        };
        let step = (fs / ANALYSIS_RATE_HZ).max(1.0);
        let rate = fs / step;
        let band = BAND_HZ.min(0.45 * fs);
        // Butterworth: section k of an order-2N filter has
        // Q = 1 / (2·cos((2k − 1)·π / 4N)).
        let n = 2 * LP_SECTIONS;
        let lp = core::array::from_fn(|k| {
            let q =
                1.0 / (2.0 * ((2 * k + 1) as f64 * core::f64::consts::PI / (2 * n) as f64).cos());
            BiquadCoefficients::low_pass(band, q, fs)
        });
        let window = WINDOW_MS * 0.001 * rate;
        Self {
            lp,
            z: [[0.0; 2]; LP_SECTIONS],
            step,
            pos: 1.0,
            taps: [0.0; 4],
            lambda: (-1.0 / window).exp(),
            r: [0.0; ORDER + 1],
            hist: [0.0; ORDER],
            pre_prev: 0.0,
            full_ms: 0.0,
            band_ms: 0.0,
            lambda_in: (-1.0 / (WINDOW_MS * 0.001 * fs)).exp(),
            seen: 0,
            warm: (window.round() as usize).max(4 * ORDER),
        }
    }

    /// Clear the state.
    pub fn reset(&mut self) {
        self.z = [[0.0; 2]; LP_SECTIONS];
        self.pos = 1.0;
        self.taps = [0.0; 4];
        self.r = [0.0; ORDER + 1];
        self.hist = [0.0; ORDER];
        self.pre_prev = 0.0;
        self.full_ms = 0.0;
        self.band_ms = 0.0;
        self.seen = 0;
    }

    /// Feed `x` (interleaved, `ch` channels).
    pub fn push(&mut self, x: &[f32], ch: usize) {
        if ch == 0 {
            return;
        }
        let l = self.lambda;
        let w = 1.0 - l;
        let inv = 1.0 / ch as f64;
        for frame in x.chunks_exact(ch) {
            let m = frame.iter().map(|&v| v as f64).sum::<f64>() * inv;
            let mut m = if m.is_finite() { m } else { 0.0 };
            self.full_ms = self.lambda_in * self.full_ms + (1.0 - self.lambda_in) * m * m;
            for (c, z) in self.lp.iter().zip(self.z.iter_mut()) {
                let y = c.b0 * m + z[0];
                z[0] = c.b1 * m - c.a1 * y + z[1];
                z[1] = c.b2 * m - c.a2 * y;
                m = y;
            }
            self.taps = [self.taps[1], self.taps[2], self.taps[3], m];
            self.pos -= 1.0;
            while self.pos < 1.0 {
                let a = catmull_rom(&self.taps, self.pos);
                self.pos += self.step;
                self.band_ms = l * self.band_ms + w * a * a;
                let y = a - PRE_EMPHASIS * self.pre_prev;
                self.pre_prev = a;
                self.r[0] = l * self.r[0] + w * y * y;
                for k in 1..=ORDER {
                    self.r[k] = l * self.r[k] + w * y * self.hist[k - 1];
                }
                self.hist.copy_within(0..ORDER - 1, 1);
                self.hist[0] = y;
                self.seen = self.seen.saturating_add(1);
            }
        }
    }

    /// Keep the sums and the filter state out of the subnormal range in
    /// silence. Call it at fixed points of the stream (e.g. once per control
    /// block), so the result does not depend on how the input was split.
    pub fn flush_denormals(&mut self) {
        if self.r[0] < 1e-30 {
            self.r = [0.0; ORDER + 1];
            self.hist = [0.0; ORDER];
            self.pre_prev = 0.0;
            self.seen = 0;
        }
        for v in [&mut self.full_ms, &mut self.band_ms] {
            if *v < 1e-30 {
                *v = 0.0;
            }
        }
        for v in self.taps.iter_mut() {
            if v.abs() < 1e-30 {
                *v = 0.0;
            }
        }
        for z in self.z.iter_mut().flatten() {
            if z.abs() < 1e-30 {
                *z = 0.0;
            }
        }
    }

    /// The prediction gain (dB, 0 … [`MAX_GAIN_DB`]) of the current window.
    pub fn gain_db(&self) -> f64 {
        let r = &self.r;
        if self.seen < self.warm {
            return 0.0;
        }
        // Programme that lives above the analysed band (a high tone, a
        // high twin-tone): nothing to judge it by, read as predictable.
        if self.full_ms > QUIET && self.band_ms < BAND_SHARE_MIN * self.full_ms {
            return MAX_GAIN_DB;
        }
        if r[0] < QUIET {
            return 0.0;
        }
        // Levinson–Durbin; a slight lag window on r₀ (white-noise
        // correction, 40 dB) keeps the recursion well conditioned. The
        // recursive window does not guarantee a positive definite estimate:
        // a reflection coefficient at or over 1 means the window is (close
        // to) a sum of pure tones, which reads as fully predictable.
        let mut e = r[0] * (1.0 + 1e-4);
        let mut a = [0.0f64; ORDER + 1];
        let mut tmp = [0.0f64; ORDER + 1];
        a[0] = 1.0;
        for i in 1..=ORDER {
            let mut acc = r[i];
            for j in 1..i {
                acc += a[j] * r[i - j];
            }
            let k = -acc / e;
            if !k.is_finite() || k.abs() >= 1.0 {
                return MAX_GAIN_DB;
            }
            tmp[..=i].copy_from_slice(&a[..=i]);
            for j in 1..i {
                a[j] = tmp[j] + k * tmp[i - j];
            }
            a[i] = k;
            e *= 1.0 - k * k;
        }
        let g = 10.0 * (r[0] / e.max(1e-300)).log10();
        if g.is_finite() {
            g.clamp(0.0, MAX_GAIN_DB)
        } else {
            0.0
        }
    }
}

/// Catmull–Rom cubic through `p[1]` (at 0) and `p[2]` (at 1), with
/// tangents from the outer points; `t` in `[0, 1)`.
#[inline(always)]
fn catmull_rom(p: &[f64; 4], t: f64) -> f64 {
    let c1 = 0.5 * (p[2] - p[0]);
    let c2 = p[0] - 2.5 * p[1] + 2.0 * p[2] - 0.5 * p[3];
    let c3 = 0.5 * (p[3] - p[0]) + 1.5 * (p[1] - p[2]);
    ((c3 * t + c2) * t + c1) * t + p[1]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gain_of(x: &[f32]) -> f64 {
        let mut p = Predictability::new(48_000.0);
        p.push(x, 1);
        p.gain_db()
    }

    #[test]
    fn tone_is_predictable_noise_is_not() {
        let n = 9_600;
        let tone: Vec<f32> = (0..n)
            .map(|i| {
                (0.3 * (2.0 * core::f64::consts::PI * 440.0 * i as f64 / 48_000.0).sin()) as f32
            })
            .collect();
        // Deterministic white noise (xorshift).
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let noise: Vec<f32> = (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32 * 0.5
            })
            .collect();
        let gt = gain_of(&tone);
        let gn = gain_of(&noise);
        assert!(gt > 30.0, "tone {gt:.1} dB");
        // White noise pre-emphasised is coloured; most of that is predicted.
        assert!(gn < 8.0, "noise {gn:.1} dB");
        assert_eq!(gain_of(&vec![0.0; n]), 0.0);
    }

    /// Deterministic white noise (xorshift), uniform in ±0.25.
    fn white(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    }

    /// `x` through `c` (direct form I).
    fn filter(c: &BiquadCoefficients, x: &[f64]) -> Vec<f64> {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
        x.iter()
            .map(|&v| {
                let y = c.b0 * v + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
                (x2, x1, y2, y1) = (x1, v, y1, y);
                y
            })
            .collect()
    }

    /// The test material at rate `fs`, 1 s each: a vowel (harmonics of
    /// 120 Hz shaped by formants at 600 and 1 400 Hz, up to 0.45·fs),
    /// white noise, white noise through a band-pass at 900 Hz (Q 2; the
    /// same analogue response at every rate), and white noise with an
    /// 8th-order low-pass at 3 kHz (band-limited, like a phone source).
    fn material(fs: f64) -> [(&'static str, Vec<f32>); 4] {
        let n = (1.0 * fs) as usize;
        let vowel = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let mut s = 0.0;
                let mut k = 1;
                while 120.0 * k as f64 <= 0.45 * fs && k <= 60 {
                    let f = 120.0 * k as f64;
                    let formant = |fc: f64, bw: f64| 1.0 / (1.0 + ((f - fc) / bw).powi(2));
                    let a = formant(600.0, 150.0) + 0.5 * formant(1_400.0, 200.0);
                    s += a * (2.0 * core::f64::consts::PI * f * t + 0.3 * k as f64).sin();
                    k += 1;
                }
                (0.1 * s) as f32
            })
            .collect();
        let w = white(n, 0x2545_f491_4f6c_dd1d);
        let noise = w.iter().map(|&v| v as f32).collect();
        let bp = BiquadCoefficients::band_pass_peak(900.0, 2.0, fs);
        let coloured = filter(&bp, &w).iter().map(|&v| v as f32).collect();
        let mut lp = w.clone();
        for k in 0..4 {
            let q = 1.0 / (2.0 * ((2 * k + 1) as f64 * core::f64::consts::PI / 16.0).cos());
            lp = filter(&BiquadCoefficients::low_pass(3_000.0, q, fs), &lp);
        }
        let phone = lp.iter().map(|&v| v as f32).collect();
        [
            ("vowel", vowel),
            ("noise", noise),
            ("band-pass noise", coloured),
            ("3 kHz noise", phone),
        ]
    }

    /// The same material reads the same prediction gain at every rate:
    /// within 0.6 dB of the 48 kHz reading for material near the clipper's
    /// thresholds (2 and 5 dB), well above them for material that reads
    /// far above them at 48 kHz.
    #[test]
    fn gain_is_independent_of_the_sample_rate() {
        let rates = [16_000.0, 22_050.0, 44_100.0, 48_000.0, 96_000.0, 192_000.0];
        let read = |fs: f64| -> Vec<(&'static str, f64)> {
            material(fs)
                .into_iter()
                .map(|(name, x)| {
                    // Mean of readings every 10 ms after the first 100 ms:
                    // one reading of a 20 ms window of noise varies by
                    // more than the tolerance.
                    let mut p = Predictability::new(fs);
                    let hop = (0.01 * fs) as usize;
                    let (mut sum, mut k) = (0.0, 0);
                    for (i, b) in x.chunks(hop).enumerate() {
                        p.push(b, 1);
                        if i >= 10 {
                            sum += p.gain_db();
                            k += 1;
                        }
                    }
                    (name, sum / k as f64)
                })
                .collect()
        };
        const TOL_DB: f64 = 0.6;
        let reference = read(48_000.0);
        for fs in rates {
            let got = read(fs);
            for ((name, g), (_, g48)) in got.iter().zip(&reference) {
                println!("{fs:>7} Hz {name:>16}: {g:6.2} dB (48 kHz {g48:6.2})");
                if *g48 < 10.0 {
                    assert!(
                        (g - g48).abs() < TOL_DB,
                        "{name} at {fs} Hz: {g:.2} dB against {g48:.2}"
                    );
                } else {
                    assert!(*g > 20.0, "{name} at {fs} Hz: {g:.2} dB against {g48:.2}");
                }
            }
        }
        let g = |n: &str| reference.iter().find(|(k, _)| *k == n).unwrap().1;
        assert!(g("vowel") > 20.0);
        assert!(g("noise") < 3.0 && g("3 kHz noise") > g("noise") + 1.0);
    }

    /// Programme with nothing in the analysed band (a 15 kHz tone, a
    /// 19 + 20 kHz twin-tone) reads as predictable at every rate that
    /// carries it; white noise, which has some, does not.
    #[test]
    fn programme_above_the_band_reads_as_predictable() {
        for fs in [44_100.0, 48_000.0, 96_000.0, 192_000.0] {
            let n = (0.2 * fs) as usize;
            let tone = |f: f64, i: usize| (2.0 * core::f64::consts::PI * f * i as f64 / fs).sin();
            let high: Vec<f32> = (0..n).map(|i| (0.5 * tone(15_000.0, i)) as f32).collect();
            let twin: Vec<f32> = (0..n)
                .map(|i| (0.25 * (tone(19_000.0, i) + tone(20_000.0, i))) as f32)
                .collect();
            for (name, x) in [("15 kHz", high), ("19 + 20 kHz", twin)] {
                let mut p = Predictability::new(fs);
                p.push(&x, 1);
                assert_eq!(p.gain_db(), MAX_GAIN_DB, "{name} at {fs} Hz");
            }
            let noise: Vec<f32> = white(n, 5).iter().map(|&v| v as f32).collect();
            let mut p = Predictability::new(fs);
            p.push(&noise, 1);
            assert!(p.gain_db() < 5.0, "noise at {fs} Hz: {:.2} dB", p.gain_db());
        }
    }

    /// Right after a reset the window is too short for an order-10
    /// estimate: noise reads 0 dB until it is full, then stays low.
    #[test]
    fn noise_is_not_predictable_while_the_window_fills() {
        let fs = 48_000.0;
        let mut worst = 0.0f64;
        for seed in 1..=16 {
            let x: Vec<f32> = white(4_800, seed).iter().map(|&v| v as f32).collect();
            let mut p = Predictability::new(fs);
            for b in x.chunks(64) {
                p.push(b, 1);
                worst = worst.max(p.gain_db());
            }
        }
        println!("worst reading on noise from a reset: {worst:.2} dB");
        assert!(worst < 4.0, "noise read as {worst:.2} dB");
    }
}
