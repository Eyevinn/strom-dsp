//! Windowed FFT magnitude spectrum with sine-calibrated levels.
//!
//! A 4-term Blackman-Harris window (Harris, "On the use of windows for
//! harmonic analysis with the discrete Fourier transform", Proc. IEEE
//! 66(1), 1978): highest side lobe −92 dB, main lobe ±4 bins, so spurs
//! 80 dB under a tone stay visible next to it. Levels are scaled so a sine
//! of amplitude 1 reads 0 dB at its bin (scalloping loss ≤ 0.83 dB between
//! bins; [`Spectrum::level_db`] takes the largest bin near a frequency).

use std::f64::consts::PI;

use realfft::RealFftPlanner;

/// Floor of every level (dB), instead of −∞.
pub const FLOOR_DB: f64 = -300.0;

/// Magnitude spectrum of one channel segment.
#[derive(Debug, Clone)]
pub struct Spectrum {
    /// Sine-calibrated amplitude per bin (0 … len/2).
    amp: Vec<f64>,
    bin_hz: f64,
}

fn db(x: f64) -> f64 {
    if x > 0.0 {
        (20.0 * x.log10()).max(FLOOR_DB)
    } else {
        FLOOR_DB
    }
}

impl Spectrum {
    /// Spectrum of `len` frames of channel `channel` of interleaved `x`
    /// (`ch` channels) starting at frame `from`. `len` should be a power of
    /// two; frames beyond the end of `x` count as zero.
    pub fn of(x: &[f32], ch: usize, channel: usize, fs: f64, from: usize, len: usize) -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(len);
        let mut buf = fft.make_input_vec();
        let mut wsum = 0.0;
        for (i, b) in buf.iter_mut().enumerate() {
            let p = 2.0 * PI * i as f64 / len as f64;
            let w =
                0.35875 - 0.48829 * p.cos() + 0.14128 * (2.0 * p).cos() - 0.01168 * (3.0 * p).cos();
            wsum += w;
            let idx = (from + i) * ch + channel;
            *b = x.get(idx).map_or(0.0, |&v| v as f64) * w;
        }
        let mut spec = fft.make_output_vec();
        fft.process(&mut buf, &mut spec).expect("forward FFT");
        let scale = 2.0 / wsum;
        Self {
            amp: spec.iter().map(|c| c.norm() * scale).collect(),
            bin_hz: fs / len as f64,
        }
    }

    /// Frequency resolution (Hz per bin).
    pub fn bin_hz(&self) -> f64 {
        self.bin_hz
    }

    /// Level (dB re a unit sine) of a tone at `f`: the largest bin within
    /// ±2 bins of it.
    pub fn level_db(&self, f: f64) -> f64 {
        let k = (f / self.bin_hz).round() as isize;
        let lo = (k - 2).max(0) as usize;
        let hi = ((k + 2).max(0) as usize).min(self.amp.len() - 1);
        db(self.amp[lo..=hi].iter().fold(0.0f64, |m, &v| m.max(v)))
    }

    /// Largest level (dB) over the bins whose centre frequency satisfies
    /// `keep`.
    pub fn max_db_where(&self, keep: impl Fn(f64) -> bool) -> f64 {
        db(self
            .amp
            .iter()
            .enumerate()
            .filter(|(k, _)| keep(*k as f64 * self.bin_hz))
            .fold(0.0f64, |m, (_, &v)| m.max(v)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_reads_its_amplitude_and_side_lobes_are_low() {
        let fs = 48_000.0;
        let len = 1 << 14;
        for (f, a) in [(1000.0, 1.0), (15_003.7, 0.25), (60.0, 0.5)] {
            let x: Vec<f32> = (0..len)
                .map(|n| (a * (2.0 * PI * f * n as f64 / fs).sin()) as f32)
                .collect();
            let s = Spectrum::of(&x, 1, 0, fs, 0, len);
            let want = 20.0 * f64::log10(a);
            assert!((s.level_db(f) - want).abs() < 0.9, "{f}: {}", s.level_db(f));
            let away = s.max_db_where(|g| (g - f).abs() > 6.0 * s.bin_hz());
            assert!(away < want - 85.0, "{f}: leakage {away}");
        }
    }
}
