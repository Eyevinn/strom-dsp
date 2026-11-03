//! Measurement helpers for the expander and bass-clipper tests: synthetic
//! signals with fixed seeds, and level / spectral measurements. They are
//! written so they can move into a shared measurement harness unchanged.

#![allow(dead_code)]

use std::f64::consts::PI;

/// Small deterministic generator (xorshift64).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `[−1, 1)`.
    pub fn bipolar(&mut self) -> f64 {
        2.0 * self.uniform() - 1.0
    }
}

pub fn db(x: f64) -> f64 {
    20.0 * x.max(1e-30).log10()
}

pub fn lin(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

/// Mono sine.
pub fn sine(frames: usize, fs: f64, f: f64, amp: f64) -> Vec<f32> {
    (0..frames)
        .map(|n| (amp * (2.0 * PI * f * n as f64 / fs).sin()) as f32)
        .collect()
}

/// Interleave a mono signal onto `ch` channels.
pub fn to_channels(mono: &[f32], ch: usize) -> Vec<f32> {
    mono.iter()
        .flat_map(|&v| std::iter::repeat_n(v, ch))
        .collect()
}

/// Channel `c` of interleaved `x`.
pub fn channel(x: &[f32], ch: usize, c: usize) -> Vec<f32> {
    x.iter().skip(c).step_by(ch).copied().collect()
}

/// White noise, uniform in `[−amp, amp)`.
pub fn white(frames: usize, amp: f64, seed: u64) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..frames).map(|_| (amp * r.bipolar()) as f32).collect()
}

/// Second-order Butterworth low-pass (bilinear, pre-warped), in place;
/// `passes` times in series. For band-limiting test noise only.
pub fn lowpass(x: &mut [f32], fs: f64, fc: f64, passes: usize) {
    let k = (PI * fc / fs).tan();
    let q = std::f64::consts::FRAC_1_SQRT_2;
    let norm = 1.0 / (1.0 + k / q + k * k);
    let b0 = k * k * norm;
    let (b1, b2) = (2.0 * b0, b0);
    let a1 = 2.0 * (k * k - 1.0) * norm;
    let a2 = (1.0 - k / q + k * k) * norm;
    for _ in 0..passes {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
        for v in x.iter_mut() {
            let x0 = *v as f64;
            let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;
            *v = y0 as f32;
        }
    }
}

/// RMS (linear) of `x`.
pub fn rms(x: &[f32]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len() as f64).sqrt()
}

/// Peak magnitude of `x`.
pub fn peak(x: &[f32]) -> f64 {
    x.iter().fold(0.0f64, |m, &v| m.max((v as f64).abs()))
}

/// Amplitude of the component at `f` (Hz) in `x`, Hann-windowed DFT
/// projection (window gain compensated). Resolution about `2·fs/len`.
pub fn tone_amplitude(x: &[f32], fs: f64, f: f64) -> f64 {
    let n = x.len();
    let w = 2.0 * PI * f / fs;
    let (mut re, mut im, mut wsum) = (0.0, 0.0, 0.0);
    for (i, &v) in x.iter().enumerate() {
        let win = 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos();
        re += win * v as f64 * (w * i as f64).cos();
        im -= win * v as f64 * (w * i as f64).sin();
        wsum += win;
    }
    2.0 * (re * re + im * im).sqrt() / wsum
}

/// Speech-like bursts: band-limited noise (two passes of a 2nd-order
/// low-pass at `bandwidth` Hz) under a raised-cosine envelope with 20 ms
/// ramps, `burst_s` on and `gap_s` off, scaled so the burst sections have
/// RMS `burst_rms`; between the bursts, white noise of RMS `floor_rms`.
/// Returns the signal and the `(start, end)` frame ranges of each burst.
pub fn speech_like_bursts(
    frames: usize,
    fs: f64,
    bandwidth: f64,
    burst_s: f64,
    gap_s: f64,
    burst_rms: f64,
    floor_rms: f64,
    seed: u64,
) -> (Vec<f32>, Vec<(usize, usize)>) {
    let mut voice = white(frames, 1.0, seed);
    lowpass(&mut voice, fs, bandwidth, 2);
    let v_rms = rms(&voice).max(1e-12);
    // Uniform noise of amplitude a has RMS a/√3.
    let floor = white(frames, floor_rms * 3f64.sqrt(), seed ^ 0x5eed);
    let on = (burst_s * fs) as usize;
    let off = (gap_s * fs) as usize;
    let ramp = (0.02 * fs) as usize;
    let mut out = floor;
    let mut ranges = Vec::new();
    let mut start = off;
    while start + on <= frames {
        for i in 0..on {
            let e = if i < ramp {
                0.5 - 0.5 * (PI * i as f64 / ramp as f64).cos()
            } else if i >= on - ramp {
                0.5 - 0.5 * (PI * (on - i) as f64 / ramp as f64).cos()
            } else {
                1.0
            };
            out[start + i] += (e * burst_rms / v_rms * voice[start + i] as f64) as f32;
        }
        ranges.push((start, start + on));
        start += on + off;
    }
    (out, ranges)
}

/// The bass-into-highs pair: a loud `f_bass` sine and a soft `f_high` sine.
pub fn bass_into_highs(
    frames: usize,
    fs: f64,
    f_bass: f64,
    bass_amp: f64,
    f_high: f64,
    high_amp: f64,
) -> Vec<f32> {
    (0..frames)
        .map(|n| {
            let t = n as f64 / fs;
            (bass_amp * (2.0 * PI * f_bass * t).sin() + high_amp * (2.0 * PI * f_high * t).sin())
                as f32
        })
        .collect()
}
