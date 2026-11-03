//! Synthetic test signals for the loudness detector and AGC tests (fixed
//! seeds, no recordings).
//!
//! Pink noise is made in the frequency domain: every FFT bin between 20 Hz
//! and 20 kHz (and below 0.45·fs) gets magnitude `1/√f` and a uniformly random
//! phase, then one inverse real FFT. The result has an exact 1/f power
//! spectrum and is periodic in the FFT length, so it loops without a seam.

#![allow(dead_code)]

use realfft::num_complex::Complex;
use realfft::RealFftPlanner;
use strom_dsp::meter::{ChannelLayout, LoudnessConfig, LoudnessMeter};

/// xorshift64* generator.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in [−1, 1).
    pub fn bipolar(&mut self) -> f64 {
        2.0 * self.uniform() - 1.0
    }
}

/// One channel of band-limited noise of `frames` (a power of two) frames
/// with power spectrum ∝ `f^(−slope)` (0 = white, 1 = pink), unit RMS.
fn coloured_channel(frames: usize, fs: f64, slope: f64, rng: &mut Rng) -> Vec<f64> {
    let mut planner = RealFftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(frames);
    let mut spec = ifft.make_input_vec();
    let hi = 20_000.0f64.min(0.45 * fs);
    for (k, bin) in spec.iter_mut().enumerate() {
        let f = k as f64 * fs / frames as f64;
        *bin = if k > 0 && k < frames / 2 && (20.0..=hi).contains(&f) {
            let phase = 2.0 * std::f64::consts::PI * rng.uniform();
            Complex::from_polar(f.powf(-0.5 * slope), phase)
        } else {
            Complex::new(0.0, 0.0)
        };
    }
    let mut out = ifft.make_output_vec();
    ifft.process(&mut spec, &mut out).expect("inverse FFT");
    let rms = (out.iter().map(|v| v * v).sum::<f64>() / frames as f64).sqrt();
    out.iter_mut().for_each(|v| *v /= rms);
    out
}

/// Interleaved noise, independent per channel, unit RMS per channel.
pub fn noise(frames: usize, channels: usize, fs: f64, slope: f64, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let chans: Vec<Vec<f64>> = (0..channels)
        .map(|_| coloured_channel(frames, fs, slope, &mut rng))
        .collect();
    (0..frames)
        .flat_map(|n| chans.iter().map(move |c| c[n] as f32))
        .collect()
}

/// Pink noise (see [`noise`]).
pub fn pink(frames: usize, channels: usize, fs: f64, seed: u64) -> Vec<f32> {
    noise(frames, channels, fs, 1.0, seed)
}

/// Integrated loudness of `buf` (looped `loops` times) measured by the
/// `ebur128`-based reference meter.
pub fn reference_loudness(buf: &[f32], channels: usize, fs: f64, layout: ChannelLayout) -> f64 {
    let mut m = LoudnessMeter::new(&LoudnessConfig {
        sample_rate: fs as f32,
        channels,
        layout,
        true_peak: false,
    })
    .unwrap();
    // Two passes, so the 400 ms gating blocks see steady-state filters.
    m.process(buf);
    m.reset();
    m.process(buf);
    m.integrated()
}

/// Scale `buf` in place so its reference loudness is `lufs`.
pub fn scale_to_lufs(buf: &mut [f32], channels: usize, fs: f64, lufs: f64) {
    let now = reference_loudness(buf, channels, fs, ChannelLayout::Unknown);
    let g = 10f64.powf((lufs - now) / 20.0) as f32;
    buf.iter_mut().for_each(|v| *v *= g);
}

/// Plays a buffer in a loop.
pub struct Looper<'a> {
    buf: &'a [f32],
    pos: usize,
}

impl<'a> Looper<'a> {
    pub fn new(buf: &'a [f32]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Fill `out` with the next samples (interleaved, whole frames).
    pub fn fill(&mut self, out: &mut [f32]) {
        for v in out.iter_mut() {
            *v = self.buf[self.pos];
            self.pos += 1;
            if self.pos == self.buf.len() {
                self.pos = 0;
            }
        }
    }
}

/// Sine of `amp` at `freq` Hz on `channel` of `channels`, others silent.
pub fn sine_on(
    frames: usize,
    channels: usize,
    channel: usize,
    fs: f64,
    freq: f64,
    amp: f64,
) -> Vec<f32> {
    let mut out = vec![0.0f32; frames * channels];
    for n in 0..frames {
        out[n * channels + channel] =
            (amp * (2.0 * std::f64::consts::PI * freq * n as f64 / fs).sin()) as f32;
    }
    out
}

/// A hot, peaky, sport-like programme (a ~10.9 s loop at 48 kHz, fixed
/// seed): a pink-noise crowd bed; percussive hits every ~250 ms (a decaying
/// white-noise burst plus a decaying 60 Hz thump); a 5 ms fs/4 sine burst
/// at 45° phase (sample peaks 3 dB under its true peak) every ~2.3 s; and
/// a single-sample click every `click_s` seconds.
pub fn peaky_programme(ch: usize, fs: f64, click_s: f64) -> Vec<f32> {
    let frames = 1 << 19;
    let mut x = pink(frames, ch, fs, 31);
    x.iter_mut().for_each(|v| *v *= 0.12);
    let mut rng = Rng::new(17);
    // Hits: every 250 ms ± 20 ms.
    let mut at = 0.0f64;
    while at < frames as f64 / fs - 0.3 {
        let start = (at * fs) as usize;
        let amp = 0.42 + 0.18 * rng.uniform();
        for k in 0..(0.25 * fs) as usize {
            let n = start + k;
            if n >= frames {
                break;
            }
            let t = k as f64 / fs;
            let thump = (2.0 * std::f64::consts::PI * 60.0 * t).sin() * (-t / 0.06).exp();
            let burst = (-t / 0.012).exp();
            for c in 0..ch {
                let v = amp * (thump + burst * rng.bipolar());
                x[n * ch + c] += v as f32;
            }
        }
        at += 0.25 + 0.04 * (rng.uniform() - 0.5);
    }
    // fs/4 bursts at 45° phase, 5 ms, every 2.3 s.
    let burst_len = (0.005 * fs) as usize;
    let mut start = (0.7 * fs) as usize;
    while start + burst_len < frames {
        for k in 0..burst_len {
            let v = (1.2
                * (std::f64::consts::PI / 2.0 * k as f64 + std::f64::consts::PI / 4.0).sin())
                as f32;
            for c in 0..ch {
                x[(start + k) * ch + c] += v;
            }
        }
        start += (2.3 * fs) as usize;
    }
    // Single-sample clicks every `click_s`.
    let mut n = (0.37 * fs) as usize;
    let mut sign = 1.0f32;
    while n < frames {
        for c in 0..ch {
            x[n * ch + c] += 4.0 * sign;
        }
        sign = -sign;
        n += (click_s * fs) as usize;
    }
    x
}
