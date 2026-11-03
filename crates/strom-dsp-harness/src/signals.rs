//! The harness signals: synthetic only, fixed seeds, generated in code.
//!
//! Every generator returns interleaved stereo (`CH` channels) of exactly the
//! requested number of frames, at an arbitrary scale: the case runner
//! scales it to a loudness or a peak level. A [`Signal`] also carries
//! **marks**: frames of short events (clicks) that a metric may treat
//! apart from the programme ("on peaks" versus "on programme").

use std::f64::consts::PI;

use realfft::num_complex::Complex;
use realfft::RealFftPlanner;

use crate::agc_support::Rng;
use crate::programme_support;

/// Channels of every harness signal.
pub const CH: usize = 2;

/// A generated signal.
#[derive(Debug, Clone)]
pub struct Signal {
    /// Interleaved samples, [`CH`] channels.
    pub data: Vec<f32>,
    /// Frames of short events (clicks), ascending.
    pub marks: Vec<usize>,
}

impl Signal {
    fn plain(data: Vec<f32>) -> Self {
        Self {
            data,
            marks: Vec::new(),
        }
    }

    /// Number of frames.
    pub fn frames(&self) -> usize {
        self.data.len() / CH
    }
}

/// The signals of the harness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalKind {
    /// Sine at a frequency (Hz).
    Sine(f64),
    /// Logarithmic sweep 20 Hz … 20 kHz over the whole signal.
    LogSweep,
    /// SMPTE RP 120 pair: 60 Hz and 7 kHz, amplitudes 4:1.
    SmptePair,
    /// CCIF twin tone: 19 kHz and 20 kHz, equal amplitudes.
    CcifPair,
    /// A loud 50 Hz tone with a soft 3 kHz tone (−20 dB).
    BassIntoHighs,
    /// Single-sample clicks every `spacing` seconds over speech-like noise.
    SpeechClicks(f64),
    /// Speech-like noise (syllable and word envelope, room tone between
    /// words), band-limited to this upper frequency (Hz).
    SpeechNoise(f64),
    /// Band-limited 100 Hz square wave through a first-order 100 Hz
    /// high-pass: the flat tops tilt (AC coupling).
    TiltedSquare,
    /// 250 Hz sine hard-clipped at half its amplitude, band-limited to
    /// 20 kHz (Fourier series, no aliasing in the source).
    ClippedSine,
    /// The music-like acceptance programme of the broadcast tests.
    Music,
    /// The hot, peaky, sport-like programme of the broadcast tests, with a
    /// single-sample click every this many seconds.
    Peaky(f64),
}

impl SignalKind {
    /// Generate `frames` frames at `fs`.
    pub fn generate(self, frames: usize, fs: f64) -> Signal {
        match self {
            Self::Sine(f) => Signal::plain(stereo(frames, |t| (2.0 * PI * f * t).sin(), fs)),
            Self::LogSweep => Signal::plain(log_sweep(frames, fs, 20.0, 20_000.0)),
            Self::SmptePair => Signal::plain(stereo(
                frames,
                |t| 0.8 * (2.0 * PI * 60.0 * t).sin() + 0.2 * (2.0 * PI * 7_000.0 * t).sin(),
                fs,
            )),
            Self::CcifPair => Signal::plain(stereo(
                frames,
                |t| 0.5 * (2.0 * PI * 19_000.0 * t).sin() + 0.5 * (2.0 * PI * 20_000.0 * t).sin(),
                fs,
            )),
            Self::BassIntoHighs => Signal::plain(stereo(
                frames,
                |t| 0.9 * (2.0 * PI * 50.0 * t).sin() + 0.09 * (2.0 * PI * 3_000.0 * t).sin(),
                fs,
            )),
            Self::SpeechClicks(spacing) => speech_clicks(frames, fs, spacing),
            Self::SpeechNoise(hi) => Signal::plain(speech_noise(frames, fs, hi, 41)),
            Self::TiltedSquare => Signal::plain(tilted_square(frames, fs)),
            Self::ClippedSine => Signal::plain(clipped_sine(frames, fs)),
            Self::Music => {
                let src = programme_support::music_programme(CH, fs);
                Signal::plain(looped(&src, frames))
            }
            Self::Peaky(click_s) => {
                let src = programme_support::peaky_programme(CH, fs, click_s);
                let one = programme_support::click_frames(fs, click_s);
                let mut marks = Vec::new();
                let mut base = 0;
                while base < frames {
                    marks.extend(one.iter().map(|&m| m + base).filter(|&m| m < frames));
                    base += programme_support::PROGRAMME_FRAMES;
                }
                Signal {
                    data: looped(&src, frames),
                    marks,
                }
            }
        }
    }
}

/// The same mono function of time on every channel.
fn stereo(frames: usize, f: impl Fn(f64) -> f64, fs: f64) -> Vec<f32> {
    (0..frames)
        .flat_map(|n| {
            let v = f(n as f64 / fs) as f32;
            std::iter::repeat_n(v, CH)
        })
        .collect()
}

/// `src` (interleaved [`CH`]) played in a loop for `frames` frames.
fn looped(src: &[f32], frames: usize) -> Vec<f32> {
    src.iter().copied().cycle().take(frames * CH).collect()
}

/// Exponential sine sweep from `f0` to `f1` over the whole signal
/// (Farina's phase law `φ(t) = 2π f0 T/ln(f1/f0) · (e^{t ln(f1/f0)/T} − 1)`).
fn log_sweep(frames: usize, fs: f64, f0: f64, f1: f64) -> Vec<f32> {
    let dur = frames as f64 / fs;
    let r = (f1 / f0).ln();
    stereo(
        frames,
        |t| (2.0 * PI * f0 * dur / r * ((t * r / dur).exp() - 1.0)).sin(),
        fs,
    )
}

/// One channel of noise of `len` (a power of two) frames, power spectrum
/// ∝ `f^(−slope)` between `lo` and `hi`, unit RMS, periodic in `len`
/// (frequency-domain synthesis with random phases).
fn band_noise(len: usize, fs: f64, lo: f64, hi: f64, slope: f64, rng: &mut Rng) -> Vec<f64> {
    let mut planner = RealFftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(len);
    let mut spec = ifft.make_input_vec();
    let hi = hi.min(0.45 * fs);
    for (k, bin) in spec.iter_mut().enumerate() {
        let f = k as f64 * fs / len as f64;
        *bin = if k > 0 && k < len / 2 && (lo..=hi).contains(&f) {
            Complex::from_polar(f.powf(-0.5 * slope), 2.0 * PI * rng.uniform())
        } else {
            Complex::new(0.0, 0.0)
        };
    }
    let mut out = ifft.make_output_vec();
    ifft.process(&mut spec, &mut out).expect("inverse FFT");
    let rms = (out.iter().map(|v| v * v).sum::<f64>() / len as f64).sqrt();
    out.iter_mut().for_each(|v| *v /= rms);
    out
}

/// Loop length of the noise-based signals (~5.5 s at 48 kHz).
const NOISE_LEN: usize = 1 << 18;

/// A speech-like envelope, one value per frame: words of 250 … 600 ms with
/// gaps of 80 … 300 ms (every fourth gap a 600 ms pause), a 4 … 6 Hz
/// syllable modulation inside words, raised-cosine word edges (10 ms).
/// Between words the envelope is 0.
fn speech_envelope(frames: usize, fs: f64, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut env = vec![0.0f64; frames];
    let mut at = (0.1 * fs) as usize;
    let mut word = 0usize;
    while at < frames {
        let len = ((0.25 + 0.35 * rng.uniform()) * fs) as usize;
        let syl = 4.0 + 2.0 * rng.uniform();
        let edge = (0.01 * fs) as usize;
        for k in 0..len.min(frames - at) {
            let t = k as f64 / fs;
            let s = 0.55 + 0.45 * (2.0 * PI * syl * t - PI / 2.0).sin();
            let w = if k < edge {
                0.5 - 0.5 * (PI * k as f64 / edge as f64).cos()
            } else if len - k < edge {
                0.5 - 0.5 * (PI * (len - k) as f64 / edge as f64).cos()
            } else {
                1.0
            };
            env[at + k] = s * w;
        }
        word += 1;
        let gap = if word.is_multiple_of(4) {
            0.6
        } else {
            0.08 + 0.22 * rng.uniform()
        };
        at += len + (gap * fs) as usize;
    }
    env
}

/// Speech-like noise: pink noise from 100 Hz to `hi` under
/// [`speech_envelope`], plus pink room tone 45 dB below the word level.
fn speech_noise(frames: usize, fs: f64, hi: f64, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let env = speech_envelope(frames, fs, seed ^ 0x5eed);
    let room = 10f64.powf(-45.0 / 20.0);
    let chans: Vec<(Vec<f64>, Vec<f64>)> = (0..CH)
        .map(|_| {
            (
                band_noise(NOISE_LEN, fs, 100.0, hi, 1.0, &mut rng),
                band_noise(NOISE_LEN, fs, 20.0, 20_000.0, 1.0, &mut rng),
            )
        })
        .collect();
    let mut out = Vec::with_capacity(frames * CH);
    for (n, e) in env.iter().enumerate() {
        for (speech, tone) in &chans {
            let i = n % NOISE_LEN;
            out.push((speech[i] * e + tone[i] * room) as f32);
        }
    }
    out
}

/// Single-sample clicks (alternating sign, 20 dB above the RMS of the
/// words) every `spacing` seconds over full-band [`speech_noise`].
fn speech_clicks(frames: usize, fs: f64, spacing: f64) -> Signal {
    let mut data = speech_noise(frames, fs, 7_000.0, 43);
    let step = ((spacing * fs) as usize).max(1);
    let marks: Vec<usize> = ((0.2 * fs) as usize..frames).step_by(step).collect();
    for (i, &m) in marks.iter().enumerate() {
        let a = if i % 2 == 0 { 10.0 } else { -10.0 };
        for c in 0..CH {
            data[m * CH + c] += a;
        }
    }
    Signal { data, marks }
}

/// Band-limited (≤ 20 kHz) 100 Hz square wave, each harmonic passed through
/// a first-order high-pass at 100 Hz (magnitude and phase), peak 1.
fn tilted_square(frames: usize, fs: f64) -> Vec<f32> {
    let f0 = 100.0;
    let fc = 100.0;
    let harmonics: Vec<(f64, f64, f64)> = (1..)
        .step_by(2)
        .map(|k| k as f64 * f0)
        .take_while(|&f| f < 20_000.0f64.min(0.45 * fs))
        .map(|f| {
            // H(jf) = (jf/fc) / (1 + jf/fc).
            let x = f / fc;
            let mag = x / (1.0 + x * x).sqrt();
            let phase = (1.0 / x).atan();
            (f, 4.0 / PI * f0 / f * mag, phase)
        })
        .collect();
    normalise_peak(stereo(
        frames,
        |t| {
            harmonics
                .iter()
                .map(|&(f, a, p)| a * (2.0 * PI * f * t + p).sin())
                .sum()
        },
        fs,
    ))
}

/// 250 Hz sine hard-clipped at half its amplitude, synthesised from its
/// Fourier series up to 20 kHz (odd sine harmonics; coefficients by
/// numerical integration over one period), peak 1.
fn clipped_sine(frames: usize, fs: f64) -> Vec<f32> {
    let f0 = 250.0;
    let grid = 8192;
    let clipped = |th: f64| th.sin().clamp(-0.5, 0.5);
    let harmonics: Vec<(f64, f64)> = (1..)
        .step_by(2)
        .map(|k| k as f64)
        .take_while(|&k| k * f0 < 20_000.0f64.min(0.45 * fs))
        .map(|k| {
            let b = (0..grid)
                .map(|i| {
                    let th = 2.0 * PI * (i as f64 + 0.5) / grid as f64;
                    clipped(th) * (k * th).sin()
                })
                .sum::<f64>()
                * 2.0
                / grid as f64;
            (k * f0, b)
        })
        .collect();
    normalise_peak(stereo(
        frames,
        |t| {
            harmonics
                .iter()
                .map(|&(f, b)| b * (2.0 * PI * f * t).sin())
                .sum()
        },
        fs,
    ))
}

fn normalise_peak(mut x: Vec<f32>) -> Vec<f32> {
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if peak > 0.0 {
        x.iter_mut().for_each(|v| *v /= peak);
    }
    x
}

/// Sample peak of `x` (absolute, over all channels).
pub fn sample_peak(x: &[f32]) -> f64 {
    x.iter().fold(0.0f64, |m, &v| m.max(v.abs() as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signals_have_the_requested_length_and_are_finite() {
        let fs = 48_000.0;
        for kind in [
            SignalKind::Sine(1000.0),
            SignalKind::LogSweep,
            SignalKind::SmptePair,
            SignalKind::CcifPair,
            SignalKind::BassIntoHighs,
            SignalKind::SpeechClicks(0.3),
            SignalKind::SpeechNoise(3_400.0),
            SignalKind::TiltedSquare,
            SignalKind::ClippedSine,
            SignalKind::Music,
            SignalKind::Peaky(0.3),
        ] {
            let s = kind.generate(30_000, fs);
            assert_eq!(s.frames(), 30_000, "{kind:?}");
            assert!(s.data.iter().all(|v| v.is_finite()), "{kind:?}");
            assert!(sample_peak(&s.data) > 0.0, "{kind:?}");
            assert!(s.marks.iter().all(|&m| m < 30_000), "{kind:?}");
        }
        // Peaky marks repeat with the loop.
        let s = SignalKind::Peaky(1.1).generate(3 * programme_support::PROGRAMME_FRAMES, fs);
        assert_eq!(s.marks.len() % 3, 0);
        for &m in &s.marks {
            assert!(s.data[m * CH].abs() > 3.0, "click at {m}");
        }
    }

    #[test]
    fn clipped_sine_is_flat_topped() {
        let s = clipped_sine(4800, 48_000.0);
        // Half the samples of a sine clipped at half its amplitude sit on
        // the flat top (|sin| ≥ 0.5 for 2/3 of the period); allow the
        // band-limiting ripple.
        let near_top = s.iter().filter(|v| v.abs() > 0.9).count();
        assert!(near_top as f64 > 0.5 * s.len() as f64, "{near_top}");
    }
}
