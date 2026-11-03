//! Synthetic programmes for the broadcast chain tests and the measurement
//! harness (`crates/strom-dsp-harness`, which includes this file). Fixed
//! seeds, no recordings. Needs `crate::agc_support`.

#![allow(dead_code)]

use std::f64::consts::PI;

use crate::agc_support::{pink, Rng};

/// Frames of each programme loop (~10.9 s at 48 kHz).
pub const PROGRAMME_FRAMES: usize = 1 << 19;

/// A music-like programme: pink noise with a 4 Hz level modulation, two
/// tones, and short speech-like gaps (a 10.9 s loop).
pub fn music_programme(ch: usize, fs: f64) -> Vec<f32> {
    let frames = PROGRAMME_FRAMES;
    let mut x = pink(frames, ch, fs, 21);
    for n in 0..frames {
        let t = n as f64 / fs;
        // 4 Hz syllable-like envelope, with a 300 ms gap every 2.7 s.
        let syl = 0.6 + 0.4 * (2.0 * PI * 4.0 * t).sin();
        let gap = if (t % 2.7) < 0.3 { 0.05 } else { 1.0 };
        let tones = 0.4 * (2.0 * PI * 110.0 * t).sin() + 0.2 * (2.0 * PI * 1_320.0 * t).sin();
        for c in 0..ch {
            let v = &mut x[n * ch + c];
            *v = ((*v as f64 * 0.5 + tones) * syl * gap) as f32;
        }
    }
    x
}

/// Frame of the first click of [`peaky_programme`] (0.37 s).
pub fn first_click_frame(fs: f64) -> usize {
    (0.37 * fs) as usize
}

/// A sport-like programme (a ~10.9 s loop, fixed seed): a pink-noise crowd
/// bed; percussive hits every ~250 ms (a decaying white-noise burst plus a
/// decaying 60 Hz thump); a 5 ms fs/4 sine burst at 45° phase (sample
/// peaks 3 dB under its true peak) every ~2.3 s; and a single-sample click
/// every `click_s` seconds, starting at 0.37 s.
pub fn peaky_programme(ch: usize, fs: f64, click_s: f64) -> Vec<f32> {
    let mut x = peaky_bed(ch, fs);
    // Single-sample clicks every `click_s`.
    for n in click_frames(fs, click_s) {
        let sign = if click_index(fs, click_s, n).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        for c in 0..ch {
            x[n * ch + c] += 4.0 * sign;
        }
    }
    x
}

/// [`peaky_programme`] without its clicks: the bed, the hits and the fs/4
/// bursts.
pub fn peaky_bed(ch: usize, fs: f64) -> Vec<f32> {
    let frames = PROGRAMME_FRAMES;
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
            let thump = (2.0 * PI * 60.0 * t).sin() * (-t / 0.06).exp();
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
            let v = (1.2 * (PI / 2.0 * k as f64 + PI / 4.0).sin()) as f32;
            for c in 0..ch {
                x[(start + k) * ch + c] += v;
            }
        }
        start += (2.3 * fs) as usize;
    }
    x
}

/// Frames of the clicks of [`peaky_programme`] within one loop.
pub fn click_frames(fs: f64, click_s: f64) -> Vec<usize> {
    let step = (click_s * fs) as usize;
    (first_click_frame(fs)..PROGRAMME_FRAMES)
        .step_by(step)
        .collect()
}

fn click_index(fs: f64, click_s: f64, n: usize) -> usize {
    (n - first_click_frame(fs)) / (click_s * fs) as usize
}
