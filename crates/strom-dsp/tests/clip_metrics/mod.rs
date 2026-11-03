//! Test-only measurements of clipping quality, written so they can move into
//! a shared measurement harness unchanged: every function takes plain
//! sample slices and returns numbers.
//!
//! - [`spectrum_db`]: amplitude spectrum (dBFS, sine-calibrated) with a
//!   4-term Blackman–Harris window (Harris, Proc. IEEE 66(1), 1978).
//! - [`spur_dbfs`]: the largest component away from given frequencies
//!   (aliasing of a clipped tone).
//! - [`level_at_dbfs`]: the level at one frequency (sidebands).
//! - [`nmr_db`]: noise-to-mask ratio of a processing error against a
//!   gain- and delay-matched reference (Brandenburg, AES 1987), with a
//!   simple Bark-band masking model (Zwicker & Fastl critical bands,
//!   Schroeder spreading function, a fixed masking offset). **A test metric
//!   only**: nothing in the processing path uses it.

#![allow(dead_code)]

use realfft::RealFftPlanner;

/// 4-term Blackman–Harris window of length `n` (−92 dB side lobes).
fn blackman_harris(n: usize) -> Vec<f64> {
    let (a0, a1, a2, a3) = (0.35875, 0.48829, 0.14128, 0.01168);
    (0..n)
        .map(|i| {
            let t = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            a0 - a1 * t.cos() + a2 * (2.0 * t).cos() - a3 * (3.0 * t).cos()
        })
        .collect()
}

/// Amplitude spectrum of `x` in dBFS (a full-scale sine at a bin centre
/// reads 0 dB), bins `k·fs/x.len()` for `k = 0 … len/2`.
pub fn spectrum_db(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let w = blackman_harris(n);
    let gain: f64 = w.iter().sum::<f64>() / 2.0;
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut input: Vec<f64> = x.iter().zip(&w).map(|(a, b)| a * b).collect();
    let mut out = fft.make_output_vec();
    fft.process(&mut input, &mut out).expect("FFT");
    out.iter()
        .map(|c| 20.0 * (c.norm() / gain).max(1e-15).log10())
        .collect()
}

/// The largest spectral component of `x` (dBFS) outside ±`guard_bins`
/// around each frequency in `exclude` (Hz) and DC. Use coherent
/// frequencies (whole bins) for exact results.
pub fn spur_dbfs(x: &[f64], fs: f64, exclude: &[f64], guard_bins: usize) -> f64 {
    let s = spectrum_db(x);
    let df = fs / x.len() as f64;
    let mut worst = -400.0f64;
    for (k, &v) in s.iter().enumerate() {
        let f = k as f64 * df;
        let near = |e: f64| ((f - e) / df).abs() <= guard_bins as f64;
        if k <= guard_bins || exclude.iter().any(|&e| near(e)) {
            continue;
        }
        worst = worst.max(v);
    }
    worst
}

/// The level (dBFS) at `freq` (Hz): the largest bin within ±1 bin.
pub fn level_at_dbfs(x: &[f64], fs: f64, freq: f64) -> f64 {
    let s = spectrum_db(x);
    let df = fs / x.len() as f64;
    let k = (freq / df).round() as isize;
    (k - 1..=k + 1)
        .filter(|&i| i >= 0 && (i as usize) < s.len())
        .map(|i| s[i as usize])
        .fold(-400.0, f64::max)
}

/// One channel of an interleaved buffer as `f64`.
pub fn channel(x: &[f32], channels: usize, c: usize) -> Vec<f64> {
    x.iter()
        .skip(c)
        .step_by(channels)
        .map(|&v| v as f64)
        .collect()
}

/// Largest absolute sample value (dBFS).
pub fn sample_peak_db(x: &[f32]) -> f64 {
    let p = x.iter().fold(0.0f32, |m, v| m.max(v.abs())) as f64;
    20.0 * p.max(1e-15).log10()
}

/// Critical-band rate (Bark) of `f` Hz (Zwicker & Terhardt 1980).
fn bark(f: f64) -> f64 {
    13.0 * (0.00076 * f).atan() + 3.5 * (f / 7500.0).powi(2).atan()
}

/// Schroeder spreading function (dB) at a distance `dz` Bark from the masker
/// (positive: toward higher frequencies).
fn spreading_db(dz: f64) -> f64 {
    15.81 + 7.5 * (dz + 0.474) - 17.5 * (1.0 + (dz + 0.474).powi(2)).sqrt()
}

/// Masking offset (dB) of the threshold below the spread masker energy: a
/// fixed value between the tonal and the noise-like masker of Johnston's
/// model (IEEE JSAC 6(2), 1988).
pub const MASK_OFFSET_DB: f64 = 10.0;
/// Frame length and hop of the NMR analysis (Hann window, 50 % overlap).
const NMR_FRAME: usize = 2048;

/// Noise-to-mask ratio (dB) of `out` against `reference` (mono, the same
/// length, `out` lagging by `delay` samples). The reference is scaled by
/// the least-squares gain first, so a pure level change has no error.
/// Per frame: error and reference energies per Bark band (1 Bark wide), the
/// reference spread across bands, the mask `MASK_OFFSET_DB` below it; the
/// frame's NMR is the mean of error/mask over the bands; the result is the
/// mean over frames, in dB (Brandenburg 1987's total NMR). Frames whose
/// reference is silent are left out.
pub fn nmr_db(reference: &[f64], out: &[f64], delay: usize, fs: f64) -> f64 {
    let n = reference.len().min(out.len().saturating_sub(delay));
    let r = &reference[..n];
    let y = &out[delay..delay + n];
    let ry: f64 = r.iter().zip(y).map(|(a, b)| a * b).sum();
    let rr: f64 = r.iter().map(|a| a * a).sum();
    let g = if rr > 0.0 { ry / rr } else { 1.0 };
    let err: Vec<f64> = r.iter().zip(y).map(|(a, b)| b - g * a).collect();
    let refg: Vec<f64> = r.iter().map(|a| g * a).collect();

    let hann: Vec<f64> = (0..NMR_FRAME)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / NMR_FRAME as f64).cos())
        .collect();
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(NMR_FRAME);
    let bins = NMR_FRAME / 2 + 1;
    let nbands = bark(fs / 2.0).floor() as usize + 1;
    let band_of: Vec<usize> = (0..bins)
        .map(|k| (bark(k as f64 * fs / NMR_FRAME as f64).floor() as usize).min(nbands - 1))
        .collect();
    let spread: Vec<Vec<f64>> = (0..nbands)
        .map(|to| {
            (0..nbands)
                .map(|from| 10f64.powf(spreading_db(to as f64 - from as f64) / 10.0))
                .collect()
        })
        .collect();
    let band_energy = |x: &[f64]| -> Vec<f64> {
        let mut input: Vec<f64> = x.iter().zip(&hann).map(|(a, w)| a * w).collect();
        let mut spec = fft.make_output_vec();
        fft.process(&mut input, &mut spec).expect("FFT");
        let mut e = vec![0.0; nbands];
        for (k, c) in spec.iter().enumerate() {
            e[band_of[k]] += c.norm_sqr();
        }
        e
    };
    let offset = 10f64.powf(-MASK_OFFSET_DB / 10.0);
    let mut sum = 0.0;
    let mut frames = 0usize;
    let mut start = 0;
    while start + NMR_FRAME <= n {
        let er = band_energy(&refg[start..start + NMR_FRAME]);
        let ee = band_energy(&err[start..start + NMR_FRAME]);
        let total: f64 = er.iter().sum();
        if total > 1e-6 {
            let floor = total * 1e-9;
            let mut acc = 0.0;
            for b in 0..nbands {
                let masker: f64 = (0..nbands).map(|f| er[f] * spread[b][f]).sum();
                let mask = (masker * offset).max(floor);
                acc += ee[b] / mask;
            }
            sum += acc / nbands as f64;
            frames += 1;
        }
        start += NMR_FRAME / 2;
    }
    if frames == 0 {
        return -200.0;
    }
    10.0 * (sum / frames as f64).max(1e-20).log10()
}
