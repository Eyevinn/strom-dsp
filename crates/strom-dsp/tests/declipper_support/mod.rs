//! Synthetic test material and metrics for the declipper tests, bench and
//! render example. Everything is generated from fixed seeds; no recordings.
//!
//! Metrics follow the evaluation protocol of Záviška et al., "A survey and an
//! extensive evaluation of popular audio declipping methods", IEEE JSTSP 2021:
//! SDR = 10·log10(Σ x² / Σ (x − y)²), ΔSDR = SDR(declipped) − SDR(clipped),
//! and the same restricted to the clipped samples (SDRc, ΔSDRc). Clip depth
//! here is the clip level below the signal peak, in dB.

#![allow(dead_code)]

use std::f64::consts::PI;

use strom_dsp::declipper::{ClipDetector, Declipper, DeclipperParams, DetectorParams};
use strom_dsp::Processor;

pub const FS: f64 = 48_000.0;

/// xorshift32 with a uniform and a Gaussian draw.
pub struct Rng(pub u32);

impl Rng {
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u32() >> 8) as f64 / (1u32 << 24) as f64
    }
    /// Standard normal (Box–Muller).
    pub fn gauss(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

pub fn normalize(x: &mut [f64], peak: f64) {
    let m = x.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    if m > 0.0 {
        for v in x.iter_mut() {
            *v *= peak / m;
        }
    }
}

/// Sum of six sines, log-uniform 80 Hz … 5 kHz, random phases, amplitudes 1/√k.
pub fn multitone(secs: f64, seed: u32) -> Vec<f64> {
    let mut rng = Rng(seed);
    let tones: Vec<(f64, f64, f64)> = (1..=6)
        .map(|k| {
            let f = 80.0 * (5000.0f64 / 80.0).powf(rng.uniform());
            (f, 2.0 * PI * rng.uniform(), 1.0 / (k as f64).sqrt())
        })
        .collect();
    let n = (secs * FS) as usize;
    let mut x: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / FS;
            tones
                .iter()
                .map(|&(f, ph, a)| a * (2.0 * PI * f * t + ph).sin())
                .sum()
        })
        .collect();
    normalize(&mut x, 1.0);
    x
}

/// Two-pole resonator `y = g·x + 2r·cos θ·y1 − r²·y2`, unity gain at DC.
#[derive(Default, Clone, Copy)]
struct Reson {
    y1: f64,
    y2: f64,
}

impl Reson {
    fn step(&mut self, x: f64, f: f64, bw: f64) -> f64 {
        let r = (-PI * bw / FS).exp();
        let c = 2.0 * r * (2.0 * PI * f / FS).cos();
        let g = 1.0 - c + r * r;
        let y = g * x + c * self.y1 - r * r * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Multitone (as [`multitone`]) with a noise-driven amplitude envelope: a new
/// random target level (0.15 … 1) every 100 ms, approached by a one-pole
/// glide, like programme whose level moves.
pub fn noise_modulated_tones(secs: f64, seed: u32) -> Vec<f64> {
    let mut x = multitone(secs, seed);
    let mut rng = Rng(seed ^ 0x9e37_79b9);
    let mut env = 0.5;
    let mut target = 0.5;
    for (i, v) in x.iter_mut().enumerate() {
        if i % 4800 == 0 {
            target = 0.15 + 0.85 * rng.uniform();
        }
        env += (target - env) * 2e-4;
        *v *= env;
    }
    normalize(&mut x, 1.0);
    x
}

/// Band-limited Gaussian noise (two resonators at 400 Hz and 2 kHz plus the
/// direct path) with a slow random amplitude envelope.
pub fn noise_modulated(secs: f64, seed: u32) -> Vec<f64> {
    let mut rng = Rng(seed);
    let n = (secs * FS) as usize;
    let (mut r1, mut r2) = (Reson::default(), Reson::default());
    let mut lp = 0.0;
    let mut env = 0.5;
    let mut env_target = 0.5;
    let mut x: Vec<f64> = (0..n)
        .map(|i| {
            if i % 4800 == 0 {
                env_target = 0.15 + 0.85 * rng.uniform();
            }
            env += (env_target - env) * 2e-4;
            let w = rng.gauss();
            lp += 0.3 * (w - lp);
            let v = 0.6 * lp + r1.step(w, 400.0, 300.0) * 0.08 + r2.step(w, 2000.0, 800.0) * 0.05;
            env * v
        })
        .collect();
    normalize(&mut x, 1.0);
    x
}

/// Formant (F1, F2, F3) of five vowels, Peterson & Barney (1952) averages.
const VOWELS: [(f64, f64, f64); 5] = [
    (730.0, 1090.0, 2440.0),
    (270.0, 2290.0, 3010.0),
    (300.0, 870.0, 2240.0),
    (530.0, 1840.0, 2480.0),
    (570.0, 840.0, 2410.0),
];

/// Speech-like signal by source-filter synthesis (Rabiner & Schafer): a
/// glottal pulse train with intonation and jitter, three formant resonators
/// gliding between vowels, syllable envelopes, short fricative noise bursts
/// and pauses. `f0` is the mean pitch (Hz).
pub fn speech_like(secs: f64, seed: u32, f0: f64) -> Vec<f64> {
    let mut rng = Rng(seed);
    let n = (secs * FS) as usize;
    let mut out = vec![0.0; n];
    let mut rs = [Reson::default(); 3];
    let mut glottal_lp = [0.0f64; 2];
    let mut phase = 0.0;
    let mut i = 0usize;
    let mut fr = Reson::default();
    let mut cur = VOWELS[0];
    let mut hp_prev = 0.0;
    while i < n {
        // A syllable: 120–260 ms voiced, then a fricative or a pause.
        let syl = ((0.12 + 0.14 * rng.uniform()) * FS) as usize;
        let target = VOWELS[(rng.next_u32() % 5) as usize];
        let start = cur;
        let loud = 0.5 + 0.5 * rng.uniform();
        for k in 0..syl.min(n - i) {
            let u = k as f64 / syl as f64;
            let glide = (u * 4.0).min(1.0);
            let f = (
                start.0 + (target.0 - start.0) * glide,
                start.1 + (target.1 - start.1) * glide,
                start.2 + (target.2 - start.2) * glide,
            );
            let t = (i + k) as f64 / FS;
            let pitch = f0 * (1.0 + 0.12 * (2.0 * PI * 0.7 * t).sin() + 0.06 * (1.0 - 2.0 * u));
            phase += pitch * (1.0 + 0.004 * rng.gauss()) / FS;
            let mut src = 0.0;
            if phase >= 1.0 {
                phase -= 1.0;
                src = 1.0;
            }
            // Glottal spectrum (two one-pole low-passes), lip radiation (difference).
            glottal_lp[0] += 0.12 * (src - glottal_lp[0]);
            glottal_lp[1] += 0.12 * (glottal_lp[0] - glottal_lp[1]);
            let g = glottal_lp[1];
            let rad = g - hp_prev;
            hp_prev = g;
            let a = rs[0].step(rad, f.0, 80.0);
            let b = rs[1].step(a, f.1, 100.0);
            let c = rs[2].step(b, f.2, 140.0);
            let env = (PI * u).sin().powf(0.6) * loud;
            out[i + k] = env * (c + 0.002 * rng.gauss());
        }
        cur = target;
        i += syl;
        if i >= n {
            break;
        }
        let gap = ((0.03 + 0.09 * rng.uniform()) * FS) as usize;
        let fric = rng.uniform() < 0.4;
        for k in 0..gap.min(n - i) {
            let u = k as f64 / gap as f64;
            out[i + k] = if fric {
                0.03 * (PI * u).sin() * fr.step(rng.gauss(), 4500.0, 2500.0)
            } else {
                0.0
            };
        }
        i += gap;
    }
    // Remove DC and normalise.
    let mean = out.iter().sum::<f64>() / n as f64;
    for v in &mut out {
        *v -= mean;
    }
    normalize(&mut out, 1.0);
    out
}

/// Music-like signal: a I–vi–IV–V progression in C (two bars per chord at
/// 100 bpm feel) of plucked additive tones with decaying harmonics, plus a
/// bass note per chord.
pub fn music_like(secs: f64, seed: u32) -> Vec<f64> {
    let mut rng = Rng(seed);
    let n = (secs * FS) as usize;
    let mut out = vec![0.0; n];
    let midi = |m: f64| 440.0 * 2f64.powf((m - 69.0) / 12.0);
    let chords: [[f64; 3]; 4] = [
        [60.0, 64.0, 67.0],
        [57.0, 60.0, 64.0],
        [53.0, 57.0, 60.0],
        [55.0, 59.0, 62.0],
    ];
    let beat = (0.3 * FS) as usize;
    let note =
        |out: &mut [f64], start: usize, f: f64, amp: f64, decay: f64, harms: usize, ph: f64| {
            let len = ((decay * 5.0) * FS) as usize;
            for k in 0..len.min(n.saturating_sub(start)) {
                let t = k as f64 / FS;
                let att = (t / 0.004).min(1.0);
                let mut v = 0.0;
                for h in 1..=harms {
                    let fh = f * h as f64;
                    if fh > 12_000.0 {
                        break;
                    }
                    v += (2.0 * PI * fh * t + ph * h as f64).sin() / (h as f64).powf(1.3)
                        * (-t * h as f64 / (decay * 1.5)).exp();
                }
                out[start + k] += amp * att * v * (-t / decay).exp();
            }
        };
    let mut b = 0usize;
    let mut pos = 0usize;
    while pos < n {
        let ch = chords[(b / 4) % 4];
        if b.is_multiple_of(4) {
            note(
                &mut out,
                pos,
                midi(ch[0] - 24.0),
                0.5,
                0.9,
                8,
                rng.uniform(),
            );
        }
        for (j, &m) in ch.iter().enumerate() {
            let at = pos + j * (0.012 * FS) as usize;
            if at < n {
                note(&mut out, at, midi(m), 0.25, 0.5, 10, rng.uniform() * 6.0);
            }
        }
        // A melody note on every beat.
        let mel = ch[(rng.next_u32() % 3) as usize] + 12.0;
        note(&mut out, pos, midi(mel), 0.3, 0.35, 6, rng.uniform() * 6.0);
        pos += beat;
        b += 1;
    }
    normalize(&mut out, 1.0);
    out
}

/// Clip `x` (peak 1) at `depth_db` below its peak. Returns the clipped signal
/// and the clip level.
pub fn clip(x: &[f64], depth_db: f64) -> (Vec<f64>, f64) {
    let peak = x.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let c = peak * 10f64.powf(-depth_db / 20.0);
    (x.iter().map(|v| v.clamp(-c, c)).collect(), c)
}

pub fn to_f32(x: &[f64]) -> Vec<f32> {
    x.iter().map(|&v| v as f32).collect()
}

/// Run a mono signal through a declipper and return the output aligned with
/// the input (latency removed), in 480-frame buffers.
pub fn declip(x: &[f32], params: &DeclipperParams) -> (Vec<f32>, Declipper) {
    let mut d = Declipper::new();
    d.set_params(params);
    d.configure(FS as f32, 1);
    let lat = d.latency_frames();
    let mut buf: Vec<f32> = x.to_vec();
    buf.extend(std::iter::repeat_n(0.0, lat));
    for b in buf.chunks_mut(480) {
        d.process(b);
    }
    (buf[lat..].to_vec(), d)
}

/// Clipped-sample mask (`|x| > c` in the original).
pub fn truth_mask(orig: &[f64], c: f64) -> Vec<bool> {
    orig.iter().map(|v| v.abs() > c).collect()
}

/// Detection mask from the streaming detector (every reported run, including
/// the too-long ones).
pub fn detect_mask(x: &[f32], p: DetectorParams) -> Vec<bool> {
    let mut det = ClipDetector::new(p);
    let mut mask = vec![false; x.len()];
    for &v in x.iter().chain(std::iter::once(&0.0)) {
        if let Some(r) = det.push(v) {
            for i in r.start..=r.end() {
                mask[i as usize] = true;
            }
        }
    }
    mask
}

/// Detector settings that match a default declipper at 48 kHz.
pub fn default_detector() -> DetectorParams {
    let p = DeclipperParams::default();
    DetectorParams {
        min_level: 10f32.powf(p.min_level_db / 20.0),
        tolerance: p.tolerance,
        max_run: (p.max_run_ms * 48.0).round() as usize,
        ceiling_hold: (strom_dsp::declipper::detector::CEILING_HOLD_S as f64 * FS) as u64,
        max_tilt: 2.0 * std::f32::consts::PI * p.tilt_hz / FS as f32,
        singles: true,
    }
}

/// (precision, recall) of `det` against `truth`.
pub fn precision_recall(det: &[bool], truth: &[bool]) -> (f64, f64) {
    let (mut tp, mut fp, mut fneg) = (0usize, 0usize, 0usize);
    for (&d, &t) in det.iter().zip(truth) {
        match (d, t) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fneg += 1,
            _ => {}
        }
    }
    let p = if tp + fp == 0 {
        1.0
    } else {
        tp as f64 / (tp + fp) as f64
    };
    let r = if tp + fneg == 0 {
        1.0
    } else {
        tp as f64 / (tp + fneg) as f64
    };
    (p, r)
}

/// SDR (dB) of `y` against `x`, optionally over the samples where `mask` is true.
pub fn sdr(x: &[f64], y: &[f32], mask: Option<&[bool]>) -> f64 {
    let (mut s, mut e) = (0.0, 0.0);
    for i in 0..x.len() {
        if mask.is_none_or(|m| m[i]) {
            s += x[i] * x[i];
            let d = x[i] - f64::from(y[i]);
            e += d * d;
        }
    }
    10.0 * (s / e.max(1e-300)).log10()
}

/// Truth mask restricted to samples in clipped runs of length ≥ 2 (what a
/// plateau detector can see at all).
pub fn truth_runs_ge2(truth: &[bool]) -> Vec<bool> {
    let mut out = vec![false; truth.len()];
    let mut i = 0;
    while i < truth.len() {
        if truth[i] {
            let mut j = i;
            while j < truth.len() && truth[j] {
                j += 1;
            }
            if j - i >= 2 {
                out[i..j].fill(true);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    out
}
