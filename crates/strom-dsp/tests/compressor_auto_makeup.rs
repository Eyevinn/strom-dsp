//! Auto-makeup: the integrated loudness (ITU-R BS.1770 via
//! `strom_dsp::meter::LoudnessMeter`) of a fixed programme-like signal stays
//! within ±1.5 LU of the unprocessed signal across ratios 2 … 10 and
//! thresholds −30 … −10 dB; explicit makeup applies on top; switching and
//! parameter changes glide.

use strom_dsp::compressor::{auto_makeup_db, CompressorDetector, AUTO_MAKEUP_REFERENCE_DB};
use strom_dsp::dynamics::GainComputer;
use strom_dsp::meter::{LoudnessConfig, LoudnessMeter};
use strom_dsp::util::{db_to_lin, lin_to_db};
use strom_dsp::{Compressor, CompressorParams, Processor};

const FS: f32 = 48_000.0;
const CH: usize = 2;

struct Rng(u64);

impl Rng {
    /// Uniform in [−1, 1).
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
    fn unit(&mut self) -> f64 {
        0.5 * (self.next() + 1.0)
    }
}

/// Programme-like test signal, `seconds` long, stereo, fixed seed:
/// syllables of 120–300 ms (raised-cosine envelope, random level over a
/// 12 dB range) with short pauses and an occasional longer one, each
/// syllable a voiced harmonic complex (f0 100–220 Hz, 12 harmonics at 1/k)
/// plus low-passed noise (unvoiced part), and a quiet noise floor. Scaled so
/// its loudest syllables peak around −6 dBFS.
fn programme(seconds: f64) -> Vec<f32> {
    let n = (seconds * FS as f64) as usize;
    let mut rng = Rng(0x00c0_ffee_1234_5678);
    let mut out = vec![0.0f64; n * CH];
    let mut pos = 0usize;
    let mut lp = [0.0f64; CH];
    while pos < n {
        let len = ((0.12 + 0.18 * rng.unit()) * FS as f64) as usize;
        let gain = db_to_lin(-12.0 * rng.unit());
        let f0 = 100.0 + 120.0 * rng.unit();
        let voiced = 0.6 + 0.4 * rng.unit();
        for i in 0..len.min(n - pos) {
            let t = i as f64 / FS as f64;
            let env = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / len as f64).cos();
            let tone: f64 = (1..=12)
                .map(|k| (2.0 * std::f64::consts::PI * f0 * k as f64 * t).sin() / k as f64)
                .sum();
            for (c, l) in lp.iter_mut().enumerate() {
                *l = 0.7 * *l + 0.3 * rng.next();
                out[(pos + i) * CH + c] =
                    gain * env * (voiced * 0.35 * tone + (1.0 - voiced) * 0.8 * *l);
            }
        }
        pos += len;
        // Pause: mostly short, now and then a longer one.
        let pause = if rng.unit() < 0.15 { 0.4 } else { 0.06 } * (0.5 + rng.unit());
        pos += (pause * FS as f64) as usize;
    }
    for v in out.iter_mut() {
        *v += 1e-4 * rng.next();
    }
    let peak = out.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let k = db_to_lin(-6.0) / peak;
    out.iter().map(|v| (v * k) as f32).collect()
}

fn loudness(x: &[f32]) -> f64 {
    let mut m = LoudnessMeter::new(&LoudnessConfig {
        sample_rate: FS,
        channels: CH,
        true_peak: false,
        ..Default::default()
    })
    .unwrap();
    m.process(x);
    m.integrated()
}

fn compressed(x: &[f32], p: &CompressorParams, det: CompressorDetector, auto: bool) -> Vec<f32> {
    let mut c = Compressor::new();
    c.set_detector(det);
    c.set_auto_makeup(auto);
    c.set_params(p);
    c.configure(FS, CH);
    let mut y = x.to_vec();
    for chunk in y.chunks_mut(1024 * CH) {
        c.process(chunk);
    }
    y
}

#[test]
fn auto_makeup_formula() {
    // Hard knee, T = −30 dB, R = 4: reduction at −18 dB is 0.75 · 12 = 9 dB.
    let gc = GainComputer::from_linear(db_to_lin(-30.0), 4.0, 1.0);
    assert!((auto_makeup_db(&gc) - 9.0).abs() < 1e-9);
    // Reference below the knee: no makeup.
    let gc = GainComputer::from_linear(db_to_lin(-10.0), 10.0, 1.0);
    assert_eq!(auto_makeup_db(&gc), 0.0);
    // Ratio 1: none.
    let gc = GainComputer::from_linear(db_to_lin(-30.0), 1.0, 0.0625);
    assert_eq!(auto_makeup_db(&gc), 0.0);
    // Soft knee: the quadratic.
    let gc = GainComputer::from_linear(db_to_lin(-20.0), 2.0, db_to_lin(-6.0));
    let (t, w, s) = (-20.0, 12.0, 0.5);
    let d: f64 = AUTO_MAKEUP_REFERENCE_DB - t + w / 2.0;
    assert!((auto_makeup_db(&gc) - s * d * d / (2.0 * w)).abs() < 1e-6);
}

/// A steady 1 kHz sine at the reference level comes out at its input level
/// (makeup 1), and an explicit makeup multiplies on top.
#[test]
fn reference_level_is_unchanged() {
    for makeup in [1.0f32, 2.0] {
        let p = CompressorParams {
            threshold: db_to_lin(-30.0) as f32,
            ratio: 6.0,
            knee: 0.5,
            attack_ms: 1.0,
            release_ms: 100.0,
            makeup_gain: makeup,
        };
        let amp = db_to_lin(AUTO_MAKEUP_REFERENCE_DB);
        let x: Vec<f32> = (0..CH * 48_000)
            .map(|i| (amp * (2.0 * std::f64::consts::PI * (i / CH) as f64 / 48.0).sin()) as f32)
            .collect();
        let y = compressed(&x, &p, CompressorDetector::Peak, true);
        let peak = y[y.len() - 96 * CH..]
            .iter()
            .fold(0.0f64, |m, &v| m.max(v.abs() as f64));
        let got = lin_to_db(peak) - AUTO_MAKEUP_REFERENCE_DB;
        assert!((got - lin_to_db(makeup as f64)).abs() < 0.01, "{got}");
    }
}

/// Loudness stays constant while the operator changes compression.
///
/// Reference: for each detector, the mean integrated loudness over the grid
/// (thresholds −30, −25, −20, −15, −10 dB × ratios 2, 4, 6, 10; default knee,
/// attack and release). Tolerance: every setting within ±1.5 LU of it.
/// The mean itself must be within ±2.5 LU of the unprocessed programme: the
/// reference level is fixed, so the remaining offset depends on how the
/// programme reads in the detector (speech peaks read higher in peak mode
/// than in RMS mode, so peak mode makes up about 2 dB less on this signal).
#[test]
fn loudness_constant_across_ratio_and_threshold() {
    let x = programme(30.0);
    let input = loudness(&x);
    eprintln!("programme: {input:.2} LUFS");
    for det in [CompressorDetector::Peak, CompressorDetector::Rms] {
        let mut auto = Vec::new();
        let mut without = Vec::new();
        for t_db in [-30.0, -25.0, -20.0, -15.0, -10.0] {
            for ratio in [2.0f32, 4.0, 6.0, 10.0] {
                let p = CompressorParams {
                    threshold: db_to_lin(t_db) as f32,
                    ratio,
                    ..Default::default()
                };
                let l = loudness(&compressed(&x, &p, det, true)) - input;
                let l0 = loudness(&compressed(&x, &p, det, false)) - input;
                eprintln!("{det:?} T {t_db} R {ratio}: auto {l:+.2} LU, without {l0:+.2} LU");
                auto.push(l);
                without.push(l0);
            }
        }
        let mean = auto.iter().sum::<f64>() / auto.len() as f64;
        let worst = auto.iter().map(|l| (l - mean).abs()).fold(0.0, f64::max);
        let mean0 = without.iter().sum::<f64>() / without.len() as f64;
        let worst0 = without
            .iter()
            .map(|l| (l - mean0).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "{det:?}: mean {mean:+.2} LU vs input, worst ±{worst:.2} LU around it \
             (without auto-makeup: ±{worst0:.2} LU)"
        );
        assert!(worst <= 1.5, "{det:?}: worst {worst:.2} LU");
        assert!(mean.abs() <= 2.5, "{det:?}: mean {mean:.2} LU");
        // The test is meaningful: without auto-makeup the loudness moves a lot.
        assert!(worst0 > 4.0, "{det:?}: {worst0}");
    }
}

/// Switching auto-makeup and changing threshold/ratio with it on glides:
/// per-frame gain change bounded as in the phase 1 parameter-change test.
#[test]
fn auto_makeup_changes_glide() {
    let level = 0.5f32;
    let mut c = Compressor::new();
    c.configure(FS, 1);
    let sets = [
        CompressorParams {
            threshold: 0.03,
            ratio: 10.0,
            knee: 1.0,
            attack_ms: 0.01,
            release_ms: 0.01,
            makeup_gain: 1.0,
        },
        CompressorParams {
            threshold: 0.3,
            ratio: 2.0,
            knee: 0.0625,
            attack_ms: 0.01,
            release_ms: 0.01,
            makeup_gain: 1.0,
        },
    ];
    c.set_params(&sets[0]);
    c.process(&mut vec![level; 4800]);
    let mut prev = None;
    let mut max_step: f64 = 0.0;
    for k in 0..12 {
        if k % 3 == 0 {
            c.set_auto_makeup(!c.auto_makeup());
        } else {
            c.set_params(&sets[k % 2]);
        }
        let mut buf = vec![level; 2400];
        c.process(&mut buf);
        for o in buf {
            let g = lin_to_db(o as f64 / level as f64);
            if let Some(p) = prev {
                max_step = max_step.max((g - p as f64).abs());
            }
            prev = Some(g);
        }
    }
    // Largest change: about 30 dB (auto-makeup of 10:1 at −30 dB), spread
    // over a 10 ms one-pole: ≈ 30/480 dB per frame at most, times 3 as in
    // the phase 1 bound.
    assert!(max_step < 0.2, "max step {max_step} dB");
}
