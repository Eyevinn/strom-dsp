//! Auto-makeup keeps the programme's loudness.
//!
//! Signal: speech-like programme (syllables, pauses, phrases with level
//! jumps between "speakers", peaks near −3 dBFS) at three input loudness
//! levels made by changing density and level spread, not by scaling.
//! Measured with `strom_dsp::meter::LoudnessMeter` (ITU-R BS.1770, gated
//! integrated loudness).
//!
//! Requirement: with auto-makeup on, the integrated output loudness is
//! within ±[`TOL_LU`] of the input's, for ratios 2 … 10, thresholds
//! −30 … −10 dB, both detector modes and all three programmes. Explicit
//! makeup applies on top; switching and parameter changes glide.
//!
//! Tolerance ±2 LU ("within a couple of LU", the operator requirement),
//! over the whole 40 s programme with no exclusion: the estimate starts
//! from the static prior and converges within the first second, which
//! moves the integrated value by far less than the margin (measured worst
//! case 1.0 LU). The remaining error is mainly that the estimator weights
//! frames by their unweighted peak energy above a −50 dB gate, while
//! BS.1770 uses K-weighting and a −10 LU relative gate. The phase 1b static
//! reference (−18 dB) fails this test at 87 of 150 grid points (worst
//! −8.1 LU, peak mode, dense programme at 10:1).

use strom_dsp::compressor::{
    static_makeup_db, CompressorDetector, AUTO_MAKEUP_GATE_DB, AUTO_MAKEUP_MAX_DB,
    AUTO_MAKEUP_PRIOR_DB,
};
use strom_dsp::dynamics::GainComputer;
use strom_dsp::meter::{LoudnessConfig, LoudnessMeter};
use strom_dsp::util::{db_to_lin, lin_to_db};
use strom_dsp::{Compressor, CompressorParams, Processor};

const FS: f32 = 48_000.0;
const CH: usize = 2;
/// Programme length (s).
const SECONDS: f64 = 40.0;
/// Tolerance of the integrated output loudness against the input (LU).
const TOL_LU: f64 = 2.0;

/// Grids. Optimised builds (`cargo test --release`, part of the required
/// checks) run the full grid; unoptimised builds run its corners, because
/// 40 s programmes through the compressor and the loudness meter take
/// seconds per setting without optimisation (debug: corners of the grid,
/// on the loudest and quietest programme).
const FULL: bool = !cfg!(debug_assertions);
const THRESHOLDS_DB: &[f64] = if FULL {
    &[-30.0, -25.0, -20.0, -15.0, -10.0]
} else {
    &[-30.0, -10.0]
};
const RATIOS: &[f32] = if FULL {
    &[2.0, 4.0, 6.0, 10.0]
} else {
    &[2.0, 10.0]
};
const TIMES: &[(f32, f32)] = if FULL {
    &[(0.1, 20.0), (1.0, 50.0), (50.0, 500.0), (200.0, 2000.0)]
} else {
    &[(0.1, 20.0), (200.0, 2000.0)]
};

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
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// Shape of a speech-like programme.
#[derive(Clone, Copy, Debug)]
struct Style {
    /// Level spread between phrases ("speakers", level jumps), dB.
    phrase_db: f64,
    /// Level spread between syllables within a phrase, dB.
    syllable_db: f64,
    /// Pause between syllables, s (scaled by 0.5 … 1.5).
    gap_s: f64,
    /// Pause between phrases, s (scaled by 0.5 … 1.5).
    phrase_gap_s: f64,
    /// Accented syllables (shouts, laughs): probability and boost (dB).
    /// Rare loud accents set the peak, so they lower the loudness at a
    /// given peak (higher crest).
    accent: (f64, f64),
    seed: u64,
}

/// Dense, even speech (about −16 LUFS at −3 dBFS peak).
const DENSE: Style = Style {
    phrase_db: 3.0,
    syllable_db: 5.0,
    gap_s: 0.03,
    phrase_gap_s: 0.25,
    accent: (0.0, 0.0),
    seed: 0x5eed_0001,
};
/// Typical speech with level jumps (about −20 LUFS).
const TYPICAL: Style = Style {
    phrase_db: 10.0,
    syllable_db: 9.0,
    gap_s: 0.08,
    phrase_gap_s: 0.5,
    accent: (0.0, 0.0),
    seed: 0x5eed_0002,
};
/// Sparse speech with large jumps and high crest (about −24 LUFS).
const SPARSE: Style = Style {
    phrase_db: 14.0,
    syllable_db: 12.0,
    gap_s: 0.16,
    phrase_gap_s: 0.9,
    accent: (0.02, 12.0),
    seed: 0x5eed_0003,
};

/// Speech-like stereo programme, `SECONDS` long, fixed seed. Phrases of
/// 2 … 6 s at a random level (the level jumps), made of syllables of
/// 80 … 300 ms with a fast rise and slower fall, each a voiced harmonic
/// complex (f0 90 … 240 Hz with a small glide, 16 harmonics at 1/k) mixed
/// with low-passed noise (the unvoiced part), and a −70 dBFS noise floor.
/// Normalised so the largest sample is at −3 dBFS.
fn programme(style: Style) -> Vec<f32> {
    let fs = FS as f64;
    let n = (SECONDS * fs) as usize;
    let mut rng = Rng(style.seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut out = vec![0.0f64; n * CH];
    let mut lp = [0.0f64; CH];
    let mut pos = 0usize;
    let tau = 2.0 * std::f64::consts::PI;
    while pos < n {
        let phrase_end = pos + (rng.range(2.0, 6.0) * fs) as usize;
        let phrase_gain = db_to_lin(-style.phrase_db * rng.unit());
        let f0_base = rng.range(90.0, 240.0);
        while pos < phrase_end.min(n) {
            let len = (rng.range(0.08, 0.3) * fs) as usize;
            let accent = if rng.unit() < style.accent.0 {
                db_to_lin(style.accent.1)
            } else {
                1.0
            };
            let gain = accent * phrase_gain * db_to_lin(-style.syllable_db * rng.unit());
            let f0 = f0_base * rng.range(0.85, 1.15);
            let glide = rng.range(-0.15, 0.15);
            let voiced = rng.range(0.4, 1.0);
            let rise = 0.15 + 0.15 * rng.unit();
            let mut phase = 0.0f64;
            for i in 0..len.min(n - pos) {
                let u = i as f64 / len as f64;
                // Fast rise (raised cosine over `rise`), slower fall.
                let env = if u < rise {
                    0.5 - 0.5 * (std::f64::consts::PI * u / rise).cos()
                } else {
                    0.5 + 0.5 * (std::f64::consts::PI * (u - rise) / (1.0 - rise)).cos()
                };
                phase += tau * f0 * (1.0 + glide * u) / fs;
                let tone: f64 = (1..=16).map(|k| (phase * k as f64).sin() / k as f64).sum();
                for (c, l) in lp.iter_mut().enumerate() {
                    *l = 0.6 * *l + 0.4 * rng.next();
                    out[(pos + i) * CH + c] =
                        gain * env * (voiced * 0.4 * tone + (1.0 - voiced) * 1.2 * *l);
                }
            }
            pos += len + (style.gap_s * rng.range(0.5, 1.5) * fs) as usize;
        }
        pos += (style.phrase_gap_s * rng.range(0.5, 1.5) * fs) as usize;
    }
    let peak = out.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let k = db_to_lin(-3.0) / peak;
    out.iter()
        .map(|v| (v * k + db_to_lin(-70.0) * rng.next()) as f32)
        .collect()
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

/// The listening-test settings: threshold −20 dB, 4:1, 10 ms / 100 ms,
/// knee 0.5012 (12 dB).
fn listening_params() -> CompressorParams {
    CompressorParams {
        threshold: 0.1,
        ratio: 4.0,
        attack_ms: 10.0,
        release_ms: 100.0,
        makeup_gain: 1.0,
        knee: 0.5012,
    }
}

#[test]
fn static_makeup_formula() {
    // Hard knee, T = −30 dB, R = 4: reduction at the prior level.
    let gc = GainComputer::from_linear(db_to_lin(-30.0), 4.0, 1.0);
    let want = 0.75 * (AUTO_MAKEUP_PRIOR_DB + 30.0);
    assert!((static_makeup_db(&gc) - want).abs() < 1e-9);
    // Prior level below the knee: no makeup.
    let gc = GainComputer::from_linear(db_to_lin(-3.0), 10.0, 1.0);
    assert_eq!(static_makeup_db(&gc), 0.0);
    // Ratio 1: none.
    let gc = GainComputer::from_linear(db_to_lin(-30.0), 1.0, 0.0625);
    assert_eq!(static_makeup_db(&gc), 0.0);
    // Soft knee: the quadratic.
    let gc = GainComputer::from_linear(db_to_lin(-15.0), 2.0, db_to_lin(-6.0));
    let (t, w, s) = (-15.0, 12.0, 0.5);
    let d: f64 = AUTO_MAKEUP_PRIOR_DB - t + w / 2.0;
    assert!(d > 0.0 && d < w);
    assert!((static_makeup_db(&gc) - s * d * d / (2.0 * w)).abs() < 1e-6);
    // Bounded.
    let gc = GainComputer::from_linear(0.0, 100.0, 1.0);
    assert_eq!(static_makeup_db(&gc), AUTO_MAKEUP_MAX_DB);
}

/// A steady 1 kHz sine comes out at its input level (makeup 1) once the
/// estimate has adapted, whatever its level and the settings, and an
/// explicit makeup multiplies on top.
#[test]
fn steady_sine_keeps_its_level() {
    for (amp_db, t_db, ratio, makeup) in [
        (-6.0, -30.0, 6.0, 1.0f32),
        (-6.0, -30.0, 6.0, 2.0),
        (-20.0, -40.0, 10.0, 1.0),
        (-1.0, -12.0, 2.0, 1.0),
    ] {
        let p = CompressorParams {
            threshold: db_to_lin(t_db) as f32,
            ratio,
            knee: 0.5,
            attack_ms: 1.0,
            release_ms: 100.0,
            makeup_gain: makeup,
        };
        let amp = db_to_lin(amp_db);
        let x: Vec<f32> = (0..CH * 48_000 * 6)
            .map(|i| (amp * (2.0 * std::f64::consts::PI * (i / CH) as f64 / 48.0).sin()) as f32)
            .collect();
        let y = compressed(&x, &p, CompressorDetector::Peak, true);
        let peak = y[y.len() - 96 * CH..]
            .iter()
            .fold(0.0f64, |m, &v| m.max(v.abs() as f64));
        let got = lin_to_db(peak) - amp_db;
        assert!(
            (got - lin_to_db(makeup as f64)).abs() < 0.05,
            "{amp_db} dB, T {t_db}, R {ratio}: {got}"
        );
    }
}

/// The table of the listening-test report on the typical programme:
/// integrated output loudness with the listening-test settings, per
/// detector, with and without auto-makeup. Prints only (`--nocapture`).
#[test]
fn listening_settings_table() {
    let x = programme(TYPICAL);
    let input = loudness(&x);
    eprintln!("typical programme: {input:.2} LUFS in");
    for det in [CompressorDetector::Peak, CompressorDetector::Rms] {
        for auto in [false, true] {
            let out = loudness(&compressed(&x, &listening_params(), det, auto));
            eprintln!(
                "{det:?} auto-makeup {auto}: {out:.2} LUFS ({:+.2} LU)",
                out - input
            );
        }
    }
}

/// Loudness stays at the input loudness while the operator changes
/// compression, on all three programmes and in both detector modes.
///
/// Grid: thresholds −30, −25, −20, −15, −10 dB × ratios 2, 4, 6, 10;
/// attack 10 ms, release 100 ms, knee 0.5012 (the listening-test times)
/// and the default knee 0.0625 at 4:1. Every point within ±`TOL_LU` of the
/// input, measured over the whole programme (no exclusion: the estimate
/// starts from the static prior and needs no convergence time to stay
/// within the tolerance).
#[test]
fn loudness_follows_input_across_settings() {
    let mut failures = Vec::new();
    let mut worst: f64 = 0.0;
    let styles: &[Style] = if FULL {
        &[DENSE, TYPICAL, SPARSE]
    } else {
        &[DENSE, SPARSE]
    };
    for &style in styles {
        let x = programme(style);
        let input = loudness(&x);
        eprintln!("programme {:x}: {input:.2} LUFS", style.seed);
        for det in [CompressorDetector::Peak, CompressorDetector::Rms] {
            let mut grid: Vec<(f64, f32, f32)> = Vec::new();
            for &t_db in THRESHOLDS_DB {
                for &ratio in RATIOS {
                    grid.push((t_db, ratio, 0.5012));
                }
                if FULL {
                    grid.push((t_db, 4.0, 0.0625));
                }
            }
            let mut line = String::new();
            for (t_db, ratio, knee) in grid {
                let p = CompressorParams {
                    threshold: db_to_lin(t_db) as f32,
                    ratio,
                    knee,
                    ..listening_params()
                };
                let l = loudness(&compressed(&x, &p, det, true)) - input;
                line += &format!(" {l:+.1}");
                worst = worst.max(l.abs());
                if l.abs() > TOL_LU {
                    failures.push(format!(
                        "{:x} {det:?} T {t_db} R {ratio} k {knee}: {l:+.2} LU",
                        style.seed
                    ));
                }
            }
            eprintln!("  {det:?}:{line}");
        }
    }
    eprintln!("worst deviation {worst:.2} LU");
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The same holds for other attack and release times (the measured
/// correction covers the difference between the static curve and the
/// smoothed gain): typical programme, T −25 dB, 6:1.
#[test]
fn loudness_follows_input_across_times() {
    let x = programme(TYPICAL);
    let input = loudness(&x);
    for det in [CompressorDetector::Peak, CompressorDetector::Rms] {
        for &(attack_ms, release_ms) in TIMES {
            let p = CompressorParams {
                threshold: db_to_lin(-25.0) as f32,
                ratio: 6.0,
                attack_ms,
                release_ms,
                ..listening_params()
            };
            let l = loudness(&compressed(&x, &p, det, true)) - input;
            let l0 = loudness(&compressed(&x, &p, det, false)) - input;
            eprintln!("{det:?} {attack_ms}/{release_ms} ms: {l:+.2} LU (off {l0:+.2} LU)");
            assert!(l.abs() <= TOL_LU, "{det:?} {attack_ms}/{release_ms}: {l}");
        }
    }
}

/// The estimate freezes below the gate (silence, pauses, low noise),
/// never rises on noise, is bounded, and follows parameter changes at once.
#[test]
fn estimate_is_gated_bounded_and_follows_settings() {
    let x = programme(TYPICAL);
    let mut c = Compressor::new();
    c.set_auto_makeup(true);
    c.set_params(&listening_params());
    c.configure(FS, CH);
    let mut y = x.clone();
    for chunk in y.chunks_mut(1024 * CH) {
        c.process(chunk);
    }
    let after_programme = c.auto_makeup_gain_db();
    assert!(after_programme > 3.0, "{after_programme}");

    // 20 s of silence, then 20 s of noise 5 dB below the gate: frozen.
    let mut rng = Rng(7);
    for amp in [0.0, db_to_lin(AUTO_MAKEUP_GATE_DB - 5.0)] {
        let mut buf: Vec<f32> = (0..CH * 20 * 48_000)
            .map(|_| (amp * rng.next()) as f32)
            .collect();
        for chunk in buf.chunks_mut(1024 * CH) {
            c.process(chunk);
        }
        assert_eq!(c.auto_makeup_gain_db(), after_programme, "amp {amp}");
    }

    // Noise above the gate but below the knee (no compression): the
    // estimate can only fall, never pump up.
    let amp = db_to_lin(-40.0);
    let mut prev = after_programme;
    for _ in 0..30 {
        let mut buf: Vec<f32> = (0..CH * 48_000)
            .map(|_| (amp * rng.next()) as f32)
            .collect();
        c.process(&mut buf);
        let now = c.auto_makeup_gain_db();
        assert!(now <= prev + 1e-9, "{now} > {prev}");
        prev = now;
    }
    assert!(prev < after_programme - 1.0, "{prev}");

    // A new setting changes the target immediately (no re-adaptation).
    let mut c2 = Compressor::new();
    c2.set_auto_makeup(true);
    c2.set_params(&listening_params());
    c2.configure(FS, CH);
    let mut y = x.clone();
    for chunk in y.chunks_mut(1024 * CH) {
        c2.process(chunk);
    }
    let before = c2.auto_makeup_gain_db();
    c2.set_params(&CompressorParams {
        ratio: 10.0,
        ..listening_params()
    });
    assert!(c2.auto_makeup_gain_db() > before + 0.5);

    // Bounded, even at the most extreme setting.
    c2.set_params(&CompressorParams {
        threshold: 0.0,
        ratio: 100.0,
        ..listening_params()
    });
    assert_eq!(c2.auto_makeup_gain_db(), AUTO_MAKEUP_MAX_DB);
}

/// The estimate is updated per frame: the output does not depend on how
/// the caller splits the buffers, with a key or without.
#[test]
fn independent_of_buffer_size() {
    let x = programme(DENSE);
    let x = &x[..CH * 48_000 * 8];
    // `resend`: the host re-applies unchanged settings before every buffer
    // (as the GStreamer element does after any property change).
    let run = |frames: usize, key: bool, resend: bool| {
        let mut c = Compressor::new();
        c.set_detector(CompressorDetector::Rms);
        c.set_auto_makeup(true);
        c.set_params(&listening_params());
        c.configure(FS, CH);
        let mut y = x.to_vec();
        for (i, chunk) in y.chunks_mut(frames * CH).enumerate() {
            if resend {
                c.set_auto_makeup(true);
                c.set_params(&listening_params());
            }
            if key {
                let k = x[i * frames * CH..][..chunk.len()].to_vec();
                c.process_with_key(chunk, &k, CH);
            } else {
                c.process(chunk);
            }
        }
        y
    };
    for key in [false, true] {
        let a = run(1024, key, false);
        for (frames, resend) in [
            (1, false),
            (37, false),
            (480, false),
            (4096, false),
            (37, true),
        ] {
            let b = run(frames, key, resend);
            assert!(
                a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()),
                "{frames} frames, key {key}, resend {resend}"
            );
        }
    }
}

/// The programmes are what they claim: three loudness levels about 4 LU
/// apart, all peaking at −3 dBFS.
#[test]
fn programmes_span_loudness_levels() {
    let l: Vec<f64> = [DENSE, TYPICAL, SPARSE]
        .iter()
        .map(|&s| loudness(&programme(s)))
        .collect();
    eprintln!("programme loudness: {l:.2?}");
    for (got, want) in l.iter().zip([-16.0, -20.0, -24.0]) {
        assert!((got - want).abs() < 1.5, "{got} vs {want}");
    }
}

/// Toggling auto-makeup, and changing threshold and ratio while it is on,
/// is click-free on running programme: per-frame change of the gain
/// (output/input, dB) bounded as in the phase 1 parameter-change test.
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
    let mut prev: Option<f64> = None;
    let mut max_step: f64 = 0.0;
    for k in 0..12usize {
        if k.is_multiple_of(3) {
            c.set_auto_makeup(!c.auto_makeup());
        } else {
            c.set_params(&sets[k % 2]);
        }
        let mut buf = vec![level; 2400];
        c.process(&mut buf);
        for o in buf {
            let g = lin_to_db(o as f64 / level as f64);
            if let Some(p) = prev {
                max_step = max_step.max((g - p).abs());
            }
            prev = Some(g);
        }
    }
    // Largest change: about 30 dB (auto-makeup of 10:1 at −30 dB with a
    // −6 dB signal, below the bound), spread over a 10 ms one-pole:
    // ≈ 30/480 dB per frame at most, times 3 as in the phase 1 bound.
    assert!(max_step < 0.2, "max step {max_step} dB");
}
