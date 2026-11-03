//! Guard test for the listening-test bug "compressor distorts bass with fast
//! times": total harmonic distortion of steady sines through the compressor
//! at attack 1 ms, release 50 ms, ratio 10:1, well above the threshold.
//!
//! With the phase 1 detector (per-frame peak `|x|` straight into the gain
//! computer) the gain followed every half-cycle of the waveform: 2.5 % THD at
//! 60 Hz and 0.77 % at 200 Hz. Targets: ≤ 1 % at 50/60 Hz, ≤ 0.3 % at
//! 100 Hz, ≤ 0.1 % at 200 Hz, and lower with slower times.
//!
//! THD is computed here from a DFT evaluated at the harmonic frequencies over
//! an integer number of cycles (one second at 48 kHz holds an integer number
//! of cycles of every test frequency), so there is no spectral leakage:
//! `THD = sqrt(Σ_{k=2..10} A_k²) / A_1`.
//!
//! The same measurement is run on the gate (open) and the limiter (well
//! above its ceiling, fast settings) and reported.

use strom_dsp::{
    Compressor, CompressorParams, Gate, GateParams, Limiter, LimiterParams, Processor,
};

const FS: f32 = 48_000.0;
const CH: usize = 2;
/// −13 dBFS peak, as in the listening test.
const AMP_DB: f64 = -13.0;

/// Amplitude of the DFT bin at `freq` Hz of `x` (one channel), `x.len()`
/// covering an integer number of cycles of `freq`.
fn dft_amplitude(x: &[f64], freq: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * freq / FS as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for (n, &v) in x.iter().enumerate() {
        let ph = w * n as f64;
        re += v * ph.cos();
        im -= v * ph.sin();
    }
    2.0 * (re * re + im * im).sqrt() / x.len() as f64
}

/// THD (ratio, not %) over harmonics 2…10 of `freq`.
fn thd(x: &[f64], freq: f64) -> f64 {
    let a1 = dft_amplitude(x, freq);
    let h: f64 = (2..=10)
        .map(|k| dft_amplitude(x, k as f64 * freq).powi(2))
        .sum();
    h.sqrt() / a1
}

/// Run a stereo sine of `freq` Hz at `AMP_DB` through `proc` in 1024-frame
/// buffers for 3 s and return the THD of the left channel over the last second.
fn measure<P: Processor>(proc: &mut P, freq: f64) -> f64 {
    proc.configure(FS, CH);
    let amp = 10f64.powf(AMP_DB / 20.0);
    let frames = 3 * FS as usize;
    let mut buf: Vec<f32> = (0..frames)
        .flat_map(|n| {
            let v = (amp * (2.0 * std::f64::consts::PI * freq * n as f64 / FS as f64).sin()) as f32;
            [v; CH]
        })
        .collect();
    for chunk in buf.chunks_mut(1024 * CH) {
        proc.process(chunk);
    }
    let last: Vec<f64> = buf[(frames - FS as usize) * CH..]
        .iter()
        .step_by(CH)
        .map(|&v| v as f64)
        .collect();
    thd(&last, freq)
}

fn listening_test_params() -> CompressorParams {
    CompressorParams {
        threshold: 0.0316,
        ratio: 10.0,
        attack_ms: 1.0,
        release_ms: 50.0,
        makeup_gain: 1.0,
        knee: 0.5012,
    }
}

fn pct(v: f64) -> String {
    format!("{:.5} %", 100.0 * v)
}

#[test]
fn thd_meter_is_accurate() {
    // A sine plus a known 3rd and 5th harmonic.
    let n = FS as usize;
    let x: Vec<f64> = (0..n)
        .map(|i| {
            let t = 2.0 * std::f64::consts::PI * 60.0 * i as f64 / FS as f64;
            t.sin() + 0.01 * (3.0 * t).sin() + 0.002 * (5.0 * t).cos()
        })
        .collect();
    let want = (0.01f64.powi(2) + 0.002f64.powi(2)).sqrt();
    assert!((thd(&x, 60.0) - want).abs() < 1e-9);
}

#[test]
fn input_sine_is_clean() {
    struct Through;
    impl Processor for Through {
        fn configure(&mut self, _: f32, _: usize) {}
        fn process(&mut self, _: &mut [f32]) {}
        fn reset(&mut self) {}
    }
    for f in [50.0, 60.0, 100.0, 200.0] {
        let t = measure(&mut Through, f);
        eprintln!("input {f} Hz: THD {}", pct(t));
        assert!(t < 1e-4);
    }
}

/// Targets of the listening-test fix with attack 1 ms, release 50 ms, 10:1.
#[test]
fn compressor_thd_fast_times_meets_targets() {
    let p = listening_test_params();
    let mut failures = Vec::new();
    for (f, limit) in [(50.0, 0.01), (60.0, 0.01), (100.0, 0.003), (200.0, 0.001)] {
        let mut c = Compressor::new();
        c.set_params(&p);
        let t = measure(&mut c, f);
        let red = c.gain_reduction_db();
        eprintln!(
            "compressor {f} Hz (1 ms / 50 ms): THD {} (limit {}), reduction {red:.2} dB",
            pct(t),
            pct(limit),
        );
        if t > limit {
            failures.push(format!("{f} Hz: THD {} > {}", pct(t), pct(limit)));
        }
        // Well above threshold: the reduction is the static one
        // (0.9 · 17 dB = 15.3 dB).
        if (red - 15.3).abs() > 0.1 {
            failures.push(format!("{f} Hz: reduction {red:.3} dB, static 15.3 dB"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Slower times distort less.
#[test]
fn compressor_thd_decreases_with_slower_times() {
    for f in [50.0, 100.0, 200.0] {
        let mut prev = f64::INFINITY;
        for (attack_ms, release_ms) in [(1.0, 50.0), (10.0, 200.0), (30.0, 1000.0)] {
            let mut c = Compressor::new();
            c.set_params(&CompressorParams {
                attack_ms,
                release_ms,
                ..listening_test_params()
            });
            let t = measure(&mut c, f);
            eprintln!(
                "compressor {f} Hz ({attack_ms} ms / {release_ms} ms): THD {}",
                pct(t)
            );
            assert!(t <= prev * 1.05 + 1e-5, "{f} Hz: THD rose to {}", pct(t));
            prev = t;
        }
    }
}

/// Gate, open (the sine is 27 dB above the open threshold): the gain is
/// constant 1, so the output is as clean as the input.
#[test]
fn gate_open_adds_no_distortion() {
    for f in [50.0, 60.0, 100.0, 200.0] {
        let mut g = Gate::new();
        g.set_params(&GateParams {
            open_threshold_db: -40.0,
            close_threshold_db: -50.0,
            attack_ms: 0.01,
            hold_ms: 0.0,
            release_ms: 1.0,
            reduction_db: -60.0,
        });
        let t = measure(&mut g, f);
        eprintln!("gate (open, fastest times) {f} Hz: THD {}", pct(t));
        assert!(t < 1e-4, "{f} Hz: {}", pct(t));
    }
}

/// Limiter well above its ceiling (−13 dBFS sine into a −25 dB ceiling) with
/// fast settings. Reported for information; the assertion is only the
/// ceiling and a loose sanity bound.
#[test]
fn limiter_thd_report() {
    for (attack_ms, release_ms, lookahead_ms) in [
        (1.0f32, 50.0f32, 5.0f32),
        (1.0, 10.0, 5.0),
        (5.0, 200.0, 5.0),
    ] {
        for f in [50.0, 60.0, 100.0, 200.0] {
            let mut l = Limiter::new();
            l.set_params(&LimiterParams {
                enabled: true,
                threshold_db: -25.0,
                attack_ms,
                release_ms,
                lookahead_ms,
            });
            let t = measure(&mut l, f);
            eprintln!(
                "limiter {f} Hz (attack {attack_ms} ms, release {release_ms} ms, lookahead {lookahead_ms} ms): THD {}",
                pct(t)
            );
            assert!(t.is_finite());
            assert_eq!(l.safety_clip_count(), 0);
        }
    }
}
