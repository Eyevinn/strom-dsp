//! Guard test for the listening-test bug "limiter distorts bass at fast
//! settings": total harmonic distortion of steady sines through the limiter,
//! well above its ceiling (−13 dBFS sine into a −25 dB ceiling), lookahead
//! 5 ms, attack 1 ms, release 50 ms and 100 ms.
//!
//! With a sliding-minimum window of only `attack + 1` frames the gain released
//! between the waveform peaks and rippled with every half-cycle: 3.4 % THD at
//! 50 Hz and 0.18 % at 200 Hz (1 ms / 50 ms). Targets: ≤ 1 % at 50 Hz,
//! ≤ 0.3 % at 100 Hz, ≤ 0.1 % at 200 Hz.
//!
//! THD is computed as in `compressor_thd.rs`: a DFT evaluated at the harmonic
//! frequencies over one second at 48 kHz, which holds an integer number of
//! cycles of every test frequency, so there is no spectral leakage:
//! `THD = sqrt(Σ_{k=2..10} A_k²) / A_1`.

use strom_dsp::{Limiter, LimiterParams, Processor};

const FS: f32 = 48_000.0;
const CH: usize = 2;
/// −13 dBFS peak, as in the listening test.
const AMP_DB: f64 = -13.0;
/// Ceiling 12 dB below the sine peak.
const CEILING_DB: f32 = -25.0;

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

/// Run a stereo sine of `freq` Hz at `AMP_DB` through a limiter with `p` in
/// 1024-frame buffers for 3 s. Returns the THD of the left channel over the
/// last second and the largest output magnitude over all of it.
fn measure(p: &LimiterParams, freq: f64) -> (f64, f32, Limiter) {
    let mut l = Limiter::new();
    l.set_params(p);
    l.configure(FS, CH);
    let amp = 10f64.powf(AMP_DB / 20.0);
    let frames = 3 * FS as usize;
    let mut buf: Vec<f32> = (0..frames)
        .flat_map(|n| {
            let v = (amp * (2.0 * std::f64::consts::PI * freq * n as f64 / FS as f64).sin()) as f32;
            [v; CH]
        })
        .collect();
    for chunk in buf.chunks_mut(1024 * CH) {
        l.process(chunk);
    }
    let peak = buf.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let last: Vec<f64> = buf[(frames - FS as usize) * CH..]
        .iter()
        .step_by(CH)
        .map(|&v| v as f64)
        .collect();
    (thd(&last, freq), peak, l)
}

fn params(attack_ms: f32, release_ms: f32) -> LimiterParams {
    LimiterParams {
        enabled: true,
        threshold_db: CEILING_DB,
        attack_ms,
        release_ms,
        lookahead_ms: 5.0,
    }
}

fn pct(v: f64) -> String {
    format!("{:.5} %", 100.0 * v)
}

#[test]
fn thd_meter_is_accurate() {
    let n = FS as usize;
    let x: Vec<f64> = (0..n)
        .map(|i| {
            let t = 2.0 * std::f64::consts::PI * 50.0 * i as f64 / FS as f64;
            t.sin() + 0.01 * (3.0 * t).sin() + 0.002 * (5.0 * t).cos()
        })
        .collect();
    let want = (0.01f64.powi(2) + 0.002f64.powi(2)).sqrt();
    assert!((thd(&x, 50.0) - want).abs() < 1e-9);
}

/// Targets of the listening-test fix: attack 1 ms, release 50 ms and 100 ms,
/// lookahead 5 ms. The ceiling and the safety clip are checked too.
#[test]
fn limiter_thd_fast_times_meets_targets() {
    let mut failures = Vec::new();
    for release_ms in [50.0f32, 100.0] {
        for (f, limit) in [(50.0, 0.01), (100.0, 0.003), (200.0, 0.001)] {
            let p = params(1.0, release_ms);
            let (t, peak, l) = measure(&p, f);
            eprintln!(
                "limiter {f} Hz (1 ms / {release_ms} ms, lookahead 5 ms): THD {} (limit {})",
                pct(t),
                pct(limit),
            );
            if t > limit {
                failures.push(format!(
                    "{f} Hz, release {release_ms} ms: THD {} > {}",
                    pct(t),
                    pct(limit)
                ));
            }
            assert!(peak <= l.ceiling_linear(), "{f} Hz: peak {peak}");
            assert_eq!(l.safety_clip_count(), 0);
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The other listening-test settings (1 ms / 10 ms, 5 ms / 200 ms) and more
/// frequencies, including ones whose period is not a whole number of frames.
/// With a 10 ms release the hold is 10 ms, still at least one peak of `|x|`
/// from 50 Hz up, so the gain is steady here too. Bound: 0.01 %.
#[test]
fn limiter_thd_other_settings() {
    for (attack_ms, release_ms) in [(1.0f32, 10.0f32), (5.0, 200.0)] {
        for f in [50.0, 55.0, 60.0, 100.0, 130.0, 200.0, 440.0] {
            let (t, peak, l) = measure(&params(attack_ms, release_ms), f);
            eprintln!(
                "limiter {f} Hz ({attack_ms} ms / {release_ms} ms, lookahead 5 ms): THD {}",
                pct(t)
            );
            assert!(
                t < 1e-4,
                "{f} Hz, {attack_ms} / {release_ms} ms: {}",
                pct(t)
            );
            assert!(peak <= l.ceiling_linear());
            assert_eq!(l.safety_clip_count(), 0);
        }
    }
}
