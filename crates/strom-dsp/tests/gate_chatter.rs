//! Guard tests for the gate's detector (a bug found by ear, "chatter on noise
//! peaks"): steady noise below the threshold must keep the gate closed even
//! when its peaks exceed the threshold and open == close, while a
//! speech-like burst above the threshold still opens it promptly.

use strom_dsp::util::{db_to_lin, lin_to_db};
use strom_dsp::{Gate, GateParams, Processor};

const FS: f32 = 48_000.0;

struct Rng(u64);

impl Rng {
    /// Uniform in [−1, 1).
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// Pink (1/f) noise by the Voss method (M. Gardner, "White and brown music,
/// fractal curves and one-over-f fluctuations", Scientific American, 1978):
/// the sum of `ROWS` random values, row `k` renewed every `2^k` samples
/// (picked by the trailing zeros of a counter), plus a white term. Scaled to
/// an RMS of `rms_db` dBFS. Fixed seed.
fn pink_noise(frames: usize, rms_db: f64, seed: u64) -> Vec<f32> {
    const ROWS: usize = 16;
    let mut rng = Rng(seed | 1);
    let mut rows = [0.0f64; ROWS];
    for r in rows.iter_mut() {
        *r = rng.next_f64();
    }
    let mut sum: f64 = rows.iter().sum();
    let raw: Vec<f64> = (1..=frames as u64)
        .map(|n| {
            let k = (n.trailing_zeros() as usize).min(ROWS - 1);
            let v = rng.next_f64();
            sum += v - rows[k];
            rows[k] = v;
            sum + rng.next_f64()
        })
        .collect();
    let mean = raw.iter().sum::<f64>() / frames as f64;
    let rms = (raw.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / frames as f64).sqrt();
    let k = db_to_lin(rms_db) / rms;
    raw.iter().map(|x| ((x - mean) * k) as f32).collect()
}

fn params() -> GateParams {
    GateParams {
        open_threshold_db: -40.0,
        close_threshold_db: -40.0,
        attack_ms: 5.0,
        release_ms: 100.0,
        ..GateParams::default()
    }
}

/// Process `x` frame by frame; returns (open/close transitions, open frames).
fn run_counting(g: &mut Gate, x: &mut [f32]) -> (usize, usize) {
    let (mut transitions, mut open_frames) = (0, 0);
    let mut state = g.is_open();
    for chunk in x.chunks_mut(1) {
        g.process(chunk);
        if g.is_open() != state {
            transitions += 1;
            state = g.is_open();
        }
        open_frames += state as usize;
    }
    (transitions, open_frames)
}

#[test]
fn steady_pink_noise_below_threshold_keeps_gate_closed() {
    let p = params();
    let n = 5 * FS as usize;
    let mut noise = pink_noise(n, -51.0, 0x5eed);
    // The premise of the bug: the noise peaks do exceed the threshold.
    let peak = noise.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    assert!(
        lin_to_db(peak as f64) > -40.0,
        "peak {}",
        lin_to_db(peak as f64)
    );

    let mut g = Gate::new();
    g.set_params(&p);
    g.configure(FS, 1);
    let (transitions, open_frames) = run_counting(&mut g, &mut noise);
    assert_eq!(
        (transitions, open_frames),
        (0, 0),
        "gate opened on noise peaks"
    );
    assert!((lin_to_db(g.current_gain()) - p.reduction_db as f64).abs() < 1e-6);
}

#[test]
fn speech_like_burst_above_threshold_opens_within_attack() {
    let p = params();
    let pre = FS as usize; // 1 s of noise first
    let burst = (0.3 * FS) as usize;
    let mut x = pink_noise(pre + burst, -51.0, 0xfeed);
    // Harmonic complex (150 Hz, 12 harmonics at 1/k) with a 40 ms raised
    // onset, scaled to an RMS of −20 dBFS: a voiced syllable 20 dB above the
    // threshold.
    let tone: Vec<f64> = (0..burst)
        .map(|i| {
            let t = i as f64 / FS as f64;
            (1..=12)
                .map(|k| (2.0 * std::f64::consts::PI * 150.0 * k as f64 * t).sin() / k as f64)
                .sum::<f64>()
        })
        .collect();
    let rms = (tone.iter().map(|v| v * v).sum::<f64>() / burst as f64).sqrt();
    let scale = db_to_lin(-20.0) / rms;
    let onset = (0.040 * FS) as usize;
    for (i, v) in tone.iter().enumerate() {
        let env = if i < onset {
            0.5 - 0.5 * (std::f64::consts::PI * i as f64 / onset as f64).cos()
        } else {
            1.0
        };
        x[pre + i] += (v * scale * env) as f32;
    }

    let mut g = Gate::new();
    g.set_params(&p);
    g.configure(FS, 1);
    let mut gains = Vec::with_capacity(x.len());
    let mut opened_at = None;
    for (i, s) in x.iter().enumerate() {
        let mut f = [*s];
        g.process(&mut f);
        gains.push(g.current_gain());
        if opened_at.is_none() && g.is_open() {
            opened_at = Some(i);
        }
    }
    let opened_at = opened_at.expect("burst did not open the gate");
    assert!(opened_at >= pre, "opened on noise at frame {opened_at}");
    // The onset ramp reaches the threshold (−40 dB vs a −20 dB plateau,
    // i.e. 10 % amplitude) within its first ~8 ms; allow the detector's rise
    // on top (well under 5 ms at 20 dB above threshold).
    let ms = |m: f64| (m * 0.001 * FS as f64) as usize;
    assert!(
        opened_at - pre <= ms(15.0),
        "opened {} ms after the burst",
        (opened_at - pre) as f64 * 1000.0 / FS as f64
    );
    // Then the attack fade: after one attack time constant 1 − 1/e of the way.
    let red = db_to_lin(p.reduction_db as f64);
    let g_tau = gains[opened_at + ms(p.attack_ms as f64)];
    assert!((g_tau - red) / (1.0 - red) > 0.6, "gain {g_tau}");
    assert!(gains[pre + burst - 1] > 0.999);
}

/// Guard test for a bug found by ear, "gate closes much more slowly than
/// its release": after a burst stops (noise continues 11 dB below the
/// threshold, open == close), the gain must reach `reduction` within
/// `t_detector + hold + release`, and must not reopen on the noise.
///
/// `t_detector` is the time for the RMS detector to fall from the burst level
/// to the close threshold: the level decays with 20 ms, i.e. about 2.3 ms per
/// dB. The burst reads ≈ −17 dB (−20 dBFS RMS, sine-calibrated) and the
/// effective close threshold is −41 dB, so ≈ 24 dB → ≈ 55 ms; allowed: 70 ms.
/// The original one-pole release needed ≈ 6.9 · release to reach −60 dB and
/// fails this test.
#[test]
fn gate_closes_within_hold_plus_release_after_burst() {
    for (release_ms, hold_ms) in [(50.0f32, 20.0f32), (100.0, 50.0)] {
        let p = GateParams {
            release_ms,
            hold_ms,
            ..params()
        };
        let pre = FS as usize / 2;
        let burst = (0.3 * FS) as usize;
        let post = 2 * FS as usize;
        let mut x = pink_noise(pre + burst + post, -51.0, 0xc105e);
        let rms_target = db_to_lin(-20.0);
        let tone: Vec<f64> = (0..burst)
            .map(|i| (2.0 * std::f64::consts::PI * 300.0 * i as f64 / FS as f64).sin())
            .collect();
        let rms = (tone.iter().map(|v| v * v).sum::<f64>() / burst as f64).sqrt();
        for (i, v) in tone.iter().enumerate() {
            x[pre + i] += (v * rms_target / rms) as f32;
        }

        let mut g = Gate::new();
        g.set_params(&p);
        g.configure(FS, 1);
        let mut gains = Vec::with_capacity(x.len());
        let mut opens_after_burst = 0;
        let mut was_open = false;
        for (i, s) in x.iter().enumerate() {
            g.process(&mut [*s]);
            gains.push(g.current_gain());
            if i >= pre + burst && g.is_open() && !was_open {
                opens_after_burst += 1;
            }
            was_open = g.is_open();
        }
        let end = pre + burst;
        let ms = |m: f64| (m * 0.001 * FS as f64) as usize;
        let red = db_to_lin(p.reduction_db as f64);
        assert!(
            gains[end - 1] > 0.999,
            "gate not open at the end of the burst"
        );
        // Still fully open through the hold time (the detector has not even
        // fallen below the threshold yet).
        assert!(gains[end + ms(hold_ms as f64)] > 0.999);
        let closed_at = gains[end..]
            .iter()
            .position(|&v| v <= red)
            .expect("gate never closed");
        let budget = ms(70.0 + hold_ms as f64 + release_ms as f64);
        assert!(
            closed_at <= budget,
            "release {release_ms}, hold {hold_ms}: closed after {:.1} ms, budget {:.1} ms",
            closed_at as f64 * 1000.0 / FS as f64,
            budget as f64 * 1000.0 / FS as f64
        );
        assert_eq!(opens_after_burst, 0, "reopened on noise after the burst");
        assert!(gains[end + closed_at..].iter().all(|&v| v <= red));
    }
}
