//! Gate tests: thresholds, hysteresis, hold, attack/release time constants,
//! attenuation, linking, robustness and real-time safety.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use strom_dsp::gate::DETECTOR_RELEASE_MS;
use strom_dsp::util::{db_to_lin, lin_to_db, one_pole_coef};
use strom_dsp::{Gate, GateParams, Processor};

// ---------------------------------------------------------------------------
// Counting allocator (per thread, so parallel tests don't interfere).

struct CountingAlloc;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn alloc_count() -> usize {
    ALLOCS.with(|c| c.get())
}

// ---------------------------------------------------------------------------
// Helpers.

const FS: f32 = 48_000.0;
/// Constant probe level on a second channel, far below any threshold (−100 dBFS),
/// used to read the applied gain while the key channel is silent.
const PROBE: f32 = 1e-5;

fn gate(fs: f32, ch: usize, p: GateParams) -> Gate {
    let mut g = Gate::new();
    g.set_params(&p);
    g.configure(fs, ch);
    g
}

/// 1 kHz sine at 48 kHz whose samples hit the exact peak (n ≡ 12 mod 48).
fn sine_1k(amp: f32, n: usize, start: usize) -> Vec<f32> {
    (0..n)
        .map(|i| amp * ((2.0 * std::f64::consts::PI * ((start + i) as f64) / 48.0).sin() as f32))
        .collect()
}

/// Run a mono key signal through a 2-channel gate (channel 1 = probe) and return
/// the per-frame gain read from the probe channel.
fn run_with_probe(g: &mut Gate, key: &[f32]) -> Vec<f64> {
    let mut buf: Vec<f32> = key.iter().flat_map(|&k| [k, PROBE]).collect();
    g.process(&mut buf);
    buf.chunks_exact(2)
        .map(|f| f[1] as f64 / PROBE as f64)
        .collect()
}

fn ms(n_ms: f64) -> usize {
    (n_ms * 0.001 * FS as f64).round() as usize
}

// ---------------------------------------------------------------------------

#[test]
fn opens_only_above_open_threshold() {
    let p = GateParams::default(); // open −20, close −30
    for (level, should_open) in [(-21.0, false), (-19.0, true), (-25.0, false)] {
        let mut g = gate(FS, 2, p);
        let gains = run_with_probe(&mut g, &sine_1k(db_to_lin(level) as f32, ms(300.0), 0));
        assert_eq!(g.is_open(), should_open, "level {level}");
        let last = *gains.last().unwrap();
        if should_open {
            assert!((last - 1.0).abs() < 1e-6, "level {level}: gain {last}");
        } else {
            assert!(
                (lin_to_db(last) - p.reduction_db as f64).abs() < 1e-6,
                "level {level}: gain {last}"
            );
        }
    }
}

#[test]
fn closes_only_below_close_threshold_hysteresis() {
    let p = GateParams::default();
    for (level, should_stay_open) in [(-29.0, true), (-25.0, true), (-31.0, false)] {
        let mut g = gate(FS, 2, p);
        run_with_probe(&mut g, &sine_1k(db_to_lin(-10.0) as f32, ms(100.0), 0));
        assert!(g.is_open());
        let gains = run_with_probe(
            &mut g,
            &sine_1k(db_to_lin(level) as f32, ms(2000.0), ms(100.0)),
        );
        assert_eq!(g.is_open(), should_stay_open, "level {level}");
        let last = *gains.last().unwrap();
        if should_stay_open {
            assert!(
                gains.iter().all(|&x| (x - 1.0).abs() < 1e-9),
                "level {level}"
            );
        } else {
            // Attenuation equals reduction once closed.
            assert!((lin_to_db(last) - p.reduction_db as f64).abs() < 0.01);
        }
    }
}

#[test]
fn close_above_open_is_treated_as_equal() {
    let p = GateParams {
        open_threshold_db: -30.0,
        close_threshold_db: -10.0,
        hold_ms: 0.0,
        ..GateParams::default()
    };
    // Level between the two: above open → opens, and must stay open (close == open).
    let mut g = gate(FS, 2, p);
    run_with_probe(&mut g, &sine_1k(db_to_lin(-20.0) as f32, ms(500.0), 0));
    assert!(g.is_open());
    // Just below the open threshold → closes.
    run_with_probe(
        &mut g,
        &sine_1k(db_to_lin(-32.0) as f32, ms(500.0), ms(500.0)),
    );
    assert!(!g.is_open());
}

#[test]
fn attenuation_equals_reduction() {
    for red in [-80.0f32, -60.0, -40.0, -20.0, -6.0, 0.0] {
        let p = GateParams {
            reduction_db: red,
            ..GateParams::default()
        };
        let mut g = gate(FS, 2, p);
        run_with_probe(&mut g, &vec![0.5; ms(100.0)]);
        // Silence: release (100 ms) needs ≈ 20 τ to reach within 0.01 dB of −80 dB.
        let gains = run_with_probe(&mut g, &vec![0.0; ms(3000.0)]);
        let got = lin_to_db(*gains.last().unwrap());
        assert!(
            (got - red as f64).abs() < 0.01,
            "reduction {red}: got {got}"
        );
    }
}

#[test]
fn hold_time() {
    for hold in [0.0f32, 20.0, 50.0, 300.0] {
        let p = GateParams {
            hold_ms: hold,
            ..GateParams::default()
        };
        let mut g = gate(FS, 2, p);
        let amp = 0.5f64;
        run_with_probe(&mut g, &vec![amp as f32; ms(100.0)]);
        let gains = run_with_probe(&mut g, &vec![0.0; ms(hold as f64 + 300.0)]);
        // The (sine-calibrated RMS) detector reads a constant |x| = amp as
        // √2·amp, and decays from there to the close threshold in
        // τ_d·ln(√2·amp/close) (discrete: n with a_d^n·√2·amp < close), then
        // hold runs.
        let start = std::f64::consts::SQRT_2 * amp;
        let a_d = one_pole_coef(DETECTOR_RELEASE_MS, FS as f64);
        let close = db_to_lin(p.close_threshold_db as f64);
        let decay = ((close / start).ln() / a_d.ln()).ceil() as usize;
        let expected = decay + ms(hold as f64);
        let first_drop = gains.iter().position(|&x| x < 1.0).unwrap();
        assert!(
            (first_drop as i64 - expected as i64).abs() <= 2,
            "hold {hold}: first drop at {first_drop}, expected {expected}"
        );
    }
}

#[test]
fn attack_time_constant() {
    for attack in [0.5f32, 5.0, 50.0, 200.0] {
        let p = GateParams {
            attack_ms: attack,
            ..GateParams::default()
        };
        let mut g = gate(FS, 1, p);
        let red = db_to_lin(p.reduction_db as f64);
        let n = ms(attack as f64);
        let mut buf = vec![0.5f32; n + ms(10.0)];
        g.process(&mut buf);
        // The RMS detector needs a moment to reach the threshold (0.2 ms
        // here); the attack fade starts on the first frame with a rising gain.
        let start = buf
            .iter()
            .position(|&x| x as f64 / 0.5 > red * 1.001)
            .unwrap();
        assert!(start <= ms(1.0), "opened after {start} frames");
        // After τ the linear gain covered 1 − 1/e of the way from `red` to 1.
        let gain = buf[start + n - 1] as f64 / 0.5;
        let frac = (gain - red) / (1.0 - red);
        let expect = 1.0 - (-1.0f64).exp();
        assert!(
            (frac - expect).abs() < 0.01,
            "attack {attack}: fraction {frac}"
        );
    }
}

/// Release is the time to close: once the gate starts falling, the gain is a
/// straight line in dB from 0 dB to `reduction` and arrives after `release` ms
/// (found by ear on speech; it used to be a one-pole time constant).
#[test]
fn release_closes_over_full_range_in_release_time() {
    for (release, red_db) in [
        (1.0f32, -60.0f32),
        (20.0, -60.0),
        (100.0, -80.0),
        (2000.0, -20.0),
    ] {
        let p = GateParams {
            release_ms: release,
            reduction_db: red_db,
            hold_ms: 0.0,
            ..GateParams::default()
        };
        let mut g = gate(FS, 2, p);
        let red = db_to_lin(red_db as f64);
        run_with_probe(&mut g, &vec![0.5; ms(300.0)]);
        assert_eq!(g.current_gain(), 1.0);
        let gains = run_with_probe(&mut g, &vec![0.0; ms(release as f64 + 200.0)]);
        let start = gains.iter().position(|&x| x < 1.0).unwrap();
        let n = ms(release as f64).max(1);
        // Halfway in time: halfway in dB (linear ramp), for ramps of several samples.
        if n >= 8 {
            let mid = lin_to_db(gains[start + n / 2 - 1]);
            assert!(
                (mid - red_db as f64 / 2.0).abs() < 0.1,
                "release {release}: {mid} dB at half time"
            );
        }
        // Closed within release (+1 frame for rounding), not before release − 1 frame.
        // The gain is measured through an f32 probe: compare with 1e-6 relative slack.
        let closed_at = gains.iter().position(|&x| x <= red * (1.0 + 1e-6)).unwrap() - start;
        assert!(
            closed_at + 1 >= n.saturating_sub(1) && closed_at <= n + 1,
            "release {release}: closed after {closed_at} frames, expected {n}"
        );
    }
}

#[test]
fn no_chatter_between_thresholds() {
    // Amplitude-modulated sine hovering between −28 and −22 dBFS (between close −30
    // and open −20), with hold 0 so only hysteresis prevents chatter.
    let p = GateParams {
        hold_ms: 0.0,
        ..GateParams::default()
    };
    let n = ms(3000.0);
    let hover: Vec<f32> = sine_1k(1.0, n, 0)
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let db = -25.0 + 3.0 * (2.0 * std::f64::consts::PI * 7.0 * i as f64 / FS as f64).sin();
            s * db_to_lin(db) as f32
        })
        .collect();

    for start_open in [true, false] {
        let mut g = gate(FS, 1, p);
        if start_open {
            g.process(&mut sine_1k(0.5, ms(50.0), 0));
            assert!(g.is_open());
        }
        let mut transitions = 0;
        let mut state = g.is_open();
        for chunk in hover.chunks(16) {
            let mut c = chunk.to_vec();
            g.process(&mut c);
            if g.is_open() != state {
                transitions += 1;
                state = g.is_open();
            }
        }
        assert_eq!(transitions, 0, "start_open {start_open}");
        assert_eq!(g.is_open(), start_open);
    }
}

#[test]
fn linked_channels() {
    for ch in [2usize, 8] {
        let mut g = gate(FS, ch, GateParams::default());
        // Key only on channel 0; all others carry the probe.
        let frames = ms(400.0);
        let mut buf = vec![PROBE; frames * ch];
        for f in 0..frames {
            buf[f * ch] = if f < ms(200.0) { 0.5 } else { PROBE };
        }
        let input = buf.clone();
        g.process(&mut buf);
        let mut saw_open = false;
        let mut saw_closed = false;
        for f in 0..frames {
            let g0 = buf[f * ch] as f64 / input[f * ch] as f64;
            for c in 1..ch {
                let gc = buf[f * ch + c] as f64 / input[f * ch + c] as f64;
                assert!((gc - g0).abs() <= 1e-6 * g0.max(1e-4), "frame {f} ch {c}");
            }
            saw_open |= g0 > 0.999;
            saw_closed |= g0 < 0.01;
        }
        assert!(saw_open && saw_closed);
    }
}

// ---------------------------------------------------------------------------
// Robustness.

fn extreme_param_sets() -> Vec<GateParams> {
    let mut sets = Vec::new();
    for bits in 0u32..64 {
        let pick = |i: u32, lo: f32, hi: f32| if bits & (1 << i) != 0 { hi } else { lo };
        sets.push(GateParams {
            open_threshold_db: pick(0, -80.0, 0.0),
            close_threshold_db: pick(1, -80.0, 0.0),
            attack_ms: pick(2, 0.01, 200.0),
            hold_ms: pick(3, 0.0, 1000.0),
            release_ms: pick(4, 1.0, 2000.0),
            reduction_db: pick(5, -80.0, 0.0),
        });
    }
    sets
}

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Test signal: loud noise (+20 dBFS), silence, an impulse, then quiet DC.
fn torture_signal(frames: usize, ch: usize, rng: &mut Rng) -> Vec<f32> {
    let mut v = vec![0.0f32; frames * ch];
    for (i, s) in v.iter_mut().enumerate() {
        let f = i / ch;
        *s = match (f * 4) / frames {
            0 => 10.0 * rng.next_f32(),
            1 => 0.0,
            2 => {
                if f == frames / 2 {
                    1.0
                } else {
                    0.0
                }
            }
            _ => 1e-3,
        };
    }
    v
}

#[test]
fn extremes_no_nan_inf_all_rates_and_channels() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let sets = extreme_param_sets();
    for fs in [8_000.0f32, 44_100.0, 48_000.0, 96_000.0, 192_000.0] {
        for ch in [1usize, 2, 8, 64] {
            // All 64 corner combinations at stereo; a subset for other counts to keep runtime sane.
            let chosen: Vec<&GateParams> = if ch == 2 {
                sets.iter().collect()
            } else {
                sets.iter().step_by(9).collect()
            };
            for p in chosen {
                let mut g = gate(fs, ch, *p);
                let frames = (fs as usize) / 50;
                let mut buf = torture_signal(frames, ch, &mut rng);
                g.process(&mut buf);
                assert!(buf.iter().all(|x| x.is_finite()), "fs {fs} ch {ch} {p:?}");
                // Switch to the opposite corner mid-stream.
                let q = sets[63 - sets.iter().position(|s| s == p).unwrap()];
                g.set_params(&q);
                let mut buf = torture_signal(frames, ch, &mut rng);
                g.process(&mut buf);
                assert!(buf.iter().all(|x| x.is_finite()), "fs {fs} ch {ch} {q:?}");
                assert!(g.current_gain().is_finite());
            }
        }
    }
}

#[test]
fn out_of_range_and_nan_params_are_clamped() {
    let mut g = gate(FS, 2, GateParams::default());
    g.set_params(&GateParams {
        open_threshold_db: 100.0,
        close_threshold_db: -1000.0,
        attack_ms: 0.0,
        hold_ms: -5.0,
        release_ms: f32::INFINITY,
        reduction_db: f32::NAN,
    });
    let p = *g.params();
    assert_eq!(p.open_threshold_db, 0.0);
    assert_eq!(p.close_threshold_db, -80.0);
    assert_eq!(p.attack_ms, 0.01);
    assert_eq!(p.hold_ms, 0.0);
    assert_eq!(p.release_ms, 2000.0);
    assert_eq!(p.reduction_db, GateParams::default().reduction_db);
    let mut buf = vec![0.3f32; 2000];
    g.process(&mut buf);
    assert!(buf.iter().all(|x| x.is_finite()));
}

/// Jump bound: whatever parameters change, the linear gain moves per frame by at most
/// `(1 − a_min)·(1 − reduction_lin_min)`, where `a_min = exp(−1/(τ_min·fs))` is the
/// one-pole coefficient of the shortest attack/release time in use — i.e. a parameter
/// change never moves the gain faster than the configured fades allow. The parameter
/// sets used here have all fade times ≥ 2 ms, so the bound is ≈ 1 % per frame at 48 kHz.
#[test]
fn parameter_changes_do_not_jump() {
    let sets = [
        GateParams {
            attack_ms: 2.0,
            release_ms: 2.0,
            ..GateParams::default()
        },
        GateParams {
            open_threshold_db: -50.0,
            close_threshold_db: -60.0,
            attack_ms: 10.0,
            hold_ms: 0.0,
            release_ms: 5.0,
            reduction_db: -20.0,
        },
        GateParams {
            open_threshold_db: -5.0,
            close_threshold_db: -6.0,
            attack_ms: 2.0,
            hold_ms: 10.0,
            release_ms: 50.0,
            reduction_db: -80.0,
        },
        GateParams {
            reduction_db: 0.0,
            attack_ms: 3.0,
            release_ms: 2.0,
            ..GateParams::default()
        },
    ];
    let bound = 1.0 - one_pole_coef(2.0, FS as f64) + 1e-9;
    let mut g = gate(FS, 2, sets[0]);
    let key = sine_1k(db_to_lin(-15.0) as f32, ms(4000.0), 0);
    let mut prev = g.current_gain();
    let mut max_step: f64 = 0.0;
    for (i, chunk) in key.chunks(ms(37.0)).enumerate() {
        g.set_params(&sets[i % sets.len()]);
        for gain in run_with_probe(&mut g, chunk) {
            max_step = max_step.max((gain - prev).abs());
            prev = gain;
        }
    }
    assert!(max_step <= bound, "max step {max_step} > bound {bound}");
    assert!(max_step > 0.0);
}

#[test]
fn reset_clears_state() {
    let mut g = gate(FS, 1, GateParams::default());
    g.process(&mut vec![0.5f32; 1000]);
    assert!(g.is_open());
    g.reset();
    assert!(!g.is_open());
    assert!((lin_to_db(g.current_gain()) - (-60.0)).abs() < 1e-9);
}

#[test]
fn unconfigured_is_passthrough() {
    let mut g = Gate::new();
    let mut buf = vec![0.25f32, -0.5, 0.0];
    g.process(&mut buf);
    assert_eq!(buf, vec![0.25, -0.5, 0.0]);
}

#[test]
fn no_allocation_in_process_or_set_params() {
    let mut g = Gate::new();
    g.configure(48_000.0, 8);
    let mut buf = vec![0.0f32; 8 * 1024];
    let mut rng = Rng(42);
    for s in buf.iter_mut() {
        *s = rng.next_f32();
    }
    let other = GateParams {
        open_threshold_db: -40.0,
        reduction_db: -10.0,
        ..GateParams::default()
    };
    let def = GateParams::default();
    let before = alloc_count();
    for i in 0..50 {
        g.set_params(if i % 2 == 0 { &other } else { &def });
        g.process(&mut buf);
    }
    g.reset();
    g.process(&mut buf);
    assert_eq!(alloc_count() - before, 0);
}

// ---------------------------------------------------------------------------
// Call order: set_params before or after configure/reset gives identical output.

#[test]
fn set_params_before_or_after_configure_is_identical() {
    let p = GateParams {
        open_threshold_db: -30.0,
        close_threshold_db: -40.0,
        attack_ms: 1.0,
        hold_ms: 10.0,
        release_ms: 20.0,
        reduction_db: -20.0,
    };
    let mut rng = Rng(7);
    let input: Vec<f32> = (0..2 * 9_600)
        .map(|i| rng.next_f32() * if (i / 2400) % 2 == 0 { 0.5 } else { 0.001 })
        .collect();

    let mut a = Gate::new();
    a.set_params(&p);
    a.configure(48_000.0, 2);
    let mut b = Gate::new();
    b.configure(48_000.0, 2);
    b.set_params(&p);
    // After a reset, too.
    let mut c = Gate::new();
    c.configure(48_000.0, 2);
    c.process(&mut input.clone());
    c.reset();
    c.set_params(&p);

    let (mut ya, mut yb, mut yc) = (input.clone(), input.clone(), input.clone());
    a.process(&mut ya);
    b.process(&mut yb);
    c.process(&mut yc);
    assert!(ya.iter().zip(&yb).all(|(x, y)| x.to_bits() == y.to_bits()));
    assert!(ya.iter().zip(&yc).all(|(x, y)| x.to_bits() == y.to_bits()));
}
