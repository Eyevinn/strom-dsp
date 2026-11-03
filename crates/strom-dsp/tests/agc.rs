//! Tests of the wideband AGC ([`strom_dsp::agc::Agc`]), from its documented
//! behaviour and BS.1770 measurement. All signals are synthetic
//! (pink noise made by inverse FFT, fixed seeds).
//!
//! - Convergence: pink noise from −40 to −6 LUFS lands on the target ±1 LU
//!   within the documented time
//!   `0.4 s·10^((gate − L)/10) + 0.41 s + (|Δ| − 1 dB)/rate`, measured on
//!   the gain and on the output with the `ebur128` reference meter.
//! - Pause: a 10 s pause (digital silence, or room tone below the gate)
//!   moves the gain ≤ 1 dB, and there is no overshoot afterwards.
//! - Gate: programme below the gate never moves the gain.
//! - Bounds and rate limits hold, also while parameters jump.
//! - Robustness, reset, bit-exact passthrough at 0 dB, zero allocations.

mod agc_support;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use agc_support::{pink, reference_loudness, scale_to_lufs, Looper, Rng};
use strom_dsp::agc::{Agc, AgcParams};
use strom_dsp::meter::{ChannelLayout, LoudnessDetector};
use strom_dsp::Processor;

// ---------------------------------------------------------------------------
// Counting allocator (per thread).

struct CountingAlloc;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn alloc_count() -> usize {
    ALLOCS.with(|c| c.get())
}

// ---------------------------------------------------------------------------

const FS: f64 = 48_000.0;
const BLOCK: usize = 480;
const NOISE_FRAMES: usize = 1 << 19;

/// Documented convergence bound (module docs of `agc`): the gate opens once
/// the momentary loudness of input at `level` passes the gate, then 400 ms of
/// gated programme and one sub-block, then the rate-limited ramp to within
/// 1 dB.
fn convergence_bound_s(level: f64, gate: f64, delta_db: f64, rate_db_per_s: f64) -> f64 {
    0.4 * 10f64.powf((gate - level) / 10.0) + 0.41 + (delta_db.abs() - 1.0).max(0.0) / rate_db_per_s
}

fn agc(params: AgcParams) -> Agc {
    let mut a = Agc::new();
    a.configure_with_layout(FS as f32, 2, ChannelLayout::Stereo);
    a.set_params(&params);
    a
}

/// Stereo pink noise at `lufs` (reference meter), looped.
fn programme(lufs: f64, seed: u64) -> Vec<f32> {
    let mut x = pink(NOISE_FRAMES, 2, FS, seed);
    scale_to_lufs(&mut x, 2, FS, lufs);
    x
}

/// A source of input blocks.
enum Source<'a> {
    Noise(Looper<'a>),
    Silence,
}

impl Source<'_> {
    fn fill(&mut self, out: &mut [f32]) {
        match self {
            Source::Noise(l) => l.fill(out),
            Source::Silence => out.fill(0.0),
        }
    }
}

/// Run `seconds` of `src` through `a` in 10 ms blocks; returns the gain after
/// every block and the output.
fn run(a: &mut Agc, src: &mut Source, seconds: f64) -> (Vec<f64>, Vec<f32>) {
    let blocks = (seconds * FS) as usize / BLOCK;
    let mut gains = Vec::with_capacity(blocks);
    let mut out = Vec::with_capacity(blocks * BLOCK * 2);
    let mut buf = vec![0.0f32; BLOCK * 2];
    for _ in 0..blocks {
        src.fill(&mut buf);
        a.process(&mut buf);
        gains.push(a.gain_db());
        out.extend_from_slice(&buf);
    }
    (gains, out)
}

#[test]
fn converges_to_target_from_minus_40_to_minus_6_lufs() {
    let target = -23.0;
    let params = AgcParams {
        target_lufs: target as f32,
        max_gain_db: 20.0,
        max_cut_db: 20.0,
        ..AgcParams::default()
    };
    for (i, level) in [-40.0, -35.0, -30.0, -26.0, -23.0, -20.0, -15.0, -10.0, -6.0]
        .into_iter()
        .enumerate()
    {
        let x = programme(level, 100 + i as u64);
        let mut a = agc(params);
        let delta = target - level;
        let rate = if delta > 0.0 {
            f64::from(params.rate_up_db_per_s)
        } else {
            f64::from(params.rate_down_db_per_s)
        };
        // Plus 0.1 s: the gated loudness of 3 s of pink noise scatters by
        // a few hundredths of a LU, which shifts the 1 dB crossing.
        let bound = convergence_bound_s(level, f64::from(params.gate_lufs), delta, rate) + 0.1;
        let seconds = bound + 12.0;
        let (gains, out) = run(&mut a, &mut Source::Noise(Looper::new(&x)), seconds);
        // Gain: within 1 dB of the ideal from the bound on.
        let first = (bound * FS / BLOCK as f64).ceil() as usize;
        let mut reached = None;
        for (b, g) in gains.iter().enumerate() {
            if (g - delta).abs() <= 1.0 && reached.is_none() {
                reached = Some(b as f64 * BLOCK as f64 / FS);
            }
            if b >= first {
                assert!(
                    (g - delta).abs() <= 1.0,
                    "level {level}: gain {g} at {:.2} s, want {delta}",
                    b as f64 * BLOCK as f64 / FS
                );
            }
        }
        // Output: integrated loudness of the last 10 s (reference meter).
        let tail = &out[out.len() - (10.0 * FS) as usize * 2..];
        let lout = reference_loudness(tail, 2, FS, ChannelLayout::Stereo);
        println!(
            "input {level:>5} LUFS: within 1 dB after {:.2} s (bound {bound:.2} s), \
             final gain {:+.2} dB, output {lout:.2} LUFS",
            reached.unwrap_or(f64::NAN),
            gains.last().unwrap()
        );
        assert!((lout - target).abs() <= 1.0, "level {level}: output {lout}");
    }
}

/// Programme, a 10 s pause, programme again; returns (gains, output) and the
/// block index where the pause starts and ends.
fn pause_scenario(
    before: f64,
    pause: Option<f64>,
    after: f64,
) -> (Vec<f64>, Vec<f32>, usize, usize) {
    let mut a = agc(AgcParams::default());
    let x1 = programme(before, 7);
    let x2 = programme(after, 8);
    let (mut gains, mut out) = run(&mut a, &mut Source::Noise(Looper::new(&x1)), 20.0);
    let p0 = gains.len();
    let (g, o) = match pause {
        None => run(&mut a, &mut Source::Silence, 10.0),
        Some(level) => {
            let tone = programme(level, 9);
            run(&mut a, &mut Source::Noise(Looper::new(&tone)), 10.0)
        }
    };
    gains.extend(g);
    out.extend(o);
    let p1 = gains.len();
    let (g, o) = run(&mut a, &mut Source::Noise(Looper::new(&x2)), 15.0);
    gains.extend(g);
    out.extend(o);
    (gains, out, p0, p1)
}

#[test]
fn pause_holds_the_gain_and_recovery_does_not_overshoot() {
    let target = -23.0;
    // (before, pause, after): silence and room tone at −65 LUFS (below the
    // −50 LUFS gate), same level after, louder after, quieter after.
    let cases = [
        (-30.0, None, -30.0),
        (-30.0, Some(-65.0), -30.0),
        (-15.0, None, -15.0),
        (-30.0, None, -20.0),
        (-20.0, Some(-65.0), -30.0),
        (-30.0, None, -40.0),
    ];
    for (before, pause, after) in cases {
        let (gains, out, p0, p1) = pause_scenario(before, pause, after);
        let held = gains[p0 - 1];
        let drift = gains[p0..p1]
            .iter()
            .map(|g| (g - held).abs())
            .fold(0.0, f64::max);
        let settled = (target - after).clamp(-12.0, 12.0);
        let after_gains = &gains[p1..];
        let (lo, hi) = after_gains
            .iter()
            .fold((f64::MAX, f64::MIN), |(l, h), &g| (l.min(g), h.max(g)));
        // No overshoot: the gain stays between where it was held and where
        // it settles (0.25 dB slack for the noise in the gated measurement).
        let (want_lo, want_hi) = (held.min(settled) - 0.25, held.max(settled) + 0.25);
        // Output momentary loudness after the pause never exceeds the larger
        // of the target and the held-gain level (no blast from the pause).
        let mut det = LoudnessDetector::new();
        det.configure(FS as f32, 2, ChannelLayout::Stereo);
        let mut max_m = f64::MIN;
        for (b, blk) in out.as_chunks::<{ BLOCK * 2 }>().0.iter().enumerate() {
            det.process(blk);
            if b >= p1 + 40 {
                max_m = max_m.max(det.momentary_lufs());
            }
        }
        let blast_bound = target.max(after + held) + 1.0;
        println!(
            "{before} / {pause:?} / {after}: held {held:+.2} dB, pause drift {drift:.3} dB, \
             after-pause gain {lo:+.2} … {hi:+.2} (settles {settled:+.2}), \
             max momentary {max_m:.2} LUFS"
        );
        assert!(drift <= 1.0, "gain moved {drift} dB during the pause");
        assert!(lo >= want_lo && hi <= want_hi, "overshoot: {lo} … {hi}");
        assert!((after_gains.last().unwrap() - settled).abs() <= 1.0);
        assert!(max_m <= blast_bound, "momentary {max_m} > {blast_bound}");
    }
}

#[test]
fn below_the_gate_the_gain_never_moves() {
    // From reset: programme at −60 LUFS (gate −50) for 30 s.
    let x = programme(-60.0, 21);
    let mut a = agc(AgcParams::default());
    let (gains, out) = run(&mut a, &mut Source::Noise(Looper::new(&x)), 30.0);
    assert!(gains.iter().all(|&g| g == 0.0));
    assert!(a.is_holding());
    // 0 dB is bit-exact passthrough.
    let expect: Vec<f32> = {
        let mut l = Looper::new(&x);
        let mut v = vec![0.0f32; out.len()];
        l.fill(&mut v);
        v
    };
    assert!(out == expect);

    // A raised gate holds programme that would otherwise be lifted.
    let x = programme(-35.0, 22);
    let mut a = agc(AgcParams {
        gate_lufs: -30.0,
        ..AgcParams::default()
    });
    let (gains, _) = run(&mut a, &mut Source::Noise(Looper::new(&x)), 20.0);
    assert!(gains.iter().all(|&g| g == 0.0));
}

#[test]
fn max_gain_and_max_cut_bounds_hold() {
    for (level, max_gain, max_cut) in [
        (-45.0, 12.0, 12.0),
        (-3.0, 12.0, 12.0),
        (-40.0, 3.0, 0.0),
        (-8.0, 0.0, 5.0),
    ] {
        let x = programme(level, 31);
        let mut a = agc(AgcParams {
            max_gain_db: max_gain,
            max_cut_db: max_cut,
            ..AgcParams::default()
        });
        let (gains, _) = run(&mut a, &mut Source::Noise(Looper::new(&x)), 25.0);
        let (lo, hi) = gains
            .iter()
            .fold((f64::MAX, f64::MIN), |(l, h), &g| (l.min(g), h.max(g)));
        assert!(hi <= f64::from(max_gain) + 1e-12 && lo >= -f64::from(max_cut) - 1e-12);
        let want = (-23.0 - level).clamp(-f64::from(max_cut), f64::from(max_gain));
        assert!(
            (gains.last().unwrap() - want).abs() < 1e-9,
            "{level}: {:?}",
            gains.last()
        );
    }
}

#[test]
fn lowering_a_bound_glides_into_range_at_the_rate_limit() {
    let x = programme(-35.0, 41);
    let mut a = agc(AgcParams::default());
    run(&mut a, &mut Source::Noise(Looper::new(&x)), 15.0);
    assert!((a.gain_db() - 12.0).abs() < 1e-9);
    a.set_params(&AgcParams {
        max_gain_db: 2.0,
        ..AgcParams::default()
    });
    let (gains, _) = run(&mut a, &mut Source::Noise(Looper::new(&x)), 5.0);
    let step = 3.0 * BLOCK as f64 / FS;
    let mut prev = 12.0;
    for &g in &gains {
        assert!(prev - g <= step + 1e-9 && g <= prev + 1e-12);
        prev = g;
    }
    assert!((gains.last().unwrap() - 2.0).abs() < 1e-9);
}

#[test]
fn rate_limits_hold_and_down_is_faster() {
    for (rate_up, rate_down) in [(1.0f32, 3.0f32), (20.0, 40.0), (0.1, 0.1), (5.0, 2.0)] {
        let params = AgcParams {
            rate_up_db_per_s: rate_up,
            rate_down_db_per_s: rate_down,
            max_gain_db: 15.0,
            max_cut_db: 15.0,
            ..AgcParams::default()
        };
        // Up: −38 LUFS needs +15 dB; then down: −8 LUFS needs −15 dB.
        let lo = programme(-38.0, 51);
        let hi = programme(-8.0, 52);
        let mut a = agc(params);
        let (mut gains, _) = run(&mut a, &mut Source::Noise(Looper::new(&lo)), 8.0);
        let switch = gains.len();
        gains.extend(run(&mut a, &mut Source::Noise(Looper::new(&hi)), 8.0).0);
        let up = f64::from(rate_up) * BLOCK as f64 / FS;
        let down = f64::from(rate_down) * BLOCK as f64 / FS;
        let mut prev = 0.0;
        let (mut max_up, mut max_down) = (0.0f64, 0.0f64);
        for &g in &gains {
            let d = g - prev;
            max_up = max_up.max(d);
            max_down = max_down.max(-d);
            prev = g;
        }
        assert!(max_up <= up + 1e-9, "up {max_up} > {up}");
        assert!(max_down <= down + 1e-9, "down {max_down} > {down}");
        // While far from target the gain moves at the full rate.
        if rate_up < 15.0 {
            assert!((max_up - up).abs() < 1e-9);
        }
        assert!((max_down - down).abs() < 1e-9);
        let _ = switch;
    }

    // Per sample, from the output/input ratio during a ramp.
    let x = programme(-35.0, 53);
    let mut a = agc(AgcParams::default());
    let mut input = vec![0.0f32; (2.0 * FS) as usize * 2];
    Looper::new(&x).fill(&mut input);
    let mut out = input.clone();
    a.process(&mut out);
    let step = 1.0 / FS; // 1 dB/s
    let mut prev: Option<f64> = None;
    let mut worst: f64 = 0.0;
    for (i, o) in input.iter().zip(&out).step_by(2) {
        if i.abs() > 0.01 {
            let g = 20.0 * (f64::from(*o) / f64::from(*i)).log10();
            if let Some(p) = prev {
                worst = worst.max(g - p);
            }
            prev = Some(g);
        } else {
            prev = None;
        }
    }
    assert!(worst <= step + 2e-5, "per-sample step {worst} dB > {step}");
}

#[test]
fn parameter_jumps_are_bounded_by_the_rate() {
    let x = programme(-28.0, 61);
    let mut a = agc(AgcParams::default());
    let mut looper = Looper::new(&x);
    let mut rng = Rng::new(62);
    let mut buf = vec![0.0f32; BLOCK * 2];
    let mut prev = 0.0;
    for b in 0..6000 {
        if b % 10 == 0 {
            let pick = |r: &mut Rng, lo: f32, hi: f32| {
                [lo, hi, lo + (hi - lo) * r.uniform() as f32][(r.next_u64() % 3) as usize]
            };
            let p = AgcParams {
                target_lufs: pick(&mut rng, -40.0, -5.0),
                max_gain_db: pick(&mut rng, 0.0, 30.0),
                max_cut_db: pick(&mut rng, 0.0, 30.0),
                rate_up_db_per_s: pick(&mut rng, 0.1, 20.0),
                rate_down_db_per_s: pick(&mut rng, 0.1, 40.0),
                gate_lufs: pick(&mut rng, -80.0, -20.0),
                window_ms: pick(&mut rng, 400.0, 10_000.0),
            };
            a.set_params(&p);
        }
        looper.fill(&mut buf);
        a.process(&mut buf);
        let g = a.gain_db();
        let p = a.params();
        let limit = f64::from(p.rate_up_db_per_s.max(p.rate_down_db_per_s)) * BLOCK as f64 / FS;
        assert!((g - prev).abs() <= limit.max(40.0 * BLOCK as f64 / FS) + 1e-9);
        assert!(buf.iter().all(|v| v.is_finite()));
        prev = g;
    }
}

#[test]
fn robust_at_every_rate_channel_count_and_extreme_input() {
    let extremes = [
        AgcParams::default(),
        AgcParams {
            target_lufs: -5.0,
            max_gain_db: 30.0,
            max_cut_db: 0.0,
            rate_up_db_per_s: 20.0,
            rate_down_db_per_s: 40.0,
            gate_lufs: -80.0,
            window_ms: 400.0,
        },
        AgcParams {
            target_lufs: -40.0,
            max_gain_db: 0.0,
            max_cut_db: 30.0,
            rate_up_db_per_s: 0.1,
            rate_down_db_per_s: 0.1,
            gate_lufs: -20.0,
            window_ms: 10_000.0,
        },
        AgcParams {
            target_lufs: f32::NAN,
            max_gain_db: f32::INFINITY,
            max_cut_db: -1.0,
            rate_up_db_per_s: f32::NAN,
            rate_down_db_per_s: 1e9,
            gate_lufs: f32::NEG_INFINITY,
            window_ms: 0.0,
        },
    ];
    let mut rng = Rng::new(71);
    for fs in [
        8_000.0f32, 22_050.0, 44_100.0, 48_000.0, 96_000.0, 192_000.0,
    ] {
        for channels in [1usize, 2, 3, 6, 8] {
            for p in &extremes {
                let mut a = Agc::new();
                a.configure(fs, channels);
                a.set_params(p);
                let cp = a.params();
                assert!(
                    cp.target_lufs.is_finite() && cp.max_gain_db <= 30.0 && cp.max_cut_db >= 0.0
                );
                let n = (fs as usize / 5) * channels;
                for round in 0..8 {
                    let mut buf: Vec<f32> = (0..n)
                        .map(|i| match round {
                            0 => 1e-3 * rng.bipolar() as f32,
                            1 => 1e-40,
                            2 => [f32::NAN, f32::INFINITY, -f32::INFINITY, 0.5][i % 4],
                            3 => f32::MAX * rng.bipolar() as f32,
                            4 => 0.0,
                            _ => 0.3 * rng.bipolar() as f32,
                        })
                        .collect();
                    let finite_in: Vec<bool> = buf.iter().map(|v| v.is_finite()).collect();
                    a.process(&mut buf);
                    assert!(a.gain_db().is_finite() && a.gain_db().abs() <= 30.0 + 1e-9);
                    assert!(a.measured_lufs().is_finite());
                    for (v, fin) in buf.iter().zip(finite_in) {
                        assert!(!fin || v.is_finite(), "fs {fs} ch {channels} round {round}");
                    }
                }
                a.reset();
                assert_eq!(a.gain_db(), 0.0);
            }
        }
    }
    // Unconfigured: a no-op.
    let mut a = Agc::new();
    let mut buf = [0.5f32, -0.5];
    a.process(&mut buf);
    assert_eq!(buf, [0.5, -0.5]);
}

#[test]
fn reset_returns_to_unity_and_passthrough() {
    let x = programme(-35.0, 81);
    let mut a = agc(AgcParams::default());
    run(&mut a, &mut Source::Noise(Looper::new(&x)), 6.0);
    assert!(a.gain_db() > 1.0);
    a.reset();
    assert_eq!(a.gain_db(), 0.0);
    assert!(a.is_holding());
    let mut buf = vec![0.0f32; BLOCK * 2];
    Looper::new(&x).fill(&mut buf);
    let orig = buf.clone();
    a.process(&mut buf);
    assert_eq!(buf, orig);
}

#[test]
fn speech_like_programme_with_gaps_reaches_target() {
    // Noise with a syllabic (4 Hz) envelope and 300 ms gaps of silence
    // every second: the gaps are gated out of the measurement.
    let target = -23.0;
    let mut x = pink(NOISE_FRAMES, 2, FS, 91);
    for (n, f) in x.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let t = n as f64 / FS;
        let env = if t.fract() >= 0.7 {
            0.0
        } else {
            0.55 + 0.45 * (2.0 * std::f64::consts::PI * 4.0 * t).sin()
        };
        f.iter_mut().for_each(|v| *v = (*v as f64 * env) as f32);
    }
    scale_to_lufs(&mut x, 2, FS, -32.0);
    let mut a = agc(AgcParams::default());
    let (_, out) = run(&mut a, &mut Source::Noise(Looper::new(&x)), 30.0);
    let tail = &out[out.len() - (10.0 * FS) as usize * 2..];
    let lout = reference_loudness(tail, 2, FS, ChannelLayout::Stereo);
    println!(
        "speech-like at −32 LUFS: gain {:+.2} dB, output {lout:.2} LUFS",
        a.gain_db()
    );
    assert!((lout - target).abs() <= 1.0, "{lout}");
}

#[test]
fn process_set_params_and_reset_do_not_allocate() {
    let x = programme(-30.0, 101);
    let mut a = Agc::new();
    a.configure_with_layout(FS as f32, 2, ChannelLayout::Stereo);
    let mut looper = Looper::new(&x);
    let mut bufs: Vec<Vec<f32>> = [480usize, 1, 7, 1024, 4096]
        .iter()
        .map(|&n| vec![0.0f32; n * 2])
        .collect();
    let before = alloc_count();
    let mut sink = 0.0;
    for i in 0..3000 {
        let buf = &mut bufs[i % 5];
        looper.fill(buf);
        a.process(buf);
        sink += a.gain_db() + a.measured_lufs() + a.detector().short_term_lufs();
        if i % 100 == 0 {
            a.set_params(&AgcParams {
                window_ms: [3000.0, 400.0, 10_000.0][(i / 100) % 3],
                target_lufs: -20.0,
                ..AgcParams::default()
            });
            a.set_channel_layout(ChannelLayout::Unknown);
        }
        if i % 1000 == 999 {
            a.reset();
        }
    }
    assert_eq!(alloc_count(), before, "allocated in the processing path");
    assert!(sink.is_finite());
}
