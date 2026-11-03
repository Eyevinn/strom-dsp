//! Stereo processor (`strom_dsp::stereo`): every mode's matrix exactly, bit-exact
//! routing where it is a pure copy/swap/invert, constant-power balance, width
//! 1 = passthrough, the M/S round trip, the smoothing bound for parameter
//! changes and mode switches, and no allocation after setup.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::f64::consts::{FRAC_1_SQRT_2, SQRT_2};

use strom_dsp::stereo::{
    balance_gains, matrix_for, ramp_frames, Matrix, MonoLaw, Stereo, StereoMode, StereoParams,
};
use strom_dsp::Processor;

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
const MODES: [StereoMode; 7] = [
    StereoMode::Stereo,
    StereoMode::Mono,
    StereoMode::LeftToBoth,
    StereoMode::RightToBoth,
    StereoMode::Swap,
    StereoMode::MsEncode,
    StereoMode::MsDecode,
];

struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn noise(frames: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..frames * 2).map(|_| rng.next_f32()).collect()
}

/// Stereo test signal with a few special values (sign of zero, subnormal,
/// large values) that bit-exact paths must preserve.
fn special_signal() -> Vec<f32> {
    let mut x = noise(4096, 3);
    x[10] = -0.0;
    x[11] = f32::MIN_POSITIVE / 4.0;
    x[12] = 3.0e38;
    x[13] = -3.0e38;
    x
}

fn processed(p: &StereoParams, input: &[f32]) -> Vec<f32> {
    let mut s = Stereo::new();
    s.set_params(p);
    s.configure(FS, 2);
    let mut out = input.to_vec();
    for chunk in out.chunks_mut(2 * 333) {
        s.process(chunk);
    }
    out
}

fn with_mode(mode: StereoMode) -> StereoParams {
    StereoParams {
        mode,
        ..StereoParams::default()
    }
}

/// Maximum absolute row sum.
fn norm(m: &Matrix) -> f64 {
    m.iter()
        .map(|r| r[0].abs() + r[1].abs())
        .fold(0.0, f64::max)
}

fn diff(a: &Matrix, b: &Matrix) -> Matrix {
    [
        [a[0][0] - b[0][0], a[0][1] - b[0][1]],
        [a[1][0] - b[1][0], a[1][1] - b[1][1]],
    ]
}

// ---------------------------------------------------------------------------
// Matrices.

#[test]
fn mode_matrices_are_exact() {
    let k3 = FRAC_1_SQRT_2;
    let expected: [(StereoMode, MonoLaw, Matrix); 8] = [
        (
            StereoMode::Stereo,
            MonoLaw::Minus3Db,
            [[1.0, 0.0], [0.0, 1.0]],
        ),
        (StereoMode::Mono, MonoLaw::Minus3Db, [[k3, k3], [k3, k3]]),
        (
            StereoMode::Mono,
            MonoLaw::Minus6Db,
            [[0.5, 0.5], [0.5, 0.5]],
        ),
        (
            StereoMode::LeftToBoth,
            MonoLaw::Minus3Db,
            [[1.0, 0.0], [1.0, 0.0]],
        ),
        (
            StereoMode::RightToBoth,
            MonoLaw::Minus3Db,
            [[0.0, 1.0], [0.0, 1.0]],
        ),
        (
            StereoMode::Swap,
            MonoLaw::Minus3Db,
            [[0.0, 1.0], [1.0, 0.0]],
        ),
        (
            StereoMode::MsEncode,
            MonoLaw::Minus3Db,
            [[0.5, 0.5], [0.5, -0.5]],
        ),
        (
            StereoMode::MsDecode,
            MonoLaw::Minus3Db,
            [[1.0, 1.0], [1.0, -1.0]],
        ),
    ];
    for (mode, law, m) in expected {
        let p = StereoParams {
            mode,
            mono_law: law,
            ..StereoParams::default()
        };
        assert_eq!(matrix_for(&p), m, "{mode:?} {law:?}");
        // Polarity is applied to the inputs: it negates matrix columns.
        for (il, ir) in [(true, false), (false, true), (true, true)] {
            let q = StereoParams {
                invert_left: il,
                invert_right: ir,
                ..p
            };
            let sl = if il { -1.0 } else { 1.0 };
            let sr = if ir { -1.0 } else { 1.0 };
            let want = [[m[0][0] * sl, m[0][1] * sr], [m[1][0] * sl, m[1][1] * sr]];
            let got = matrix_for(&q);
            for r in 0..2 {
                for c in 0..2 {
                    assert_eq!(got[r][c], want[r][c], "{mode:?} inv ({il},{ir})");
                }
            }
        }
    }
    // Disabled is the identity whatever the settings.
    let p = StereoParams {
        enabled: false,
        mode: StereoMode::Mono,
        balance: 0.7,
        width: 0.0,
        invert_left: true,
        ..StereoParams::default()
    };
    assert_eq!(matrix_for(&p), [[1.0, 0.0], [0.0, 1.0]]);
}

/// Pure routing (copy/swap/invert) is bit-exact, including -0.0, subnormals
/// and values near f32::MAX; the defaults leave the buffer untouched (NaN
/// payloads included).
#[test]
fn routing_modes_are_bit_exact() {
    let x = special_signal();
    for mode in [
        StereoMode::Stereo,
        StereoMode::LeftToBoth,
        StereoMode::RightToBoth,
        StereoMode::Swap,
    ] {
        for (il, ir) in [(false, false), (true, false), (false, true), (true, true)] {
            let p = StereoParams {
                mode,
                invert_left: il,
                invert_right: ir,
                ..StereoParams::default()
            };
            let out = processed(&p, &x);
            for (o, i) in out.chunks_exact(2).zip(x.chunks_exact(2)) {
                let l = if il { -i[0] } else { i[0] };
                let r = if ir { -i[1] } else { i[1] };
                let want = match mode {
                    StereoMode::Stereo => [l, r],
                    StereoMode::LeftToBoth => [l, l],
                    StereoMode::RightToBoth => [r, r],
                    _ => [r, l],
                };
                assert_eq!(o[0].to_bits(), want[0].to_bits(), "{mode:?} ({il},{ir})");
                assert_eq!(o[1].to_bits(), want[1].to_bits(), "{mode:?} ({il},{ir})");
            }
        }
    }
    let mut y = x.clone();
    y[20] = f32::from_bits(0x7fc0_1234); // NaN with payload
    let out = processed(&StereoParams::default(), &y);
    for (a, b) in out.iter().zip(&y) {
        assert_eq!(a.to_bits(), b.to_bits(), "defaults must be bit-exact");
    }
}

/// Non-routing modes compute `T·x` in f64 with a single rounding to f32.
#[test]
fn mixing_modes_match_their_formula() {
    let x = noise(4096, 11);
    type Formula = fn(f64, f64) -> (f64, f64);
    let cases: [(StereoParams, Formula); 5] = [
        (with_mode(StereoMode::Mono), |l, r| {
            (
                FRAC_1_SQRT_2 * l + FRAC_1_SQRT_2 * r,
                FRAC_1_SQRT_2 * l + FRAC_1_SQRT_2 * r,
            )
        }),
        (
            StereoParams {
                mono_law: MonoLaw::Minus6Db,
                ..with_mode(StereoMode::Mono)
            },
            |l, r| ((l + r) * 0.5, (l + r) * 0.5),
        ),
        (with_mode(StereoMode::MsEncode), |l, r| {
            ((l + r) * 0.5, (l - r) * 0.5)
        }),
        (with_mode(StereoMode::MsDecode), |m, s| (m + s, m - s)),
        (
            StereoParams {
                width: 0.0,
                ..StereoParams::default()
            },
            |l, r| ((l + r) * 0.5, (l + r) * 0.5),
        ),
    ];
    for (p, f) in cases {
        let out = processed(&p, &x);
        for (o, i) in out.chunks_exact(2).zip(x.chunks_exact(2)) {
            let (l, r) = f(f64::from(i[0]), f64::from(i[1]));
            assert_eq!(o[0], l as f32, "{p:?}");
            assert_eq!(o[1], r as f32, "{p:?}");
        }
    }
}

#[test]
#[allow(clippy::needless_range_loop)] // matrix product reads clearest with indices
fn ms_round_trip() {
    let x = noise(48_000, 5);
    let enc = processed(&with_mode(StereoMode::MsEncode), &x);
    let dec = processed(&with_mode(StereoMode::MsDecode), &enc);
    for (o, i) in dec.chunks_exact(2).zip(x.chunks_exact(2)) {
        // Two f32 roundings of the intermediate M/S signal: ≤ 1 ulp of the
        // larger input each.
        let tol = 2.0 * f32::EPSILON * i[0].abs().max(i[1].abs());
        assert!((o[0] - i[0]).abs() <= tol && (o[1] - i[1]).abs() <= tol);
    }
    // The matrices are exact inverses.
    let e = matrix_for(&with_mode(StereoMode::MsEncode));
    let d = matrix_for(&with_mode(StereoMode::MsDecode));
    for (r, dr) in d.iter().enumerate() {
        for c in 0..2 {
            let v = dr[0] * e[0][c] + dr[1] * e[1][c];
            assert_eq!(v, if r == c { 1.0 } else { 0.0 });
        }
    }
}

#[test]
fn balance_is_constant_power() {
    assert_eq!(balance_gains(0.0), (1.0, 1.0));
    assert_eq!(balance_gains(-1.0), (SQRT_2, 0.0));
    assert_eq!(balance_gains(1.0), (0.0, SQRT_2));
    let mut prev = balance_gains(-1.0);
    for i in -100..=100 {
        let b = i as f32 / 100.0;
        let (gl, gr) = balance_gains(b);
        assert!((gl * gl + gr * gr - 2.0).abs() < 1e-12, "b = {b}");
        // Pan law: θ = (b+1)π/4.
        let theta = (f64::from(b) + 1.0) * std::f64::consts::FRAC_PI_4;
        assert!((gl - SQRT_2 * theta.cos()).abs() < 1e-12);
        assert!(gl <= prev.0 && gr >= prev.1, "monotonic");
        prev = (gl, gr);
    }
    // Measured: uncorrelated noise keeps its total power at every balance.
    let x = noise(48_000, 9);
    let power = |v: &[f32]| v.iter().map(|&s| f64::from(s).powi(2)).sum::<f64>();
    let p_in = power(&x);
    for b in [-1.0f32, -0.6, -0.2, 0.3, 0.8, 1.0] {
        let p = StereoParams {
            balance: b,
            ..StereoParams::default()
        };
        let out = processed(&p, &x);
        let ratio = power(&out) / p_in;
        assert!((ratio - 1.0).abs() < 0.02, "b = {b}: power ratio {ratio}");
    }
    // Out-of-range and NaN are clamped / neutral.
    assert_eq!(balance_gains(5.0), balance_gains(1.0));
    assert_eq!(balance_gains(f32::NAN), (1.0, 1.0));
}

#[test]
fn width_definition() {
    let x = noise(4096, 13);
    // Width 1 is an exact passthrough (identity matrix, buffer untouched).
    let out = processed(
        &StereoParams {
            width: 1.0,
            ..StereoParams::default()
        },
        &x,
    );
    assert_eq!(out, x);
    // S' = w·S with M unchanged.
    for w in [0.0f32, 0.5, 1.5, 2.0] {
        let p = StereoParams {
            width: w,
            ..StereoParams::default()
        };
        let out = processed(&p, &x);
        for (o, i) in out.chunks_exact(2).zip(x.chunks_exact(2)) {
            let (m_in, s_in) = (
                (f64::from(i[0]) + f64::from(i[1])) / 2.0,
                (f64::from(i[0]) - f64::from(i[1])) / 2.0,
            );
            let (m_out, s_out) = (
                (f64::from(o[0]) + f64::from(o[1])) / 2.0,
                (f64::from(o[0]) - f64::from(o[1])) / 2.0,
            );
            assert!((m_out - m_in).abs() < 1e-6, "w {w}");
            assert!((s_out - f64::from(w) * s_in).abs() < 1e-6, "w {w}");
        }
    }
}

// ---------------------------------------------------------------------------
// Smoothing.

/// After any change the matrix ramps linearly over `ramp_frames`; every output
/// step stays within `‖T‖·D + ‖ΔT‖·A` (module docs), and once the ramp is over
/// the output equals a processor that started with the new settings.
#[test]
fn changes_are_smoothed_within_bound() {
    let settings: Vec<StereoParams> = {
        let mut v: Vec<StereoParams> = MODES.iter().map(|&m| with_mode(m)).collect();
        v.push(StereoParams {
            balance: -1.0,
            width: 2.0,
            invert_left: true,
            ..StereoParams::default()
        });
        v.push(StereoParams {
            balance: 0.9,
            width: 0.0,
            mode: StereoMode::MsDecode,
            ..StereoParams::default()
        });
        v.push(StereoParams {
            enabled: false,
            ..with_mode(StereoMode::Swap)
        });
        v
    };
    let n = ramp_frames(FS);
    assert_eq!(n, 480);
    // A smooth input: two sines, amplitude 0.8.
    let frames = 3 * n;
    let x: Vec<f32> = (0..frames * 2)
        .map(|i| {
            let (f, c) = (i / 2, i % 2);
            let hz = if c == 0 { 440.0 } else { 1234.0 };
            0.8 * (2.0 * std::f32::consts::PI * hz * f as f32 / FS).sin()
        })
        .collect();
    let mut d_in = 0.0f64;
    for f in 1..frames {
        for c in 0..2 {
            d_in = d_in.max(f64::from((x[f * 2 + c] - x[(f - 1) * 2 + c]).abs()));
        }
    }
    let a_in = 0.8f64;
    for from in &settings {
        for to in &settings {
            let mut s = Stereo::new();
            s.set_params(from);
            s.configure(FS, 2);
            // Settle on `from` with the first half buffer.
            let mut warm = x[..n * 2].to_vec();
            s.process(&mut warm);
            let t0 = matrix_for(from);
            let t1 = matrix_for(to);
            s.set_params(to);
            let mut out = x.clone();
            s.process(&mut out[..7 * 2]);
            s.process(&mut out[7 * 2..]);
            assert!(!s.is_ramping());
            assert_eq!(s.current_matrix(), t1);
            let bound = norm(&t0).max(norm(&t1)) * d_in + norm(&diff(&t1, &t0)) / n as f64 * a_in;
            let mut prev = [warm[warm.len() - 2], warm[warm.len() - 1]];
            for f in 0..frames {
                for c in 0..2 {
                    let y = out[f * 2 + c];
                    // The step from the last warm-up frame crosses the input's
                    // own jump between x[n−1] and x[0]; skip that one.
                    if f > 0 {
                        let step = f64::from((y - prev[c]).abs());
                        assert!(
                            step <= bound + 1e-6,
                            "{from:?} -> {to:?}: frame {f} step {step} > {bound}"
                        );
                    }
                    prev[c] = y;
                }
            }
            // After the ramp: identical to a fresh processor with `to`.
            let fresh = processed(to, &x);
            assert_eq!(&out[n * 2..], &fresh[n * 2..], "{from:?} -> {to:?}");
        }
    }
}

#[test]
fn change_during_ramp_restarts_from_reached_matrix() {
    let mut s = Stereo::new();
    s.configure(FS, 2);
    let mut buf = noise(100, 1);
    s.process(&mut buf);
    s.set_params(&with_mode(StereoMode::Swap));
    s.process(&mut buf[..2 * 100]);
    let mid = s.current_matrix();
    assert!(s.is_ramping());
    assert!(mid[0][1] > 0.0 && mid[0][1] < 1.0);
    s.set_params(&with_mode(StereoMode::Mono));
    let mut one = [0.25f32, -0.5];
    s.process(&mut one);
    // One step of 1/N from `mid` toward the mono matrix.
    let t = matrix_for(&with_mode(StereoMode::Mono));
    let n = ramp_frames(FS) as f64;
    let m = s.current_matrix();
    for r in 0..2 {
        for c in 0..2 {
            assert!((m[r][c] - (mid[r][c] + (t[r][c] - mid[r][c]) / n)).abs() < 1e-12);
        }
    }
}

// ---------------------------------------------------------------------------
// Robustness and real-time safety.

#[test]
fn robust_inputs_parameters_and_channel_counts() {
    let mut s = Stereo::new();
    s.configure(FS, 2);
    let mut x = noise(10_000, 77);
    for v in x.iter_mut() {
        *v *= f32::MAX;
    }
    let params = [
        StereoParams {
            balance: f32::NAN,
            width: f32::NAN,
            ..StereoParams::default()
        },
        StereoParams {
            balance: -1.0,
            width: 2.0,
            mode: StereoMode::MsDecode,
            ..StereoParams::default()
        },
        StereoParams {
            balance: 1e9,
            width: -5.0,
            ..with_mode(StereoMode::Mono)
        },
    ];
    for (i, chunk) in x.chunks_mut(2 * 500).enumerate() {
        s.set_params(&params[i % params.len()]);
        s.process(chunk);
        assert!(chunk.iter().all(|v| v.is_finite()), "overflow not clamped");
    }
    // Other channel counts: no-op.
    for ch in [1usize, 3, 8] {
        let mut t = Stereo::new();
        t.set_params(&with_mode(StereoMode::Swap));
        t.configure(FS, ch);
        let mut b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let orig = b.clone();
        t.process(&mut b);
        assert_eq!(b, orig);
    }
    // Odd trailing sample left alone.
    let mut t = Stereo::new();
    t.set_params(&with_mode(StereoMode::Swap));
    t.configure(FS, 2);
    let mut b = [1.0f32, 2.0, 3.0];
    t.process(&mut b);
    assert_eq!(b, [2.0, 1.0, 3.0]);
}

#[test]
fn no_allocation_after_configure() {
    let mut s = Stereo::new();
    s.configure(FS, 2);
    let mut buf = noise(1024, 4);
    let before = alloc_count();
    for i in 0..500 {
        let p = StereoParams {
            mode: MODES[i % MODES.len()],
            mono_law: if i % 2 == 0 {
                MonoLaw::Minus3Db
            } else {
                MonoLaw::Minus6Db
            },
            invert_left: i % 3 == 0,
            invert_right: i % 5 == 0,
            balance: ((i % 21) as f32 - 10.0) / 10.0,
            width: (i % 9) as f32 / 4.0,
            enabled: i % 13 != 0,
        };
        s.set_params(&p);
        let n = [1usize, 64, 480, 1024, 7][i % 5];
        s.process(&mut buf[..n * 2]);
        if i % 77 == 76 {
            s.reset();
        }
    }
    assert_eq!(alloc_count() - before, 0, "process/set_params allocated");
}
