//! Equalizer tests, verified against theory: the analytic magnitude of the RBJ
//! cookbook transfer functions (computed independently in this file from the
//! cookbook equations in their original form), stability, bit-exactness,
//! robustness and real-time safety.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::f64::consts::PI;

use strom_dsp::equalizer::{band_coefficients, band_magnitude, CROSSFADE_MS};
use strom_dsp::{BandParams, BandType, Equalizer, EqualizerParams, Processor, MAX_BANDS};

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

fn allocs() -> usize {
    ALLOCS.with(|c| c.get())
}

// ---------------------------------------------------------------------------
// Independent reference: cookbook coefficients in their literal form, and
// |H(e^jw)| evaluated as a complex ratio.

const ALL_TYPES: [BandType; 10] = [
    BandType::Off,
    BandType::LowPass,
    BandType::HighPass,
    BandType::BandPassSkirt,
    BandType::BandPassPeak,
    BandType::Notch,
    BandType::AllPass,
    BandType::Peaking,
    BandType::LowShelf,
    BandType::HighShelf,
];

/// Returns (b0, b1, b2, a0, a1, a2), not normalised.
fn reference_coefs(ty: BandType, f0: f64, gain_db: f64, q: f64, fs: f64) -> [f64; 6] {
    let w0 = 2.0 * PI * f0 / fs;
    let cw = w0.cos();
    let sw = w0.sin();
    let alpha = sw / (2.0 * q);
    let a = 10f64.powf(gain_db / 40.0);
    let sq = a.sqrt();
    match ty {
        BandType::Off => [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        BandType::LowPass => [
            (1.0 - cw) / 2.0,
            1.0 - cw,
            (1.0 - cw) / 2.0,
            1.0 + alpha,
            -2.0 * cw,
            1.0 - alpha,
        ],
        BandType::HighPass => [
            (1.0 + cw) / 2.0,
            -(1.0 + cw),
            (1.0 + cw) / 2.0,
            1.0 + alpha,
            -2.0 * cw,
            1.0 - alpha,
        ],
        BandType::BandPassSkirt => [
            q * alpha,
            0.0,
            -q * alpha,
            1.0 + alpha,
            -2.0 * cw,
            1.0 - alpha,
        ],
        BandType::BandPassPeak => [alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * cw, 1.0 - alpha],
        BandType::Notch => [1.0, -2.0 * cw, 1.0, 1.0 + alpha, -2.0 * cw, 1.0 - alpha],
        BandType::AllPass => [
            1.0 - alpha,
            -2.0 * cw,
            1.0 + alpha,
            1.0 + alpha,
            -2.0 * cw,
            1.0 - alpha,
        ],
        BandType::Peaking => [
            1.0 + alpha * a,
            -2.0 * cw,
            1.0 - alpha * a,
            1.0 + alpha / a,
            -2.0 * cw,
            1.0 - alpha / a,
        ],
        BandType::LowShelf => [
            a * ((a + 1.0) - (a - 1.0) * cw + 2.0 * sq * alpha),
            2.0 * a * ((a - 1.0) - (a + 1.0) * cw),
            a * ((a + 1.0) - (a - 1.0) * cw - 2.0 * sq * alpha),
            (a + 1.0) + (a - 1.0) * cw + 2.0 * sq * alpha,
            -2.0 * ((a - 1.0) + (a + 1.0) * cw),
            (a + 1.0) + (a - 1.0) * cw - 2.0 * sq * alpha,
        ],
        BandType::HighShelf => [
            a * ((a + 1.0) + (a - 1.0) * cw + 2.0 * sq * alpha),
            -2.0 * a * ((a - 1.0) + (a + 1.0) * cw),
            a * ((a + 1.0) + (a - 1.0) * cw - 2.0 * sq * alpha),
            (a + 1.0) - (a - 1.0) * cw + 2.0 * sq * alpha,
            2.0 * ((a - 1.0) - (a + 1.0) * cw),
            (a + 1.0) - (a - 1.0) * cw - 2.0 * sq * alpha,
        ],
    }
}

fn reference_magnitude(c: [f64; 6], f: f64, fs: f64) -> f64 {
    let w = 2.0 * PI * f / fs;
    let eval = |k0: f64, k1: f64, k2: f64| {
        let re = k0 + k1 * w.cos() + k2 * (2.0 * w).cos();
        let im = -(k1 * w.sin() + k2 * (2.0 * w).sin());
        (re * re + im * im).sqrt()
    };
    eval(c[0], c[1], c[2]) / eval(c[3], c[4], c[5])
}

// ---------------------------------------------------------------------------
// Helpers.

fn band(ty: BandType, f: f32, g: f32, q: f32) -> BandParams {
    BandParams {
        enabled: true,
        band_type: ty,
        frequency: f,
        gain_db: g,
        q,
    }
}

fn params_with(bands: &[BandParams]) -> EqualizerParams {
    let mut p = EqualizerParams {
        num_bands: bands.len().max(1),
        ..Default::default()
    };
    p.bands[..bands.len()].copy_from_slice(bands);
    p
}

fn eq(fs: f32, ch: usize, p: &EqualizerParams) -> Equalizer {
    let mut e = Equalizer::new();
    e.configure(fs, ch);
    e.set_params(p);
    e
}

fn sine(f: f64, fs: f64, amp: f64, start: usize, n: usize) -> Vec<f32> {
    (start..start + n)
        .map(|i| (amp * (2.0 * PI * f * i as f64 / fs).sin()) as f32)
        .collect()
}

/// Amplitude of the component at `f` in `x` (which must hold an integer number
/// of cycles), by projection onto sin/cos.
fn amplitude_at(x: &[f32], f: f64, fs: f64, start: usize) -> f64 {
    let (mut s, mut c) = (0.0, 0.0);
    for (i, &v) in x.iter().enumerate() {
        let ph = 2.0 * PI * f * (start + i) as f64 / fs;
        s += v as f64 * ph.sin();
        c += v as f64 * ph.cos();
    }
    2.0 * (s * s + c * c).sqrt() / x.len() as f64
}

/// Steady-state gain (linear) of the equalizer at integer frequency `f`.
fn measure_gain(p: &EqualizerParams, fs: f64, f: f64, settle: usize) -> f64 {
    let mut e = eq(fs as f32, 1, p);
    let amp = 0.25;
    let mut warm = sine(f, fs, amp, 0, settle);
    e.process(&mut warm);
    // One second window: integer Hz -> integer number of cycles, no leakage.
    let n = fs as usize;
    let mut x = sine(f, fs, amp, settle, n);
    for chunk in x.chunks_mut(512) {
        e.process(chunk);
    }
    amplitude_at(&x, f, fs, settle) / amp
}

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-30).log10()
}

fn noise(n: usize, seed: &mut u64, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            ((*seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * amp
        })
        .collect()
}

fn assert_finite(x: &[f32], ctx: &str) {
    for (i, v) in x.iter().enumerate() {
        assert!(v.is_finite(), "{ctx}: non-finite {v} at {i}");
    }
}

// ---------------------------------------------------------------------------
// Tests.

#[test]
fn from_i32_maps_interface_values() {
    for (i, t) in ALL_TYPES.iter().enumerate() {
        assert_eq!(BandType::from_i32(i as i32), *t);
        assert_eq!(*t as i32, i as i32);
    }
    assert_eq!(BandType::from_i32(-1), BandType::Off);
    assert_eq!(BandType::from_i32(10), BandType::Off);
}

/// The library's analytic magnitude equals the literal cookbook formulas.
#[test]
fn analytic_magnitude_matches_cookbook() {
    for &fs in &[8_000.0, 44_100.0, 48_000.0, 96_000.0, 192_000.0] {
        for &ty in &ALL_TYPES {
            for &(f0, g, q) in &[
                (100.0, 12.0, 0.707),
                (1000.0, -9.0, 2.0),
                (3000.0, 6.0, 0.3),
                (20.0, 60.0, 0.1),
                (1000.0, -60.0, 10.0),
            ] {
                let b = band(ty, f0 as f32, g as f32, q as f32);
                let r = reference_coefs(ty, f0, g, q, fs);
                for &f in &[10.0, 50.0, 300.0, 999.0, 2500.0, 3900.0] {
                    let lib = band_magnitude(&b, fs as f32, f);
                    let refm = reference_magnitude(r, f, fs);
                    if refm > 1e-6 {
                        let err = (db(lib) - db(refm)).abs();
                        assert!(err < 1e-6, "{ty:?} fs {fs} f0 {f0} f {f}: {lib} vs {refm}");
                    }
                }
            }
        }
    }
}

/// Characteristic points of the cookbook responses.
#[test]
fn analytic_characteristic_points() {
    let fs = 48_000.0;
    let m = |ty, f0: f32, g: f32, q: f32, f: f64| db(band_magnitude(&band(ty, f0, g, q), fs, f));
    // Low/high-pass: |H(f0)| = Q.
    assert!((m(BandType::LowPass, 1000.0, 0.0, 2.0, 1000.0) - db(2.0)).abs() < 1e-9);
    assert!((m(BandType::HighPass, 1000.0, 0.0, 2.0, 1000.0) - db(2.0)).abs() < 1e-9);
    // Constant skirt BP peak gain = Q; constant peak BP = 0 dB.
    assert!((m(BandType::BandPassSkirt, 1000.0, 0.0, 4.0, 1000.0) - db(4.0)).abs() < 1e-9);
    assert!(m(BandType::BandPassPeak, 1000.0, 0.0, 4.0, 1000.0).abs() < 1e-9);
    // All-pass: unity everywhere.
    for f in [20.0, 1000.0, 15000.0] {
        assert!(m(BandType::AllPass, 1000.0, 0.0, 4.0, f).abs() < 1e-9);
    }
    // Peaking: gain at f0, unity at DC.
    assert!((m(BandType::Peaking, 1000.0, 9.0, 1.0, 1000.0) - 9.0).abs() < 1e-9);
    // Shelves: half the gain (dB) at f0, full gain far into the shelf.
    assert!((m(BandType::LowShelf, 1000.0, 12.0, 0.707, 1000.0) - 6.0).abs() < 1e-6);
    assert!((m(BandType::HighShelf, 1000.0, 12.0, 0.707, 1000.0) - 6.0).abs() < 1e-6);
    assert!((m(BandType::LowShelf, 1000.0, 12.0, 0.707, 5.0) - 12.0).abs() < 1e-3);
    assert!((m(BandType::HighShelf, 1000.0, 12.0, 0.707, 23_000.0) - 12.0).abs() < 0.05);
    // Gain ignored for non-gain types.
    for ty in [BandType::LowPass, BandType::Notch, BandType::BandPassPeak] {
        assert_eq!(
            m(ty, 1000.0, 30.0, 1.0, 700.0),
            m(ty, 1000.0, -30.0, 1.0, 700.0)
        );
    }
}

/// For each type: measured steady-state sine gain vs the analytic cookbook
/// magnitude, within 0.1 dB where the response is not near a zero.
#[test]
fn measured_gain_matches_analytic_for_every_type() {
    let fs = 48_000.0;
    let freqs = [50.0, 200.0, 700.0, 1000.0, 1400.0, 3000.0, 9000.0, 18000.0];
    let cases = [
        (1000.0, 6.0, 0.707),
        (1000.0, -12.0, 3.0),
        (300.0, 15.0, 0.4),
    ];
    for &ty in &ALL_TYPES {
        for &(f0, g, q) in &cases {
            let p = params_with(&[band(ty, f0, g, q)]);
            let r = reference_coefs(ty, f0 as f64, g as f64, q as f64, fs);
            for &f in &freqs {
                let expect = reference_magnitude(r, f, fs);
                if expect < 0.01 {
                    continue; // near a zero (notch centre, deep stopband)
                }
                let got = measure_gain(&p, fs, f, 24_000);
                let err = (db(got) - db(expect)).abs();
                assert!(
                    err <= 0.1,
                    "{ty:?} f0 {f0} g {g} q {q} at {f} Hz: got {:.4} dB, expected {:.4} dB",
                    db(got),
                    db(expect)
                );
            }
        }
    }
}

/// Several bands in series multiply their responses.
#[test]
fn bands_in_series_multiply() {
    let fs = 44_100.0;
    let bands = [
        band(BandType::HighPass, 40.0, 0.0, 0.707),
        band(BandType::Peaking, 250.0, -6.0, 1.0),
        band(BandType::Peaking, 3000.0, 4.0, 2.0),
        band(BandType::HighShelf, 8000.0, 3.0, 0.707),
        band(BandType::LowShelf, 120.0, 5.0, 0.5),
        band(BandType::LowPass, 16000.0, 0.0, 0.707),
        band(BandType::AllPass, 2000.0, 0.0, 1.0),
        band(BandType::Notch, 60.0, 0.0, 10.0),
    ];
    let mut p = params_with(&bands);
    p.num_bands = MAX_BANDS;
    for &f in &[100.0, 250.0, 1000.0, 3000.0, 10000.0] {
        let expect: f64 = bands
            .iter()
            .map(|b| band_magnitude(b, fs as f32, f))
            .product();
        let got = measure_gain(&p, fs, f, 44_100);
        assert!(
            (db(got) - db(expect)).abs() <= 0.1,
            "{f}: {} vs {}",
            db(got),
            db(expect)
        );
    }
}

/// Run an impulse through, then check the tail decays; also run full-scale
/// noise and check the output is bounded by the analytic worst-case gain.
fn check_stable(b: BandParams, fs: f32, tail_secs: f64) {
    let p = params_with(&[b]);
    let mut e = eq(fs, 1, &p);
    let n = (fs as f64 * tail_secs) as usize;
    let mut x = vec![0.0f32; n];
    x[0] = 1.0;
    e.process(&mut x);
    assert_finite(&x, "impulse");
    // Poles strictly inside the unit circle (stability triangle).
    let c = band_coefficients(&b, fs).unwrap();
    assert!(
        c.a2.abs() < 1.0 && c.a1.abs() < 1.0 + c.a2,
        "{b:?}: unstable {c:?}"
    );
    // The impulse response decays: the last tenth is tiny, or smaller than the
    // tenth before it. (Low-Q low-frequency designs have a real pole very
    // close to z = 1, e.g. 20 Hz / Q 0.1 / −60 dB peaking has a time constant
    // of about 2.5 s, with a small residue; they decay slowly but surely.)
    let seg = |k: usize| {
        x[k * n / 10..(k + 1) * n / 10]
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()))
    };
    let tail = seg(9);
    assert!(
        tail < 1e-6 || tail < seg(8),
        "{b:?} @ {fs}: impulse tail {tail}, previous {}",
        seg(8)
    );

    e.reset();
    let mut seed = 0x1234_5678_9abc_def1u64;
    let mut y = noise(n, &mut seed, 1.0);
    e.process(&mut y);
    assert_finite(&y, "noise");
    let peak = y.iter().fold(0.0f32, |m, v| m.max(v.abs())) as f64;
    // Max magnitude over a dense grid, times a generous crest allowance.
    let max_gain = (1..2000)
        .map(|k| band_magnitude(&b, fs, k as f64 * fs as f64 / 4000.0))
        .fold(1.0f64, f64::max);
    assert!(
        peak < max_gain * 20.0,
        "{b:?}: noise peak {peak}, max gain {max_gain}"
    );
}

#[test]
fn stability_low_frequency_low_q() {
    for &ty in &ALL_TYPES[1..] {
        for g in [-60.0, 60.0] {
            check_stable(band(ty, 20.0, g, 0.1), 48_000.0, 4.0);
        }
    }
}

#[test]
fn stability_high_frequency() {
    for &ty in &ALL_TYPES[1..] {
        for g in [-60.0, 60.0] {
            for q in [0.1, 0.707, 10.0] {
                check_stable(band(ty, 20_000.0, g, q), 48_000.0, 2.0);
            }
        }
    }
}

#[test]
fn type_off_disabled_and_unused_bands_are_bit_exact() {
    let mut seed = 42u64;
    let input = noise(4096 * 2, &mut seed, 0.9);

    // All bands off / disabled / beyond num_bands.
    let mut p = EqualizerParams {
        num_bands: 3,
        ..Default::default()
    };
    p.bands[0] = band(BandType::Off, 1000.0, 12.0, 1.0);
    p.bands[1] = BandParams {
        enabled: false,
        ..band(BandType::Peaking, 1000.0, 12.0, 1.0)
    };
    p.bands[2] = band(BandType::Off, 100.0, -12.0, 4.0);
    for i in 3..MAX_BANDS {
        p.bands[i] = band(BandType::Peaking, 500.0 * i as f32, 12.0, 1.0);
    }
    let mut e = eq(48_000.0, 2, &p);
    let mut x = input.clone();
    e.process(&mut x);
    assert_eq!(x, input);

    // Inactive bands next to an active one don't change its output.
    let mut with_extras = p;
    with_extras.bands[1].band_type = BandType::LowShelf; // still disabled
    with_extras.bands[2] = band(BandType::Peaking, 2000.0, 6.0, 1.0);
    let only = params_with(&[
        band(BandType::Off, 1.0, 0.0, 1.0),
        band(BandType::Off, 1.0, 0.0, 1.0),
        band(BandType::Peaking, 2000.0, 6.0, 1.0),
    ]);
    let mut e1 = eq(48_000.0, 2, &with_extras);
    let mut e2 = eq(48_000.0, 2, &only);
    let mut a = input.clone();
    let mut b = input.clone();
    e1.process(&mut a);
    e2.process(&mut b);
    assert_eq!(a, b);
    assert_ne!(a, input);
}

#[test]
fn band_switched_off_while_running_returns_to_bit_exact() {
    let fs = 48_000.0;
    let mut p = params_with(&[band(BandType::Peaking, 1000.0, 9.0, 1.0)]);
    let mut e = eq(fs, 2, &p);
    let mut seed = 7u64;
    let mut x = noise(4800 * 2, &mut seed, 0.5);
    e.process(&mut x);
    p.bands[0].enabled = false;
    e.set_params(&p);
    // Let the fade-out complete.
    let fade = (CROSSFADE_MS * 0.001 * fs as f64) as usize + 1;
    let mut y = noise(fade * 2, &mut seed, 0.5);
    e.process(&mut y);
    let input = noise(4096 * 2, &mut seed, 0.5);
    let mut z = input.clone();
    e.process(&mut z);
    assert_eq!(z, input);

    // num_bands shrinking deactivates the band as well.
    p.bands[0].enabled = true;
    p.bands[1] = band(BandType::Peaking, 3000.0, 6.0, 1.0);
    p.num_bands = 2;
    e.set_params(&p);
    let mut y = noise(fs as usize * 2, &mut seed, 0.5);
    e.process(&mut y);
    p.num_bands = 1;
    p.bands[0].band_type = BandType::Off;
    e.set_params(&p);
    let mut y = noise(fade * 2, &mut seed, 0.5);
    e.process(&mut y);
    let mut z = input.clone();
    e.process(&mut z);
    assert_eq!(z, input);
}

#[test]
fn reset_clears_state() {
    let p = params_with(&[band(BandType::LowPass, 200.0, 0.0, 5.0)]);
    let mut e = eq(48_000.0, 1, &p);
    let mut x = vec![1.0f32; 1000];
    e.process(&mut x);
    e.reset();
    let mut z = vec![0.0f32; 256];
    e.process(&mut z);
    assert!(z.iter().all(|&v| v == 0.0));
    // After reset the response to an impulse equals a fresh instance.
    let mut f = eq(48_000.0, 1, &p);
    let mut a = vec![0.0f32; 512];
    a[0] = 1.0;
    let mut b = a.clone();
    e.process(&mut a);
    f.process(&mut b);
    assert_eq!(a, b);
}

#[test]
fn channels_are_independent() {
    let p = params_with(&[band(BandType::Peaking, 1000.0, 12.0, 2.0)]);
    let mut e = eq(48_000.0, 3, &p);
    let mut x = vec![0.0f32; 300];
    x[1] = 1.0; // impulse on channel 1 only
    e.process(&mut x);
    for (i, v) in x.iter().enumerate() {
        if i % 3 != 1 {
            assert_eq!(*v, 0.0);
        }
    }
}

/// Every parameter at its extremes, at every rate and several channel counts.
#[test]
fn robustness_extremes() {
    let freqs = [1.0f32, 24_000.0];
    let gains = [-60.0f32, 60.0];
    let qs = [0.01f32, 100.0];
    let mut combos = Vec::new();
    for &ty in &ALL_TYPES {
        for &f in &freqs {
            for &g in &gains {
                for &q in &qs {
                    combos.push(band(ty, f, g, q));
                }
            }
        }
    }
    let mut seed = 99u64;
    for &fs in &[8_000.0f32, 44_100.0, 48_000.0, 96_000.0, 192_000.0] {
        for &ch in &[1usize, 2, 8, 64] {
            let frames = if ch == 64 { 512 } else { 2048 };
            for group in combos.chunks(MAX_BANDS) {
                let mut p = params_with(group);
                p.num_bands = MAX_BANDS;
                let mut e = eq(fs, ch, &p);
                let mut x = noise(frames * ch, &mut seed, 1.0);
                x[0] = 1.0;
                e.process(&mut x);
                assert_finite(&x, &format!("fs {fs} ch {ch} {group:?}"));
                // Change every band to the opposite extreme while running.
                for b in p.bands.iter_mut() {
                    b.frequency = if b.frequency > 1.0 { 1.0 } else { 24_000.0 };
                    b.gain_db = -b.gain_db;
                    b.q = if b.q > 1.0 { 0.01 } else { 100.0 };
                }
                e.set_params(&p);
                let mut y = noise(frames * ch, &mut seed, 1.0);
                e.process(&mut y);
                assert_finite(&y, &format!("fs {fs} ch {ch} changed"));
            }
        }
    }
    // Out-of-range and non-finite parameter values are clamped.
    let mut p = params_with(&[
        band(BandType::Peaking, f32::NAN, f32::INFINITY, -1.0),
        band(BandType::LowShelf, -5.0, 1e9, 0.0),
        band(BandType::LowPass, 1e9, 0.0, f32::NAN),
    ]);
    p.num_bands = 99;
    let mut e = eq(8_000.0, 2, &p);
    let mut x = noise(4096, &mut seed, 1.0);
    e.process(&mut x);
    assert_finite(&x, "out of range");
    // Non-finite input does not poison later output.
    let mut x = vec![f32::NAN, f32::INFINITY, 0.5, 0.5];
    e.process(&mut x);
    let mut x = noise(4096, &mut seed, 1.0);
    e.process(&mut x);
    assert_finite(&x, "after NaN input");
}

/// Eight +60 dB bands at the same frequency cannot overflow to Inf.
#[test]
fn extreme_cumulative_gain_stays_finite() {
    let mut p = params_with(&[band(BandType::Peaking, 1000.0, 60.0, 100.0); MAX_BANDS]);
    p.num_bands = MAX_BANDS;
    let mut e = eq(48_000.0, 1, &p);
    let mut x = sine(1000.0, 48_000.0, 1.0, 0, 96_000);
    e.process(&mut x);
    assert_finite(&x, "cumulative");
}

#[test]
fn unconfigured_is_noop() {
    let mut e = Equalizer::new();
    e.set_params(&params_with(&[band(BandType::Peaking, 1000.0, 12.0, 1.0)]));
    let mut x = vec![0.3f32; 64];
    e.process(&mut x);
    assert!(x.iter().all(|&v| v == 0.3));
}

/// Max |y[n] − y[n−1]| over a slice.
fn max_step(x: &[f32]) -> f64 {
    x.windows(2)
        .map(|w| (w[1] as f64 - w[0] as f64).abs())
        .fold(0.0, f64::max)
}

/// Parameter-change click bound.
///
/// A steady sine of amplitude `a` at `f` passes through the equalizer and the
/// parameters change abruptly while it runs. For a pure sine of amplitude `A`
/// the largest sample-to-sample step is `2·A·sin(π·f/fs)`. The documented
/// bound: during and after the transition, no step exceeds
/// `1.25 × 2·a·sin(π·f/fs) × G + 1e-4`, where `G` is the largest analytic
/// steady-state gain at `f` of the old setting, the new setting and (for a
/// change within one type) every setting on the smoothing path between them. An abrupt coefficient switch without smoothing/crossfade would
/// violate it in several of these scenarios.
#[test]
fn parameter_changes_do_not_click() {
    let fs = 48_000.0f32;
    let f = 1000.0;
    let a = 0.5;
    let scenarios: Vec<(&str, Vec<BandParams>, Vec<BandParams>)> = vec![
        (
            "gain 0 -> +18",
            vec![band(BandType::Peaking, 1000.0, 0.0, 1.0)],
            vec![band(BandType::Peaking, 1000.0, 18.0, 1.0)],
        ),
        (
            "gain +18 -> -18",
            vec![band(BandType::Peaking, 1000.0, 18.0, 1.0)],
            vec![band(BandType::Peaking, 1000.0, -18.0, 1.0)],
        ),
        (
            "gain -60 -> +60",
            vec![band(BandType::Peaking, 1000.0, -60.0, 1.0)],
            vec![band(BandType::Peaking, 1000.0, 60.0, 1.0)],
        ),
        (
            "freq sweep peaking",
            vec![band(BandType::Peaking, 100.0, 12.0, 4.0)],
            vec![band(BandType::Peaking, 8000.0, 12.0, 4.0)],
        ),
        (
            "lowpass corner",
            vec![band(BandType::LowPass, 20_000.0, 0.0, 0.707)],
            vec![band(BandType::LowPass, 300.0, 0.0, 0.707)],
        ),
        (
            "Q change",
            vec![band(BandType::Peaking, 1000.0, 12.0, 0.3)],
            vec![band(BandType::Peaking, 1000.0, 12.0, 20.0)],
        ),
        (
            "shelf gain",
            vec![band(BandType::LowShelf, 2000.0, -12.0, 0.707)],
            vec![band(BandType::LowShelf, 2000.0, 12.0, 0.707)],
        ),
        (
            "type LP -> HP",
            vec![band(BandType::LowPass, 1000.0, 0.0, 0.707)],
            vec![band(BandType::HighPass, 1000.0, 0.0, 0.707)],
        ),
        (
            "type off -> peaking",
            vec![band(BandType::Off, 1000.0, 0.0, 1.0)],
            vec![band(BandType::Peaking, 1000.0, 12.0, 1.0)],
        ),
        (
            "type notch -> off",
            vec![band(BandType::Notch, 1000.0, 0.0, 1.0)],
            vec![band(BandType::Off, 1000.0, 0.0, 1.0)],
        ),
        (
            "type allpass -> bandpass",
            vec![band(BandType::AllPass, 1000.0, 0.0, 1.0)],
            vec![band(BandType::BandPassSkirt, 1000.0, 0.0, 4.0)],
        ),
    ];
    for (name, before, after) in scenarios {
        let gain =
            |bs: &[BandParams]| -> f64 { bs.iter().map(|b| band_magnitude(b, fs, f)).product() };
        // Largest steady-state gain along the parameter path. Same type:
        // frequency and Q move geometrically, gain linearly in dB (as the
        // smoother does). Type change: the two endpoint settings only.
        let mut path_gain = gain(&before).max(gain(&after));
        if before[0].band_type == after[0].band_type {
            for k in 0..=400 {
                let t = k as f32 / 400.0;
                let (b0, b1) = (before[0], after[0]);
                let mid = BandParams {
                    frequency: b0.frequency * (b1.frequency / b0.frequency).powf(t),
                    gain_db: b0.gain_db + (b1.gain_db - b0.gain_db) * t,
                    q: b0.q * (b1.q / b0.q).powf(t),
                    ..b0
                };
                path_gain = path_gain.max(band_magnitude(&mid, fs, f));
            }
        }
        let unit_step = 2.0 * a * (PI * f / fs as f64).sin();
        let bound = 1.25 * unit_step * path_gain + 1e-4;

        let p0 = params_with(&before);
        let p1 = params_with(&after);
        let mut e = eq(fs, 1, &p0);
        let warm = 48_000;
        let mut x = sine(f, fs as f64, a, 0, warm);
        e.process(&mut x);
        let before_step = max_step(&x[warm - 4800..]);
        assert!(
            before_step <= bound,
            "{name}: steady before {before_step} > {bound}"
        );
        e.set_params(&p1);
        // Also include the last pre-change sample in the step check.
        let mut y = sine(f, fs as f64, a, warm, 48_000);
        for chunk in y.chunks_mut(256) {
            e.process(chunk);
        }
        let mut joined = vec![x[warm - 1]];
        joined.extend_from_slice(&y);
        let step = max_step(&joined);
        assert!(step <= bound, "{name}: step {step} > bound {bound}");
        // And it ends at the new steady state.
        let end = amplitude_at(&y[24_000..], f, fs as f64, warm + 24_000) / a;
        let expect = gain(&after);
        if expect > 0.01 {
            assert!(
                (db(end) - db(expect)).abs() < 0.1,
                "{name}: end gain {end} vs {expect}"
            );
        } else {
            assert!(end < 0.02, "{name}: end gain {end} vs {expect}");
        }
    }
}

#[test]
fn no_allocation_in_process_and_set_params() {
    let mut p = params_with(&[
        band(BandType::Peaking, 1000.0, 6.0, 1.0),
        band(BandType::LowShelf, 100.0, -3.0, 0.707),
        band(BandType::HighPass, 30.0, 0.0, 0.707),
        band(BandType::Notch, 50.0, 0.0, 10.0),
    ]);
    let mut e = Equalizer::new();
    e.configure(48_000.0, 8);
    let mut seed = 5u64;
    let mut buf = noise(1024 * 8, &mut seed, 0.5);
    let mut buf2 = buf.clone();

    let before = allocs();
    e.set_params(&p);
    e.process(&mut buf);
    for i in 0..50 {
        // Parameter changes of every kind, including type changes and toggles.
        p.bands[0].gain_db = (i as f32 * 1.7) % 24.0 - 12.0;
        p.bands[1].frequency = 50.0 + i as f32 * 37.0;
        p.bands[2].band_type = BandType::from_i32(i % 10);
        p.bands[3].enabled = i % 3 != 0;
        p.num_bands = 1 + (i as usize % MAX_BANDS);
        e.set_params(&p);
        e.process(&mut buf2);
        e.process(&mut buf);
    }
    e.reset();
    e.process(&mut buf);
    let after = allocs();
    assert_eq!(after - before, 0, "allocations in set_params/process");
}

/// Every channel of a multichannel equalizer behaves exactly like a mono
/// equalizer fed that channel alone, for every lane-group layout (pairs plus
/// an odd channel), in the steady state and while parameters change.
#[test]
fn multichannel_equals_mono_per_channel() {
    let fs = 48_000.0;
    let p0 = params_with(&[
        band(BandType::HighPass, 80.0, 0.0, 0.707),
        band(BandType::Peaking, 300.0, -3.0, 1.2),
        band(BandType::Peaking, 3000.0, 2.5, 0.9),
        band(BandType::HighShelf, 10_000.0, 2.0, 0.707),
    ]);
    let mut p1 = p0;
    p1.bands[1].gain_db = 9.0; // ramp
    p1.bands[2].band_type = BandType::LowShelf; // crossfade
    p1.bands[3].enabled = false; // fade out
    let frames = 4000;
    for ch in [1usize, 2, 3, 5, 8] {
        let mut seed = 77 + ch as u64;
        let input = noise(frames * ch, &mut seed, 0.8);
        let mut multi = input.clone();
        let mut e = eq(fs, ch, &p0);
        let (a, b) = multi.split_at_mut(1000 * ch);
        e.process(a);
        e.set_params(&p1);
        for chunk in b.chunks_mut(256 * ch) {
            e.process(chunk);
        }
        for k in 0..ch {
            let mut mono: Vec<f32> = input.iter().skip(k).step_by(ch).copied().collect();
            let mut m = eq(fs, 1, &p0);
            let (a, b) = mono.split_at_mut(1000);
            m.process(a);
            m.set_params(&p1);
            for chunk in b.chunks_mut(256) {
                m.process(chunk);
            }
            let got: Vec<f32> = multi.iter().skip(k).step_by(ch).copied().collect();
            assert_eq!(got, mono, "ch {ch}, channel {k}");
        }
    }
}

/// `set_params` before or after `configure` gives the same output.
#[test]
fn set_params_before_or_after_configure_is_identical() {
    let p = params_with(&[
        band(BandType::LowShelf, 120.0, 4.0, 0.707),
        band(BandType::Peaking, 2500.0, -6.0, 2.0),
    ]);
    let mut a = Equalizer::new();
    a.set_params(&p);
    a.configure(44_100.0, 2);
    let mut b = Equalizer::new();
    b.configure(44_100.0, 2);
    b.set_params(&p);
    let mut seed = 3u64;
    let input = noise(2048 * 2, &mut seed, 0.7);
    let (mut x, mut y) = (input.clone(), input);
    a.process(&mut x);
    b.process(&mut y);
    assert_eq!(x, y);
}
