//! `BandCorrelation` tests: identical L/R reads +1 and inverted L/R −1 in
//! every band, a signal correlated only in one band reads accordingly,
//! independent noise reads about 0, and nothing allocates after `new`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::f64::consts::PI;

use strom_dsp::meter::{BandCorrelation, CORRELATION_BANDS, DEFAULT_BAND_SPLITS_HZ};

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

const FS: f32 = 48_000.0;
const BLOCK: usize = 480;

/// Xorshift32 white noise in −1 … 1 (fixed seed).
struct Noise(u32);

impl Noise {
    fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }
}

fn sine(n: usize, f: f64, phase: f64) -> f32 {
    (2.0 * PI * f * n as f64 / FS as f64 + phase).sin() as f32
}

/// Feed interleaved stereo in `BLOCK`-frame blocks.
fn feed(m: &mut BandCorrelation, stereo: &[f32]) {
    for b in stereo.chunks(2 * BLOCK) {
        m.process_interleaved(b, 2);
    }
}

fn assert_bands(what: &str, got: [f64; CORRELATION_BANDS], want: [f64; 3], tol: f64) {
    for k in 0..CORRELATION_BANDS {
        assert!(
            (got[k] - want[k]).abs() <= tol,
            "{what}: band {k} reads {:.5}, want {} ± {tol} (all: {got:?})",
            got[k],
            want[k]
        );
    }
}

#[test]
fn defaults() {
    let m = BandCorrelation::new(FS, 300.0);
    let s = m.split_frequencies_hz();
    assert!((s[0] - DEFAULT_BAND_SPLITS_HZ[0] as f64).abs() < 1e-3);
    assert!((s[1] - DEFAULT_BAND_SPLITS_HZ[1] as f64).abs() < 1e-3);
    assert_eq!(m.integration_ms(), 300.0);
    // Silence reads 0 everywhere.
    assert_eq!(m.correlations(), [0.0; 3]);
    assert_eq!(m.correlation(7), 0.0);
}

#[test]
fn identical_is_plus_one_inverted_is_minus_one_in_every_band() {
    for rate in [44_100.0f32, 48_000.0, 96_000.0] {
        for inverted in [false, true] {
            let mut noise = Noise(0x9e37_79b9);
            let mut m = BandCorrelation::new(rate, 300.0);
            let stereo: Vec<f32> = (0..rate as usize)
                .flat_map(|_| {
                    let x = 0.5 * noise.next();
                    [x, if inverted { -x } else { x }]
                })
                .collect();
            feed(&mut m, &stereo);
            let want = if inverted { -1.0 } else { 1.0 };
            assert_bands(
                &format!("{rate} Hz inverted={inverted}"),
                m.correlations(),
                [want; 3],
                1e-6,
            );
        }
    }
}

/// Low band: identical 60 Hz sine (+1). Mid band: 1 kHz in quadrature
/// (sin/cos, uncorrelated: 0). High band: 8 kHz with one channel inverted
/// (−1). Each tone is ≥ 2 octaves from a split, so the LR4 leakage into the
/// neighbouring bands is below −45 dB.
#[test]
fn correlated_only_in_the_low_band() {
    let mut m = BandCorrelation::new(FS, 300.0);
    let stereo: Vec<f32> = (0..2 * FS as usize)
        .flat_map(|n| {
            let low = 0.4 * sine(n, 60.0, 0.0);
            let mid_l = 0.3 * sine(n, 1000.0, 0.0);
            let mid_r = 0.3 * sine(n, 1000.0, PI / 2.0);
            let high = 0.2 * sine(n, 8000.0, 0.0);
            [low + mid_l + high, low + mid_r - high]
        })
        .collect();
    feed(&mut m, &stereo);
    assert_bands(
        "low/quadrature/inverted",
        m.correlations(),
        [1.0, 0.0, -1.0],
        0.02,
    );

    // A common 50 Hz tone plus independent white noise in L and R: only the
    // low band is dominated by common signal (white noise puts about 0.5 %
    // of its power below 250 Hz).
    let mut m = BandCorrelation::new(FS, 300.0);
    let (mut nl, mut nr) = (Noise(1), Noise(0xdead_beef));
    let stereo: Vec<f32> = (0..2 * FS as usize)
        .flat_map(|n| {
            let low = 0.5 * sine(n, 50.0, 0.0);
            let (a, b) = (nl.next(), nr.next());
            [low + 0.2 * a, low + 0.2 * b]
        })
        .collect();
    feed(&mut m, &stereo);
    let c = m.correlations();
    assert!(c[0] > 0.95, "low band {c:?}");
    assert!(c[1].abs() < 0.15 && c[2].abs() < 0.1, "mid/high {c:?}");
}

#[test]
fn independent_noise_is_about_zero_in_every_band() {
    let mut m = BandCorrelation::new(FS, 300.0);
    let (mut a, mut b) = (Noise(12345), Noise(0x0bad_cafe));
    let stereo: Vec<f32> = (0..2 * FS as usize)
        .flat_map(|_| [0.5 * a.next(), 0.5 * b.next()])
        .collect();
    feed(&mut m, &stereo);
    // The low band (250 Hz of bandwidth, 300 ms) has the largest estimator
    // spread, about 1/sqrt(B·T) ≈ 0.12.
    assert_bands("independent", m.correlations(), [0.0; 3], 0.3);
    let c = m.correlations();
    assert!(c[1].abs() < 0.15 && c[2].abs() < 0.1, "{c:?}");
}

#[test]
fn planar_and_multichannel_input_match_interleaved_stereo() {
    let mut noise = Noise(77);
    let frames = 10_000;
    let l: Vec<f32> = (0..frames).map(|_| noise.next()).collect();
    let r: Vec<f32> = (0..frames)
        .map(|n| 0.5 * l[n] + 0.5 * noise.next())
        .collect();
    let stereo: Vec<f32> = l.iter().zip(&r).flat_map(|(&a, &b)| [a, b]).collect();
    let quad: Vec<f32> = l
        .iter()
        .zip(&r)
        .flat_map(|(&a, &b)| [a, b, 9.0, -9.0])
        .collect();

    let mut a = BandCorrelation::new(FS, 100.0);
    feed(&mut a, &stereo);
    let mut b = BandCorrelation::new(FS, 100.0);
    for (cl, cr) in l.chunks(333).zip(r.chunks(333)) {
        b.process_pair(cl, cr);
    }
    let mut c = BandCorrelation::new(FS, 100.0);
    for q in quad.chunks(4 * 1000) {
        c.process_interleaved(q, 4);
    }
    assert_eq!(a.correlations(), b.correlations());
    assert_eq!(a.correlations(), c.correlations());
    // r = 0.5·l + independent: correlation ≈ 0.5/√0.5 ≈ 0.71 in every band.
    assert_bands("partial", a.correlations(), [0.707; 3], 0.2);

    // Fewer than two channels: ignored.
    let before = a.correlations();
    a.process_interleaved(&l, 1);
    assert_eq!(a.correlations(), before);
}

#[test]
fn reset_and_parameter_changes() {
    let mut m = BandCorrelation::new(FS, 300.0);
    let stereo: Vec<f32> = (0..FS as usize)
        .flat_map(|n| {
            let x = sine(n, 1000.0, 0.0);
            [x, -x]
        })
        .collect();
    feed(&mut m, &stereo);
    assert!(m.correlation(1) < -0.99);
    m.reset();
    assert_eq!(m.correlations(), [0.0; 3]);

    m.set_split_frequencies([100.0, 30_000.0]);
    let s = m.split_frequencies_hz();
    assert!((s[0] - 100.0).abs() < 1e-3);
    assert!(
        (s[1] - 0.499 * FS as f64).abs() < 1e-3,
        "clamped to Nyquist: {s:?}"
    );
    m.set_integration_ms(50.0);
    assert_eq!(m.integration_ms(), 50.0);
    feed(&mut m, &stereo);
    // 1 kHz is now in the mid band; the high band is silent (reads 0).
    let c = m.correlations();
    assert!(c[1] < -0.99, "{c:?}");
    assert_eq!(c[2], 0.0, "{c:?}");
}

/// Zero allocations after `new`, for every entry point and block size.
#[test]
fn never_allocates_after_new() {
    let mut noise = Noise(5);
    let stereo: Vec<f32> = (0..20_000).map(|_| noise.next()).collect();
    let (l, r): (Vec<f32>, Vec<f32>) = stereo
        .as_chunks::<2>()
        .0
        .iter()
        .map(|f| (f[0], f[1]))
        .unzip();
    for rate in [8_000.0f32, 48_000.0, 192_000.0] {
        let mut m = BandCorrelation::new(rate, 300.0);
        let before = alloc_count();
        let mut sink = 0.0;
        for (i, size) in [1usize, 7, 255, 256, 257, 480, 1024, 4096]
            .into_iter()
            .enumerate()
        {
            m.process_interleaved(&stereo[..2 * size], 2);
            m.process_pair(&l[..size], &r[..size]);
            m.set_integration_ms(10.0 + i as f32 * 50.0);
            m.set_split_frequencies([150.0 + i as f32 * 10.0, 3000.0]);
            sink += m.correlations().iter().sum::<f64>() + m.correlation(1);
            sink += m.split_frequencies_hz()[0];
            if i == 4 {
                m.reset();
            }
        }
        assert_eq!(alloc_count() - before, 0, "{rate} Hz");
        assert!(sink.is_finite());
    }
}
