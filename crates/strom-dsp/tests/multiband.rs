//! Multiband compressor and the crossover requirements it
//! relies on, verified against theory:
//!
//! - the analytic magnitude of the bilinear-transformed (pre-warped) LR4
//!   bands: with `r_s(f) = tan(πf/fs)/tan(πf_s/fs)` for split `s`,
//!   `|LP_s|² = 1/(1 + r_s⁴)` and `|HP_s|² = r_s⁴/(1 + r_s⁴)`. Because each
//!   LR4 band is the overall all-pass times a real, non-negative factor
//!   (`L²/(L² + H²) = 1/(1 + s⁴)` on the jω axis), band `k` at frequency `f`
//!   is the all-pass times `m_k(f) = Π_{s<k} |HP_s|² · |LP_k|²` (no `LP` for
//!   the top band), and `Σ_k m_k = 1`. A steady sine of amplitude `A` thus
//!   reaches band `k` with amplitude `A·m_k`, and leaves the chain with
//!   amplitude `A·Σ_k g_k·m_k` when band `k` has the steady gain `g_k`;
//! - the gain computer of Giannoulis, Massberg & Reiss (JAES 2012), written
//!   out here independently: `r = 0` below `T − W/2`,
//!   `S·(x − T + W/2)²/(2W)` in the knee, `S·(x − T)` above, `S = 1 − 1/R`.
//!
//! Synthetic signals only (sines, xorshift noise with fixed seeds). Sine
//! amplitudes are measured by a DFT projection over exactly one second of
//! integer-Hz tones, so different tones are exactly orthogonal.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::f64::consts::PI;

use strom_dsp::compressor::CompressorDetector;
use strom_dsp::filter::{Crossover, CrossoverParams, LrOrder, MAX_CROSSOVER_BANDS, MAX_SPLITS};
use strom_dsp::multiband::{
    MultibandBandParams, MultibandCompressor, MultibandParams, DEFAULT_FREQUENCIES_HZ,
};
use strom_dsp::{Compressor, Processor};

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
// Helpers.

const FS: f32 = 48_000.0;

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-300).log10()
}

fn lin(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

/// `r_s(f)` of the pre-warped bilinear transform.
fn warp_ratio(f: f64, fc: f64, fs: f64) -> f64 {
    (PI * f / fs).tan() / (PI * fc / fs).tan()
}

/// Analytic magnitude factor `m_k(f)` of LR4 band `k` of `nb`.
fn band_factor(k: usize, nb: usize, splits: &[f32], f: f64, fs: f64) -> f64 {
    let mut m = 1.0;
    for &s in &splits[..k] {
        let r4 = warp_ratio(f, s as f64, fs).powi(4);
        m *= r4 / (1.0 + r4);
    }
    if k + 1 < nb {
        let r4 = warp_ratio(f, splits[k] as f64, fs).powi(4);
        m *= 1.0 / (1.0 + r4);
    }
    m
}

/// Static gain reduction (dB) of the gain computer, written from the paper.
fn reduction_formula(x: f64, t: f64, w: f64, ratio: f64) -> f64 {
    let s = 1.0 - 1.0 / ratio;
    let over = x - t;
    if 2.0 * over <= -w {
        0.0
    } else if 2.0 * over < w {
        s * (over + w / 2.0).powi(2) / (2.0 * w)
    } else {
        s * over
    }
}

/// Response `Σ h[n]·e^{−jωn}` of a mono response at `f` (Hz).
fn dft_at(h: &[f32], f: f64, fs: f64) -> (f64, f64) {
    // e^{−jωn} by complex multiplication with e^{−jω}, re-seeded from
    // sin_cos every 256 samples so rounding cannot accumulate.
    let w = 2.0 * PI * f / fs;
    let (ws, wc) = w.sin_cos();
    let (mut re, mut im) = (0.0, 0.0);
    let (mut c, mut s) = (1.0f64, 0.0f64);
    for (n, &x) in h.iter().enumerate() {
        if n % 256 == 0 {
            (s, c) = (w * n as f64).sin_cos();
        }
        re += x as f64 * c;
        im -= x as f64 * s;
        // (c + js)·(wc + j·ws): advance the angle by ω.
        (c, s) = (c * wc - s * ws, s * wc + c * ws);
    }
    (re, im)
}

fn mag(z: (f64, f64)) -> f64 {
    (z.0 * z.0 + z.1 * z.1).sqrt()
}

/// Amplitude of the integer-Hz tone `f` in channel `c` of interleaved `x`,
/// over exactly one second starting at frame `start`.
fn tone_amp(x: &[f32], ch: usize, c: usize, f: f64, start: usize) -> f64 {
    let n = FS as usize;
    let mono: Vec<f32> = x[start * ch..(start + n) * ch]
        .iter()
        .skip(c)
        .step_by(ch)
        .copied()
        .collect();
    // Phase reference at `start`: the projection magnitude does not care.
    2.0 * mag(dft_at(&mono, f, FS as f64)) / n as f64
}

/// Interleaved signal of `frames` frames from `gen(frame, channel)`.
fn signal(frames: usize, ch: usize, gen: impl Fn(usize, usize) -> f64) -> Vec<f32> {
    let mut v = Vec::with_capacity(frames * ch);
    for i in 0..frames {
        for c in 0..ch {
            v.push(gen(i, c) as f32);
        }
    }
    v
}

fn tone(f: f64, amp: f64, i: usize) -> f64 {
    amp * (2.0 * PI * f * i as f64 / FS as f64).sin()
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

/// Band params with no compression (ratio 1, makeup 0).
fn flat_band() -> MultibandBandParams {
    MultibandBandParams {
        threshold_db: -30.0,
        ratio: 1.0,
        attack_ms: 5.0,
        release_ms: 50.0,
        makeup_db: 0.0,
        knee_db: 0.0,
    }
}

fn flat_params(nb: usize) -> MultibandParams {
    MultibandParams {
        num_bands: nb,
        frequencies_hz: DEFAULT_FREQUENCIES_HZ,
        drive_db: 0.0,
        detector: CompressorDetector::Peak,
        bands: [flat_band(); MAX_CROSSOVER_BANDS],
    }
}

fn multiband(fs: f32, ch: usize, p: &MultibandParams) -> MultibandCompressor {
    let mut m = MultibandCompressor::new();
    m.set_params(p);
    m.configure(fs, ch);
    m
}

/// Integer-Hz test frequency in the middle of band `k` (geometric centre of
/// its edges; 50 Hz for the lowest band, `√(f_top·20 kHz)` for the top band).
fn centre(k: usize, nb: usize, splits: &[f32]) -> f64 {
    let f = if nb == 1 {
        1000.0
    } else if k == 0 {
        splits[0] as f64 / 2.4
    } else if k + 1 == nb {
        (splits[k - 1] as f64 * 20_000.0).sqrt()
    } else {
        (splits[k - 1] as f64 * splits[k] as f64).sqrt()
    };
    // Integer Hz, coprime to the 48 kHz rate, so the sampled waveform hits
    // many phases and the peak detector reads the true peak (a 4000 Hz tone
    // has 12 samples per period and reads up to 0.06 dB low).
    let mut f = f.round() as u64;
    while gcd(f, FS as u64) != 1 {
        f += 1;
    }
    f as f64
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn log_freqs(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo * (hi / lo).powf(i as f64 / (n - 1) as f64))
        .collect()
}

// ---------------------------------------------------------------------------
// Crossover requirements, on `filter::Crossover`.

fn lr4(fs: f32, nb: usize, splits: [f32; MAX_SPLITS]) -> Crossover {
    let mut x = Crossover::new();
    x.configure(fs, 1);
    x.set_params(&CrossoverParams {
        order: LrOrder::Lr4,
        num_bands: nb,
        frequencies_hz: splits,
    });
    x
}

fn band_impulses(x: &mut Crossover, n: usize) -> Vec<Vec<f32>> {
    let mut input = vec![0.0f32; n];
    input[0] = 1.0;
    let mut bands = vec![vec![0.0f32; n]; MAX_CROSSOVER_BANDS];
    x.process(&input, &mut bands);
    bands.truncate(x.num_bands());
    bands
}

const SPLIT_SETS: [[f32; MAX_SPLITS]; 2] = [DEFAULT_FREQUENCIES_HZ, [60.0, 250.0, 1000.0, 5000.0]];

/// 1–5 bands, LR4, the plain band sum is flat within ±0.05 dB from 20 Hz to
/// 20 kHz (it is an all-pass, so the error is rounding only: < 0.001 dB).
#[test]
fn crossover_lr4_band_sum_is_flat() {
    for fs in [44_100.0f32, 48_000.0, 96_000.0] {
        for set in SPLIT_SETS {
            for nb in 1..=MAX_CROSSOVER_BANDS {
                let mut x = lr4(fs, nb, set);
                let b = band_impulses(&mut x, fs as usize / 4);
                let mut sum = vec![0.0f32; b[0].len()];
                for band in &b {
                    for (s, &v) in sum.iter_mut().zip(band) {
                        *s += v;
                    }
                }
                let worst = log_freqs(20.0, 20_000.0, 80)
                    .into_iter()
                    .map(|f| db(mag(dft_at(&sum, f, fs as f64))).abs())
                    .fold(0.0, f64::max);
                eprintln!("crossover sum: fs={fs} bands={nb} {set:?}: max |dev| {worst:.2e} dB");
                assert!(worst < 0.05, "fs={fs} bands={nb} {set:?}: {worst} dB");
                assert!(worst < 1e-3, "fs={fs} bands={nb} {set:?}: {worst} dB");
            }
        }
    }
}

/// Each LR4 split is at −6.02 dB (`20·log10(1/2)`) at its frequency: with two
/// bands exactly (±0.01 dB) for every default crossover frequency. With 3–5
/// bands every band matches the analytic product `m_k(f)` within 0.01 dB
/// over 20 Hz–20 kHz, so at an edge it is −6.02 dB plus the (analytic)
/// attenuation of the neighbouring split.
#[test]
fn crossover_lr4_bands_are_minus_6_db_at_crossover() {
    let fs = FS as f64;
    let edge = db(0.5);
    for fc in DEFAULT_FREQUENCIES_HZ {
        let mut x = lr4(FS, 2, [fc, 0.0, 0.0, 0.0]);
        let b = band_impulses(&mut x, FS as usize / 4);
        for (k, band) in b.iter().enumerate() {
            let g = db(mag(dft_at(band, fc as f64, fs)));
            eprintln!("2 bands, split {fc} Hz: band {k} at {fc} Hz = {g:.4} dB");
            assert!((g - edge).abs() < 0.01, "split {fc}: band {k} {g} dB");
        }
    }
    for set in SPLIT_SETS {
        for nb in 3..=MAX_CROSSOVER_BANDS {
            let mut x = lr4(FS, nb, set);
            let b = band_impulses(&mut x, FS as usize / 4);
            for (k, band) in b.iter().enumerate() {
                for f in log_freqs(20.0, 20_000.0, 40) {
                    let want = band_factor(k, nb, &set, f, fs);
                    if want > 1e-4 {
                        let got = mag(dft_at(band, f, fs));
                        assert!(
                            (db(got) - db(want)).abs() < 0.01,
                            "{set:?} {nb} bands, band {k} at {f}: {} vs {}",
                            db(got),
                            db(want)
                        );
                    }
                }
                // Edges: the split's own factor is exactly 1/2 (−6.02 dB).
                for s in [k.wrapping_sub(1), k] {
                    if s < nb - 1 {
                        let f = set[s] as f64;
                        let got = db(mag(dft_at(band, f, fs)));
                        let want = db(band_factor(k, nb, &set, f, fs));
                        eprintln!(
                            "{nb} bands {set:?}: band {k} at {f} Hz = {got:.4} dB \
                             (analytic {want:.4}, of which −6.0206 from the split)"
                        );
                        assert!((got - want).abs() < 0.01);
                    }
                }
            }
        }
    }
}

/// Zero latency: the crossover and the multiband compressor report no
/// latency, the band sum's impulse response starts at sample 0, and with
/// one band the chain is the identity sample for sample.
#[test]
fn crossover_and_multiband_have_zero_latency() {
    for nb in 1..=MAX_CROSSOVER_BANDS {
        let mut x = lr4(FS, nb, DEFAULT_FREQUENCIES_HZ);
        let b = band_impulses(&mut x, 64);
        let h0: f32 = b.iter().map(|band| band[0]).sum();
        assert!(h0.abs() > 0.01, "bands={nb}: h[0] = {h0}");
        let m = multiband(FS, 1, &flat_params(nb));
        assert_eq!(m.latency_frames(), 0);
    }
    let mut seed = 5u64;
    let input = noise(4000, &mut seed, 0.5);
    let mut m = multiband(FS, 2, &flat_params(1));
    let mut out = input.clone();
    m.process(&mut out);
    assert_eq!(out, input);
}

// ---------------------------------------------------------------------------
// Multiband compressor.

/// At ratio 1 and makeup 0 the whole chain is the crossover's all-pass: its
/// impulse response equals the crossover band sum bit for bit, and its
/// magnitude is flat within ±0.05 dB from 20 Hz to 20 kHz (2–5 bands, mono
/// and stereo, three rates).
#[test]
fn ratio_one_chain_is_flat() {
    for fs in [44_100.0f32, 48_000.0, 96_000.0] {
        for nb in 1..=MAX_CROSSOVER_BANDS {
            for ch in [1usize, 2] {
                let n = fs as usize / 4;
                let mut m = multiband(fs, ch, &flat_params(nb));
                let mut h = vec![0.0f32; n * ch];
                h[0] = 1.0; // impulse on channel 0
                m.process(&mut h);
                let h0: Vec<f32> = h.iter().step_by(ch).copied().collect();
                // Reference: the crossover's band sum, summed in the same order.
                let mut x = Crossover::new();
                x.configure(fs, 1);
                x.set_params(&CrossoverParams {
                    order: LrOrder::Lr4,
                    num_bands: nb,
                    frequencies_hz: DEFAULT_FREQUENCIES_HZ,
                });
                let b = band_impulses(&mut x, n);
                let mut sum = b[0].clone();
                for band in &b[1..] {
                    for (s, &v) in sum.iter_mut().zip(band) {
                        *s += v;
                    }
                }
                assert_eq!(h0, sum, "fs={fs} bands={nb} ch={ch}");
                let worst = log_freqs(20.0, 20_000.0, 60)
                    .into_iter()
                    .map(|f| db(mag(dft_at(&h0, f, fs as f64))).abs())
                    .fold(0.0, f64::max);
                assert!(worst < 0.05, "fs={fs} bands={nb}: {worst} dB");
                if ch == 2 {
                    assert!(h.iter().skip(1).step_by(2).all(|&v| v == 0.0));
                }
            }
        }
    }
}

/// Per band, once settled, the gain reduction equals the gain-computer
/// formula at the band's level, and the output tone has the predicted
/// amplitude `A·Σ_j g_j·m_j(f)`.
///
/// For 2–5 bands and every band `k`: a stereo sine at the band's centre
/// (`A` such that the band level is 10 dB above threshold), band `k`
/// compressing (threshold −30 dB, ratio 4, makeup +3 dB, hard knee and a
/// 12 dB soft knee with the level inside the knee), all other bands at
/// ratio 1. Tolerances: 0.005 dB on the gain reduction (the peak detector
/// reads the sampled, not the true, peak: measured errors are at most
/// 1.2e-4 dB, at 4001 Hz with 12 samples per period) and 0.05 dB on the
/// output amplitude (measured at most 0.013 dB: at 4001 Hz the sampled peak
/// beats at 1 Hz against the 12-sample grid, a slight gain modulation).
#[test]
fn static_curve_per_band_matches_gain_computer() {
    let fs = FS as f64;
    let splits = DEFAULT_FREQUENCIES_HZ;
    for nb in 2..=MAX_CROSSOVER_BANDS {
        for k in 0..nb {
            for (knee, over) in [(0.0f32, 10.0f64), (12.0, 3.0)] {
                let f = centre(k, nb, &splits);
                let mk = band_factor(k, nb, &splits, f, fs);
                let (t, ratio, makeup) = (-30.0f64, 4.0f64, 3.0f64);
                let level = t + over;
                let amp = lin(level) / mk;
                let mut p = flat_params(nb);
                p.bands[k] = MultibandBandParams {
                    threshold_db: t as f32,
                    ratio: ratio as f32,
                    attack_ms: 5.0,
                    release_ms: 50.0,
                    makeup_db: makeup as f32,
                    knee_db: knee,
                };
                let mut m = multiband(FS, 2, &p);
                let frames = 2 * FS as usize;
                let mut x = signal(frames, 2, |i, _| tone(f, amp, i));
                m.process(&mut x);
                let gr = m.band_gain_reduction_db(k);
                let want = reduction_formula(level, t, knee as f64, ratio);
                eprintln!(
                    "static: {nb} bands, band {k} at {f} Hz, knee {knee}: GR {gr:.4} dB, \
                     error {:.1e} dB",
                    gr - want
                );
                assert!(
                    (gr - want).abs() < 0.005,
                    "{nb} bands, band {k} at {f} Hz, knee {knee}: GR {gr} vs {want}"
                );
                for j in (0..nb).filter(|&j| j != k) {
                    assert_eq!(m.band_gain_reduction_db(j), 0.0);
                }
                let gk = lin(makeup - want);
                let total: f64 = (0..nb)
                    .map(|j| {
                        let g = if j == k { gk } else { 1.0 };
                        g * band_factor(j, nb, &splits, f, fs)
                    })
                    .sum();
                for c in 0..2 {
                    let got = tone_amp(&x, 2, c, f, FS as usize);
                    let want_amp = amp * total;
                    eprintln!("static: output error {:.1e} dB", db(got) - db(want_amp));
                    assert!(
                        (db(got) - db(want_amp)).abs() < 0.05,
                        "{nb} bands, band {k}, ch {c}: {} vs {} dB",
                        db(got),
                        db(want_amp)
                    );
                }
            }
        }
    }
}

/// One band is a plain wideband compressor: bit-identical to the plain
/// `Compressor` with the converted parameters.
#[test]
fn one_band_is_wideband_compressor() {
    let band = MultibandBandParams {
        threshold_db: -24.0,
        ratio: 6.0,
        attack_ms: 2.0,
        release_ms: 80.0,
        makeup_db: 4.0,
        knee_db: 6.0,
    };
    let mut p = flat_params(1);
    p.bands[0] = band;
    let mut m = multiband(FS, 2, &p);
    let mut c = Compressor::new();
    c.set_params(&band.compressor_params());
    c.configure(FS, 2);
    let mut seed = 11u64;
    let input = noise(2 * 20_000, &mut seed, 0.8);
    let (mut a, mut b) = (input.clone(), input);
    for (ca, cb) in a.chunks_mut(960).zip(b.chunks_mut(960)) {
        m.process(ca);
        c.process(cb);
    }
    assert_eq!(a, b);
    assert_eq!(m.band_gain_reduction_db(0), c.gain_reduction_db());
}

/// Bands are independent: a loud low tone compressing band 0 does not change
/// band 2's gain or the level of a tone in band 2, and band 2's reduction
/// stays at the formula whether band 0 compresses or not.
///
/// Bands 0 and 2 compress (threshold −30 dB, 10:1); bands 1 and 3 are at
/// ratio 1. (If band 1 compressed too, the LR4 skirts of both tones that
/// reach it, about −30 dB and −14 dB, would legitimately make band 1 react
/// to the low tone and change its share `m_1` of the mid tone: that is
/// crossover overlap, not coupling between compressors.) The mid tone's
/// output amplitude is `A·Σ g_j·m_j`; band 0's share `m_0(1789 Hz)` is
/// about −94 dB, so compressing band 0 by > 20 dB must leave it within
/// 0.01 dB.
#[test]
fn bands_are_independent() {
    let fs = FS as f64;
    let splits = DEFAULT_FREQUENCIES_HZ;
    let nb = 4;
    let (f_lo, f_mid) = (50.0, 1789.0);
    let mut p = flat_params(nb);
    for k in [0, 2] {
        p.bands[k] = MultibandBandParams {
            threshold_db: -30.0,
            ratio: 10.0,
            attack_ms: 5.0,
            release_ms: 50.0,
            makeup_db: 0.0,
            knee_db: 0.0,
        };
    }
    for mid_amp in [0.01f64, 0.2] {
        let run = |lo_amp: f64| {
            let mut m = multiband(FS, 2, &p);
            let mut x = signal(2 * FS as usize, 2, |i, _| {
                tone(f_lo, lo_amp, i) + tone(f_mid, mid_amp, i)
            });
            m.process(&mut x);
            (
                m.gain_reduction_db(),
                tone_amp(&x, 2, 0, f_mid, FS as usize),
            )
        };
        let (gr_quiet, mid_quiet) = run(0.0);
        let (gr_loud, mid_loud) = run(0.9);
        assert!(gr_loud[0] > 20.0, "band 0 should compress: {}", gr_loud[0]);
        assert_eq!(gr_quiet[0], 0.0);
        let level = db(mid_amp * band_factor(2, nb, &splits, f_mid, fs));
        let want = reduction_formula(level, -30.0, 0.0, 10.0);
        for gr in [gr_quiet[2], gr_loud[2]] {
            assert!((gr - want).abs() < 0.02, "band 2 GR {gr} vs {want}");
        }
        assert!(
            (db(mid_loud) - db(mid_quiet)).abs() < 0.01,
            "mid tone {} vs {} dB",
            db(mid_loud),
            db(mid_quiet)
        );
        for k in [1, 3] {
            assert!(gr_loud[k] < 0.01 && gr_quiet[k] < 0.01);
        }
    }
}

/// Channels are linked within a band: a loud tone on the left and a quiet
/// one on the right in the same band get the same gain.
#[test]
fn channels_are_linked_within_a_band() {
    let mut p = flat_params(3);
    p.bands[1] = MultibandBandParams {
        threshold_db: -30.0,
        ratio: 4.0,
        attack_ms: 5.0,
        release_ms: 50.0,
        makeup_db: 0.0,
        knee_db: 0.0,
    };
    let f = centre(1, 3, &DEFAULT_FREQUENCIES_HZ);
    let mut m = multiband(FS, 2, &p);
    let amps = [0.5, 0.005];
    let input = signal(2 * FS as usize, 2, |i, c| tone(f, amps[c], i));
    let mut x = input.clone();
    m.process(&mut x);
    let gain =
        |c: usize| tone_amp(&x, 2, c, f, FS as usize) / tone_amp(&input, 2, c, f, FS as usize);
    let (gl, gr) = (db(gain(0)), db(gain(1)));
    assert!(gl < -10.0, "left should be compressed: {gl}");
    assert!((gl - gr).abs() < 0.01, "left {gl} dB, right {gr} dB");
}

/// Drive feeds all bands: below threshold it raises the output by the drive;
/// above a hard-knee threshold (ratio 4) the band's gain reduction rises by
/// drive·(1 − 1/ratio), and the output matches `A·10^(drive/20)·Σ g_j·m_j`
/// with every band's `g_j` from the formula at its level plus drive (with
/// one band: the output rises by drive/ratio). 1 and 4 bands.
#[test]
fn drive_raises_band_levels() {
    let fs = FS as f64;
    let splits = DEFAULT_FREQUENCIES_HZ;
    for nb in [1usize, 4] {
        let k = nb / 2;
        let f = centre(k, nb, &splits);
        let mk = band_factor(k, nb, &splits, f, fs);
        let theory = |level: f64, drive: f64| {
            let amp = lin(level) / mk;
            let sum: f64 = (0..nb)
                .map(|j| {
                    let m = band_factor(j, nb, &splits, f, fs);
                    let gr = reduction_formula(db(amp * m) + drive, -30.0, 0.0, 4.0);
                    lin(-gr) * m
                })
                .sum();
            db(amp * lin(drive) * sum)
        };
        let measure = |level: f64, drive: f32| {
            let mut p = flat_params(nb);
            for b in &mut p.bands[..nb] {
                b.ratio = 4.0;
            }
            p.drive_db = drive;
            let mut m = multiband(FS, 2, &p);
            let amp = lin(level) / mk;
            let mut x = signal(2 * FS as usize, 2, |i, _| tone(f, amp, i));
            m.process(&mut x);
            (
                db(tone_amp(&x, 2, 0, f, FS as usize)),
                m.band_gain_reduction_db(k),
            )
        };
        // Below threshold (−30 dB): −45 dB band level, +6 dB drive.
        let (o0, g0) = measure(-45.0, 0.0);
        let (o6, g6) = measure(-45.0, 6.0);
        assert_eq!((g0, g6), (0.0, 0.0));
        assert!(((o6 - o0) - 6.0).abs() < 0.01, "bands={nb}: {}", o6 - o0);
        // Above threshold: −20 dB band level.
        let (o0, g0) = measure(-20.0, 0.0);
        let (o6, g6) = measure(-20.0, 6.0);
        assert!(((g6 - g0) - 4.5).abs() < 0.02, "bands={nb}: {}", g6 - g0);
        assert!((o0 - theory(-20.0, 0.0)).abs() < 0.05, "bands={nb}");
        assert!((o6 - theory(-20.0, 6.0)).abs() < 0.05, "bands={nb}");
        if nb == 1 {
            assert!(((o6 - o0) - 1.5).abs() < 0.02, "{}", o6 - o0);
        }
        // Negative drive.
        let (om, _) = measure(-45.0, -12.0);
        let (o0, _) = measure(-45.0, 0.0);
        assert!(((om - o0) + 12.0).abs() < 0.01);
    }
}

/// A drive jump glides without a click: with one band at ratio 1 the output
/// is `g[n]·x[n]`, so the gain trajectory is read directly. After a jump
/// from −12 to +12 dB it starts with zero speed (first step < 0.002 dB),
/// moves monotonically, never faster than the critically damped glide
/// allows (24 dB/(τ·e) ≈ 0.037 dB per sample at 48 kHz, τ = 5 ms), and ends
/// at +12 dB.
#[test]
fn drive_changes_glide() {
    let mut p = flat_params(1);
    p.drive_db = -12.0;
    let mut m = multiband(FS, 1, &p);
    let n = 4800;
    let input: Vec<f32> = (0..2 * n).map(|_| 0.25).collect();
    let mut a = input[..n].to_vec();
    m.process(&mut a);
    assert!((db(a[n - 1] as f64 / 0.25) + 12.0).abs() < 1e-4);
    p.drive_db = 12.0;
    m.set_params(&p);
    let mut b = input[n..].to_vec();
    m.process(&mut b);
    let g: Vec<f64> = b.iter().map(|&v| db(v as f64 / 0.25)).collect();
    assert!(g[0] + 12.0 < 0.002, "first step {}", g[0] + 12.0);
    let mut prev = -12.0;
    for &v in &g {
        assert!(v >= prev - 1e-6, "not monotonic");
        assert!(v - prev < 0.04, "too fast: {}", v - prev);
        prev = v;
    }
    assert!((g[n - 1] - 12.0).abs() < 1e-3, "end {}", g[n - 1]);
}

/// Parameter jumps between extremes while a programme-like signal runs give
/// finite output and no gain step: the compressors glide threshold, ratio and
/// makeup (10 ms), so the per-sample change of the output stays within the
/// signal's own slope times the largest possible gain.
#[test]
fn parameter_jumps_are_smooth() {
    let mut p = flat_params(4);
    let mut m = multiband(FS, 2, &p);
    let f = 300.0;
    let amp = 0.3;
    let n = 480;
    let max_gain = lin(12.0 + 24.0);
    // Largest slope of the input times the largest gain, plus a margin for
    // a 10 ms gain glide over the whole range.
    let bound = max_gain * amp * (2.0 * PI * f / FS as f64) * 3.0;
    let mut pos = 0usize;
    for step in 0..40 {
        let ext = step % 2 == 0;
        p.drive_db = if ext { 12.0 } else { -12.0 };
        for b in &mut p.bands {
            b.threshold_db = if ext { -60.0 } else { 0.0 };
            b.ratio = if ext { 20.0 } else { 1.0 };
            b.makeup_db = if ext { 24.0 } else { -12.0 };
            b.attack_ms = if ext { 0.1 } else { 500.0 };
            b.release_ms = if ext { 5.0 } else { 5000.0 };
            b.knee_db = if ext { 0.0 } else { 24.0 };
        }
        m.set_params(&p);
        let mut x = signal(n, 2, |i, _| tone(f, amp, pos + i));
        pos += n;
        m.process(&mut x);
        assert!(x.iter().all(|v| v.is_finite()));
        let step_max = x
            .as_chunks::<2>()
            .0
            .iter()
            .map(|fr| fr[0] as f64)
            .collect::<Vec<_>>()
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0, f64::max);
        assert!(step_max < bound, "step {step}: {step_max} > {bound}");
    }
}

/// No NaN/Inf at any parameter extreme, sample rate or nasty input
/// (+20 dBFS noise, full-scale square, impulses, tiny values, silence),
/// including NaN/Inf parameters and non-ascending crossover frequencies.
#[test]
fn robust_at_extremes() {
    let extreme = |hi: bool| MultibandBandParams {
        threshold_db: if hi { 0.0 } else { -60.0 },
        ratio: if hi { 1.0 } else { 20.0 },
        attack_ms: if hi { 500.0 } else { 0.1 },
        release_ms: if hi { 5000.0 } else { 5.0 },
        makeup_db: if hi { -12.0 } else { 24.0 },
        knee_db: if hi { 24.0 } else { 0.0 },
    };
    let silly = MultibandBandParams {
        threshold_db: f32::NAN,
        ratio: f32::INFINITY,
        attack_ms: -1.0,
        release_ms: f32::NAN,
        makeup_db: f32::INFINITY,
        knee_db: f32::NEG_INFINITY,
    };
    let freq_sets = [
        DEFAULT_FREQUENCIES_HZ,
        [20.0, 20.0, 20_000.0, 20_000.0],
        [5000.0, 100.0, f32::NAN, f32::INFINITY],
    ];
    for fs in [8_000.0f32, 44_100.0, 96_000.0, 192_000.0] {
        let frames = (fs as usize) / 8;
        let mut seed = 99u64;
        let loud = noise(frames * 2, &mut seed, 10.0);
        let square = signal(frames, 2, |i, _| if (i / 37) % 2 == 0 { 1.0 } else { -1.0 });
        let impulses = signal(frames, 2, |i, _| if i % 1000 == 0 { 1.0 } else { 0.0 });
        let tiny = signal(frames, 2, |i, _| if i % 2 == 0 { 1e-38 } else { -1e-38 });
        let silence = vec![0.0f32; frames * 2];
        for nb in 1..=MAX_CROSSOVER_BANDS {
            for (bi, band) in [extreme(false), extreme(true), silly]
                .into_iter()
                .enumerate()
            {
                for (fi, freqs) in freq_sets.iter().enumerate() {
                    // Rotate the drive extremes through the combinations.
                    {
                        let drive = [-12.0f32, 12.0, f32::NAN][(nb + bi + fi) % 3];
                        let p = MultibandParams {
                            num_bands: nb,
                            frequencies_hz: *freqs,
                            drive_db: drive,
                            detector: if bi == 1 {
                                CompressorDetector::Rms
                            } else {
                                CompressorDetector::Peak
                            },
                            bands: [band; MAX_CROSSOVER_BANDS],
                        };
                        let mut m = multiband(fs, 2, &p);
                        for input in [&loud, &square, &impulses, &tiny, &silence] {
                            let mut x = input.clone();
                            m.process(&mut x);
                            assert!(
                                x.iter().all(|v| v.is_finite()),
                                "fs={fs} bands={nb} band set {bi} freqs {fi} drive {drive}"
                            );
                            let gr = m.gain_reduction_db();
                            assert!(gr.iter().all(|g| g.is_finite() && *g >= 0.0));
                        }
                        // After a reset, silence gives exact silence.
                        m.reset();
                        let mut x = silence.clone();
                        m.process(&mut x);
                        assert!(x.iter().all(|&v| v == 0.0));
                    }
                }
            }
        }
    }
}

/// `reset` returns to the freshly configured state: the same input gives the
/// same output, bit for bit.
#[test]
fn reset_restores_fresh_state() {
    let p = MultibandParams {
        drive_db: 3.0,
        ..MultibandParams::default()
    };
    let mut seed = 3u64;
    let input = noise(2 * 9600, &mut seed, 0.7);
    let mut fresh = multiband(FS, 2, &p);
    let mut a = input.clone();
    fresh.process(&mut a);
    let mut used = multiband(FS, 2, &p);
    let mut junk = noise(2 * 5000, &mut seed, 1.0);
    used.process(&mut junk);
    used.reset();
    let mut b = input;
    used.process(&mut b);
    assert_eq!(a, b);
}

/// Buffer partitioning does not change the output (steady parameters):
/// one long buffer (several internal chunks) equals many odd-sized ones.
#[test]
fn buffer_size_does_not_change_output() {
    let p = MultibandParams::default();
    let mut seed = 21u64;
    let input = noise(2 * 7000, &mut seed, 0.8);
    let mut m1 = multiband(FS, 2, &p);
    let mut a = input.clone();
    m1.process(&mut a);
    let mut m2 = multiband(FS, 2, &p);
    let mut b = input;
    let mut pos = 0;
    for (i, len) in [1usize, 7, 480, 1023, 64, 2048].iter().cycle().enumerate() {
        if pos >= b.len() || i > 10_000 {
            break;
        }
        let end = (pos + len * 2).min(b.len());
        m2.process(&mut b[pos..end]);
        pos = end;
    }
    assert_eq!(a, b);
}

/// Band-count changes while running (allowed by the library, READY-only in
/// the element) stay finite and bounded, and the realised count follows.
#[test]
fn band_count_changes_are_safe() {
    let mut p = MultibandParams::default();
    let mut m = multiband(FS, 2, &p);
    let mut pos = 0;
    for nb in [4usize, 1, 5, 2, 3, 5, 1] {
        p.num_bands = nb;
        m.set_params(&p);
        assert_eq!(m.num_bands(), nb);
        let mut x = signal(4800, 2, |i, _| tone(440.0, 0.5, pos + i));
        pos += 4800;
        m.process(&mut x);
        assert!(x.iter().all(|v| v.is_finite() && v.abs() < 2.0));
        for k in nb..MAX_CROSSOVER_BANDS {
            assert_eq!(m.band_gain_reduction_db(k), 0.0);
        }
    }
}

/// `set_params`, `process` and `reset` never allocate (any buffer size,
/// band-count and detector changes included).
#[test]
fn no_allocation() {
    let mut p = MultibandParams::default();
    let mut m = multiband(FS, 2, &p);
    let mut seed = 77u64;
    let input = noise(2 * 3000, &mut seed, 0.9);
    let mut buf = input.clone();
    let before = allocs();
    for i in 0..80 {
        p.num_bands = 1 + i % MAX_CROSSOVER_BANDS;
        p.drive_db = (i % 25) as f32 - 12.0;
        p.detector = if i % 3 == 0 {
            CompressorDetector::Rms
        } else {
            CompressorDetector::Peak
        };
        p.frequencies_hz[1] = 500.0 + 10.0 * i as f32;
        for b in &mut p.bands {
            b.threshold_db = -(i as f32 % 60.0);
            b.ratio = 1.0 + (i % 20) as f32;
            b.makeup_db = (i % 36) as f32 - 12.0;
        }
        m.set_params(&p);
        let len = [2usize, 960, 6000, 14, 2048][i % 5];
        buf[..len].copy_from_slice(&input[..len]);
        m.process(&mut buf[..len]);
        if i % 13 == 12 {
            m.reset();
        }
        let _ = m.gain_reduction_db();
    }
    assert_eq!(
        allocs() - before,
        0,
        "allocations in set_params/process/reset"
    );
}
