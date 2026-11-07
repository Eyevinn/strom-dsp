//! Oversampler tests: passband ripple and stopband
//! measured with sine sweeps through the real processor, latency by impulse,
//! round trip, every factor at 44.1/48/96 kHz, robustness, SIMD level
//! equivalence and zero allocations after `configure`.
//!
//! Sweeps use exact DFT bins (whole numbers of cycles in the analysis window),
//! so a rectangular window has no leakage and each image/alias is measured on
//! its own bin.

use std::f64::consts::PI;

use strom_dsp::biquad::uses_fma;
use strom_dsp::oversample::{OversampleFactor, Oversampler, PASSBAND_EDGE};
use strom_dsp::simd::Level;

use crate::simd_support::levels;

use crate::counting_alloc::allocs;

// ---------------------------------------------------------------------------
// Helpers.

const FACTORS: [OversampleFactor; 3] = [
    OversampleFactor::X2,
    OversampleFactor::X4,
    OversampleFactor::X8,
];
const RATES: [f64; 3] = [44_100.0, 48_000.0, 96_000.0];

/// Base-rate analysis length (DFT size at the base rate).
const NB: usize = 2048;
/// Settling frames before the analysis window (≫ any latency).
const WARM: usize = 256;

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-30).log10()
}

/// Amplitude of DFT bin `bin` of channel `c` of interleaved `y`, over `n`
/// frames starting at frame `start`.
fn bin_amp(y: &[f32], ch: usize, c: usize, start: usize, n: usize, bin: usize) -> f64 {
    let (mut re, mut im) = (0.0, 0.0);
    for k in 0..n {
        let ph = 2.0 * PI * ((bin * k) % n) as f64 / n as f64;
        let v = y[(start + k) * ch + c] as f64;
        re += v * ph.cos();
        im -= v * ph.sin();
    }
    2.0 * (re * re + im * im).sqrt() / n as f64
}

/// Highest base-rate bin inside the passband.
fn top_bin() -> usize {
    (PASSBAND_EDGE * NB as f64).floor() as usize
}

/// Sweep bins from 1 to the passband edge.
fn sweep_bins() -> Vec<usize> {
    let top = top_bin();
    let mut v: Vec<usize> = (0..12).map(|i| 1 + i * (top - 1) / 11).collect();
    v.extend([3, 97, top - 1]);
    v
}

fn make(f: OversampleFactor, ch: usize) -> Oversampler {
    let mut o = Oversampler::new();
    o.configure(f, ch);
    o
}

/// Interleaved stereo sine at base-rate bin `bin` (both channels; the second
/// in cosine phase), amplitude `a`.
fn sine(frames: usize, ch: usize, bin: usize, n: usize, a: f64) -> Vec<f32> {
    let mut x = vec![0.0f32; frames * ch];
    for i in 0..frames {
        for c in 0..ch {
            let ph = 2.0 * PI * bin as f64 * i as f64 / n as f64 + c as f64 * PI / 2.0;
            x[i * ch + c] = (a * ph.sin()) as f32;
        }
    }
    x
}

/// Deterministic white noise (xorshift64*), uniform in [-1, 1).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn noise(len: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..len).map(|_| r.next()).collect()
}

// ---------------------------------------------------------------------------
// Frequency response (sweeps).

/// Upsampling: the fundamental keeps its level within 0.1 dB up to the
/// passband edge, and every image `k·fs ± f` is at least 90 dB down.
#[test]
fn upsample_ripple_and_images() {
    let ch = 2;
    let (mut worst_ripple, mut worst_image) = (0.0f64, -400.0f64);
    for &rate in &RATES {
        for &f in &FACTORS {
            let l = f.ratio();
            let n = NB * l;
            for &m in &sweep_bins() {
                let mut o = make(f, ch);
                let x = sine(WARM + NB, ch, m, NB, 0.5);
                let mut y = vec![0.0f32; x.len() * l];
                o.upsample(&x, &mut y);
                for c in 0..ch {
                    let a = bin_amp(&y, ch, c, WARM * l, n, m);
                    let ripple = db(a / 0.5).abs();
                    worst_ripple = worst_ripple.max(ripple);
                    assert!(
                        ripple <= 0.1,
                        "{rate} Hz {l}x bin {m}: passband error {ripple:.4} dB"
                    );
                    for k in 1..l {
                        for b in [k * NB - m, k * NB + m] {
                            if b > n / 2 {
                                continue;
                            }
                            let img = db(bin_amp(&y, ch, c, WARM * l, n, b) / 0.5);
                            worst_image = worst_image.max(img);
                            assert!(
                                img <= -90.0,
                                "{rate} Hz {l}x bin {m}: image at bin {b} only {img:.1} dB down"
                            );
                        }
                    }
                }
            }
        }
    }
    println!("upsample: worst ripple {worst_ripple:.5} dB, worst image {worst_image:.1} dB");
}

/// Downsampling: a high-rate tone in the passband keeps its level within
/// 0.1 dB; a tone that would alias into the passband (`k·fs ± f`) is at least
/// 90 dB down.
#[test]
fn downsample_ripple_and_aliases() {
    let ch = 2;
    let (mut worst_ripple, mut worst_alias) = (0.0f64, -400.0f64);
    for &rate in &RATES {
        for &f in &FACTORS {
            let l = f.ratio();
            let n = NB * l;
            let bins = sweep_bins();
            let mut cases: Vec<(usize, usize, bool)> = bins.iter().map(|&m| (m, m, true)).collect();
            for &m in &bins {
                for k in 1..l {
                    for b in [k * NB - m, k * NB + m] {
                        if b < n / 2 {
                            cases.push((b, m, false));
                        }
                    }
                }
            }
            for (b, alias, pass) in cases {
                let mut o = make(f, ch);
                let x = sine((WARM + NB) * l, ch, b, n, 0.5);
                let mut y = vec![0.0f32; (WARM + NB) * ch];
                o.downsample(&x, &mut y);
                for c in 0..ch {
                    let a = db(bin_amp(&y, ch, c, WARM, NB, alias) / 0.5);
                    if pass {
                        worst_ripple = worst_ripple.max(a.abs());
                        assert!(
                            a.abs() <= 0.1,
                            "{rate} Hz {l}x bin {b}: passband error {a:.4} dB"
                        );
                    } else {
                        worst_alias = worst_alias.max(a);
                        assert!(
                            a <= -90.0,
                            "{rate} Hz {l}x high-rate bin {b}: alias at {alias} only {a:.1} dB down"
                        );
                    }
                }
            }
        }
    }
    println!("downsample: worst ripple {worst_ripple:.5} dB, worst alias {worst_alias:.1} dB");
}

// ---------------------------------------------------------------------------
// Latency and round trip.

/// Index of the largest |value| of channel `c`.
fn argmax(y: &[f32], ch: usize, c: usize) -> usize {
    (0..y.len() / ch)
        .max_by(|&a, &b| y[a * ch + c].abs().total_cmp(&y[b * ch + c].abs()))
        .unwrap()
}

/// The response of channel `c` is symmetric around `centre` (linear phase).
fn assert_symmetric(y: &[f32], ch: usize, c: usize, centre: usize, span: usize, what: &str) {
    for t in 1..span.min(centre) {
        let (a, b) = (y[(centre + t) * ch + c], y[(centre - t) * ch + c]);
        assert!(
            (a - b).abs() <= 1e-6,
            "{what}: asymmetric at ±{t}: {a} vs {b}"
        );
    }
}

#[test]
fn latency_is_exact_by_impulse() {
    let ch = 2;
    let n0 = 100;
    for &f in &FACTORS {
        let l = f.ratio();
        let mut o = make(f, ch);
        let frames = 512;

        // Up: impulse at base frame n0 peaks at high-rate frame (n0 + up)·L.
        let mut x = vec![0.0f32; frames * ch];
        x[n0 * ch] = 1.0;
        x[n0 * ch + 1] = -0.5;
        let mut up = vec![0.0f32; frames * l * ch];
        o.upsample(&x, &mut up);
        let centre = (n0 + o.up_latency_frames()) * l;
        for c in 0..ch {
            assert_eq!(argmax(&up, ch, c), centre, "{l}x up peak");
            assert_symmetric(&up, ch, c, centre, 400, "up");
        }
        // Gain of the centre tap: exactly the impulse.
        assert_eq!(up[centre * ch], 1.0);

        // Round trip: peak at n0 + latency, symmetric.
        let mut rt = vec![0.0f32; frames * ch];
        o.downsample(&up, &mut rt);
        let centre = n0 + o.latency_frames();
        for c in 0..ch {
            assert_eq!(argmax(&rt, ch, c), centre, "{l}x round-trip peak");
            assert_symmetric(&rt, ch, c, centre, 90, "round trip");
        }
        assert!((rt[centre * ch] - 1.0).abs() < 0.05);

        // Down alone: high-rate impulse at frame n0·L peaks at n0 + down.
        let mut o = make(f, ch);
        let mut hx = vec![0.0f32; frames * l * ch];
        hx[n0 * l * ch] = 1.0;
        hx[n0 * l * ch + 1] = 1.0;
        let mut dy = vec![0.0f32; frames * ch];
        o.downsample(&hx, &mut dy);
        let centre = n0 + o.down_latency_frames();
        for c in 0..ch {
            assert_eq!(argmax(&dy, ch, c), centre, "{l}x down peak");
            assert_symmetric(&dy, ch, c, centre, 60, "down");
        }
        if l == 2 {
            assert_eq!(dy[centre * ch], 0.5, "down centre tap is ½");
        }
    }
}

/// Up then down returns the input delayed by exactly `latency_frames`, within
/// the ripple, for a multi-tone signal inside the passband.
#[test]
fn round_trip_returns_delayed_input() {
    let ch = 3;
    for &rate in &RATES {
        let tones = [100.0, 997.0, 5_000.0, 12_345.0, PASSBAND_EDGE * rate * 0.99];
        for &f in &FACTORS {
            let l = f.ratio();
            let mut o = make(f, ch);
            let frames = 8192;
            let mut x = vec![0.0f32; frames * ch];
            for i in 0..frames {
                for c in 0..ch {
                    let t = i as f64 / rate;
                    let v: f64 = tones
                        .iter()
                        .enumerate()
                        .map(|(k, &fr)| 0.18 * (2.0 * PI * fr * t + (k + c) as f64).sin())
                        .sum();
                    x[i * ch + c] = v as f32;
                }
            }
            let mut up = vec![0.0f32; frames * l * ch];
            let mut y = vec![0.0f32; frames * ch];
            // Varying block sizes through both paths.
            let mut pos = 0;
            for (k, sz) in [1usize, 480, 7, 1024, 65, 333].iter().cycle().enumerate() {
                if pos >= frames {
                    break;
                }
                let n = (*sz).min(frames - pos);
                let (a, b) = (pos * ch, (pos + n) * ch);
                o.upsample(&x[a..b], &mut up[a * l..b * l]);
                o.downsample(&up[a * l..b * l], &mut y[a..b]);
                pos += n;
                let _ = k;
            }
            let d = o.latency_frames();
            let (mut err, mut sig) = (0.0f64, 0.0f64);
            for i in 1000..frames {
                for c in 0..ch {
                    let e = (y[i * ch + c] - x[(i - d) * ch + c]) as f64;
                    err = err.max(e.abs());
                    sig = sig.max((x[(i - d) * ch + c] as f64).abs());
                }
            }
            let rel = db(err / sig);
            println!("{rate} Hz {l}x: round-trip max error {rel:.1} dB re peak");
            // 0.1 dB of ripple would allow about −39 dB; the design does far better.
            assert!(rel < -60.0, "{rate} Hz {l}x: round-trip error {rel:.1} dB");
        }
    }
}

// ---------------------------------------------------------------------------
// Structure, robustness, SIMD and real-time safety.

/// Block size and channel count change nothing: any split of the input gives
/// bit-identical output, and each channel equals a mono run of that channel.
#[test]
fn block_size_and_channel_independence() {
    for &f in &FACTORS {
        let l = f.ratio();
        for ch in [1usize, 2, 3, 8, 64] {
            let frames = 700;
            let x = noise(frames * ch, 7 + ch as u64);
            let mut one = make(f, ch);
            let mut ref_up = vec![0.0f32; frames * l * ch];
            one.upsample(&x, &mut ref_up);
            let mut ref_dn = vec![0.0f32; frames * ch];
            one.downsample(&ref_up, &mut ref_dn);

            let mut o = make(f, ch);
            let mut up = vec![0.0f32; frames * l * ch];
            let mut dn = vec![0.0f32; frames * ch];
            let mut pos = 0;
            for sz in [1usize, 7, 63, 64, 65, 129, 3].iter().cycle() {
                if pos >= frames {
                    break;
                }
                let n = (*sz).min(frames - pos);
                let (a, b) = (pos * ch, (pos + n) * ch);
                o.upsample(&x[a..b], &mut up[a * l..b * l]);
                o.downsample(&up[a * l..b * l], &mut dn[a..b]);
                pos += n;
            }
            assert_eq!(up, ref_up, "{l}x {ch} ch: up differs with block size");
            assert_eq!(dn, ref_dn, "{l}x {ch} ch: down differs with block size");

            if ch > 1 {
                let c = ch - 1;
                let mono: Vec<f32> = (0..frames).map(|i| x[i * ch + c]).collect();
                let mut m = make(f, 1);
                let mut mu = vec![0.0f32; frames * l];
                m.upsample(&mono, &mut mu);
                for i in 0..frames * l {
                    assert_eq!(mu[i], ref_up[i * ch + c]);
                }
            }
        }
    }
}

/// `upsample_peaks` is the per-frame max |·| of what `upsample` produces.
#[test]
fn peaks_match_upsampled_signal() {
    let ch = 2;
    for &f in &FACTORS {
        let l = f.ratio();
        let frames = 300;
        let x = noise(frames * ch, 99);
        let mut a = make(f, ch);
        let mut up = vec![0.0f32; frames * l * ch];
        a.upsample(&x, &mut up);
        let mut b = make(f, ch);
        let mut peaks = vec![0.0f32; frames];
        b.upsample_peaks(&x[..100 * ch], &mut peaks[..100]);
        b.upsample_peaks(&x[100 * ch..], &mut peaks[100..]);
        for (i, p) in peaks.iter().enumerate() {
            let m = up[i * l * ch..(i + 1) * l * ch]
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs()));
            assert_eq!(*p, m);
        }
        // An inter-sample peak: fs/4 sine at 45° phase has samples at ±0.707
        // but a true peak of 1.
        let mut c = make(f, 1);
        let s: Vec<f32> = (0..400)
            .map(|i| (PI / 2.0 * i as f64 + PI / 4.0).sin() as f32)
            .collect();
        let mut pk = vec![0.0f32; 400];
        c.upsample_peaks(&s, &mut pk);
        let tp = pk[200..].iter().fold(0.0f32, |m, v| m.max(*v));
        let expect = 1.0;
        assert!((tp - expect).abs() < 0.01, "{l}x: true peak {tp}");
    }
}

/// Non-finite and extreme inputs: no panic, finite output for finite input,
/// and full recovery after a NaN/Inf burst (an FIR forgets in finite time).
#[test]
fn robustness() {
    for &f in &FACTORS {
        let l = f.ratio();
        let ch = 2;
        let mut o = make(f, ch);
        let frames = 400;
        let mut x = vec![0.0f32; frames * ch];
        for i in 10..20 {
            x[i * ch] = f32::NAN;
            x[i * ch + 1] = f32::INFINITY;
        }
        x[30 * ch] = f32::NEG_INFINITY;
        let mut up = vec![0.0f32; frames * l * ch];
        o.upsample(&x, &mut up);
        let mut y = vec![0.0f32; frames * ch];
        o.downsample(&up, &mut y);
        assert!(up[200 * l * ch..].iter().all(|v| *v == 0.0));
        assert!(y[250 * ch..].iter().all(|v| *v == 0.0));

        // Large finite input stays finite.
        let big: Vec<f32> = noise(frames * ch, 3).iter().map(|v| v * 1e6).collect();
        o.reset();
        o.upsample(&big, &mut up);
        o.downsample(&up, &mut y);
        assert!(up.iter().chain(&y).all(|v| v.is_finite()));

        // Reset gives the same output as a fresh instance.
        let s = noise(frames * ch, 5);
        o.reset();
        o.upsample(&s, &mut up);
        let mut fresh = make(f, ch);
        let mut up2 = vec![0.0f32; frames * l * ch];
        fresh.upsample(&s, &mut up2);
        assert_eq!(up, up2);

        // Empty and short buffers.
        o.upsample(&[], &mut []);
        o.downsample(&[], &mut []);
        o.upsample(&s[..ch], &mut up[..l * ch]);
    }
    // Unconfigured: no-op, no panic. Zero channels is treated as one.
    let mut u = Oversampler::new();
    let mut out = [7.0f32; 8];
    u.upsample(&[1.0; 4], &mut out);
    assert_eq!(out, [7.0; 8]);
    u.configure(OversampleFactor::X4, 0);
    assert_eq!(u.channels(), 1);
}

/// Every SIMD level the CPU supports matches the fallback (upsample,
/// downsample and `upsample_peaks`): bit-identical between levels with the
/// same FMA policy, otherwise within f32 rounding.
#[test]
fn every_level_matches_fallback() {
    for &f in &FACTORS {
        let l = f.ratio();
        for ch in [1usize, 2, 3, 8] {
            let frames = 1000;
            let x = noise(frames * ch, 11);
            let run = |level: Level| {
                let mut o = make(f, ch);
                o.set_simd_level(level);
                let mut up = vec![0.0f32; frames * l * ch];
                o.upsample(&x, &mut up);
                let mut dn = vec![0.0f32; frames * ch];
                o.downsample(&up, &mut dn);
                o.reset();
                let mut pk = vec![0.0f32; frames];
                o.upsample_peaks(&x, &mut pk);
                [up, dn, pk].concat()
            };
            let runs: Vec<_> = levels()
                .into_iter()
                .map(|(name, level)| (name, level, run(level)))
                .collect();
            for (name, level, y) in &runs {
                // Reference: the first level with the same FMA policy (exact)
                // and the fallback (within rounding).
                let same = runs.iter().find(|r| uses_fma(r.1) == uses_fma(*level));
                let (_, _, exact) = same.expect("a level matches itself");
                assert!(
                    y.iter().zip(exact).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{l}x {ch} ch: {name} differs from a level with the same FMA policy"
                );
                for (a, b) in y.iter().zip(&runs[0].2) {
                    assert!((a - b).abs() <= 1e-5, "{l}x {ch} ch {name}: {a} vs {b}");
                }
            }
        }
    }
}

#[test]
fn no_allocation_after_configure() {
    for &f in &FACTORS {
        let l = f.ratio();
        let ch = 2;
        let mut o = make(f, ch);
        let x = noise(1024 * ch, 1);
        let mut up = vec![0.0f32; 1024 * l * ch];
        let mut y = vec![0.0f32; 1024 * ch];
        let mut pk = vec![0.0f32; 1024];
        let before = allocs();
        for sz in [1usize, 480, 1024, 7, 64] {
            o.upsample(&x[..sz * ch], &mut up[..sz * l * ch]);
            o.downsample(&up[..sz * l * ch], &mut y[..sz * ch]);
            o.upsample_peaks(&x[..sz * ch], &mut pk[..sz]);
            let _ = o.latency_frames();
        }
        o.reset();
        assert_eq!(allocs() - before, 0, "{l}x allocated");
    }
}
