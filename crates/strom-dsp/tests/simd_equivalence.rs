//! Every SIMD level the CPU supports against the scalar fallback level, sample
//! by sample, for the equalizer and the biquad.
//!
//! Each lane of a vector kernel executes exactly the operations of the scalar
//! fallback, in the same order. The only permitted difference is the FMA
//! policy (`strom_dsp::biquad::uses_fma`): levels with the same policy must be
//! **bit-identical**. Levels with different policies (on x86-64: AVX2/AVX-512
//! fuse, the fallback and SSE do not) may differ by the rounding of the
//! product-sums. The documented bound for that case is
//! `|a − b| ≤ 1e-5·max(|a|, |b|) + 1e-9` per output sample (a few `f32` ulps;
//! the filter state is `f64`, so the difference starts at about 1e-16 relative
//! and is amplified only by the filters' own error gain).
//!
//! Scenarios cover the steady state, coefficient ramps (frequency, gain and Q
//! automation), crossfades (type changes, enable/disable, `num_bands` changes,
//! a ramp during a crossfade), `reset`, buffers of odd lengths, and 1, 2, 3, 4,
//! 5, 8, 13 and 64 channels (every lane-group layout of every level: full
//! groups, padded `f64x2`, padded native-width groups).

use strom_dsp::biquad::{uses_fma, Biquad, BiquadCoefficients};
use strom_dsp::simd::{detect, Level};
use strom_dsp::{BandParams, BandType, Equalizer, EqualizerParams, Processor, MAX_BANDS};

use fearless_simd::Simd;

const CHANNELS: [usize; 8] = [1, 2, 3, 4, 5, 8, 13, 64];
const FS: f32 = 48_000.0;

/// The fallback plus every vector level the running CPU supports.
fn levels() -> Vec<(&'static str, Level)> {
    let d = detect();
    #[allow(unused_mut)]
    let mut v = vec![("fallback", Level::fallback())];
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = d.as_neon() {
        v.push(("neon", t.level()));
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if let Some(t) = d.as_sse2() {
            v.push(("sse2", t.level()));
        }
        if let Some(t) = d.as_sse4_2() {
            v.push(("sse4.2", t.level()));
        }
        if let Some(t) = d.as_avx2() {
            v.push(("avx2", t.level()));
        }
        if let Some(t) = d.as_avx512() {
            v.push(("avx512", t.level()));
        }
    }
    let _ = d;
    v
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

fn band(ty: BandType, f: f32, g: f32, q: f32) -> BandParams {
    BandParams {
        enabled: true,
        band_type: ty,
        frequency: f,
        gain_db: g,
        q,
    }
}

/// Compare a level's output with the fallback's (see the module docs).
fn compare(ctx: &str, level: Level, got: &[f32], reference: &[f32]) -> f64 {
    assert_eq!(got.len(), reference.len(), "{ctx}");
    let exact = uses_fma(level) == uses_fma(Level::fallback());
    let mut worst = 0.0f64;
    for (i, (&a, &b)) in got.iter().zip(reference).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "{ctx}: non-finite at {i}");
        if exact {
            assert_eq!(a.to_bits(), b.to_bits(), "{ctx}: sample {i}: {a} vs {b}");
        } else {
            let (a, b) = (a as f64, b as f64);
            let d = (a - b).abs();
            let bound = 1e-5 * a.abs().max(b.abs()) + 1e-9;
            assert!(d <= bound, "{ctx}: sample {i}: {a} vs {b}");
            worst = worst.max(d / bound);
        }
    }
    worst
}

/// One equalizer run: parameter changes of every kind between buffers of
/// varying length. Returns all output samples.
fn run_equalizer(level: Level, ch: usize) -> Vec<f32> {
    let mut e = Equalizer::new();
    e.configure(FS, ch);
    e.set_simd_level(level);
    let mut p = EqualizerParams {
        num_bands: 4,
        ..Default::default()
    };
    p.bands[0] = band(BandType::HighPass, 30.0, 0.0, 0.707);
    p.bands[1] = band(BandType::Peaking, 300.0, -6.0, 1.2);
    p.bands[2] = band(BandType::Peaking, 3000.0, 4.5, 2.5);
    p.bands[3] = band(BandType::HighShelf, 9000.0, 3.0, 0.707);
    p.bands[4] = band(BandType::LowShelf, 120.0, 5.0, 0.6);
    p.bands[5] = band(BandType::Notch, 50.0, 0.0, 8.0);
    p.bands[6] = band(BandType::AllPass, 1500.0, 0.0, 0.9);
    p.bands[7] = band(BandType::LowPass, 15_000.0, 0.0, 1.1);
    e.set_params(&p);

    let mut seed = 0x5eed_0000 + ch as u64;
    let mut out = Vec::new();
    let mut run = |e: &mut Equalizer, frames: usize, seed: &mut u64| {
        let mut x = noise(frames * ch, seed, 0.7);
        if let Some(v) = x.first_mut() {
            *v = 0.95; // an impulse at every buffer start
        }
        e.process(&mut x);
        out.extend_from_slice(&x);
    };
    // Steady state, odd buffer sizes.
    for frames in [480, 1, 7, 333, 64] {
        run(&mut e, frames, &mut seed);
    }
    // Ramps: gain, frequency and Q automation on several bands at once.
    for k in 0..6 {
        p.bands[1].gain_db = -6.0 + 3.0 * k as f32;
        p.bands[2].frequency = 3000.0 * (1.0 + 0.3 * k as f32);
        p.bands[3].q = 0.5 + 0.2 * k as f32;
        e.set_params(&p);
        run(&mut e, [100, 31, 256, 5, 480, 77][k], &mut seed);
    }
    // Crossfades: type change, disable, num_bands growing (fade-ins), with a
    // ramp running on another band at the same time.
    p.bands[2].band_type = BandType::LowShelf;
    p.bands[3].enabled = false;
    p.num_bands = 7;
    p.bands[1].gain_db = 9.0;
    e.set_params(&p);
    for frames in [64, 200, 3, 333] {
        run(&mut e, frames, &mut seed);
    }
    // A type change requested during a running crossfade, then all 8 bands.
    p.bands[2].band_type = BandType::BandPassPeak;
    p.num_bands = MAX_BANDS;
    p.bands[3].enabled = true;
    e.set_params(&p);
    for frames in [100, 480, 480, 1000] {
        run(&mut e, frames, &mut seed);
    }
    // Shrinking to one band (fade-outs), then steady.
    p.num_bands = 1;
    e.set_params(&p);
    for frames in [250, 480, 480] {
        run(&mut e, frames, &mut seed);
    }
    // Reset, new parameters, steady.
    e.reset();
    p.num_bands = 5;
    p.bands[0].band_type = BandType::BandPassSkirt;
    e.set_params(&p);
    for frames in [480, 13] {
        run(&mut e, frames, &mut seed);
    }
    out
}

#[test]
fn equalizer_levels_match_fallback() {
    let levels = levels();
    eprintln!(
        "levels: {:?}",
        levels.iter().map(|l| l.0).collect::<Vec<_>>()
    );
    for ch in CHANNELS {
        let reference = run_equalizer(Level::fallback(), ch);
        for &(name, level) in &levels[1..] {
            let got = run_equalizer(level, ch);
            let worst = compare(&format!("eq {name} ch {ch}"), level, &got, &reference);
            if worst > 0.0 {
                eprintln!("eq {name} ch {ch}: worst difference {worst:.3} of the bound");
            }
        }
    }
}

/// Switching the level between `process` calls is seamless (the state layout
/// does not depend on the level): alternating between levels with the same FMA
/// policy gives exactly the output of a single level.
#[test]
fn switching_levels_mid_stream_is_seamless() {
    let levels = levels();
    for ch in [1, 3, 8, 13] {
        let same: Vec<Level> = levels
            .iter()
            .map(|l| l.1)
            .filter(|&l| uses_fma(l) == uses_fma(Level::fallback()))
            .collect();
        let mut e = Equalizer::new();
        e.configure(FS, ch);
        let mut p = EqualizerParams::default();
        p.bands[0] = band(BandType::Peaking, 700.0, 6.0, 1.0);
        p.bands[1] = band(BandType::HighPass, 40.0, 0.0, 0.707);
        p.bands[2] = band(BandType::LowShelf, 200.0, -4.0, 0.707);
        e.set_params(&p);
        let mut seed = 99;
        let input = noise(4000 * ch, &mut seed, 0.6);
        let mut a = input.clone();
        e.set_simd_level(Level::fallback());
        e.process(&mut a);
        let mut b = input;
        let mut f = Equalizer::new();
        f.configure(FS, ch);
        f.set_params(&p);
        for (i, chunk) in b.chunks_mut(97 * ch).enumerate() {
            f.set_simd_level(same[i % same.len()]);
            f.process(chunk);
        }
        assert_eq!(a, b, "ch {ch}");
    }
}

fn run_biquad(level: Level, ch: usize) -> Vec<f32> {
    let fs = FS as f64;
    let mut bq = Biquad::new(BiquadCoefficients::peaking(1000.0, 9.0, 2.0, fs), ch);
    bq.set_simd_level(level);
    let mut seed = 7 + ch as u64;
    let mut out = Vec::new();
    for (k, frames) in [480usize, 3, 100, 333, 480, 64].into_iter().enumerate() {
        match k {
            2 => bq.ramp_to(BiquadCoefficients::low_shelf(200.0, -6.0, 0.7, fs), 250),
            4 => bq.set_coefficients(BiquadCoefficients::high_pass(25.0, 0.707, fs)),
            _ => {}
        }
        let mut x = noise(frames * ch, &mut seed, 0.8);
        bq.process(&mut x);
        out.extend_from_slice(&x);
    }
    out
}

#[test]
fn biquad_levels_match_fallback() {
    let levels = levels();
    for ch in CHANNELS {
        let reference = run_biquad(Level::fallback(), ch);
        for &(name, level) in &levels[1..] {
            let got = run_biquad(level, ch);
            compare(&format!("biquad {name} ch {ch}"), level, &got, &reference);
        }
    }
}

/// The vector path is really used: the detected level is not the fallback
/// on the platforms CI runs on.
#[test]
fn detected_level_is_a_vector_level() {
    if cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
        assert!(!detect().is_fallback());
    }
}
