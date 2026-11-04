//! Tests of the real-time BS.1770 loudness detector
//! ([`strom_dsp::meter::LoudnessDetector`]) and its K-weighting design, from
//! the standard and from theory:
//!
//! - the coefficients derived for 48 kHz equal the BS.1770 tables;
//! - at every other rate the digital magnitude equals the analog prototype at
//!   the pre-warped frequency, and stays close to the 48 kHz reference;
//! - a 997 Hz sine at −20 dBFS in one channel reads −23.0 LUFS (BS.1770:
//!   a 0 dBFS 1 kHz sine in one front channel reads −3.01 LKFS);
//! - momentary and short-term agree with the `ebur128` reference meter
//!   within 0.2 LU on white and pink noise;
//! - channel weights, block-size independence, running-sum drift,
//!   robustness and zero allocations.

use crate::agc_support::{noise, pink, sine_on, Looper};
use strom_dsp::biquad::BiquadCoefficients;
use strom_dsp::meter::{
    k_weighting, k_weighting_prototypes, lufs_from_mean_square, ChannelLayout, ChannelRole,
    LoudnessConfig, LoudnessDetector, LoudnessMeter, BS1770_PRE_FILTER_48K, BS1770_RLB_48K,
    SILENCE_LUFS,
};

use crate::counting_alloc::allocs;

// ---------------------------------------------------------------------------

const RATES: [f64; 10] = [
    8_000.0, 16_000.0, 22_050.0, 32_000.0, 44_100.0, 48_000.0, 88_200.0, 96_000.0, 176_400.0,
    192_000.0,
];

fn db(x: f64) -> f64 {
    20.0 * x.log10()
}

fn detector(fs: f64, channels: usize, layout: ChannelLayout) -> LoudnessDetector {
    let mut d = LoudnessDetector::new();
    d.configure(fs as f32, channels, layout);
    d
}

/// Magnitude of the two K-weighting stages in series.
fn k_mag(c: &[BiquadCoefficients; 2], f: f64, fs: f64) -> f64 {
    c[0].magnitude_at(f, fs) * c[1].magnitude_at(f, fs)
}

#[test]
fn coefficients_at_48k_match_bs1770_tables() {
    let [pre, rlb] = k_weighting(48_000.0);
    for (got, want) in [(pre, BS1770_PRE_FILTER_48K), (rlb, BS1770_RLB_48K)] {
        for (g, w) in [
            (got.b0, want.b0),
            (got.b1, want.b1),
            (got.b2, want.b2),
            (got.a1, want.a1),
            (got.a2, want.a2),
        ] {
            assert!((g - w).abs() < 1e-12, "{g} vs {w}");
        }
    }
    // The published table values themselves (BS.1770-4 Annex 1, tables 1 and 2).
    assert_eq!(BS1770_PRE_FILTER_48K.b0, 1.53512485958697);
    assert_eq!(BS1770_PRE_FILTER_48K.b1, -2.69169618940638);
    assert_eq!(BS1770_PRE_FILTER_48K.b2, 1.19839281085285);
    assert_eq!(BS1770_PRE_FILTER_48K.a1, -1.69065929318241);
    assert_eq!(BS1770_PRE_FILTER_48K.a2, 0.73248077421585);
    assert_eq!(BS1770_RLB_48K.a1, -1.99004745483398);
    assert_eq!(BS1770_RLB_48K.a2, 0.99007225036621);
}

#[test]
fn analog_prototypes_are_the_expected_shapes() {
    let [pre, rlb] = k_weighting_prototypes();
    println!("pre-filter: {pre:?} ({:.4} dB)", db(pre.high_gain));
    println!("RLB:        {rlb:?}");
    // Pre-filter: a +4 dB high shelf near 1.68 kHz, Butterworth-like Q,
    // unity at DC, band gain ≈ √V_H (a symmetric shelf).
    assert!((pre.f0_hz - 1681.97).abs() < 0.01, "{}", pre.f0_hz);
    assert!((pre.q - 0.7072).abs() < 1e-3, "{}", pre.q);
    assert!((db(pre.high_gain) - 4.0).abs() < 0.01);
    assert!((pre.low_gain - 1.0).abs() < 1e-9);
    assert!((pre.band_gain - pre.high_gain.sqrt()).abs() < 1e-3);
    // RLB: a second-order high-pass near 38 Hz with Q ≈ 0.5.
    assert!((rlb.f0_hz - 38.135).abs() < 0.01, "{}", rlb.f0_hz);
    assert!((rlb.q - 0.5003).abs() < 1e-3, "{}", rlb.q);
    assert!(rlb.low_gain.abs() < 1e-12 && rlb.band_gain.abs() < 1e-12);
    assert!((rlb.high_gain - 1.0).abs() < 0.01);
}

#[test]
fn magnitude_at_every_rate_equals_the_warped_analog_prototype() {
    let protos = k_weighting_prototypes();
    for fs in RATES {
        let c = k_weighting(fs);
        assert!(c[0].is_stable() && c[1].is_stable(), "unstable at {fs}");
        let mut f = 5.0;
        while f < 0.49 * fs {
            let digital = k_mag(&c, f, fs);
            let analog = protos[0].magnitude_warped(f, fs) * protos[1].magnitude_warped(f, fs);
            assert!(
                (db(digital) - db(analog)).abs() < 1e-6,
                "fs {fs} f {f}: {digital} vs {analog}"
            );
            f *= 1.07;
        }
    }
}

#[test]
fn response_at_other_rates_stays_close_to_the_48k_reference() {
    let reference = k_weighting(48_000.0);
    for fs in RATES {
        let c = k_weighting(fs);
        let top = 16_000.0f64.min(0.45 * fs);
        let mut worst: f64 = 0.0;
        let mut f = 10.0;
        while f <= top {
            let d = db(k_mag(&c, f, fs)) - db(k_mag(&reference, f, 48_000.0));
            worst = worst.max(d.abs());
            f *= 1.03;
        }
        println!("fs {fs:>8}: max |deviation| from 48 kHz up to {top:.0} Hz = {worst:.4} dB");
        // 44.1 kHz and up: within 0.1 dB to 16 kHz. Lower rates warp the
        // shelf more near their Nyquist frequency; bound measured with margin.
        let bound = if fs >= 44_100.0 { 0.1 } else { 1.0 };
        assert!(worst < bound, "fs {fs}: {worst} dB");
    }
}

#[test]
fn sine_997hz_at_minus_20_dbfs_reads_minus_23_lufs() {
    for fs in [44_100.0, 48_000.0, 96_000.0] {
        for (channels, ch, layout) in [
            (1, 0, ChannelLayout::Mono),
            (2, 0, ChannelLayout::Stereo),
            (2, 1, ChannelLayout::Stereo),
            (6, 2, ChannelLayout::Surround51),
        ] {
            let amp = 10f64.powf(-20.0 / 20.0);
            let x = sine_on((4.0 * fs) as usize, channels, ch, fs, 997.0, amp);
            let mut d = detector(fs, channels, layout);
            d.process(&x);
            // Analytic: mean square amp²/2 times the K-weighting power gain.
            let c = k_weighting(fs);
            let k = k_mag(&c, 997.0, fs);
            let expected = lufs_from_mean_square(0.5 * amp * amp * k * k);
            let st = d.short_term_lufs();
            let m = d.momentary_lufs();
            assert!((expected - -23.0).abs() < 0.02, "expected {expected}");
            assert!((st - -23.0).abs() < 0.1, "fs {fs} ch {ch}: short-term {st}");
            assert!((m - -23.0).abs() < 0.1, "fs {fs} ch {ch}: momentary {m}");
            assert!((st - expected).abs() < 0.01, "{st} vs {expected}");
        }
    }
}

/// Our readings against the `ebur128` reference at sub-block boundaries.
fn compare_with_reference(fs: f64, channels: usize, layout: ChannelLayout, slope: f64, seed: u64) {
    let frames = 1 << 18;
    let mut sig = noise(frames, channels, fs, slope, seed);
    // A slow level contour (±10 dB) so the windows see changing loudness.
    for (n, f) in sig.chunks_exact_mut(channels).enumerate() {
        let t = n as f64 / fs;
        let g = 10f64.powf((-20.0 + 10.0 * (2.0 * std::f64::consts::PI * 0.2 * t).sin()) / 20.0);
        f.iter_mut().for_each(|v| *v = (*v as f64 * g) as f32);
    }
    let mut d = detector(fs, channels, layout);
    let mut m = LoudnessMeter::new(&LoudnessConfig {
        sample_rate: fs as f32,
        channels,
        layout,
        true_peak: false,
    })
    .unwrap();
    let sub = d.sub_block_frames();
    let mut worst_m: f64 = 0.0;
    let mut worst_st: f64 = 0.0;
    for (i, block) in sig.chunks_exact(sub * channels).enumerate() {
        d.process(block);
        m.process(block);
        if i >= 300 && i % 7 == 0 {
            worst_m = worst_m.max((d.momentary_lufs() - m.momentary()).abs());
            worst_st = worst_st.max((d.short_term_lufs() - m.short_term()).abs());
        }
    }
    println!(
        "fs {fs} ch {channels} slope {slope}: max |Δ| momentary {worst_m:.4} LU, short-term {worst_st:.4} LU"
    );
    assert!(worst_m < 0.2 && worst_st < 0.2, "{worst_m} {worst_st}");
}

#[test]
fn agrees_with_ebur128_on_noise() {
    for fs in [44_100.0, 48_000.0, 96_000.0] {
        compare_with_reference(fs, 2, ChannelLayout::Stereo, 1.0, 11);
        compare_with_reference(fs, 2, ChannelLayout::Stereo, 0.0, 12);
    }
    compare_with_reference(48_000.0, 1, ChannelLayout::Mono, 1.0, 13);
    compare_with_reference(48_000.0, 6, ChannelLayout::Surround51, 1.0, 14);
    compare_with_reference(32_000.0, 3, ChannelLayout::Unknown, 1.0, 15);
}

#[test]
fn channel_weights_follow_bs1770() {
    let fs = 48_000.0;
    let amp = 0.1;
    let read = |layout: ChannelLayout, ch: usize| {
        let x = sine_on(fs as usize * 4, 6, ch, fs, 997.0, amp);
        let mut d = detector(fs, 6, layout);
        d.process(&x);
        d.short_term_lufs()
    };
    let front = read(ChannelLayout::Surround51, 0);
    let surround = read(ChannelLayout::Surround51, 4);
    let lfe = read(ChannelLayout::Surround51, 3);
    assert!((surround - front - 10.0 * 1.41f64.log10()).abs() < 1e-6);
    assert_eq!(lfe, SILENCE_LUFS);
    // Unknown layout: every channel 1.0, the LFE position included.
    assert!((read(ChannelLayout::Unknown, 3) - front).abs() < 1e-6);

    // Explicit roles; DualMono counts twice (+3.01 dB).
    let mut d = detector(fs, 1, ChannelLayout::Mono);
    assert!(d.set_channel_roles(&[ChannelRole::DualMono]).is_ok());
    let x = sine_on(fs as usize * 4, 1, 0, fs, 997.0, amp);
    d.process(&x);
    assert!((d.short_term_lufs() - (front + 10.0 * 2f64.log10())).abs() < 1e-6);
    let mut d2 = detector(fs, 2, ChannelLayout::Stereo);
    assert!(d2
        .set_channel_roles(&[ChannelRole::DualMono, ChannelRole::Front])
        .is_err());
    assert!(d2.set_channel_roles(&[ChannelRole::Front]).is_err());
}

#[test]
fn readings_do_not_depend_on_block_size() {
    let fs = 48_000.0;
    let sig = pink(1 << 17, 2, fs, 21);
    let mut whole = detector(fs, 2, ChannelLayout::Stereo);
    whole.process(&sig);
    for size in [1usize, 7, 64, 480, 1024, 4095] {
        let mut d = detector(fs, 2, ChannelLayout::Stereo);
        for b in sig.chunks(size * 2) {
            d.process(b);
        }
        assert_eq!(d.sub_blocks_completed(), whole.sub_blocks_completed());
        assert!((d.momentary_lufs() - whole.momentary_lufs()).abs() < 1e-9);
        assert!((d.short_term_lufs() - whole.short_term_lufs()).abs() < 1e-9);
    }
}

#[test]
fn running_sums_do_not_drift() {
    // 10 minutes of alternating loud noise, quiet noise and digital silence.
    let fs = 48_000.0;
    let loud = noise(1 << 16, 2, fs, 0.0, 31);
    let mut d = detector(fs, 2, ChannelLayout::Stereo);
    let mut looper = Looper::new(&loud);
    let block = 480;
    let mut buf = vec![0.0f32; block * 2];
    let total_blocks = (10.0 * 60.0 * fs) as usize / block;
    for i in 0..total_blocks {
        looper.fill(&mut buf);
        let g = match (i / 700) % 3 {
            0 => 3.0,
            1 => 1e-4,
            _ => 0.0,
        };
        buf.iter_mut().for_each(|v| *v *= g);
        d.process(&buf);
        if i % 97 == 0 {
            let exact_m = d.recent_mean_square(40);
            let exact_st = d.recent_mean_square(300);
            let rel = |a: f64, b: f64| (a - b).abs() / b.max(1e-30);
            assert!(rel(d.momentary_mean_square(), exact_m) < 1e-9 || exact_m == 0.0);
            assert!(rel(d.short_term_mean_square(), exact_st) < 1e-9 || exact_st == 0.0);
        }
    }
    // After 3 s of digital silence everything is exactly silent.
    let zeros = vec![0.0f32; (3.1 * fs) as usize * 2];
    d.process(&zeros);
    assert_eq!(d.momentary_lufs(), SILENCE_LUFS);
    assert_eq!(d.short_term_lufs(), SILENCE_LUFS);
}

#[test]
fn robust_against_non_finite_and_extreme_input() {
    for fs in [8_000.0, 48_000.0, 192_000.0] {
        let mut d = detector(fs, 2, ChannelLayout::Stereo);
        let mut x = vec![0.0f32; 4800 * 2];
        for (i, v) in x.iter_mut().enumerate() {
            *v = match i % 5 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::MAX,
                3 => -f32::MAX,
                _ => 1e-40,
            };
        }
        d.process(&x);
        assert!(d.momentary_lufs().is_finite() && d.short_term_lufs().is_finite());
        // Recovers to a correct reading afterwards.
        let amp = 0.1;
        let s = sine_on((4.0 * fs) as usize, 2, 0, fs, 997.0, amp);
        d.process(&s);
        let k = k_mag(&k_weighting(fs), 997.0, fs);
        let expected = lufs_from_mean_square(0.5 * amp * amp * k * k);
        let st = d.short_term_lufs();
        assert!((st - expected).abs() < 0.01, "fs {fs}: {st} vs {expected}");
        // Denormal input: stays finite, reads as (near) silence.
        d.reset();
        d.process(&vec![1e-40f32; 48_000]);
        assert!(d.short_term_lufs() <= -150.0);
    }
    // Unconfigured: a no-op.
    let mut d = LoudnessDetector::new();
    assert_eq!(d.process(&[1.0, 2.0]), 0);
    assert_eq!(d.short_term_lufs(), SILENCE_LUFS);
}

#[test]
fn process_and_readings_do_not_allocate() {
    let fs = 48_000.0;
    let sig = pink(1 << 15, 6, fs, 41);
    let mut d = detector(fs, 6, ChannelLayout::Surround51);
    let before = allocs();
    let mut sink = 0.0;
    for (i, b) in sig
        .chunks(6 * [1usize, 480, 7, 1024][0])
        .enumerate()
        .take(5000)
    {
        d.process(b);
        sink += d.momentary_lufs() + d.short_term_lufs() + d.recent_mean_square(10);
        if i % 1000 == 0 {
            d.set_layout(ChannelLayout::Unknown);
            let _ = d.set_channel_roles(&[ChannelRole::Front; 6]);
            d.reset();
        }
    }
    for size in [480usize, 7, 1024] {
        for b in sig.chunks(6 * size) {
            d.process(b);
            sink += d.short_term_lufs();
        }
    }
    assert_eq!(allocs(), before, "allocated in the processing path");
    assert!(sink.is_finite());
}
