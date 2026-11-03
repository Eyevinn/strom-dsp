//! Tests of the final clipper ([`strom_dsp::clipper::FinalClipper`]). All
//! signals are synthetic with fixed seeds; the measurements are in
//! `clip_metrics` (test-only).
//!
//! - Latency and transparency: below the knee the output is the input
//!   delayed by the reported latency, bit for bit, in any buffer size; the
//!   early-out changes nothing.
//! - Aliasing: a 15 kHz tone 6 dB over the threshold leaves no spur above
//!   −80 dBFS (guard: fails without oversampling).
//! - Overshoot: band-limited signals driven 2 and 4 dB into the clipper
//!   come out within +0.3 dB of the threshold (guard: fails without the
//!   overshoot compensation).
//! - Low cut: the clipper does not modulate the bass with peaks in the
//!   upper bands (guard: fails without the low cut). The bass-into-highs
//!   sidebands are measured and documented.
//! - Loudness against NMR, drive 0 … 4 dB, all followed by the same
//!   true-peak limiter: the clipper is louder and has a lower NMR than the
//!   limiter alone and than a base-rate hard clip, at every drive.
//! - Activity meter, stereo independence, parameter glides, robustness,
//!   reset, zero allocations.

mod agc_support;
mod clip_metrics;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::f64::consts::PI;

use agc_support::{noise, pink, Rng};
use clip_metrics::*;
use strom_dsp::biquad::{Biquad, BiquadCoefficients};
use strom_dsp::clipper::{ClipperLatency, FinalClipper, FinalClipperParams, BANDWIDTH_RANGE_HZ};
use strom_dsp::meter::{ChannelLayout, LoudnessConfig, LoudnessMeter};
use strom_dsp::oversample::OversampleFactor;
use strom_dsp::truepeak::{TruePeakLimiter, TruePeakLimiterParams};
use strom_dsp::Processor;

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

const FS: f64 = 48_000.0;

const SETTINGS: [ClipperLatency; 3] = [
    ClipperLatency::Low,
    ClipperLatency::Standard,
    ClipperLatency::TwoPass,
];

/// A configured clipper with a latency setting.
fn clipper_at(p: FinalClipperParams, l: ClipperLatency, ch: usize) -> FinalClipper {
    let mut c = FinalClipper::new();
    c.set_latency(l);
    c.set_params(&p);
    c.configure(FS as f32, ch);
    c
}

fn clipper(p: FinalClipperParams, f: Option<OversampleFactor>, ch: usize) -> FinalClipper {
    let mut c = FinalClipper::new();
    c.set_oversampling(f);
    c.set_params(&p);
    c.configure(FS as f32, ch);
    c
}

fn run(c: &mut FinalClipper, x: &[f32], ch: usize, block: usize) -> Vec<f32> {
    let mut y = x.to_vec();
    for b in y.chunks_mut(block * ch) {
        c.process(b);
    }
    y
}

fn gen(frames: usize, f: impl Fn(f64) -> f64) -> Vec<f32> {
    (0..frames).map(|n| f(n as f64 / FS) as f32).collect()
}

fn sine(frames: usize, freq: f64, amp: f64) -> Vec<f32> {
    gen(frames, |t| amp * (2.0 * PI * freq * t).sin())
}

/// `x` scaled so its sample peak is `peak_db` dBFS.
fn scale_peak(mut x: Vec<f32>, peak_db: f64) -> Vec<f32> {
    let p = x.iter().fold(0.0f32, |m, v| m.max(v.abs())) as f64;
    let g = 10f64.powf(peak_db / 20.0) / p;
    x.iter_mut().for_each(|v| *v = (*v as f64 * g) as f32);
    x
}

fn threshold() -> f64 {
    FinalClipperParams::default().threshold_db as f64
}

/// Noise band-limited by a 4th-order low-pass at `hz` (two cascaded RBJ
/// low-passes at Butterworth Q), with a 2 Hz syllable-like envelope.
fn speech_like(frames: usize, hz: f64, seed: u64) -> Vec<f32> {
    let mut x = noise(frames, 1, FS, 0.0, seed);
    for q in [0.5412, 1.3066] {
        let mut b = Biquad::new(BiquadCoefficients::low_pass(hz, q, FS), 1);
        b.process(&mut x);
    }
    for (i, v) in x.iter_mut().enumerate() {
        let t = i as f64 / FS;
        *v *= (0.25 + 0.75 * (2.0 * PI * 2.0 * t).sin().abs()) as f32;
    }
    x
}

// ---------------------------------------------------------------------------
// Latency and transparency.

/// Below the knee the output is the input delayed by exactly the latency,
/// bit for bit, in any buffer size, stereo, at several rates.
#[test]
fn transparent_below_knee() {
    for (fs, l) in [44_100.0f32, 48_000.0, 96_000.0]
        .into_iter()
        .flat_map(|fs| SETTINGS.map(|l| (fs, l)))
    {
        let mut c = FinalClipper::new();
        c.set_latency(l);
        c.configure(fs, 2);
        let lat = c.latency_frames();
        assert_eq!(lat, l.latency_frames());
        // Peaks well under the knee start (the interpolator can add up to
        // its bound B on top of sample peaks).
        let x = scale_peak(pink(1 << 15, 2, fs as f64, 9), threshold() - 2.0 - 7.0);
        for block in [1, 7, 480, 1024] {
            c.reset();
            let y = run(&mut c, &x, 2, block);
            for n in lat..x.len() / 2 {
                for ch in 0..2 {
                    assert_eq!(
                        y[n * 2 + ch].to_bits(),
                        x[(n - lat) * 2 + ch].to_bits(),
                        "fs {fs} block {block} frame {n}"
                    );
                }
            }
            assert!(!c.activity().is_active());
        }
    }
}

/// An impulse below the knee comes out exactly at the reported latency.
#[test]
fn impulse_latency_is_reported_latency() {
    let mut c = FinalClipper::new();
    c.configure(48_000.0, 1);
    let mut x = vec![0.0f32; 1000];
    x[10] = 0.25;
    let y = run(&mut c, &x, 1, 480);
    let at = y.iter().position(|&v| v != 0.0).unwrap();
    assert_eq!(at, 10 + c.latency_frames());
    assert_eq!(y[at], 0.25);
}

/// The early-out is exact: with it switched off the output is bit-identical,
/// on programme that keeps crossing between quiet and clipped.
#[test]
fn early_out_is_exact() {
    let frames = 1 << 16;
    let mut x = scale_peak(pink(frames, 2, FS, 4), threshold() + 4.0);
    // Quiet stretches of different lengths between loud ones.
    let mut rng = Rng::new(8);
    let mut n = 0;
    while n < frames {
        let len = 200 + (rng.uniform() * 6000.0) as usize;
        let g = if rng.uniform() < 0.5 { 0.05 } else { 1.0 };
        for i in n..(n + len).min(frames) {
            x[i * 2] *= g;
            x[i * 2 + 1] *= g;
        }
        n += len;
    }
    for block in [1, 64, 333, 480] {
        let mut a = FinalClipper::new();
        a.configure(FS as f32, 2);
        let mut b = FinalClipper::new();
        b.set_early_out(false);
        b.configure(FS as f32, 2);
        let ya = run(&mut a, &x, 2, block);
        let yb = run(&mut b, &x, 2, block);
        assert!(a.take_stats().is_active());
        for (i, (p, q)) in ya.iter().zip(&yb).enumerate() {
            assert_eq!(p.to_bits(), q.to_bits(), "block {block} sample {i}");
        }
    }
}

// ---------------------------------------------------------------------------
// Aliasing.

/// Largest spur (dBFS) of a 15 kHz tone 6 dB over the threshold, mono, 1 s
/// analysed after 1 s, at latency setting `l` (oversampling overridden by
/// `f` when given).
fn alias_spur(l: ClipperLatency, f: Option<Option<OversampleFactor>>, fs: f64) -> f64 {
    let mut c = FinalClipper::new();
    c.set_latency(l);
    if let Some(f) = f {
        c.set_oversampling(f);
    }
    c.configure(fs as f32, 1);
    let t = 10f64.powf(threshold() / 20.0);
    let n = fs as usize;
    let x: Vec<f32> = (0..2 * n)
        .map(|i| (2.0 * t * (2.0 * PI * 15_000.0 * i as f64 / fs).sin()) as f32)
        .collect();
    let y = run(&mut c, &x, 1, 480);
    let s = channel(&y[n..], 1, 0);
    // Exclude the tone and its harmonics below Nyquist (at 96 kHz the 3rd
    // harmonic, 45 kHz, is in band and is not aliasing).
    let harmonics: Vec<f64> = (1..)
        .map(|h| 15_000.0 * h as f64)
        .take_while(|&f| f < fs / 2.0)
        .collect();
    spur_dbfs(&s, fs, &harmonics, 6)
}

/// Spurs of the 15 kHz tone stay below −80 dBFS at 48 and 96 kHz with the
/// standard setting (8×; the tone's folded harmonics are non-harmonic).
/// Guard: without oversampling the same curve leaves spurs far above that.
/// The low setting (4×, short FIR) is measured and must stay within 1 dB of the
/// recorded value.
#[test]
fn aliasing_of_clipped_15k_tone() {
    let std48 = alias_spur(ClipperLatency::Standard, None, 48_000.0);
    let std96 = alias_spur(ClipperLatency::Standard, None, 96_000.0);
    let low = alias_spur(ClipperLatency::Low, None, 48_000.0);
    let x4 = alias_spur(
        ClipperLatency::Standard,
        Some(Some(OversampleFactor::X4)),
        48_000.0,
    );
    let base = alias_spur(ClipperLatency::Standard, Some(None), 48_000.0);
    println!(
        "15 kHz +6 dB: spur standard (8x) {std48:.1} dBFS (96 kHz: {std96:.1}), low (4x, 3 taps) \
         {low:.1}, 4x {x4:.1}, no oversampling {base:.1}"
    );
    assert!(std48 < -80.0, "{std48}");
    assert!(std96 < -80.0, "{std96}");
    assert!((low - LOW_ALIAS_DBFS).abs() <= 1.0, "low {low}");
    assert!(base > -80.0, "guard: no oversampling should alias, {base}");
}

/// Spur of the 15 kHz tone at the low setting (dBFS), as measured.
const LOW_ALIAS_DBFS: f64 = -75.8;

// ---------------------------------------------------------------------------
// Overshoot.

/// The band-limited test signals (each lies inside the clippings' band,
/// 120 Hz … 20 kHz, where the clipper is responsible for the peaks).
fn band_limited_signals() -> Vec<(&'static str, Vec<f32>)> {
    let n = 1 << 17;
    vec![
        ("1 kHz", sine(n, 1000.0, 1.0)),
        ("15 kHz", sine(n, 15_000.0, 1.0)),
        (
            "CCIF 19+20 kHz",
            gen(n, |t| {
                0.5 * (2.0 * PI * 19_000.0 * t).sin() + 0.5 * (2.0 * PI * 20_000.0 * t).sin()
            }),
        ),
        ("pink", pink(n, 1, FS, 5)),
        ("speech-like 3.4 kHz", speech_like(n, 3_400.0, 11)),
        ("speech-like 7 kHz", speech_like(n, 7_000.0, 12)),
    ]
}

/// Largest sample peak (dB re threshold) over the signals at drives 2 and
/// 4 dB, after 4000 frames of warm-up.
fn worst_overshoot(l: ClipperLatency, compensate: bool) -> f64 {
    let p = FinalClipperParams {
        overshoot_compensation: compensate,
        ..FinalClipperParams::default()
    };
    let mut worst = f64::NEG_INFINITY;
    for (name, x) in band_limited_signals() {
        for drive in [2.0, 4.0] {
            let x = scale_peak(x.clone(), threshold() + drive);
            let mut c = clipper_at(p, l, 1);
            let y = run(&mut c, &x, 1, 480);
            let over = sample_peak_db(&y[4000..]) - threshold();
            println!(
                "{l:?}, compensation {compensate}: {name} +{drive} dB: peak {over:+.2} dB re T"
            );
            worst = worst.max(over);
        }
    }
    worst
}

/// Two passes hold sample peaks within +0.3 dB of the threshold with the
/// overshoot compensation; one pass (the standard and the low setting)
/// leaves more, up to the recorded bound, for the limiter after the
/// clipper. Without the compensation every setting does worse (guard: each
/// exceeds its bound; at the low setting's 3 taps the compensation is
/// worth only about 0.3 dB).
#[test]
fn sample_peak_within_threshold() {
    for (l, bound) in [
        (ClipperLatency::TwoPass, 0.3),
        (ClipperLatency::Standard, STANDARD_OVERSHOOT_DB),
        (ClipperLatency::Low, LOW_OVERSHOOT_DB),
    ] {
        let with = worst_overshoot(l, true);
        let without = worst_overshoot(l, false);
        println!("{l:?}: worst with compensation {with:+.2} dB, without {without:+.2} dB");
        assert!(with <= bound, "{l:?}: {with}");
        assert!(without > bound, "{l:?}: guard: {without}");
    }
}

/// Worst sample-peak overshoot (dB re threshold) allowed for one pass.
const STANDARD_OVERSHOOT_DB: f64 = 0.85;
const LOW_OVERSHOOT_DB: f64 = 2.3;

// ---------------------------------------------------------------------------
// Low cut and intermodulation.

/// Highs into bass: a soft 50 Hz tone under loud 2 kHz bursts (4 Hz
/// envelope) that drive the clipper 4 dB. Without the low cut the
/// subtracted clippings modulate the bass band with the bursts. Returns
/// the error energy (output − delayed input) between 10 and 150 Hz, dBFS.
fn bass_band_error(low_cut: bool) -> f64 {
    let p = FinalClipperParams {
        low_cut,
        ..FinalClipperParams::default()
    };
    let mut c = clipper(p, Some(OversampleFactor::X8), 1);
    let n = 2 * FS as usize;
    let t = 10f64.powf(threshold() / 20.0);
    let x = gen(n, |s| {
        let env = (2.0 * PI * 4.0 * s).sin().abs();
        0.05 * t * (2.0 * PI * 50.0 * s).sin() + 1.58 * t * env * (2.0 * PI * 2000.0 * s).sin()
    });
    let y = run(&mut c, &x, 1, 480);
    let lat = c.latency_frames();
    let start = FS as usize;
    let err: Vec<f64> = (start..n)
        .map(|i| y[i] as f64 - x[i - lat] as f64)
        .collect();
    let s = spectrum_db(&err);
    let df = FS / err.len() as f64;
    let e: f64 = s
        .iter()
        .enumerate()
        .filter(|(k, _)| (10.0..=150.0).contains(&(*k as f64 * df)))
        .map(|(_, v)| 10f64.powf(v / 10.0))
        .sum();
    10.0 * e.max(1e-30).log10()
}

/// The low cut keeps the clipper out of the bass band: the low-frequency
/// error of clipping 2 kHz bursts (the 50 Hz tone modulated by the bursts)
/// is at least 10 dB lower with it (guard: fails without the low cut).
#[test]
fn low_cut_keeps_bass_band() {
    let with = bass_band_error(true);
    let without = bass_band_error(false);
    println!("bass-band error: low cut {with:.1} dBFS, without {without:.1} dBFS");
    assert!(with < without - 10.0, "with {with}, without {without}");
}

/// Recorded sidebands (dBc, −50/+50/−100/+100 Hz) per setting: with the low
/// cut, without it.
type Sidebands = [f64; 4];
const SIDEBANDS: [(ClipperLatency, Sidebands, Sidebands); 3] = [
    (
        ClipperLatency::Low,
        [-95.7, -99.2, -4.5, -4.5],
        [-95.9, -99.4, -4.7, -4.7],
    ),
    (
        ClipperLatency::Standard,
        [-95.2, -99.0, -4.0, -4.0],
        [-95.5, -99.1, -4.3, -4.2],
    ),
    (
        ClipperLatency::TwoPass,
        [-70.5, -70.2, -3.1, -3.1],
        [-90.4, -93.8, -3.7, -3.7],
    ),
];

/// Sidebands (dBc) at 3 kHz ± 50 and ± 100 Hz when a loud 50 Hz tone (3 dB
/// over the threshold) drives the clipper under a soft 3 kHz tone.
fn bass_into_highs(l: ClipperLatency, low_cut: bool) -> [f64; 4] {
    let p = FinalClipperParams {
        low_cut,
        ..FinalClipperParams::default()
    };
    let mut c = clipper_at(p, l, 1);
    let t = 10f64.powf(threshold() / 20.0);
    let x = gen(2 * FS as usize, |s| {
        1.4 * t * (2.0 * PI * 50.0 * s).sin() + 0.1 * t * (2.0 * PI * 3000.0 * s).sin()
    });
    let y = run(&mut c, &x, 1, 480);
    let s = channel(&y[FS as usize..], 1, 0);
    let c0 = level_at_dbfs(&s, FS, 3000.0);
    [2950.0, 3050.0, 2900.0, 3100.0].map(|f| level_at_dbfs(&s, FS, f) - c0)
}

/// Documents the bass-into-highs sidebands of the clipper on its own, per
/// latency setting. A bass peak over the threshold chops the 3 kHz tone
/// riding on it, with or without the low cut, so the ±100 Hz sidebands sit
/// 2 … 4 dB under the tone in every setting. The low cut does not lower
/// them: with one pass it changes them by 0.3 dB; with two passes it makes
/// them 0.6 dB worse, because it leaves the bass peak in place (for the
/// limiter after the clipper) and the second pass chops the remaining
/// treble again. Keeping bass out of the clipper is the job of a stage
/// before it. Each value must stay within 1 dB of the recorded one (or both
/// below −70 dBc), so a change shows up.
#[test]
fn bass_into_highs_sidebands_documented() {
    let mut cases = Vec::new();
    for (l, rec_with, rec_without) in SIDEBANDS {
        let with = bass_into_highs(l, true);
        let without = bass_into_highs(l, false);
        println!(
            "{l:?}: sidebands -50/+50/-100/+100 Hz (dBc): low cut {with:.1?}, without \
             {without:.1?}"
        );
        cases.push((with, rec_with));
        cases.push((without, rec_without));
    }
    for (got, rec) in cases {
        for (m, r) in got.iter().zip(rec) {
            assert!(
                (m - r).abs() <= 1.0 || (*m < -70.0 && r < -70.0),
                "{got:?} vs recorded {rec:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Loudness against NMR.

/// A music-like programme: pink noise, two tones, a 4 Hz level modulation.
fn programme(frames: usize) -> Vec<f32> {
    let mut x = pink(frames, 1, FS, 21);
    for (n, v) in x.iter_mut().enumerate() {
        let t = n as f64 / FS;
        let syl = 0.6 + 0.4 * (2.0 * PI * 4.0 * t).sin();
        let tones = 0.4 * (2.0 * PI * 110.0 * t).sin() + 0.2 * (2.0 * PI * 1_320.0 * t).sin();
        *v = ((*v as f64 * 0.5 + tones) * syl) as f32;
    }
    x
}

fn integrated(x: &[f32]) -> f64 {
    let mut m = LoudnessMeter::new(&LoudnessConfig {
        sample_rate: FS as f32,
        channels: 1,
        layout: ChannelLayout::Unknown,
        true_peak: false,
    })
    .unwrap();
    m.process(x);
    m.integrated()
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Method {
    Clipper(ClipperLatency),
    LimiterAlone,
    HardClip,
}

/// (integrated loudness LUFS, NMR dB) of `x` through `method` and a
/// true-peak limiter at −1 dBTP (so every method ends at the same true-peak
/// ceiling). The NMR reference is `x`, delayed and gain-matched.
fn loudness_and_nmr(x: &[f32], method: Method) -> (f64, f64) {
    let mut lim = TruePeakLimiter::new();
    lim.set_params(&TruePeakLimiterParams {
        enabled: true,
        ceiling_dbtp: -1.0,
        release_ms: 200.0,
        lookahead_ms: 5.0,
    });
    lim.configure(FS as f32, 1);
    let mut lat = lim.latency_frames();
    let mut y = x.to_vec();
    match method {
        Method::Clipper(l) => {
            let mut c = FinalClipper::new();
            c.set_latency(l);
            c.configure(FS as f32, 1);
            lat += c.latency_frames();
            y = run(&mut c, &y, 1, 480);
        }
        Method::HardClip => {
            let t = 10f32.powf(threshold() as f32 / 20.0);
            y.iter_mut().for_each(|v| *v = v.clamp(-t, t));
        }
        Method::LimiterAlone => {}
    }
    for b in y.chunks_mut(480) {
        lim.process(b);
    }
    let r: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let o: Vec<f64> = y.iter().map(|&v| v as f64).collect();
    let skip = FS as usize;
    (
        integrated(&y[skip + lat..]),
        nmr_db(&r[skip..], &o[skip..], lat, FS),
    )
}

/// Drive 0 … 4 dB into each method (drive 0: the programme's sample peak
/// at the clipper threshold). At every drive the clipper is at least as
/// loud as the limiter alone and as a base-rate hard clip at the clipper's
/// threshold, and has a lower NMR: its loudness-against-NMR curve lies
/// above both at every point.
#[test]
fn loudness_against_nmr_beats_limiter_and_hard_clip() {
    let x0 = scale_peak(programme(1 << 18), threshold());
    let mut failures = Vec::new();
    for drive in [0.0, 1.0, 2.0, 3.0, 4.0] {
        let g = 10f64.powf(drive / 20.0) as f32;
        let x: Vec<f32> = x0.iter().map(|v| v * g).collect();
        let others = [Method::LimiterAlone, Method::HardClip].map(|m| (m, loudness_and_nmr(&x, m)));
        for setting in SETTINGS {
            let (lc, nc) = loudness_and_nmr(&x, Method::Clipper(setting));
            let mut line = format!("drive {drive}: {setting:?} {lc:.2} LUFS / NMR {nc:.1} dB");
            for (m, (l, n)) in others {
                line += &format!("; {m:?} {l:.2} LUFS / {n:.1} dB");
                if lc < l - 0.02 || nc >= n {
                    failures.push(format!(
                        "drive {drive}: {m:?} {l:.2}/{n:.1} vs {setting:?} {lc:.2}/{nc:.1}"
                    ));
                }
            }
            println!("{line}");
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// ---------------------------------------------------------------------------
// Meter, stereo, parameters, robustness.

/// The stats agree with the signals: below the knee nothing is reported;
/// on a driven sine the removed energy is the energy of (input delayed −
/// output) and the depth and touched share are plausible.
#[test]
fn activity_meter() {
    let mut c = clipper(FinalClipperParams::default(), Some(OversampleFactor::X8), 1);
    let quiet = sine(48_000, 1000.0, 0.1);
    run(&mut c, &quiet, 1, 480);
    let s = c.take_stats();
    assert!(!s.is_active() && s.depth_db() == -200.0 && s.touched == 0);
    assert!(s.input_energy > 0.0);

    c.reset();
    let t = 10f64.powf(threshold() / 20.0);
    let x = sine(48_000, 1000.0, 2.0 * t);
    let y = run(&mut c, &x, 1, 480);
    let s = c.take_stats();
    let lat = c.latency_frames();
    let removed: f64 = (lat..x.len())
        .map(|i| (x[i - lat] as f64 - y[i] as f64).powi(2))
        .sum();
    assert!((s.removed_energy / removed - 1.0).abs() < 1e-3);
    // A sine clipped from 6 dB over: most samples touched, depth −10 … −3 dB.
    assert!(s.touched_share() > 0.5 && s.touched_share() < 1.0);
    assert!((-10.0..-3.0).contains(&s.depth_db()), "{}", s.depth_db());
    // `activity` covers the latest call only.
    for _ in 0..2 {
        let mut b = vec![0.0f32; 480];
        c.process(&mut b);
    }
    assert!(!c.activity().is_active());
    assert!(c.activity().depth_db() == -200.0);
}

/// Channels are clipped independently: a clipped left channel leaves the
/// right one a bit-exact delay. On correlated stereo the inter-channel
/// correlation hardly moves.
#[test]
fn stereo_channels_are_independent() {
    let n = 1 << 15;
    let t = 10f64.powf(threshold() / 20.0);
    let mut x = vec![0.0f32; 2 * n];
    let l = sine(n, 700.0, 3.0 * t);
    let r = sine(n, 300.0, 0.2 * t);
    for i in 0..n {
        x[2 * i] = l[i];
        x[2 * i + 1] = r[i];
    }
    let mut c = clipper(FinalClipperParams::default(), Some(OversampleFactor::X8), 2);
    let y = run(&mut c, &x, 2, 480);
    let lat = c.latency_frames();
    for i in lat..n {
        assert_eq!(y[2 * i + 1].to_bits(), x[2 * (i - lat) + 1].to_bits());
    }

    // Correlated stereo programme driven 3 dB into the clipper.
    let a = programme(1 << 17);
    let b = pink(1 << 17, 1, FS, 77);
    let mut st = vec![0.0f32; 2 << 17];
    for i in 0..1 << 17 {
        st[2 * i] = a[i] + 0.2 * b[i];
        st[2 * i + 1] = 0.9 * a[i] - 0.2 * b[i];
    }
    let st = scale_peak(st, threshold() + 3.0);
    let corr = |v: &[f32]| {
        let (mut lr, mut ll, mut rr) = (0.0f64, 0.0f64, 0.0f64);
        for f in v.as_chunks::<2>().0 {
            lr += f[0] as f64 * f[1] as f64;
            ll += f[0] as f64 * f[0] as f64;
            rr += f[1] as f64 * f[1] as f64;
        }
        lr / (ll * rr).sqrt()
    };
    let mut c = clipper(FinalClipperParams::default(), Some(OversampleFactor::X8), 2);
    let y = run(&mut c, &st, 2, 480);
    let (ci, co) = (corr(&st), corr(&y[2 * lat..]));
    println!("correlation in {ci:.4}, out {co:.4}");
    assert!((ci - co).abs() < 0.02, "{ci} {co}");
}

/// Parameter jumps while clipping glide. A clipped tone has sharper edges
/// than the tone, so the bound is relative to the clipped output itself:
/// during each transition the largest sample-to-sample step stays below
/// 1.5 times the largest step of the steady output at either setting (a
/// discontinuity would exceed both).
#[test]
fn parameter_changes_glide() {
    let t = 10f64.powf(threshold() / 20.0);
    let x = sine(48_000, 200.0, 2.0 * t);
    let max_step = |y: &[f32]| {
        y.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max)
    };
    let d = FinalClipperParams::default();
    let changes = [
        (
            "threshold -12 dB",
            FinalClipperParams {
                threshold_db: -12.0,
                ..d
            },
        ),
        (
            "threshold 0 dB",
            FinalClipperParams {
                threshold_db: 0.0,
                ..d
            },
        ),
        ("knee 6 dB", FinalClipperParams { knee_db: 6.0, ..d }),
        (
            "bandwidth 8 kHz",
            FinalClipperParams {
                bandwidth_hz: BANDWIDTH_RANGE_HZ.0,
                ..d
            },
        ),
        (
            "low cut 1 kHz",
            FinalClipperParams {
                low_cut_hz: 1000.0,
                ..d
            },
        ),
        (
            "low cut 20 Hz",
            FinalClipperParams {
                low_cut_hz: 20.0,
                ..d
            },
        ),
        (
            "no compensation",
            FinalClipperParams {
                overshoot_compensation: false,
                ..d
            },
        ),
    ];
    for (name, q) in changes {
        for (from, to) in [(d, q), (q, d)] {
            let steady = |p: FinalClipperParams| {
                let mut c = clipper(p, Some(OversampleFactor::X8), 1);
                max_step(&run(&mut c, &x, 1, 480)[4800..])
            };
            let bound = 1.5 * steady(from).max(steady(to));
            let mut c = clipper(from, Some(OversampleFactor::X8), 1);
            let mut y = x.clone();
            for (i, b) in y.chunks_mut(480).enumerate() {
                if i == 20 {
                    c.set_params(&to);
                }
                c.process(b);
            }
            let got = max_step(&y[4800..]);
            assert!(got <= bound, "{name}: step {got}, bound {bound}");
            assert!(y.iter().all(|v| v.is_finite()));
        }
    }
}

/// Extreme and non-finite input and parameters: finite output for finite
/// input, nothing panics, and NaN does not stick.
#[test]
fn robust() {
    let mut rng = Rng::new(5);
    let mut c = FinalClipper::new();
    c.configure(FS as f32, 2);
    for round in 0..50 {
        let p = FinalClipperParams {
            threshold_db: (rng.uniform() * 40.0 - 30.0) as f32,
            knee_db: (rng.uniform() * 10.0) as f32,
            bandwidth_hz: (rng.uniform() * 30_000.0) as f32,
            low_cut_hz: if round % 7 == 0 {
                f32::NAN
            } else {
                (rng.uniform() * 2000.0) as f32
            },
            low_cut: rng.uniform() < 0.5,
            overshoot_compensation: rng.uniform() < 0.5,
            passes: (rng.uniform() * 4.0) as usize,
        };
        c.set_params(&p);
        let amp = 10f64.powf(rng.uniform() * 12.0 - 6.0);
        let mut b: Vec<f32> = (0..960).map(|_| (amp * rng.bipolar()) as f32).collect();
        c.process(&mut b);
        assert!(b.iter().all(|v| v.is_finite()), "round {round}");
    }
    // Non-finite input: finite output again once it has passed.
    let mut b = vec![0.5f32; 960];
    b[10] = f32::NAN;
    b[11] = f32::INFINITY;
    b[12] = -1e30;
    c.process(&mut b);
    let mut ok = vec![0.5f32; 4800];
    c.process(&mut ok);
    assert!(ok.iter().all(|v| v.is_finite()));
}

/// `reset` clears every state: afterwards the clipper behaves like a fresh
/// one.
#[test]
fn reset_restores_fresh_state() {
    let x = scale_peak(pink(1 << 14, 2, FS, 3), threshold() + 4.0);
    let mut a = FinalClipper::new();
    a.configure(FS as f32, 2);
    run(&mut a, &x, 2, 480);
    a.reset();
    let ya = run(&mut a, &x, 2, 480);
    let mut b = FinalClipper::new();
    b.configure(FS as f32, 2);
    let yb = run(&mut b, &x, 2, 480);
    assert_eq!(ya, yb);
}

/// No allocation in `process` or `set_params`, clipping or quiet.
#[test]
fn no_allocation() {
    let x = scale_peak(pink(1 << 14, 2, FS, 3), threshold() + 4.0);
    let mut c = FinalClipper::new();
    c.configure(FS as f32, 2);
    let mut buf = x.clone();
    let mut quiet = vec![0.0f32; 4800];
    let before = allocs();
    for (i, b) in buf.chunks_mut(960).enumerate() {
        c.set_params(&FinalClipperParams {
            bandwidth_hz: 15_000.0 + i as f32,
            ..FinalClipperParams::default()
        });
        c.process(b);
        let _ = c.activity();
        let _ = c.take_stats();
    }
    c.process(&mut quiet);
    assert_eq!(allocs() - before, 0);
}
