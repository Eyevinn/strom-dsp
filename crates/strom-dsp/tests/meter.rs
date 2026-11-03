//! Meter tests: EBU Tech 3341 / 3342 minimum-requirements signals (generated
//! in code from the published descriptions), sample and true peak, channel
//! weights, correlation, the mono detector and real-time safety.
//!
//! Sources of the expected values:
//! - EBU Tech 3341 (v4, November 2023), Table 1 "Minimum requirements test
//!   signals". Cases 1–6, 9–19 are generated here; 7, 8 (authentic programme)
//!   and 20–23 (synthesised at 4·fs, then low-pass filtered and decimated) are
//!   not.
//! - EBU Tech 3342 (2023), Table 1, cases 1–4 (5, 6 are authentic programme).
//!
//! "Per-channel peak level" in both documents is the sine amplitude in dBFS.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::f64::consts::PI;

use strom_dsp::meter::{
    ChannelLayout, ChannelRole, CorrelationMeter, LoudnessConfig, LoudnessMeter, MeterError,
    MonoDetector, MonoDetectorParams, StereoState, MAX_LOUDNESS_CHANNELS,
};

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
// Signal helpers.

const FS: f64 = 48_000.0;
/// Block size used to feed the meters (10 ms at 48 kHz, as a live meter would).
const BLOCK: usize = 480;

fn db(level_db: f64) -> f64 {
    10f64.powf(level_db / 20.0)
}

/// Stereo 1 kHz sine generator with continuous phase across segments.
struct Tone {
    n: u64,
}

impl Tone {
    fn new() -> Self {
        Self { n: 0 }
    }

    /// `secs` of stereo (identical, in phase) 1 kHz sine at `level_db` dBFS
    /// per-channel peak; `None` = silence.
    fn segment(&mut self, out: &mut Vec<f32>, level_db: Option<f64>, secs: f64) {
        let frames = (secs * FS).round() as u64;
        for _ in 0..frames {
            let v = match level_db {
                Some(l) => (db(l) * (2.0 * PI * 1000.0 * self.n as f64 / FS).sin()) as f32,
                None => 0.0,
            };
            out.push(v);
            out.push(v);
            self.n += 1;
        }
    }
}

fn stereo_meter(true_peak: bool) -> LoudnessMeter {
    LoudnessMeter::new(&LoudnessConfig {
        sample_rate: FS as f32,
        channels: 2,
        layout: ChannelLayout::Stereo,
        true_peak,
    })
    .unwrap()
}

/// Feed `signal` (interleaved, `ch` channels) in `BLOCK`-frame blocks and call
/// `after` after every block with the frame count so far.
fn feed(
    m: &mut LoudnessMeter,
    signal: &[f32],
    ch: usize,
    mut after: impl FnMut(&LoudnessMeter, usize),
) {
    let mut frames = 0;
    for b in signal.chunks(BLOCK * ch) {
        m.process(b);
        frames += b.len() / ch;
        after(m, frames);
    }
}

fn assert_near(what: &str, got: f64, want: f64, tol: f64) {
    // Visible with `--nocapture`, to see the margins.
    eprintln!("{what}: {got:.4} (want {want:.4} ± {tol})");
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got:.4}, want {want:.4} ± {tol}"
    );
}

// ---------------------------------------------------------------------------
// EBU Tech 3341, cases 1–6: stationary and stepped sines.

/// Run a stepped stereo tone through a fresh meter and return it.
fn run_steps(steps: &[(Option<f64>, f64)]) -> LoudnessMeter {
    let mut sig = Vec::new();
    let mut t = Tone::new();
    for &(l, s) in steps {
        t.segment(&mut sig, l, s);
    }
    let mut m = stereo_meter(false);
    feed(&mut m, &sig, 2, |_, _| {});
    m
}

#[test]
fn tech3341_case1_and_2_stationary_sine() {
    for level in [-23.0, -33.0] {
        let m = run_steps(&[(Some(level), 20.0)]);
        assert_near("M", m.momentary(), level, 0.1);
        assert_near("S", m.short_term(), level, 0.1);
        assert_near("I", m.integrated(), level, 0.1);
        // Relative scale: LU = LUFS + 23.
        assert_near("I (LU)", m.integrated() + 23.0, level + 23.0, 0.1);
    }
}

#[test]
fn tech3341_case3_gating_three_tones() {
    let m = run_steps(&[
        (Some(-36.0), 10.0),
        (Some(-23.0), 60.0),
        (Some(-36.0), 10.0),
    ]);
    assert_near("I", m.integrated(), -23.0, 0.1);
}

#[test]
fn tech3341_case4_gating_five_tones() {
    let m = run_steps(&[
        (Some(-72.0), 10.0),
        (Some(-36.0), 10.0),
        (Some(-23.0), 60.0),
        (Some(-36.0), 10.0),
        (Some(-72.0), 10.0),
    ]);
    assert_near("I", m.integrated(), -23.0, 0.1);
}

#[test]
fn tech3341_case5_relative_gate() {
    let m = run_steps(&[
        (Some(-26.0), 20.0),
        (Some(-20.0), 20.1),
        (Some(-26.0), 20.0),
    ]);
    assert_near("I", m.integrated(), -23.0, 0.1);
}

/// 5.0 (or 5.1 with a loud LFE) 1 kHz sine, 20 s, levels per Tech 3341 case 6.
fn surround_case6(layout: ChannelLayout) -> LoudnessMeter {
    let ch = layout.channels().unwrap();
    // Peak level per channel (dBFS) in L R C [LFE] Ls Rs order.
    let levels: Vec<f64> = match layout {
        ChannelLayout::Surround50 => vec![-28.0, -28.0, -24.0, -30.0, -30.0],
        // The LFE carries a loud tone that must not count.
        ChannelLayout::Surround51 => vec![-28.0, -28.0, -24.0, -6.0, -30.0, -30.0],
        _ => unreachable!(),
    };
    let frames = (20.0 * FS) as usize;
    let mut sig = Vec::with_capacity(frames * ch);
    for n in 0..frames {
        let s = (2.0 * PI * 1000.0 * n as f64 / FS).sin();
        for l in &levels {
            sig.push((db(*l) * s) as f32);
        }
    }
    let mut m = LoudnessMeter::new(&LoudnessConfig {
        sample_rate: FS as f32,
        channels: ch,
        layout,
        true_peak: false,
    })
    .unwrap();
    feed(&mut m, &sig, ch, |_, _| {});
    m
}

#[test]
fn tech3341_case6_surround_weights() {
    assert_near(
        "I 5.0",
        surround_case6(ChannelLayout::Surround50).integrated(),
        -23.0,
        0.1,
    );
    // Same programme in 5.1 with an LFE at -6 dBFS: the LFE is excluded.
    assert_near(
        "I 5.1",
        surround_case6(ChannelLayout::Surround51).integrated(),
        -23.0,
        0.1,
    );
}

// ---------------------------------------------------------------------------
// EBU Tech 3341, cases 9–14: constant readings and maximum M / S.

#[test]
fn tech3341_case9_short_term_constant_after_3s() {
    let mut sig = Vec::new();
    let mut t = Tone::new();
    for _ in 0..5 {
        t.segment(&mut sig, Some(-20.0), 1.34);
        t.segment(&mut sig, Some(-30.0), 1.66);
    }
    let mut m = stereo_meter(false);
    let mut checked = 0;
    feed(&mut m, &sig, 2, |m, frames| {
        if frames as f64 >= 3.0 * FS {
            assert_near("S", m.short_term(), -23.0, 0.1);
            checked += 1;
        }
    });
    assert!(checked > 1000);
}

#[test]
fn tech3341_case12_momentary_constant_after_1s() {
    let mut sig = Vec::new();
    let mut t = Tone::new();
    for _ in 0..25 {
        t.segment(&mut sig, Some(-20.0), 0.18);
        t.segment(&mut sig, Some(-30.0), 0.22);
    }
    let mut m = stereo_meter(false);
    let mut checked = 0;
    feed(&mut m, &sig, 2, |m, frames| {
        if frames as f64 >= FS {
            assert_near("M", m.momentary(), -23.0, 0.1);
            checked += 1;
        }
    });
    assert!(checked > 800);
}

/// Cases 10 and 13 (file-based: a fresh measurement per segment).
/// `window` selects S (3 s tone) or M (400 ms tone); `gap` is the per-index
/// leading silence (0.15 s or 20 ms).
fn file_based_max(short_term: bool) {
    let (tone_s, gap) = if short_term { (3.0, 0.15) } else { (0.4, 0.02) };
    for i in 0..20 {
        let mut sig = Vec::new();
        let mut t = Tone::new();
        t.segment(&mut sig, None, i as f64 * gap);
        t.segment(&mut sig, Some(-23.0), tone_s);
        t.segment(&mut sig, None, 1.0);
        let mut m = stereo_meter(false);
        let mut max = f64::NEG_INFINITY;
        feed(&mut m, &sig, 2, |m, _| {
            let v = if short_term {
                m.short_term()
            } else {
                m.momentary()
            };
            max = max.max(v);
        });
        assert_near(&format!("max segment {i}"), max, -23.0, 0.1);
    }
}

#[test]
fn tech3341_case10_max_short_term_per_file() {
    file_based_max(true);
}

#[test]
fn tech3341_case13_max_momentary_per_file() {
    file_based_max(false);
}

/// Cases 11 and 14 (live meter: one long signal, 20 successive maxima).
fn live_successive_max(short_term: bool) {
    let (tone_s, gap) = if short_term { (3.0, 0.15) } else { (0.4, 0.02) };
    let mut sig = Vec::new();
    let mut t = Tone::new();
    let mut bounds = Vec::new(); // segment end frame
    for i in 0..20 {
        t.segment(&mut sig, None, i as f64 * gap);
        t.segment(&mut sig, Some(-38.0 + i as f64), tone_s);
        t.segment(&mut sig, None, tone_s - i as f64 * gap);
        bounds.push(sig.len() / 2);
    }
    let mut m = stereo_meter(false);
    let mut maxima = [f64::NEG_INFINITY; 20];
    let mut seg = 0;
    feed(&mut m, &sig, 2, |m, frames| {
        let v = if short_term {
            m.short_term()
        } else {
            m.momentary()
        };
        maxima[seg] = maxima[seg].max(v);
        if seg < 19 && frames >= bounds[seg] {
            seg += 1;
        }
    });
    for (i, &mx) in maxima.iter().enumerate() {
        assert_near(&format!("max segment {i}"), mx, -38.0 + i as f64, 0.1);
    }
}

#[test]
fn tech3341_case11_live_successive_max_short_term() {
    live_successive_max(true);
}

#[test]
fn tech3341_case14_live_successive_max_momentary() {
    live_successive_max(false);
}

// ---------------------------------------------------------------------------
// EBU Tech 3341, cases 15–19: true peak. Tolerance +0.2 / −0.4 dB.

/// Stereo sine at `fs/div` with amplitude `amp` and phase `phase_deg`, 1 s,
/// with 10 ms raised-cosine fade-in and fade-out.
fn tp_signal(div: f64, amp: f64, phase_deg: f64) -> Vec<f32> {
    let frames = FS as usize;
    let fade = (0.01 * FS) as usize;
    let mut sig = Vec::with_capacity(2 * frames);
    for n in 0..frames {
        let env = if n < fade {
            0.5 - 0.5 * (PI * n as f64 / fade as f64).cos()
        } else if n >= frames - fade {
            0.5 - 0.5 * (PI * (frames - 1 - n) as f64 / fade as f64).cos()
        } else {
            1.0
        };
        let v = (env * amp * (2.0 * PI * n as f64 / div + phase_deg.to_radians()).sin()) as f32;
        sig.push(v);
        sig.push(v);
    }
    sig
}

fn true_peak_dbtp(sig: &[f32]) -> f64 {
    let mut m = stereo_meter(true);
    feed(&mut m, sig, 2, |_, _| {});
    let tp = m.true_peak(0).max(m.true_peak(1));
    assert_eq!(m.true_peak(0), m.true_peak(1));
    20.0 * tp.log10()
}

#[test]
fn tech3341_cases15_to_19_true_peak() {
    let cases = [
        (15, 4.0, 0.50, 0.0, -6.0),
        (16, 4.0, 0.50, 45.0, -6.0),
        (17, 6.0, 0.50, 60.0, -6.0),
        (18, 8.0, 0.50, 67.5, -6.0),
        (19, 4.0, 1.41, 45.0, 3.0),
    ];
    for (case, div, amp, phase, want) in cases {
        let got = true_peak_dbtp(&tp_signal(div, amp, phase));
        eprintln!("true peak case {case}: {got:.3} dBTP (want {want})");
        assert!(
            got <= want + 0.2 && got >= want - 0.4,
            "case {case}: true peak {got:.3} dBTP, want {want} +0.2/−0.4"
        );
    }
}

#[test]
fn intersample_peak_is_found_and_sample_peak_is_exact() {
    // fs/4 at 45°: every sample is ±A·√½, the peaks fall exactly between
    // samples. Sample peak = A/√2 (−3 dB), true peak = A.
    let sig = tp_signal(4.0, 1.0, 45.0);
    let mut m = stereo_meter(true);
    feed(&mut m, &sig, 2, |_, _| {});
    let sp_expected = sig.iter().fold(0.0f32, |a, &x| a.max(x.abs())) as f64;
    assert_eq!(m.sample_peak(0), sp_expected);
    assert_near(
        "sample peak dB",
        20.0 * m.sample_peak(0).log10(),
        -3.01,
        0.01,
    );
    let tp = 20.0 * m.true_peak(0).log10();
    assert!((-0.4..=0.2).contains(&tp), "true peak {tp:.3} dBTP, want 0");
}

#[test]
fn sample_peak_is_exact_per_channel() {
    // Distinct, exactly representable peaks per channel, at arbitrary
    // positions (including the first and the last frame).
    let peaks = [0.8125f32, 0.25, 0.5, 1.5];
    let ch = peaks.len();
    let frames = 10_000;
    let mut sig = vec![0.0f32; frames * ch];
    for (n, f) in sig.chunks_exact_mut(ch).enumerate() {
        for (c, s) in f.iter_mut().enumerate() {
            *s = 0.1 * ((n * (c + 1)) as f32 * 0.01).sin();
        }
    }
    sig[0] = -peaks[0];
    sig[(frames - 1) * ch + 1] = peaks[1];
    sig[1234 * ch + 2] = -peaks[2];
    sig[5000 * ch + 3] = peaks[3];
    let mut m = LoudnessMeter::new(&LoudnessConfig {
        channels: ch,
        true_peak: false,
        ..LoudnessConfig::default()
    })
    .unwrap();
    feed(&mut m, &sig, ch, |_, _| {});
    for (c, &p) in peaks.iter().enumerate() {
        assert_eq!(m.sample_peak(c), p as f64, "channel {c}");
        // Without true peak, the true-peak reading is the sample peak.
        assert_eq!(m.true_peak(c), p as f64, "channel {c}");
        assert_eq!(m.interval_sample_peak(c), p as f64, "channel {c}");
    }
    assert_eq!(m.reading().sample_peak, 1.5);
    m.clear_interval_peaks();
    assert_eq!(m.interval_sample_peak(3), 0.0);
    assert_eq!(m.sample_peak(3), 1.5, "since-reset peak survives");
    m.reset();
    assert_eq!(m.sample_peak(3), 0.0);
}

// ---------------------------------------------------------------------------
// EBU Tech 3342, cases 1–4: loudness range. Tolerance ±1 LU.

#[test]
fn tech3342_cases1_to_4_loudness_range() {
    let cases: [(&[f64], f64); 4] = [
        (&[-20.0, -30.0], 10.0),
        (&[-20.0, -15.0], 5.0),
        (&[-40.0, -20.0], 20.0),
        (&[-50.0, -35.0, -20.0, -35.0, -50.0], 15.0),
    ];
    for (i, (levels, want)) in cases.iter().enumerate() {
        let steps: Vec<_> = levels.iter().map(|&l| (Some(l), 20.0)).collect();
        let m = run_steps(&steps);
        assert_near(
            &format!("LRA case {}", i + 1),
            m.loudness_range(),
            *want,
            1.0,
        );
    }
}

#[test]
fn tech3342_case1_repeated_is_unchanged() {
    // "The expected response is unchanged if the test signal is repeated."
    let m = run_steps(&[
        (Some(-20.0), 20.0),
        (Some(-30.0), 20.0),
        (Some(-20.0), 20.0),
        (Some(-30.0), 20.0),
    ]);
    assert_near("LRA", m.loudness_range(), 10.0, 1.0);
}

// ---------------------------------------------------------------------------
// Channel layouts and weights.

/// Integrated loudness of a 1 kHz sine at -23 dBFS in `channels` channels,
/// all equal, under `layout` (5 s).
fn integrated_with(layout: ChannelLayout, channels: usize) -> f64 {
    let frames = (5.0 * FS) as usize;
    let mut sig = Vec::with_capacity(frames * channels);
    for n in 0..frames {
        let v = (db(-23.0) * (2.0 * PI * 1000.0 * n as f64 / FS).sin()) as f32;
        sig.extend(std::iter::repeat_n(v, channels));
    }
    let mut m = LoudnessMeter::new(&LoudnessConfig {
        sample_rate: FS as f32,
        channels,
        layout,
        true_peak: false,
    })
    .unwrap();
    feed(&mut m, &sig, channels, |_, _| {});
    m.integrated()
}

#[test]
fn channel_weights_follow_layout() {
    let mono = integrated_with(ChannelLayout::Mono, 1);
    let dual = integrated_with(ChannelLayout::DualMono, 1);
    let stereo = integrated_with(ChannelLayout::Stereo, 2);
    // Stereo at -23 dBFS reads -23 LUFS (Tech 3341 case 1); one channel 3 dB less.
    assert_near("stereo", stereo, -23.0, 0.1);
    assert_near("mono", mono, -23.0 - 10.0 * 2f64.log10(), 0.1);
    assert_near("dual mono", dual, stereo, 0.05);
    // Unknown layout: all channels weight 1.0, so 6 equal channels = +4.77 dB
    // over mono, while 5.1 drops the LFE and weights Ls/Rs by 1.41.
    let unknown6 = integrated_with(ChannelLayout::Unknown, 6);
    assert_near("unknown 6", unknown6, mono + 10.0 * 6f64.log10(), 0.1);
    let s51 = integrated_with(ChannelLayout::Surround51, 6);
    assert_near("5.1", s51, mono + 10.0 * (3.0 + 2.0 * 1.41f64).log10(), 0.1);
    // Mismatched layout falls back to Unknown.
    assert_near(
        "stereo layout on 6 ch",
        integrated_with(ChannelLayout::Stereo, 6),
        unknown6,
        1e-9,
    );
}

#[test]
fn layout_roles_and_custom_roles() {
    use ChannelRole::*;
    let roles: Vec<_> = (0..8)
        .map(|c| ChannelLayout::Surround71.role(c, 8))
        .collect();
    assert_eq!(
        roles,
        [Front, Front, Front, Lfe, Front, Front, Surround, Surround]
    );
    assert_eq!(Surround.weight(), 1.41);
    assert_eq!(Lfe.weight(), 0.0);

    let mut m = LoudnessMeter::new(&LoudnessConfig {
        channels: 2,
        true_peak: false,
        ..LoudnessConfig::default()
    })
    .unwrap();
    assert_eq!(m.channel_roles(), [Front, Front]);
    m.set_channel_roles(&[Front, Excluded]).unwrap();
    assert_eq!(
        m.set_channel_roles(&[DualMono, Front]),
        Err(MeterError::InvalidChannelRoles)
    );
    assert_eq!(
        m.set_channel_roles(&[Front]),
        Err(MeterError::InvalidChannelRoles)
    );
    // Signal only in the excluded channel: no loudness, but its peak counts.
    let sig: Vec<f32> = (0..48_000)
        .flat_map(|n| [0.0, (n as f32 * 0.1).sin() * 0.5])
        .collect();
    m.process(&sig);
    assert_eq!(m.momentary(), f64::NEG_INFINITY);
    assert_eq!(m.integrated(), f64::NEG_INFINITY);
    assert!(m.sample_peak(1) > 0.49);
}

#[test]
fn setup_errors_and_limits() {
    let cfg = |channels, sample_rate| LoudnessConfig {
        channels,
        sample_rate,
        ..LoudnessConfig::default()
    };
    assert_eq!(
        LoudnessMeter::new(&cfg(0, 48_000.0)).unwrap_err(),
        MeterError::UnsupportedChannelCount(0)
    );
    assert_eq!(
        LoudnessMeter::new(&cfg(MAX_LOUDNESS_CHANNELS + 1, 48_000.0)).unwrap_err(),
        MeterError::UnsupportedChannelCount(65)
    );
    assert_eq!(
        LoudnessMeter::new(&cfg(2, f32::NAN)).unwrap_err(),
        MeterError::UnsupportedSampleRate
    );
    for rate in [8_000.0, 44_100.0, 48_000.0, 96_000.0, 192_000.0] {
        for ch in [1, 2, 8, MAX_LOUDNESS_CHANNELS] {
            let mut m = LoudnessMeter::new(&cfg(ch, rate)).unwrap();
            let half_s = rate as usize / 2;
            let sig: Vec<f32> = (0..ch * half_s)
                .map(|i| ((i / ch) as f32 * 0.05).sin() * 0.3)
                .collect();
            m.process(&sig);
            m.process(&sig[..ch * 3 + 1]); // partial trailing frame is ignored
            let r = m.reading();
            assert!(
                r.momentary.is_finite() && r.integrated.is_finite(),
                "{rate} {ch}: {r:?}"
            );
            assert!(r.true_peak >= r.sample_peak && r.sample_peak > 0.29);
        }
    }
    // Silence reads -inf / 0, never NaN.
    let mut m = LoudnessMeter::new(&cfg(2, 48_000.0)).unwrap();
    m.process(&[0.0; 96_000]);
    let r = m.reading();
    assert_eq!(r.momentary, f64::NEG_INFINITY);
    assert_eq!(r.short_term, f64::NEG_INFINITY);
    assert_eq!(r.integrated, f64::NEG_INFINITY);
    assert_eq!(r.loudness_range, 0.0);
}

// ---------------------------------------------------------------------------
// Real-time behaviour of the loudness meter.

/// Long run (10 min at 8 kHz, > 5000 gating blocks, where `ebur128`'s default
/// queue would reallocate) with block sizes 1 … 1024 and every reading after
/// each block: zero allocations after `new`.
#[test]
fn loudness_meter_never_allocates_after_setup() {
    for true_peak in [false, true] {
        for ch in [1usize, 2, 6, 9] {
            let rate = 8_000.0;
            let mut m = LoudnessMeter::new(&LoudnessConfig {
                sample_rate: rate,
                channels: ch,
                layout: ChannelLayout::Unknown,
                true_peak,
            })
            .unwrap();
            let sizes = [1usize, 480, 7, 1024, 64, 333];
            let src: Vec<f32> = (0..1024 * ch)
                .map(|i| ((i * 7919) % 2001) as f32 / 1000.0 - 1.0)
                .collect();
            let total_frames = (if ch <= 2 { 610.0 } else { 30.0 } * rate) as usize;
            let before = alloc_count();
            let mut frames = 0;
            let mut i = 0;
            let mut sink = 0.0;
            while frames < total_frames {
                let n = sizes[i % sizes.len()];
                // Vary the level so the histogram fills many bins.
                m.process(&src[..n * ch]);
                frames += n;
                i += 1;
                if i % 64 == 0 {
                    let r = m.reading();
                    sink += r.momentary + r.short_term + r.integrated + r.loudness_range;
                    sink += m.interval_true_peak(0);
                    m.clear_interval_peaks();
                }
                if i % 5000 == 0 {
                    m.reset();
                }
            }
            assert_eq!(alloc_count() - before, 0, "tp={true_peak} ch={ch}");
            assert!(!sink.is_nan());
        }
    }
}

/// Documents why histogram mode is used: in `ebur128`'s default mode, the
/// block history reallocates once it exceeds 5000 blocks (500 s), and
/// reading the loudness range allocates every time.
#[test]
fn ebur128_default_mode_allocates_histogram_mode_does_not() {
    use ebur128::{EbuR128, Mode};
    let rate = 8_000u32;
    // 100 ms of a 1 kHz sine (whole periods), mono.
    let block: Vec<f32> = (0..800)
        .map(|n| 0.25 * (2.0 * PI * n as f64 / 8.0).sin() as f32)
        .collect();
    let run = |mode: Mode| -> (usize, usize) {
        let mut e = EbuR128::new(1, rate, mode).unwrap();
        let start = alloc_count();
        for _ in 0..5_200 {
            e.add_frames_f32(&block).unwrap();
        }
        let add = alloc_count() - start;
        let start = alloc_count();
        let _ = e.loudness_range().unwrap();
        let _ = e.loudness_global().unwrap();
        (add, alloc_count() - start)
    };
    let (add_q, read_q) = run(Mode::I | Mode::LRA);
    let (add_h, read_h) = run(Mode::I | Mode::LRA | Mode::HISTOGRAM);
    assert!(
        add_q > 0,
        "queue mode: add_frames reallocates past 5000 blocks"
    );
    assert!(read_q > 0, "queue mode: loudness_range allocates");
    assert_eq!((add_h, read_h), (0, 0), "histogram mode never allocates");
}

// ---------------------------------------------------------------------------
// Correlation meter.

/// xorshift32 white noise, uniform in [-1, 1).
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
    (0.5 * (2.0 * PI * f * n as f64 / FS + phase).sin()) as f32
}

#[test]
fn correlation_identical_inverted_independent() {
    let n = 48_000;
    let mut noise = Noise(1);
    let x: Vec<f32> = (0..n).map(|_| noise.next() * 0.5).collect();
    let neg: Vec<f32> = x.iter().map(|v| -v).collect();
    let mut noise2 = Noise(0xdead_beef);
    let y: Vec<f32> = (0..n).map(|_| noise2.next() * 0.5).collect();

    let mut m = CorrelationMeter::new(FS as f32, 300.0);
    m.process_pair(&x, &x);
    assert_near("identical", m.correlation(), 1.0, 1e-12);
    assert!(m.side_mid_ratio_db() <= -99.0);

    m.reset();
    m.process_pair(&x, &neg);
    assert_near("inverted", m.correlation(), -1.0, 1e-12);
    assert!(m.side_mid_ratio_db() >= 99.0);

    m.reset();
    m.process_pair(&x, &y);
    assert!(
        m.correlation().abs() < 0.05,
        "independent: {}",
        m.correlation()
    );
    assert_near("independent S/M", m.side_mid_ratio_db(), 0.0, 0.5);

    // Interleaved path gives the same result.
    let inter: Vec<f32> = x.iter().zip(&y).flat_map(|(&a, &b)| [a, b, 0.3]).collect();
    let mut m2 = CorrelationMeter::new(FS as f32, 300.0);
    m2.process_interleaved(&inter, 3);
    assert_eq!(m2.correlation(), m.correlation());
}

#[test]
fn correlation_of_phase_shifted_sines_is_cos_phi() {
    for deg in [0.0f64, 60.0, 90.0, 120.0, 180.0] {
        let mut m = CorrelationMeter::new(FS as f32, 300.0);
        for n in 0..96_000 {
            m.push(sine(n, 997.0, 0.0), sine(n, 997.0, deg.to_radians()));
        }
        assert_near(
            &format!("{deg}°"),
            m.correlation(),
            deg.to_radians().cos(),
            0.01,
        );
    }
}

#[test]
fn correlation_silent_channel_is_zero() {
    let mut m = CorrelationMeter::new(FS as f32, 300.0);
    assert_eq!(m.correlation(), 0.0, "fresh meter");
    for n in 0..48_000 {
        m.push(sine(n, 1000.0, 0.0), 0.0);
    }
    assert_eq!(m.correlation(), 0.0, "right silent");
    m.reset();
    for n in 0..48_000 {
        m.push(0.0, sine(n, 1000.0, 0.0));
    }
    assert_eq!(m.correlation(), 0.0, "left silent");
    m.reset();
    for _ in 0..48_000 {
        m.push(0.0, 0.0);
    }
    assert_eq!(m.correlation(), 0.0, "both silent");
    assert_eq!(m.side_mid_ratio_db(), 0.0);
}

#[test]
fn correlation_integration_time_and_side_mid_ratio() {
    // After a switch from identical to inverted noise, with constant channel
    // energies ⟨LR⟩ decays as P·(2e^(−t/τ) − 1), so r(τ) = 2/e − 1.
    let tau_ms = 50.0;
    let mut m = CorrelationMeter::new(FS as f32, tau_ms);
    let mut noise = Noise(7);
    for _ in 0..48_000 {
        let v = noise.next();
        m.push(v, v);
    }
    let tau = (tau_ms * 0.001 * FS as f32) as usize;
    for _ in 0..tau {
        let v = noise.next();
        m.push(v, -v);
    }
    assert_near(
        "r(τ)",
        m.correlation(),
        2.0 / std::f64::consts::E - 1.0,
        0.03,
    );
    assert_eq!(m.integration_ms(), 50.0);

    // R = L/2: S = L/4, M = 3L/4, so ⟨S²⟩/⟨M²⟩ = 1/9 = −9.54 dB.
    let mut m = CorrelationMeter::new(FS as f32, 300.0);
    for n in 0..48_000 {
        let v = sine(n, 440.0, 0.0);
        m.push(v, 0.5 * v);
    }
    assert_near(
        "S/M",
        m.side_mid_ratio_db(),
        20.0 * (1.0f64 / 3.0).log10(),
        1e-6,
    );
    assert_near("r", m.correlation(), 1.0, 1e-9);
}

// ---------------------------------------------------------------------------
// Mono detector.

/// Drive a detector frame by frame (in 480-frame blocks) and record every
/// state change (frame index, new state).
fn run_detector(
    d: &mut MonoDetector,
    frames: usize,
    mut gen: impl FnMut(usize) -> (f32, f32),
) -> Vec<(usize, StereoState)> {
    let mut changes = Vec::new();
    let mut buf = vec![0.0f32; 2 * BLOCK];
    let mut last = d.state();
    let mut n = 0;
    while n < frames {
        for f in buf.as_chunks_mut::<2>().0 {
            let (l, r) = gen(n);
            f[0] = l;
            f[1] = r;
            n += 1;
        }
        d.process_interleaved(&buf, 2);
        if d.state() != last {
            last = d.state();
            changes.push((n, last));
        }
    }
    changes
}

#[test]
fn mono_detector_classifies_mono_stereo_inverted() {
    let p = MonoDetectorParams::default();
    let mut d = MonoDetector::new(FS as f32, &p);
    assert_eq!(d.state(), StereoState::Stereo);
    let mut noise = Noise(3);
    let mut noise2 = Noise(4);

    // 2 s mono: switches once, after at least the hold time.
    let ch = run_detector(&mut d, 96_000, |_| {
        let v = noise.next() * 0.3;
        (v, v)
    });
    assert_eq!(ch.len(), 1);
    assert_eq!(ch[0].1, StereoState::Mono);
    assert!(ch[0].0 as f64 >= p.hold_ms as f64 * 0.001 * FS);

    // Silence afterwards: frozen.
    assert!(run_detector(&mut d, 96_000, |_| (0.0, 0.0)).is_empty());
    assert_eq!(d.state(), StereoState::Mono);

    // Inverted.
    let ch = run_detector(&mut d, 96_000, |_| {
        let v = noise.next() * 0.3;
        (v, -v)
    });
    assert_eq!(ch.last().map(|c| c.1), Some(StereoState::Inverted));
    assert!(ch.len() <= 2, "{ch:?}");

    // Independent noise: stereo.
    let ch = run_detector(&mut d, 96_000, |_| {
        (noise.next() * 0.3, noise2.next() * 0.3)
    });
    assert_eq!(ch.last().map(|c| c.1), Some(StereoState::Stereo));
    assert_eq!(ch.len(), 1);

    // A mono source panned off-centre (L = 2R) is not "mono".
    let ch = run_detector(&mut d, 96_000, |_| {
        let v = noise.next() * 0.3;
        (v, 0.5 * v)
    });
    assert!(ch.is_empty());

    d.reset();
    assert_eq!(d.state(), StereoState::Stereo);
}

/// Mono content plus independent noise in each channel, with the noise level
/// chosen so that ⟨S²⟩/⟨M²⟩ follows `ratio_db(t)` (σn² = 2ρσx²/(1−ρ)).
fn near_threshold(ratio_db: impl Fn(f64) -> f64, polarity: f32) -> impl FnMut(usize) -> (f32, f32) {
    let mut a = Noise(11);
    let mut b = Noise(12);
    let mut c = Noise(13);
    move |n| {
        let rho = 10f64.powf(ratio_db(n as f64 / FS) / 10.0);
        let g = (2.0 * rho / (1.0 - rho)).sqrt() as f32;
        let x = a.next() * 0.3;
        (x + g * 0.3 * b.next(), polarity * (x + g * 0.3 * c.next()))
    }
}

#[test]
fn mono_detector_does_not_flap_near_thresholds() {
    let p = MonoDetectorParams::default();
    for polarity in [1.0f32, -1.0] {
        let want = if polarity > 0.0 {
            StereoState::Mono
        } else {
            StereoState::Inverted
        };
        // Hovering around the enter threshold (−25 dB ± 3 dB, 1.5 s period)
        // with estimation noise on top: enters once, never leaves.
        let mut d = MonoDetector::new(FS as f32, &p);
        let ch = run_detector(
            &mut d,
            (30.0 * FS) as usize,
            near_threshold(|t| -25.0 + 3.0 * (2.0 * PI * t / 1.5).sin(), polarity),
        );
        assert_eq!(ch.len(), 1, "enter threshold, {want:?}: {ch:?}");
        assert_eq!(ch[0].1, want);

        // Hovering around the exit threshold (−15 dB ± 3 dB) from stereo:
        // never enters.
        let mut d = MonoDetector::new(FS as f32, &p);
        let ch = run_detector(
            &mut d,
            (30.0 * FS) as usize,
            near_threshold(|t| -15.0 + 3.0 * (2.0 * PI * t / 1.5).sin(), polarity),
        );
        assert!(
            ch.is_empty(),
            "exit threshold from stereo, {want:?}: {ch:?}"
        );

        // Once in the state, hovering around the exit threshold flips at most
        // a couple of times (the hold time suppresses the 1.5 s wobble),
        // never every half period.
        let mut d = MonoDetector::new(FS as f32, &p);
        let ch = run_detector(
            &mut d,
            (30.0 * FS) as usize,
            near_threshold(
                |t| {
                    if t < 3.0 {
                        -40.0
                    } else {
                        -15.0 + 1.0 * (2.0 * PI * t / 0.4).sin()
                    }
                },
                polarity,
            ),
        );
        assert!(ch.len() <= 2, "exit threshold from {want:?}: {ch:?}");
    }
}

#[test]
fn correlation_and_detector_never_allocate() {
    let mut m = CorrelationMeter::new(FS as f32, 300.0);
    let mut d = MonoDetector::new(FS as f32, &MonoDetectorParams::default());
    let mut noise = Noise(5);
    let sig: Vec<f32> = (0..2 * 4096).map(|_| noise.next()).collect();
    let (l, r): (Vec<f32>, Vec<f32>) = sig.as_chunks::<2>().0.iter().map(|f| (f[0], f[1])).unzip();
    let before = alloc_count();
    let mut sink = 0.0;
    for i in 0..500 {
        let n = [1usize, 480, 7, 4096][i % 4];
        m.process_interleaved(&sig[..2 * n], 2);
        m.process_pair(&l[..n], &r[..n]);
        m.set_integration_ms(10.0 + i as f32);
        d.process_interleaved(&sig[..2 * n], 2);
        d.process_pair(&l[..n], &l[..n]);
        d.set_params(&MonoDetectorParams {
            hold_ms: (i % 50) as f32 * 10.0,
            ..MonoDetectorParams::default()
        });
        sink += m.correlation() + m.side_mid_ratio_db() + d.meter().correlation();
        if i % 97 == 0 {
            m.reset();
            d.reset();
        }
    }
    assert_eq!(alloc_count() - before, 0);
    assert!(sink.is_finite());
}
