//! The broadcast chain in its latency modes ([`LatencyMode`]): standard
//! (spike guard 0.5 ms, bass clipper 1 ms, standard clipper, limiter
//! 1.5 ms: 269 frames, 5.60 ms at 48 kHz) and low (none, none, low
//! clipper, 0.6 ms: 133 frames, 2.77 ms).
//!
//! - Latency budget: each mode stays within its budget (6 ms, 3 ms) at
//!   44.1, 48 and 96 kHz, and an impulse comes out at the reported latency
//!   in both modes.
//! - The true-peak ceiling holds (ceiling + 0.1 dB, `ebur128`) on nasty
//!   signals at 44.1, 48 and 96 kHz, in both modes, with the clipper on
//!   and off.
//! - What the shorter settings cost on programme: the limiter's mean and
//!   largest gain reduction, the clipper's depth and the true peak,
//!   against longer settings of each stage.

use std::f64::consts::PI;

use crate::agc_support::{noise, pink, scale_to_lufs, Looper, Rng};
use crate::programme_support::peaky_programme;
use strom_dsp::broadcast::{BroadcastParams, BroadcastProcessor, LatencyMode};
use strom_dsp::meter::{ChannelLayout, LoudnessConfig, LoudnessMeter};
use strom_dsp::presets::BroadcastPreset;
use strom_dsp::Processor;

const TP_MARGIN_DB: f64 = 0.1;

/// A latency mode with optional overrides of single settings.
#[derive(Debug, Clone, Copy)]
struct Config {
    name: &'static str,
    mode: LatencyMode,
    clipper: bool,
    /// Limiter lookahead instead of the mode's (ms).
    lookahead_ms: Option<f32>,
    /// Bass clipper lookahead instead of the mode's (ms).
    bass_ms: Option<f32>,
}

const STANDARD: Config = Config {
    name: "standard",
    mode: LatencyMode::Standard,
    clipper: true,
    lookahead_ms: None,
    bass_ms: None,
};
const LOW: Config = Config {
    name: "low",
    mode: LatencyMode::Low,
    ..STANDARD
};

fn params(c: Config, base: BroadcastParams) -> BroadcastParams {
    let mut p = base;
    c.mode.apply(&mut p);
    p.clipper_enabled = c.clipper;
    if let Some(ms) = c.lookahead_ms {
        p.limiter_lookahead_ms = ms;
    }
    if let Some(ms) = c.bass_ms {
        p.bass_clipper_lookahead_ms = ms;
    }
    p
}

/// Each latency mode stays within its budget at 44.1, 48 and 96 kHz: the
/// latency of the mode's settings, and of a chain configured with them.
/// The budgets are on the element page (standard 6 ms, low 3 ms); the low
/// mode is tightest at 44.1 kHz, where the stages' fixed frame counts cost
/// the most time.
#[test]
fn latency_budget_per_mode() {
    let mut failures = Vec::new();
    for mode in LatencyMode::ALL {
        for fs in [44_100.0f32, 48_000.0, 96_000.0] {
            let mut p = BroadcastParams::preset(BroadcastPreset::EbuR128, 4);
            mode.apply(&mut p);
            let frames = BroadcastProcessor::latency_frames_for_params(fs, &p);
            let mut bp = BroadcastProcessor::new();
            bp.set_params(&p);
            bp.configure(fs, 2);
            assert_eq!(bp.latency_frames(), frames, "{} at {fs}", mode.nick());
            let ms = frames as f64 * 1000.0 / fs as f64;
            println!(
                "{} at {fs} Hz: {frames} frames, {ms:.3} ms (budget {} ms)",
                mode.nick(),
                mode.budget_ms()
            );
            if ms > mode.budget_ms() {
                failures.push(format!(
                    "{} at {fs} Hz: {ms:.3} ms over {} ms",
                    mode.nick(),
                    mode.budget_ms()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The latency of each mode, in frames at 48 kHz, as documented.
#[test]
fn latency_formula_per_mode() {
    let mut p = BroadcastParams::default();
    LatencyMode::Standard.apply(&mut p);
    // Spike guard 24 + 24, bass 48, clipper 69, limiter 72 + 32.
    assert_eq!(
        BroadcastProcessor::latency_frames_for_params(48_000.0, &p),
        48 + 48 + 69 + 72 + 32
    );
    LatencyMode::Low.apply(&mut p);
    // Spike guard 24, bass 0, clipper 48, limiter round(28.8) = 29 + 32.
    assert_eq!(
        BroadcastProcessor::latency_frames_for_params(48_000.0, &p),
        24 + 48 + 29 + 32
    );
    assert_eq!(
        BroadcastProcessor::latency_frames_for_params(44_100.0, &p),
        24 + 48 + 26 + 32
    );
    // The library default is the standard mode.
    let d = BroadcastParams::default();
    let mut s = d;
    LatencyMode::Standard.apply(&mut s);
    assert_eq!(d, s);
}

/// An impulse comes out at the reported latency in both modes, with the
/// chain on and off (flat chain: AGC and spike guard off, no compression).
#[test]
fn impulse_at_reported_latency_per_mode() {
    for mode in LatencyMode::ALL {
        for fs in [44_100.0f32, 48_000.0, 96_000.0] {
            for enabled in [true, false] {
                let mut p = BroadcastParams {
                    enabled,
                    agc_enabled: false,
                    spike_guard_enabled: false,
                    true_peak_ceiling_dbtp: 0.0,
                    ..BroadcastParams::default()
                };
                for b in &mut p.bands {
                    b.ratio = 1.0;
                }
                mode.apply(&mut p);
                let mut bp = BroadcastProcessor::new();
                bp.set_params(&p);
                bp.configure(fs, 1);
                let lat = bp.latency_frames();
                assert_eq!(lat, BroadcastProcessor::latency_frames_for_params(fs, &p));
                let mut x = vec![0.0f32; lat + 4096];
                x[1000] = 0.25;
                for c in x.chunks_mut(333) {
                    bp.process(c);
                }
                let at = x.iter().position(|&v| v != 0.0).unwrap();
                assert_eq!(at, 1000 + lat, "{} at {fs}, enabled {enabled}", mode.nick());
            }
        }
    }
}

/// Nasty signals for the ceiling (stereo, `frames` long, fixed seeds).
fn nasty(fs: f64, frames: usize) -> Vec<(String, Vec<f32>)> {
    let mut v: Vec<(String, Vec<f32>)> = Vec::new();
    let stereo =
        |f: &dyn Fn(usize) -> f32| -> Vec<f32> { (0..frames).flat_map(|n| [f(n), f(n)]).collect() };
    v.push((
        "square +20 dBFS".into(),
        stereo(&|n| if (n / 48) % 2 == 0 { 10.0 } else { -10.0 }),
    ));
    v.push((
        "impulse train +20 dBFS".into(),
        stereo(&|n| if n % 997 == 0 { 10.0 } else { 0.0 }),
    ));
    v.push((
        "fs/4 at 45° +6 dBFS".into(),
        stereo(&|n| (2.0 * (PI / 2.0 * n as f64 + PI / 4.0).sin()) as f32),
    ));
    for f in [15_000.0, 19_000.0, 0.45 * fs] {
        for ph in [0.0, 0.7] {
            v.push((
                format!("{f:.0} Hz phase {ph} +10 dBFS"),
                stereo(&|n| (3.16 * (2.0 * PI * f * n as f64 / fs + ph).sin()) as f32),
            ));
        }
    }
    let mut w = noise(frames.next_power_of_two(), 2, fs, 0.0, 3);
    w.truncate(frames * 2);
    w.iter_mut().for_each(|x| *x *= 10.0);
    v.push(("white noise +20 dBFS".into(), w));
    let mut rng = Rng::new(9);
    let mut clicks: Vec<f32> = pink(frames.next_power_of_two(), 2, fs, 4)
        .into_iter()
        .take(frames * 2)
        .map(|x| 0.1 * x)
        .collect();
    let mut n = 100;
    while n < frames {
        let a = (10.0 * rng.bipolar()) as f32;
        clicks[2 * n] += a;
        clicks[2 * n + 1] -= a;
        n += 300 + (rng.uniform() * 3000.0) as usize;
    }
    v.push(("random clicks over noise".into(), clicks));
    v.push((
        "step into spikes".into(),
        stereo(&|n| {
            if n < frames / 2 {
                0.05
            } else if n % 4801 == 0 {
                30.0
            } else {
                0.9
            }
        }),
    ));
    v
}

/// The true-peak ceiling holds in both configurations (and with the
/// clipper off, the limiter alone at the short lookaheads), at 44.1, 48
/// and 96 kHz, on every nasty signal. AGC off, so the signals reach the
/// limiter at their full level.
#[test]
fn true_peak_ceiling_holds_at_short_lookahead() {
    let mut failures = Vec::new();
    let mut worst = f64::NEG_INFINITY;
    let base = BroadcastParams {
        agc_enabled: false,
        ..BroadcastParams::preset(BroadcastPreset::Streaming, 4)
    };
    let ceiling = base.true_peak_ceiling_dbtp as f64;
    for fs in [44_100.0f64, 48_000.0, 96_000.0] {
        let frames = fs as usize; // 1 s
        let signals = nasty(fs, frames);
        for cfg in [
            STANDARD,
            LOW,
            Config {
                name: "standard, clipper off",
                clipper: false,
                ..STANDARD
            },
            Config {
                name: "low, clipper off",
                clipper: false,
                ..LOW
            },
        ] {
            for (name, x) in &signals {
                let mut bp = BroadcastProcessor::new();
                bp.set_params(&params(cfg, base));
                bp.configure(fs as f32, 2);
                let mut m = LoudnessMeter::new(&LoudnessConfig {
                    sample_rate: fs as f32,
                    channels: 2,
                    layout: ChannelLayout::Unknown,
                    true_peak: true,
                })
                .unwrap();
                let mut y = x.clone();
                for b in y.chunks_mut(960) {
                    bp.process(b);
                }
                m.process(&y);
                let tp = 20.0 * (0..2).map(|c| m.true_peak(c)).fold(0.0, f64::max).log10();
                worst = worst.max(tp - ceiling);
                if tp > ceiling + TP_MARGIN_DB {
                    failures.push(format!("{fs} Hz, {}: {name}: {tp:.3} dBTP", cfg.name));
                }
            }
        }
    }
    println!("worst true peak re ceiling: {worst:+.3} dB");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Limiter gain reduction, clip depth, true peak and loudness on programme
/// for each configuration (streaming preset, 60 s, measured from 10 s):
/// the sport-like peaky programme and a music-like programme at −23 LUFS
/// input, and the same peaky programme hot (AGC off, −10 LUFS, 6 dB over
/// the target into the multiband, clipper and limiter). The bounds: true
/// peak ≤ ceiling + 0.1 dB and mean limiter gain reduction ≤ 2 dB in every
/// configuration. "Clipper off" measures the plain limiter: its peak
/// control is off too (with the clipper off it would otherwise take every
/// peak with its slow gain, 3 dB under the ceiling, which is what it is
/// for, not limiter work).
#[test]
fn limiter_work_on_programme() {
    let fs = 48_000.0f64;
    let base = BroadcastParams::preset(BroadcastPreset::Streaming, 4);
    let ceiling = base.true_peak_ceiling_dbtp as f64;
    let mut music = pink(1 << 19, 2, fs, 21);
    for n in 0..1 << 19 {
        let t = n as f64 / fs;
        let syl = 0.6 + 0.4 * (2.0 * PI * 4.0 * t).sin();
        let tones = 0.4 * (2.0 * PI * 110.0 * t).sin() + 0.2 * (2.0 * PI * 1_320.0 * t).sin();
        for c in 0..2 {
            let v = &mut music[n * 2 + c];
            *v = ((*v as f64 * 0.5 + tones) * syl) as f32;
        }
    }
    let peaky = peaky_programme(2, fs, 1.1);
    let programmes = [
        ("peaky", peaky.clone(), -23.0, true),
        ("music", music, -23.0, true),
        ("peaky hot", peaky, -10.0, false),
    ];
    let mut failures = Vec::new();
    for (pname, src, input_lufs, agc) in &programmes {
        let mut x = src.clone();
        scale_to_lufs(&mut x, 2, fs, *input_lufs);
        let base = BroadcastParams {
            agc_enabled: *agc,
            ..base
        };
        for cfg in [
            Config {
                name: "clipper off, 5 ms",
                clipper: false,
                lookahead_ms: Some(5.0),
                ..STANDARD
            },
            Config {
                name: "standard, limiter 5 ms",
                lookahead_ms: Some(5.0),
                ..STANDARD
            },
            Config {
                name: "standard, bass 3 ms",
                bass_ms: Some(3.0),
                ..STANDARD
            },
            STANDARD,
            Config {
                name: "standard, bass 0",
                bass_ms: Some(0.0),
                ..STANDARD
            },
            Config {
                name: "low, limiter 0.75 ms",
                lookahead_ms: Some(0.75),
                ..LOW
            },
            LOW,
        ] {
            let mut bp = BroadcastProcessor::new();
            bp.set_params(&params(cfg, base));
            bp.configure(fs as f32, 2);
            bp.set_limiter_peak_control(cfg.clipper);
            let lat_ms = bp.latency_frames() as f64 / fs * 1000.0;
            let mut m = LoudnessMeter::new(&LoudnessConfig {
                sample_rate: fs as f32,
                channels: 2,
                layout: ChannelLayout::Unknown,
                true_peak: true,
            })
            .unwrap();
            let mut looper = Looper::new(&x);
            let mut buf = vec![0.0f32; 960];
            let (mut sum, mut n, mut max) = (0.0, 0, 0.0f64);
            let mut depth = 0.0f64;
            for i in 0..6000 {
                looper.fill(&mut buf);
                bp.process(&mut buf);
                let meters = bp.take_meters();
                if i >= 1000 {
                    m.process(&buf);
                    sum += meters.limiter_gain_reduction_db;
                    max = max.max(meters.limiter_gain_reduction_db);
                    depth += 10f64.powf(meters.clip_depth_db / 10.0);
                    n += 1;
                }
            }
            let tp = 20.0 * (0..2).map(|c| m.true_peak(c)).fold(0.0, f64::max).log10();
            let mean = sum / n as f64;
            let depth_db = 10.0 * (depth / n as f64).max(1e-30).log10();
            println!(
                "{pname}, {} ({lat_ms:.2} ms): limiter GR mean {mean:.2} dB, max {max:.2} dB; \
                 clip depth {depth_db:.1} dB; true peak {tp:+.2} dBTP; integrated {:.2} LUFS",
                cfg.name,
                m.integrated()
            );
            if tp > ceiling + TP_MARGIN_DB {
                failures.push(format!("{pname}, {}: true peak {tp:.2}", cfg.name));
            }
            if mean > 2.0 {
                failures.push(format!("{pname}, {}: limiter mean GR {mean:.2}", cfg.name));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
