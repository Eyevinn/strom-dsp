//! The broadcast chain at its two latency configurations: standard
//! (standard clipper setting, 1.5 ms limiter lookahead, 173 frames at
//! 48 kHz) and low (low clipper setting, 0.75 ms, 116 frames).
//!
//! - The true-peak ceiling holds (ceiling + 0.1 dB, `ebur128`) on nasty
//!   signals at 44.1, 48 and 96 kHz, in both configurations, with the
//!   clipper on and off.
//! - What the shorter lookahead costs on programme: the limiter's mean and
//!   largest gain reduction, measured against the old 5 ms lookahead.

mod agc_support;
mod programme_support;

use std::f64::consts::PI;

use agc_support::{noise, pink, scale_to_lufs, Looper, Rng};
use programme_support::peaky_programme;
use strom_dsp::broadcast::{BroadcastParams, BroadcastProcessor, LOW_LATENCY_LOOKAHEAD_MS};
use strom_dsp::clipper::ClipperLatency;
use strom_dsp::meter::{ChannelLayout, LoudnessConfig, LoudnessMeter};
use strom_dsp::presets::BroadcastPreset;
use strom_dsp::Processor;

const TP_MARGIN_DB: f64 = 0.1;

#[derive(Debug, Clone, Copy)]
struct Config {
    name: &'static str,
    clipper: bool,
    clipper_latency: ClipperLatency,
    lookahead_ms: f32,
}

const STANDARD: Config = Config {
    name: "standard",
    clipper: true,
    clipper_latency: ClipperLatency::Standard,
    lookahead_ms: 1.5,
};
const LOW: Config = Config {
    name: "low",
    clipper: true,
    clipper_latency: ClipperLatency::Low,
    lookahead_ms: LOW_LATENCY_LOOKAHEAD_MS,
};

fn params(c: Config, base: BroadcastParams) -> BroadcastParams {
    BroadcastParams {
        clipper_enabled: c.clipper,
        clipper_latency: c.clipper_latency,
        limiter_lookahead_ms: c.lookahead_ms,
        ..base
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

/// Limiter gain reduction, true peak and loudness on programme for each
/// configuration (streaming preset, −23 LUFS input, 60 s, measured from
/// 10 s): the sport-like peaky programme and a music-like programme.
/// The bounds: true peak ≤ ceiling + 0.1 dB and mean limiter gain
/// reduction ≤ 2 dB in every configuration.
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
    let programmes = [("peaky", peaky_programme(2, fs, 1.1)), ("music", music)];
    let mut failures = Vec::new();
    for (pname, src) in &programmes {
        let mut x = src.clone();
        scale_to_lufs(&mut x, 2, fs, -23.0);
        for cfg in [
            Config {
                name: "clipper off, 5 ms",
                clipper: false,
                lookahead_ms: 5.0,
                ..STANDARD
            },
            Config {
                name: "standard clipper, 5 ms",
                lookahead_ms: 5.0,
                ..STANDARD
            },
            STANDARD,
            LOW,
        ] {
            let mut bp = BroadcastProcessor::new();
            bp.set_params(&params(cfg, base));
            bp.configure(fs as f32, 2);
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
            for i in 0..6000 {
                looper.fill(&mut buf);
                bp.process(&mut buf);
                let meters = bp.take_meters();
                if i >= 1000 {
                    m.process(&buf);
                    sum += meters.limiter_gain_reduction_db;
                    max = max.max(meters.limiter_gain_reduction_db);
                    n += 1;
                }
            }
            let tp = 20.0 * (0..2).map(|c| m.true_peak(c)).fold(0.0, f64::max).log10();
            let mean = sum / n as f64;
            println!(
                "{pname}, {} ({lat_ms:.2} ms): limiter GR mean {mean:.2} dB, max {max:.2} dB; \
                 true peak {tp:+.2} dBTP; integrated {:.2} LUFS",
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
