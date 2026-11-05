//! The broadcast chain's latency modes on programme, moved here from
//! `strom-dsp/tests/broadcast_latency.rs` (seven configurations × three
//! programmes × 60 s): what the shorter lookaheads cost in limiter work,
//! clip depth and true peak, against longer settings of each stage.
//!
//! Per change, the harness baseline guards the same readings for the
//! standard and low modes at the target level (`streaming/peaky`,
//! `streaming/music` and their `_low_latency` twins):
//! `limiter_gr_programme_mean_db`, `max_true_peak_dbtp`, `clip_depth_db`.
//! The default suite keeps the true-peak ceiling on nasty signals in both
//! modes (`broadcast_latency::true_peak_ceiling_holds_at_short_lookahead`).
//! Only this run checks the hot, AGC-off programme and the intermediate
//! settings (5 ms limiter, 3 ms and no bass clipper lookahead, 0.75 ms).

use std::f64::consts::PI;

use strom_dsp::broadcast::{BroadcastParams, BroadcastProcessor, LatencyMode};
use strom_dsp::meter::{ChannelLayout, LoudnessConfig, LoudnessMeter};
use strom_dsp::presets::BroadcastPreset;
use strom_dsp::Processor;
use strom_dsp_harness::agc_support::{pink, scale_to_lufs, Looper};
use strom_dsp_harness::programme_support::peaky_programme;

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
#[ignore = "long programme run; runs in the extended workflow"]
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
