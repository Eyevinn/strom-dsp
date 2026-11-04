//! Spike guard measurements that need the harness metrics.
//!
//! - **No spurs from capping a click:** single-sample clicks (±1.0) on a
//!   steady 15 kHz tone of 0.1 are spikes the guard brings down to its cap.
//!   A memoryless clipper's harmonics of the tone would fold back below
//!   10 kHz (45 kHz → 3 kHz, 105 kHz → 9 kHz at 48 kHz). The clicks
//!   themselves put broadband content there, so the test asserts that the
//!   guard's output stays at least 6 dB under the input's own level in that
//!   band, in both latency modes and with a 2 ms lookahead: the gain dip
//!   scales the tone around each click, it adds nothing there.
//! - **Swells are programme:** 0.2 ms Hann-shaped swells of the same tone
//!   (from 0.1 to 1.0) build up over about ten samples, so the rise gate
//!   leaves them alone: no event, the output is the delayed input.
//! - **Events:** the clicks of the speech-like and the peaky harness
//!   signals are caught, one event each: never more events than clicks, and
//!   at least 80 % of them (a click that lands on a loud hit or burst is
//!   not far enough above the programme to be a spike).

use std::f64::consts::PI;

use strom_dsp::spike_guard::{
    SpikeGuard, SpikeGuardParams, LOW_LATENCY_LOOKAHEAD_MS, STANDARD_LOOKAHEAD_MS,
};
use strom_dsp::Processor;
use strom_dsp_harness::metrics::FFT_LEN;
use strom_dsp_harness::signals::{SignalKind, CH};
use strom_dsp_harness::spectrum::Spectrum;

const FS: f64 = 48_000.0;

/// The lookaheads of the element's latency modes.
const MODES: [(&str, f32); 2] = [
    ("standard", STANDARD_LOOKAHEAD_MS),
    ("low latency", LOW_LATENCY_LOOKAHEAD_MS),
];

fn guard(lookahead_ms: f32) -> SpikeGuard {
    let mut g = SpikeGuard::new();
    g.set_params(&SpikeGuardParams::default());
    g.set_lookahead_ms(lookahead_ms);
    g.configure(FS as f32, CH);
    g
}

/// Steady 15 kHz tone of 0.1 with a single-sample click (±1.0,
/// alternating) every 20 ms (stereo, identical channels).
fn clicks_on_tone(frames: usize) -> Vec<f32> {
    let period = (0.02 * FS) as usize;
    (0..frames)
        .flat_map(|n| {
            let k = (n + period / 2) % period;
            let click = match (k, (n + period / 2) / period % 2) {
                (0, 0) => 1.0,
                (0, _) => -1.0,
                _ => 0.0,
            };
            let v = (0.1 * (2.0 * PI * 15_000.0 * n as f64 / FS).sin() + click) as f32;
            [v; CH]
        })
        .collect()
}

/// Steady 15 kHz tone of 0.1 with a 0.2 ms Hann-shaped swell to 1.0 every
/// 20 ms (stereo, identical channels).
fn swells(frames: usize) -> Vec<f32> {
    let len = (0.0002 * FS) as usize;
    let period = (0.02 * FS) as usize;
    (0..frames)
        .flat_map(|n| {
            let k = (n + period / 2) % period;
            let w = if k < len {
                0.5 - 0.5 * (2.0 * PI * k as f64 / len as f64).cos()
            } else {
                0.0
            };
            let v = ((0.1 + 0.9 * w) * (2.0 * PI * 15_000.0 * n as f64 / FS).sin()) as f32;
            [v; CH]
        })
        .collect()
}

/// Largest spur from 20 Hz to 10 kHz (dB re full-scale sine), channel 0:
/// where a clipped 15 kHz tone's harmonics fold (45 kHz → 3 kHz, 105 kHz →
/// 9 kHz).
fn spur_db(x: &[f32], from: usize) -> f64 {
    Spectrum::of(x, CH, 0, FS, from, FFT_LEN).max_db_where(|f| (20.0..=10_000.0).contains(&f))
}

#[test]
fn capped_clicks_on_15k_add_no_spurs() {
    let frames = FFT_LEN + 9600;
    let x = clicks_on_tone(frames);
    let p = SpikeGuardParams::default();
    let cap = 0.1 * 10f64.powf(p.threshold_db as f64 / 20.0);
    let input_spur = spur_db(&x, 9600);
    println!("input: {input_spur:.1} dBFS in 20 Hz … 10 kHz");
    for (name, la) in [MODES[0], MODES[1], ("2 ms lookahead", 2.0)] {
        let mut g = guard(la);
        let mut y = x.clone();
        for c in y.chunks_mut(480 * CH) {
            g.process(c);
        }
        let events = g.take_events();
        let expected = frames / (0.02 * FS) as usize;
        assert!(
            events as usize + 2 >= expected,
            "{name}: {events} events, {expected} clicks"
        );
        // The clicks come down to the cap (0.1 · 8 dB = 0.25, within 0.5 dB).
        let lat = g.latency_frames();
        let out_peak = y[(4800 + lat) * CH..]
            .iter()
            .fold(0.0f64, |m, &v| m.max(v.abs() as f64));
        assert!(
            out_peak <= cap * 1.06,
            "{name}: peak {out_peak:.3}, cap {cap:.3}"
        );
        let spur = spur_db(&y, 9600);
        println!("{name}: spur {spur:.1} dBFS in 20 Hz … 10 kHz");
        // The clicks put content there (`input`); the dip takes them down
        // and only scales the tone, so the output stays at least 6 dB under.
        assert!(
            spur <= input_spur - 6.0,
            "{name}: {spur:.1} dBFS, input {input_spur:.1} dBFS"
        );
    }
}

#[test]
fn swells_are_programme() {
    let frames = 9600 * 2;
    let x = swells(frames);
    for (name, la) in [MODES[0], MODES[1]] {
        let mut g = guard(la);
        let mut y = x.clone();
        for c in y.chunks_mut(480 * CH) {
            g.process(c);
        }
        assert_eq!(g.take_events(), 0, "{name}");
        let lat = g.latency_frames() * CH;
        for (i, (a, b)) in y[lat..].iter().zip(&x).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{name}: sample {i}");
        }
    }
}

#[test]
fn harness_clicks_are_caught_once() {
    for (name, la) in MODES {
        for kind in [
            SignalKind::SpeechClicks(0.1),
            SignalKind::SpeechClicks(0.3),
            SignalKind::Peaky(0.3),
            SignalKind::Peaky(1.1),
        ] {
            let s = kind.generate(8 * FS as usize, FS);
            let mut g = guard(la);
            let mut y = s.data.clone();
            for c in y.chunks_mut(480 * CH) {
                g.process(c);
            }
            let events = g.take_events() as usize;
            println!(
                "{name}, {kind:?}: {events} events, {} clicks",
                s.marks.len()
            );
            assert!(
                events <= s.marks.len() && events * 10 >= s.marks.len() * 8,
                "{name}, {kind:?}: {events} events, {} clicks",
                s.marks.len()
            );
        }
    }
}
