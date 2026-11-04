//! Harness tests for `stromdsp-deesser`: the element output matches the
//! library on the same input and buffer boundaries (48 kHz stereo and an odd
//! rate/channel count), including property changes, DISCONT and flushes;
//! `enabled=false` is bit-exact.

use crate::util;

use crate::util::{Action, FORMATS, TOL};
use gst::glib;
use gst::prelude::*;
use strom_dsp::deesser::{Deesser, DeesserMode, DeesserParams};

/// Tolerance while the crossover glides its frequency. The element splits a
/// buffer into whole-frame chunks of at most 4096 samples (1365 frames for 3
/// channels), and the crossover redesigns its sections every 32 frames counted
/// from the start of each call, so during a frequency glide the coefficients
/// are rounded differently (measured: 5.6e-6). Without a split (48 kHz stereo,
/// 1024-frame buffers) the output is bit-identical.
const GLIDE_TOL: f32 = 2e-5;

fn params() -> DeesserParams {
    DeesserParams {
        frequency_hz: 4_000.0,
        threshold_db: -40.0,
        max_reduction_db: 12.0,
        attack_ms: 0.5,
        release_ms: 40.0,
        mode: DeesserMode::SplitBand,
    }
}

fn props(p: &DeesserParams) -> Vec<(&'static str, glib::Value)> {
    vec![
        ("frequency", p.frequency_hz.to_value()),
        ("threshold", p.threshold_db.to_value()),
        ("max-reduction", p.max_reduction_db.to_value()),
        ("attack", p.attack_ms.to_value()),
        ("release", p.release_ms.to_value()),
    ]
}

/// Test signal with sibilance: the shared tone signal plus 7 kHz bursts.
fn signal(rate: u32, channels: usize) -> Vec<f32> {
    let mut x = util::test_signal(rate, channels, rate as usize);
    for (i, v) in x.iter_mut().enumerate() {
        let n = i / channels;
        if (n / (rate as usize / 10)).is_multiple_of(2) {
            let t = n as f32 / rate as f32;
            *v += 0.3 * (2.0 * std::f32::consts::PI * 7_000.0 * t).sin();
        }
    }
    x
}

#[test]
fn deesser_matches_library() {
    for (rate, channels) in FORMATS {
        let fpb = if channels == 2 { 1024 } else { 1500 };
        let input = signal(rate, channels);
        let p = params();
        let actions = [
            (3, Action::SetProperty("threshold", (-20.0f32).to_value())),
            (5, Action::Discont),
            (6, Action::SetProperty("mode", mode_value("wideband"))),
            (8, Action::SetProperty("frequency", 8_000.0f32.to_value())),
            (10, Action::Flush),
            (12, Action::SetProperty("mode", mode_value("split-band"))),
        ];
        let out = util::run_element(
            "stromdsp-deesser",
            &props(&p),
            rate,
            channels,
            &input,
            fpb,
            &actions,
        );
        let mut dsp = Deesser::new();
        dsp.set_params(&p);
        let reference = util::run_reference(
            dsp,
            rate,
            channels,
            &input,
            fpb,
            &actions,
            |dsp, name, value| {
                let mut q = *dsp.params();
                match name {
                    "threshold" => q.threshold_db = value.get().unwrap(),
                    "frequency" => q.frequency_hz = value.get().unwrap(),
                    "mode" => {
                        let (_, v) = glib::EnumValue::from_value(value).unwrap();
                        q.mode = if v.nick() == "wideband" {
                            DeesserMode::Wideband
                        } else {
                            DeesserMode::SplitBand
                        };
                    }
                    _ => unreachable!(),
                }
                dsp.set_params(&q);
            },
        );
        if channels == 2 {
            util::assert_bit_exact(&out, &reference);
        }
        util::assert_close(
            &out,
            &reference,
            if channels == 2 { TOL } else { GLIDE_TOL },
        );
        // The sibilance was actually reduced somewhere (the test is not vacuous).
        let changed = out
            .iter()
            .zip(&input)
            .filter(|(a, b)| (*a - *b).abs() > 0.05)
            .count();
        assert!(changed > 1000, "{rate}/{channels}: de-esser did nothing");
    }
}

/// A `glib::Value` of the de-esser's mode enum, by nick.
fn mode_value(nick: &str) -> glib::Value {
    util::init();
    let ty = glib::Type::from_name("StromDspDeesserMode").expect("enum type registered");
    let class = glib::EnumClass::with_type(ty).unwrap();
    class.to_value_by_nick(nick).unwrap()
}

#[test]
fn deesser_disabled_is_bit_exact() {
    for (rate, channels) in FORMATS {
        let input = signal(rate, channels);
        let mut pr = props(&params());
        pr.push(("enabled", false.to_value()));
        let out = util::run_element("stromdsp-deesser", &pr, rate, channels, &input, 512, &[]);
        util::assert_bit_exact(&out, &input);
    }
}
