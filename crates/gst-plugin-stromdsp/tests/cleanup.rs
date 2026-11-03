//! Harness tests for `stromdsp-cleanup`: the element output matches the
//! library on the same input and buffer boundaries (48 kHz stereo and an odd
//! rate/channel count), including property changes, DISCONT and flushes; the
//! defaults and `enabled=false` are bit-exact; hum is actually removed.

mod util;

use gst::glib;
use gst::prelude::*;
use strom_dsp::cleanup::{Cleanup, CleanupParams, HumFrequency};
use util::{Action, FORMATS, TOL};

/// Tolerance while the high-pass glides its cutoff: the element splits a
/// buffer into whole-frame chunks of at most 4096 samples (1365 frames for 3
/// channels), and the Butterworth filter redesigns its sections every 32
/// frames counted from the start of each call, so during a glide the
/// coefficients are rounded differently. Without a split (48 kHz stereo,
/// 1024-frame buffers) the output is bit-identical.
const GLIDE_TOL: f32 = 2e-5;

fn params() -> CleanupParams {
    CleanupParams {
        dc_block: true,
        highpass_hz: 60.0,
        highpass_order: 4,
        hum: HumFrequency::Hz50,
        hum_harmonics: 6,
        hum_q: 12.0,
    }
}

fn hum_value(nick: &str) -> glib::Value {
    util::init();
    let ty = glib::Type::from_name("StromDspCleanupHum").expect("enum type registered");
    glib::EnumClass::with_type(ty)
        .unwrap()
        .to_value_by_nick(nick)
        .unwrap()
}

fn props(p: &CleanupParams) -> Vec<(&'static str, glib::Value)> {
    vec![
        ("dc-block", p.dc_block.to_value()),
        ("highpass", p.highpass_hz.to_value()),
        ("highpass-order", (p.highpass_order as u32).to_value()),
        ("hum", hum_value("50")),
        ("hum-harmonics", (p.hum_harmonics as u32).to_value()),
        ("hum-q", p.hum_q.to_value()),
    ]
}

/// The shared test signal plus a DC offset and 50 Hz hum with harmonics.
fn signal(rate: u32, channels: usize) -> Vec<f32> {
    let mut x = util::test_signal(rate, channels, rate as usize);
    for (i, v) in x.iter_mut().enumerate() {
        let t = (i / channels) as f32 / rate as f32;
        let w = 2.0 * std::f32::consts::PI * 50.0 * t;
        *v += 0.05 + 0.1 * w.sin() + 0.05 * (3.0 * w).sin();
    }
    x
}

#[test]
fn cleanup_matches_library() {
    for (rate, channels) in FORMATS {
        let fpb = if channels == 2 { 1024 } else { 1500 };
        let input = signal(rate, channels);
        let p = params();
        let actions = [
            (3, Action::SetProperty("highpass", 120.0f32.to_value())),
            (4, Action::SetProperty("hum", hum_value("60"))),
            (5, Action::Discont),
            (6, Action::SetProperty("highpass-order", 7u32.to_value())),
            (7, Action::SetProperty("dc-block", false.to_value())),
            (8, Action::SetProperty("hum-harmonics", 10u32.to_value())),
            (9, Action::SetProperty("hum-q", 40.0f32.to_value())),
            (10, Action::Flush),
            (12, Action::SetProperty("highpass", 0.0f32.to_value())),
            (13, Action::SetProperty("hum", hum_value("off"))),
        ];
        let out = util::run_element(
            "stromdsp-cleanup",
            &props(&p),
            rate,
            channels,
            &input,
            fpb,
            &actions,
        );
        let mut dsp = Cleanup::new();
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
                    "highpass" => q.highpass_hz = value.get().unwrap(),
                    "highpass-order" => q.highpass_order = value.get::<u32>().unwrap() as usize,
                    "dc-block" => q.dc_block = value.get().unwrap(),
                    "hum-harmonics" => q.hum_harmonics = value.get::<u32>().unwrap() as usize,
                    "hum-q" => q.hum_q = value.get().unwrap(),
                    "hum" => {
                        let (_, v) = glib::EnumValue::from_value(value).unwrap();
                        q.hum = match v.value() {
                            50 => HumFrequency::Hz50,
                            60 => HumFrequency::Hz60,
                            _ => HumFrequency::Off,
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
        // After every stage is off again, the output is the input, bit-exact.
        let tail = 14 * fpb * channels;
        util::assert_bit_exact(&out[tail..], &input[tail..]);
    }
}

/// With the default properties the element changes nothing (bit-exact), and
/// with `enabled=false` and every stage on it changes nothing either.
#[test]
fn cleanup_defaults_and_disabled_are_bit_exact() {
    for (rate, channels) in FORMATS {
        let input = signal(rate, channels);
        let out = util::run_element("stromdsp-cleanup", &[], rate, channels, &input, 512, &[]);
        util::assert_bit_exact(&out, &input);
        let mut pr = props(&params());
        pr.push(("enabled", false.to_value()));
        let out = util::run_element("stromdsp-cleanup", &pr, rate, channels, &input, 512, &[]);
        util::assert_bit_exact(&out, &input);
    }
}

/// End to end: 50 Hz hum (fundamental and third harmonic) and DC are removed
/// by the element, the 220 Hz programme tone passes.
#[test]
fn cleanup_removes_hum_and_dc() {
    let rate = 48_000u32;
    let n = rate as usize * 2;
    let tone = |f: f64, i: usize| (2.0 * std::f64::consts::PI * f * i as f64 / rate as f64).sin();
    let input: Vec<f32> = (0..n)
        .map(|i| (0.1 + 0.3 * tone(50.0, i) + 0.1 * tone(150.0, i) + 0.2 * tone(220.0, i)) as f32)
        .collect();
    let p = CleanupParams {
        highpass_hz: 0.0,
        ..params()
    };
    let mut pr = props(&p);
    pr.retain(|(name, _)| *name != "highpass");
    let out = util::run_element("stromdsp-cleanup", &pr, rate, 1, &input, 960, &[]);
    let tail = &out[n / 2..];
    let amp = |f: f64| {
        let (mut s, mut c) = (0.0, 0.0);
        for (k, &v) in tail.iter().enumerate() {
            let ph = 2.0 * std::f64::consts::PI * f * (n / 2 + k) as f64 / rate as f64;
            s += v as f64 * ph.sin();
            c += v as f64 * ph.cos();
        }
        2.0 * (s * s + c * c).sqrt() / tail.len() as f64
    };
    let mean = tail.iter().map(|&v| v as f64).sum::<f64>() / tail.len() as f64;
    assert!(mean.abs() < 1e-5, "DC {mean}");
    assert!(amp(50.0) < 0.3 * 0.01, "50 Hz {}", amp(50.0));
    assert!(amp(150.0) < 0.1 * 0.01, "150 Hz {}", amp(150.0));
    let a220 = amp(220.0);
    assert!((a220 / 0.2 - 1.0).abs() < 0.06, "220 Hz {a220}");
}
