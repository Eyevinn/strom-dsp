//! Harness tests for `stromdsp-compressor`.

mod util;

use gst::glib;
use gst::prelude::*;
use strom_dsp::{Compressor, CompressorParams};
use util::{Action, FORMATS, TOL};

fn params() -> CompressorParams {
    CompressorParams {
        threshold: 0.25,
        ratio: 4.0,
        attack_ms: 5.0,
        release_ms: 50.0,
        makeup_gain: 1.5,
        knee: 0.5,
    }
}

fn props(p: &CompressorParams) -> Vec<(&'static str, glib::Value)> {
    vec![
        ("threshold", p.threshold.to_value()),
        ("ratio", p.ratio.to_value()),
        ("attack", p.attack_ms.to_value()),
        ("release", p.release_ms.to_value()),
        ("makeup-gain", p.makeup_gain.to_value()),
        ("knee", p.knee.to_value()),
    ]
}

#[test]
fn compressor_matches_library() {
    for (rate, channels) in FORMATS {
        let fpb = if channels == 2 { 1024 } else { 1500 };
        let input = util::test_signal(rate, channels, rate as usize);
        let p = params();
        let actions = [
            (4, Action::SetProperty("ratio", 10.0f32.to_value())),
            (6, Action::Discont),
            (7, Action::SetProperty("knee", 1.0f32.to_value())),
            (7, Action::SetProperty("makeup-gain", 0.5f32.to_value())),
            (11, Action::Flush),
        ];
        let out = util::run_element(
            "stromdsp-compressor",
            &props(&p),
            rate,
            channels,
            &input,
            fpb,
            &actions,
        );
        let mut dsp = Compressor::new();
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
                let v: f32 = value.get().unwrap();
                match name {
                    "ratio" => q.ratio = v,
                    "knee" => q.knee = v,
                    "makeup-gain" => q.makeup_gain = v,
                    _ => unreachable!(),
                }
                dsp.set_params(&q);
            },
        );
        util::assert_close(&out, &reference, TOL);
    }
}

/// `detector`, `rms-window` and `auto-makeup`, set at start and changed
/// while running, give the library's output.
#[test]
fn compressor_detector_and_auto_makeup_match_library() {
    use gststromdsp::compressor::StromDspCompressorDetector;
    use strom_dsp::compressor::CompressorDetector;
    for (rate, channels) in FORMATS {
        let fpb = if channels == 2 { 1024 } else { 1500 };
        let input = util::test_signal(rate, channels, rate as usize);
        let p = params();
        let mut pr = props(&p);
        pr.push(("detector", StromDspCompressorDetector::Rms.to_value()));
        pr.push(("rms-window", 25.0f32.to_value()));
        pr.push(("auto-makeup", true.to_value()));
        let actions = [
            (3, Action::SetProperty("rms-window", 5.0f32.to_value())),
            (
                5,
                Action::SetProperty("detector", StromDspCompressorDetector::Peak.to_value()),
            ),
            (8, Action::SetProperty("auto-makeup", false.to_value())),
            (9, Action::SetProperty("ratio", 8.0f32.to_value())),
        ];
        let out = util::run_element(
            "stromdsp-compressor",
            &pr,
            rate,
            channels,
            &input,
            fpb,
            &actions,
        );
        let mut dsp = Compressor::new();
        dsp.set_detector(CompressorDetector::Rms);
        dsp.set_rms_window(25.0);
        dsp.set_auto_makeup(true);
        dsp.set_params(&p);
        let reference = util::run_reference(
            dsp,
            rate,
            channels,
            &input,
            fpb,
            &actions,
            |dsp, name, value| match name {
                "rms-window" => dsp.set_rms_window(value.get().unwrap()),
                "detector" => {
                    dsp.set_detector(match value.get::<StromDspCompressorDetector>().unwrap() {
                        StromDspCompressorDetector::Peak => CompressorDetector::Peak,
                        StromDspCompressorDetector::Rms => CompressorDetector::Rms,
                    })
                }
                "auto-makeup" => dsp.set_auto_makeup(value.get().unwrap()),
                "ratio" => {
                    let mut q = *dsp.params();
                    q.ratio = value.get().unwrap();
                    dsp.set_params(&q);
                }
                _ => unreachable!(),
            },
        );
        util::assert_close(&out, &reference, TOL);
    }
}

#[test]
fn compressor_disabled_is_bit_exact() {
    for (rate, channels) in FORMATS {
        let input = util::test_signal(rate, channels, rate as usize / 2);
        let mut pr = props(&params());
        pr.push(("enabled", false.to_value()));
        let out = util::run_element("stromdsp-compressor", &pr, rate, channels, &input, 512, &[]);
        util::assert_bit_exact(&out, &input);
    }
}

#[test]
fn compressor_knee_reads_back_exactly() {
    util::init();
    let e = gst::ElementFactory::make("stromdsp-compressor")
        .build()
        .unwrap();
    for k in [
        0.0f32,
        0.001,
        0.0625,
        0.1,
        1.0 / 3.0,
        std::f32::consts::FRAC_1_SQRT_2,
        0.999_999_9,
        1.0,
    ] {
        e.set_property("knee", k);
        assert_eq!(e.property::<f32>("knee").to_bits(), k.to_bits());
    }
}
