//! Harness tests for `stromdsp-stereo`: output matches the library
//! through mode switches and parameter changes, the defaults and
//! `enabled=false` are bit-exact, routing modes are bit-exact, caps other than
//! 2 channels are refused, and the latency query is unchanged.

use crate::util;

use crate::util::Action;
use gst::glib;
use gst::prelude::*;
use gst_check::Harness;
use gststromdsp::stereo::{MonoLaw, StereoMode};
use strom_dsp::stereo::{Stereo, StereoParams};

const FORMATS: [u32; 2] = [48_000, 22_050];

fn on_set(dsp: &mut Stereo, name: &str, value: &glib::Value) {
    let mut p = *dsp.params();
    match name {
        "enabled" => p.enabled = value.get().unwrap(),
        "mode" => p.mode = value.get::<StereoMode>().unwrap().into(),
        "mono-law" => p.mono_law = value.get::<MonoLaw>().unwrap().into(),
        "invert-left" => p.invert_left = value.get().unwrap(),
        "invert-right" => p.invert_right = value.get().unwrap(),
        "balance" => p.balance = value.get().unwrap(),
        "width" => p.width = value.get().unwrap(),
        _ => unreachable!(),
    }
    dsp.set_params(&p);
}

#[test]
fn stereo_matches_library() {
    util::init();
    for rate in FORMATS {
        let input = util::test_signal(rate, 2, rate as usize);
        let props = vec![
            ("width", 1.5f32.to_value()),
            ("balance", (-0.25f32).to_value()),
        ];
        let actions = [
            (2, Action::SetProperty("mode", StereoMode::Mono.to_value())),
            (
                3,
                Action::SetProperty("mono-law", MonoLaw::Minus6Db.to_value()),
            ),
            (4, Action::SetProperty("invert-right", true.to_value())),
            (5, Action::Discont),
            (
                6,
                Action::SetProperty("mode", StereoMode::MsEncode.to_value()),
            ),
            (8, Action::SetProperty("mode", StereoMode::Swap.to_value())),
            (9, Action::SetProperty("width", 0.0f32.to_value())),
            (10, Action::SetProperty("enabled", false.to_value())),
            (12, Action::Flush),
            (13, Action::SetProperty("enabled", true.to_value())),
            (14, Action::SetProperty("balance", 1.0f32.to_value())),
        ];
        let out = util::run_element("stromdsp-stereo", &props, rate, 2, &input, 1024, &actions);
        let mut dsp = Stereo::new();
        dsp.set_params(&StereoParams {
            width: 1.5,
            balance: -0.25,
            ..StereoParams::default()
        });
        let reference = util::run_reference(dsp, rate, 2, &input, 1024, &actions, on_set);
        util::assert_bit_exact(&out, &reference);
    }
}

#[test]
fn defaults_and_disabled_are_bit_exact() {
    util::init();
    for rate in FORMATS {
        let mut input = util::test_signal(rate, 2, rate as usize / 2);
        input[7] = f32::from_bits(0x7fc0_0042); // NaN with payload
        input[8] = -0.0;
        let out = util::run_element("stromdsp-stereo", &[], rate, 2, &input, 480, &[]);
        util::assert_bit_exact(&out, &input);
        let props = vec![
            ("enabled", false.to_value()),
            ("mode", StereoMode::Mono.to_value()),
            ("width", 2.0f32.to_value()),
        ];
        let out = util::run_element("stromdsp-stereo", &props, rate, 2, &input, 480, &[]);
        util::assert_bit_exact(&out, &input);
    }
}

#[test]
fn routing_is_bit_exact() {
    util::init();
    let input = util::test_signal(48_000, 2, 9600);
    let props = vec![
        ("mode", StereoMode::Swap.to_value()),
        ("invert-left", true.to_value()),
    ];
    let out = util::run_element("stromdsp-stereo", &props, 48_000, 2, &input, 480, &[]);
    for (o, i) in out.as_chunks::<2>().0.iter().zip(input.as_chunks::<2>().0) {
        // invert-left acts on the input, then swap: L out = R in, R out = −L in.
        assert_eq!(o[0].to_bits(), i[1].to_bits());
        assert_eq!(o[1].to_bits(), (-i[0]).to_bits());
    }
}

#[test]
fn enum_properties_from_strings() {
    util::init();
    let e = gst::ElementFactory::make("stromdsp-stereo")
        .build()
        .unwrap();
    for (nick, v) in [
        ("stereo", StereoMode::Stereo),
        ("mono", StereoMode::Mono),
        ("left-to-both", StereoMode::LeftToBoth),
        ("right-to-both", StereoMode::RightToBoth),
        ("swap", StereoMode::Swap),
        ("ms-encode", StereoMode::MsEncode),
        ("ms-decode", StereoMode::MsDecode),
    ] {
        e.set_property_from_str("mode", nick);
        assert_eq!(e.property::<StereoMode>("mode"), v);
    }
    e.set_property_from_str("mono-law", "minus-6db");
    assert_eq!(e.property::<MonoLaw>("mono-law"), MonoLaw::Minus6Db);
    e.set_property_from_str("mono-law", "minus-3db");
    assert_eq!(e.property::<MonoLaw>("mono-law"), MonoLaw::Minus3Db);
}

#[test]
fn refuses_other_channel_counts() {
    util::init();
    for channels in [1usize, 3] {
        let mut h = Harness::new("stromdsp-stereo");
        h.set_src_caps_str(&util::caps_str(48_000, channels));
        h.play();
        let buf = gst::Buffer::from_mut_slice(vec![0u8; 64 * channels * 4]);
        assert!(h.push(buf).is_err(), "{channels} channels accepted");
    }
}

#[test]
fn latency_query_is_unchanged() {
    util::init();
    let mut h = Harness::new("stromdsp-stereo");
    let upstream = gst::ClockTime::from_mseconds(7);
    h.set_upstream_latency(upstream);
    h.set_src_caps_str(&util::caps_str(48_000, 2));
    h.play();
    let buf = gst::Buffer::from_mut_slice(vec![0u8; 64 * 2 * 4]);
    assert_eq!(h.push(buf), Ok(gst::FlowSuccess::Ok));
    let _ = h.pull().unwrap();
    assert_eq!(h.query_latency(), Some(upstream));
}
