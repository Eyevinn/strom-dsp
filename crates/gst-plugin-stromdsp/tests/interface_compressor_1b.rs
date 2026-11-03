//! Interface test of the compressor additions after the listening tests:
//! `detector` (GLib enum `StromDspCompressorDetector`, `peak` = 0,
//! `rms` = 1 default), `rms-window` (Float, ms, 1 … 100, default 10) and
//! `auto-makeup` (Boolean, default false). The defaults are the broadcast
//! preset (`strom_dsp::presets`). All readable, writable,
//! controllable and mutable in PLAYING. The phase 1 properties are checked
//! by `tests/interface.rs`.

mod util;

use gst::glib;
use gst::prelude::*;
use gststromdsp::compressor::StromDspCompressorDetector;

fn gst_flags(ps: &glib::ParamSpec) -> glib::ParamFlags {
    ps.flags()
        & (gst::PARAM_FLAG_CONTROLLABLE
            | gst::PARAM_FLAG_MUTABLE_READY
            | gst::PARAM_FLAG_MUTABLE_PAUSED
            | gst::PARAM_FLAG_MUTABLE_PLAYING)
}

fn make() -> gst::Element {
    util::init();
    gst::ElementFactory::make("stromdsp-compressor")
        .build()
        .unwrap()
}

fn check_flags(ps: &glib::ParamSpec) {
    assert!(ps.flags().contains(glib::ParamFlags::READWRITE));
    assert!(!ps.flags().contains(glib::ParamFlags::CONSTRUCT_ONLY));
    assert_eq!(
        gst_flags(ps),
        gst::PARAM_FLAG_CONTROLLABLE | gst::PARAM_FLAG_MUTABLE_PLAYING
    );
}

#[test]
fn compressor_has_exactly_phase1_plus_additions() {
    let e = make();
    let mut own: Vec<String> = e
        .list_properties()
        .iter()
        .filter(|ps| ps.owner_type() == e.type_())
        .map(|ps| ps.name().to_string())
        .collect();
    own.sort();
    assert_eq!(
        own,
        [
            "attack",
            "auto-makeup",
            "detector",
            "enabled",
            "knee",
            "makeup-gain",
            "ratio",
            "release",
            "rms-window",
            "threshold"
        ]
    );
}

#[test]
fn compressor_detector_property() {
    let e = make();
    let ps = e.find_property("detector").expect("detector");
    check_flags(&ps);
    assert_eq!(ps.value_type().name(), "StromDspCompressorDetector");
    let ps = ps
        .downcast_ref::<glib::ParamSpecEnum>()
        .expect("a GLib enum");
    assert_eq!(
        ps.default_value::<StromDspCompressorDetector>().unwrap(),
        StromDspCompressorDetector::Rms
    );
    let values: Vec<(i32, &str)> = ps
        .enum_class()
        .values()
        .iter()
        .map(|v| (v.value(), v.nick()))
        .collect();
    assert_eq!(values, [(0, "peak"), (1, "rms")]);

    assert_eq!(
        e.property::<StromDspCompressorDetector>("detector"),
        StromDspCompressorDetector::Rms
    );
    e.set_property("detector", StromDspCompressorDetector::Peak);
    assert_eq!(
        e.property::<StromDspCompressorDetector>("detector"),
        StromDspCompressorDetector::Peak
    );
    // Settable by nick, as from gst-launch-1.0.
    for (nick, v) in [
        ("peak", StromDspCompressorDetector::Peak),
        ("rms", StromDspCompressorDetector::Rms),
    ] {
        e.set_property_from_str("detector", nick);
        assert_eq!(e.property::<StromDspCompressorDetector>("detector"), v);
    }
}

#[test]
fn compressor_rms_window_property() {
    let e = make();
    let ps = e.find_property("rms-window").expect("rms-window");
    check_flags(&ps);
    assert_eq!(ps.value_type(), glib::Type::F32);
    let ps = ps
        .downcast_ref::<glib::ParamSpecFloat>()
        .expect("a Float (f32)");
    assert_eq!(
        (ps.minimum(), ps.maximum(), ps.default_value()),
        (1.0, 100.0, 10.0)
    );
    assert_eq!(e.property::<f32>("rms-window"), 10.0);
    for v in [1.0f32, 100.0, 50.5, 1.0 + 99.0 / 3.0, 10.0] {
        e.set_property("rms-window", v);
        assert_eq!(e.property::<f32>("rms-window").to_bits(), v.to_bits());
    }
}

#[test]
fn compressor_auto_makeup_property() {
    let e = make();
    let ps = e.find_property("auto-makeup").expect("auto-makeup");
    check_flags(&ps);
    assert_eq!(ps.value_type(), glib::Type::BOOL);
    let ps = ps
        .downcast_ref::<glib::ParamSpecBoolean>()
        .expect("a Boolean");
    assert!(!ps.default_value());
    assert!(!e.property::<bool>("auto-makeup"));
    for v in [true, false] {
        e.set_property("auto-makeup", v);
        assert_eq!(e.property::<bool>("auto-makeup"), v);
    }
}
