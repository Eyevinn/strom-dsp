//! Interface test of the phase 1b gate additions (STROM_DSP_PHASE1B_SPEC.md
//! 3.3): `mode` (GLib enum `StromDspGateMode`, `gate` = 0 default,
//! `expander` = 1) and `ratio` (Float, 1 … 20, default 2). Both are
//! readable, writable, controllable and mutable in PLAYING. The phase 1
//! properties are checked, unchanged, by `tests/interface.rs`.

mod util;

use gst::glib;
use gst::prelude::*;
use gststromdsp::gate::StromDspGateMode;

fn gst_flags(ps: &glib::ParamSpec) -> glib::ParamFlags {
    ps.flags()
        & (gst::PARAM_FLAG_CONTROLLABLE
            | gst::PARAM_FLAG_MUTABLE_READY
            | gst::PARAM_FLAG_MUTABLE_PAUSED
            | gst::PARAM_FLAG_MUTABLE_PLAYING)
}

fn make() -> gst::Element {
    util::init();
    gst::ElementFactory::make("stromdsp-gate").build().unwrap()
}

#[test]
fn gate_has_exactly_phase1_plus_mode_and_ratio() {
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
            "close-threshold",
            "enabled",
            "hold",
            "mode",
            "open-threshold",
            "ratio",
            "reduction",
            "release"
        ]
    );
}

#[test]
fn gate_mode_property() {
    let e = make();
    let ps = e.find_property("mode").expect("mode");
    assert!(ps.flags().contains(glib::ParamFlags::READWRITE));
    assert!(!ps.flags().contains(glib::ParamFlags::CONSTRUCT_ONLY));
    assert_eq!(
        gst_flags(&ps),
        gst::PARAM_FLAG_CONTROLLABLE | gst::PARAM_FLAG_MUTABLE_PLAYING
    );
    assert_eq!(ps.value_type().name(), "StromDspGateMode");
    let ps = ps
        .downcast_ref::<glib::ParamSpecEnum>()
        .expect("a GLib enum");
    assert_eq!(
        ps.default_value::<StromDspGateMode>().unwrap(),
        StromDspGateMode::Gate
    );

    // Values and nicks are part of the interface.
    let class = ps.enum_class();
    let values: Vec<(i32, &str)> = class
        .values()
        .iter()
        .map(|v| (v.value(), v.nick()))
        .collect();
    assert_eq!(values, [(0, "gate"), (1, "expander")]);

    assert_eq!(
        e.property::<StromDspGateMode>("mode"),
        StromDspGateMode::Gate
    );
    e.set_property("mode", StromDspGateMode::Expander);
    assert_eq!(
        e.property::<StromDspGateMode>("mode"),
        StromDspGateMode::Expander
    );
    // Settable by nick, as from gst-launch-1.0.
    e.set_property_from_str("mode", "gate");
    assert_eq!(
        e.property::<StromDspGateMode>("mode"),
        StromDspGateMode::Gate
    );
    e.set_property_from_str("mode", "expander");
    assert_eq!(
        e.property::<StromDspGateMode>("mode"),
        StromDspGateMode::Expander
    );
}

#[test]
fn gate_ratio_property() {
    let e = make();
    let ps = e.find_property("ratio").expect("ratio");
    assert!(ps.flags().contains(glib::ParamFlags::READWRITE));
    assert_eq!(
        gst_flags(&ps),
        gst::PARAM_FLAG_CONTROLLABLE | gst::PARAM_FLAG_MUTABLE_PLAYING
    );
    assert_eq!(ps.value_type(), glib::Type::F32);
    let ps = ps
        .downcast_ref::<glib::ParamSpecFloat>()
        .expect("a Float (f32)");
    assert_eq!(
        (ps.minimum(), ps.maximum(), ps.default_value()),
        (1.0, 20.0, 2.0)
    );
    assert_eq!(e.property::<f32>("ratio"), 2.0);
    for v in [1.0f32, 20.0, 10.5, 1.0 + 19.0 / 3.0, 2.0] {
        e.set_property("ratio", v);
        assert_eq!(e.property::<f32>("ratio").to_bits(), v.to_bits());
    }
}
