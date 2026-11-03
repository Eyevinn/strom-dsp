//! Interface test for `stromdsp-stereo`, modelled on
//! `tests/interface.rs`: every property's name, GLib type, range or enum
//! values, default and flags, no extra properties, klass, 2-channel pads, and
//! that the element is never a BaseTransform passthrough.

mod util;

use gst::glib;
use gst::prelude::*;
use gst_base::prelude::BaseTransformExt;

const FACTORY: &str = "stromdsp-stereo";

fn gst_flags(flags: glib::ParamFlags) -> glib::ParamFlags {
    [
        gst::PARAM_FLAG_CONTROLLABLE,
        gst::PARAM_FLAG_MUTABLE_READY,
        gst::PARAM_FLAG_MUTABLE_PAUSED,
        gst::PARAM_FLAG_MUTABLE_PLAYING,
    ]
    .iter()
    .fold(glib::ParamFlags::empty(), |acc, f| acc | (flags & *f))
}

fn assert_flags(ps: &glib::ParamSpec) {
    let name = ps.name();
    let flags = ps.flags();
    assert!(flags.contains(glib::ParamFlags::READABLE), "{name}");
    assert!(flags.contains(glib::ParamFlags::WRITABLE), "{name}");
    assert!(!flags.contains(glib::ParamFlags::CONSTRUCT_ONLY), "{name}");
    assert_eq!(
        gst_flags(flags),
        gst::PARAM_FLAG_CONTROLLABLE | gst::PARAM_FLAG_MUTABLE_PLAYING,
        "{name}: GStreamer flags"
    );
}

fn boolean(e: &gst::Element, name: &str, def: bool) {
    let ps = e.find_property(name).unwrap();
    assert_flags(&ps);
    let b = ps.downcast_ref::<glib::ParamSpecBoolean>().expect(name);
    assert_eq!(ps.value_type(), glib::Type::BOOL);
    assert_eq!(b.default_value(), def, "{name}");
    assert_eq!(e.property::<bool>(name), def, "{name}: initial");
    for v in [!def, def] {
        e.set_property(name, v);
        assert_eq!(e.property::<bool>(name), v);
    }
}

fn float(e: &gst::Element, name: &str, min: f32, max: f32, def: f32) {
    let ps = e.find_property(name).unwrap();
    assert_flags(&ps);
    let f = ps.downcast_ref::<glib::ParamSpecFloat>().expect(name);
    assert_eq!(ps.value_type(), glib::Type::F32, "{name}");
    assert_eq!(
        (f.minimum(), f.maximum(), f.default_value()),
        (min, max, def)
    );
    assert_eq!(e.property::<f32>(name), def, "{name}: initial");
    for v in [min, max, (min + max) / 2.0, min + (max - min) / 3.0, def] {
        e.set_property(name, v);
        assert_eq!(e.property::<f32>(name).to_bits(), v.to_bits(), "{name}");
    }
}

/// GLib enum with exactly `values` as (value, nick) and default `def`.
fn enumeration(e: &gst::Element, name: &str, type_name: &str, values: &[(i32, &str)], def: i32) {
    let ps = e.find_property(name).unwrap();
    assert_flags(&ps);
    let en = ps.downcast_ref::<glib::ParamSpecEnum>().expect(name);
    assert_eq!(ps.value_type().name(), type_name, "{name}");
    assert_eq!(en.default_value_as_i32(), def, "{name}: default");
    let class = en.enum_class();
    let got: Vec<(i32, String)> = class
        .values()
        .iter()
        .map(|v| (v.value(), v.nick().to_string()))
        .collect();
    let want: Vec<(i32, String)> = values.iter().map(|(v, n)| (*v, n.to_string())).collect();
    assert_eq!(got, want, "{name}: enum values");
    for (v, nick) in values {
        e.set_property_from_str(name, nick);
        let val = e.property_value(name);
        assert_eq!(class.value_by_nick(nick).unwrap().value(), *v);
        assert_eq!(
            glib::EnumValue::from_value(&val).unwrap().1.value(),
            *v,
            "{name}: read-back {nick}"
        );
    }
    let def_nick = values.iter().find(|(v, _)| *v == def).unwrap().1;
    e.set_property_from_str(name, def_nick);
}

#[test]
fn stereo_interface() {
    util::init();
    let e = gst::ElementFactory::make(FACTORY).build().unwrap();
    assert_eq!(e.type_().name(), "StromDspStereo");
    let f = e.factory().unwrap();
    assert_eq!(f.metadata("klass"), Some("Filter/Effect/Audio"));

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
            "balance",
            "enabled",
            "invert-left",
            "invert-right",
            "mode",
            "mono-law",
            "width"
        ]
    );

    boolean(&e, "enabled", true);
    enumeration(
        &e,
        "mode",
        "StromDspStereoMode",
        &[
            (0, "stereo"),
            (1, "mono"),
            (2, "left-to-both"),
            (3, "right-to-both"),
            (4, "swap"),
            (5, "ms-encode"),
            (6, "ms-decode"),
        ],
        0,
    );
    enumeration(
        &e,
        "mono-law",
        "StromDspStereoMonoLaw",
        &[(0, "minus-3db"), (1, "minus-6db")],
        0,
    );
    boolean(&e, "invert-left", false);
    boolean(&e, "invert-right", false);
    float(&e, "balance", -1.0, 1.0, 0.0);
    float(&e, "width", 0.0, 2.0, 1.0);

    // Pads: any rate, exactly 2 channels.
    let expected: gst::Caps =
        "audio/x-raw, format=F32LE, layout=interleaved, rate=[1, 2147483647], channels=2"
            .parse()
            .unwrap();
    let templates = f.static_pad_templates();
    assert_eq!(templates.len(), 2);
    for t in templates {
        assert_eq!(t.presence(), gst::PadPresence::Always);
        assert!(t.caps() == expected, "{}", t.caps());
    }

    // Never a BaseTransform passthrough (enabled=false ramps in the library).
    let bt = e.downcast_ref::<gst_base::BaseTransform>().unwrap();
    assert!(!bt.is_passthrough());
    e.set_property("enabled", false);
    assert!(!bt.is_passthrough());
}
