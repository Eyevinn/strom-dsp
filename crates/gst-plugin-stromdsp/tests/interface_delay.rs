//! Interface test for `stromdsp-delay`, modelled on
//! `tests/interface.rs`: every property's name, GLib type, range, default and
//! flags, no extra properties, klass, pads, and that the element is never a
//! BaseTransform passthrough.

use crate::util;

use gst::glib;
use gst::prelude::*;
use gst_base::prelude::BaseTransformExt;

const FACTORY: &str = "stromdsp-delay";

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

fn assert_rw(ps: &glib::ParamSpec, controllable: bool) {
    let name = ps.name();
    let flags = ps.flags();
    assert!(flags.contains(glib::ParamFlags::READABLE), "{name}");
    assert!(flags.contains(glib::ParamFlags::WRITABLE), "{name}");
    assert!(!flags.contains(glib::ParamFlags::CONSTRUCT_ONLY), "{name}");
    let want = gst::PARAM_FLAG_MUTABLE_PLAYING
        | if controllable {
            gst::PARAM_FLAG_CONTROLLABLE
        } else {
            glib::ParamFlags::empty()
        };
    assert_eq!(gst_flags(flags), want, "{name}: GStreamer flags");
}

fn float(e: &gst::Element, name: &str, min: f32, max: f32, def: f32) {
    let ps = e.find_property(name).unwrap();
    assert_rw(&ps, true);
    let f = ps.downcast_ref::<glib::ParamSpecFloat>().expect(name);
    assert_eq!(ps.value_type(), glib::Type::F32, "{name}");
    assert_eq!(
        (f.minimum(), f.maximum(), f.default_value()),
        (min, max, def)
    );
    assert_eq!(e.property::<f32>(name), def, "{name}: initial");
    for v in [min, max, (min + max) / 2.0, def] {
        e.set_property(name, v);
        assert_eq!(e.property::<f32>(name).to_bits(), v.to_bits(), "{name}");
    }
}

#[test]
fn delay_interface() {
    util::init();
    let e = gst::ElementFactory::make(FACTORY).build().unwrap();
    assert_eq!(e.type_().name(), "StromDspDelay");
    let f = e.factory().unwrap();
    assert_eq!(f.metadata("klass"), Some("Filter/Effect/Audio"));
    let desc = f.metadata("description").unwrap();
    assert!(desc.contains("NOT latency"), "description: {desc}");

    let mut own: Vec<String> = e
        .list_properties()
        .iter()
        .filter(|ps| ps.owner_type() == e.type_())
        .map(|ps| ps.name().to_string())
        .collect();
    own.sort();
    assert_eq!(own, ["channel-delays", "delay", "enabled"]);

    // enabled: Boolean, default true.
    let ps = e.find_property("enabled").unwrap();
    assert_rw(&ps, true);
    let b = ps.downcast_ref::<glib::ParamSpecBoolean>().unwrap();
    assert!(b.default_value());
    assert!(e.property::<bool>("enabled"));
    for v in [false, true] {
        e.set_property("enabled", v);
        assert_eq!(e.property::<bool>("enabled"), v);
    }

    // delay: Float ms, 0 … 2000, default 0.
    float(&e, "delay", 0.0, 2000.0, 0.0);

    // channel-delays: GstValueArray of Float (ms, −2000 … 2000), default empty,
    // mutable in PLAYING, not controllable (arrays cannot be interpolated).
    let ps = e.find_property("channel-delays").unwrap();
    assert_rw(&ps, false);
    assert_eq!(ps.value_type(), gst::Array::static_type());
    let arr = ps.downcast_ref::<gst::ParamSpecArray>().unwrap();
    let el = arr.element_spec().expect("element spec");
    let el = el
        .downcast_ref::<glib::ParamSpecFloat>()
        .expect("float elements");
    assert_eq!(
        (el.minimum(), el.maximum(), el.default_value()),
        (-2000.0, 2000.0, 0.0)
    );
    assert!(e.property::<gst::Array>("channel-delays").is_empty());

    // Pads: any rate and channel count.
    let expected: gst::Caps =
        "audio/x-raw, format=F32LE, layout=interleaved, rate=[1, 2147483647], channels=[1, 2147483647]"
            .parse()
            .unwrap();
    let templates = f.static_pad_templates();
    assert_eq!(templates.len(), 2);
    for t in templates {
        assert_eq!(t.presence(), gst::PadPresence::Always);
        assert!(t.caps() == expected, "{}", t.caps());
    }

    // Never a BaseTransform passthrough: enabled=false is handled click-free
    // by the library (crossfade to delay 0, then bit-exact).
    let bt = e.downcast_ref::<gst_base::BaseTransform>().unwrap();
    assert!(!bt.is_passthrough());
    e.set_property("enabled", false);
    assert!(!bt.is_passthrough());
}
