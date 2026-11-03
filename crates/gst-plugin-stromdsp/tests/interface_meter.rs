//! Interface test of `stromdsp-meter` (phase 1b spec §3.7): names,
//! GLib types, ranges, defaults and flags of every property, the `reset`
//! action signal, the pad templates, metadata and the passthrough mode.

mod util;

use gst::glib;
use gst::prelude::*;
use gst_base::prelude::BaseTransformExt;

const GST_FLAGS: glib::ParamFlags = gst::PARAM_FLAG_CONTROLLABLE
    .union(gst::PARAM_FLAG_MUTABLE_READY)
    .union(gst::PARAM_FLAG_MUTABLE_PAUSED)
    .union(gst::PARAM_FLAG_MUTABLE_PLAYING);

fn make() -> gst::Element {
    util::init();
    gst::ElementFactory::make("stromdsp-meter").build().unwrap()
}

fn own_properties(e: &gst::Element) -> Vec<String> {
    let mut v: Vec<String> = e
        .list_properties()
        .iter()
        .filter(|ps| ps.owner_type() == e.type_())
        .map(|ps| ps.name().to_string())
        .collect();
    v.sort();
    v
}

fn check_rw(ps: &glib::ParamSpec, mutable: glib::ParamFlags) {
    let flags = ps.flags();
    assert!(flags.contains(glib::ParamFlags::READWRITE), "{}", ps.name());
    assert!(!flags.contains(glib::ParamFlags::CONSTRUCT_ONLY));
    assert_eq!(flags & GST_FLAGS, mutable, "{}: GStreamer flags", ps.name());
}

#[test]
fn meter_identity_and_metadata() {
    let e = make();
    assert_eq!(e.type_().name(), "StromDspMeter");
    let f = e.factory().unwrap();
    assert_eq!(f.metadata("klass"), Some("Filter/Analyzer/Audio"));
    assert_eq!(own_properties(&e), ["interval", "true-peak"]);
    // Always a passthrough element.
    let bt = e.downcast_ref::<gst_base::BaseTransform>().unwrap();
    assert!(bt.is_passthrough());
}

#[test]
fn meter_properties() {
    let e = make();

    let ps = e.find_property("interval").unwrap();
    check_rw(&ps, gst::PARAM_FLAG_MUTABLE_PLAYING);
    assert_eq!(ps.value_type(), glib::Type::U32);
    let ps = ps.downcast_ref::<glib::ParamSpecUInt>().expect("UInt");
    assert_eq!(
        (ps.minimum(), ps.maximum(), ps.default_value()),
        (10, 60_000, 100)
    );
    assert_eq!(e.property::<u32>("interval"), 100);
    for v in [10u32, 60_000, 250] {
        e.set_property("interval", v);
        assert_eq!(e.property::<u32>("interval"), v);
    }

    let ps = e.find_property("true-peak").unwrap();
    check_rw(&ps, gst::PARAM_FLAG_MUTABLE_READY);
    assert_eq!(ps.value_type(), glib::Type::BOOL);
    let ps = ps
        .downcast_ref::<glib::ParamSpecBoolean>()
        .expect("Boolean");
    assert!(ps.default_value());
    assert!(e.property::<bool>("true-peak"));
    e.set_property("true-peak", false);
    assert!(!e.property::<bool>("true-peak"));
}

#[test]
fn meter_reset_signal() {
    let e = make();
    let id = glib::subclass::SignalId::lookup("reset", e.type_()).expect("reset signal");
    let q = id.query();
    assert!(
        q.flags().contains(glib::SignalFlags::ACTION),
        "action signal"
    );
    assert!(q.param_types().is_empty(), "no arguments");
    assert_eq!(q.return_type().type_(), glib::Type::UNIT, "no return value");
    // Emitting it outside a running pipeline is harmless.
    e.emit_by_name::<()>("reset", &[]);
}

#[test]
fn meter_pads() {
    util::init();
    let f = gst::ElementFactory::find("stromdsp-meter").unwrap();
    let expected: gst::Caps =
        "audio/x-raw, format=F32LE, layout=interleaved, rate=[1, 2147483647], channels=[1, 64]"
            .parse()
            .unwrap();
    let templates = f.static_pad_templates();
    assert_eq!(templates.len(), 2);
    for t in templates {
        assert_eq!(t.presence(), gst::PadPresence::Always);
        assert!(
            t.caps() == expected,
            "{} caps {}",
            t.name_template(),
            t.caps()
        );
    }
}
