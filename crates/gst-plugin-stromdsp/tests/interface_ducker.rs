//! Interface test for `stromdsp-ducker`, modelled on `tests/interface.rs`:
//! every property's name, GLib type, range, default and flags (readable,
//! writable, controllable, mutable in PLAYING), that no other own properties
//! exist, read-back of in-range values, the `range-mode` enum, the element
//! metadata and the pad templates. This is the contract documented in the
//! README; the test fails if anything drifts from it.

mod util;

use gst::glib;
use gst::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    Bool(bool),
    /// (min, max, default)
    Float(f32, f32, f32),
    UInt(u32, u32, u32),
    /// GLib enum type name, default nick, (value, nick) pairs.
    Enum(&'static str, &'static str, &'static [(i32, &'static str)]),
}

use Kind::*;

struct Prop {
    name: &'static str,
    kind: Kind,
    controllable: bool,
}

const fn p(name: &'static str, kind: Kind) -> Prop {
    Prop {
        name,
        kind,
        controllable: true,
    }
}

const RANGE_MODES: &[(i32, &str)] = &[(0, "fixed"), (1, "proportional")];

/// The documented interface.
fn ducker_spec() -> Vec<Prop> {
    vec![
        p("enabled", Bool(true)),
        p("threshold", Float(-80.0, 0.0, -30.0)),
        p("depth", Float(-80.0, 0.0, -12.0)),
        p("attack", Float(0.1, 2000.0, 20.0)),
        p("hold", Float(0.0, 5000.0, 300.0)),
        p("release", Float(1.0, 10_000.0, 500.0)),
        p("key-highpass", Float(0.0, 20_000.0, 100.0)),
        p(
            "range-mode",
            Enum("StromDspDuckerRangeMode", "fixed", RANGE_MODES),
        ),
        Prop {
            name: "meter-interval",
            kind: UInt(0, 10_000, 100),
            controllable: false,
        },
    ]
}

const GST_FLAGS: [glib::ParamFlags; 4] = [
    gst::PARAM_FLAG_CONTROLLABLE,
    gst::PARAM_FLAG_MUTABLE_READY,
    gst::PARAM_FLAG_MUTABLE_PAUSED,
    gst::PARAM_FLAG_MUTABLE_PLAYING,
];

#[test]
fn ducker_properties() {
    util::init();
    let factory = "stromdsp-ducker";
    let element = gst::ElementFactory::make(factory).build().unwrap();
    assert_eq!(element.type_().name(), "StromDspDucker");
    assert!(element.is::<gst_base::Aggregator>());
    let f = element.factory().unwrap();
    assert_eq!(f.metadata("klass"), Some("Filter/Effect/Audio"));

    // Exactly the documented own properties (GstAggregator's are inherited).
    let spec = ducker_spec();
    let mut own: Vec<String> = element
        .list_properties()
        .iter()
        .filter(|ps| ps.owner_type() == element.type_())
        .map(|ps| ps.name().to_string())
        .collect();
    own.sort();
    let mut names: Vec<String> = spec.iter().map(|p| p.name.to_string()).collect();
    names.sort();
    assert_eq!(own, names, "property set differs");

    for prop in &spec {
        let name = prop.name;
        let ctx = format!("{factory}.{name}");
        let ps = element.find_property(name).unwrap();
        let flags = ps.flags();
        assert!(
            flags.contains(glib::ParamFlags::READWRITE),
            "{ctx}: not read/write"
        );
        assert!(!flags.contains(glib::ParamFlags::CONSTRUCT_ONLY), "{ctx}");
        let mut expected = gst::PARAM_FLAG_MUTABLE_PLAYING;
        if prop.controllable {
            expected |= gst::PARAM_FLAG_CONTROLLABLE;
        }
        let actual = GST_FLAGS
            .iter()
            .fold(glib::ParamFlags::empty(), |acc, f| acc | (flags & *f));
        assert_eq!(actual, expected, "{ctx}: GStreamer flags");

        let initial = element.property_value(name);
        let base_type = ps.value_type();
        match prop.kind {
            Bool(def) => {
                let ps = ps
                    .downcast_ref::<glib::ParamSpecBoolean>()
                    .expect("Boolean");
                assert_eq!(ps.default_value(), def, "{ctx}");
                assert_eq!(initial.get::<bool>().unwrap(), def, "{ctx}");
                for v in [!def, def] {
                    element.set_property(name, v);
                    assert_eq!(element.property::<bool>(name), v, "{ctx}");
                }
            }
            Float(min, max, def) => {
                let ps = ps
                    .downcast_ref::<glib::ParamSpecFloat>()
                    .expect("Float (f32)");
                assert_eq!(base_type, glib::Type::F32, "{ctx}");
                assert_eq!(ps.minimum(), min, "{ctx}: minimum");
                assert_eq!(ps.maximum(), max, "{ctx}: maximum");
                assert_eq!(ps.default_value(), def, "{ctx}: default");
                assert_eq!(initial.get::<f32>().unwrap(), def, "{ctx}: initial");
                for v in [min, max, (min + max) * 0.5, def] {
                    element.set_property(name, v);
                    assert_eq!(
                        element.property::<f32>(name).to_bits(),
                        v.to_bits(),
                        "{ctx}"
                    );
                }
            }
            UInt(min, max, def) => {
                let ps = ps.downcast_ref::<glib::ParamSpecUInt>().expect("UInt");
                assert_eq!(base_type, glib::Type::U32, "{ctx}");
                assert_eq!(ps.minimum(), min, "{ctx}: minimum");
                assert_eq!(ps.maximum(), max, "{ctx}: maximum");
                assert_eq!(ps.default_value(), def, "{ctx}: default");
                assert_eq!(initial.get::<u32>().unwrap(), def, "{ctx}: initial");
                for v in [min, max, def] {
                    element.set_property(name, v);
                    assert_eq!(element.property::<u32>(name), v, "{ctx}");
                }
            }
            Enum(type_name, def_nick, values) => {
                let ps = ps.downcast_ref::<glib::ParamSpecEnum>().expect("Enum");
                assert_eq!(base_type.name(), type_name, "{ctx}: enum type");
                let class = ps.enum_class();
                let got: Vec<(i32, String)> = class
                    .values()
                    .iter()
                    .map(|v| (v.value(), v.nick().to_string()))
                    .collect();
                let want: Vec<(i32, String)> =
                    values.iter().map(|(v, n)| (*v, n.to_string())).collect();
                assert_eq!(got, want, "{ctx}: enum values");
                let base: &glib::ParamSpec = ps.as_ref();
                let def = glib::EnumValue::from_value(base.default_value()).unwrap().1;
                assert_eq!(def.nick(), def_nick, "{ctx}: default");
                // Also typed, through the public Rust enum.
                assert_eq!(
                    ps.default_value::<gststromdsp::ducker::RangeMode>()
                        .unwrap(),
                    gststromdsp::ducker::RangeMode::Fixed
                );
                let init = glib::EnumValue::from_value(&initial).unwrap().1;
                assert_eq!(init.nick(), def_nick, "{ctx}: initial");
                for (_, nick) in values.iter().chain([&(0, def_nick)]) {
                    element.set_property_from_str(name, nick);
                    let v = element.property_value(name);
                    assert_eq!(glib::EnumValue::from_value(&v).unwrap().1.nick(), *nick);
                }
            }
        }
    }
}

#[test]
fn ducker_pad_templates() {
    util::init();
    let f = gst::ElementFactory::find("stromdsp-ducker").unwrap();
    let caps: gst::Caps =
        "audio/x-raw, format=F32LE, layout=interleaved, rate=[1, 2147483647], channels=[1, 2147483647]"
            .parse()
            .unwrap();
    let mut t: Vec<(String, gst::PadDirection, gst::PadPresence)> = f
        .static_pad_templates()
        .iter()
        .map(|t| {
            assert!(t.caps() == caps, "{}: caps {}", t.name_template(), t.caps());
            (t.name_template().to_string(), t.direction(), t.presence())
        })
        .collect();
    t.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        t,
        vec![
            (
                "sidechain".to_string(),
                gst::PadDirection::Sink,
                gst::PadPresence::Request
            ),
            (
                "sink".to_string(),
                gst::PadDirection::Sink,
                gst::PadPresence::Always
            ),
            (
                "src".to_string(),
                gst::PadDirection::Src,
                gst::PadPresence::Always
            ),
        ]
    );

    // The always pads exist; one sidechain can be requested, not two; it can
    // be released and requested again.
    let e = f.create().build().unwrap();
    assert!(e.static_pad("sink").is_some() && e.static_pad("src").is_some());
    let side = e.request_pad_simple("sidechain").unwrap();
    assert_eq!(side.name(), "sidechain");
    assert!(side.is::<gst_base::AggregatorPad>());
    let templ = e.pad_template("sidechain").unwrap();
    assert!(
        e.request_pad(&templ, None, None).is_none(),
        "second sidechain"
    );
    e.release_request_pad(&side);
    assert!(e.request_pad_simple("sidechain").is_some());
}
