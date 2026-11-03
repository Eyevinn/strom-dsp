//! Property-interface checker for the element interface tests
//! (`interface_deesser.rs`, `interface_cleanup.rs`), modelled on
//! `tests/interface.rs`: name, GLib type, range, default, flags (readable,
//! writable, controllable, mutable in PLAYING), no extra properties, klass,
//! pad templates, and exact read-back of every value tested.

#![allow(dead_code)]

use gst::glib;
use gst::prelude::*;

/// Expected type, range and default of one property.
#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    Bool(bool),
    /// (min, max, default)
    Float(f32, f32, f32),
    /// (min, max, default)
    UInt(u32, u32, u32),
    /// GLib enum: type name, every (value, nick), default value.
    Enum(&'static str, Vec<(i32, &'static str)>, i32),
}

/// One expected property (always mutable in PLAYING).
pub struct Prop {
    pub name: &'static str,
    pub kind: Kind,
}

pub fn p(name: &'static str, kind: Kind) -> Prop {
    Prop { name, kind }
}

const GST_FLAGS: [glib::ParamFlags; 4] = [
    gst::PARAM_FLAG_CONTROLLABLE,
    gst::PARAM_FLAG_MUTABLE_READY,
    gst::PARAM_FLAG_MUTABLE_PAUSED,
    gst::PARAM_FLAG_MUTABLE_PLAYING,
];

/// Initialise GStreamer and register the plugin statically (once per binary).
pub fn init() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        gst::init().unwrap();
        gststromdsp::plugin_register_static().expect("static registration failed");
    });
}

/// Integer value of a GLib enum `Value`.
fn enum_int(v: &glib::Value) -> i32 {
    glib::EnumValue::from_value(v)
        .map(|(_, e)| e.value())
        .expect("not an enum value")
}

/// Check every property of `factory` against `spec`, the GObject type name,
/// the klass and the pad templates.
pub fn check_element(factory: &str, type_name: &str, spec: &[Prop]) {
    init();
    let element = gst::ElementFactory::make(factory).build().unwrap();
    assert_eq!(element.type_().name(), type_name);
    let f = element.factory().unwrap();
    assert_eq!(f.metadata("klass"), Some("Filter/Effect/Audio"));

    let mut own: Vec<String> = element
        .list_properties()
        .iter()
        .filter(|ps| ps.owner_type() == element.type_())
        .map(|ps| ps.name().to_string())
        .collect();
    own.sort();
    let mut names: Vec<String> = spec.iter().map(|p| p.name.to_string()).collect();
    names.sort();
    assert_eq!(own, names, "{factory}: property set differs");

    for prop in spec {
        let name = prop.name;
        let ps = element.find_property(name).unwrap();
        let ctx = format!("{factory}.{name}");
        let flags = ps.flags();
        assert!(flags.contains(glib::ParamFlags::READABLE), "{ctx}");
        assert!(flags.contains(glib::ParamFlags::WRITABLE), "{ctx}");
        assert!(!flags.contains(glib::ParamFlags::CONSTRUCT_ONLY), "{ctx}");
        let actual_gst = GST_FLAGS
            .iter()
            .fold(glib::ParamFlags::empty(), |acc, f| acc | (flags & *f));
        assert_eq!(
            actual_gst,
            gst::PARAM_FLAG_CONTROLLABLE | gst::PARAM_FLAG_MUTABLE_PLAYING,
            "{ctx}: GStreamer flags"
        );

        let initial = element.property_value(name);
        let vt = ps.value_type();
        match &prop.kind {
            Kind::Bool(def) => {
                let ps = ps.downcast_ref::<glib::ParamSpecBoolean>().expect(&ctx);
                assert_eq!(vt, glib::Type::BOOL, "{ctx}");
                assert_eq!(ps.default_value(), *def, "{ctx}: default");
                assert_eq!(initial.get::<bool>().unwrap(), *def, "{ctx}: initial");
                for v in [!def, *def] {
                    element.set_property(name, v);
                    assert_eq!(element.property::<bool>(name), v, "{ctx}: read-back");
                }
            }
            Kind::Float(min, max, def) => {
                let ps = ps.downcast_ref::<glib::ParamSpecFloat>().expect(&ctx);
                assert_eq!(vt, glib::Type::F32, "{ctx}");
                assert_eq!(ps.minimum(), *min, "{ctx}: minimum");
                assert_eq!(ps.maximum(), *max, "{ctx}: maximum");
                assert_eq!(ps.default_value(), *def, "{ctx}: default");
                assert_eq!(initial.get::<f32>().unwrap(), *def, "{ctx}: initial");
                for v in [*min, *max, (min + max) * 0.5, min + (max - min) / 3.0, *def] {
                    element.set_property(name, v);
                    assert_eq!(
                        element.property::<f32>(name).to_bits(),
                        v.to_bits(),
                        "{ctx}: read-back"
                    );
                }
            }
            Kind::UInt(min, max, def) => {
                let ps = ps.downcast_ref::<glib::ParamSpecUInt>().expect(&ctx);
                assert_eq!(vt, glib::Type::U32, "{ctx}");
                assert_eq!(ps.minimum(), *min, "{ctx}: minimum");
                assert_eq!(ps.maximum(), *max, "{ctx}: maximum");
                assert_eq!(ps.default_value(), *def, "{ctx}: default");
                assert_eq!(initial.get::<u32>().unwrap(), *def, "{ctx}: initial");
                for v in (*min..=*max).chain([*def]) {
                    element.set_property(name, v);
                    assert_eq!(element.property::<u32>(name), v, "{ctx}: read-back");
                }
            }
            Kind::Enum(type_name, values, def) => {
                let ps = ps.downcast_ref::<glib::ParamSpecEnum>().expect(&ctx);
                assert_eq!(vt.name(), *type_name, "{ctx}: enum type");
                let class = ps.enum_class();
                let mut got: Vec<(i32, String)> = class
                    .values()
                    .iter()
                    .map(|v| (v.value(), v.nick().to_string()))
                    .collect();
                got.sort();
                let mut want: Vec<(i32, String)> =
                    values.iter().map(|(v, n)| (*v, n.to_string())).collect();
                want.sort();
                assert_eq!(got, want, "{ctx}: enum values");
                assert_eq!(ps.default_value_as_i32(), *def, "{ctx}: default");
                assert_eq!(enum_int(&initial), *def, "{ctx}: initial");
                // Every value can be set by nick (as gst-launch does) and reads back.
                for (v, nick) in values.iter().chain(values.iter().filter(|(v, _)| v == def)) {
                    element.set_property_from_str(name, nick);
                    let got = enum_int(&element.property_value(name));
                    assert_eq!(got, *v, "{ctx}: read-back of {nick}");
                }
            }
        }
    }

    // Pads: always sink and src with the shared audio caps.
    let expected: gst::Caps = "audio/x-raw, format=F32LE, layout=interleaved, \
                               rate=[1, 2147483647], channels=[1, 2147483647]"
        .parse()
        .unwrap();
    let templates = f.static_pad_templates();
    assert_eq!(templates.len(), 2, "{factory}");
    for t in templates {
        assert_eq!(t.presence(), gst::PadPresence::Always, "{factory}");
        assert!(t.caps() == expected, "{factory}: caps {}", t.caps());
    }
}
