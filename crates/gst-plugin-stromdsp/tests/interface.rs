//! Interface test: the drop-in contract of phase 1 spec §4.
//!
//! The tables below are a verbatim transcription of section 4, except the
//! defaults, which are strom-dsp's own broadcast choice (§4, revised
//! 2026-11-03; `strom_dsp::presets`, README "Why these defaults"). Every property
//! of every element is checked for name, GLib value type, range, default,
//! flags (readable, writable, controllable, mutable in PLAYING vs READY), and
//! that no extra properties exist. If anything drifts from section 4, this
//! test fails.

mod util;

use gst::glib;
use gst::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    Bool(bool),
    /// (min, max, default)
    Float(f32, f32, f32),
    UInt(u32, u32, u32),
    Int(i32, i32, i32),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Mutable {
    Playing,
    Ready,
}

use Kind::*;
use Mutable::*;

struct Prop {
    name: String,
    kind: Kind,
    mutable: Mutable,
}

fn p(name: &str, kind: Kind) -> Prop {
    Prop {
        name: name.to_string(),
        kind,
        mutable: Playing,
    }
}

/// Section 4.1.
fn gate_spec() -> Vec<Prop> {
    vec![
        p("enabled", Bool(true)),
        p("open-threshold", Float(-80.0, 0.0, -45.0)),
        p("close-threshold", Float(-80.0, 0.0, -50.0)),
        p("attack", Float(0.01, 200.0, 3.0)),
        p("hold", Float(0.0, 1000.0, 120.0)),
        p("release", Float(1.0, 2000.0, 200.0)),
        p("reduction", Float(-80.0, 0.0, -35.0)),
    ]
}

/// Section 4.2.
fn compressor_spec() -> Vec<Prop> {
    vec![
        p("enabled", Bool(true)),
        p("threshold", Float(0.0, 1.0, 0.125)),
        p("ratio", Float(1.0, 100.0, 3.0)),
        p("attack", Float(0.01, 10000.0, 10.0)),
        p("release", Float(0.01, 10000.0, 150.0)),
        p("makeup-gain", Float(0.0, 100.0, 1.0)),
        p("knee", Float(0.0, 1.0, 0.5)),
    ]
}

/// Section 4.3.
fn equalizer_spec() -> Vec<Prop> {
    let mut v = vec![
        p("enabled", Bool(true)),
        Prop {
            name: "num-bands".into(),
            kind: UInt(1, 8, 4),
            mutable: Ready,
        },
    ];
    for n in 0..8 {
        v.push(p(&format!("band{n}-enabled"), Bool(true)));
        v.push(p(&format!("band{n}-type"), Int(0, 9, 0)));
        v.push(p(
            &format!("band{n}-frequency"),
            Float(1.0, 24000.0, 1000.0),
        ));
        v.push(p(&format!("band{n}-gain"), Float(-60.0, 60.0, 0.0)));
        v.push(p(&format!("band{n}-q"), Float(0.01, 100.0, 0.707)));
    }
    v
}

/// Section 4.4.
fn limiter_spec() -> Vec<Prop> {
    vec![
        p("enabled", Bool(true)),
        p("threshold", Float(-60.0, 0.0, -1.0)),
        p("attack", Float(0.01, 50.0, 1.0)),
        p("release", Float(1.0, 2000.0, 200.0)),
        p("lookahead", Float(0.0, 20.0, 5.0)),
    ]
}

const GST_FLAGS: [glib::ParamFlags; 4] = [
    gst::PARAM_FLAG_CONTROLLABLE,
    gst::PARAM_FLAG_MUTABLE_READY,
    gst::PARAM_FLAG_MUTABLE_PAUSED,
    gst::PARAM_FLAG_MUTABLE_PLAYING,
];

/// Properties added by the phase 1b spec (additive only). These are
/// the only extra properties allowed on a phase 1 element.
fn phase1b_additions(factory: &str) -> &'static [&'static str] {
    match factory {
        "stromdsp-gate" => &["mode", "ratio"],
        "stromdsp-compressor" => &["detector", "rms-window", "auto-makeup"],
        _ => &[],
    }
}

fn check_element(factory: &str, type_name: &str, spec: &[Prop]) {
    util::init();
    let element = gst::ElementFactory::make(factory).build().unwrap();
    assert_eq!(element.type_().name(), type_name);
    let f = element.factory().unwrap();
    assert_eq!(f.metadata("klass"), Some("Filter/Effect/Audio"));

    // Exactly the spec's properties, apart from those inherited from
    // GstObject/GstBaseTransform and the documented phase 1b additions
    // (checked in full by `tests/interface_gate_1b.rs`).
    let own: Vec<String> = element
        .list_properties()
        .iter()
        .filter(|ps| ps.owner_type() == element.type_())
        .map(|ps| ps.name().to_string())
        .filter(|name| !phase1b_additions(factory).contains(&name.as_str()))
        .collect();
    let mut own_sorted = own.clone();
    own_sorted.sort();
    let mut spec_names: Vec<String> = spec.iter().map(|p| p.name.clone()).collect();
    spec_names.sort();
    assert_eq!(own_sorted, spec_names, "{factory}: property set differs");

    for prop in spec {
        let name = prop.name.as_str();
        let ps = element
            .find_property(name)
            .unwrap_or_else(|| panic!("{factory}: missing property {name}"));
        let ctx = format!("{factory}.{name}");

        // Flags.
        let flags = ps.flags();
        assert!(
            flags.contains(glib::ParamFlags::READABLE),
            "{ctx}: not readable"
        );
        assert!(
            flags.contains(glib::ParamFlags::WRITABLE),
            "{ctx}: not writable"
        );
        assert!(
            !flags.contains(glib::ParamFlags::CONSTRUCT_ONLY),
            "{ctx}: construct-only"
        );
        let expected_gst = gst::PARAM_FLAG_CONTROLLABLE
            | match prop.mutable {
                Playing => gst::PARAM_FLAG_MUTABLE_PLAYING,
                Ready => gst::PARAM_FLAG_MUTABLE_READY,
            };
        let actual_gst = GST_FLAGS
            .iter()
            .fold(glib::ParamFlags::empty(), |acc, f| acc | (flags & *f));
        assert_eq!(actual_gst, expected_gst, "{ctx}: GStreamer flags");

        // Type, range, default; and the element's initial value equals the default.
        let initial = element.property_value(name);
        let vt = ps.value_type();
        match prop.kind {
            Bool(def) => {
                let ps = ps
                    .downcast_ref::<glib::ParamSpecBoolean>()
                    .unwrap_or_else(|| panic!("{ctx}: not a Boolean"));
                assert_eq!(vt, glib::Type::BOOL, "{ctx}");
                assert_eq!(ps.default_value(), def, "{ctx}: default");
                assert_eq!(initial.get::<bool>().unwrap(), def, "{ctx}: initial");
            }
            Float(min, max, def) => {
                let ps = ps
                    .downcast_ref::<glib::ParamSpecFloat>()
                    .unwrap_or_else(|| panic!("{ctx}: not a Float (f32)"));
                assert_eq!(vt, glib::Type::F32, "{ctx}");
                assert_eq!(ps.minimum(), min, "{ctx}: minimum");
                assert_eq!(ps.maximum(), max, "{ctx}: maximum");
                assert_eq!(ps.default_value(), def, "{ctx}: default");
                assert_eq!(initial.get::<f32>().unwrap(), def, "{ctx}: initial");
            }
            UInt(min, max, def) => {
                let ps = ps
                    .downcast_ref::<glib::ParamSpecUInt>()
                    .unwrap_or_else(|| panic!("{ctx}: not a UInt"));
                assert_eq!(vt, glib::Type::U32, "{ctx}");
                assert_eq!(ps.minimum(), min, "{ctx}: minimum");
                assert_eq!(ps.maximum(), max, "{ctx}: maximum");
                assert_eq!(ps.default_value(), def, "{ctx}: default");
                assert_eq!(initial.get::<u32>().unwrap(), def, "{ctx}: initial");
            }
            Int(min, max, def) => {
                let ps = ps
                    .downcast_ref::<glib::ParamSpecInt>()
                    .unwrap_or_else(|| panic!("{ctx}: not an Int (and not an enum)"));
                assert_eq!(vt, glib::Type::I32, "{ctx}");
                assert_eq!(ps.minimum(), min, "{ctx}: minimum");
                assert_eq!(ps.maximum(), max, "{ctx}: maximum");
                assert_eq!(ps.default_value(), def, "{ctx}: default");
                assert_eq!(initial.get::<i32>().unwrap(), def, "{ctx}: initial");
            }
        }

        // Every value in range, including the extremes, reads back exactly.
        match prop.kind {
            Bool(def) => {
                for v in [!def, def] {
                    element.set_property(name, v);
                    assert_eq!(element.property::<bool>(name), v, "{ctx}: read-back");
                }
            }
            Float(min, max, def) => {
                for v in [min, max, (min + max) * 0.5, min + (max - min) / 3.0, def] {
                    element.set_property(name, v);
                    assert_eq!(
                        element.property::<f32>(name).to_bits(),
                        v.to_bits(),
                        "{ctx}: read-back"
                    );
                }
            }
            UInt(min, max, def) => {
                for v in [min, max, def] {
                    element.set_property(name, v);
                    assert_eq!(element.property::<u32>(name), v, "{ctx}: read-back");
                }
            }
            Int(min, max, def) => {
                for v in min..=max {
                    element.set_property(name, v);
                    assert_eq!(element.property::<i32>(name), v, "{ctx}: read-back");
                }
                element.set_property(name, def);
            }
        }
    }
}

fn check_pads(factory: &str) {
    util::init();
    let f = gst::ElementFactory::find(factory).unwrap();
    let expected: gst::Caps =
        "audio/x-raw, format=F32LE, layout=interleaved, rate=[1, 2147483647], channels=[1, 2147483647]"
            .parse()
            .unwrap();
    let templates = f.static_pad_templates();
    assert_eq!(templates.len(), 2, "{factory}");
    for t in templates {
        assert_eq!(t.presence(), gst::PadPresence::Always, "{factory}");
        assert!(
            t.caps() == expected,
            "{factory}: {} caps {}",
            t.name_template(),
            t.caps()
        );
    }
}

#[test]
fn gate_interface() {
    check_element("stromdsp-gate", "StromDspGate", &gate_spec());
    check_pads("stromdsp-gate");
}

#[test]
fn compressor_interface() {
    check_element(
        "stromdsp-compressor",
        "StromDspCompressor",
        &compressor_spec(),
    );
    check_pads("stromdsp-compressor");
}

#[test]
fn equalizer_interface() {
    check_element("stromdsp-equalizer", "StromDspEqualizer", &equalizer_spec());
    check_pads("stromdsp-equalizer");
}

#[test]
fn limiter_interface() {
    check_element("stromdsp-limiter", "StromDspLimiter", &limiter_spec());
    check_pads("stromdsp-limiter");
}

/// `enabled` maps to BaseTransform passthrough for gate/compressor/EQ; the
/// limiter never goes passthrough (it keeps its delay).
#[test]
fn enabled_maps_to_passthrough() {
    util::init();
    for factory in [
        "stromdsp-gate",
        "stromdsp-compressor",
        "stromdsp-equalizer",
        "stromdsp-limiter",
    ] {
        let e = gst::ElementFactory::make(factory).build().unwrap();
        let bt = e.downcast_ref::<gst_base::BaseTransform>().unwrap();
        use gst_base::prelude::BaseTransformExt;
        assert!(!bt.is_passthrough(), "{factory}: passthrough by default");
        e.set_property("enabled", false);
        let expect = factory != "stromdsp-limiter";
        assert_eq!(bt.is_passthrough(), expect, "{factory}: enabled=false");
        e.set_property("enabled", true);
        assert!(!bt.is_passthrough(), "{factory}: enabled=true again");
    }
}
