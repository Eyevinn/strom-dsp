//! Interface test of `stromdsp-broadcastprocessor`: name, GLib type, range,
//! default and flags of every property, enum values and nicks, the additive
//! `crossover-frequencies`, the action signals, metadata and pad templates.
//! This is the element's public contract: it must fail if anything drifts.

mod util;

use gst::glib;
use gst::prelude::*;

const GST_FLAGS: glib::ParamFlags = gst::PARAM_FLAG_CONTROLLABLE
    .union(gst::PARAM_FLAG_MUTABLE_READY)
    .union(gst::PARAM_FLAG_MUTABLE_PAUSED)
    .union(gst::PARAM_FLAG_MUTABLE_PLAYING);

/// PLAYING-mutable and controllable (the normal case).
const LIVE: glib::ParamFlags = gst::PARAM_FLAG_CONTROLLABLE.union(gst::PARAM_FLAG_MUTABLE_PLAYING);
/// READY-only, not controllable.
const READY: glib::ParamFlags = gst::PARAM_FLAG_MUTABLE_READY;

#[derive(Debug, Clone, Copy)]
enum Kind {
    Bool(bool),
    /// (min, max, default)
    Float(f32, f32, f32),
    /// (min, max, default)
    UInt(u32, u32, u32),
    Preset,
    LatencyMode,
    FloatArray,
}

fn make() -> gst::Element {
    util::init();
    gst::ElementFactory::make("stromdsp-broadcastprocessor")
        .build()
        .unwrap()
}

/// The expected interface: (name, kind, GStreamer flags).
fn spec() -> Vec<(String, Kind, glib::ParamFlags)> {
    let mut v: Vec<(String, Kind, glib::ParamFlags)> = vec![
        ("enabled".into(), Kind::Bool(true), LIVE),
        ("preset".into(), Kind::Preset, LIVE),
        ("input-gain".into(), Kind::Float(-24.0, 24.0, 0.0), LIVE),
        (
            "target-loudness".into(),
            Kind::Float(-40.0, -5.0, -23.0),
            LIVE,
        ),
        (
            "true-peak-ceiling".into(),
            Kind::Float(-12.0, 0.0, -1.0),
            LIVE,
        ),
        ("spike-guard-enabled".into(), Kind::Bool(true), LIVE),
        ("spike-threshold".into(), Kind::Float(6.0, 30.0, 8.0), LIVE),
        ("agc-enabled".into(), Kind::Bool(true), LIVE),
        ("agc-max-gain".into(), Kind::Float(0.0, 30.0, 24.0), LIVE),
        ("agc-max-cut".into(), Kind::Float(0.0, 30.0, 24.0), LIVE),
        ("agc-rate-up".into(), Kind::Float(0.1, 20.0, 1.0), LIVE),
        ("agc-rate-down".into(), Kind::Float(0.1, 40.0, 3.0), LIVE),
        ("agc-gate".into(), Kind::Float(-80.0, -20.0, -50.0), LIVE),
        ("multiband-enabled".into(), Kind::Bool(true), LIVE),
        ("num-bands".into(), Kind::UInt(1, 5, 4), READY),
        (
            "multiband-drive".into(),
            Kind::Float(-12.0, 12.0, 0.0),
            LIVE,
        ),
        (
            "crossover-frequencies".into(),
            Kind::FloatArray,
            gst::PARAM_FLAG_MUTABLE_PLAYING,
        ),
        ("clipper-enabled".into(), Kind::Bool(true), LIVE),
        (
            "clipper-bandwidth".into(),
            Kind::Float(8000.0, 22_000.0, 20_000.0),
            LIVE,
        ),
        ("governor-enabled".into(), Kind::Bool(true), LIVE),
        ("clip-budget".into(), Kind::Float(0.0, 6.0, 2.0), LIVE),
        ("limiter-enabled".into(), Kind::Bool(true), LIVE),
        (
            "limiter-release".into(),
            Kind::Float(1.0, 2000.0, 200.0),
            LIVE,
        ),
        ("latency-mode".into(), Kind::LatencyMode, READY),
        (
            "limiter-lookahead".into(),
            Kind::Float(0.5, 20.0, 1.5),
            READY,
        ),
        ("meter-interval".into(), Kind::UInt(0, 10_000, 100), LIVE),
        (
            "expander-hold".into(),
            Kind::Float(0.0, 1000.0, 100.0),
            LIVE,
        ),
        ("bass-clipper-enabled".into(), Kind::Bool(true), LIVE),
        (
            "bass-clip-threshold".into(),
            Kind::Float(-40.0, 0.0, -14.0),
            LIVE,
        ),
    ];
    for (k, d) in [120.0, 800.0, 4000.0, 10_000.0].into_iter().enumerate() {
        v.push((
            format!("crossover{k}-frequency"),
            Kind::Float(20.0, 20_000.0, d),
            LIVE,
        ));
    }
    // Band defaults: the ebu-r128 preset at 4 bands (documented choice).
    let attack = [30.0, 15.0, 8.0, 5.0, 3.0];
    let release = [300.0, 200.0, 150.0, 100.0, 80.0];
    for k in 0..5 {
        for (f, kind) in [
            ("threshold", Kind::Float(-60.0, 0.0, -18.0)),
            ("ratio", Kind::Float(1.0, 20.0, 2.0)),
            ("attack", Kind::Float(0.1, 500.0, attack[k])),
            ("release", Kind::Float(5.0, 5000.0, release[k])),
            ("makeup", Kind::Float(-12.0, 24.0, 0.0)),
            ("expander-threshold", Kind::Float(-80.0, 0.0, -47.0)),
            ("expander-ratio", Kind::Float(1.0, 10.0, 1.5)),
            ("expander-floor", Kind::Float(0.0, 40.0, 4.0)),
        ] {
            v.push((format!("band{k}-{f}"), kind, LIVE));
        }
    }
    v
}

#[test]
fn identity_metadata_and_pads() {
    let e = make();
    assert_eq!(e.type_().name(), "StromDspBroadcastProcessor");
    let f = e.factory().unwrap();
    assert_eq!(f.metadata("klass"), Some("Filter/Effect/Audio"));
    let expected: gst::Caps = "audio/x-raw, format=F32LE, layout=interleaved, \
                               rate=[1, 2147483647], channels=[1, 2147483647]"
        .parse()
        .unwrap();
    let templates = f.static_pad_templates();
    assert_eq!(templates.len(), 2);
    for t in templates {
        assert_eq!(t.presence(), gst::PadPresence::Always);
        assert!(t.caps() == expected, "caps {}", t.caps());
    }
    // Never a BaseTransform passthrough (enabled=false keeps the delay).
    let bt = e.downcast_ref::<gst_base::BaseTransform>().unwrap();
    assert!(!gst_base::prelude::BaseTransformExt::is_passthrough(bt));
}

#[test]
fn property_set_is_exact() {
    let e = make();
    let mut own: Vec<String> = e
        .list_properties()
        .iter()
        .filter(|ps| ps.owner_type() == e.type_())
        .map(|ps| ps.name().to_string())
        .collect();
    own.sort();
    let mut want: Vec<String> = spec().into_iter().map(|(n, _, _)| n).collect();
    want.sort();
    assert_eq!(own, want);
    assert_eq!(want.len(), 26 + 3 + 4 + 40);
}

#[test]
fn every_property_type_range_default_and_flags() {
    for (name, kind, mutable) in spec() {
        // A fresh element per property: writes do not interact (presets,
        // crossover ordering).
        let e = make();
        let ps = e.find_property(&name).unwrap();
        let ctx = name.as_str();
        let flags = ps.flags();
        assert!(flags.contains(glib::ParamFlags::READWRITE), "{ctx}");
        assert!(!flags.contains(glib::ParamFlags::CONSTRUCT_ONLY), "{ctx}");
        assert_eq!(flags & GST_FLAGS, mutable, "{ctx}: GStreamer flags");
        let vt = ps.value_type();
        match kind {
            Kind::Bool(def) => {
                let ps = ps.downcast_ref::<glib::ParamSpecBoolean>().expect(ctx);
                assert_eq!(vt, glib::Type::BOOL, "{ctx}");
                assert_eq!(ps.default_value(), def, "{ctx}");
                assert_eq!(e.property::<bool>(ctx), def, "{ctx}: initial");
                for v in [!def, def] {
                    e.set_property(ctx, v);
                    assert_eq!(e.property::<bool>(ctx), v, "{ctx}: read-back");
                }
            }
            Kind::Float(min, max, def) => {
                let ps = ps.downcast_ref::<glib::ParamSpecFloat>().expect(ctx);
                assert_eq!(vt, glib::Type::F32, "{ctx}");
                assert_eq!(ps.minimum(), min, "{ctx}: minimum");
                assert_eq!(ps.maximum(), max, "{ctx}: maximum");
                assert_eq!(ps.default_value(), def, "{ctx}: default");
                assert_eq!(e.property::<f32>(ctx), def, "{ctx}: initial");
                if !ctx.starts_with("crossover") {
                    for v in [min, max, (min + max) * 0.5, def] {
                        e.set_property(ctx, v);
                        assert_eq!(e.property::<f32>(ctx), v, "{ctx}: read-back");
                    }
                }
            }
            Kind::UInt(min, max, def) => {
                let ps = ps.downcast_ref::<glib::ParamSpecUInt>().expect(ctx);
                assert_eq!(vt, glib::Type::U32, "{ctx}");
                assert_eq!(
                    (ps.minimum(), ps.maximum(), ps.default_value()),
                    (min, max, def),
                    "{ctx}"
                );
                assert_eq!(e.property::<u32>(ctx), def, "{ctx}: initial");
                for v in [min, max, def] {
                    e.set_property(ctx, v);
                    assert_eq!(e.property::<u32>(ctx), v, "{ctx}: read-back");
                }
            }
            Kind::Preset => {
                let ps = ps.downcast_ref::<glib::ParamSpecEnum>().expect(ctx);
                assert_eq!(vt.name(), "StromDspBroadcastPreset");
                let mut got: Vec<(i32, String)> = ps
                    .enum_class()
                    .values()
                    .iter()
                    .map(|v| (v.value(), v.nick().to_string()))
                    .collect();
                got.sort();
                let want: Vec<(i32, String)> = [
                    (0, "ebu-r128"),
                    (1, "atsc-a85"),
                    (2, "streaming"),
                    (3, "speech"),
                    (4, "custom"),
                ]
                .into_iter()
                .map(|(v, n)| (v, n.to_string()))
                .collect();
                assert_eq!(got, want, "preset values and nicks");
                assert_eq!(ps.default_value_as_i32(), 0, "default ebu-r128");
                for nick in ["atsc-a85", "streaming", "speech", "custom", "ebu-r128"] {
                    e.set_property_from_str(ctx, nick);
                    let v = e.property_value(ctx);
                    let (_, ev) = glib::EnumValue::from_value(&v).unwrap();
                    assert_eq!(ev.nick(), nick, "preset read-back");
                }
            }
            Kind::LatencyMode => {
                let ps = ps.downcast_ref::<glib::ParamSpecEnum>().expect(ctx);
                assert_eq!(vt.name(), "StromDspBroadcastLatencyMode");
                let mut got: Vec<(i32, String)> = ps
                    .enum_class()
                    .values()
                    .iter()
                    .map(|v| (v.value(), v.nick().to_string()))
                    .collect();
                got.sort();
                let want: Vec<(i32, String)> = [(0, "standard"), (1, "low")]
                    .into_iter()
                    .map(|(v, n)| (v, n.to_string()))
                    .collect();
                assert_eq!(got, want, "latency-mode values and nicks");
                assert_eq!(ps.default_value_as_i32(), 0, "default standard");
                for nick in ["low", "standard"] {
                    e.set_property_from_str(ctx, nick);
                    let v = e.property_value(ctx);
                    let (_, ev) = glib::EnumValue::from_value(&v).unwrap();
                    assert_eq!(ev.nick(), nick, "latency-mode read-back");
                }
            }
            Kind::FloatArray => {
                let ps = ps.downcast_ref::<gst::ParamSpecArray>().expect(ctx);
                assert_eq!(vt, gst::Array::static_type(), "{ctx}");
                let entry = ps.element_spec().expect("element spec");
                assert_eq!(entry.value_type(), glib::Type::F32, "{ctx}: entries");
                let a = e.property::<gst::Array>(ctx);
                let got: Vec<f32> = a.iter().map(|v| v.get::<f32>().unwrap()).collect();
                assert_eq!(got, [120.0, 800.0, 4000.0, 10_000.0], "{ctx}: initial");
            }
        }
    }
}

/// The individual crossover properties read back exactly within their
/// neighbours; outside they are clamped (separate preset/ordering tests in
/// `broadcastprocessor.rs` cover the rules in detail).
#[test]
fn crossover_read_back_within_neighbours() {
    let e = make();
    for (name, v) in [
        ("crossover0-frequency", 20.0f32),
        ("crossover0-frequency", 119.5),
        ("crossover1-frequency", 1234.5),
        ("crossover2-frequency", 9999.0),
        ("crossover3-frequency", 20_000.0),
    ] {
        e.set_property(name, v);
        assert_eq!(e.property::<f32>(name), v, "{name}");
    }
}

#[test]
fn action_signals() {
    let e = make();
    let class = e.type_();
    let id = glib::subclass::SignalId::lookup("get-preset-values", class).expect("signal");
    let q = id.query();
    assert!(q.flags().contains(glib::SignalFlags::ACTION));
    assert_eq!(q.return_type().type_(), gst::Structure::static_type());
    let params: Vec<glib::Type> = q.param_types().iter().map(|t| t.type_()).collect();
    assert_eq!(params, [glib::Type::STRING]);

    let id =
        glib::subclass::SignalId::lookup("get-preset-values-for-bands", class).expect("signal");
    let q = id.query();
    assert!(q.flags().contains(glib::SignalFlags::ACTION));
    assert_eq!(q.return_type().type_(), gst::Structure::static_type());
    let params: Vec<glib::Type> = q.param_types().iter().map(|t| t.type_()).collect();
    assert_eq!(params, [glib::Type::STRING, glib::Type::U32]);

    let id = glib::subclass::SignalId::lookup("reset", class).expect("signal");
    let q = id.query();
    assert!(q.flags().contains(glib::SignalFlags::ACTION));
    assert_eq!(q.return_type().type_(), glib::Type::UNIT);
    assert!(q.param_types().is_empty());

    // Callable as an application would.
    let s = e.emit_by_name::<Option<gst::Structure>>("get-preset-values", &[&"ebu-r128"]);
    assert_eq!(s.unwrap().name(), "ebu-r128");
    assert!(e
        .emit_by_name::<Option<gst::Structure>>("get-preset-values", &[&"custom"])
        .is_none());
    let s = e
        .emit_by_name::<Option<gst::Structure>>("get-preset-values-for-bands", &[&"speech", &2u32]);
    assert_eq!(s.unwrap().name(), "speech");
    assert!(e
        .emit_by_name::<Option<gst::Structure>>("get-preset-values-for-bands", &[&"custom", &2u32])
        .is_none());
    e.emit_by_name::<()>("reset", &[]);
}
