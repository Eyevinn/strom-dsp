//! Tests of `stromdsp-broadcastprocessor` beyond the interface: presets
//! (against the documented tables and `get-preset-values`), the `custom`
//! read-back, crossover ordering, and harness tests of processing, latency,
//! meter messages, the `reset` signal and `enabled=false`.
//!
//! Processing is always compared against the library's `BroadcastProcessor`
//! run directly on the same input with the same buffer boundaries, so the
//! tests stay meaningful whatever the chain does (including while it is a
//! passthrough stub).

mod util;

use gst::glib;
use gst::prelude::*;
use gst_check::Harness;
use gststromdsp::broadcastprocessor::{MESSAGE_NAME, SILENCE_DB};
use strom_dsp::broadcast::{BroadcastMeters, BroadcastParams, BroadcastProcessor};
use strom_dsp::presets::BroadcastPreset;
use strom_dsp::Processor;
use util::{Action, FORMATS, TOL};

const FACTORY: &str = "stromdsp-broadcastprocessor";

fn make() -> gst::Element {
    util::init();
    gst::ElementFactory::make(FACTORY).build().unwrap()
}

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

/// Properties a preset never sets (and `preset` itself, and the array that
/// only mirrors the individual crossovers).
const NOT_SET_BY_PRESET: [&str; 6] = [
    "enabled",
    "preset",
    "meter-interval",
    "num-bands",
    "limiter-lookahead",
    "crossover-frequencies",
];

/// Every own property name of the element.
fn all_props(e: &gst::Element) -> Vec<String> {
    e.list_properties()
        .iter()
        .filter(|ps| ps.owner_type() == e.type_())
        .map(|ps| ps.name().to_string())
        .collect()
}

/// The documented preset tables (`docs/elements/broadcastprocessor.md` /
/// `strom_dsp::presets`), written out
/// independently of the code: (target, ceiling, agc-gate, band threshold,
/// band ratio, limiter-release).
fn table(nick: &str) -> (f32, f32, f32, f32, f32, f32) {
    match nick {
        "ebu-r128" => (-23.0, -1.0, -50.0, -18.0, 2.0, 200.0),
        "atsc-a85" => (-24.0, -2.0, -50.0, -19.0, 2.0, 200.0),
        "streaming" => (-16.0, -1.0, -50.0, -11.0, 2.0, 200.0),
        "speech" => (-18.0, -1.0, -40.0, -11.0, 1.5, 150.0),
        _ => unreachable!(),
    }
}

/// Documented crossovers per band count.
fn table_crossovers(n: u32) -> [f32; 4] {
    match n {
        2 => [200.0, 800.0, 4000.0, 10_000.0],
        3 => [200.0, 3000.0, 4000.0, 10_000.0],
        _ => [120.0, 800.0, 4000.0, 10_000.0],
    }
}

/// Documented (attack, release) per band, per band count; inactive bands
/// get the 5-band values of their index.
fn table_times(n: u32) -> [(f32, f32); 5] {
    let five = [
        (30.0, 300.0),
        (15.0, 200.0),
        (8.0, 150.0),
        (5.0, 100.0),
        (3.0, 80.0),
    ];
    let active: Vec<(f32, f32)> = match n {
        1 => vec![(30.0, 300.0)],
        2 => vec![(30.0, 300.0), (10.0, 150.0)],
        3 => vec![(30.0, 300.0), (10.0, 150.0), (5.0, 100.0)],
        4 => five[..4].to_vec(),
        _ => five.to_vec(),
    };
    std::array::from_fn(|k| active.get(k).copied().unwrap_or(five[k]))
}

/// The value `docs/elements/broadcastprocessor.md` says `nick` sets for property `name` at `n` bands.
fn documented(nick: &str, n: u32, name: &str) -> glib::Value {
    let (target, ceiling, gate, threshold, ratio, lim_release) = table(nick);
    let f = |v: f32| v.to_value();
    match name {
        "input-gain" | "multiband-drive" => f(0.0),
        "target-loudness" => f(target),
        "true-peak-ceiling" => f(ceiling),
        "agc-enabled"
        | "multiband-enabled"
        | "clipper-enabled"
        | "governor-enabled"
        | "limiter-enabled"
        | "spike-guard-enabled" => true.to_value(),
        // Every preset: spike guard at 8 dB.
        "spike-threshold" => f(8.0),
        "clipper-bandwidth" => f(20_000.0),
        "clip-budget" if nick == "streaming" => f(3.0),
        "clip-budget" => f(2.0),
        // Every preset has ±24 dB of AGC range; ebu-r128 rides slower.
        "agc-max-gain" | "agc-max-cut" => f(24.0),
        "agc-rate-up" if nick == "ebu-r128" => f(1.0),
        "agc-rate-up" => f(3.0),
        "agc-rate-down" if nick == "ebu-r128" => f(3.0),
        "agc-rate-down" => f(6.0),
        "agc-gate" => f(gate),
        "limiter-release" => f(lim_release),
        // Expander and bass clipper: see the second preset table.
        "expander-hold" => f(100.0),
        "bass-clipper-enabled" => true.to_value(),
        "bass-clip-threshold" => f(target + 9.0),
        _ => {
            if let Some(k) = name
                .strip_prefix("crossover")
                .and_then(|r| r.strip_suffix("-frequency"))
            {
                return f(table_crossovers(n)[k.parse::<usize>().unwrap()]);
            }
            let (k, field) = name
                .strip_prefix("band")
                .and_then(|r| r.split_once('-'))
                .unwrap_or_else(|| panic!("undocumented {name}"));
            let k: usize = k.parse().unwrap();
            match field {
                "threshold" => f(threshold),
                "ratio" => f(ratio),
                "attack" => f(table_times(n)[k].0),
                "release" => f(table_times(n)[k].1),
                "makeup" => f(0.0),
                "expander-threshold" if nick == "speech" => f(target - 22.0),
                "expander-threshold" => f(target - 24.0),
                "expander-ratio" if nick == "speech" => f(2.0),
                "expander-ratio" => f(1.5),
                "expander-floor" if nick == "speech" => f(6.0),
                "expander-floor" => f(4.0),
                _ => panic!("undocumented {name}"),
            }
        }
    }
}

fn preset_nick(e: &gst::Element) -> String {
    let v = e.property_value("preset");
    glib::EnumValue::from_value(&v)
        .unwrap()
        .1
        .nick()
        .to_string()
}

fn values_equal(a: &glib::Value, b: &glib::Value) -> bool {
    if let (Ok(x), Ok(y)) = (a.get::<f32>(), b.get::<f32>()) {
        return x.to_bits() == y.to_bits();
    }
    if let (Ok(x), Ok(y)) = (a.get::<bool>(), b.get::<bool>()) {
        return x == y;
    }
    if let (Ok(x), Ok(y)) = (a.get::<u32>(), b.get::<u32>()) {
        return x == y;
    }
    if let (Ok(x), Ok(y)) = (a.get::<gst::Array>(), b.get::<gst::Array>()) {
        return x.len() == y.len() && x.iter().zip(y.iter()).all(|(p, q)| values_equal(p, q));
    }
    if a.type_() == b.type_() && a.type_().is_a(glib::Type::ENUM) {
        let ia = glib::EnumValue::from_value(a).unwrap().1.value();
        let ib = glib::EnumValue::from_value(b).unwrap().1.value();
        return ia == ib;
    }
    panic!("cannot compare {a:?} and {b:?}");
}

/// Snapshot of every own property.
fn snapshot(e: &gst::Element) -> Vec<(String, glib::Value)> {
    all_props(e)
        .into_iter()
        .map(|n| {
            let v = e.property_value(&n);
            (n, v)
        })
        .collect()
}

/// Names whose value differs between two snapshots.
fn changed(a: &[(String, glib::Value)], b: &[(String, glib::Value)]) -> Vec<String> {
    a.iter()
        .zip(b)
        .filter(|((_, x), (_, y))| !values_equal(x, y))
        .map(|((n, _), _)| n.clone())
        .collect()
}

/// For every preset and band count: the element's values after setting the
/// preset equal the documented tables, and `get-preset-values` returns
/// exactly those fields (names, types, values) for the current band count.
#[test]
fn presets_set_exactly_the_documented_values() {
    for nick in ["ebu-r128", "atsc-a85", "streaming", "speech"] {
        for n in 1..=5u32 {
            let e = make();
            e.set_property("num-bands", n);
            e.set_property_from_str("preset", nick);
            assert_eq!(preset_nick(&e), nick);
            let s = e
                .emit_by_name::<Option<gst::Structure>>("get-preset-values", &[&nick])
                .expect("structure");
            assert_eq!(s.name(), nick);

            let mut expected: Vec<String> = all_props(&e)
                .into_iter()
                .filter(|p| !NOT_SET_BY_PRESET.contains(&p.as_str()))
                .collect();
            expected.sort();
            let mut fields: Vec<String> = s.fields().map(|f| f.to_string()).collect();
            fields.sort();
            assert_eq!(fields, expected, "{nick} @ {n}: fields");

            for name in &expected {
                let ctx = format!("{nick} @ {n} bands: {name}");
                let field = s.value(name.as_str()).unwrap();
                let prop = e.property_value(name);
                assert_eq!(field.type_(), prop.type_(), "{ctx}: type");
                assert!(values_equal(field, &prop), "{ctx}: {field:?} vs {prop:?}");
                let doc = documented(nick, n, name);
                assert!(
                    values_equal(&doc, &prop),
                    "{ctx}: docs/elements/broadcastprocessor.md {doc:?} vs {prop:?}"
                );
            }
        }
    }
}

/// The element's defaults are the ebu-r128 preset at 4 bands.
#[test]
fn defaults_are_ebu_r128() {
    let fresh = make();
    let e = make();
    e.set_property_from_str("preset", "ebu-r128");
    assert!(changed(&snapshot(&fresh), &snapshot(&e)).is_empty());
    assert_eq!(preset_nick(&fresh), "ebu-r128");
}

#[test]
fn get_preset_values_null_for_custom_and_unknown() {
    let e = make();
    for nick in ["custom", "loud", "", "EBU-R128"] {
        assert!(
            e.emit_by_name::<Option<gst::Structure>>("get-preset-values", &[&nick])
                .is_none(),
            "{nick}"
        );
    }
    // get-preset-values does not change anything.
    let before = snapshot(&e);
    let _ = e.emit_by_name::<Option<gst::Structure>>("get-preset-values", &[&"speech"]);
    assert!(changed(&before, &snapshot(&e)).is_empty());
}

/// The value of property `name` in the library parameters `p`.
fn library_value(p: &BroadcastParams, name: &str) -> glib::Value {
    match name {
        "input-gain" => p.input_gain_db.to_value(),
        "target-loudness" => p.target_loudness_lufs.to_value(),
        "true-peak-ceiling" => p.true_peak_ceiling_dbtp.to_value(),
        "agc-enabled" => p.agc_enabled.to_value(),
        "spike-guard-enabled" => p.spike_guard_enabled.to_value(),
        "spike-threshold" => p.spike_threshold_db.to_value(),
        "agc-max-gain" => p.agc_max_gain_db.to_value(),
        "agc-max-cut" => p.agc_max_cut_db.to_value(),
        "agc-rate-up" => p.agc_rate_up_db_s.to_value(),
        "agc-rate-down" => p.agc_rate_down_db_s.to_value(),
        "agc-gate" => p.agc_gate_lufs.to_value(),
        "multiband-enabled" => p.multiband_enabled.to_value(),
        "multiband-drive" => p.multiband_drive_db.to_value(),
        "clipper-enabled" => p.clipper_enabled.to_value(),
        "clipper-bandwidth" => p.clipper_bandwidth_hz.to_value(),
        "governor-enabled" => p.governor_enabled.to_value(),
        "clip-budget" => p.clip_budget_db.to_value(),
        "limiter-enabled" => p.limiter_enabled.to_value(),
        "limiter-release" => p.limiter_release_ms.to_value(),
        "expander-hold" => p.expander_hold_ms.to_value(),
        "bass-clipper-enabled" => p.bass_clipper_enabled.to_value(),
        "bass-clip-threshold" => p.bass_clip_threshold_db.to_value(),
        _ => {
            if let Some(k) = name
                .strip_prefix("crossover")
                .and_then(|r| r.strip_suffix("-frequency"))
            {
                return p.crossover_hz[k.parse::<usize>().unwrap()].to_value();
            }
            let (k, field) = name
                .strip_prefix("band")
                .and_then(|r| r.split_once('-'))
                .unwrap_or_else(|| panic!("unknown {name}"));
            let k = k.parse::<usize>().unwrap();
            let b = &p.bands[k];
            let e = &p.band_expanders[k];
            match field {
                "threshold" => b.threshold_db.to_value(),
                "ratio" => b.ratio.to_value(),
                "attack" => b.attack_ms.to_value(),
                "release" => b.release_ms.to_value(),
                "makeup" => b.makeup_db.to_value(),
                "expander-threshold" => e.threshold_db.to_value(),
                "expander-ratio" => e.ratio.to_value(),
                "expander-floor" => e.floor_db.to_value(),
                _ => panic!("unknown {name}"),
            }
        }
    }
}

/// For every preset and band count 1 … 5, `get-preset-values-for-bands`
/// returns the same fields as `get-preset-values`, with the values of
/// `BroadcastParams::preset` (and the documented tables) for that band
/// count, whatever the element's own `num-bands` and settings are, and it
/// changes nothing on the element.
#[test]
fn get_preset_values_for_bands_matches_library() {
    let mut expected: Vec<String> = all_props(&make())
        .into_iter()
        .filter(|p| !NOT_SET_BY_PRESET.contains(&p.as_str()))
        .collect();
    expected.sort();

    for own_bands in [2u32, 4] {
        let e = make();
        e.set_property("num-bands", own_bands);
        e.set_property_from_str("preset", "speech");
        e.set_property("input-gain", 3.0f32);
        let before = snapshot(&e);

        for preset in BroadcastPreset::ALL {
            let nick = preset.nick();
            for n in 1..=5u32 {
                let ctx = format!("{nick} @ {n} bands (element at {own_bands})");
                let s = e
                    .emit_by_name::<Option<gst::Structure>>(
                        "get-preset-values-for-bands",
                        &[&nick, &n],
                    )
                    .unwrap_or_else(|| panic!("{ctx}: structure"));
                assert_eq!(s.name(), nick, "{ctx}");

                let mut fields: Vec<String> = s.fields().map(|f| f.to_string()).collect();
                fields.sort();
                assert_eq!(fields, expected, "{ctx}: fields");

                let lib = BroadcastParams::preset(preset, n as usize);
                for name in &expected {
                    let field = s.value(name.as_str()).unwrap();
                    assert_eq!(
                        field.type_(),
                        e.property_value(name).type_(),
                        "{ctx}: {name} type"
                    );
                    let want = library_value(&lib, name);
                    assert!(
                        values_equal(field, &want),
                        "{ctx}: {name}: {field:?} vs library {want:?}"
                    );
                    let doc = documented(nick, n, name);
                    assert!(
                        values_equal(field, &doc),
                        "{ctx}: {name}: {field:?} vs docs/elements/broadcastprocessor.md {doc:?}"
                    );
                }

                // Same as `get-preset-values` on an element at `n` bands.
                let other = make();
                other.set_property("num-bands", n);
                let t = other
                    .emit_by_name::<Option<gst::Structure>>("get-preset-values", &[&nick])
                    .unwrap();
                assert_eq!(s, t, "{ctx}: vs get-preset-values");
            }
        }

        assert!(
            changed(&before, &snapshot(&e)).is_empty(),
            "element at {own_bands} bands changed"
        );
        assert_eq!(e.property::<u32>("num-bands"), own_bands);
        assert_eq!(preset_nick(&e), "custom");
    }
}

#[test]
fn get_preset_values_for_bands_null_cases() {
    let e = make();
    for nick in ["custom", "loud", "", "EBU-R128"] {
        for n in 1..=5u32 {
            assert!(
                e.emit_by_name::<Option<gst::Structure>>(
                    "get-preset-values-for-bands",
                    &[&nick, &n]
                )
                .is_none(),
                "{nick} @ {n}"
            );
        }
    }
    for n in [0u32, 6, 100, u32::MAX] {
        assert!(
            e.emit_by_name::<Option<gst::Structure>>(
                "get-preset-values-for-bands",
                &[&"ebu-r128", &n]
            )
            .is_none(),
            "{n}"
        );
    }
}

/// Presets never touch `enabled`, `meter-interval`, `num-bands` or
/// `limiter-lookahead`.
#[test]
fn presets_leave_excluded_properties_alone() {
    let e = make();
    e.set_property("enabled", false);
    e.set_property("meter-interval", 250u32);
    e.set_property("num-bands", 3u32);
    e.set_property("limiter-lookahead", 12.0f32);
    for nick in ["speech", "streaming", "atsc-a85", "ebu-r128", "custom"] {
        e.set_property_from_str("preset", nick);
        assert!(!e.property::<bool>("enabled"), "{nick}");
        assert_eq!(e.property::<u32>("meter-interval"), 250, "{nick}");
        assert_eq!(e.property::<u32>("num-bands"), 3, "{nick}");
        assert_eq!(e.property::<f32>("limiter-lookahead"), 12.0, "{nick}");
    }
}

/// Writing any preset-set property changes only that property and switches
/// to `custom`; writing `custom` changes nothing else.
#[test]
fn custom_read_back() {
    let e = make();
    e.set_property_from_str("preset", "streaming");
    let before = snapshot(&e);
    e.set_property("input-gain", 3.0f32);
    assert_eq!(preset_nick(&e), "custom");
    let mut diff = changed(&before, &snapshot(&e));
    diff.sort();
    assert_eq!(diff, ["input-gain", "preset"]);

    // Every preset-set property, from every preset, switches to custom.
    for name in all_props(&e)
        .into_iter()
        .filter(|p| !NOT_SET_BY_PRESET.contains(&p.as_str()))
    {
        let e = make();
        e.set_property_from_str("preset", "speech");
        let before = snapshot(&e);
        let v = e.property_value(&name);
        let new = if let Ok(b) = v.get::<bool>() {
            (!b).to_value()
        } else {
            let x = v.get::<f32>().unwrap();
            let ps = e.find_property(&name).unwrap();
            let ps = ps.downcast_ref::<glib::ParamSpecFloat>().unwrap();
            // A small move that stays inside the range (and between the
            // neighbours for a crossover).
            let y = if x + 0.5 <= ps.maximum() {
                x + 0.5
            } else {
                x - 0.5
            };
            y.to_value()
        };
        e.set_property_from_value(&name, &new);
        assert_eq!(preset_nick(&e), "custom", "{name}");
        let mut diff = changed(&before, &snapshot(&e));
        diff.retain(|d| d != "crossover-frequencies");
        diff.sort();
        let mut want = vec![name.clone(), "preset".to_string()];
        want.sort();
        assert_eq!(diff, want, "{name}");
    }

    // Writing the value it already has keeps the preset.
    let e = make();
    e.set_property_from_str("preset", "speech");
    e.set_property("target-loudness", -18.0f32);
    assert_eq!(preset_nick(&e), "speech");
    // So do the properties a preset does not set.
    e.set_property("enabled", false);
    e.set_property("meter-interval", 0u32);
    assert_eq!(preset_nick(&e), "speech");

    // Writing custom changes nothing else.
    let before = snapshot(&e);
    e.set_property_from_str("preset", "custom");
    assert_eq!(preset_nick(&e), "custom");
    assert_eq!(changed(&before, &snapshot(&e)), ["preset"]);
    // And a preset after custom restores every preset value.
    e.set_property("input-gain", 6.0f32);
    e.set_property_from_str("preset", "speech");
    assert_eq!(e.property::<f32>("input-gain"), 0.0);
}

/// `num-bands` with a preset selected re-applies the preset's band set for
/// the new count; with `custom` it changes only `num-bands`.
#[test]
fn num_bands_follows_the_selected_preset() {
    let e = make();
    e.set_property_from_str("preset", "speech");
    e.set_property("num-bands", 2u32);
    assert_eq!(preset_nick(&e), "speech");
    assert_eq!(e.property::<f32>("crossover0-frequency"), 200.0);
    assert_eq!(e.property::<f32>("band1-attack"), 10.0);

    e.set_property("band0-ratio", 3.0f32);
    assert_eq!(preset_nick(&e), "custom");
    let before = snapshot(&e);
    e.set_property("num-bands", 5u32);
    assert_eq!(changed(&before, &snapshot(&e)), ["num-bands"]);
}

#[test]
fn notify_on_preset() {
    let e = make();
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let s2 = seen.clone();
    e.connect_notify(None, move |_, ps| {
        s2.lock().unwrap().push(ps.name().to_string());
    });
    e.set_property_from_str("preset", "speech");
    let got = std::mem::take(&mut *seen.lock().unwrap());
    for name in ["target-loudness", "band4-ratio", "crossover0-frequency"] {
        assert!(got.iter().any(|n| n == name), "{name}: {got:?}");
    }
    e.set_property("input-gain", 1.0f32);
    let got = std::mem::take(&mut *seen.lock().unwrap());
    assert!(got.iter().any(|n| n == "preset"), "{got:?}");
}

// ---------------------------------------------------------------------------
// Crossover ordering
// ---------------------------------------------------------------------------

fn crossovers(e: &gst::Element) -> [f32; 4] {
    std::array::from_fn(|k| e.property::<f32>(&format!("crossover{k}-frequency")))
}

fn array_crossovers(e: &gst::Element) -> Vec<f32> {
    e.property::<gst::Array>("crossover-frequencies")
        .iter()
        .map(|v| v.get::<f32>().unwrap())
        .collect()
}

#[test]
fn crossover_writes_are_clamped_to_neighbours() {
    let e = make();
    e.set_property("crossover1-frequency", 50.0f32);
    assert_eq!(crossovers(&e), [120.0, 121.0, 4000.0, 10_000.0]);
    e.set_property("crossover1-frequency", 15_000.0f32);
    assert_eq!(crossovers(&e), [120.0, 3999.0, 4000.0, 10_000.0]);
    e.set_property("crossover3-frequency", 20.0f32);
    assert_eq!(crossovers(&e), [120.0, 3999.0, 4000.0, 4001.0]);
    assert_eq!(array_crossovers(&e), crossovers(&e));
}

/// The documented write order: lowering → crossover0 first; raising →
/// crossover3 first. Both reach the target exactly; the other order clamps.
#[test]
fn crossover_write_order_rule() {
    let lower = [30.0f32, 40.0, 50.0, 60.0];
    let raise = [5000.0f32, 10_000.0, 15_000.0, 19_000.0];
    let write = |e: &gst::Element, target: &[f32; 4], order: &[usize]| {
        for &k in order {
            e.set_property(&format!("crossover{k}-frequency"), target[k]);
        }
    };
    let e = make();
    write(&e, &lower, &[0, 1, 2, 3]);
    assert_eq!(crossovers(&e), lower, "lowering, ascending order");
    let e = make();
    write(&e, &raise, &[3, 2, 1, 0]);
    assert_eq!(crossovers(&e), raise, "raising, descending order");
    // The wrong order is clamped (still strictly ascending).
    let e = make();
    write(&e, &lower, &[3, 2, 1, 0]);
    assert_ne!(crossovers(&e), lower);
    let e = make();
    write(&e, &raise, &[0, 1, 2, 3]);
    assert_ne!(crossovers(&e), raise);
}

#[test]
fn crossover_frequencies_array() {
    let e = make();
    // Unsorted, partial: sorted into crossover 0, 1, 2; crossover 3 kept.
    e.set_property_from_str("crossover-frequencies", "<3000.0, 150.0, 900.0>");
    assert_eq!(crossovers(&e), [150.0, 900.0, 3000.0, 10_000.0]);
    assert_eq!(preset_nick(&e), "custom");
    // Duplicates are separated, upward from the lowest …
    e.set_property_from_str("crossover-frequencies", "<500.0, 500.0, 500.0, 500.0>");
    assert_eq!(crossovers(&e), [500.0, 501.0, 502.0, 503.0]);
    // … and downward from the top of the range.
    e.set_property_from_str(
        "crossover-frequencies",
        "<19999.5, 20000.0, 20000.0, 20000.0>",
    );
    assert_eq!(crossovers(&e), [19_997.0, 19_998.0, 19_999.0, 20_000.0]);
    // A partial set above the kept ones pushes them up.
    let e = make();
    e.set_property_from_str("crossover-frequencies", "<5000.0, 12000.0>");
    assert_eq!(crossovers(&e), [5000.0, 12_000.0, 12_001.0, 12_002.0]);
    // As a gst::Array of floats.
    let arr = gst::Array::new([100.0f32, 1000.0, 5000.0, 12_000.0]);
    e.set_property("crossover-frequencies", &arr);
    assert_eq!(array_crossovers(&e), [100.0, 1000.0, 5000.0, 12_000.0]);
    // An empty array is ignored. (Out-of-range, NaN and non-float
    // entries are rejected by GObject against the element spec already.)
    let before = crossovers(&e);
    e.set_property("crossover-frequencies", gst::Array::new(Vec::<f32>::new()));
    assert_eq!(crossovers(&e), before);
    assert_eq!(array_crossovers(&e), before);
}

// ---------------------------------------------------------------------------
// Harness tests
// ---------------------------------------------------------------------------

/// A non-default configuration (streaming preset at 3 bands plus edits).
fn params() -> BroadcastParams {
    let mut p = BroadcastParams::preset(BroadcastPreset::Streaming, 3);
    p.input_gain_db = 2.0;
    p.bands[1].ratio = 3.0;
    p.limiter_lookahead_ms = 3.0;
    p
}

fn props(p: &BroadcastParams) -> Vec<(&'static str, glib::Value)> {
    vec![
        ("num-bands", (p.num_bands as u32).to_value()),
        ("limiter-lookahead", p.limiter_lookahead_ms.to_value()),
        ("preset", preset_value("streaming")),
        ("input-gain", p.input_gain_db.to_value()),
        ("band1-ratio", p.bands[1].ratio.to_value()),
        ("enabled", p.enabled.to_value()),
    ]
}

fn on_set(dsp: &mut BroadcastProcessor, name: &str, value: &glib::Value) {
    let mut q = *dsp.params();
    match name {
        "enabled" => q.enabled = value.get().unwrap(),
        "target-loudness" => q.target_loudness_lufs = value.get().unwrap(),
        "multiband-drive" => q.multiband_drive_db = value.get().unwrap(),
        "crossover0-frequency" => q.crossover_hz[0] = value.get().unwrap(),
        "limiter-enabled" => q.limiter_enabled = value.get().unwrap(),
        _ => unreachable!("{name}"),
    }
    dsp.set_params(&q);
}

/// The `preset` enum value with this nick.
fn preset_value(nick: &str) -> glib::Value {
    let e = make();
    e.set_property_from_str("preset", nick);
    e.property_value("preset")
}

fn set_props(e: &gst::Element, props: &[(&str, glib::Value)]) {
    for (name, v) in props {
        e.set_property_from_value(name, v);
    }
}

fn run(
    props: &[(&str, glib::Value)],
    rate: u32,
    channels: usize,
    input: &[f32],
    fpb: usize,
    actions: &[(usize, Action)],
) -> Vec<f32> {
    util::run_element(FACTORY, props, rate, channels, input, fpb, actions)
}

#[test]
fn processing_matches_library() {
    for (rate, channels) in FORMATS {
        let fpb = if channels == 2 { 1024 } else { 1500 };
        let input = util::test_signal(rate, channels, rate as usize);
        let p = params();
        let actions = [
            (
                2,
                Action::SetProperty("target-loudness", (-20.0f32).to_value()),
            ),
            (3, Action::SetProperty("multiband-drive", 3.0f32.to_value())),
            (5, Action::Discont),
            (
                6,
                Action::SetProperty("crossover0-frequency", 300.0f32.to_value()),
            ),
            (7, Action::SetProperty("limiter-enabled", false.to_value())),
            (8, Action::SetProperty("enabled", false.to_value())),
            (10, Action::Flush),
            (12, Action::SetProperty("enabled", true.to_value())),
        ];
        let out = run(&props(&p), rate, channels, &input, fpb, &actions);
        let mut dsp = BroadcastProcessor::new();
        dsp.set_params(&p);
        let reference = util::run_reference(dsp, rate, channels, &input, fpb, &actions, on_set);
        util::assert_close(&out, &reference, TOL);
    }
}

/// The structural properties take effect at the next caps negotiation: a
/// write while playing changes neither the processing nor the latency.
#[test]
fn ready_only_properties_wait_for_caps() {
    let (rate, channels) = (48_000, 2);
    let input = util::test_signal(rate, channels, rate as usize / 2);
    let p = params();
    let actions = [
        (2, Action::SetProperty("num-bands", 5u32.to_value())),
        (
            2,
            Action::SetProperty("limiter-lookahead", 15.0f32.to_value()),
        ),
    ];
    let out = run(&props(&p), rate, channels, &input, 1024, &actions);
    let mut dsp = BroadcastProcessor::new();
    dsp.set_params(&p);
    // The reference never sees the structural change.
    let reference = util::run_reference(dsp, rate, channels, &input, 1024, &actions, |_, _, _| {});
    util::assert_close(&out, &reference, TOL);
}

fn expected_latency(rate: u32, lookahead_ms: f32) -> gst::ClockTime {
    let frames = BroadcastProcessor::latency_frames_for(rate as f32, lookahead_ms) as u64;
    gst::ClockTime::from_nseconds((frames * 1_000_000_000).div_ceil(u64::from(rate)))
}

fn query_latency(h: &Harness) -> (bool, gst::ClockTime, Option<gst::ClockTime>) {
    let src = h.element().unwrap().static_pad("src").unwrap();
    let mut q = gst::query::Latency::new();
    assert!(src.query(&mut q), "latency query failed");
    q.result()
}

fn push_one(h: &mut Harness, channels: usize) {
    let buf = gst::Buffer::from_mut_slice(vec![0u8; 64 * channels * 4]);
    assert_eq!(h.push(buf), Ok(gst::FlowSuccess::Ok));
    let _ = h.pull().unwrap();
}

#[test]
fn latency_query_adds_chain_latency() {
    util::init();
    for (rate, lookahead, bands) in [
        (48_000u32, 5.0f32, 4u32),
        (44_100, 5.0, 1),
        (22_050, 3.3, 5),
        (96_000, 0.5, 2),
        (48_000, 20.0, 3),
    ] {
        let mut h = Harness::new(FACTORY);
        let upstream = gst::ClockTime::from_mseconds(10);
        h.set_upstream_latency(upstream);
        let e = h.element().unwrap();
        e.set_property("limiter-lookahead", lookahead);
        e.set_property("num-bands", bands);
        h.set_src_caps_str(&util::caps_str(rate, 2));
        h.play();
        push_one(&mut h, 2);
        let ours = expected_latency(rate, lookahead);
        let ctx = format!("rate {rate}, lookahead {lookahead}, {bands} bands");
        let (live, min, max) = query_latency(&h);
        assert!(live, "{ctx}");
        assert_eq!(min, upstream + ours, "{ctx}");
        let mut upstream_q = gst::query::Latency::new();
        assert!(h.srcpad().unwrap().query(&mut upstream_q));
        let (_, _, upstream_max) = upstream_q.result();
        assert_eq!(max, upstream_max.map(|m| m + ours), "{ctx}");
        // Unchanged by enabled and stage bypass.
        for (name, v) in [
            ("enabled", false),
            ("spike-guard-enabled", false),
            ("agc-enabled", false),
            ("multiband-enabled", false),
            ("limiter-enabled", false),
        ] {
            e.set_property(name, v);
            push_one(&mut h, 2);
            assert_eq!(query_latency(&h).1, upstream + ours, "{ctx}: {name}");
        }
    }
}

#[test]
fn latency_message_on_renegotiation() {
    util::init();
    let mut h = Harness::new(FACTORY);
    let e = h.element().unwrap();
    let bus = gst::Bus::new();
    e.set_bus(Some(&bus));
    h.set_upstream_latency(gst::ClockTime::ZERO);
    h.set_src_caps_str(&util::caps_str(48_000, 2));
    h.play();
    push_one(&mut h, 2);
    while bus.pop().is_some() {}
    let initial = expected_latency(48_000, 1.5);
    assert_eq!(query_latency(&h).1, initial);

    // READY-only lookahead written while playing: no change yet, no message.
    e.set_property("limiter-lookahead", 12.0f32);
    for (name, v) in [
        ("input-gain", 3.0f32.to_value()),
        ("enabled", false.to_value()),
    ] {
        e.set_property_from_value(name, &v);
    }
    e.set_property_from_str("preset", "speech");
    push_one(&mut h, 2);
    assert!(bus.pop_filtered(&[gst::MessageType::Latency]).is_none());
    assert_eq!(query_latency(&h).1, initial);

    // Renegotiation applies it and posts a Latency message.
    h.set_src_caps_str(&util::caps_str(44_100, 2));
    push_one(&mut h, 2);
    let msg = bus
        .pop_filtered(&[gst::MessageType::Latency])
        .expect("no latency message after renegotiation");
    assert_eq!(msg.src(), Some(e.upcast_ref::<gst::Object>()));
    assert_eq!(query_latency(&h).1, expected_latency(44_100, 12.0));

    // Same configuration again: no message.
    h.set_src_caps_str(&util::caps_str(44_100, 1));
    push_one(&mut h, 1);
    assert!(bus.pop_filtered(&[gst::MessageType::Latency]).is_none());
    e.set_bus(None);
}

/// Does the library chain process at all (the stub is an exact
/// passthrough)? Used only to decide whether an exact-delay check can apply.
fn chain_is_stub() -> bool {
    let mut dsp = BroadcastProcessor::new();
    dsp.set_params(&BroadcastParams::preset(BroadcastPreset::EbuR128, 4));
    dsp.configure(48_000.0, 1);
    let input: Vec<f32> = (0..4800)
        .map(|n| ((n * 7919) % 101) as f32 * 0.002)
        .collect();
    let mut out = input.clone();
    dsp.process(&mut out);
    out == input
}

/// `enabled=false`: the output equals the library's, and (once the real
/// chain is merged) it is the input delayed by exactly the reported latency.
#[test]
fn disabled_is_input_delayed_by_reported_latency() {
    for (rate, channels) in FORMATS {
        let input = util::test_signal(rate, channels, rate as usize / 2);
        let mut p = params();
        p.enabled = false;
        let out = run(&props(&p), rate, channels, &input, 512, &[]);
        let mut dsp = BroadcastProcessor::new();
        dsp.set_params(&p);
        let reference = util::run_reference(dsp, rate, channels, &input, 512, &[], on_set);
        util::assert_close(&out, &reference, TOL);

        if chain_is_stub() {
            println!("BroadcastProcessor is the passthrough stub: exact-delay check skipped");
            continue;
        }
        let delay = BroadcastProcessor::latency_frames_for(rate as f32, p.limiter_lookahead_ms);
        let d = delay * channels;
        assert!(out[..d].iter().all(|&v| v == 0.0), "{rate}: leading zeros");
        util::assert_bit_exact(&out[d..], &input[..input.len() - d]);
    }
}

// ---------------------------------------------------------------------------
// Meter messages
// ---------------------------------------------------------------------------

/// One received meter message and the index of the buffer it followed.
struct Received {
    after_buffer: usize,
    end: gst::ClockTime,
    s: gst::Structure,
}

/// Push `input` in buffers of `fpb` frames from PTS 0; `before(idx, e)`
/// runs before buffer `idx`. Returns the meter messages.
fn run_meter(
    props: &[(&str, glib::Value)],
    rate: u32,
    channels: usize,
    input: &[f32],
    fpb: usize,
    mut before: impl FnMut(usize, &gst::Element),
) -> Vec<Received> {
    util::init();
    let mut h = Harness::new(FACTORY);
    let e = h.element().unwrap();
    set_props(&e, props);
    let bus = gst::Bus::new();
    e.set_bus(Some(&bus));
    h.set_src_caps_str(&util::caps_str(rate, channels));
    h.play();
    let mut msgs = Vec::new();
    let mut pts = gst::ClockTime::ZERO;
    for (idx, chunk) in input.chunks(fpb * channels).enumerate() {
        before(idx, &e);
        let frames = (chunk.len() / channels) as u64;
        let duration = gst::ClockTime::SECOND
            .mul_div_floor(frames, u64::from(rate))
            .unwrap();
        let bytes: Vec<u8> = chunk.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut buf = gst::Buffer::from_mut_slice(bytes);
        {
            let b = buf.get_mut().unwrap();
            b.set_pts(pts);
            b.set_duration(duration);
        }
        assert_eq!(h.push(buf), Ok(gst::FlowSuccess::Ok));
        let _ = h.pull().expect("no output buffer");
        while let Some(msg) = bus.pop() {
            if let gst::MessageView::Element(m) = msg.view() {
                let s = m.structure().unwrap().to_owned();
                if s.name() == MESSAGE_NAME {
                    msgs.push(Received {
                        after_buffer: idx,
                        end: pts + duration,
                        s,
                    });
                }
            }
        }
        pts += duration;
    }
    e.set_bus(None);
    msgs
}

/// The library run directly with the element's documented scheduling:
/// meters taken after the buffer that completes each interval. `reset_at`
/// buffers get `reset_integrated` first. Returns (buffer index, meters).
fn reference_meters(
    p: &BroadcastParams,
    rate: u32,
    channels: usize,
    input: &[f32],
    fpb: usize,
    interval_ms: u32,
    reset_at: &[usize],
) -> Vec<(usize, BroadcastMeters)> {
    let mut dsp = BroadcastProcessor::new();
    dsp.set_params(p);
    dsp.configure(rate as f32, channels);
    let interval = (f64::from(interval_ms) * 0.001 * f64::from(rate)).round() as u64;
    let mut phase = 0u64;
    let mut out = Vec::new();
    let mut buf = input.to_vec();
    for (idx, chunk) in buf.chunks_mut(fpb * channels).enumerate() {
        if reset_at.contains(&idx) {
            dsp.reset_integrated();
        }
        dsp.process(chunk);
        phase += (chunk.len() / channels) as u64;
        if phase >= interval {
            phase -= interval;
            if phase >= interval {
                phase = 0;
            }
            out.push((idx, dsp.take_meters()));
        }
    }
    out
}

const FIELDS: [&str; 18] = [
    "running-time",
    "stream-time",
    "input-momentary",
    "input-short-term",
    "output-momentary",
    "output-short-term",
    "output-integrated",
    "input-true-peak",
    "output-true-peak",
    "spike-events",
    "agc-gain",
    "band-gain-reduction",
    "limiter-gain-reduction",
    "band-expander-gain-reduction",
    "bass-clip-gr",
    "clip-depth",
    "loudness-deficit",
    "governor-state",
];

fn clean(x: f64) -> f64 {
    if x.is_finite() {
        x.max(SILENCE_DB)
    } else {
        SILENCE_DB
    }
}

fn get_f64(s: &gst::Structure, name: &str) -> f64 {
    let v = s
        .get::<f64>(name)
        .unwrap_or_else(|e| panic!("field {name}: {e}"));
    assert!(v.is_finite() && v >= SILENCE_DB, "{name} = {v}");
    v
}

fn check_message(r: &Received, m: &BroadcastMeters, num_bands: usize) {
    let s = &r.s;
    let mut names: Vec<&str> = s.fields().map(|f| f.as_str()).collect();
    names.sort();
    let mut want: Vec<&str> = FIELDS.to_vec();
    want.sort();
    assert_eq!(names, want, "field set");
    // Times: end of the buffer; the harness segment starts at 0.
    assert_eq!(s.get::<u64>("running-time").unwrap(), r.end.nseconds());
    assert_eq!(s.get::<u64>("stream-time").unwrap(), r.end.nseconds());
    let ctx = format!("message after buffer {}", r.after_buffer);
    for (name, v) in [
        ("input-momentary", m.input_momentary_lufs),
        ("input-short-term", m.input_short_term_lufs),
        ("output-momentary", m.output_momentary_lufs),
        ("output-short-term", m.output_short_term_lufs),
        ("output-integrated", m.output_integrated_lufs),
        ("input-true-peak", m.input_true_peak_dbtp),
        ("output-true-peak", m.output_true_peak_dbtp),
    ] {
        assert!(
            (get_f64(s, name) - clean(v)).abs() < 1e-9,
            "{ctx}: {name} {} vs library {v}",
            get_f64(s, name)
        );
    }
    assert_eq!(
        s.get::<u64>("spike-events").unwrap(),
        m.spike_events,
        "{ctx}: spike-events"
    );
    let agc = s.get::<f64>("agc-gain").unwrap();
    assert!((agc - m.agc_gain_db).abs() < 1e-9, "{ctx}: agc-gain");
    assert!((get_f64(s, "clip-depth") - clean(m.clip_depth_db)).abs() < 1e-9);
    let deficit = s.get::<f64>("loudness-deficit").unwrap();
    assert!(deficit >= 0.0 && (deficit - m.loudness_deficit_lu.max(0.0)).abs() < 1e-9);
    assert_eq!(
        s.get::<&str>("governor-state").unwrap(),
        m.governor_state.nick(),
        "{ctx}: governor-state"
    );
    let lim = s.get::<f64>("limiter-gain-reduction").unwrap();
    assert!(lim >= 0.0 && (lim - m.limiter_gain_reduction_db.max(0.0)).abs() < 1e-9);
    let bands = s.get::<gst::Array>("band-gain-reduction").unwrap();
    assert_eq!(bands.len(), num_bands, "{ctx}: one value per active band");
    for (k, v) in bands.iter().enumerate() {
        let v = v.get::<f64>().expect("gdouble entries");
        assert!(v >= 0.0 && v.is_finite(), "{ctx}: band {k} = {v}");
        assert!(
            (v - m.band_gain_reduction_db[k].max(0.0)).abs() < 1e-9,
            "{ctx}: band {k}"
        );
    }
    let exp = s.get::<gst::Array>("band-expander-gain-reduction").unwrap();
    assert_eq!(
        exp.len(),
        num_bands,
        "{ctx}: one expander value per active band"
    );
    for (k, v) in exp.iter().enumerate() {
        let v = v.get::<f64>().expect("gdouble entries");
        assert!(v >= 0.0 && v.is_finite(), "{ctx}: expander {k} = {v}");
        assert!(
            (v - m.band_expander_gain_reduction_db[k].max(0.0)).abs() < 1e-9,
            "{ctx}: expander {k}"
        );
    }
    let bass = s.get::<f64>("bass-clip-gr").unwrap();
    assert!(
        bass >= 0.0 && (bass - m.bass_clip_gain_reduction_db.max(0.0)).abs() < 1e-9,
        "{ctx}: bass-clip-gr {bass}"
    );
}

#[test]
fn meter_messages_match_library_and_rate() {
    for (rate, channels) in FORMATS {
        let fpb = (rate / 100) as usize; // 10 ms buffers
        let input = util::test_signal(rate, channels, rate as usize * 2);
        let p = params();
        let mut pr = props(&p);
        pr.push(("meter-interval", 100u32.to_value()));
        let msgs = run_meter(&pr, rate, channels, &input, fpb, |_, _| {});
        let reference = reference_meters(&p, rate, channels, &input, fpb, 100, &[]);
        // Rate: one message per 100 ms of stream time (2 s → 20), after the
        // buffer that completes each 100 ms (every 10th at 48 kHz).
        assert_eq!(msgs.len(), 20, "{rate}");
        assert_eq!(msgs.len(), reference.len());
        let buffer = gst::ClockTime::SECOND
            .mul_div_floor(fpb as u64, u64::from(rate))
            .unwrap();
        for (i, (r, (idx, m))) in msgs.iter().zip(&reference).enumerate() {
            assert_eq!(r.after_buffer, *idx);
            let due = gst::ClockTime::from_mseconds(100 * (i as u64 + 1));
            assert!(
                r.end + gst::ClockTime::from_useconds(1) >= due && r.end < due + buffer,
                "{rate}: {i}"
            );
            if rate == 48_000 {
                assert_eq!(r.after_buffer, i * 10 + 9);
            }
            check_message(r, m, p.num_bands);
        }
    }
}

#[test]
fn meter_interval_rate_limiting() {
    let (rate, channels) = (48_000u32, 1usize);
    let input = util::test_signal(rate, channels, rate as usize);
    let count = |interval: u32, fpb: usize| {
        let pr = [("meter-interval", interval.to_value())];
        run_meter(&pr, rate, channels, &input, fpb, |_, _| {})
    };
    // Off.
    assert!(count(0, 480).is_empty());
    // Shorter than a buffer: at most one per buffer.
    let m = count(1, 480);
    assert_eq!(m.len(), 100);
    for (i, r) in m.iter().enumerate() {
        assert_eq!(r.after_buffer, i);
    }
    // 35 ms with 10 ms buffers: after the buffers that complete 35, 70,
    // 105, … ms (buffers 3, 6, 10, 13, …), 28 in one second.
    let m = count(35, 480);
    let idx: Vec<usize> = m.iter().map(|r| r.after_buffer).collect();
    assert_eq!(&idx[..6], &[3, 6, 10, 13, 17, 20]);
    assert_eq!(m.len(), 28);
    // Changing the interval while playing takes effect.
    let input2 = util::test_signal(rate, channels, rate as usize);
    let m = run_meter(
        &[("meter-interval", 100u32.to_value())],
        rate,
        channels,
        &input2,
        480,
        |idx, e| {
            if idx == 50 {
                e.set_property("meter-interval", 0u32);
            }
        },
    );
    assert_eq!(m.len(), 5);
    // Defaults: 100 ms, 4 bands.
    let m = run_meter(&[], rate, channels, &input, 480, |_, _| {});
    assert_eq!(m.len(), 10);
    assert_eq!(
        m[0].s
            .get::<gst::Array>("band-gain-reduction")
            .unwrap()
            .len(),
        4
    );
}

/// Clicks every 50 ms over a tone: the messages count the spike guard's
/// events, the same as the library, about one per click.
#[test]
fn meter_counts_spike_events() {
    let (rate, channels) = (48_000u32, 2usize);
    let frames = rate as usize * 2;
    let mut input = vec![0.0f32; frames * channels];
    for n in 0..frames {
        let v = (0.05 * (2.0 * std::f64::consts::PI * 440.0 * n as f64 / 48_000.0).sin()) as f32;
        let click = if n % 2400 == 1200 { 0.8 } else { 0.0 };
        for c in 0..channels {
            input[n * channels + c] = v + click;
        }
    }
    let msgs = run_meter(&[], rate, channels, &input, 480, |_, _| {});
    let reference = reference_meters(
        &BroadcastParams::preset(BroadcastPreset::EbuR128, 4),
        rate,
        channels,
        &input,
        480,
        100,
        &[],
    );
    assert_eq!(msgs.len(), reference.len());
    let mut total = 0;
    for (r, (_, lib)) in msgs.iter().zip(&reference) {
        check_message(r, lib, 4);
        total += r.s.get::<u64>("spike-events").unwrap();
    }
    // 40 clicks; the last ones are still inside the latency at the end.
    assert!((36..=40).contains(&total), "{total} spike events");
}

/// Silence reads −200, never NaN or −∞.
#[test]
fn meter_silence() {
    let (rate, channels) = (48_000u32, 2usize);
    let input = vec![0.0f32; rate as usize * channels];
    let m = run_meter(&[], rate, channels, &input, 480, |_, _| {});
    assert_eq!(m.len(), 10);
    let reference = reference_meters(
        &BroadcastParams::preset(BroadcastPreset::EbuR128, 4),
        rate,
        channels,
        &input,
        480,
        100,
        &[],
    );
    for (r, (_, lib)) in m.iter().zip(&reference) {
        check_message(r, lib, 4);
        for name in ["input-momentary", "output-integrated", "output-true-peak"] {
            assert_eq!(get_f64(&r.s, name), SILENCE_DB, "{name}");
        }
    }
}

/// `reset` restarts the output integrated loudness: the element matches
/// the library with `reset_integrated` at the same buffer.
#[test]
fn reset_signal_restarts_integrated() {
    let (rate, channels) = (48_000u32, 2usize);
    let input = util::test_signal(rate, channels, rate as usize * 2);
    let p = BroadcastParams::preset(BroadcastPreset::EbuR128, 4);
    let msgs = run_meter(&[], rate, channels, &input, 480, |idx, e| {
        if idx == 120 {
            e.emit_by_name::<()>("reset", &[]);
        }
    });
    let reference = reference_meters(&p, rate, channels, &input, 480, 100, &[120]);
    assert_eq!(msgs.len(), reference.len());
    for (r, (idx, m)) in msgs.iter().zip(&reference) {
        assert_eq!(r.after_buffer, *idx);
        check_message(r, m, 4);
    }
}
