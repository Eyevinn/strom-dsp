//! Tests of `stromdsp-declipper`: interface, processing against the
//! library, bypass, latency per setting and its budget, transparency on
//! clean programme, READY-only latency setting and the meter message.
//! The knob test is in `knobs.rs`; real-time safety of the processing path
//! is checked in the library (`tests/declipper.rs` counting allocator,
//! `tests/rtsan_declipper.rs`).

use crate::util;

use crate::util::{Action, FORMATS, TOL};
use gst::glib;
use gst::prelude::*;
use gst_base::prelude::BaseTransformExt;
use gst_check::Harness;
use gststromdsp::declipper::{StromDspDeclipperLatency, MESSAGE_NAME};
use strom_dsp::declipper::{Declipper, DeclipperLatency, DeclipperParams};
use strom_dsp::Processor;

const FACTORY: &str = "stromdsp-declipper";

/// (nick, library setting, budget in ms on the element page).
const MODES: [(&str, DeclipperLatency, f64); 3] = [
    ("low", DeclipperLatency::Low, 3.0),
    ("standard", DeclipperLatency::Normal, 5.0),
    ("high", DeclipperLatency::High, 10.0),
];

fn make() -> gst::Element {
    util::init();
    gst::ElementFactory::make(FACTORY).build().unwrap()
}

fn mode_value(nick: &str) -> glib::Value {
    let t = StromDspDeclipperLatency::static_type();
    let class = glib::EnumClass::with_type(t).unwrap();
    class.to_value_by_nick(nick).expect("nick")
}

fn mode_nick(e: &gst::Element) -> String {
    let v = e.property_value("latency-mode");
    glib::EnumValue::from_value(&v)
        .unwrap()
        .1
        .nick()
        .to_string()
}

/// The util test signal (peaks up to 1.4), hard-clipped at `c`.
fn clipped(rate: u32, channels: usize, frames: usize, c: f32) -> Vec<f32> {
    util::test_signal(rate, channels, frames)
        .into_iter()
        .map(|v| v.clamp(-c, c))
        .collect()
}

// ---------------------------------------------------------------------------
// Interface
// ---------------------------------------------------------------------------

const GST_FLAGS: glib::ParamFlags = gst::PARAM_FLAG_CONTROLLABLE
    .union(gst::PARAM_FLAG_MUTABLE_READY)
    .union(gst::PARAM_FLAG_MUTABLE_PAUSED)
    .union(gst::PARAM_FLAG_MUTABLE_PLAYING);
const PLAYING: glib::ParamFlags =
    gst::PARAM_FLAG_CONTROLLABLE.union(gst::PARAM_FLAG_MUTABLE_PLAYING);
const READY: glib::ParamFlags = gst::PARAM_FLAG_MUTABLE_READY;

fn flags(e: &gst::Element, name: &str, want: glib::ParamFlags) -> glib::ParamSpec {
    let ps = e.find_property(name).expect(name);
    let f = ps.flags();
    assert!(f.contains(glib::ParamFlags::READWRITE), "{name}");
    assert!(!f.contains(glib::ParamFlags::CONSTRUCT_ONLY), "{name}");
    assert_eq!(f & GST_FLAGS, want, "{name}: GStreamer flags");
    ps
}

#[test]
fn interface() {
    let e = make();
    assert_eq!(e.type_().name(), "StromDspDeclipper");
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
            "enabled",
            "latency-mode",
            "max-overshoot",
            "meter-interval",
            "min-level"
        ]
    );

    let ps = flags(&e, "enabled", PLAYING);
    let b = ps.downcast_ref::<glib::ParamSpecBoolean>().unwrap();
    assert!(b.default_value());
    for v in [false, true] {
        e.set_property("enabled", v);
        assert_eq!(e.property::<bool>("enabled"), v);
    }

    for (name, min, max, def) in [
        ("max-overshoot", 0.0f32, 24.0f32, 6.0f32),
        ("min-level", -60.0, 0.0, -20.0),
    ] {
        let ps = flags(&e, name, PLAYING);
        let fl = ps.downcast_ref::<glib::ParamSpecFloat>().expect(name);
        assert_eq!(
            (fl.minimum(), fl.maximum(), fl.default_value()),
            (min, max, def),
            "{name}"
        );
        assert_eq!(e.property::<f32>(name), def, "{name}: initial");
        for v in [min, max, (min + max) / 2.0, def] {
            e.set_property(name, v);
            assert_eq!(e.property::<f32>(name).to_bits(), v.to_bits(), "{name}");
        }
    }

    let ps = flags(&e, "meter-interval", PLAYING);
    let u = ps.downcast_ref::<glib::ParamSpecUInt>().unwrap();
    assert_eq!(
        (u.minimum(), u.maximum(), u.default_value()),
        (0, 10_000, 1000)
    );
    for v in [0u32, 10_000, 100, 1000] {
        e.set_property("meter-interval", v);
        assert_eq!(e.property::<u32>("meter-interval"), v);
    }

    let ps = flags(&e, "latency-mode", READY);
    let en = ps.downcast_ref::<glib::ParamSpecEnum>().unwrap();
    assert_eq!(ps.value_type().name(), "StromDspDeclipperLatency");
    let got: Vec<(i32, String)> = en
        .enum_class()
        .values()
        .iter()
        .map(|v| (v.value(), v.nick().to_string()))
        .collect();
    assert_eq!(
        got,
        [
            (0, "low".to_string()),
            (1, "standard".to_string()),
            (2, "high".to_string())
        ]
    );
    assert_eq!(en.default_value_as_i32(), 1);
    assert_eq!(mode_nick(&e), "standard");
    for nick in ["low", "high", "standard"] {
        e.set_property_from_str("latency-mode", nick);
        assert_eq!(mode_nick(&e), nick);
    }

    // Shared caps on both pads; never a BaseTransform passthrough.
    let expected: gst::Caps = "audio/x-raw, format=F32LE, layout=interleaved, \
                               rate=[1, 2147483647], channels=[1, 2147483647]"
        .parse()
        .unwrap();
    for t in f.static_pad_templates() {
        assert_eq!(t.presence(), gst::PadPresence::Always);
        assert!(t.caps() == expected, "caps {}", t.caps());
    }
    let bt = e.downcast_ref::<gst_base::BaseTransform>().unwrap();
    e.set_property("enabled", false);
    assert!(!bt.is_passthrough());
}

// ---------------------------------------------------------------------------
// Processing
// ---------------------------------------------------------------------------

fn on_set(dsp: &mut Declipper, name: &str, value: &glib::Value) {
    let mut p = dsp.params();
    match name {
        "enabled" => p.enabled = value.get().unwrap(),
        "max-overshoot" => p.max_overshoot_db = value.get().unwrap(),
        "min-level" => p.min_level_db = value.get().unwrap(),
        "meter-interval" => {}
        _ => unreachable!("{name}"),
    }
    dsp.set_params(&p);
}

/// The element gives the library's output, sample for sample, with
/// property changes, DISCONT and flush on the way, at every format.
#[test]
fn processing_matches_library() {
    for (rate, channels) in FORMATS {
        let input = clipped(rate, channels, rate as usize / 2, 0.9);
        let actions = [
            (5, Action::SetProperty("max-overshoot", 3.0f32.to_value())),
            (9, Action::SetProperty("min-level", (-6.0f32).to_value())),
            (13, Action::SetProperty("enabled", false.to_value())),
            (16, Action::Discont),
            (18, Action::SetProperty("enabled", true.to_value())),
            (19, Action::SetProperty("min-level", (-20.0f32).to_value())),
            (20, Action::Flush),
        ];
        let out = util::run_element(FACTORY, &[], rate, channels, &input, 512, &actions);
        let reference = util::run_reference(
            Declipper::new(),
            rate,
            channels,
            &input,
            512,
            &actions,
            on_set,
        );
        util::assert_close(&out, &reference, TOL);
        assert!(
            out.iter().zip(&input).any(|(a, b)| a != b),
            "nothing changed at {rate} Hz"
        );
    }
}

/// Latency (frames) of `mode` at `rate`.
fn frames(rate: u32, mode: DeclipperLatency) -> usize {
    Declipper::latency_frames_for(rate as f32, mode)
}

fn frames_time(rate: u32, frames: usize) -> gst::ClockTime {
    gst::ClockTime::from_nseconds((frames as u64 * 1_000_000_000).div_ceil(u64::from(rate)))
}

/// Bypass: `enabled=false` on clipped audio crossfades from the repaired
/// to the dry signal (the wet weight never rises and stays within 0 … 1,
/// so no step), and once the fade is over the output is the input delayed
/// by exactly the latency, sample for sample.
#[test]
fn bypass_fades_to_delayed_input() {
    let (rate, ch) = (48_000u32, 2usize);
    let input = clipped(rate, ch, rate as usize / 2, 0.9);
    let lat = frames(rate, DeclipperLatency::Normal);
    // Buffer index of the toggle (480-frame buffers): the fade then runs
    // over the loudest (clipped) segment of the signal.
    let at = 31usize;
    let wet = util::run_element(FACTORY, &[], rate, ch, &input, 480, &[]);
    let toggled = util::run_element(
        FACTORY,
        &[],
        rate,
        ch,
        &input,
        480,
        &[(at, Action::SetProperty("enabled", false.to_value()))],
    );
    let dry = |n: usize, c: usize| {
        if n < lat {
            0.0
        } else {
            input[(n - lat) * ch + c]
        }
    };
    let start = at * 480;
    let fade = (strom_dsp::declipper::XFADE_MS as usize) * 48;
    let mut last = 1.0f32;
    let mut weighed = 0;
    for n in start..start + fade {
        for c in 0..ch {
            let d = dry(n, c);
            let w = wet[n * ch + c];
            if (w - d).abs() > 1e-3 {
                let g = (toggled[n * ch + c] - d) / (w - d);
                assert!((-1e-4..=1.0 + 1e-4).contains(&g), "frame {n}: weight {g}");
                assert!(g <= last + 1e-4, "frame {n}: weight rose {last} -> {g}");
                last = g;
                weighed += 1;
            }
        }
    }
    assert!(weighed > 0, "no repaired samples inside the fade");
    for n in start + fade..input.len() / ch {
        for c in 0..ch {
            assert_eq!(
                toggled[n * ch + c].to_bits(),
                dry(n, c).to_bits(),
                "frame {n} after the fade"
            );
        }
    }
    // Before the toggle the output is the repaired one.
    util::assert_bit_exact(&toggled[..start * ch], &wet[..start * ch]);
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

/// Latency per setting at 44.1, 48 and 96 kHz: the LATENCY query adds
/// exactly the configured delay, which stays within the setting's budget
/// (3, 5 and 10 ms); an impulse comes out delayed by exactly that many
/// frames with `enabled` true and false.
#[test]
fn latency_per_setting_query_budget_and_delay() {
    util::init();
    for (nick, mode, budget_ms) in MODES {
        for rate in [44_100u32, 48_000, 96_000] {
            let ctx = format!("{nick} at {rate} Hz");
            let lat = frames(rate, mode);
            let ours = frames_time(rate, lat);
            assert!(
                ours.nseconds() as f64 <= budget_ms * 1e6,
                "{ctx}: {ours} over the {budget_ms} ms budget"
            );
            let mut h = Harness::new(FACTORY);
            let upstream = gst::ClockTime::from_mseconds(10);
            h.set_upstream_latency(upstream);
            let e = h.element().unwrap();
            e.set_property_from_value("latency-mode", &mode_value(nick));
            h.set_src_caps_str(&util::caps_str(rate, 2));
            h.play();
            push_one(&mut h, 2);
            let (live, min, max) = query_latency(&h);
            assert!(live, "{ctx}");
            assert_eq!(min, upstream + ours, "{ctx}");
            let mut upstream_q = gst::query::Latency::new();
            assert!(h.srcpad().unwrap().query(&mut upstream_q));
            assert_eq!(max, upstream_q.result().2.map(|m| m + ours), "{ctx}");
            e.set_property("enabled", false);
            push_one(&mut h, 2);
            assert_eq!(query_latency(&h).1, upstream + ours, "{ctx}: disabled");

            for enabled in [true, false] {
                let mut input = vec![0.0f32; rate as usize / 20];
                input[100] = 0.8;
                let out = util::run_element(
                    FACTORY,
                    &[
                        ("latency-mode", mode_value(nick)),
                        ("enabled", enabled.to_value()),
                    ],
                    rate,
                    1,
                    &input,
                    256,
                    &[],
                );
                let peak = out
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                    .unwrap();
                assert_eq!(peak.0, 100 + lat, "{ctx}, enabled {enabled}");
                assert_eq!(*peak.1, 0.8, "{ctx}, enabled {enabled}");
            }
        }
    }
}

/// A `latency-mode` write while playing changes nothing until the next
/// caps negotiation, which applies it and posts a Latency message; the
/// value reads back at once.
#[test]
fn latency_mode_waits_for_caps_and_posts_latency() {
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
    let standard = frames_time(48_000, frames(48_000, DeclipperLatency::Normal));
    assert_eq!(query_latency(&h).1, standard);

    e.set_property_from_value("latency-mode", &mode_value("low"));
    assert_eq!(mode_nick(&e), "low");
    for _ in 0..3 {
        push_one(&mut h, 2);
    }
    assert!(bus.pop_filtered(&[gst::MessageType::Latency]).is_none());
    assert_eq!(query_latency(&h).1, standard);

    h.set_src_caps_str(&util::caps_str(44_100, 2));
    push_one(&mut h, 2);
    let msg = bus
        .pop_filtered(&[gst::MessageType::Latency])
        .expect("no latency message after renegotiation");
    assert_eq!(msg.src(), Some(e.upcast_ref::<gst::Object>()));
    assert_eq!(
        query_latency(&h).1,
        frames_time(44_100, frames(44_100, DeclipperLatency::Low))
    );

    // Same latency again: no message.
    h.set_src_caps_str(&util::caps_str(44_100, 1));
    push_one(&mut h, 1);
    assert!(bus.pop_filtered(&[gst::MessageType::Latency]).is_none());
    e.set_bus(None);
}

/// Transparency: clean programme (the util signal scaled under full scale,
/// a full-scale 1 kHz tone, a 997 Hz tone at −0.1 dBFS) comes out bit-exact,
/// delayed by the latency, in every latency setting.
#[test]
fn clean_programme_is_bit_exact() {
    let rate = 48_000u32;
    let n = rate as usize / 2;
    let mut programme = util::test_signal(rate, 2, n);
    let peak = programme.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    for v in &mut programme {
        *v *= 0.99 / peak;
    }
    let tone = |f: f64, a: f64| -> Vec<f32> {
        (0..n)
            .flat_map(|i| {
                let v = (a * (2.0 * std::f64::consts::PI * f * i as f64 / 48_000.0).sin()) as f32;
                [v, v]
            })
            .collect()
    };
    for (what, input) in [
        ("programme", programme),
        ("1 kHz at 0 dBFS", tone(1000.0, 1.0)),
        ("997 Hz at -0.1 dBFS", tone(997.0, 0.988_6)),
    ] {
        for (nick, mode, _) in MODES {
            let lat = frames(rate, mode);
            let out = util::run_element(
                FACTORY,
                &[("latency-mode", mode_value(nick))],
                rate,
                2,
                &input,
                480,
                &[],
            );
            assert!(out[..lat * 2].iter().all(|&v| v == 0.0), "{what} {nick}");
            util::assert_bit_exact(&out[lat * 2..], &input[..input.len() - lat * 2]);
        }
    }
}

// ---------------------------------------------------------------------------
// Meter message
// ---------------------------------------------------------------------------

/// Push `input` (stereo, 48 kHz) in 10 ms buffers and collect the meter
/// messages.
fn messages(input: &[f32], interval: u32) -> Vec<gst::Structure> {
    util::init();
    let mut h = Harness::new(FACTORY);
    let e = h.element().unwrap();
    e.set_property("meter-interval", interval);
    let bus = gst::Bus::new();
    e.set_bus(Some(&bus));
    h.set_src_caps_str(&util::caps_str(48_000, 2));
    h.play();
    let dur = gst::ClockTime::from_mseconds(10);
    for (i, chunk) in input.chunks(960).enumerate() {
        let mut b = gst::Buffer::from_mut_slice(
            chunk
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        {
            let b = b.get_mut().unwrap();
            b.set_pts(dur * i as u64);
            b.set_duration(dur);
        }
        assert_eq!(h.push(b), Ok(gst::FlowSuccess::Ok));
        let _ = h.pull().unwrap();
    }
    let mut v = Vec::new();
    while let Some(m) = bus.pop_filtered(&[gst::MessageType::Element]) {
        if let gst::MessageView::Element(el) = m.view() {
            let s = el.structure().unwrap();
            if s.name() == MESSAGE_NAME {
                v.push(s.to_owned());
            }
        }
    }
    e.set_bus(None);
    v
}

#[test]
fn meter_message_counts_repairs() {
    let input = clipped(48_000, 2, 48_000, 0.7);
    let msgs = messages(&input, 100);
    assert_eq!(msgs.len(), 10, "one message per 100 ms");
    let sum = |f: &str| msgs.iter().map(|s| s.get::<u64>(f).unwrap()).sum::<u64>();
    assert!(sum("runs-repaired") > 0);
    assert!(sum("samples-repaired") >= sum("runs-repaired"));
    assert!(sum("fallbacks") <= sum("runs-repaired"));
    // Every count field is there (a missing one panics in `sum`).
    let _ = sum("runs-too-long");
    let _ = sum("runs-rejected");
    let _ = sum("runs-failed");
    let over = msgs
        .iter()
        .map(|s| s.get::<f64>("largest-overshoot").unwrap())
        .fold(0.0f64, f64::max);
    assert!(over > 0.0 && over <= 6.0 + 1e-6, "overshoot {over}");
    let times: Vec<u64> = msgs
        .iter()
        .map(|s| s.get::<u64>("running-time").unwrap())
        .collect();
    assert_eq!(times[0], 100_000_000);
    assert!(times.windows(2).all(|w| w[1] - w[0] == 100_000_000));

    // The same counts as the library over the same audio.
    let mut d = Declipper::new();
    d.set_params(&DeclipperParams::default());
    d.configure(48_000.0, 2);
    let mut buf = input.clone();
    d.process(&mut buf);
    assert_eq!(sum("runs-repaired"), d.stats().runs_repaired);
    assert_eq!(sum("runs-rejected"), d.stats().runs_rejected);
    assert_eq!(sum("runs-failed"), d.stats().runs_failed);

    // Clean audio: messages with nothing repaired. Interval 0: none.
    let clean: Vec<f32> = util::test_signal(48_000, 2, 48_000)
        .iter()
        .map(|v| v * 0.7)
        .collect();
    let msgs = messages(&clean, 250);
    assert_eq!(msgs.len(), 4);
    assert!(msgs
        .iter()
        .all(|s| s.get::<u64>("runs-repaired").unwrap() == 0
            && s.get::<f64>("largest-overshoot").unwrap() == 0.0));
    assert!(messages(&input, 0).is_empty());
}
