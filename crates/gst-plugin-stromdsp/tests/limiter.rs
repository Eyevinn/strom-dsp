//! Harness tests for `stromdsp-limiter`, including latency reporting.

mod util;

use gst::glib;
use gst::prelude::*;
use gst_check::Harness;
use strom_dsp::{Limiter, LimiterParams};
use util::{Action, FORMATS, TOL};

fn params() -> LimiterParams {
    LimiterParams {
        enabled: true,
        threshold_db: -6.0,
        attack_ms: 0.5,
        release_ms: 50.0,
        lookahead_ms: 3.0,
    }
}

fn props(p: &LimiterParams) -> Vec<(&'static str, glib::Value)> {
    vec![
        ("enabled", p.enabled.to_value()),
        ("threshold", p.threshold_db.to_value()),
        ("attack", p.attack_ms.to_value()),
        ("release", p.release_ms.to_value()),
        ("lookahead", p.lookahead_ms.to_value()),
    ]
}

fn on_set(dsp: &mut Limiter, name: &str, value: &glib::Value) {
    let mut q = *dsp.params();
    match name {
        "enabled" => q.enabled = value.get().unwrap(),
        "threshold" => q.threshold_db = value.get().unwrap(),
        "lookahead" => q.lookahead_ms = value.get().unwrap(),
        _ => unreachable!(),
    }
    dsp.set_params(&q);
}

#[test]
fn limiter_matches_library() {
    for (rate, channels) in FORMATS {
        let fpb = if channels == 2 { 1024 } else { 1500 };
        let input = util::test_signal(rate, channels, rate as usize);
        let p = params();
        let actions = [
            (3, Action::SetProperty("threshold", (-3.0f32).to_value())),
            (5, Action::Discont),
            (7, Action::SetProperty("lookahead", 10.0f32.to_value())),
            (10, Action::Flush),
        ];
        let out = util::run_element(
            "stromdsp-limiter",
            &props(&p),
            rate,
            channels,
            &input,
            fpb,
            &actions,
        );
        let mut dsp = Limiter::new();
        dsp.set_params(&p);
        let reference = util::run_reference(dsp, rate, channels, &input, fpb, &actions, on_set);
        util::assert_close(&out, &reference, TOL);
    }
}

/// The limiter is never a BaseTransform passthrough: `enabled=false` is handed
/// to the library (unity gain, delay kept). Compare against the library with
/// the same setting, including toggling while running.
#[test]
fn limiter_disabled_matches_library() {
    for (rate, channels) in FORMATS {
        let input = util::test_signal(rate, channels, rate as usize / 2);
        let mut p = params();
        p.enabled = false;
        let actions = [(4, Action::SetProperty("enabled", true.to_value()))];
        let out = util::run_element(
            "stromdsp-limiter",
            &props(&p),
            rate,
            channels,
            &input,
            512,
            &actions,
        );
        let mut dsp = Limiter::new();
        dsp.set_params(&p);
        let reference = util::run_reference(dsp, rate, channels, &input, 512, &actions, on_set);
        util::assert_close(&out, &reference, TOL);
    }
}

/// Expected lookahead latency per the interface: round(ms * rate / 1000)
/// frames, expressed as time rounded up to whole nanoseconds.
fn expected_latency(lookahead_ms: f64, rate: u64) -> gst::ClockTime {
    let frames = (lookahead_ms * rate as f64 / 1000.0).round() as u64;
    gst::ClockTime::from_nseconds((frames * 1_000_000_000).div_ceil(rate))
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
fn limiter_latency_query_adds_lookahead() {
    util::init();
    for (rate, lookahead) in [
        (48_000u32, 5.0f32),
        (44_100, 5.0),
        (22_050, 3.3),
        (48_000, 0.0),
    ] {
        let mut h = Harness::new("stromdsp-limiter");
        let upstream = gst::ClockTime::from_mseconds(10);
        h.set_upstream_latency(upstream);
        h.element().unwrap().set_property("lookahead", lookahead);
        h.set_src_caps_str(&util::caps_str(rate, 2));
        h.play();
        push_one(&mut h, 2);

        let ours = expected_latency(f64::from(lookahead), u64::from(rate));
        let (live, min, max) = query_latency(&h);
        assert!(live);
        assert_eq!(min, upstream + ours, "rate {rate}, lookahead {lookahead}");
        // Whatever max upstream reports, ours is added to it.
        let mut upstream_q = gst::query::Latency::new();
        let harness_src = h.srcpad().unwrap();
        assert!(harness_src.query(&mut upstream_q));
        let (_, _, upstream_max) = upstream_q.result();
        assert_eq!(max, upstream_max.map(|m| m + ours));
        assert_eq!(h.query_latency(), Some(upstream + ours));
    }
}

#[test]
fn limiter_latency_message_on_lookahead_change() {
    util::init();
    let mut h = Harness::new("stromdsp-limiter");
    let element = h.element().unwrap();
    let bus = gst::Bus::new();
    element.set_bus(Some(&bus));
    h.set_upstream_latency(gst::ClockTime::ZERO);
    h.set_src_caps_str(&util::caps_str(48_000, 2));
    h.play();
    push_one(&mut h, 2);
    // Drain anything posted so far.
    while bus.pop().is_some() {}

    element.set_property("lookahead", 12.0f32);
    let msg = bus
        .pop_filtered(&[gst::MessageType::Latency])
        .expect("no latency message after lookahead change");
    assert_eq!(msg.src(), Some(element.upcast_ref::<gst::Object>()));
    assert_eq!(query_latency(&h).1, gst::ClockTime::from_mseconds(12));

    // Setting the same value again does not change the latency: no message.
    element.set_property("lookahead", 12.0f32);
    assert!(bus.pop_filtered(&[gst::MessageType::Latency]).is_none());

    // Other properties never post latency messages.
    element.set_property("threshold", -3.0f32);
    element.set_property("enabled", false);
    assert!(bus.pop_filtered(&[gst::MessageType::Latency]).is_none());

    // Disabling keeps the latency.
    assert_eq!(query_latency(&h).1, gst::ClockTime::from_mseconds(12));
}

/// Other elements have no internal latency and must not change the query.
#[test]
fn other_elements_do_not_add_latency() {
    util::init();
    for factory in ["stromdsp-gate", "stromdsp-compressor", "stromdsp-equalizer"] {
        let mut h = Harness::new(factory);
        let upstream = gst::ClockTime::from_mseconds(7);
        h.set_upstream_latency(upstream);
        h.set_src_caps_str(&util::caps_str(48_000, 2));
        h.play();
        push_one(&mut h, 2);
        assert_eq!(query_latency(&h).1, upstream, "{factory}");
    }
}
