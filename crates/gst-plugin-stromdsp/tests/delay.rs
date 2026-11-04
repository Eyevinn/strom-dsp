//! Harness tests for `stromdsp-delay`: output matches the library (with
//! property changes, DISCONT and flush), timestamps are unchanged (checked by
//! `util::run_element` for every buffer), the LATENCY query is unchanged,
//! `channel-delays` works with arrays shorter or longer than the channel
//! count, and the defaults / `enabled=false` are bit-exact.

use crate::util;

use crate::util::{Action, FORMATS};
use gst::glib;
use gst::prelude::*;
use gst_check::Harness;
use strom_dsp::delay::{Delay, DelayParams};

fn array(v: &[f32]) -> glib::Value {
    gst::Array::new(v.iter().copied()).to_value()
}

fn on_set(dsp: &mut Delay, name: &str, value: &glib::Value) {
    let mut p = *dsp.params();
    match name {
        "enabled" => p.enabled = value.get().unwrap(),
        "delay" => p.delay_ms = value.get().unwrap(),
        "channel-delays" => {
            let a: gst::Array = value.get().unwrap();
            let v: Vec<f32> = a.iter().map(|x| x.get::<f32>().unwrap()).collect();
            p.set_offsets(&v);
        }
        _ => unreachable!(),
    }
    dsp.set_params(&p);
}

#[test]
fn delay_matches_library() {
    util::init();
    for (rate, channels) in FORMATS {
        let fpb = if channels == 2 { 1024 } else { 1500 };
        let input = util::test_signal(rate, channels, rate as usize);
        let offsets = [0.0f32, 1.5, -0.5];
        let props = vec![
            ("delay", 12.0f32.to_value()),
            ("channel-delays", array(&offsets)),
        ];
        let actions = [
            (3, Action::SetProperty("delay", 30.0f32.to_value())),
            (4, Action::SetProperty("channel-delays", array(&[2.0]))),
            (6, Action::Discont),
            (8, Action::SetProperty("enabled", false.to_value())),
            (10, Action::SetProperty("enabled", true.to_value())),
            (12, Action::Flush),
            (13, Action::SetProperty("delay", 0.0f32.to_value())),
        ];
        let out = util::run_element(
            "stromdsp-delay",
            &props,
            rate,
            channels,
            &input,
            fpb,
            &actions,
        );
        let mut dsp = Delay::new();
        let mut p = DelayParams {
            delay_ms: 12.0,
            ..DelayParams::default()
        };
        p.set_offsets(&offsets);
        dsp.set_params(&p);
        let reference = util::run_reference(dsp, rate, channels, &input, fpb, &actions, on_set);
        // The element hands the library the same samples: exact.
        util::assert_bit_exact(&out, &reference);
    }
}

#[test]
fn defaults_and_disabled_are_bit_exact() {
    util::init();
    for (rate, channels) in FORMATS {
        let input = util::test_signal(rate, channels, rate as usize / 2);
        let out = util::run_element("stromdsp-delay", &[], rate, channels, &input, 480, &[]);
        util::assert_bit_exact(&out, &input);
        let props = vec![
            ("enabled", false.to_value()),
            ("delay", 100.0f32.to_value()),
            ("channel-delays", array(&[5.0, 7.0])),
        ];
        let out = util::run_element("stromdsp-delay", &props, rate, channels, &input, 480, &[]);
        util::assert_bit_exact(&out, &input);
    }
}

/// Impulse through the element: per-channel delay = round((delay + offset)
/// · rate / 1000) frames; missing offsets are 0, extra ones are ignored.
#[test]
fn channel_delays_shorter_and_longer_than_channel_count() {
    util::init();
    let rate = 48_000u32;
    let channels = 3usize;
    for offsets in [vec![1.0f32], vec![1.0, 2.0, 3.0, 4.0, 5.0]] {
        let frames = 4800;
        let mut input = vec![0.0f32; frames * channels];
        input[..channels].fill(1.0);
        let props = vec![
            ("delay", 10.0f32.to_value()),
            ("channel-delays", array(&offsets)),
        ];
        let out = util::run_element("stromdsp-delay", &props, rate, channels, &input, 256, &[]);
        for c in 0..channels {
            let off = offsets.get(c).copied().unwrap_or(0.0);
            let want = ((10.0 + off) as f64 * f64::from(rate) / 1000.0).round() as usize;
            let hits: Vec<usize> = (0..frames)
                .filter(|&f| out[f * channels + c] != 0.0)
                .collect();
            assert_eq!(hits, vec![want], "offsets {offsets:?} channel {c}");
        }
    }
}

#[test]
fn channel_delays_property_round_trip() {
    util::init();
    let e = gst::ElementFactory::make("stromdsp-delay").build().unwrap();
    let get = |e: &gst::Element| -> Vec<f32> {
        let a = e.property::<gst::Array>("channel-delays");
        a.iter().map(|v| v.get::<f32>().unwrap()).collect()
    };
    assert!(get(&e).is_empty());
    e.set_property("channel-delays", gst::Array::new([1.5f32, -2.0, 0.25]));
    assert_eq!(get(&e), vec![1.5, -2.0, 0.25]);
    e.set_property("channel-delays", gst::Array::new([-2000.0f32, 2000.0]));
    assert_eq!(get(&e), vec![-2000.0, 2000.0]);
    // From a string, as gst-launch does: untyped numbers are deserialised
    // with the float element spec.
    e.set_property_from_str("channel-delays", "<1.0, 2.5, 3>");
    assert_eq!(get(&e), vec![1.0, 2.5, 3.0]);
    e.set_property("channel-delays", gst::Array::new(Vec::<f32>::new()));
    assert!(get(&e).is_empty());
}

fn push_one(h: &mut Harness, channels: usize, pts: gst::ClockTime) {
    let mut buf = gst::Buffer::from_mut_slice(vec![0u8; 64 * channels * 4]);
    buf.get_mut().unwrap().set_pts(pts);
    assert_eq!(h.push(buf), Ok(gst::FlowSuccess::Ok));
    let out = h.pull().unwrap();
    assert_eq!(
        out.pts(),
        Some(pts),
        "timestamps must pass through unchanged"
    );
}

/// The delay is intentional and must not be reported as latency, at any
/// setting, including while it changes.
#[test]
fn latency_query_is_unchanged() {
    util::init();
    let mut h = Harness::new("stromdsp-delay");
    let element = h.element().unwrap();
    let bus = gst::Bus::new();
    element.set_bus(Some(&bus));
    let upstream = gst::ClockTime::from_mseconds(7);
    h.set_upstream_latency(upstream);
    h.set_src_caps_str(&util::caps_str(48_000, 2));
    h.play();
    let src = element.static_pad("src").unwrap();
    let mut pts = gst::ClockTime::ZERO;
    for (delay, offs) in [
        (0.0f32, Vec::<f32>::new()),
        (500.0, vec![10.0f32]),
        (2000.0, vec![0.0, -100.0]),
    ] {
        element.set_property("delay", delay);
        element.set_property("channel-delays", gst::Array::new(offs));
        push_one(&mut h, 2, pts);
        pts += gst::ClockTime::from_mseconds(1);
        let mut q = gst::query::Latency::new();
        assert!(src.query(&mut q));
        let (live, min, _) = q.result();
        assert!(live);
        assert_eq!(min, upstream, "delay {delay} must not add latency");
        assert_eq!(h.query_latency(), Some(upstream));
    }
    assert!(
        bus.pop_filtered(&[gst::MessageType::Latency]).is_none(),
        "no latency message for an intentional delay"
    );
}
