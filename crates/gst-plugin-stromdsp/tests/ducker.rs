//! Tests for `stromdsp-ducker`:
//!
//! - `gst_check::Harness` on both pads (non-live, so the aggregator waits for
//!   both): output equals the library run on the same programme and key spans,
//!   with a property change while playing, a key at another rate and channel
//!   count, and timestamps/caps passed through;
//! - `enabled=false` is bit-exact; no sidechain pad and a key that reaches EOS
//!   never block and end at unity;
//! - meter messages: content (against the library) and rate;
//! - live pipelines (`audiotestsrc is-live=true` on both inputs): a key that
//!   stops and is unlinked, and a requested but never linked sidechain, never
//!   stall the programme, which recovers to unity.

use crate::util;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use gst::prelude::*;
use gst_check::Harness;
use strom_dsp::ducker::{Ducker, DuckerParams, RangeMode};
use strom_dsp::Processor;

const RATE: u32 = 48_000;
const CH: usize = 2;
/// Programme frames per buffer (10 ms).
const FRAMES: usize = 480;

fn to_bytes(s: &[f32]) -> Vec<u8> {
    s.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn from_bytes(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn buffer(samples: &[f32], pts: gst::ClockTime, dur: gst::ClockTime) -> gst::Buffer {
    let mut b = gst::Buffer::from_mut_slice(to_bytes(samples));
    {
        let b = b.get_mut().unwrap();
        b.set_pts(pts);
        b.set_duration(dur);
    }
    b
}

/// Key: 1 kHz bursts (on 300 ms, off 500 ms) at amplitude `amp`.
fn key_signal(rate: u32, ch: usize, frames: usize, amp: f32) -> Vec<f32> {
    (0..frames)
        .flat_map(|i| {
            let t = i as f64 / rate as f64;
            let on = (t % 0.8) < 0.3;
            let v = if on {
                amp * (2.0 * std::f64::consts::PI * 1000.0 * t).sin() as f32
            } else {
                0.0
            };
            (0..ch).map(move |c| if c % 2 == 0 { v } else { -v })
        })
        .collect()
}

fn params() -> DuckerParams {
    DuckerParams {
        threshold_db: -30.0,
        depth_db: -15.0,
        attack_ms: 10.0,
        hold_ms: 50.0,
        release_ms: 150.0,
        key_highpass_hz: 100.0,
        range_mode: RangeMode::Fixed,
    }
}

fn set_props(e: &gst::Element, p: &DuckerParams) {
    e.set_property("threshold", p.threshold_db);
    e.set_property("depth", p.depth_db);
    e.set_property("attack", p.attack_ms);
    e.set_property("hold", p.hold_ms);
    e.set_property("release", p.release_ms);
    e.set_property("key-highpass", p.key_highpass_hz);
    e.set_property_from_str(
        "range-mode",
        match p.range_mode {
            RangeMode::Fixed => "fixed",
            RangeMode::Proportional => "proportional",
        },
    );
}

/// A ducker in a programme harness (`h`) plus, optionally, a key harness on a
/// requested `sidechain` pad (`hk`). Both non-live.
struct Rig {
    h: Harness,
    hk: Option<Harness>,
    element: gst::Element,
}

fn rig(key: Option<(u32, usize)>, p: &DuckerParams, extra: &[(&str, glib::Value)]) -> Rig {
    util::init();
    let mut h = Harness::new("stromdsp-ducker");
    let element = h.element().unwrap();
    set_props(&element, p);
    for (n, v) in extra {
        element.set_property_from_value(n, v);
    }
    h.set_live(false);
    h.set_src_caps_str(&util::caps_str(RATE, CH));
    let hk = key.map(|(kr, kc)| {
        let mut hk = Harness::with_element(&element, Some("sidechain"), None);
        hk.set_live(false);
        hk.set_src_caps_str(&util::caps_str(kr, kc));
        hk
    });
    h.play();
    Rig { h, hk, element }
}

use gst::glib;

/// Push `prog` (and the matching spans of `key`) buffer by buffer and pull
/// the output; `before(i, element)` runs before buffer `i`. Checks that
/// timestamps, durations and sizes pass through.
fn run(
    rig: &mut Rig,
    prog: &[f32],
    key: Option<(&[f32], u32, usize)>,
    mut before: impl FnMut(usize, &gst::Element),
) -> Vec<f32> {
    let mut out = Vec::with_capacity(prog.len());
    let dur = gst::ClockTime::from_mseconds(10);
    for (i, chunk) in prog.chunks(FRAMES * CH).enumerate() {
        before(i, &rig.element);
        let pts = dur * i as u64;
        assert_eq!(
            rig.h.push(buffer(chunk, pts, dur)),
            Ok(gst::FlowSuccess::Ok)
        );
        if let (Some(hk), Some((k, kr, kc))) = (rig.hk.as_mut(), key) {
            let per = kr as usize / 100;
            let ks = &k[i * per * kc..(i + 1) * per * kc];
            assert_eq!(hk.push(buffer(ks, pts, dur)), Ok(gst::FlowSuccess::Ok));
        }
        let o = rig.h.pull().expect("no output");
        assert_eq!(o.pts(), Some(pts), "PTS changed");
        assert_eq!(o.duration(), Some(dur), "duration changed");
        assert_eq!(o.size(), chunk.len() * 4);
        out.extend(from_bytes(&o.map_readable().unwrap()));
    }
    let caps = rig.h.sinkpad().unwrap().current_caps().unwrap();
    let expected: gst::Caps = util::caps_str(RATE, CH).parse().unwrap();
    assert!(caps.is_strictly_equal(&expected), "caps changed: {caps}");
    out
}

/// Library reference for the same buffers.
fn reference(
    p: &DuckerParams,
    prog: &[f32],
    key: Option<(&[f32], u32, usize)>,
    mut before: impl FnMut(usize, &mut Ducker),
) -> (Vec<f32>, Vec<(f64, bool, f64)>) {
    let mut d = Ducker::new();
    d.set_params(p);
    d.configure(RATE as f32, CH);
    if let Some((_, kr, kc)) = key {
        d.set_key_format(kr as f32, kc);
    }
    let mut out = prog.to_vec();
    let mut meters = Vec::new();
    for (i, chunk) in out.chunks_mut(FRAMES * CH).enumerate() {
        before(i, &mut d);
        match key {
            Some((k, kr, kc)) => {
                let per = kr as usize / 100;
                d.process_with_key(chunk, &k[i * per * kc..(i + 1) * per * kc], kc);
            }
            None => d.process(chunk),
        }
        meters.push((d.reduction_db(), d.key_active(), d.key_level_db()));
    }
    (out, meters)
}

fn programme(secs: f64) -> Vec<f32> {
    let n = (RATE as f64 * secs) as usize;
    (0..n)
        .flat_map(|i| {
            let v = 0.5 * (2.0 * std::f64::consts::PI * 220.0 * i as f64 / RATE as f64).sin();
            [v as f32, -(v as f32) * 0.5]
        })
        .collect()
}

#[test]
fn harness_matches_library() {
    for (kr, kc) in [(48_000u32, 1usize), (16_000, 2), (44_100, 1)] {
        let p = params();
        let prog = programme(2.0);
        let key = key_signal(kr, kc, kr as usize * 2, 0.3);
        let k = Some((key.as_slice(), kr, kc));
        let mut r = rig(Some((kr, kc)), &p, &[]);
        let out = run(&mut r, &prog, k, |i, e| {
            if i == 120 {
                e.set_property("depth", -6.0f32);
                e.set_property_from_str("range-mode", "proportional");
            }
        });
        let (expected, _) = reference(&p, &prog, k, |i, d| {
            if i == 120 {
                d.set_params(&DuckerParams {
                    depth_db: -6.0,
                    range_mode: RangeMode::Proportional,
                    ..p
                });
            }
        });
        util::assert_bit_exact(&out, &expected);
        // And it really ducked.
        let min = out.iter().map(|v| v.abs()).fold(f32::MAX, f32::min);
        assert!(
            expected
                .iter()
                .zip(&prog)
                .any(|(o, i)| (o / i - 10f32.powf(-0.75)).abs() < 1e-3 && i.abs() > 0.1),
            "never reached depth (min {min})"
        );
    }
}

#[test]
fn disabled_is_bit_exact() {
    let p = params();
    let prog = programme(1.0);
    let key = key_signal(RATE, 1, RATE as usize, 0.5);
    let mut r = rig(Some((RATE, 1)), &p, &[("enabled", false.to_value())]);
    let out = run(&mut r, &prog, Some((&key, RATE, 1)), |_, _| {});
    util::assert_bit_exact(&out, &prog);
}

/// Without a sidechain pad the programme passes at unity (bit-exact: the
/// gain is exactly 1), and nothing waits for a key.
#[test]
fn no_sidechain_is_unity() {
    let prog = programme(0.5);
    let mut r = rig(None, &params(), &[]);
    let out = run(&mut r, &prog, None, |_, _| {});
    util::assert_bit_exact(&out, &prog);
}

/// The key reaches EOS while ducking: the programme keeps flowing (no
/// blocking, even non-live) and recovers to unity as the library does.
#[test]
fn key_eos_releases_to_unity() {
    // Constant key: the key high-pass must be off.
    let p = DuckerParams {
        key_highpass_hz: 0.0,
        ..params()
    };
    let prog = programme(1.5);
    let key = vec![0.3f32; RATE as usize * 2];
    let mut r = rig(Some((RATE, 1)), &p, &[]);
    let dur = gst::ClockTime::from_mseconds(10);
    let mut out = Vec::new();
    for (i, chunk) in prog.chunks(FRAMES * CH).enumerate() {
        let pts = dur * i as u64;
        r.h.push(buffer(chunk, pts, dur)).unwrap();
        if i < 30 {
            let ks = &key[i * FRAMES..(i + 1) * FRAMES];
            r.hk.as_mut().unwrap().push(buffer(ks, pts, dur)).unwrap();
        } else if i == 30 {
            assert!(r.hk.as_mut().unwrap().push_event(gst::event::Eos::new()));
        }
        out.extend(from_bytes(&r.h.pull().unwrap().map_readable().unwrap()));
    }
    // Reference: key for 30 buffers, then none (silence).
    let (expected, _) = {
        let mut d = Ducker::new();
        d.set_params(&p);
        d.configure(RATE as f32, CH);
        d.set_key_format(RATE as f32, 1);
        let mut o = prog.clone();
        for (i, c) in o.chunks_mut(FRAMES * CH).enumerate() {
            if i < 30 {
                d.process_with_key(c, &key[i * FRAMES..(i + 1) * FRAMES], 1);
            } else {
                // The element passes zeros for the missing key span.
                d.process_with_key(c, &[0.0; FRAMES], 1);
            }
        }
        (o, ())
    };
    util::assert_bit_exact(&out, &expected);
    let tail = &out[out.len() - FRAMES * CH..];
    let src = &prog[prog.len() - FRAMES * CH..];
    assert_eq!(tail, src, "not back at unity");
    let ducked = &out[25 * FRAMES * CH..26 * FRAMES * CH];
    assert!(ducked
        .iter()
        .zip(&prog[25 * FRAMES * CH..])
        .all(|(o, i)| (o - i * 10f32.powf(-0.75)).abs() < 1e-6));
}

/// Meter messages: one at the first buffer, then whenever at least
/// `meter-interval` of running time has passed; the content matches the
/// library's meters at that buffer.
#[test]
fn meter_messages_content_and_rate() {
    let p = params();
    let prog = programme(2.0);
    let key = key_signal(RATE, 1, RATE as usize * 2, 0.3);
    let k = Some((key.as_slice(), RATE, 1));
    let mut r = rig(Some((RATE, 1)), &p, &[("meter-interval", 45u32.to_value())]);
    let bus = gst::Bus::new();
    r.element.set_bus(Some(&bus));
    run(&mut r, &prog, k, |_, _| {});
    let (_, meters) = reference(&p, &prog, k, |_, _| {});

    let msgs: Vec<gst::Structure> =
        std::iter::from_fn(|| bus.pop_filtered(&[gst::MessageType::Element]))
            .filter_map(|m| m.structure().map(|s| s.to_owned()))
            .filter(|s| s.name() == "stromdsp-ducker")
            .collect();
    // Buffers end at 10, 20, … ms: messages at 10, 60, 110, … (every 5th
    // buffer for 45 ms), 2 s → 40 messages.
    assert_eq!(msgs.len(), 40, "message count");
    let mut active_seen = (false, false);
    for (n, s) in msgs.iter().enumerate() {
        let rt = s.get::<gst::ClockTime>("running-time").unwrap();
        let st = s.get::<gst::ClockTime>("stream-time").unwrap();
        let buf = 5 * n; // index of the buffer that posted it
        assert_eq!(rt, gst::ClockTime::from_mseconds(10 * (buf as u64 + 1)));
        assert_eq!(st, rt, "default segment: stream time = running time");
        let (red, act, lvl) = meters[buf];
        assert_eq!(s.get::<f64>("reduction").unwrap(), red, "msg {n}");
        assert_eq!(s.get::<bool>("key-active").unwrap(), act, "msg {n}");
        assert_eq!(s.get::<f64>("key-level").unwrap(), lvl, "msg {n}");
        assert!(s.get::<bool>("key-present").unwrap());
        if act {
            active_seen.0 = true;
        } else {
            active_seen.1 = true;
        }
    }
    assert!(active_seen.0 && active_seen.1, "both states reported");
    r.element.set_bus(None::<&gst::Bus>);
}

#[test]
fn meter_interval_zero_is_off() {
    let mut r = rig(None, &params(), &[("meter-interval", 0u32.to_value())]);
    let bus = gst::Bus::new();
    r.element.set_bus(Some(&bus));
    run(&mut r, &programme(0.3), None, |_, _| {});
    assert!(bus.pop_filtered(&[gst::MessageType::Element]).is_none());
    r.element.set_bus(None::<&gst::Bus>);
}

// ---------------------------------------------------------------------------
// Live pipelines.

/// (time received, peak of the left channel) per output buffer.
type Trace = Arc<Mutex<Vec<(Instant, f32)>>>;

fn live_pipeline(sidechain: &str) -> (gst::Pipeline, Trace) {
    util::init();
    let desc = format!(
        "audiotestsrc name=prog is-live=true wave=sine freq=220 volume=0.5 samplesperbuffer=480 \
           ! audio/x-raw,format=F32LE,rate=48000,channels=2 \
           ! stromdsp-ducker name=d threshold=-30 depth=-15 attack=10 hold=50 release=150 \
           ! fakesink name=out sync=true \
         {sidechain}"
    );
    let pipeline = gst::parse::launch(&desc)
        .unwrap()
        .downcast::<gst::Pipeline>()
        .unwrap();
    let trace: Trace = Arc::default();
    let d = pipeline.by_name("d").unwrap();
    let t = trace.clone();
    d.static_pad("src")
        .unwrap()
        .add_probe(gst::PadProbeType::BUFFER, move |_, info| {
            if let Some(b) = info.buffer() {
                let m = b.map_readable().unwrap();
                let s = from_bytes(&m);
                let peak = s.iter().step_by(2).fold(0.0f32, |a, v| a.max(v.abs()));
                t.lock().unwrap().push((Instant::now(), peak));
            }
            gst::PadProbeReturn::Ok
        });
    (pipeline, trace)
}

fn play_for(pipeline: &gst::Pipeline, d: Duration) {
    let bus = pipeline.bus().unwrap();
    let end = Instant::now() + d;
    while Instant::now() < end {
        if let Some(m) = bus.timed_pop_filtered(
            gst::ClockTime::from_mseconds(20),
            &[gst::MessageType::Error],
        ) {
            panic!("pipeline error: {m:?}");
        }
    }
}

/// Buffers received within `[from, to)`, and the last peak.
fn window(trace: &Trace, from: Instant, to: Instant) -> (usize, f32) {
    let t = trace.lock().unwrap();
    let w: Vec<_> = t.iter().filter(|(i, _)| *i >= from && *i < to).collect();
    (w.len(), w.last().map_or(0.0, |x| x.1))
}

const DUCKED: f32 = 0.5 * 0.177_827_94; // 0.5 · 10^(−15/20)

#[test]
fn live_key_stops_and_unlinks_without_stalling() {
    let (pipeline, trace) = live_pipeline(
        "audiotestsrc name=key is-live=true wave=sine freq=1000 volume=0.3 samplesperbuffer=160 \
           ! audio/x-raw,format=F32LE,rate=16000,channels=1 ! d.sidechain",
    );
    pipeline.set_state(gst::State::Playing).unwrap();
    play_for(&pipeline, Duration::from_millis(800));
    let t_stop = Instant::now();
    let (_, peak) = window(&trace, t_stop - Duration::from_millis(200), t_stop);
    assert!((peak - DUCKED).abs() < 0.01, "not ducked: {peak}");

    // Stop the key source, then unlink it from the sidechain.
    let key = pipeline.by_name("key").unwrap();
    key.set_locked_state(true);
    key.set_state(gst::State::Null).unwrap();
    let side = pipeline
        .by_name("d")
        .unwrap()
        .static_pad("sidechain")
        .unwrap();
    if let Some(peer) = side.peer() {
        peer.unlink(&side).unwrap();
    }
    play_for(&pipeline, Duration::from_millis(1200));
    let t_end = Instant::now();
    // The programme kept flowing at (about) real time: 10 ms buffers.
    let (n, _) = window(&trace, t_stop, t_end);
    let expected = (t_end - t_stop).as_millis() as usize / 10;
    assert!(
        n * 10 >= expected * 7,
        "stalled: {n} buffers in {expected}0 ms"
    );
    // ... and recovered to unity (hold 50 + release 150 ms + detector decay).
    let (_, peak) = window(&trace, t_end - Duration::from_millis(200), t_end);
    assert!((peak - 0.5).abs() < 0.005, "not released: {peak}");
    pipeline.set_state(gst::State::Null).unwrap();
}

#[test]
fn live_requested_but_unlinked_sidechain_does_not_block() {
    let (pipeline, trace) = live_pipeline("");
    let d = pipeline.by_name("d").unwrap();
    let _side = d.request_pad_simple("sidechain").unwrap();
    pipeline.set_state(gst::State::Playing).unwrap();
    let t0 = Instant::now();
    play_for(&pipeline, Duration::from_millis(800));
    let t1 = Instant::now();
    let (n, peak) = window(&trace, t0, t1);
    assert!(n >= 50, "only {n} buffers in 800 ms");
    assert!((peak - 0.5).abs() < 0.005, "ducked without a key: {peak}");
    pipeline.set_state(gst::State::Null).unwrap();
}

#[test]
fn live_without_sidechain_pad() {
    let (pipeline, trace) = live_pipeline("");
    pipeline.set_state(gst::State::Playing).unwrap();
    let t0 = Instant::now();
    play_for(&pipeline, Duration::from_millis(500));
    let (n, peak) = window(&trace, t0, Instant::now());
    assert!(n >= 30, "only {n} buffers in 500 ms");
    assert!((peak - 0.5).abs() < 0.005);
    pipeline.set_state(gst::State::Null).unwrap();
}
