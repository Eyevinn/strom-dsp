//! Shared helpers for the plugin integration tests.
//!
//! The harness tests push a deterministic signal through an element with
//! `gst_check::Harness` and compare the output against running the same
//! `strom-dsp` processor directly on the same input, with the same parameters
//! and the same buffer boundaries. They also check that buffer sizes, timestamps,
//! durations, offsets and caps are preserved.

#![allow(dead_code)]

use std::sync::Once;

use gst::glib;
use gst::prelude::*;
use gst_check::Harness;
use strom_dsp::Processor;

/// Initialise GStreamer and register the plugin statically (once per test binary).
pub fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        gst::init().unwrap();
        gststromdsp::plugin_register_static().expect("static registration failed");
    });
}

/// Caps string for interleaved F32LE.
pub fn caps_str(rate: u32, channels: usize) -> String {
    format!("audio/x-raw,format=F32LE,layout=interleaved,rate={rate},channels={channels}")
}

/// Deterministic multichannel test signal: a tone whose level alternates
/// between loud (above 0 dBFS peaks, to exercise limiter/compressor), medium and
/// very quiet (to make the gate close), plus a little pseudo-random noise. Each
/// channel has its own frequency and phase so channels differ.
pub fn test_signal(rate: u32, channels: usize, frames: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames * channels);
    let mut lcg: u32 = 0x1234_5678;
    let seg = (rate as usize / 20).max(1); // 50 ms segments
    let levels = [1.4f32, 0.3, 0.002, 0.8, 0.01, 0.05];
    for n in 0..frames {
        let level = levels[(n / seg) % levels.len()];
        let t = n as f32 / rate as f32;
        for c in 0..channels {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = ((lcg >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.002;
            let f = 220.0 * (c as f32 + 1.0);
            let phase = c as f32 * 0.7;
            out.push(level * (2.0 * std::f32::consts::PI * f * t + phase).sin() + noise);
        }
    }
    out
}

/// An action applied before the buffer with the given index is pushed.
pub enum Action {
    /// Set a property on the element; the reference applies `apply` instead.
    SetProperty(&'static str, glib::Value),
    /// Mark the buffer DISCONT (reference: `reset()` before processing it).
    Discont,
    /// Send FLUSH_START/FLUSH_STOP + a new segment (reference: `reset()`).
    Flush,
}

fn f32s_to_bytes(s: &[f32]) -> Vec<u8> {
    s.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Push `input` through `factory` in buffers of `frames_per_buffer` frames,
/// after setting `props`. `actions` are (buffer index, action). Returns the
/// concatenated output samples. Asserts that every output buffer has the same
/// size, PTS, duration and offsets as its input and that output caps equal
/// input caps.
pub fn run_element(
    factory: &str,
    props: &[(&str, glib::Value)],
    rate: u32,
    channels: usize,
    input: &[f32],
    frames_per_buffer: usize,
    actions: &[(usize, Action)],
) -> Vec<f32> {
    init();
    let mut h = Harness::new(factory);
    let element = h.element().unwrap();
    for (name, value) in props {
        element.set_property_from_value(name, value);
    }
    h.set_src_caps_str(&caps_str(rate, channels));
    h.play();

    let mut output = Vec::with_capacity(input.len());
    let samples_per_buffer = frames_per_buffer * channels;
    let mut pts = gst::ClockTime::ZERO;
    let mut offset = 0u64;
    for (idx, chunk) in input.chunks(samples_per_buffer).enumerate() {
        let mut discont = false;
        for (_, action) in actions.iter().filter(|(i, _)| *i == idx) {
            match action {
                Action::SetProperty(name, value) => element.set_property_from_value(name, value),
                Action::Discont => discont = true,
                Action::Flush => {
                    assert!(h.push_event(gst::event::FlushStart::new()));
                    assert!(h.push_event(gst::event::FlushStop::new(true)));
                    let segment = gst::FormattedSegment::<gst::ClockTime>::new();
                    assert!(h.push_event(gst::event::Segment::new(&segment)));
                }
            }
        }
        let frames = (chunk.len() / channels) as u64;
        let duration = gst::ClockTime::SECOND
            .mul_div_floor(frames, rate as u64)
            .unwrap();
        let mut buf = gst::Buffer::from_mut_slice(f32s_to_bytes(chunk));
        {
            let b = buf.get_mut().unwrap();
            b.set_pts(pts);
            b.set_duration(duration);
            b.set_offset(offset);
            b.set_offset_end(offset + frames);
            if discont {
                b.set_flags(gst::BufferFlags::DISCONT);
            }
        }
        assert_eq!(h.push(buf), Ok(gst::FlowSuccess::Ok));
        let out = h.pull().expect("no output buffer");
        assert_eq!(out.size(), chunk.len() * 4, "buffer size changed");
        assert_eq!(out.pts(), Some(pts), "PTS changed");
        assert_eq!(out.duration(), Some(duration), "duration changed");
        assert_eq!(out.offset(), offset, "offset changed");
        assert_eq!(out.offset_end(), offset + frames, "offset_end changed");
        let map = out.map_readable().unwrap();
        output.extend(bytes_to_f32s(&map));
        pts += duration;
        offset += frames;
    }

    let sinkpad = h.sinkpad().unwrap();
    let out_caps = sinkpad.current_caps().expect("no caps on output");
    let in_caps: gst::Caps = caps_str(rate, channels).parse().unwrap();
    assert!(
        out_caps.is_strictly_equal(&in_caps),
        "caps changed: {out_caps}"
    );
    output
}

/// Run `dsp` directly on `input` with the same buffer boundaries and actions as
/// [`run_element`]. `on_set` maps a `SetProperty` action to a parameter change.
pub fn run_reference<P: Processor>(
    mut dsp: P,
    rate: u32,
    channels: usize,
    input: &[f32],
    frames_per_buffer: usize,
    actions: &[(usize, Action)],
    mut on_set: impl FnMut(&mut P, &str, &glib::Value),
) -> Vec<f32> {
    // `dsp` must already carry its initial parameters; configure keeps them.
    dsp.configure(rate as f32, channels);
    let mut output = input.to_vec();
    for (idx, chunk) in output.chunks_mut(frames_per_buffer * channels).enumerate() {
        for (_, action) in actions.iter().filter(|(i, _)| *i == idx) {
            match action {
                Action::SetProperty(name, value) => on_set(&mut dsp, name, value),
                Action::Discont | Action::Flush => dsp.reset(),
            }
        }
        dsp.process(chunk);
    }
    output
}

/// Assert two sample vectors match within `tol` (absolute), reporting the first mismatch.
pub fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length differs");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite(), "non-finite output at {i}: {a}");
        assert!(
            (a - e).abs() <= tol,
            "sample {i}: element {a} vs library {e} (tol {tol})"
        );
    }
}

/// Assert bit-exact equality.
pub fn assert_bit_exact(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "length differs");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), e.to_bits(), "sample {i}: {a} vs {e}");
    }
}

/// Tolerance for element-vs-library comparisons. The element feeds the library
/// exactly the same samples with the same buffer boundaries (internally split
/// into whole-frame chunks), so the only possible difference is a processor
/// that is not perfectly invariant to how a buffer is split.
pub const TOL: f32 = 1e-6;

/// The two formats every harness test runs at: 48 kHz stereo and an odd
/// rate/channel count.
pub const FORMATS: [(u32, usize); 2] = [(48_000, 2), (22_050, 3)];
