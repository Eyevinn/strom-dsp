//! Harness tests for `stromdsp-meter`: message content against the library
//! meters fed the same input with the same buffer boundaries, message rate,
//! bit-exact passthrough, the `reset` signal, `true-peak=false`, silence and
//! the channel-position mapping.

mod util;

use gst::glib;
use gst::prelude::*;
use gst_check::Harness;
use gststromdsp::meter::{MESSAGE_NAME, SILENCE_DB};
use strom_dsp::meter::{
    BandCorrelation, ChannelLayout, CorrelationMeter, LoudnessConfig, LoudnessMeter, MonoDetector,
    MonoDetectorParams, StereoState,
};

/// One received meter message and the index of the buffer it followed.
struct Received {
    after_buffer: usize,
    /// PTS + duration of that buffer.
    end: gst::ClockTime,
    s: gst::Structure,
}

/// Push `input` through `stromdsp-meter` with `caps`, in buffers of
/// `fpb` frames, timestamped from 0. `before(idx, element)` runs before buffer
/// `idx` is pushed. Returns the output samples and the messages.
fn run_meter(
    props: &[(&str, glib::Value)],
    caps: &str,
    rate: u32,
    channels: usize,
    input: &[f32],
    fpb: usize,
    mut before: impl FnMut(usize, &gst::Element),
) -> (Vec<f32>, Vec<Received>) {
    util::init();
    let mut h = Harness::new("stromdsp-meter");
    let element = h.element().unwrap();
    for (name, value) in props {
        element.set_property_from_value(name, value);
    }
    let bus = gst::Bus::new();
    element.set_bus(Some(&bus));
    h.set_src_caps_str(caps);
    h.play();

    let mut out = Vec::with_capacity(input.len());
    let mut msgs = Vec::new();
    let mut pts = gst::ClockTime::ZERO;
    for (idx, chunk) in input.chunks(fpb * channels).enumerate() {
        before(idx, &element);
        let frames = (chunk.len() / channels) as u64;
        let duration = gst::ClockTime::SECOND
            .mul_div_floor(frames, rate as u64)
            .unwrap();
        let bytes: Vec<u8> = chunk.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut buf = gst::Buffer::from_mut_slice(bytes);
        {
            let b = buf.get_mut().unwrap();
            b.set_pts(pts);
            b.set_duration(duration);
        }
        assert_eq!(h.push(buf), Ok(gst::FlowSuccess::Ok));
        let o = h.pull().expect("no output buffer");
        assert_eq!(o.pts(), Some(pts));
        let map = o.map_readable().unwrap();
        out.extend(
            map.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        );
        while let Some(msg) = bus.pop() {
            if let gst::MessageView::Element(e) = msg.view() {
                let s = e.structure().unwrap().to_owned();
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
    element.set_bus(None);
    (out, msgs)
}

fn f(s: &gst::Structure, name: &str) -> f64 {
    let v = s
        .get::<f64>(name)
        .unwrap_or_else(|e| panic!("field {name}: {e}"));
    assert!(v.is_finite(), "{name} = {v}");
    v
}

fn arr(s: &gst::Structure, name: &str) -> Vec<f64> {
    let a = s
        .get::<gst::Array>(name)
        .unwrap_or_else(|e| panic!("field {name}: {e}"));
    a.iter()
        .map(|v| {
            let x = v.get::<f64>().unwrap();
            assert!(x.is_finite(), "{name}: {x}");
            x
        })
        .collect()
}

fn db(x: f64) -> f64 {
    if x > 0.0 {
        (20.0 * x.log10()).max(SILENCE_DB)
    } else {
        SILENCE_DB
    }
}

fn lufs(x: f64) -> f64 {
    if x.is_finite() {
        x.max(SILENCE_DB)
    } else {
        SILENCE_DB
    }
}

/// Expected values of one message, from the library meters.
#[derive(Debug)]
struct Expected {
    after_buffer: usize,
    duration_frames: u64,
    momentary: f64,
    short_term: f64,
    integrated: f64,
    lra: f64,
    sample_peak: Vec<f64>,
    true_peak: Vec<f64>,
    programme_sample_peak: f64,
    programme_true_peak: f64,
    rms: Vec<f64>,
    /// correlation, low, mid, high, state
    stereo: Option<([f64; 4], &'static str)>,
}

/// Library reference with the element's documented scheduling rule.
fn reference(
    rate: u32,
    channels: usize,
    input: &[f32],
    fpb: usize,
    interval_ms: u32,
) -> Vec<Expected> {
    let fs = rate as f32;
    let mut loud = LoudnessMeter::new(&LoudnessConfig {
        sample_rate: fs,
        channels,
        layout: ChannelLayout::Unknown,
        true_peak: true,
    })
    .unwrap();
    let mut corr = CorrelationMeter::new(fs, 300.0);
    let mut bands = BandCorrelation::new(fs, 300.0);
    let mut mono = MonoDetector::new(fs, &MonoDetectorParams::default());
    let interval = (interval_ms as f64 * 0.001 * rate as f64).round() as u64;
    let (mut phase, mut since) = (0u64, 0u64);
    let mut sum_sq = vec![0.0f64; channels];
    let mut out = Vec::new();
    for (idx, chunk) in input.chunks(fpb * channels).enumerate() {
        loud.process(chunk);
        for fr in chunk.chunks_exact(channels) {
            for (a, &x) in sum_sq.iter_mut().zip(fr) {
                *a += x as f64 * x as f64;
            }
        }
        if channels == 2 {
            corr.process_interleaved(chunk, 2);
            bands.process_interleaved(chunk, 2);
            mono.process_interleaved(chunk, 2);
        }
        let frames = (chunk.len() / channels) as u64;
        phase += frames;
        since += frames;
        if phase >= interval {
            phase -= interval;
            if phase >= interval {
                phase = 0;
            }
            let state = match mono.state() {
                StereoState::Mono => "mono",
                StereoState::Stereo => "stereo",
                StereoState::Inverted => "inverted",
            };
            let c = bands.correlations();
            out.push(Expected {
                after_buffer: idx,
                duration_frames: since,
                momentary: lufs(loud.momentary()),
                short_term: lufs(loud.short_term()),
                integrated: lufs(loud.integrated()),
                lra: loud.loudness_range(),
                sample_peak: (0..channels)
                    .map(|c| db(loud.interval_sample_peak(c)))
                    .collect(),
                true_peak: (0..channels)
                    .map(|c| db(loud.interval_true_peak(c)))
                    .collect(),
                programme_sample_peak: db((0..channels)
                    .map(|c| loud.sample_peak(c))
                    .fold(0.0, f64::max)),
                programme_true_peak: db((0..channels)
                    .map(|c| loud.true_peak(c))
                    .fold(0.0, f64::max)),
                rms: sum_sq
                    .iter()
                    .map(|&e| db((e / since as f64).sqrt()))
                    .collect(),
                stereo: (channels == 2).then_some(([corr.correlation(), c[0], c[1], c[2]], state)),
            });
            loud.clear_interval_peaks();
            sum_sq.fill(0.0);
            since = 0;
        }
    }
    out
}

fn assert_vec(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-9,
            "{what}[{i}]: element {g} vs library {w}"
        );
    }
}

/// Stereo programme with a common part (correlated) and a channel-specific
/// part, so every field has a non-trivial value.
fn programme(rate: u32, channels: usize, secs: f64) -> Vec<f32> {
    let base = util::test_signal(rate, channels, (rate as f64 * secs) as usize);
    let common = util::test_signal(rate, 1, (rate as f64 * secs) as usize);
    base.chunks_exact(channels)
        .zip(&common)
        .flat_map(|(fr, &c)| {
            fr.iter()
                .map(move |&x| 0.4 * x + 0.5 * c)
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn message_content_matches_library() {
    for (rate, channels, fpb) in [
        (48_000u32, 2usize, 1024usize),
        (44_100, 3, 441),
        (48_000, 1, 960),
    ] {
        let input = programme(rate, channels, 4.0);
        let (_, msgs) = run_meter(
            &[],
            &util::caps_str(rate, channels),
            rate,
            channels,
            &input,
            fpb,
            |_, _| {},
        );
        let want = reference(rate, channels, &input, fpb, 100);
        assert_eq!(msgs.len(), want.len(), "{rate}/{channels}: message count");
        assert!(msgs.len() >= 35);
        for (m, w) in msgs.iter().zip(&want) {
            let s = &m.s;
            let ctx = format!("{rate}/{channels} after buffer {}", w.after_buffer);
            assert_eq!(m.after_buffer, w.after_buffer, "{ctx}");
            assert_eq!(s.get::<i32>("channels").unwrap(), channels as i32);
            let dur = s.get::<gst::ClockTime>("duration").unwrap();
            assert_eq!(
                dur,
                gst::ClockTime::SECOND
                    .mul_div_floor(w.duration_frames, rate as u64)
                    .unwrap(),
                "{ctx}"
            );
            // End (PTS + duration) of the buffer the message followed; the
            // segment starts at 0, so stream and running time equal it.
            assert_eq!(
                s.get::<gst::ClockTime>("stream-time").unwrap(),
                m.end,
                "{ctx}"
            );
            assert_eq!(
                s.get::<gst::ClockTime>("running-time").unwrap(),
                m.end,
                "{ctx}"
            );
            assert_vec(&ctx, &[f(s, "momentary")], &[w.momentary]);
            assert_vec(&ctx, &[f(s, "short-term")], &[w.short_term]);
            assert_vec(&ctx, &[f(s, "integrated")], &[w.integrated]);
            assert_vec(&ctx, &[f(s, "loudness-range")], &[w.lra]);
            assert_vec(&ctx, &arr(s, "sample-peak"), &w.sample_peak);
            assert_vec(&ctx, &arr(s, "true-peak"), &w.true_peak);
            let spm = w.sample_peak.iter().copied().fold(SILENCE_DB, f64::max);
            let tpm = w.true_peak.iter().copied().fold(SILENCE_DB, f64::max);
            assert_vec(&ctx, &[f(s, "sample-peak-max")], &[spm]);
            assert_vec(&ctx, &[f(s, "true-peak-max")], &[tpm]);
            assert_vec(
                &ctx,
                &[f(s, "programme-sample-peak")],
                &[w.programme_sample_peak],
            );
            assert_vec(
                &ctx,
                &[f(s, "programme-true-peak")],
                &[w.programme_true_peak],
            );
            assert_vec(&ctx, &arr(s, "rms"), &w.rms);
            match w.stereo {
                Some((c, state)) => {
                    let got = [
                        f(s, "correlation"),
                        f(s, "correlation-low"),
                        f(s, "correlation-mid"),
                        f(s, "correlation-high"),
                    ];
                    assert_vec(&ctx, &got, &c);
                    assert_eq!(s.get::<&str>("stereo-state").unwrap(), state, "{ctx}");
                }
                None => {
                    for name in [
                        "correlation",
                        "correlation-low",
                        "correlation-mid",
                        "correlation-high",
                        "stereo-state",
                    ] {
                        assert!(!s.has_field(name), "{ctx}: {name} without stereo");
                    }
                }
            }
        }
        // The values are plausible, not just equal: after 4 s the integrated
        // loudness is measured and the programme is mostly correlated.
        let last = &msgs.last().unwrap().s;
        assert!(f(last, "integrated") > -40.0 && f(last, "integrated") < 0.0);
        if channels == 2 {
            assert!(f(last, "correlation") > 0.3);
        }
    }
}

#[test]
fn message_rate_follows_interval_in_stream_time() {
    let (rate, channels) = (48_000u32, 2usize);
    let input = programme(rate, channels, 2.0);
    // 1024-frame buffers, 100 ms = 4800 frames: ⌊96000/4800⌋ = 20 messages.
    let (_, msgs) = run_meter(
        &[],
        &util::caps_str(rate, channels),
        rate,
        channels,
        &input,
        1024,
        |_, _| {},
    );
    assert_eq!(msgs.len(), 20);
    // 10 ms buffers and a 50 ms interval: exactly every 5th buffer, and the
    // stream time advances by exactly 50 ms per message.
    let (_, msgs) = run_meter(
        &[("interval", 50u32.to_value())],
        &util::caps_str(rate, channels),
        rate,
        channels,
        &input,
        480,
        |_, _| {},
    );
    assert_eq!(msgs.len(), 40);
    for (i, m) in msgs.iter().enumerate() {
        assert_eq!(m.after_buffer, 5 * i + 4);
        let t = m.s.get::<gst::ClockTime>("stream-time").unwrap();
        assert_eq!(t, gst::ClockTime::from_mseconds(50 * (i as u64 + 1)));
        assert_eq!(
            m.s.get::<gst::ClockTime>("duration").unwrap(),
            gst::ClockTime::from_mseconds(50)
        );
    }
    // Interval changed while playing: 1 s of 100 ms (10 messages), then 1 s
    // of 500 ms (2 messages).
    let (_, msgs) = run_meter(
        &[],
        &util::caps_str(rate, channels),
        rate,
        channels,
        &input,
        480,
        |idx, e| {
            if idx == 100 {
                e.set_property("interval", 500u32);
            }
        },
    );
    assert_eq!(msgs.len(), 12);
    // Buffers longer than an interval: one message per buffer.
    let (_, msgs) = run_meter(
        &[("interval", 10u32.to_value())],
        &util::caps_str(rate, channels),
        rate,
        channels,
        &input,
        4800,
        |_, _| {},
    );
    assert_eq!(msgs.len(), 20);
}

#[test]
fn samples_are_unchanged_bit_exact() {
    for (rate, channels) in [(48_000u32, 2usize), (22_050, 3), (48_000, 8)] {
        let mut input = util::test_signal(rate, channels, rate as usize / 2);
        // Specials pass untouched too.
        input[1] = f32::MIN_POSITIVE / 2.0;
        input[2] = -0.0;
        input[3] = 3.5;
        let out = util::run_element(
            "stromdsp-meter",
            &[("interval", 10u32.to_value())],
            rate,
            channels,
            &input,
            333,
            &[],
        );
        util::assert_bit_exact(&out, &input);
    }
}

fn sine(rate: u32, channels: usize, secs: f64, amp: f32) -> Vec<f32> {
    (0..(rate as f64 * secs) as usize)
        .flat_map(|n| {
            let x = amp * (2.0 * std::f32::consts::PI * 997.0 * n as f32 / rate as f32).sin();
            std::iter::repeat_n(x, channels)
        })
        .collect()
}

#[test]
fn reset_signal_restarts_integrated_and_lra() {
    let (rate, ch) = (48_000u32, 2usize);
    // 4 s at −6 dBFS, then 4 s at −26 dBFS.
    let mut input = sine(rate, ch, 4.0, 0.5);
    input.extend(sine(rate, ch, 4.0, 0.05));
    let caps = util::caps_str(rate, ch);
    let reset_at = 4 * rate as usize / 480;
    let run = |reset: bool| {
        run_meter(&[], &caps, rate, ch, &input, 480, |idx, e| {
            if reset && idx == reset_at {
                e.emit_by_name::<()>("reset", &[]);
            }
        })
        .1
    };
    let plain = run(false);
    let with_reset = run(true);
    let at_switch = &with_reset[39].s;
    let end_plain = &plain.last().unwrap().s;
    let end_reset = &with_reset.last().unwrap().s;
    // A stereo 997 Hz sine at amplitude A reads 20·log10(A) + 0.0 LUFS
    // (−3.01 dB for RMS, +3.01 dB for two channels, K-weighting ≈ 0 dB at 1 kHz).
    assert!((f(at_switch, "integrated") - (-6.02)).abs() < 0.2);
    assert!(
        (f(end_reset, "integrated") - (-26.02)).abs() < 0.2,
        "{end_reset}"
    );
    // Without the reset the quiet part is gated out relative to the loud one,
    // and the range spans both levels.
    assert!(f(end_plain, "integrated") > -10.0, "{end_plain}");
    assert!(f(end_plain, "loudness-range") > 10.0);
    assert!(f(end_reset, "loudness-range") < 1.0);
    assert!((f(end_reset, "programme-sample-peak") - (-26.02)).abs() < 0.1);
    assert!((f(end_plain, "programme-sample-peak") - (-6.02)).abs() < 0.1);
}

#[test]
fn true_peak_false_omits_true_peak_fields() {
    let (rate, ch) = (48_000u32, 2usize);
    let input = programme(rate, ch, 1.0);
    let caps = util::caps_str(rate, ch);
    let (_, off) = run_meter(
        &[("true-peak", false.to_value())],
        &caps,
        rate,
        ch,
        &input,
        480,
        |_, _| {},
    );
    let (_, on) = run_meter(&[], &caps, rate, ch, &input, 480, |_, _| {});
    assert_eq!(off.len(), on.len());
    for (a, b) in off.iter().zip(&on) {
        for name in ["true-peak", "true-peak-max", "programme-true-peak"] {
            assert!(!a.s.has_field(name), "{name} present with true-peak=false");
            assert!(b.s.has_field(name), "{name} missing with true-peak=true");
        }
        // Everything else is identical.
        assert_eq!(arr(&a.s, "sample-peak"), arr(&b.s, "sample-peak"));
        assert_eq!(f(&a.s, "momentary"), f(&b.s, "momentary"));
        assert!(f(&b.s, "true-peak-max") >= f(&b.s, "sample-peak-max") - 1e-9);
    }
}

#[test]
fn silence_is_finite_sentinel() {
    let (rate, ch) = (48_000u32, 2usize);
    let input = vec![0.0f32; rate as usize * ch];
    let (_, msgs) = run_meter(
        &[],
        &util::caps_str(rate, ch),
        rate,
        ch,
        &input,
        480,
        |_, _| {},
    );
    assert_eq!(msgs.len(), 10);
    for m in &msgs {
        let s = &m.s;
        for name in [
            "momentary",
            "short-term",
            "integrated",
            "sample-peak-max",
            "true-peak-max",
            "programme-sample-peak",
            "programme-true-peak",
        ] {
            assert_eq!(f(s, name), SILENCE_DB, "{name}");
        }
        for name in ["sample-peak", "true-peak", "rms"] {
            assert_eq!(arr(s, name), [SILENCE_DB; 2], "{name}");
        }
        assert_eq!(f(s, "loudness-range"), 0.0);
        assert_eq!(f(s, "correlation"), 0.0);
        assert_eq!(f(s, "correlation-low"), 0.0);
        assert_eq!(s.get::<&str>("stereo-state").unwrap(), "stereo");
    }
}

#[test]
fn stereo_state_and_band_correlation_through_the_element() {
    let (rate, ch) = (48_000u32, 2usize);
    // Inverted mono for 2 s.
    let input: Vec<f32> = util::test_signal(rate, 1, 2 * rate as usize)
        .into_iter()
        .flat_map(|x| [0.5 * x, -0.5 * x])
        .collect();
    let (_, msgs) = run_meter(
        &[],
        &util::caps_str(rate, ch),
        rate,
        ch,
        &input,
        480,
        |_, _| {},
    );
    let last = &msgs.last().unwrap().s;
    assert_eq!(last.get::<&str>("stereo-state").unwrap(), "inverted");
    assert!(f(last, "correlation") < -0.99);
    assert!(f(last, "correlation-low") < -0.99);
    assert!(f(last, "correlation-mid") < -0.99);
}

/// 5.1 with a channel mask: LFE excluded, rear pair weighted 1.41. Without a
/// mask every channel counts 1.0.
#[test]
fn channel_mask_maps_to_bs1770_weights() {
    let rate = 48_000u32;
    let ch = 6usize;
    let with_mask = format!("{},channel-mask=(bitmask)0x3f", util::caps_str(rate, ch));
    let integrated = |caps: &str, active: usize| {
        let input: Vec<f32> = sine(rate, 1, 3.0, 0.1)
            .into_iter()
            .flat_map(|x| (0..ch).map(move |c| if c == active { x } else { 0.0 }))
            .collect();
        let (_, msgs) = run_meter(&[], caps, rate, ch, &input, 960, |_, _| {});
        f(&msgs.last().unwrap().s, "integrated")
    };
    let front = integrated(&with_mask, 0);
    assert!((front - (-23.0)).abs() < 0.3, "front {front}");
    assert_eq!(integrated(&with_mask, 3), SILENCE_DB, "LFE excluded");
    let rear = integrated(&with_mask, 4);
    assert!(
        (rear - front - 10.0 * 1.41f64.log10()).abs() < 0.05,
        "rear {rear}"
    );
    // No mask: Unknown layout, LFE position counted with weight 1.0.
    let plain = util::caps_str(rate, ch);
    assert!((integrated(&plain, 3) - front).abs() < 0.05);
    assert!((integrated(&plain, 4) - front).abs() < 0.05);
}
