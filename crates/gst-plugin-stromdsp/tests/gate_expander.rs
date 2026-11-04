//! Harness tests of the gate's `mode` and `ratio` properties: expander mode
//! through the element matches the library and its static curve, and
//! `mode=gate` (with any `ratio`) is bit-identical to the classic gate.

use crate::util;

use crate::util::{Action, FORMATS, TOL};
use gst::glib;
use gst::prelude::*;
use gststromdsp::gate::StromDspGateMode;
use strom_dsp::gate::GateMode;
use strom_dsp::{Gate, GateParams};

fn params() -> GateParams {
    GateParams {
        open_threshold_db: -30.0,
        close_threshold_db: -40.0,
        attack_ms: 1.0,
        hold_ms: 10.0,
        release_ms: 20.0,
        reduction_db: -40.0,
    }
}

fn props(p: &GateParams) -> Vec<(&'static str, glib::Value)> {
    vec![
        ("open-threshold", p.open_threshold_db.to_value()),
        ("close-threshold", p.close_threshold_db.to_value()),
        ("attack", p.attack_ms.to_value()),
        ("hold", p.hold_ms.to_value()),
        ("release", p.release_ms.to_value()),
        ("reduction", p.reduction_db.to_value()),
    ]
}

/// Classic gate behaviour: the library gate with only `set_params` (no mode
/// or ratio calls), as the element drove it before `mode` and `ratio` existed.
fn classic_gate_reference(rate: u32, channels: usize, input: &[f32], fpb: usize) -> Vec<f32> {
    let mut dsp = Gate::new();
    dsp.set_params(&params());
    util::run_reference(dsp, rate, channels, input, fpb, &[], |_, _, _| {})
}

#[test]
fn mode_gate_is_bit_identical_to_classic_gate() {
    // The standard harness signal, and long segments in which the gate closes.
    let signals: [fn(u32, usize) -> Vec<f32>; 2] = [
        |r, c| util::test_signal(r, c, r as usize),
        |r, c| long_segments(r, c, 2),
    ];
    for ((rate, channels), signal) in FORMATS.into_iter().flat_map(|f| signals.map(|s| (f, s))) {
        let input = signal(rate, channels);
        let want = classic_gate_reference(rate, channels, &input, 1024);
        // Default mode, explicit mode=gate, and mode=gate with a non-default
        // ratio (ignored in gate mode).
        for extra in [
            vec![],
            vec![("mode", StromDspGateMode::Gate.to_value())],
            vec![
                ("mode", StromDspGateMode::Gate.to_value()),
                ("ratio", 7.5f32.to_value()),
            ],
        ] {
            let mut pr = props(&params());
            pr.extend(extra);
            let out = util::run_element("stromdsp-gate", &pr, rate, channels, &input, 1024, &[]);
            util::assert_bit_exact(&out, &want);
        }
        // The ratio changed while running in gate mode: still bit-identical.
        let actions = [
            (3, Action::SetProperty("ratio", 15.0f32.to_value())),
            (7, Action::SetProperty("ratio", 1.0f32.to_value())),
        ];
        let out = util::run_element(
            "stromdsp-gate",
            &props(&params()),
            rate,
            channels,
            &input,
            1024,
            &actions,
        );
        util::assert_bit_exact(&out, &want);
    }
}

#[test]
fn expander_mode_matches_library() {
    for (rate, channels) in FORMATS {
        let input = long_segments(rate, channels, 3);
        let mut pr = props(&params());
        pr.push(("mode", StromDspGateMode::Expander.to_value()));
        pr.push(("ratio", 1.5f32.to_value()));
        // Change the ratio, switch to gate and back while running (all after
        // the first 600 ms, which the gate-vs-expander check below uses).
        let buffers = |ms: usize| ms * rate as usize / 1000 / 1024;
        let actions = [
            (
                buffers(700),
                Action::SetProperty("ratio", 3.0f32.to_value()),
            ),
            (
                buffers(1300),
                Action::SetProperty("mode", StromDspGateMode::Gate.to_value()),
            ),
            (
                buffers(2000),
                Action::SetProperty("mode", StromDspGateMode::Expander.to_value()),
            ),
        ];
        let out = util::run_element("stromdsp-gate", &pr, rate, channels, &input, 1024, &actions);
        let mut dsp = Gate::new();
        dsp.set_params(&params());
        dsp.set_mode(GateMode::Expander);
        dsp.set_expander_ratio(1.5);
        let want = util::run_reference(
            dsp,
            rate,
            channels,
            &input,
            1024,
            &actions,
            |dsp, name, value| match name {
                "ratio" => dsp.set_expander_ratio(value.get().unwrap()),
                "mode" => dsp.set_mode(match value.get::<StromDspGateMode>().unwrap() {
                    StromDspGateMode::Gate => GateMode::Gate,
                    StromDspGateMode::Expander => GateMode::Expander,
                }),
                _ => unreachable!(),
            },
        );
        util::assert_close(&out, &want, TOL);
        // And it really differs from gate mode: at the end of the first quiet
        // segment (−54 dBFS, gate closed) the expander at ratio 1.5 attenuates
        // by 12 dB instead of 40 dB.
        let gate = classic_gate_reference(rate, channels, &input, 1024);
        let seg = SEGMENT_MS * rate as usize / 1000 * channels;
        let tail = 2 * seg - seg / 5..2 * seg;
        let peak = |v: &[f32]| v[tail.clone()].iter().fold(0f32, |m, x| m.max(x.abs()));
        let ratio_db = 20.0 * (peak(&out) / peak(&gate)).log10();
        assert!(
            (ratio_db - 28.0).abs() < 1.0,
            "expander vs gate: {ratio_db} dB"
        );
    }
}

const SEGMENT_MS: usize = 300;

/// Sines (a different frequency per channel) in 300 ms segments at
/// −10, −54, −6 and −46 dBFS, repeated: long enough for the gate to close.
fn long_segments(rate: u32, channels: usize, secs: usize) -> Vec<f32> {
    let seg = SEGMENT_MS * rate as usize / 1000;
    let levels = [0.316f32, 0.002, 0.5, 0.005];
    (0..secs * rate as usize)
        .flat_map(|n| {
            let a = levels[(n / seg) % levels.len()];
            let t = n as f32 / rate as f32;
            (0..channels)
                .map(move |c| a * (2.0 * std::f32::consts::PI * 300.0 * (c as f32 + 1.0) * t).sin())
        })
        .collect()
}

/// Static curve through the element: a steady sine 10 dB below the open
/// threshold at ratio 2 is attenuated by (2 − 1)·10 = 10 dB; at ratio 4 by
/// 30 dB; at ratio 20 the floor `reduction` (−40 dB) applies. In gate mode
/// it is attenuated by `reduction`.
#[test]
fn expander_static_curve_through_element() {
    let (rate, channels) = (48_000u32, 2usize);
    let amp = 10f32.powf(-40.0 / 20.0); // −40 dBFS peak, threshold −30
    let input: Vec<f32> = (0..rate as usize)
        .flat_map(|n| {
            let x = amp * (2.0 * std::f32::consts::PI * 1000.0 * n as f32 / rate as f32).sin();
            [x, x]
        })
        .collect();
    let peak_db = |out: &[f32]| {
        // Last 100 ms, after settling.
        let tail = &out[out.len() - 2 * rate as usize / 10..];
        20.0 * tail.iter().fold(0.0f32, |m, x| m.max(x.abs())).log10()
    };
    for (mode, ratio, want_gain_db) in [
        (StromDspGateMode::Expander, 2.0f32, -10.0f32),
        (StromDspGateMode::Expander, 4.0, -30.0),
        (StromDspGateMode::Expander, 20.0, -40.0),
        (StromDspGateMode::Expander, 1.0, 0.0),
        (StromDspGateMode::Gate, 2.0, -40.0),
    ] {
        let mut pr = props(&params());
        pr.push(("mode", mode.to_value()));
        pr.push(("ratio", ratio.to_value()));
        let out = util::run_element("stromdsp-gate", &pr, rate, channels, &input, 480, &[]);
        let got = peak_db(&out) - (-40.0);
        assert!(
            (got - want_gain_db).abs() < 0.5,
            "{mode:?} ratio {ratio}: gain {got} dB, want {want_gain_db}"
        );
    }
}
