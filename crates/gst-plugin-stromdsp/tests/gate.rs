//! Harness tests for `stromdsp-gate`.

use crate::util;

use crate::util::{Action, FORMATS, TOL};
use gst::glib;
use gst::prelude::*;
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

#[test]
fn gate_matches_library() {
    for (rate, channels) in FORMATS {
        // 1024 frames at 48k/2ch fits one internal chunk; 1500 frames at 3ch
        // (4500 samples) exercises the element's internal chunking.
        let fpb = if channels == 2 { 1024 } else { 1500 };
        let input = util::test_signal(rate, channels, rate as usize);
        let p = params();
        let actions = [
            (5, Action::Discont),
            (8, Action::SetProperty("reduction", (-20.0f32).to_value())),
            (12, Action::Flush),
        ];
        let out = util::run_element(
            "stromdsp-gate",
            &props(&p),
            rate,
            channels,
            &input,
            fpb,
            &actions,
        );
        let mut dsp = Gate::new();
        dsp.set_params(&p);
        let reference = util::run_reference(
            dsp,
            rate,
            channels,
            &input,
            fpb,
            &actions,
            |dsp, name, value| {
                assert_eq!(name, "reduction");
                let mut q = *dsp.params();
                q.reduction_db = value.get().unwrap();
                dsp.set_params(&q);
            },
        );
        util::assert_close(&out, &reference, TOL);
    }
}

#[test]
fn gate_disabled_is_bit_exact() {
    for (rate, channels) in FORMATS {
        let input = util::test_signal(rate, channels, rate as usize / 2);
        let mut pr = props(&params());
        pr.push(("enabled", false.to_value()));
        let out = util::run_element("stromdsp-gate", &pr, rate, channels, &input, 512, &[]);
        util::assert_bit_exact(&out, &input);
    }
}
