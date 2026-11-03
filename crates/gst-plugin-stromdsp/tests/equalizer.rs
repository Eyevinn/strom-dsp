//! Harness tests for `stromdsp-equalizer`.

mod util;

use gst::glib;
use gst::prelude::*;
use strom_dsp::{BandType, Equalizer, EqualizerParams};
use util::{Action, FORMATS, TOL};

fn params() -> EqualizerParams {
    let mut p = EqualizerParams {
        num_bands: 3,
        ..Default::default()
    };
    p.bands[0].band_type = BandType::Peaking;
    p.bands[0].frequency = 1000.0;
    p.bands[0].gain_db = 6.0;
    p.bands[0].q = 1.0;
    p.bands[1].band_type = BandType::HighPass;
    p.bands[1].frequency = 100.0;
    p.bands[2].band_type = BandType::HighShelf;
    p.bands[2].frequency = 8000.0;
    p.bands[2].gain_db = -6.0;
    // Band 3 is configured but inactive because num_bands = 3.
    p.bands[3].band_type = BandType::Peaking;
    p.bands[3].gain_db = 12.0;
    // Band 4 is disabled.
    p.bands[4].enabled = false;
    p.bands[4].band_type = BandType::LowPass;
    p
}

fn props(p: &EqualizerParams) -> Vec<(String, glib::Value)> {
    let mut v = vec![("num-bands".to_string(), (p.num_bands as u32).to_value())];
    for (n, b) in p.bands.iter().enumerate() {
        v.push((format!("band{n}-enabled"), b.enabled.to_value()));
        v.push((format!("band{n}-type"), (b.band_type as i32).to_value()));
        v.push((format!("band{n}-frequency"), b.frequency.to_value()));
        v.push((format!("band{n}-gain"), b.gain_db.to_value()));
        v.push((format!("band{n}-q"), b.q.to_value()));
    }
    v
}

fn as_refs(v: &[(String, glib::Value)]) -> Vec<(&str, glib::Value)> {
    v.iter().map(|(n, x)| (n.as_str(), x.clone())).collect()
}

#[test]
fn equalizer_matches_library() {
    for (rate, channels) in FORMATS {
        let fpb = if channels == 2 { 1024 } else { 1500 };
        let input = util::test_signal(rate, channels, rate as usize);
        let p = params();
        let actions = [
            (3, Action::SetProperty("band0-gain", (-9.0f32).to_value())),
            (5, Action::SetProperty("band1-type", 7i32.to_value())),
            (6, Action::Discont),
            (
                9,
                Action::SetProperty("band2-frequency", 3000.0f32.to_value()),
            ),
            (12, Action::Flush),
        ];
        let pr = props(&p);
        let out = util::run_element(
            "stromdsp-equalizer",
            &as_refs(&pr),
            rate,
            channels,
            &input,
            fpb,
            &actions,
        );
        let mut dsp = Equalizer::new();
        dsp.set_params(&p);
        let reference = util::run_reference(
            dsp,
            rate,
            channels,
            &input,
            fpb,
            &actions,
            |dsp, name, value| {
                let mut q = *dsp.params();
                match name {
                    "band0-gain" => q.bands[0].gain_db = value.get().unwrap(),
                    "band1-type" => q.bands[1].band_type = BandType::from_i32(value.get().unwrap()),
                    "band2-frequency" => q.bands[2].frequency = value.get().unwrap(),
                    _ => unreachable!(),
                }
                dsp.set_params(&q);
            },
        );
        util::assert_close(&out, &reference, TOL);
    }
}

#[test]
fn equalizer_disabled_is_bit_exact() {
    for (rate, channels) in FORMATS {
        let input = util::test_signal(rate, channels, rate as usize / 2);
        let mut pr = props(&params());
        pr.push(("enabled".to_string(), false.to_value()));
        let out = util::run_element(
            "stromdsp-equalizer",
            &as_refs(&pr),
            rate,
            channels,
            &input,
            512,
            &[],
        );
        util::assert_bit_exact(&out, &input);
    }
}
