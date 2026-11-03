//! Interface test of `stromdsp-deesser`: every property's
//! name, GLib type, range, default and flags. Fails if the interface drifts.

mod iface_check;

use iface_check::{check_element, p, Kind::*};

#[test]
fn deesser_interface() {
    check_element(
        "stromdsp-deesser",
        "StromDspDeesser",
        &[
            p("enabled", Bool(true)),
            p("frequency", Float(1_000.0, 20_000.0, 6_000.0)),
            p("threshold", Float(-60.0, 0.0, -30.0)),
            p("max-reduction", Float(0.0, 40.0, 10.0)),
            p("attack", Float(0.01, 200.0, 1.0)),
            p("release", Float(1.0, 2000.0, 50.0)),
            p(
                "mode",
                Enum(
                    "StromDspDeesserMode",
                    vec![(0, "split-band"), (1, "wideband")],
                    0,
                ),
            ),
        ],
    );
}

/// `enabled=false` maps to BaseTransform passthrough.
#[test]
fn deesser_enabled_maps_to_passthrough() {
    use gst::prelude::*;
    use gst_base::prelude::BaseTransformExt;
    iface_check::init();
    let e = gst::ElementFactory::make("stromdsp-deesser")
        .build()
        .unwrap();
    let bt = e.downcast_ref::<gst_base::BaseTransform>().unwrap();
    assert!(!bt.is_passthrough());
    e.set_property("enabled", false);
    assert!(bt.is_passthrough());
    e.set_property("enabled", true);
    assert!(!bt.is_passthrough());
}
