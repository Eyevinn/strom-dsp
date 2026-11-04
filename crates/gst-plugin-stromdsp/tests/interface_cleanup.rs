//! Interface test of `stromdsp-cleanup`: every property's
//! name, GLib type, range, default and flags. Fails if the interface drifts.

use crate::iface_check;

use crate::iface_check::{check_element, p, Kind::*};

#[test]
fn cleanup_interface() {
    check_element(
        "stromdsp-cleanup",
        "StromDspCleanup",
        &[
            p("enabled", Bool(true)),
            p("dc-block", Bool(false)),
            p("highpass", Float(0.0, 1_000.0, 0.0)),
            p("highpass-order", UInt(1, 8, 2)),
            p(
                "hum",
                Enum(
                    "StromDspCleanupHum",
                    vec![(0, "off"), (50, "50"), (60, "60")],
                    0,
                ),
            ),
            p("hum-harmonics", UInt(1, 10, 5)),
            p("hum-q", Float(6.0, 100.0, 10.0)),
        ],
    );
}

/// `enabled=false` maps to BaseTransform passthrough.
#[test]
fn cleanup_enabled_maps_to_passthrough() {
    use gst::prelude::*;
    use gst_base::prelude::BaseTransformExt;
    iface_check::init();
    let e = gst::ElementFactory::make("stromdsp-cleanup")
        .build()
        .unwrap();
    let bt = e.downcast_ref::<gst_base::BaseTransform>().unwrap();
    assert!(!bt.is_passthrough());
    e.set_property("enabled", false);
    assert!(bt.is_passthrough());
    e.set_property("enabled", true);
    assert!(!bt.is_passthrough());
}
