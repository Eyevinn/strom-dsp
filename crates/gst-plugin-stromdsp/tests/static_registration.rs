//! `plugin_register_static()` followed by `ElementFactory::make` for all four
//! elements, as Strom does when it links the plugin statically.

use gst::prelude::*;

#[test]
fn static_registration_makes_all_elements() {
    gst::init().unwrap();
    gststromdsp::plugin_register_static().expect("plugin_register_static failed");
    let plugin = gst::Registry::get()
        .find_plugin("stromdsp")
        .expect("plugin not in registry");
    assert_eq!(plugin.plugin_name(), "stromdsp");
    for (factory, type_name) in [
        ("stromdsp-gate", "StromDspGate"),
        ("stromdsp-compressor", "StromDspCompressor"),
        ("stromdsp-equalizer", "StromDspEqualizer"),
        ("stromdsp-limiter", "StromDspLimiter"),
    ] {
        let e = gst::ElementFactory::make(factory)
            .build()
            .unwrap_or_else(|_| panic!("cannot make {factory}"));
        assert_eq!(e.type_().name(), type_name);
        assert!(e.is::<gst_base::BaseTransform>());
        let f = e.factory().unwrap();
        assert_eq!(f.metadata("klass"), Some("Filter/Effect/Audio"));
        assert_eq!(f.plugin_name().as_deref(), Some("stromdsp"));
    }
}
