//! `plugin_register_static()` called from many threads at once, as parallel
//! tests in a host application do, must not crash, must succeed on every
//! thread, and must leave every element creatable. Repeated calls afterwards
//! are no-ops.

use std::sync::{Arc, Barrier};
use std::thread;

const FACTORIES: [&str; 12] = [
    "stromdsp-gate",
    "stromdsp-compressor",
    "stromdsp-equalizer",
    "stromdsp-limiter",
    "stromdsp-ducker",
    "stromdsp-deesser",
    "stromdsp-delay",
    "stromdsp-stereo",
    "stromdsp-cleanup",
    "stromdsp-declipper",
    "stromdsp-meter",
    "stromdsp-broadcastprocessor",
];

#[test]
fn concurrent_static_registration_is_safe_and_idempotent() {
    gst::init().unwrap();
    const THREADS: usize = 32;
    let barrier = Arc::new(Barrier::new(THREADS));
    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                gststromdsp::plugin_register_static().expect("registration failed");
                // Every thread can create every element right away.
                for factory in FACTORIES {
                    gst::ElementFactory::make(factory)
                        .build()
                        .unwrap_or_else(|_| panic!("cannot make {factory}"));
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("registration thread panicked");
    }

    // Later calls are no-ops and still succeed.
    for _ in 0..10 {
        gststromdsp::plugin_register_static().unwrap();
    }
    let plugin = gst::Registry::get()
        .find_plugin("stromdsp")
        .expect("plugin not in registry");
    assert_eq!(plugin.plugin_name(), "stromdsp");
    let factories = gst::Registry::get().features_by_plugin("stromdsp");
    assert_eq!(factories.len(), FACTORIES.len());
}
