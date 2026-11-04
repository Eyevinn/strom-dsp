//! GStreamer plugin `stromdsp`: audio elements built on the `strom-dsp`
//! library.
//!
//! Elements: `stromdsp-gate`, `stromdsp-compressor`, `stromdsp-equalizer`,
//! `stromdsp-limiter`, `stromdsp-ducker`, `stromdsp-deesser`,
//! `stromdsp-delay`, `stromdsp-stereo`, `stromdsp-cleanup`,
//! `stromdsp-declipper`, `stromdsp-meter` and `stromdsp-broadcastprocessor`. All work on
//! interleaved F32LE. All except `stromdsp-ducker` are in-place
//! `BaseTransform`s; the ducker is an `Aggregator` with a `sidechain`
//! request pad for its key.
//!
//! Use it dynamically (`GST_PLUGIN_PATH` pointing at the built `cdylib`) or link
//! it statically and call [`plugin_register_static`] once after `gst::init()`.

use gst::glib;

pub mod broadcastprocessor;
pub mod cleanup;
mod common;
pub mod compressor;
pub mod declipper;
pub mod deesser;
pub mod delay;
pub mod ducker;
pub mod equalizer;
pub mod gate;
pub mod limiter;
pub mod meter;
pub mod stereo;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gate::register(plugin)?;
    compressor::register(plugin)?;
    equalizer::register(plugin)?;
    limiter::register(plugin)?;
    ducker::register(plugin)?;
    deesser::register(plugin)?;
    delay::register(plugin)?;
    stereo::register(plugin)?;
    cleanup::register(plugin)?;
    declipper::register(plugin)?;
    meter::register(plugin)?;
    broadcastprocessor::register(plugin)?;
    Ok(())
}

/// The plugin descriptor and the C entry points for dynamic loading
/// (`gst_plugin_stromdsp_get_desc`, `gst_plugin_stromdsp_register`). Kept in
/// a private module so that the crate-root [`plugin_register_static`] can wrap
/// the generated registration function.
mod registration {
    use super::plugin_init;

    gst::plugin_define!(
        stromdsp,
        env!("CARGO_PKG_DESCRIPTION"),
        plugin_init,
        concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
        "MIT/X11",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_REPOSITORY"),
        env!("BUILD_REL_DATE")
    );
}

/// Register the plugin and all its elements when the crate is linked
/// statically (as an `rlib`). Call it after `gst::init()`.
///
/// Thread-safe and idempotent: concurrent calls are serialised, and once a
/// registration has succeeded every later call returns `Ok(())` without
/// registering again. If a call fails (for example before `gst::init()`), a
/// later call retries.
pub fn plugin_register_static() -> Result<(), glib::BoolError> {
    static REGISTERED: std::sync::Mutex<bool> = std::sync::Mutex::new(false);
    // A panic while holding the lock cannot leave a half-registered state we
    // could repair, so a poisoned lock is treated like an unlocked one.
    let mut registered = REGISTERED.lock().unwrap_or_else(|e| e.into_inner());
    if !*registered {
        registration::plugin_register_static()?;
        *registered = true;
    }
    Ok(())
}
