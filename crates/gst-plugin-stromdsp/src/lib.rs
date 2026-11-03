//! GStreamer plugin `stromdsp`: audio elements built on the `strom-dsp`
//! library.
//!
//! Elements: `stromdsp-gate`, `stromdsp-compressor`, `stromdsp-equalizer`,
//! `stromdsp-limiter`, `stromdsp-ducker`, `stromdsp-deesser`,
//! `stromdsp-delay`, `stromdsp-stereo`, `stromdsp-cleanup`,
//! `stromdsp-meter` and `stromdsp-broadcastprocessor`. All work on
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
    meter::register(plugin)?;
    broadcastprocessor::register(plugin)?;
    Ok(())
}

// `plugin_define!` also generates the public `plugin_register_static()`
// (re-exported at the crate root), which registers the plugin and its
// elements when the crate is linked statically as an `rlib`. Call it once after
// `gst::init()`.
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
