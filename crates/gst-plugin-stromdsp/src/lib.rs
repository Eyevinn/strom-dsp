//! GStreamer plugin `stromdsp`: gate, compressor, equalizer and limiter elements
//! built on the `strom-dsp` library.
//!
//! Elements: `stromdsp-gate`, `stromdsp-compressor`, `stromdsp-equalizer`,
//! `stromdsp-limiter`. All are in-place `BaseTransform`s on interleaved F32LE.
//!
//! Use it dynamically (`GST_PLUGIN_PATH` pointing at the built `cdylib`) or link
//! it statically and call [`plugin_register_static`] once after `gst::init()`.

use gst::glib;

mod common;
pub mod compressor;
pub mod gate;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gate::register(plugin)?;
    compressor::register(plugin)?;
    Ok(())
}

// `plugin_define!` also generates the public `plugin_register_static()`
// (re-exported at the crate root), which registers the plugin and its four
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
