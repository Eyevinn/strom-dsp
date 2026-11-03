//! GStreamer plugin `stromdsp`: gate, compressor, equalizer and limiter elements
//! built on the `strom-dsp` library.
//!
//! STUB: elements to be written.

use gst::glib;

fn plugin_init(_plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    Ok(())
}

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
