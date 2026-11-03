//! `stromdsp-meter`: STUB, to be implemented in phase 1b.

use gst::glib;

/// Register `stromdsp-meter` with `plugin` (stub: registers nothing yet).
pub fn register(_plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    Ok(())
}
