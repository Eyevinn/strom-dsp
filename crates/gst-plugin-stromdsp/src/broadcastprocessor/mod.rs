//! `stromdsp-broadcastprocessor`: STUB, to be implemented in phase 2.

use gst::glib;

/// Register `stromdsp-broadcastprocessor` with `plugin` (stub: registers nothing yet).
pub fn register(_plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    Ok(())
}
