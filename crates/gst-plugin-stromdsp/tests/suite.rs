//! The plugin integration tests, as one test binary.
//!
//! Every `tests/*.rs` file is a module here rather than a binary of its own,
//! so the plugin and GStreamer are linked once instead of twenty times. Run
//! one file's tests with a filter on its module, e.g.
//! `cargo test -p gst-plugin-stromdsp --test suite limiter::`.
//!
//! The `static_registration*` tests stay separate binaries: they check the
//! first registration in a fresh process.

mod iface_check;
mod util;

mod broadcastprocessor;
mod cleanup;
mod compressor;
mod declipper;
mod deesser;
mod delay;
mod ducker;
mod equalizer;
mod gate;
mod gate_expander;
mod interface;
mod interface_broadcastprocessor;
mod interface_cleanup;
mod interface_compressor_1b;
mod interface_deesser;
mod interface_delay;
mod interface_ducker;
mod interface_gate_1b;
mod interface_meter;
mod interface_stereo;
mod knobs;
mod limiter;
mod meter;
mod stereo;
