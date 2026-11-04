//! Long programme runs of the broadcast chain: tens of seconds to minutes
//! of synthetic programme per case, over presets, input levels and
//! latency modes. They are measurements with hard bounds rather than unit
//! tests, so they are ignored in the default suite and run in the extended
//! workflow (release build):
//!
//! ```sh
//! cargo test --release -p strom-dsp-harness --test long_runs -- --ignored
//! ```
//!
//! Each run moved here from the `strom-dsp` suite. Per change, the harness
//! baseline (`tests/baseline.rs`, default suite) guards the same readings
//! on short, pre-levelled cases; each module names the baseline metrics
//! that stand in for it and what only the long run checks.

mod governor;
