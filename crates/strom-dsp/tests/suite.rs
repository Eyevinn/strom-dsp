//! The strom-dsp integration tests, as one test binary.
//!
//! Every `tests/*.rs` file is a module here rather than a binary of its own:
//! each binary links the library and instantiates its generics again, so one
//! binary builds and links far faster than fifty. Run one file's tests with a
//! filter on its module, e.g. `cargo test -p strom-dsp --test suite limiter::`.
//!
//! The `rtsan*` modules are only built with `--features rtsan` (see
//! docs/PERFORMANCE.md).

mod agc_support;
mod clip_metrics;
mod counting_alloc;
mod declipper_support;
mod measure_support;
mod programme_support;
mod simd_support;

mod agc;
mod band_correlation;
mod bass_clipper;
mod broadcast;
mod broadcast_expander_bass;
mod broadcast_latency;
mod cleanup;
mod clipper;
mod clipper_adaptive;
mod compressor;
mod compressor_1b;
mod compressor_auto_makeup;
mod compressor_thd;
mod declipper;
mod declipper_known_level;
mod declipper_quality;
mod deesser;
mod delay;
mod denormals;
mod ducker;
mod equalizer;
mod equalizer_fallback;
mod filter;
mod gate;
mod gate_chatter;
mod governor;
mod limiter;
mod limiter_thd;
mod loudness_detector;
mod meter;
mod multiband;
mod multiband_expander;
mod oversample;
mod rt_source;
mod rtsan;
mod rtsan_agc;
mod rtsan_broadcast;
mod rtsan_compressor_1b;
mod rtsan_declipper;
mod rtsan_deesser_cleanup;
mod rtsan_delay_stereo;
mod rtsan_ducker;
mod rtsan_filter;
mod rtsan_meter;
mod rtsan_multiband;
mod rtsan_multiband_expander_bass;
mod rtsan_sampling;
mod rtsan_sidechain;
mod rtsan_spike_guard;
mod rtsan_truepeak;
mod sidechain;
mod simd_equivalence;
mod simd_levels;
mod spike_guard;
mod stereo;
mod stft;
mod truepeak;

/// Cases per property test: `default`, or `PROPTEST_CASES` when set (the
/// extended workflow runs many more).
fn proptest_cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
