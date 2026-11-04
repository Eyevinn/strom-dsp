//! The whole equalizer suite (`tests/equalizer.rs`) once more, with every
//! equalizer forced to the scalar fallback SIMD level
//! (`Level::fallback()`, from `fearless_simd`'s `force_support_fallback`
//! feature, enabled for tests only). `tests/equalizer.rs` itself runs with the
//! level detected at `configure`; `tests/simd_equivalence.rs` compares the two
//! paths sample by sample.

// Deliberately a second copy of `equalizer.rs` (the first is `crate::equalizer`).
#[allow(clippy::duplicate_mod)]
#[path = "equalizer.rs"]
mod suite;
