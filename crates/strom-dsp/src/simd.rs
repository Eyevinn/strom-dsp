//! SIMD level selection shared by all processors.
//!
//! Processors detect the best instruction set once, in `configure`, and
//! dispatch their hot loops on it (via `fearless_simd`). Hosts and tests can
//! override the choice, e.g. force [`Level::fallback`] to compare a vector
//! path against the portable scalar path.

pub use fearless_simd::Level;

/// Best SIMD level supported by the running CPU (runtime detection:
/// AVX-512/AVX2/SSE on x86-64, NEON on aarch64, scalar fallback otherwise).
/// Cheap enough for `configure`, not meant for the per-buffer path.
pub fn detect() -> Level {
    Level::new()
}
