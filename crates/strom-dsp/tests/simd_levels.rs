//! Every SIMD level gives bit-identical output: each processor run with
//! `set_simd_level(Level::fallback())` must match the same processor on the
//! detected level (NEON on aarch64, SSE4.2/AVX2/AVX-512 on x86-64), sample
//! for sample, including parameter changes, glides, lookahead cross-fades and
//! odd buffer sizes.
//!
//! `Level::fallback()` needs fearless_simd's `force_support_fallback`
//! feature, which the dev-dependency enables for tests only.

use strom_dsp::simd::{detect, Level};
use strom_dsp::{
    Compressor, CompressorParams, Gate, GateParams, Limiter, LimiterParams, Processor,
};

const CHANNELS: [usize; 5] = [1, 2, 3, 8, 64];
const FS: f32 = 48_000.0;

struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// Noise with a level that jumps between −60 dBFS and +12 dBFS every
/// 1000 frames, plus exact silence and a few full-scale impulses, so every
/// stage (detector, knee, attack/release, limiting) is exercised.
fn signal(ch: usize, frames: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed | 1);
    let levels = [0.001f32, 0.5, 4.0, 0.0, 0.05, 1.0, 0.2];
    let mut out = Vec::with_capacity(frames * ch);
    for f in 0..frames {
        let level = levels[(f / 1000) % levels.len()];
        for c in 0..ch {
            let mut x = level * rng.next_f32();
            if f % 2777 == 5 && c == ch - 1 {
                x = 1.0;
            }
            out.push(x);
        }
    }
    out
}

/// Buffer sizes cycling through odd and block-crossing lengths.
const BUFFERS: [usize; 7] = [480, 1, 63, 64, 65, 1024, 7];

/// Run `p` over `input` in varying buffer sizes, calling `change(p, k)` before
/// buffer `k`.
fn run<P: Processor>(
    p: &mut P,
    input: &[f32],
    ch: usize,
    mut change: impl FnMut(&mut P, usize),
) -> Vec<f32> {
    let mut out = input.to_vec();
    let mut pos = 0;
    let mut k = 0;
    while pos < out.len() {
        let len = (BUFFERS[k % BUFFERS.len()] * ch).min(out.len() - pos);
        change(p, k);
        p.process(&mut out[pos..pos + len]);
        pos += len;
        k += 1;
    }
    out
}

fn assert_bits_equal(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: sample {i}: {x} vs {y}");
    }
}

fn levels() -> (Level, Level) {
    (Level::fallback(), detect())
}

#[test]
fn set_simd_level_is_kept_across_configure() {
    eprintln!("detected SIMD level: {:?}", detect());
    let mut g = Gate::new();
    g.set_simd_level(Level::fallback());
    g.configure(FS, 2);
    assert!(g.simd_level().is_fallback());
    let mut c = Compressor::new();
    c.configure(FS, 2);
    assert_eq!(c.simd_level().is_fallback(), detect().is_fallback());
}

#[test]
fn gate_levels_are_bit_identical() {
    let (lo, hi) = levels();
    for ch in CHANNELS {
        let input = signal(ch, 12_000, 1 + ch as u64);
        let change = |g: &mut Gate, k: usize| {
            if k % 5 == 3 {
                g.set_params(&GateParams {
                    open_threshold_db: -20.0 - (k % 4) as f32 * 10.0,
                    close_threshold_db: -45.0,
                    attack_ms: 0.5 + k as f32,
                    hold_ms: (k % 3) as f32 * 5.0,
                    release_ms: 20.0,
                    reduction_db: -40.0 - k as f32,
                });
            }
        };
        let mut a = Gate::new();
        a.set_simd_level(lo);
        a.configure(FS, ch);
        let mut b = Gate::new();
        b.set_simd_level(hi);
        b.configure(FS, ch);
        let ya = run(&mut a, &input, ch, change);
        let yb = run(&mut b, &input, ch, change);
        assert_bits_equal(&ya, &yb, &format!("gate ch={ch}"));
    }
}

#[test]
fn compressor_levels_are_bit_identical() {
    let (lo, hi) = levels();
    for ch in CHANNELS {
        let input = signal(ch, 12_000, 7 + ch as u64);
        // Changes start glides (threshold, ratio, knee, makeup) mid-stream.
        let change = |c: &mut Compressor, k: usize| {
            if k % 4 == 1 {
                c.set_params(&CompressorParams {
                    threshold: [0.1f32, 0.5, 0.02][k % 3],
                    ratio: 1.0 + (k % 7) as f32 * 3.0,
                    attack_ms: 0.01 + (k % 5) as f32,
                    release_ms: 5.0 + (k % 3) as f32 * 40.0,
                    makeup_gain: 1.0 + (k % 2) as f32,
                    knee: [0.0625f32, 1.0, 0.0][k % 3],
                });
            }
        };
        let mut a = Compressor::new();
        a.set_simd_level(lo);
        a.configure(FS, ch);
        let mut b = Compressor::new();
        b.set_simd_level(hi);
        b.configure(FS, ch);
        let ya = run(&mut a, &input, ch, change);
        let yb = run(&mut b, &input, ch, change);
        assert_bits_equal(&ya, &yb, &format!("compressor ch={ch}"));
        assert_eq!(
            a.gain_reduction_db().to_bits(),
            b.gain_reduction_db().to_bits()
        );
    }
}

#[test]
fn limiter_levels_are_bit_identical() {
    let (lo, hi) = levels();
    for ch in CHANNELS {
        let input = signal(ch, 12_000, 13 + ch as u64);
        // Threshold, attack (rebuild), lookahead (cross-fade) and enable changes.
        let change = |l: &mut Limiter, k: usize| {
            if k % 3 == 2 {
                l.set_params(&LimiterParams {
                    enabled: k % 11 != 8,
                    threshold_db: -((k % 6) as f32) * 2.0,
                    attack_ms: 0.5 + (k % 4) as f32,
                    release_ms: 10.0 + (k % 5) as f32 * 20.0,
                    lookahead_ms: [5.0f32, 2.0, 0.0, 8.0][(k / 3) % 4],
                });
            }
        };
        let mut a = Limiter::new();
        a.set_simd_level(lo);
        a.configure(FS, ch);
        let mut b = Limiter::new();
        b.set_simd_level(hi);
        b.configure(FS, ch);
        let ya = run(&mut a, &input, ch, change);
        let yb = run(&mut b, &input, ch, change);
        assert_bits_equal(&ya, &yb, &format!("limiter ch={ch}"));
        assert_eq!(a.safety_clip_count(), 0);
        assert_eq!(b.safety_clip_count(), 0);
    }
}

#[test]
fn compressor_detector_modes_are_bit_identical() {
    use strom_dsp::compressor::CompressorDetector;
    let (lo, hi) = levels();
    for ch in CHANNELS {
        let input = signal(ch, 12_000, 17 + ch as u64);
        // Detector switches (seeded while running), rms-window and release
        // (hold length) changes, and parameter glides.
        let change = |c: &mut Compressor, k: usize| {
            if k % 3 == 1 {
                c.set_detector(if k % 2 == 0 {
                    CompressorDetector::Peak
                } else {
                    CompressorDetector::Rms
                });
                c.set_rms_window(1.0 + (k % 9) as f32 * 7.0);
                c.set_auto_makeup(k % 5 < 3);
                c.set_params(&CompressorParams {
                    threshold: [0.1f32, 0.5, 0.02][k % 3],
                    ratio: 1.0 + (k % 7) as f32 * 3.0,
                    attack_ms: 0.01 + (k % 5) as f32,
                    release_ms: 1.0 + (k % 4) as f32 * 15.0,
                    makeup_gain: 1.0 + (k % 2) as f32,
                    knee: [0.0625f32, 1.0, 0.0][k % 3],
                });
            }
        };
        let mut a = Compressor::new();
        a.set_simd_level(lo);
        a.set_detector(CompressorDetector::Rms);
        a.configure(FS, ch);
        let mut b = Compressor::new();
        b.set_simd_level(hi);
        b.set_detector(CompressorDetector::Rms);
        b.configure(FS, ch);
        let ya = run(&mut a, &input, ch, change);
        let yb = run(&mut b, &input, ch, change);
        assert_bits_equal(&ya, &yb, &format!("compressor detector modes ch={ch}"));
    }
}
