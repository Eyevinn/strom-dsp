//! Instruction-count regression guard for the vector kernels (gungraun /
//! Valgrind Callgrind), run by the weekly extended workflow.
//!
//! Each case counts the instructions for one 480-frame (10 ms), 48 kHz buffer
//! at a fixed SIMD level and fails when the count exceeds a **hard limit**,
//! 20 % over the value measured when the limit was set (see
//! `docs/PERFORMANCE.md`, "Instruction counts", for how to update them).
//!
//! What it guards against: code outlined from `fearless_simd::vectorize` (a
//! closure the compiler does not inline) is compiled without the level's
//! target features, so on x86-64 every vector operation in it becomes a
//! function call and the kernel's state spills to memory. aarch64 does not
//! show this (NEON is part of its baseline), and the output stays correct, so
//! only a cost measurement catches it. Both known cases broke the limits:
//! the decimation closure in `oversample::down_kernel` (#8) and the
//! `for_each` closure in `biquad::kernel::flush_all` (#11).
//!
//! **Why AVX2 is forced:** the count must not depend on the CI runner.
//! Valgrind does not implement AVX-512 and hides it from CPUID, so under
//! Callgrind the detected level is at most AVX2 anyway; forcing AVX2 makes
//! that explicit for every processor that takes a level, and every x86-64
//! runner GitHub offers supports it. The broadcast chain has no level
//! setter: its setup asserts that the detected level is AVX2.
//!
//! On other architectures the cases run at the detected level and are
//! report-only (the limits are calibrated for x86-64).
//!
//! ```text
//! cargo install --locked gungraun-runner --version 0.20.0
//! cargo bench -p strom-dsp --features gungraun --bench simd_gungraun
//! ```

use std::f64::consts::PI;
use std::hint::black_box;

use gungraun::{
    library_benchmark, library_benchmark_group, main, Callgrind, EventKind, LibraryBenchmarkConfig,
};
use strom_dsp::broadcast::{BroadcastParams, BroadcastProcessor};
use strom_dsp::oversample::{OversampleFactor, Oversampler};
use strom_dsp::presets::BroadcastPreset;
use strom_dsp::simd::{detect, Level};
use strom_dsp::truepeak::{TruePeakLimiter, TruePeakLimiterParams};
use strom_dsp::{BandParams, BandType, Equalizer, EqualizerParams, Processor};

const FS: f32 = 48_000.0;
const FRAMES: usize = 480;
/// Buffers processed before the measured one (steady state).
const WARMUP_BUFFERS: usize = 16;

/// Hard limit on the instruction count (`Ir`) of one case: 20 % over the
/// count measured when it was set. x86-64 only; elsewhere report-only.
fn limit(measured: u64) -> LibraryBenchmarkConfig {
    let mut config = LibraryBenchmarkConfig::default();
    if cfg!(target_arch = "x86_64") {
        config.tool(Callgrind::default().hard_limits([(EventKind::Ir, measured * 6 / 5)]));
    }
    config
}

/// The level every case runs at: AVX2 on x86-64 (see the module docs), the
/// detected level elsewhere.
fn level() -> Level {
    #[cfg(target_arch = "x86_64")]
    {
        detect()
            .as_avx2()
            .map(Level::Avx2)
            .expect("simd_gungraun needs an x86-64 CPU with AVX2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        detect()
    }
}

/// Deterministic programme-like stereo/multichannel signal: tones spread
/// over the bands, a slow level wobble and a little noise.
fn signal(ch: usize, start_frame: usize) -> Vec<f32> {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64 ^ start_frame as u64;
    let mut out = Vec::with_capacity(FRAMES * ch);
    for n in start_frame..start_frame + FRAMES {
        let t = n as f64 / FS as f64;
        let env = 0.6 + 0.4 * (2.0 * PI * 1.7 * t).sin();
        for c in 0..ch {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let nz = (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            let v = env
                * (0.5 * (2.0 * PI * (220.0 + 10.0 * c as f64) * t).sin()
                    + 0.2 * (2.0 * PI * 2_100.0 * t).sin()
                    + 0.1 * (2.0 * PI * 7_000.0 * t).sin())
                + 0.02 * nz;
            out.push(v as f32);
        }
    }
    out
}

/// Warm `p` up on `WARMUP_BUFFERS` buffers, then return it with the buffer
/// to be measured.
fn warmed<P>(mut p: P, ch: usize, mut process: impl FnMut(&mut P, &mut [f32])) -> (P, Vec<f32>) {
    for i in 0..WARMUP_BUFFERS {
        let mut buf = signal(ch, i * FRAMES);
        process(&mut p, &mut buf);
    }
    (p, signal(ch, WARMUP_BUFFERS * FRAMES))
}

/// An oversampler with its up and down buffers.
struct Os {
    o: Oversampler,
    up: Vec<f32>,
}

impl Os {
    fn process(&mut self, buf: &mut [f32]) {
        self.o.upsample(buf, &mut self.up);
        self.o.downsample(&self.up, buf);
    }
}

fn setup_oversampler(factor: OversampleFactor) -> (Os, Vec<f32>) {
    let mut o = Oversampler::new();
    o.set_simd_level(level());
    o.configure(factor, 2);
    let up = vec![0.0; FRAMES * 2 * factor.ratio()];
    warmed(Os { o, up }, 2, Os::process)
}

/// HPF + two peaking bands + high shelf (as in `strip_gungraun`).
fn setup_equalizer(ch: usize) -> (Equalizer, Vec<f32>) {
    let mut p = EqualizerParams {
        num_bands: 4,
        ..EqualizerParams::default()
    };
    let bands = [
        (BandType::HighPass, 80.0, 0.0, 0.707),
        (BandType::Peaking, 300.0, -3.0, 1.0),
        (BandType::Peaking, 3000.0, 2.0, 1.4),
        (BandType::HighShelf, 10_000.0, 3.0, 0.707),
    ];
    for (b, (band_type, frequency, gain_db, q)) in p.bands.iter_mut().zip(bands) {
        *b = BandParams {
            enabled: true,
            band_type,
            frequency,
            gain_db,
            q,
        };
    }
    let mut e = Equalizer::new();
    e.set_simd_level(level());
    e.set_params(&p);
    e.configure(FS, ch);
    warmed(e, ch, |p, b| p.process(b))
}

fn setup_truepeak() -> (TruePeakLimiter, Vec<f32>) {
    let mut l = TruePeakLimiter::new();
    l.set_simd_level(level());
    l.set_params(&TruePeakLimiterParams {
        lookahead_ms: 5.0,
        ..TruePeakLimiterParams::default()
    });
    l.configure(FS, 2);
    // Drive it into limiting.
    let (l, mut buf) = warmed(l, 2, |p, b| {
        b.iter_mut().for_each(|x| *x *= 2.0);
        p.process(b)
    });
    buf.iter_mut().for_each(|x| *x *= 2.0);
    (l, buf)
}

/// The full broadcast chain with the `ebu-r128` defaults (standard latency).
fn setup_broadcast() -> (BroadcastProcessor, Vec<f32>) {
    #[cfg(target_arch = "x86_64")]
    assert!(
        detect().as_avx2().is_some() && detect().as_avx512().is_none(),
        "simd_gungraun: the broadcast chain runs at the detected level, which \
         must be AVX2 for its limit to apply (got {:?})",
        detect()
    );
    let mut bp = BroadcastProcessor::new();
    bp.set_params(&BroadcastParams::preset(BroadcastPreset::EbuR128, 4));
    bp.configure(FS, 2);
    warmed(bp, 2, |p, b| {
        p.process(b);
        black_box(p.take_meters());
    })
}

// Each benchmark returns its input so that the processors and buffers are
// dropped outside the measured function.

#[library_benchmark]
#[bench::x4(args = (setup_oversampler(OversampleFactor::X4)), config = limit(251_980))]
#[bench::x8(args = (setup_oversampler(OversampleFactor::X8)), config = limit(394_764))]
fn oversampler_up_down(mut input: (Os, Vec<f32>)) -> (Os, Vec<f32>) {
    input.0.process(black_box(&mut input.1));
    black_box(input)
}

#[library_benchmark]
#[bench::stereo(args = (setup_equalizer(2)), config = limit(26_692))]
#[bench::ch8(args = (setup_equalizer(8)), config = limit(71_000))]
fn equalizer_4_bands(mut input: (Equalizer, Vec<f32>)) -> (Equalizer, Vec<f32>) {
    input.0.process(black_box(&mut input.1));
    black_box(input)
}

#[library_benchmark]
#[bench::lookahead_5ms(args = (setup_truepeak()), config = limit(351_405))]
fn truepeak(mut input: (TruePeakLimiter, Vec<f32>)) -> (TruePeakLimiter, Vec<f32>) {
    input.0.process(black_box(&mut input.1));
    black_box(input)
}

#[library_benchmark]
#[bench::ebu_r128(args = (setup_broadcast()), config = limit(2_941_953))]
fn broadcast_chain(mut input: (BroadcastProcessor, Vec<f32>)) -> (BroadcastProcessor, Vec<f32>) {
    input.0.process(black_box(&mut input.1));
    black_box(input.0.take_meters());
    black_box(input)
}

library_benchmark_group!(
    name = kernels,
    benchmarks = [
        oversampler_up_down,
        equalizer_4_bands,
        truepeak,
        broadcast_chain
    ]
);

main!(library_benchmark_groups = kernels);
