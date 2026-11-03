//! Instruction-count benchmarks (gungraun / Valgrind Callgrind).
//!
//! Deterministic counterpart of the criterion bench in `strip.rs`: counts the
//! instructions for one 1024-frame, 48 kHz stereo buffer through the channel
//! strip (gate → compressor → 4-band EQ → limiter) and through each processor
//! alone. Only the stable public API is used.
//!
//! Construction, `configure`, parameter setup, signal generation and a warm-up
//! (so every processor is in steady state, past its first-buffer snap path)
//! all happen in the bench arguments, which gungraun does not count.
//!
//! Linux only (needs Valgrind and a `gungraun-runner` of the same version as
//! the `gungraun` crate):
//!
//! ```text
//! cargo install gungraun-runner --version 0.20.0
//! cargo bench -p strom-dsp --features gungraun --bench strip_gungraun
//! ```
//!
//! No regression limits are configured yet: results are report-only.

use std::hint::black_box;

use gungraun::{library_benchmark, library_benchmark_group, main};
use strom_dsp::{
    BandParams, BandType, Compressor, CompressorParams, Equalizer, EqualizerParams, Gate,
    GateParams, Limiter, LimiterParams, Processor,
};

const SAMPLE_RATE: f32 = 48_000.0;
const CHANNELS: usize = 2;
const FRAMES: usize = 1024;
/// Buffers processed before the measured one.
const WARMUP_BUFFERS: usize = 16;

/// Deterministic programme-like test signal: a 220 Hz + 3.1 kHz mix with a
/// slow amplitude wobble and a little decorrelation between the channels, at
/// a level that keeps the gate open and the compressor and limiter working.
fn signal(start_frame: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(FRAMES * CHANNELS);
    for n in start_frame..start_frame + FRAMES {
        let t = n as f32 / SAMPLE_RATE;
        let env = 0.6 + 0.4 * (2.0 * std::f32::consts::PI * 1.7 * t).sin();
        let a = (2.0 * std::f32::consts::PI * 220.0 * t).sin();
        let b = (2.0 * std::f32::consts::PI * 3100.0 * t).sin();
        out.push(env * (0.8 * a + 0.3 * b));
        out.push(env * (0.7 * a - 0.35 * b));
    }
    out
}

fn gate() -> Gate {
    let mut g = Gate::new();
    g.set_params(&GateParams {
        open_threshold_db: -45.0,
        close_threshold_db: -55.0,
        ..GateParams::default()
    });
    g.configure(SAMPLE_RATE, CHANNELS);
    g
}

fn compressor() -> Compressor {
    let mut c = Compressor::new();
    c.set_params(&CompressorParams {
        threshold: 0.25,
        ratio: 4.0,
        attack_ms: 10.0,
        release_ms: 150.0,
        makeup_gain: 1.5,
        knee: 0.25,
    });
    c.configure(SAMPLE_RATE, CHANNELS);
    c
}

/// HPF + two peaking bands + high shelf.
fn equalizer() -> Equalizer {
    let mut p = EqualizerParams {
        num_bands: 4,
        ..EqualizerParams::default()
    };
    p.bands[0] = BandParams {
        band_type: BandType::HighPass,
        frequency: 80.0,
        q: 0.707,
        ..BandParams::default()
    };
    p.bands[1] = BandParams {
        band_type: BandType::Peaking,
        frequency: 300.0,
        gain_db: -3.0,
        q: 1.0,
        ..BandParams::default()
    };
    p.bands[2] = BandParams {
        band_type: BandType::Peaking,
        frequency: 3000.0,
        gain_db: 2.0,
        q: 1.4,
        ..BandParams::default()
    };
    p.bands[3] = BandParams {
        band_type: BandType::HighShelf,
        frequency: 10_000.0,
        gain_db: 3.0,
        q: 0.707,
        ..BandParams::default()
    };
    let mut e = Equalizer::new();
    e.set_params(&p);
    e.configure(SAMPLE_RATE, CHANNELS);
    e
}

fn limiter() -> Limiter {
    let mut l = Limiter::new();
    l.set_params(&LimiterParams {
        threshold_db: -1.0,
        attack_ms: 1.0,
        release_ms: 100.0,
        lookahead_ms: 5.0,
        enabled: true,
    });
    l.configure(SAMPLE_RATE, CHANNELS);
    l
}

/// The full channel strip.
struct Strip {
    gate: Gate,
    comp: Compressor,
    eq: Equalizer,
    lim: Limiter,
}

impl Strip {
    fn process(&mut self, buf: &mut [f32]) {
        self.gate.process(buf);
        self.comp.process(buf);
        self.eq.process(buf);
        self.lim.process(buf);
    }
}

/// Warm `process` up on `WARMUP_BUFFERS` buffers, then return it with the
/// buffer to be measured.
fn warmed<P>(mut p: P, mut process: impl FnMut(&mut P, &mut [f32])) -> (P, Vec<f32>) {
    for i in 0..WARMUP_BUFFERS {
        let mut buf = signal(i * FRAMES);
        process(&mut p, &mut buf);
    }
    (p, signal(WARMUP_BUFFERS * FRAMES))
}

fn setup_strip() -> (Strip, Vec<f32>) {
    let strip = Strip {
        gate: gate(),
        comp: compressor(),
        eq: equalizer(),
        lim: limiter(),
    };
    warmed(strip, Strip::process)
}

fn setup_gate() -> (Gate, Vec<f32>) {
    warmed(gate(), |p, b| p.process(b))
}

fn setup_compressor() -> (Compressor, Vec<f32>) {
    warmed(compressor(), |p, b| p.process(b))
}

fn setup_equalizer() -> (Equalizer, Vec<f32>) {
    warmed(equalizer(), |p, b| p.process(b))
}

fn setup_limiter() -> (Limiter, Vec<f32>) {
    warmed(limiter(), |p, b| p.process(b))
}

// Each benchmark returns its input so that the processors and buffers are
// dropped outside the measured function.

#[library_benchmark]
#[bench::stereo_48k_1024(setup_strip())]
fn strip(mut input: (Strip, Vec<f32>)) -> (Strip, Vec<f32>) {
    input.0.process(black_box(&mut input.1));
    black_box(input)
}

#[library_benchmark]
#[bench::stereo_48k_1024(setup_gate())]
fn gate_alone(mut input: (Gate, Vec<f32>)) -> (Gate, Vec<f32>) {
    input.0.process(black_box(&mut input.1));
    black_box(input)
}

#[library_benchmark]
#[bench::stereo_48k_1024(setup_compressor())]
fn compressor_alone(mut input: (Compressor, Vec<f32>)) -> (Compressor, Vec<f32>) {
    input.0.process(black_box(&mut input.1));
    black_box(input)
}

#[library_benchmark]
#[bench::stereo_48k_1024(setup_equalizer())]
fn equalizer_alone(mut input: (Equalizer, Vec<f32>)) -> (Equalizer, Vec<f32>) {
    input.0.process(black_box(&mut input.1));
    black_box(input)
}

#[library_benchmark]
#[bench::stereo_48k_1024(setup_limiter())]
fn limiter_alone(mut input: (Limiter, Vec<f32>)) -> (Limiter, Vec<f32>) {
    input.0.process(black_box(&mut input.1));
    black_box(input)
}

library_benchmark_group!(
    name = processors,
    benchmarks = [
        strip,
        gate_alone,
        compressor_alone,
        equalizer_alone,
        limiter_alone
    ]
);

main!(library_benchmark_groups = processors);
