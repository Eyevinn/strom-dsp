//! Criterion benchmarks (spec 6.3): one channel strip (gate → compressor →
//! 4-band EQ) plus a limiter at 48 kHz, and each processor on its own.
//!
//! Only the stable public API is used (`new`, `configure`, `set_params`,
//! `process`, the `*Params` structs with `..Default::default()`).
//!
//! Every benchmark processes one buffer per iteration. The input is copied from
//! a 2 s synthetic programme signal (noise plus a sine with a slow level
//! envelope, so the gate opens and closes and the compressor and limiter
//! actually work), advancing through it buffer by buffer, so the processors
//! see realistic, changing material. The copy (a `memcpy` of one buffer) is
//! included in the timings and is negligible.
//!
//! Throughput is reported in frames, so the share of one core is
//! `48 000 frames/s ÷ measured frames/s` (equivalently: time per buffer ÷
//! buffer duration; 480 frames = 10 ms, 1024 frames ≈ 21.3 ms).

// The parameter structs are built with `..Default::default()` even where every
// field is named, so the benchmark keeps compiling if fields are added.
#![allow(clippy::needless_update)]

use std::f64::consts::PI;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use strom_dsp::{
    BandParams, BandType, Compressor, CompressorParams, Equalizer, EqualizerParams, Gate,
    GateParams, Limiter, LimiterParams, Processor,
};

const FS: f32 = 48_000.0;
const BUFFER_FRAMES: [usize; 2] = [480, 1024];

/// Interleaved test programme of `secs` seconds and `ch` channels.
fn programme(ch: usize, secs: f64) -> Vec<f32> {
    let n = (FS as f64 * secs) as usize;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut out = Vec::with_capacity(n * ch);
    for i in 0..n {
        let t = i as f64 / FS as f64;
        // Level envelope 0.02 … 0.9 at 0.7 Hz: crosses gate/compressor/limiter thresholds.
        let env = 0.02 + 0.88 * (0.5 - 0.5 * (2.0 * PI * 0.7 * t).cos());
        for c in 0..ch {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let nz = (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            let tone = (2.0 * PI * (220.0 + 30.0 * c as f64) * t).sin();
            out.push((env * (0.7 * tone + 0.3 * nz)) as f32);
        }
    }
    out
}

/// Cycles through a programme signal one buffer at a time.
struct Feeder {
    src: Vec<f32>,
    pos: usize,
    len: usize,
}

impl Feeder {
    fn new(ch: usize, frames: usize) -> Self {
        let src = programme(ch, 2.0);
        Self {
            src,
            pos: 0,
            len: frames * ch,
        }
    }

    #[inline]
    fn fill(&mut self, buf: &mut [f32]) {
        if self.pos + self.len > self.src.len() {
            self.pos = 0;
        }
        buf.copy_from_slice(&self.src[self.pos..self.pos + self.len]);
        self.pos += self.len;
    }
}

fn band(band_type: BandType, frequency: f32, gain_db: f32, q: f32) -> BandParams {
    BandParams {
        enabled: true,
        band_type,
        frequency,
        gain_db,
        q,
    }
}

/// Realistic 4-band vocal/channel EQ: HPF, two bells, high shelf.
fn eq_params() -> EqualizerParams {
    let mut p = EqualizerParams {
        num_bands: 4,
        ..Default::default()
    };
    p.bands[0] = band(BandType::HighPass, 80.0, 0.0, 0.707);
    p.bands[1] = band(BandType::Peaking, 300.0, -3.0, 1.2);
    p.bands[2] = band(BandType::Peaking, 3000.0, 2.5, 0.9);
    p.bands[3] = band(BandType::HighShelf, 10_000.0, 2.0, 0.707);
    p
}

fn gate_params() -> GateParams {
    GateParams {
        open_threshold_db: -40.0,
        close_threshold_db: -50.0,
        attack_ms: 2.0,
        hold_ms: 50.0,
        release_ms: 150.0,
        reduction_db: -40.0,
        ..Default::default()
    }
}

fn compressor_params() -> CompressorParams {
    CompressorParams {
        threshold: 0.25,
        ratio: 4.0,
        attack_ms: 10.0,
        release_ms: 120.0,
        makeup_gain: 1.5,
        knee: 0.25,
        ..Default::default()
    }
}

fn limiter_params() -> LimiterParams {
    LimiterParams {
        enabled: true,
        threshold_db: -1.0,
        attack_ms: 1.0,
        release_ms: 80.0,
        lookahead_ms: 5.0,
        ..Default::default()
    }
}

fn make<P: Processor>(mut p: P, ch: usize, set: impl FnOnce(&mut P)) -> P {
    p.configure(FS, ch);
    set(&mut p);
    p
}

fn equalizer(ch: usize) -> Equalizer {
    make(Equalizer::new(), ch, |e| e.set_params(&eq_params()))
}

/// Run the closure from `setup` on one freshly fed buffer per iteration.
fn run(
    c: &mut Criterion,
    group: &str,
    ch: usize,
    mut setup: impl FnMut() -> Box<dyn FnMut(&mut [f32])>,
) {
    let mut g = c.benchmark_group(group);
    for &frames in &BUFFER_FRAMES {
        g.throughput(Throughput::Elements(frames as u64));
        let mut feeder = Feeder::new(ch, frames);
        let mut buf = vec![0.0f32; frames * ch];
        let mut proc_buf = setup();
        // Warm up the processors on real material (envelopes, filter states).
        for _ in 0..200 {
            feeder.fill(&mut buf);
            proc_buf(&mut buf);
        }
        g.bench_with_input(BenchmarkId::from_parameter(frames), &frames, |b, _| {
            b.iter(|| {
                feeder.fill(&mut buf);
                proc_buf(black_box(&mut buf));
            })
        });
    }
    g.finish();
}

fn bench_strip(c: &mut Criterion) {
    run(c, "strip_stereo", 2, || {
        let mut gate = make(Gate::new(), 2, |g| g.set_params(&gate_params()));
        let mut comp = make(Compressor::new(), 2, |p| p.set_params(&compressor_params()));
        let mut eq = equalizer(2);
        let mut lim = make(Limiter::new(), 2, |l| l.set_params(&limiter_params()));
        Box::new(move |buf| {
            gate.process(buf);
            comp.process(buf);
            eq.process(buf);
            lim.process(buf);
        })
    });
}

fn bench_processors(c: &mut Criterion) {
    run(c, "gate_stereo", 2, || {
        let mut p = make(Gate::new(), 2, |g| g.set_params(&gate_params()));
        Box::new(move |buf| p.process(buf))
    });
    run(c, "compressor_stereo", 2, || {
        let mut p = make(Compressor::new(), 2, |g| g.set_params(&compressor_params()));
        Box::new(move |buf| p.process(buf))
    });
    run(c, "limiter_stereo", 2, || {
        let mut p = make(Limiter::new(), 2, |g| g.set_params(&limiter_params()));
        Box::new(move |buf| p.process(buf))
    });
    run(c, "limiter_stereo_hot", 2, || {
        let mut p = make(Limiter::new(), 2, |g| g.set_params(&limiter_params()));
        Box::new(move |buf| {
            for x in buf.iter_mut() {
                *x *= 10.0;
            }
            p.process(buf)
        })
    });
}

fn bench_eq(c: &mut Criterion) {
    for (name, ch) in [("eq4_mono", 1), ("eq4_stereo", 2), ("eq4_8ch", 8)] {
        run(c, name, ch, move || {
            let mut e = equalizer(ch);
            Box::new(move |buf| e.process(buf))
        });
    }
    // Parameter automation: a bell's gain changes on every buffer, so the
    // smoothing/coefficient-interpolation path runs continuously.
    run(c, "eq4_stereo_automated", 2, || {
        let mut e = equalizer(2);
        let mut p = eq_params();
        let mut k = 0u32;
        Box::new(move |buf| {
            k = k.wrapping_add(1);
            p.bands[2].gain_db = if k.is_multiple_of(2) { 2.5 } else { 4.0 };
            e.set_params(&p);
            e.process(buf);
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(3));
    targets = bench_eq, bench_processors, bench_strip
}
criterion_main!(benches);
