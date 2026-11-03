//! Criterion benchmarks of the higher-order filters (phase 1b, 2.4):
//! Butterworth order 4 and 8, and the Linkwitz-Riley LR4 crossover with 3 and
//! 5 bands, all at 48 kHz stereo with 480-frame (10 ms) buffers.
//!
//! Every iteration processes one buffer copied from a 2 s noise-plus-tone
//! signal, advancing through it buffer by buffer; the copy is included and
//! negligible. Throughput is in frames, so the share of one core is
//! `48 000 frames/s ÷ measured frames/s`.

use std::f64::consts::PI;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use strom_dsp::filter::{
    Butterworth, ButterworthParams, Crossover, CrossoverParams, FilterKind, LrOrder,
    MAX_CROSSOVER_BANDS,
};
use strom_dsp::Processor;

const FS: f32 = 48_000.0;
const CH: usize = 2;
const FRAMES: usize = 480;

fn programme() -> Vec<f32> {
    let n = 2 * FS as usize;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut out = Vec::with_capacity(n * CH);
    for i in 0..n {
        let t = i as f64 / FS as f64;
        for c in 0..CH {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let nz = (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            let tone = (2.0 * PI * (220.0 + 30.0 * c as f64) * t).sin();
            out.push((0.5 * tone + 0.3 * nz) as f32);
        }
    }
    out
}

struct Feeder {
    src: Vec<f32>,
    pos: usize,
}

impl Feeder {
    fn new() -> Self {
        Self {
            src: programme(),
            pos: 0,
        }
    }

    fn next(&mut self) -> &[f32] {
        let len = FRAMES * CH;
        if self.pos + len > self.src.len() {
            self.pos = 0;
        }
        let s = &self.src[self.pos..self.pos + len];
        self.pos += len;
        s
    }
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("filters_48k_stereo_480");
    g.throughput(Throughput::Elements(FRAMES as u64));

    for order in [4usize, 8] {
        let mut flt = Butterworth::new();
        flt.configure(FS, CH);
        flt.set_params(&ButterworthParams {
            kind: FilterKind::HighPass,
            order,
            cutoff_hz: 80.0,
        });
        let mut feed = Feeder::new();
        let mut buf = vec![0.0f32; FRAMES * CH];
        g.bench_function(format!("butterworth_order{order}"), |b| {
            b.iter(|| {
                buf.copy_from_slice(feed.next());
                flt.process(black_box(&mut buf));
            })
        });
    }

    for (bands, freqs) in [
        (3usize, [250.0f32, 2500.0, 0.0, 0.0]),
        (5, [120.0, 500.0, 2000.0, 6000.0]),
    ] {
        let mut xo = Crossover::new();
        xo.configure(FS, CH);
        xo.set_params(&CrossoverParams {
            order: LrOrder::Lr4,
            num_bands: bands,
            frequencies_hz: freqs,
        });
        let mut feed = Feeder::new();
        let mut out = vec![vec![0.0f32; FRAMES * CH]; MAX_CROSSOVER_BANDS];
        g.bench_function(format!("crossover_lr4_{bands}band"), |b| {
            b.iter(|| {
                xo.process(black_box(feed.next()), &mut out);
                black_box(&out);
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
