//! Downward expander per band of the multiband compressor
//! ([`MultibandCompressor::set_expanders`]).
//!
//! - Static curve: per band, compression above the band threshold, unity
//!   between, expansion below the expander threshold, with both soft knees,
//!   matches the gain-computer formula within 0.1 dB.
//! - The attenuation never exceeds the floor.
//! - No chatter on a level hovering at the expander threshold (guard: the
//!   same signal with hold 0 and a fast close does chatter).
//! - Speech-like bursts with low-level noise between them: the noise drops
//!   by the floor, the bursts are unchanged within 0.5 dB.
//! - Ratio 1 is the multiband without an expander, bit for bit, also after
//!   the expander has been on.
//! - One band is a wideband expander; no allocation while processing.

use crate::measure_support;

use crate::measure_support::{db, lin, rms, sine, speech_like_bursts, Bursts, Rng};
use strom_dsp::compressor::CompressorDetector;
use strom_dsp::dynamics::ExpanderParams;
use strom_dsp::filter::{Crossover, CrossoverParams, LrOrder};
use strom_dsp::multiband::{
    MultibandBandParams, MultibandCompressor, MultibandParams, DEFAULT_FREQUENCIES_HZ, MAX_BANDS,
};
use strom_dsp::Processor;

use crate::counting_alloc::allocs;

const FS: f64 = 48_000.0;

fn band(threshold_db: f32, ratio: f32, knee_db: f32) -> MultibandBandParams {
    MultibandBandParams {
        threshold_db,
        ratio,
        attack_ms: 5.0,
        release_ms: 100.0,
        makeup_db: 0.0,
        knee_db,
    }
}

fn params(nb: usize, b: MultibandBandParams) -> MultibandParams {
    MultibandParams {
        num_bands: nb,
        frequencies_hz: DEFAULT_FREQUENCIES_HZ,
        drive_db: 0.0,
        detector: CompressorDetector::Peak,
        bands: [b; MAX_BANDS],
    }
}

fn multiband(ch: usize, p: &MultibandParams, e: &ExpanderParams) -> MultibandCompressor {
    let mut m = MultibandCompressor::new();
    m.set_params(p);
    m.set_expanders(&[*e; MAX_BANDS]);
    m.configure(FS as f32, ch);
    m
}

/// Static reduction of the combined curve, written out from the formulas
/// (Giannoulis et al. 2012 for the compressor; its mirror below `E`).
fn formula(
    x: f64,
    (t, w, ratio): (f64, f64, f64),
    (e, we, er, floor): (f64, f64, f64, f64),
) -> f64 {
    let s = 1.0 - 1.0 / ratio;
    let over = x - t;
    let comp = if over <= -w / 2.0 {
        0.0
    } else if over < w / 2.0 {
        s * (over + w / 2.0).powi(2) / (2.0 * w)
    } else {
        s * over
    };
    let se = er - 1.0;
    let under = e - x;
    let exp = if under <= -we / 2.0 {
        0.0
    } else if under < we / 2.0 {
        se * (under + we / 2.0).powi(2) / (2.0 * we)
    } else {
        se * under
    };
    comp + exp.min(floor)
}

/// Integer-Hz frequency in the middle of band `k` of `nb` (geometric centre).
fn centre(k: usize, nb: usize) -> f64 {
    let lo = if k == 0 {
        30.0
    } else {
        DEFAULT_FREQUENCIES_HZ[k - 1] as f64
    };
    let hi = if k + 1 == nb {
        16_000.0
    } else {
        DEFAULT_FREQUENCIES_HZ[k] as f64
    };
    (lo * hi).sqrt().round()
}

/// Settled sampled peak of band `k` of the crossover for a mono input.
fn band_level(nb: usize, k: usize, x: &[f32]) -> f64 {
    let mut xo = Crossover::new();
    xo.configure(FS as f32, 1);
    xo.set_params(&CrossoverParams {
        order: LrOrder::Lr4,
        num_bands: nb,
        frequencies_hz: DEFAULT_FREQUENCIES_HZ,
    });
    let mut bands = vec![vec![0.0f32; x.len()]; MAX_BANDS];
    xo.process(x, &mut bands[..nb]);
    let tail = &bands[k][x.len() - 4800..];
    measure_support::peak(tail)
}

/// Per band (2–5 bands, every band) and wideband: the settled gain
/// reduction (compressor + expander) equals the formula at the band level
/// over levels spanning compression, both knees, the unity region, the
/// expansion slope and the floor, within 0.1 dB. One band: the output
/// amplitude matches too (the crossover passes one band bit for bit).
#[test]
fn static_curve_per_band_matches_formula() {
    let (t, w, ratio) = (-20.0, 6.0, 3.0);
    let e = ExpanderParams {
        threshold_db: -40.0,
        ratio: 3.0,
        floor_db: 15.0,
        knee_db: 8.0,
        ..ExpanderParams::default()
    };
    let p_band = band(t as f32, ratio as f32, w as f32);
    let mut worst = 0.0f64;
    for nb in 1..=MAX_BANDS {
        for k in 0..nb {
            let f = centre(k, nb);
            for x_db in [
                -60.0, -50.0, -46.0, -42.0, -40.0, -37.0, -30.0, -22.0, -18.0, -10.0,
            ] {
                // Amplitude so that the band reads x_db.
                let probe = sine(24_000, FS, f, 0.1);
                let gain = band_level(nb, k, &probe) / 0.1;
                let amp = lin(x_db) / gain;
                let x = sine(96_000, FS, f, amp);
                let level = db(band_level(nb, k, &x));
                let mut m = multiband(1, &params(nb, p_band), &e);
                let mut y = x.clone();
                m.process(&mut y);
                let got = m.band_gain_reduction_db(k) + m.band_expander_reduction_db(k);
                let want = formula(
                    level,
                    (t, w, ratio),
                    (
                        e.threshold_db as f64,
                        e.knee_db as f64,
                        e.ratio as f64,
                        e.floor_db as f64,
                    ),
                );
                worst = worst.max((got - want).abs());
                assert!(
                    (got - want).abs() < 0.1,
                    "{nb} bands, band {k} at {f} Hz, level {level:.2} dB: reduction {got:.3}, \
                     formula {want:.3}"
                );
                if nb == 1 {
                    let out = db(measure_support::peak(&y[y.len() - 4800..]));
                    assert!(
                        (out - (level - want)).abs() < 0.1,
                        "wideband at {level:.2} dB: out {out:.3}, want {:.3}",
                        level - want
                    );
                }
            }
        }
    }
    eprintln!("static curve: worst deviation {worst:.4} dB");
}

/// Random noise with level jumps and random expander settings: the
/// expander attenuation never exceeds its floor (checked every 32 frames).
#[test]
fn attenuation_never_exceeds_floor() {
    let mut r = Rng::new(17);
    for trial in 0..12 {
        let e = ExpanderParams {
            threshold_db: -20.0 - 50.0 * r.uniform() as f32,
            ratio: 1.0 + 9.0 * r.uniform() as f32,
            floor_db: 40.0 * r.uniform() as f32,
            knee_db: 12.0 * r.uniform() as f32,
            hold_ms: 100.0 * r.uniform() as f32,
            open_ms: 0.1 + 10.0 * r.uniform() as f32,
            close_ms: 1.0 + 300.0 * r.uniform() as f32,
        };
        let nb = 1 + trial % MAX_BANDS;
        let mut m = multiband(2, &params(nb, band(-10.0, 2.0, 6.0)), &e);
        let mut x = Vec::new();
        for seg in 0..20 {
            let amp = lin(-90.0 + 85.0 * r.uniform());
            let n = 1000 + (r.uniform() * 6000.0) as usize;
            let seed = (trial * 100 + seg) as u64 + 1;
            x.extend(measure_support::white(n * 2, amp, seed));
        }
        for chunk in x.chunks_mut(64) {
            m.process(chunk);
            for k in 0..nb {
                let a = m.band_expander_reduction_db(k);
                assert!(
                    a <= e.floor_db as f64 + 1e-9 && a >= 0.0,
                    "trial {trial}: band {k} attenuation {a} dB, floor {} dB",
                    e.floor_db
                );
            }
        }
    }
}

/// Total variation (dB per second: the sum of every movement of the gain,
/// up or down) of the expander attenuation of band `k` over the last two
/// seconds, for a sine at the band centre whose level hovers ±1.5 dB around
/// the expander threshold with a new random level every 20 … 60 ms.
fn hover_ripple(nb: usize, k: usize, e: &ExpanderParams) -> f64 {
    let f = centre(k, nb);
    let probe = sine(24_000, FS, f, 0.1);
    let gain = band_level(nb, k, &probe) / 0.1;
    let mut r = Rng::new(99);
    let frames = 4 * FS as usize;
    let mut x = vec![0.0f32; frames];
    let mut i = 0;
    let mut phase = 0.0f64;
    while i < frames {
        let level = e.threshold_db as f64 + 3.0 * (r.uniform() - 0.5);
        let amp = lin(level) / gain;
        let n = ((0.02 + 0.04 * r.uniform()) * FS) as usize;
        for v in x[i..(i + n).min(frames)].iter_mut() {
            *v = (amp * phase.sin()) as f32;
            phase += 2.0 * std::f64::consts::PI * f / FS;
        }
        i += n;
    }
    let mut m = multiband(1, &params(nb, band(-10.0, 1.0, 6.0)), e);
    let mut lo = f64::INFINITY;
    let mut hi = 0.0f64;
    let mut tv = 0.0;
    let mut prev: Option<f64> = None;
    for (j, chunk) in x.chunks_mut(16).enumerate() {
        m.process(chunk);
        if j * 16 >= 2 * FS as usize {
            let a = m.band_expander_reduction_db(k);
            lo = lo.min(a);
            hi = hi.max(a);
            if let Some(p) = prev {
                tv += (a - p).abs();
            }
            prev = Some(a);
        }
    }
    eprintln!(
        "p-p {:.3} dB, total variation {:.3} dB/s",
        hi - lo,
        tv / 2.0
    );
    tv / 2.0
}

/// A level hovering at the expander threshold does not make the gain
/// chatter. The static attenuation swings by 1.5 dB about 25 times a second;
/// with the hold and the slower close the gain moves less than 3 dB per
/// second in total (it rests near its most open value and closes slowly).
/// Guard: the same signal with hold 0 and a close as fast as the open moves
/// the gain more than 6 dB/s and more than three times as much.
#[test]
fn no_chatter_at_threshold() {
    let e = ExpanderParams {
        threshold_db: -40.0,
        ratio: 2.0,
        floor_db: 12.0,
        knee_db: 6.0,
        ..ExpanderParams::default()
    };
    let fast = ExpanderParams {
        hold_ms: 0.0,
        close_ms: e.open_ms,
        ..e
    };
    for (nb, k) in [(1, 0), (4, 1), (4, 2)] {
        let calm = hover_ripple(nb, k, &e);
        let chatter = hover_ripple(nb, k, &fast);
        eprintln!(
            "{nb} bands, band {k}: gain movement {calm:.3} dB/s (no hold, fast close: \
             {chatter:.3} dB/s)"
        );
        assert!(calm < 3.0, "{nb} bands, band {k}: {calm} dB/s");
        assert!(
            chatter > 6.0 && chatter > 3.0 * calm,
            "{nb} bands, band {k}: guard {chatter} dB/s"
        );
    }
}

/// Speech-like bursts (noise band-limited to 3.4 kHz, −24 dB RMS, 300 ms on,
/// 700 ms off) with white noise at −66 dB RMS between them, through 4 bands
/// with the expander (threshold −45 dB, 1:3, floor 10 dB; all band levels
/// of the noise lie far below it) and no compression. Against the same
/// multiband with the expander at ratio 1: the noise in the last 150 ms of
/// each gap is 10 ± 0.5 dB lower, and each burst from 20 ms after its onset
/// to its end is within 0.5 dB.
#[test]
fn noise_between_bursts_is_lowered() {
    let frames = 12 * FS as usize;
    let (x, bursts) = speech_like_bursts(
        frames,
        FS,
        &Bursts {
            bandwidth: 3400.0,
            burst_s: 0.3,
            gap_s: 0.7,
            burst_rms: lin(-24.0),
            floor_rms: lin(-66.0),
            seed: 7,
        },
    );
    let e = ExpanderParams {
        threshold_db: -45.0,
        ratio: 3.0,
        floor_db: 10.0,
        knee_db: 6.0,
        ..ExpanderParams::default()
    };
    let p = params(4, band(-6.0, 1.0, 6.0));
    let run = |e: &ExpanderParams| {
        let mut m = multiband(1, &p, e);
        let mut y = x.clone();
        for c in y.chunks_mut(480) {
            m.process(c);
        }
        y
    };
    let y = run(&e);
    let reference = run(&ExpanderParams { ratio: 1.0, ..e });
    let (mut worst_gap, mut worst_burst) = (0.0f64, 0.0f64);
    for (i, &(s, end)) in bursts.iter().enumerate().skip(1) {
        let b = s + (0.02 * FS) as usize..end;
        let burst = db(rms(&y[b.clone()]) / rms(&reference[b]));
        worst_burst = worst_burst.max(burst.abs());
        assert!(burst.abs() < 0.5, "burst {i}: {burst:.3} dB");
        let g = s - (0.15 * FS) as usize..s;
        let gap = db(rms(&y[g.clone()]) / rms(&reference[g]));
        worst_gap = worst_gap.max((gap + 10.0).abs());
        assert!(
            (gap + 10.0).abs() < 0.5,
            "gap before burst {i}: {gap:.3} dB"
        );
    }
    eprintln!(
        "bursts: worst change {worst_burst:.3} dB; gaps: worst deviation from −10 dB \
         {worst_gap:.3} dB"
    );
}

/// Ratio 1 is the multiband without an expander, bit for bit: with other
/// expander fields set to anything, and after the expander has been on and
/// was switched back to ratio 1.
#[test]
fn ratio_one_is_bit_exact() {
    let p = params(4, band(-30.0, 4.0, 6.0));
    let mut seed = Rng::new(3);
    let x: Vec<f32> = (0..96_000)
        .map(|n| {
            let env = if (n / 9000) % 2 == 0 { 0.5 } else { 0.001 };
            (env * seed.bipolar()) as f32
        })
        .collect();
    let to_ch = |m: &[f32]| measure_support::to_channels(m, 2);
    let mut plain = MultibandCompressor::new();
    plain.set_params(&p);
    plain.configure(FS as f32, 2);
    let mut a = to_ch(&x);
    plain.process(&mut a);

    let odd = ExpanderParams {
        threshold_db: -10.0,
        ratio: 1.0,
        floor_db: 60.0,
        knee_db: 0.0,
        open_ms: 0.1,
        close_ms: 1.0,
        hold_ms: 0.0,
    };
    let mut m = multiband(2, &p, &odd);
    let mut b = to_ch(&x);
    m.process(&mut b);
    assert!(a == b, "ratio 1 with other fields set changed the output");

    // On, then back to ratio 1: once the expander has opened and turned
    // itself off, the output is again the plain multiband's.
    let on = ExpanderParams {
        threshold_db: -20.0,
        ratio: 4.0,
        ..odd
    };
    let mut m = multiband(2, &p, &on);
    let mut plain = MultibandCompressor::new();
    plain.set_params(&p);
    plain.configure(FS as f32, 2);
    let mut warm = to_ch(&x);
    let mut warm2 = warm.clone();
    m.process(&mut warm);
    plain.process(&mut warm2);
    m.set_expanders(&[odd; MAX_BANDS]);
    let mut settle = to_ch(&x[..24_000]);
    let mut settle2 = settle.clone();
    m.process(&mut settle);
    plain.process(&mut settle2);
    let mut c = to_ch(&x);
    let mut d = c.clone();
    m.process(&mut c);
    plain.process(&mut d);
    assert!(c == d, "after switching back to ratio 1 the output differs");
}

/// `set_expanders` and processing with active expanders never allocate.
#[test]
fn no_allocation() {
    let e = ExpanderParams {
        threshold_db: -40.0,
        ratio: 3.0,
        ..ExpanderParams::default()
    };
    let mut m = multiband(2, &params(5, band(-20.0, 2.0, 6.0)), &e);
    let mut x = measure_support::white(2 * 48_000, 0.01, 5);
    let before = allocs();
    for (i, c) in x.chunks_mut(512).enumerate() {
        if i % 10 == 0 {
            m.set_expanders(
                &[ExpanderParams {
                    ratio: 1.0 + (i % 7) as f32,
                    ..e
                }; MAX_BANDS],
            );
        }
        m.process(c);
    }
    assert_eq!(allocs() - before, 0);
}
