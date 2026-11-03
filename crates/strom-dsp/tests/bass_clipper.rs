//! Bass clipper on band 0 of the multiband ([`strom_dsp::bassclip`]).
//!
//! - A 60 Hz sine 6 dB over the bass threshold: 2nd and 3rd harmonics below
//!   −50 dBc (guard: the same with the rectified waveform as envelope fails).
//! - IM feedback, bass-into-highs (loud 50 Hz, soft 3 kHz) into a fixed
//!   upper clipper: the sidebands at 3 kHz ± 50/100 Hz are at least 10 dB
//!   lower with the feedback than without. The upper clipper here is a
//!   stand-in (a hard clip at the base rate, its activity = share of
//!   samples clipped per 10 ms block, fed back with one block of delay); the
//!   end-to-end check with the real final clipper belongs with it.
//! - Bands stay aligned: with the lookahead, the impulse response is the
//!   crossover sum delayed by the lookahead, flat within ±0.05 dB, and the
//!   latency is reported.
//! - Threshold and enable changes are click-free; no allocation.

mod measure_support;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use measure_support::{bass_into_highs, db, lin, sine, tone_amplitude};
use strom_dsp::bassclip::{BassClipper, BassClipperParams, BassEnvelope};
use strom_dsp::compressor::CompressorDetector;
use strom_dsp::filter::{Crossover, CrossoverParams, LrOrder};
use strom_dsp::multiband::{
    MultibandBandParams, MultibandCompressor, MultibandParams, DEFAULT_FREQUENCIES_HZ, MAX_BANDS,
};
use strom_dsp::Processor;

struct CountingAlloc;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn allocs() -> usize {
    ALLOCS.with(|c| c.get())
}

const FS: f64 = 48_000.0;
const LOOKAHEAD_MS: f32 = 3.0;

fn flat_band() -> MultibandBandParams {
    MultibandBandParams {
        threshold_db: -30.0,
        ratio: 1.0,
        attack_ms: 5.0,
        release_ms: 50.0,
        makeup_db: 0.0,
        knee_db: 0.0,
    }
}

fn params(nb: usize) -> MultibandParams {
    MultibandParams {
        num_bands: nb,
        frequencies_hz: DEFAULT_FREQUENCIES_HZ,
        drive_db: 0.0,
        detector: CompressorDetector::Peak,
        bands: [flat_band(); MAX_BANDS],
    }
}

fn multiband(ch: usize, nb: usize, bass: BassClipperParams) -> MultibandCompressor {
    let mut m = MultibandCompressor::new();
    m.set_params(&params(nb));
    m.set_bass_lookahead_ms(LOOKAHEAD_MS);
    m.set_bass_clipper(&bass);
    m.configure(FS as f32, ch);
    m
}

/// (2nd, 3rd) harmonic of `f` in `y` relative to its fundamental (dBc).
fn harmonics(y: &[f32], f: f64) -> (f64, f64, f64) {
    let a1 = tone_amplitude(y, FS, f);
    (
        db(a1),
        db(tone_amplitude(y, FS, 2.0 * f) / a1),
        db(tone_amplitude(y, FS, 3.0 * f) / a1),
    )
}

/// The band-0 clipper alone on a 60 Hz sine 6 dB over its threshold:
/// returns (output level, 2nd, 3rd harmonic in dBc) over the last second.
fn clip_sine(kind: BassEnvelope) -> (f64, f64, f64) {
    let mut b = BassClipper::new();
    b.set_allowed(true);
    b.set_envelope(kind);
    b.set_params(&BassClipperParams {
        enabled: true,
        threshold_db: -12.0,
    });
    b.configure(FS as f32, 1, (LOOKAHEAD_MS as f64 * 0.001 * FS) as usize);
    let mut x = sine(3 * FS as usize, FS, 60.0, lin(-6.0));
    for c in x.chunks_mut(480) {
        b.process(c);
    }
    harmonics(&x[2 * FS as usize..], 60.0)
}

/// A 60 Hz sine 6 dB over the bass threshold comes out at the threshold
/// with 2nd and 3rd harmonics below −50 dBc: the analytic envelope of a
/// steady sine is flat, so the gain is constant. Guard: with the rectified
/// waveform as envelope (the same gain pipeline) the harmonics are above
/// −50 dBc. Also through the whole 4-band multiband.
#[test]
fn bass_clip_adds_no_harmonics() {
    let (level, h2, h3) = clip_sine(BassEnvelope::Analytic);
    eprintln!("analytic: level {level:.2} dB, H2 {h2:.1} dBc, H3 {h3:.1} dBc");
    assert!((level + 12.0).abs() < 0.2, "level {level} dB");
    assert!(h2 < -50.0 && h3 < -50.0, "H2 {h2} dBc, H3 {h3} dBc");
    let (_, g2, g3) = clip_sine(BassEnvelope::Rectified);
    eprintln!("rectified (guard): H2 {g2:.1} dBc, H3 {g3:.1} dBc");
    assert!(g2.max(g3) > -50.0, "guard: rectified H2 {g2}, H3 {g3}");

    // Through the multiband: band 0 carries 60 Hz (LR4 at 120 Hz).
    let mut m = multiband(
        2,
        4,
        BassClipperParams {
            enabled: true,
            threshold_db: -12.0,
        },
    );
    let mono = sine(3 * FS as usize, FS, 60.0, lin(-6.0));
    let mut x = measure_support::to_channels(&mono, 2);
    for c in x.chunks_mut(960) {
        m.process(c);
    }
    let y = measure_support::channel(&x, 2, 0);
    let (level, h2, h3) = harmonics(&y[2 * FS as usize..], 60.0);
    eprintln!("multiband: level {level:.2} dB, H2 {h2:.1} dBc, H3 {h3:.1} dBc");
    assert!(
        h2 < -50.0 && h3 < -50.0,
        "multiband: H2 {h2} dBc, H3 {h3} dBc"
    );
    assert!(level < -10.5, "multiband: level {level} dB not reduced");
}

/// Stand-in upper clipper: hard clip at `c` (base rate). Returns the share
/// of samples it clipped.
fn hard_clip(x: &mut [f32], c: f32) -> f32 {
    let mut n = 0usize;
    for v in x.iter_mut() {
        if v.abs() > c {
            *v = v.signum() * c;
            n += 1;
        }
    }
    n as f32 / x.len().max(1) as f32
}

/// Largest sideband (dBFS) at 3 kHz ± 50 Hz and ± 100 Hz of the
/// bass-into-highs pair after the multiband and the stand-in upper
/// clipper, over the last 4 s of 8 s. With `feedback` the clipper's
/// activity of each 10 ms block goes to the multiband before the next.
fn sidebands(feedback: bool) -> (f64, f64) {
    let mut m = multiband(
        1,
        4,
        BassClipperParams {
            enabled: true,
            threshold_db: -12.0,
        },
    );
    let x = bass_into_highs(8 * FS as usize, FS, 50.0, 0.25, 3000.0, 0.06);
    let mut y = x.clone();
    let mut im = 0.0f64;
    for c in y.chunks_mut(480) {
        m.process(c);
        let a = hard_clip(c, 0.29);
        if feedback {
            m.set_upper_clip_activity(a);
        }
        im = m.bass_im_reduction_db();
    }
    let tail = &y[4 * FS as usize..];
    let side = [2900.0, 2950.0, 3050.0, 3100.0]
        .into_iter()
        .map(|f| db(tone_amplitude(tail, FS, f)))
        .fold(f64::NEG_INFINITY, f64::max);
    (side, im)
}

/// IM feedback: with the upper clipper's activity fed back, the bass
/// threshold comes down while bass peaks and clipping coincide, and the
/// intermodulation sidebands of the soft 3 kHz tone drop by at least
/// 10 dB. Guard: without the feedback (activity 0, the default) they stay.
#[test]
fn im_feedback_lowers_sidebands() {
    let (without, im0) = sidebands(false);
    let (with, im1) = sidebands(true);
    eprintln!(
        "sidebands at 3 kHz ±50/±100 Hz: without feedback {without:.1} dBFS, with \
         {with:.1} dBFS ({:.1} dB lower); IM threshold reduction {im1:.2} dB",
        without - with
    );
    assert_eq!(im0, 0.0, "no activity input, no feedback");
    assert!(im1 > 0.0 && im1 <= 3.0, "IM reduction {im1} dB");
    assert!(
        without - with >= 10.0,
        "only {:.1} dB lower",
        without - with
    );
}

/// Bands stay aligned: with the bass lookahead (clipper on, not clipping),
/// the impulse response is the crossover band sum delayed by the lookahead
/// (within float rounding), its magnitude is flat within ±0.05 dB, and the
/// reported latency is the lookahead. With one band: the input delayed.
#[test]
fn bands_stay_aligned_with_lookahead() {
    let l = MultibandCompressor::bass_lookahead_frames(LOOKAHEAD_MS, FS as f32);
    assert_eq!(l, 144);
    for nb in 1..=MAX_BANDS {
        let mut m = multiband(
            1,
            nb,
            BassClipperParams {
                enabled: true,
                threshold_db: 0.0,
            },
        );
        assert_eq!(m.latency_frames(), l);
        let n = FS as usize / 4;
        let mut h = vec![0.0f32; n];
        h[0] = 0.5;
        m.process(&mut h);
        let mut xo = Crossover::new();
        xo.configure(FS as f32, 1);
        xo.set_params(&CrossoverParams {
            order: LrOrder::Lr4,
            num_bands: nb,
            frequencies_hz: DEFAULT_FREQUENCIES_HZ,
        });
        let mut imp = vec![0.0f32; n];
        imp[0] = 0.5;
        let mut bands = vec![vec![0.0f32; n]; MAX_BANDS];
        xo.process(&imp, &mut bands[..nb]);
        for i in 0..n {
            let want = if i >= l {
                bands[..nb].iter().map(|b| b[i - l] as f64).sum::<f64>()
            } else {
                0.0
            };
            assert!(
                (h[i] as f64 - want).abs() < 1e-6,
                "{nb} bands: sample {i}: {} vs {want}",
                h[i]
            );
        }
        // Flat magnitude (the crossover's all-pass), measured on the output.
        let mut worst = 0.0f64;
        let mut f = 20.0;
        while f < 20_000.0 {
            let w = 2.0 * std::f64::consts::PI * f / FS;
            let (mut re, mut im) = (0.0, 0.0);
            for (i, &v) in h.iter().enumerate() {
                re += v as f64 * (w * i as f64).cos();
                im -= v as f64 * (w * i as f64).sin();
            }
            worst = worst.max(db((re * re + im * im).sqrt() / 0.5).abs());
            f *= 1.12;
        }
        assert!(worst < 0.05, "{nb} bands: {worst} dB");
    }
}

/// Peak (dB re the steady tone) of the second difference of a 1 kHz-free
/// low tone's output around a change: a click shows as a spike above the
/// smooth signal's own second difference.
fn click_residual(y: &[f32], range: std::ops::Range<usize>) -> f64 {
    // Sinusoid annihilator for 60 Hz: y[n] − 2cos(w)y[n−1] + y[n−2] = 0.
    let w = 2.0 * std::f64::consts::PI * 60.0 / FS;
    let c = 2.0 * w.cos();
    let mut m = 0.0f64;
    for n in range.start.max(2)..range.end {
        let r = y[n] as f64 - c * y[n - 1] as f64 + y[n - 2] as f64;
        m = m.max(r.abs());
    }
    m
}

/// Changing the threshold (−40 ↔ 0 dB) and toggling `enabled` while a
/// loud 60 Hz tone is clipped is click-free: the 60 Hz annihilator residual
/// stays below −60 dB re the tone amplitude.
#[test]
fn threshold_and_enable_changes_are_click_free() {
    let mut m = multiband(
        1,
        4,
        BassClipperParams {
            enabled: true,
            threshold_db: -12.0,
        },
    );
    let amp = lin(-6.0);
    let mut y = sine(6 * FS as usize, FS, 60.0, amp);
    let changes = [
        (
            FS as usize,
            BassClipperParams {
                enabled: true,
                threshold_db: -40.0,
            },
        ),
        (
            2 * FS as usize,
            BassClipperParams {
                enabled: true,
                threshold_db: 0.0,
            },
        ),
        (
            3 * FS as usize,
            BassClipperParams {
                enabled: true,
                threshold_db: -30.0,
            },
        ),
        (
            4 * FS as usize,
            BassClipperParams {
                enabled: false,
                threshold_db: -30.0,
            },
        ),
        (
            5 * FS as usize,
            BassClipperParams {
                enabled: true,
                threshold_db: -30.0,
            },
        ),
    ];
    let mut next = 0;
    for (i, c) in y.chunks_mut(240).enumerate() {
        if next < changes.len() && i * 240 >= changes[next].0 {
            m.set_bass_clipper(&changes[next].1);
            next += 1;
        }
        m.process(c);
    }
    let worst = click_residual(&y, FS as usize / 2..y.len());
    let rel = db(worst / amp);
    eprintln!("worst annihilator residual {rel:.1} dB re the tone");
    assert!(rel < -60.0, "{rel} dB");
}

/// Processing with the bass clipper, activity input, parameter changes and
/// reads never allocates.
#[test]
fn no_allocation() {
    let mut m = multiband(
        2,
        4,
        BassClipperParams {
            enabled: true,
            threshold_db: -20.0,
        },
    );
    let mut x = measure_support::to_channels(&sine(96_000, FS, 55.0, 0.5), 2);
    let before = allocs();
    for (i, c) in x.chunks_mut(512).enumerate() {
        m.set_upper_clip_activity((i % 5) as f32 * 0.01);
        m.set_bass_clipper(&BassClipperParams {
            enabled: i % 40 < 30,
            threshold_db: -20.0 + (i % 10) as f32,
        });
        m.process(c);
        let _ = m.take_bass_clip_gain_reduction_db();
    }
    assert_eq!(allocs() - before, 0);
}
