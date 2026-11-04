//! The final clipper's content-adaptive depth and bandwidth guard, and the
//! limiter peak control that takes what the clipper leaves. Short synthetic
//! signals with fixed seeds.
//!
//! - Content-adaptive depth: a synthetic vowel driven 3 dB into the clipper
//!   loses (almost) nothing, while the plain clipper clips it; white noise
//!   is clipped exactly as by the plain clipper.
//! - Bandwidth guard: 3.4 kHz band-limited noise driven 3 dB into the
//!   clipper leaves at least 30 dB less clipping energy above 4 kHz than
//!   without the guard; full-band noise is unchanged by the guard; the
//!   estimate follows programme that changes band every 2 s and does not
//!   flap on programme that alternates every 50 ms.
//! - Peak control: off (drive 0) it is the plain limiter, bit for bit; on,
//!   the true-peak ceiling holds; on a step it reduces to its static curve
//!   and no further (its gain and the fast limiter's are the smaller of
//!   the two, not their product).

use std::f64::consts::PI;

use crate::agc_support::pink;
use crate::counting_alloc::allocs;
use crate::measure_support::{channel, db, lin, lowpass, white};
use strom_dsp::bandwidth_guard::BandwidthGuard;
use strom_dsp::clipper::{FinalClipper, FinalClipperParams};
use strom_dsp::limiter::PeakControlParams;
use strom_dsp::truepeak::{TruePeakLimiter, TruePeakLimiterParams};
use strom_dsp::Processor;

const FS: f64 = 48_000.0;
/// Clipper threshold (dBFS).
const THRESHOLD_DB: f32 = -1.5;

fn clipper(adaptive: bool, guard: bool) -> FinalClipper {
    let mut c = FinalClipper::new();
    c.set_params(&FinalClipperParams {
        threshold_db: THRESHOLD_DB,
        adaptive_depth: adaptive,
        bandwidth_guard: guard,
        ..FinalClipperParams::default()
    });
    c.configure(FS as f32, 1);
    c
}

/// `x` (mono) through `c` in 480-frame buffers.
fn run(c: &mut FinalClipper, x: &[f32]) -> Vec<f32> {
    let mut y = x.to_vec();
    for b in y.chunks_mut(480) {
        c.process(b);
    }
    y
}

/// Scale `x` so its peak is `over_db` over the clipper threshold.
fn drive(x: &mut [f32], over_db: f64) {
    let p = x.iter().fold(0.0f32, |m, &v| m.max(v.abs())) as f64;
    let g = lin(THRESHOLD_DB as f64 + over_db) / p;
    x.iter_mut().for_each(|v| *v = (*v as f64 * g) as f32);
}

/// A sustained synthetic vowel: 30 harmonics of 120 Hz shaped by two
/// formant peaks (600 and 1 400 Hz), with a slow 3 Hz amplitude swell.
fn vowel(frames: usize) -> Vec<f32> {
    (0..frames)
        .map(|n| {
            let t = n as f64 / FS;
            let swell = 0.7 + 0.3 * (2.0 * PI * 3.0 * t).sin();
            let s: f64 = (1..=30)
                .map(|k| {
                    let f = 120.0 * k as f64;
                    let formant = |fc: f64, bw: f64| 1.0 / (1.0 + ((f - fc) / bw).powi(2));
                    let a = formant(600.0, 150.0) + 0.5 * formant(1_400.0, 200.0);
                    a * (2.0 * PI * f * t + 0.3 * k as f64).sin()
                })
                .sum();
            (swell * s) as f32
        })
        .collect()
}

/// Energy of what the clipper removed (delayed input − output), from
/// frame `from`, above `hp_hz` when given (8th-order high-pass).
fn removed_energy(c: &FinalClipper, x: &[f32], y: &[f32], from: usize, hp_hz: Option<f64>) -> f64 {
    let lat = c.latency_frames();
    let mut e: Vec<f32> = (0..y.len())
        .map(|n| if n >= lat { x[n - lat] - y[n] } else { 0.0 })
        .collect();
    if let Some(f) = hp_hz {
        highpass(&mut e, FS, f, 4);
    }
    e[from..].iter().map(|&v| v as f64 * v as f64).sum()
}

/// Second-order Butterworth high-pass, `passes` times (test only).
fn highpass(x: &mut [f32], fs: f64, fc: f64, passes: usize) {
    let k = (PI * fc / fs).tan();
    let q = std::f64::consts::FRAC_1_SQRT_2;
    let norm = 1.0 / (1.0 + k / q + k * k);
    let b0 = norm;
    let (b1, b2) = (-2.0 * b0, b0);
    let a1 = 2.0 * (k * k - 1.0) * norm;
    let a2 = (1.0 - k / q + k * k) * norm;
    for _ in 0..passes {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
        for v in x.iter_mut() {
            let x0 = *v as f64;
            let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;
            *v = y0 as f32;
        }
    }
}

#[test]
fn adaptive_depth_spares_predictable_material() {
    let frames = (0.6 * FS) as usize;
    let from = (0.2 * FS) as usize;
    let mut x = vowel(frames);
    drive(&mut x, 3.0);
    let mut plain = clipper(false, false);
    let mut adaptive = clipper(true, false);
    let yp = run(&mut plain, &x);
    let ya = run(&mut adaptive, &x);
    let input: f64 = x[from..].iter().map(|&v| v as f64 * v as f64).sum();
    let ep = db(removed_energy(&plain, &x, &yp, from, None).sqrt() / input.sqrt());
    let ea = db(removed_energy(&adaptive, &x, &ya, from, None).sqrt() / input.sqrt());
    println!(
        "vowel 3 dB into the clipper: removed {ep:.1} dB (plain), {ea:.1} dB (adaptive); \
         predictability {:.2}",
        adaptive.predictability()
    );
    assert!(
        ep > -40.0,
        "the plain clipper should clip the vowel: {ep:.1} dB"
    );
    assert!(
        ea < -100.0,
        "the adaptive clipper should leave the vowel: {ea:.1} dB"
    );
    assert!(adaptive.predictability() > 0.99);

    // Noise is not predictable: clipped exactly as by the plain clipper.
    // (Pink: white noise reads up to 2.6 dB after the pre-emphasis.)
    let mut n = channel(&pink(frames, 1, FS, 7), 1, 0);
    drive(&mut n, 3.0);
    let mut plain = clipper(false, false);
    let mut adaptive = clipper(true, false);
    assert_eq!(run(&mut plain, &n), run(&mut adaptive, &n));
    assert_eq!(adaptive.depth(), 1.0);
}

#[test]
fn bandwidth_guard_keeps_clippings_out_of_an_empty_band() {
    let frames = (2.5 * FS) as usize;
    let from = (1.5 * FS) as usize;
    // 3.4 kHz band-limited noise (16th-order Butterworth).
    let mut x = white(frames, 1.0, 11);
    lowpass(&mut x, FS, 3_400.0, 8);
    drive(&mut x, 3.0);
    let mut plain = clipper(false, false);
    let mut guarded = clipper(false, true);
    let yp = run(&mut plain, &x);
    let yg = run(&mut guarded, &x);
    let ep = removed_energy(&plain, &x, &yp, from, Some(4_000.0));
    let eg = removed_energy(&guarded, &x, &yg, from, Some(4_000.0));
    let gain = db((ep / eg.max(1e-30)).sqrt());
    println!(
        "3.4 kHz noise 3 dB into the clipper: clippings above 4 kHz {gain:.1} dB lower with the \
         guard (edge {:?} Hz)",
        guarded.guard_edge_hz()
    );
    assert!(ep > 0.0, "the plain clipper should clip");
    assert!(gain >= 30.0, "only {gain:.1} dB lower");
    assert!(guarded.guard_edge_hz().is_some());

    // Full-band programme: the guard changes nothing.
    let mut n = white(frames, 1.0, 12);
    drive(&mut n, 3.0);
    let mut plain = clipper(false, false);
    let mut guarded = clipper(false, true);
    assert_eq!(run(&mut plain, &n), run(&mut guarded, &n));
    assert_eq!(guarded.guard_edge_hz(), None);
}

/// Count the guard's decisions changing over `x`, in 256-frame blocks.
fn guard_changes(x: &[f32]) -> usize {
    let mut g = BandwidthGuard::new(FS);
    let mut last = None;
    let mut changes = 0;
    for b in x.chunks(256) {
        g.push(b, 1);
        g.end_block();
        let now = g.edge().is_some();
        if last.is_some_and(|l| l != now) {
            changes += 1;
        }
        last = Some(now);
    }
    changes
}

/// Alternate band-limited (3.4 kHz) and full-band noise every `period`
/// frames, at the same level.
fn alternating(frames: usize, period: usize) -> Vec<f32> {
    let full = white(frames, 0.3, 21);
    let mut narrow = white(frames, 0.3, 22);
    lowpass(&mut narrow, FS, 3_400.0, 8);
    let g = (full.iter().map(|&v| v as f64 * v as f64).sum::<f64>()
        / narrow.iter().map(|&v| v as f64 * v as f64).sum::<f64>())
    .sqrt();
    (0..frames)
        .map(|n| {
            if (n / period).is_multiple_of(2) {
                (narrow[n] as f64 * g) as f32
            } else {
                full[n]
            }
        })
        .collect()
}

#[test]
fn bandwidth_guard_follows_the_programme_without_flapping() {
    let frames = (24.0 * FS) as usize;
    // Narrow and full band every 4 s: the guard follows, one change at
    // each of the 5 switches (the first narrow stretch is found from the
    // start, before any decision).
    let slow = guard_changes(&alternating(frames, (4.0 * FS) as usize));
    // Every 50 ms: the windows see a mixture; the decision holds.
    let fast = guard_changes(&alternating(frames, (0.05 * FS) as usize));
    println!("guard changes: {slow} (4 s alternation), {fast} (50 ms alternation)");
    assert_eq!(slow, 5, "{slow} changes on 4 s alternation");
    assert!(fast <= 1, "{fast} changes on 50 ms alternation");
}

// ---------------------------------------------------------------------------
// Limiter peak control.

const CEILING_DB: f32 = -1.0;
const PC: PeakControlParams = PeakControlParams {
    headroom_db: 3.0,
    ratio: 8.0,
    knee_db: 3.0,
    attack_ms: 4.0,
    release_ms: 1_000.0,
};

fn limiter(peak_control: bool) -> TruePeakLimiter {
    let mut l = TruePeakLimiter::new();
    l.set_params(&TruePeakLimiterParams {
        enabled: true,
        ceiling_dbtp: CEILING_DB,
        release_ms: 200.0,
        lookahead_ms: 1.5,
    });
    if peak_control {
        l.set_peak_control(&PC);
    }
    l.configure(FS as f32, 1);
    l
}

#[test]
fn peak_control_at_drive_zero_is_the_plain_limiter() {
    let mut x = white((0.5 * FS) as usize, 1.0, 31);
    drive(&mut x, 6.0);
    let mut plain = limiter(false);
    let mut pc = limiter(true);
    pc.set_peak_control_drive(0.0);
    let (mut a, mut b) = (x.clone(), x.clone());
    plain.process(&mut a);
    pc.process(&mut b);
    assert_eq!(a, b);
}

#[test]
fn peak_control_keeps_the_true_peak_ceiling() {
    use ebur128::{EbuR128, Mode};
    let frames = (1.0 * FS) as usize;
    let mut x: Vec<f32> = white(frames, 1.0, 41);
    // Bursts 12 dB over the ceiling on a quieter bed, a 12 kHz tone, and
    // single-sample spikes at +20 dBFS, faster than the peak control's
    // attack: the fast limiter has to take them.
    for (n, v) in x.iter_mut().enumerate() {
        let burst = if (n / 4_800).is_multiple_of(3) {
            4.0
        } else {
            0.5
        };
        let tone = 0.5 * (2.0 * PI * 12_000.0 * n as f64 / FS).sin();
        let spike = if n % 4_801 == 2_000 { 10.0 } else { 0.0 };
        *v = (*v as f64 * burst + tone + spike) as f32;
    }
    let mut l = limiter(true);
    let mut y = x.clone();
    for b in y.chunks_mut(333) {
        l.process(b);
    }
    let mut m = EbuR128::new(1, FS as u32, Mode::TRUE_PEAK).unwrap();
    m.add_frames_f32(&y).unwrap();
    let tp = db(m.true_peak(0).unwrap());
    println!("peak control on: true peak {tp:+.3} dBTP (ceiling {CEILING_DB})");
    assert!(tp <= CEILING_DB as f64 + 0.1, "{tp:.3} dBTP");
}

/// A 1 kHz tone steps from far below to 6 dB over the ceiling. The gain
/// is the smaller of the fast limiter's and the peak control's, so the
/// reduction ends at the peak control's static curve, `S·(6 + H)`, and
/// never goes further (a peak control in series with the limiter would
/// add the two: up to 6 dB more while the fast gain releases).
#[test]
fn peak_control_reduction_is_the_smaller_gain_not_the_product() {
    let frames = (1.0 * FS) as usize;
    let step = frames / 4;
    let amp = lin(CEILING_DB as f64 + 6.0);
    let x: Vec<f32> = (0..frames)
        .map(|n| {
            let a = if n < step { amp * lin(-40.0) } else { amp };
            (a * (2.0 * PI * 1_000.0 * n as f64 / FS).sin()) as f32
        })
        .collect();
    let mut l = limiter(true);
    let mut y = x.clone();
    let mut max_gr = 0.0f64;
    let mut last = 0.0;
    for b in y.chunks_mut(48) {
        l.process(b);
        max_gr = max_gr.max(l.gain_reduction_db());
        last = l.gain_reduction_db();
    }
    let slope = 1.0 - 1.0 / PC.ratio as f64;
    let want = slope * (6.0 + PC.headroom_db as f64);
    println!(
        "step 6 dB over: reduction {last:.2} dB at the end, at most {max_gr:.2} (curve {want:.2})"
    );
    assert!((last - want).abs() < 0.3, "settles at {last:.2} dB");
    assert!(max_gr < want + 0.5, "overshoots to {max_gr:.2} dB");
}

/// The content-adaptive depth, the bandwidth guard and the peak control
/// do not allocate, on programme that drives each of them: a vowel
/// (predictable: the depth falls), 3.4 kHz band-limited noise (the guard
/// finds an edge), after full-band pink noise (full depth), all over the clipper
/// threshold, in buffers of varying size, with a parameter update before
/// each. The test checks that every path was taken, so a change that stops
/// exercising one fails here instead of passing vacuously.
#[test]
fn no_allocation_on_programme_that_drives_every_path() {
    let mut x = vowel(FS as usize);
    let mut band = white((2.5 * FS) as usize, 1.0, 21);
    lowpass(&mut band, FS, 3_400.0, 8);
    let mut full = channel(&pink(FS as usize, 1, FS, 22), 1, 0);
    drive(&mut x, 3.0);
    drive(&mut band, 3.0);
    drive(&mut full, 3.0);
    // Full band first: the guard reopens slowly after band-limited input.
    let x = [full, x, band].concat();
    let params = FinalClipperParams {
        threshold_db: THRESHOLD_DB,
        adaptive_depth: true,
        bandwidth_guard: true,
        ..FinalClipperParams::default()
    };
    let mut c = clipper(true, true);
    let mut l = limiter(true);
    let drive_ramp: Vec<f32> = (0..3_000).map(|n| (n % 1_000) as f32 / 999.0).collect();
    let sizes = [480usize, 1, 3_000, 37, 1_024];
    let (mut max_pred, mut min_depth, mut max_depth) = (0.0f64, 1.0f32, 0.0f32);
    let mut edge_seen = false;
    let mut y = x.clone();
    let mut rest = &mut y[..];
    let before = allocs();
    let mut i = 0;
    while !rest.is_empty() {
        let n = sizes[i % sizes.len()].min(rest.len());
        let (b, tail) = rest.split_at_mut(n);
        c.set_params(&params);
        c.process(b);
        l.process_with_drive(b, &drive_ramp[..n]);
        max_pred = max_pred.max(c.predictability());
        min_depth = min_depth.min(c.depth());
        max_depth = max_depth.max(c.depth());
        edge_seen |= c.guard_edge_hz().is_some();
        rest = tail;
        i += 1;
    }
    let n = allocs() - before;
    assert_eq!(n, 0, "allocations in the clipper / peak control path");
    assert!(max_pred > 0.9, "adaptive depth not exercised: {max_pred}");
    assert!(edge_seen, "bandwidth guard never found an edge");
    assert!(
        min_depth < 0.1 && max_depth > 0.9,
        "depth {min_depth} … {max_depth}"
    );
}
