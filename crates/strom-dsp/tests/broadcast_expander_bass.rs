//! The per-band expander and the bass clipper inside the broadcast chain
//! ([`strom_dsp::broadcast::BroadcastProcessor`]).
//!
//! - Latency: an impulse comes out exactly at the reported latency with the
//!   bass clipper on, and with the multiband stage bypassed.
//! - Meters: expander and bass-clip gain reductions are positive while they
//!   work and read 0 with the multiband stage off.
//! - Noise between words: speech-like bursts with noise between them, the
//!   `speech` preset against the same preset with the expanders off: the
//!   noise between the bursts is lower by close to the floor, the bursts
//!   keep their level within 0.5 dB.
//! - Bypass of the multiband stage stays bit-exact after the fade, with
//!   the expander and bass clipper active; no allocation.

mod measure_support;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use measure_support::{db, lin, rms, sine, speech_like_bursts, to_channels, Bursts};
use strom_dsp::broadcast::{BandExpanderParams, BroadcastParams, BroadcastProcessor};
use strom_dsp::multiband::MAX_BANDS;
use strom_dsp::presets::BroadcastPreset;
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

const FS: f32 = 48_000.0;

/// Flat chain (AGC off, no compression, limiter ceiling 0 dBTP) with the
/// bass clipper on at 0 dB, so a small impulse passes untouched.
fn flat_with_bass() -> BroadcastParams {
    let mut p = BroadcastParams {
        // An impulse over silence is exactly what the spike guard removes.
        spike_guard_enabled: false,
        agc_enabled: false,
        true_peak_ceiling_dbtp: 0.0,
        bass_clipper_enabled: true,
        bass_clip_threshold_db: 0.0,
        ..BroadcastParams::default()
    };
    for b in &mut p.bands {
        b.ratio = 1.0;
    }
    p
}

fn processor(p: &BroadcastParams, ch: usize) -> BroadcastProcessor {
    let mut bp = BroadcastProcessor::new();
    bp.set_params(p);
    bp.configure(FS, ch);
    bp
}

/// An impulse comes out exactly at the reported latency (spike guard + bass
/// lookahead + clipper + limiter lookahead + detector delay), with the
/// multiband on and bypassed, at bass lookaheads of 0, 1 (standard) and
/// 3 ms.
#[test]
fn impulse_at_reported_latency() {
    for (bass_ms, bass_frames) in [(0.0f32, 0usize), (1.0, 48), (3.0, 144)] {
        for multiband in [true, false] {
            let p = BroadcastParams {
                multiband_enabled: multiband,
                bass_clipper_lookahead_ms: bass_ms,
                ..flat_with_bass()
            };
            let mut bp = processor(&p, 1);
            let lat = bp.latency_frames();
            assert_eq!(lat, 48 + bass_frames + 69 + 72 + 32);
            let mut x = vec![0.0f32; lat + 4096];
            x[1000] = 0.25;
            for c in x.chunks_mut(333) {
                bp.process(c);
            }
            // The crossover is an all-pass: its response starts at sample 0.
            let at = x.iter().position(|&v| v != 0.0).unwrap();
            assert_eq!(at, 1000 + lat, "multiband {multiband}, bass {bass_ms} ms");
            if !multiband {
                assert_eq!(x[at], 0.25);
                assert!(x.iter().filter(|&&v| v != 0.0).count() == 1);
            }
        }
    }
}

/// The meters show the expanders working on quiet noise and the bass
/// clipper on a loud bass; both read 0 with the multiband stage off.
#[test]
fn meters_report_expander_and_bass_clip() {
    let mut p = flat_with_bass();
    p.bass_clip_threshold_db = -20.0;
    p.band_expanders = [BandExpanderParams {
        threshold_db: -30.0,
        ratio: 3.0,
        floor_db: 10.0,
    }; MAX_BANDS];
    let mut bp = processor(&p, 2);
    let quiet = to_channels(&measure_support::white(48_000, 0.001, 3), 2);
    let mut x = quiet.clone();
    bp.process(&mut x);
    let m = bp.take_meters();
    for k in 0..4 {
        assert!(
            (m.band_expander_gain_reduction_db[k] - 10.0).abs() < 0.1,
            "band {k}: {}",
            m.band_expander_gain_reduction_db[k]
        );
    }
    assert_eq!(m.band_expander_gain_reduction_db[4], 0.0, "inactive band");
    let mut bass = to_channels(&sine(48_000, FS as f64, 60.0, 0.5), 2);
    bp.process(&mut bass);
    let m = bp.take_meters();
    assert!(
        m.bass_clip_gain_reduction_db > 10.0,
        "bass clip {}",
        m.bass_clip_gain_reduction_db
    );
    p.multiband_enabled = false;
    bp.set_params(&p);
    let mut bass = to_channels(&sine(48_000, FS as f64, 60.0, 0.5), 2);
    bp.process(&mut bass);
    let m = bp.take_meters();
    assert_eq!(m.bass_clip_gain_reduction_db, 0.0);
    assert!(m.band_expander_gain_reduction_db.iter().all(|&v| v == 0.0));
}

/// Speech-like bursts (3.4 kHz band-limited noise, 300 ms on, 700 ms off)
/// at about −23 LUFS with noise 40 dB below them between the bursts,
/// through the `speech` preset (4 bands): against the same preset with the
/// expanders off, the noise in the last 150 ms of each gap is at least 4 dB
/// lower (the preset's floor is 6 dB, the knee and the noise peaks take
/// some), and every burst from 20 ms after its onset keeps its level within
/// 0.5 dB.
#[test]
fn speech_preset_lowers_noise_between_words() {
    let fs = FS as f64;
    let frames = 20 * FS as usize;
    let (mono, bursts) = speech_like_bursts(
        frames,
        fs,
        &Bursts {
            bandwidth: 3400.0,
            burst_s: 0.3,
            gap_s: 0.7,
            burst_rms: lin(-20.0),
            floor_rms: lin(-60.0),
            seed: 11,
        },
    );
    let x = to_channels(&mono, 2);
    let on = BroadcastParams::preset(BroadcastPreset::Speech, 4);
    let mut off = on;
    for e in &mut off.band_expanders {
        e.ratio = 1.0;
    }
    let run = |p: &BroadcastParams| {
        let mut bp = processor(p, 2);
        let mut y = x.clone();
        for c in y.chunks_mut(960) {
            bp.process(c);
        }
        measure_support::channel(&y, 2, 0)
    };
    let a = run(&on);
    let b = run(&off);
    let lat = BroadcastProcessor::latency_frames_for(FS, 5.0);
    let (mut gap_min, mut burst_worst) = (f64::INFINITY, 0.0f64);
    for &(s, end) in bursts
        .iter()
        .filter(|&&(s, e)| s > 10 * FS as usize && e + lat < frames)
    {
        let r = s + lat + (0.02 * fs) as usize..end + lat;
        let burst = db(rms(&a[r.clone()]) / rms(&b[r]));
        burst_worst = burst_worst.max(burst.abs());
        let g = s + lat - (0.15 * fs) as usize..s + lat;
        let gap = db(rms(&b[g.clone()]) / rms(&a[g]));
        gap_min = gap_min.min(gap);
    }
    eprintln!(
        "speech preset: noise between bursts lowered by at least {gap_min:.2} dB, bursts \
         changed by at most {burst_worst:.3} dB"
    );
    assert!(gap_min >= 4.0, "noise lowered by only {gap_min} dB");
    assert!(burst_worst < 0.5, "bursts changed by {burst_worst} dB");
}

/// With expanders and bass clipper working, bypassing the multiband stage
/// gives, after the fade, the chain with the multiband's input delayed by
/// its latency: the same as a chain whose multiband never ran. Checked as:
/// output with `enabled = false` is the input delayed by the latency, bit
/// for bit, and processing allocates nothing.
#[test]
fn bypass_exact_and_no_allocation() {
    let mut p = BroadcastParams::preset(BroadcastPreset::Streaming, 4);
    let mut bp = processor(&p, 2);
    let lat = bp.latency_frames();
    let src = to_channels(&sine(96_000, FS as f64, 55.0, 0.7), 2);
    let mut warm = src.clone();
    for c in warm.chunks_mut(480) {
        bp.process(c);
    }
    let mut y = src.clone();
    let before = allocs();
    p.enabled = false;
    bp.set_params(&p);
    for c in y.chunks_mut(480) {
        bp.process(c);
    }
    assert_eq!(allocs() - before, 0);
    // After the 10 ms fade (and the latency), the output is the input delayed.
    let start = lat + 960;
    for n in start..96_000 {
        for c in 0..2 {
            assert_eq!(y[n * 2 + c], src[(n - lat) * 2 + c], "frame {n}");
        }
    }
}
