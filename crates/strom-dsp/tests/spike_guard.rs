//! Tests of the spike guard ([`strom_dsp::spike_guard::SpikeGuard`]):
//! latency, bit-exactness outside events (with the event gate as the guard),
//! clicks caught and confined, crest factor of dense percussion, robustness
//! and zero allocations. Aliasing is measured in the harness crate
//! (`crates/strom-dsp-harness/tests/spike_guard.rs`). All signals are
//! synthetic with fixed seeds.

mod agc_support;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::f64::consts::PI;

use agc_support::{pink, Rng};
use strom_dsp::spike_guard::{
    latency_frames_for, SpikeGuard, SpikeGuardParams, DEFAULT_THRESHOLD_DB, EVENT_MAX_MS,
    LOW_LATENCY_LOOKAHEAD_MS, RAMP_MS, STANDARD_LOOKAHEAD_MS, THRESHOLD_RANGE_DB,
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

const FS: f32 = 48_000.0;

/// The two lookaheads of the element's latency modes.
const MODES: [(&str, f32); 2] = [
    ("standard", STANDARD_LOOKAHEAD_MS),
    ("low", LOW_LATENCY_LOOKAHEAD_MS),
];

fn guard(fs: f32, ch: usize, p: &SpikeGuardParams) -> SpikeGuard {
    guard_la(fs, ch, p, STANDARD_LOOKAHEAD_MS)
}

fn guard_la(fs: f32, ch: usize, p: &SpikeGuardParams, lookahead_ms: f32) -> SpikeGuard {
    let mut g = SpikeGuard::new();
    g.set_params(p);
    g.set_lookahead_ms(lookahead_ms);
    g.configure(fs, ch);
    g
}

/// Run `x` through `g` in buffers of `block` frames.
fn run(g: &mut SpikeGuard, x: &[f32], ch: usize, block: usize) -> Vec<f32> {
    let mut y = x.to_vec();
    for c in y.chunks_mut(block * ch) {
        g.process(c);
    }
    y
}

/// `y` is `x` delayed by `lat` frames (zeros first), bit for bit.
fn assert_delayed_bit_exact(x: &[f32], y: &[f32], ch: usize, lat: usize, what: &str) {
    for (i, v) in y.iter().enumerate() {
        let want = if i < lat * ch { 0.0 } else { x[i - lat * ch] };
        assert_eq!(
            v.to_bits(),
            want.to_bits(),
            "{what}: sample {i} (frame {}) = {v}, expected {want}",
            i / ch
        );
    }
}

fn add_tone(x: &mut [f32], ch: usize, from: usize, len: usize, f: f64, amp: f64) {
    for n in from..(from + len).min(x.len() / ch) {
        let v = (amp * (2.0 * PI * f * (n - from) as f64 / FS as f64).sin()) as f32;
        for c in 0..ch {
            x[n * ch + c] += v;
        }
    }
}

// ---------------------------------------------------------------------------
// Latency.

#[test]
fn latency_formula() {
    // 4× interpolator (24) + round(lookahead · fs).
    assert_eq!(latency_frames_for(48_000.0, STANDARD_LOOKAHEAD_MS), 24 + 24);
    assert_eq!(latency_frames_for(48_000.0, LOW_LATENCY_LOOKAHEAD_MS), 24);
    assert_eq!(latency_frames_for(44_100.0, 0.5), 24 + 22);
    assert_eq!(latency_frames_for(96_000.0, 0.5), 24 + 48);
    assert_eq!(latency_frames_for(48_000.0, 9.0), 24 + 96); // clamped to 2 ms
                                                            // At most 1 ms (standard) and 0.5 ms (low) at 48 kHz.
    assert!(latency_frames_for(48_000.0, STANDARD_LOOKAHEAD_MS) <= 48);
    assert!(latency_frames_for(48_000.0, LOW_LATENCY_LOOKAHEAD_MS) <= 24);
    for fs in [8_000.0f32, 44_100.0, 48_000.0, 96_000.0, 192_000.0] {
        for (_, la) in MODES {
            let g = guard_la(fs, 2, &SpikeGuardParams::default(), la);
            assert_eq!(g.latency_frames(), latency_frames_for(fs, la), "{fs}");
        }
    }
}

/// An impulse comes out at exactly the reported latency when disabled
/// (with the guard on, an impulse over silence is the event it removes).
#[test]
fn impulse_latency_when_disabled() {
    for fs in [44_100.0f32, 48_000.0, 96_000.0] {
        let ch = 2;
        let p = SpikeGuardParams {
            enabled: false,
            ..SpikeGuardParams::default()
        };
        let mut g = guard_la(fs, ch, &p, LOW_LATENCY_LOOKAHEAD_MS);
        let lat = g.latency_frames();
        let mut x = vec![0.0f32; (lat + 2000) * ch];
        x[500 * ch] = 0.5;
        x[500 * ch + 1] = -0.25;
        let y = run(&mut g, &x, ch, 333);
        assert_delayed_bit_exact(&x, &y, ch, lat, &format!("{fs}"));
        assert_eq!(g.take_events(), 0);
    }
}

// ---------------------------------------------------------------------------
// Bit-exact outside events (guard: the event gate).

/// A programme with large but long excursions: a sudden sustained chord
/// 20 dB over the bed, a dense drum pattern with sharp stick transients,
/// and sudden 2 kHz and 200 Hz tones 20 dB over the bed. None is a spike:
/// no event is counted, and the output is the input delayed by the latency,
/// bit for bit, except within an abandoned event's reach after each onset
/// (`EVENT_MAX_MS − lookahead` plus a ramp, where the onset is nipped). With
/// a lookahead of 2 ms an event is abandoned before the output reaches it,
/// and the output is bit-exact everywhere.
///
/// Guard: every section rises far more than the threshold above the
/// envelope (checked below), so a spike guard that capped everything above
/// `envelope + threshold` (no event gate: no "short" test) would change
/// whole sections and fail.
#[test]
fn clean_programme_is_bit_exact() {
    let ch = 2;
    let fs = FS as f64;
    let frames = 4 * FS as usize;
    let mut x = pink(1 << 18, ch, fs, 3);
    x.truncate(frames * ch);
    x.iter_mut().for_each(|v| *v *= 0.02); // bed, peaks around −20 dBFS
    let bed_peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs())) as f64;
    let ms = |t: f64| (t * 0.001 * fs) as usize;
    // The start (from silence) is an onset too.
    let mut onsets = vec![0, ms(300.0), ms(2700.0), ms(3200.0)];
    // Sustained chord at 0.3 s for 400 ms.
    for (f, a) in [(220.0, 0.25), (277.0, 0.2), (330.0, 0.2)] {
        add_tone(&mut x, ch, ms(300.0), ms(400.0), f, a);
    }
    // Dense drums from 1.0 s for 1.4 s: hits every 90 … 130 ms, each a
    // stick transient (0.5 ms at twice the burst level) on a noise burst
    // (τ 5 ms) and a 100 Hz thump (τ 30 ms).
    let mut rng = Rng::new(9);
    let mut at = ms(1000.0);
    while at < ms(2400.0) {
        onsets.push(at);
        let amp = 0.35 + 0.2 * rng.uniform();
        for k in 0..ms(150.0) {
            let t = k as f64 / fs;
            let stick = if t < 0.0005 { 2.0 } else { 1.0 };
            let burst = stick * (-t / 0.005).exp() * rng.bipolar();
            let thump = (2.0 * PI * 100.0 * t).sin() * (-t / 0.03).exp();
            for c in 0..ch {
                x[(at + k) * ch + c] += (amp * (burst + thump)) as f32;
            }
        }
        at += ms(90.0 + 40.0 * rng.uniform());
    }
    // Sudden tones: 2 kHz at 2.7 s, 200 Hz at 3.2 s, 300 ms each.
    add_tone(&mut x, ch, ms(2700.0), ms(300.0), 2_000.0, 0.5);
    add_tone(&mut x, ch, ms(3200.0), ms(300.0), 200.0, 0.5);

    // Each section rises well over the default threshold above the bed.
    let section_peak = |from: f64, to: f64| {
        x[ms(from) * ch..ms(to) * ch]
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs())) as f64
    };
    for (from, to) in [
        (300.0, 700.0),
        (1000.0, 2400.0),
        (2700.0, 3000.0),
        (3200.0, 3500.0),
    ] {
        let rise = 20.0 * (section_peak(from, to) / bed_peak).log10();
        assert!(
            rise > DEFAULT_THRESHOLD_DB as f64 + 6.0,
            "section {from} ms: rise {rise:.1} dB"
        );
    }

    for (name, la) in [MODES[0], MODES[1], ("2 ms", 2.0)] {
        // Frames after an onset (input time) an abandoned dip can reach.
        let reach = if la as f64 > EVENT_MAX_MS + 0.03 {
            0
        } else {
            ms(EVENT_MAX_MS - la as f64 + RAMP_MS + 0.2)
        };
        for block in [480usize, 1, 1024] {
            let mut g = guard_la(FS, ch, &SpikeGuardParams::default(), la);
            let lat = g.latency_frames();
            let y = run(&mut g, &x, ch, block);
            assert_eq!(g.take_events(), 0, "{name} block {block}: events counted");
            let mut nipped = 0;
            let mut max_cut_db = 0.0f64;
            for f in lat..frames {
                let n = f - lat;
                let near = onsets.iter().any(|&o| n + ms(0.6) >= o && n < o + reach);
                for c in 0..ch {
                    let (got, want) = (y[f * ch + c], x[n * ch + c]);
                    if got.to_bits() != want.to_bits() {
                        let o = onsets
                            .iter()
                            .map(|&o| n as i64 - o as i64)
                            .min_by_key(|d| d.abs());
                        assert!(
                            near,
                            "{name} block {block}: frame {f} ch {c} {got} vs {want}, \
                             nearest onset {o:?} frames away"
                        );
                        nipped += 1;
                        let cut = (want.abs() as f64 / got.abs().max(1e-12) as f64).log10() * 20.0;
                        max_cut_db = max_cut_db.max(cut);
                    }
                }
            }
            if block == 480 {
                println!(
                    "{name}: {nipped} samples changed at onsets, largest cut {max_cut_db:.1} dB"
                );
            }
            if reach == 0 {
                assert_eq!(nipped, 0, "{name}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Events.

/// Single-sample clicks over a steady tone are caught, one event each; the
/// output peak near each click is close to the clip level (envelope +
/// threshold); away from the clicks (beyond ±3 ms) the output is the
/// delayed input, bit for bit.
#[test]
fn clicks_are_caught_and_confined() {
    let ch = 2;
    let frames = 2 * FS as usize;
    let amp = 0.1;
    let mut x = vec![0.0f32; frames * ch];
    add_tone(&mut x, ch, 0, frames, 440.0, amp);
    let spacing = 2400; // 50 ms
    let clicks: Vec<usize> = (4800..frames - 4800).step_by(spacing).collect();
    for (i, &n) in clicks.iter().enumerate() {
        let a = if i % 2 == 0 { 1.0 } else { -0.8 };
        x[n * ch] += a;
        if i % 3 != 0 {
            x[n * ch + 1] += a; // some clicks on one channel only
        }
    }
    let p = SpikeGuardParams::default();
    for (name, la) in MODES {
        let mut g = guard_la(FS, ch, &p, la);
        let lat = g.latency_frames();
        let y = run(&mut g, &x, ch, 480);
        assert_eq!(g.take_events() as usize, clicks.len(), "{name}");
        let level = amp * 10f64.powf(p.threshold_db as f64 / 20.0);
        let w = (0.003 * FS as f64) as usize;
        for &n in &clicks {
            let out = n + lat;
            let peak = y[(out - w) * ch..(out + w) * ch]
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs())) as f64;
            // The gain dip scales the click to the cap; the tone under it adds
            // at most its own amplitude scaled by the dip.
            assert!(
                peak <= level * 1.05,
                "{name}: click at {n}: output peak {peak:.3}, clip level {level:.3}"
            );
            assert!(peak >= amp * 0.9, "click at {n}: tone gone ({peak})");
        }
        // Bit-exact outside ±3 ms of the clicks (after the first 3 ms: with
        // no lookahead the tone's start from silence is an onset, nipped).
        for f in w + lat..frames {
            let near = clicks.iter().any(|&n| f + w >= n + lat && f < n + lat + w);
            if near {
                continue;
            }
            for c in 0..ch {
                let want = if f < lat { 0.0 } else { x[(f - lat) * ch + c] };
                assert_eq!(
                    y[f * ch + c].to_bits(),
                    want.to_bits(),
                    "{name}: frame {f} ch {c}"
                );
            }
        }
    }
}

/// Crest factor (sample peak / RMS, dB) of `x`.
fn crest_db(x: &[f32]) -> f64 {
    let peak = x.iter().fold(0.0f64, |m, &v| m.max(v.abs() as f64));
    let rms = (x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
    20.0 * (peak / rms).log10()
}

/// A dense percussive signal at programme level (hits every 60 … 100 ms,
/// sharp stick transients, no bed) loses at most 1 dB of crest factor with
/// the standard lookahead, at the default threshold and at the lowest one.
#[test]
fn dense_percussion_keeps_crest_factor() {
    let ch = 2;
    let fs = FS as f64;
    let frames = 3 * FS as usize;
    let mut x = vec![0.0f32; frames * ch];
    let mut rng = Rng::new(21);
    let mut at = 1000;
    while at + 7200 < frames {
        let amp = 0.2 + 0.4 * rng.uniform();
        let decay = 0.003 + 0.01 * rng.uniform();
        for k in 0..7200 {
            let t = k as f64 / fs;
            let stick = if t < 0.0005 { 2.5 } else { 1.0 };
            let v = amp
                * (stick * (-t / decay).exp() * rng.bipolar()
                    + 0.6 * (2.0 * PI * 70.0 * t).sin() * (-t / 0.04).exp());
            for c in 0..ch {
                x[(at + k) * ch + c] += v as f32;
            }
        }
        at += (0.06 * fs + 0.04 * fs * rng.uniform()) as usize;
    }
    // Standard lookahead: within 1 dB (the requirement). No lookahead: the
    // hits' onsets lose up to 0.4 ms to the cap; the loss is reported and
    // bounded at 3 dB so it cannot grow unnoticed.
    for (name, la, bound) in [(MODES[0].0, MODES[0].1, 1.0), (MODES[1].0, MODES[1].1, 3.0)] {
        for threshold_db in [DEFAULT_THRESHOLD_DB, THRESHOLD_RANGE_DB.0] {
            let p = SpikeGuardParams {
                enabled: true,
                threshold_db,
            };
            let mut g = guard_la(FS, ch, &p, la);
            let lat = g.latency_frames();
            let y = run(&mut g, &x, ch, 480);
            let loss = crest_db(&x) - crest_db(&y[lat * ch..]);
            println!(
                "{name}, threshold {threshold_db} dB: crest {:.2} → {:.2} dB (loss {loss:.2}), \
                 {} events",
                crest_db(&x),
                crest_db(&y[lat * ch..]),
                g.take_events()
            );
            assert!(
                loss <= bound,
                "{name}, threshold {threshold_db}: crest loss {loss:.2} dB"
            );
        }
    }
}

/// A threshold change applies from the next event; an event in progress
/// keeps its level, so the output never steps. Toggling `enabled` stops new
/// events, and the output returns to the delayed input.
#[test]
fn parameter_changes_take_effect_between_events() {
    let ch = 1;
    let frames = FS as usize;
    let mut x = vec![0.0f32; frames];
    add_tone(&mut x, ch, 0, frames, 300.0, 0.1);
    let clicks: Vec<usize> = (2400..frames - 2400).step_by(1200).collect();
    for &n in &clicks {
        x[n] += 1.0;
    }
    let mut g = guard(FS, ch, &SpikeGuardParams::default());
    let lat = g.latency_frames();
    let mut y = x.clone();
    let mut rng = Rng::new(4);
    let mut pos = 0;
    let mut events = 0;
    while pos < frames {
        let n = (1 + (rng.uniform() * 300.0) as usize).min(frames - pos);
        g.set_params(&SpikeGuardParams {
            enabled: pos < frames / 2,
            threshold_db: (6.0 + 24.0 * rng.uniform()) as f32,
        });
        g.process(&mut y[pos..pos + n]);
        events += g.take_events();
        pos += n;
    }
    assert!(events > 0 && events as usize <= clicks.len());
    for (i, v) in y.iter().enumerate() {
        assert!(v.is_finite() && v.abs() <= 1.2, "{i}: {v}");
    }
    // Disabled from the middle on (after the last event drains): exact.
    let from = frames / 2 + lat + 2400;
    for f in from..frames {
        assert_eq!(y[f].to_bits(), x[f - lat].to_bits(), "frame {f}");
    }
}

// ---------------------------------------------------------------------------
// Robustness and real-time safety.

#[test]
fn params_are_clamped() {
    let mut g = SpikeGuard::new();
    for (v, want) in [
        (f32::NAN, DEFAULT_THRESHOLD_DB),
        (-5.0, THRESHOLD_RANGE_DB.0),
        (99.0, THRESHOLD_RANGE_DB.1),
        (12.0, 12.0),
    ] {
        g.set_params(&SpikeGuardParams {
            enabled: true,
            threshold_db: v,
        });
        assert_eq!(g.params().threshold_db, want);
    }
}

/// Loud, tiny, silent and non-finite-free random input with random
/// parameters every buffer, odd buffer sizes, several rates and channel
/// counts: finite output, and no allocation after `configure`.
#[test]
fn robust_and_allocation_free() {
    let mut rng = Rng::new(77);
    for (fs, ch) in [
        (8_000.0f32, 1usize),
        (44_100.0, 2),
        (48_000.0, 2),
        (96_000.0, 6),
        (192_000.0, 2),
    ] {
        let mut g = guard(fs, ch, &SpikeGuardParams::default());
        let sizes = [1usize, 480, 7, 2000, 64, 513, 256];
        let mut buf = vec![0.0f32; 2000 * ch];
        let before = allocs();
        for i in 0..150 {
            let n = sizes[i % sizes.len()] * ch;
            let level = match i % 5 {
                0 => 30.0,
                1 => 1e-30,
                2 => 0.0,
                3 => 0.3,
                _ => 1.0,
            };
            for (k, v) in buf[..n].iter_mut().enumerate() {
                let click = if k % 97 == 0 { 20.0 } else { 1.0 };
                *v = (level * click * rng.bipolar()) as f32;
            }
            g.set_params(&SpikeGuardParams {
                enabled: rng.uniform() > 0.2,
                threshold_db: (rng.uniform() * 40.0 - 5.0) as f32,
            });
            g.process(&mut buf[..n]);
            assert!(buf[..n].iter().all(|v| v.is_finite()), "fs {fs} ch {ch}");
            let _ = g.take_events();
            if i % 60 == 59 {
                g.reset();
            }
        }
        assert_eq!(allocs() - before, 0, "allocations at fs {fs}, {ch} ch");
    }
    // Unconfigured / zero channels: no-ops.
    let mut g = SpikeGuard::new();
    let mut buf = [0.5f32; 64];
    g.process(&mut buf);
    assert!(buf.iter().all(|&v| v == 0.5));
    g.configure(48_000.0, 0);
    g.process(&mut buf);
    assert!(buf.iter().all(|&v| v == 0.5));
}

/// After `reset`, the guard behaves exactly like a fresh one.
#[test]
fn reset_matches_fresh() {
    let ch = 2;
    let mut x = pink(1 << 15, ch, FS as f64, 8);
    x.iter_mut().for_each(|v| *v *= 0.05);
    for n in (1000..(1 << 15)).step_by(3000) {
        x[n * ch] += 0.9;
    }
    let mut a = guard(FS, ch, &SpikeGuardParams::default());
    let _ = run(&mut a, &x, ch, 480);
    a.reset();
    let ya = run(&mut a, &x, ch, 480);
    let mut b = guard(FS, ch, &SpikeGuardParams::default());
    let yb = run(&mut b, &x, ch, 480);
    assert_eq!(ya, yb);
    assert!(a.take_events() > 0);
}
