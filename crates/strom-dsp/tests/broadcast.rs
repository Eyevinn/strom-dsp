//! Tests of the broadcast chain ([`strom_dsp::broadcast::BroadcastProcessor`]),
//! its meters, latency and loudness acceptance. All signals are
//! synthetic with fixed seeds.
//!
//! - Latency: an impulse through a flat configuration comes out exactly at
//!   the reported latency, also with each stage bypassed and with
//!   `enabled = false`, at several rates and lookaheads.
//! - Bypass: toggling each stage and `enabled` is click-free (sinusoid
//!   annihilator, bound −50 dB as in the plugin's `tests/knobs.rs`); after
//!   the fade the output is the input delayed by the latency, bit for bit.
//! - Meters: sign conventions, silence reads −200, agreement with
//!   `meter::LoudnessMeter` (`ebur128`) on the same signal.
//! - Robustness and zero allocations.
//! - Acceptance (60 s programme per preset at −35/−23/−10 LUFS): see
//!   [`acceptance`], using the presets of `strom_dsp::presets`: integrated
//!   loudness (from 10 s) within ±1 LU of the target, true peak ≤ ceiling +
//!   0.1 dB, both measured with `ebur128`.

mod agc_support;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::f64::consts::PI;

use agc_support::{pink, scale_to_lufs, Looper, Rng};
use strom_dsp::broadcast::{BroadcastMeters, BroadcastParams, BroadcastProcessor, SILENCE_DB};
use strom_dsp::meter::{ChannelLayout, LoudnessConfig, LoudnessMeter};
use strom_dsp::multiband::MAX_BANDS;
use strom_dsp::Processor;

// ---------------------------------------------------------------------------
// Counting allocator (per thread, so parallel tests don't interfere).

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

// ---------------------------------------------------------------------------
// Helpers.

const FS: f32 = 48_000.0;
/// Frames per buffer (10 ms at 48 kHz).
const BLOCK: usize = 480;

/// A configuration whose wet path is exactly a delay for small signals:
/// one band at ratio 1 (the crossover passes one band bit for bit), drive
/// and input gain 0 dB; the AGC holds at 0 dB below its gate, and the
/// limiter does not act below its ceiling.
fn flat_params() -> BroadcastParams {
    let mut p = BroadcastParams {
        num_bands: 1,
        ..BroadcastParams::default()
    };
    for b in &mut p.bands {
        b.ratio = 1.0;
        b.makeup_db = 0.0;
    }
    p
}

fn processor(fs: f32, ch: usize, p: &BroadcastParams) -> BroadcastProcessor {
    let mut bp = BroadcastProcessor::new();
    bp.set_params(p);
    bp.configure(fs, ch);
    bp
}

/// Run `input` through `bp` in buffers of `block` frames.
fn run(bp: &mut BroadcastProcessor, input: &[f32], ch: usize, block: usize) -> Vec<f32> {
    let mut out = input.to_vec();
    for c in out.chunks_mut(block * ch) {
        bp.process(c);
    }
    out
}

/// Interleaved sine of `amp` at `freq` on every channel.
fn sine(frames: usize, ch: usize, fs: f64, freq: f64, amp: f64) -> Vec<f32> {
    (0..frames)
        .flat_map(|n| {
            let v = (amp * (2.0 * PI * freq * n as f64 / fs).sin()) as f32;
            std::iter::repeat_n(v, ch)
        })
        .collect()
}

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-30).log10()
}

// ---------------------------------------------------------------------------
// Latency.

/// Which stage(s) to bypass in a latency or bypass scenario.
#[derive(Debug, Clone, Copy)]
enum Bypass {
    None,
    Agc,
    Multiband,
    Limiter,
    AllStages,
    Chain,
}

impl Bypass {
    const ALL: [Bypass; 6] = [
        Bypass::None,
        Bypass::Agc,
        Bypass::Multiband,
        Bypass::Limiter,
        Bypass::AllStages,
        Bypass::Chain,
    ];

    fn apply(self, p: &mut BroadcastParams, on: bool) {
        match self {
            Bypass::None => {}
            Bypass::Agc => p.agc_enabled = on,
            Bypass::Multiband => p.multiband_enabled = on,
            Bypass::Limiter => p.limiter_enabled = on,
            Bypass::AllStages => {
                p.agc_enabled = on;
                p.multiband_enabled = on;
                p.limiter_enabled = on;
            }
            Bypass::Chain => p.enabled = on,
        }
    }
}

/// An impulse comes out exactly at the reported latency, whatever is
/// bypassed, and the reported latency does not depend on bypass.
#[test]
fn impulse_latency_is_reported_latency() {
    for fs in [44_100.0f32, 48_000.0, 96_000.0] {
        for lookahead in [0.5f32, 5.0, 20.0] {
            for by in Bypass::ALL {
                let mut p = flat_params();
                p.limiter_lookahead_ms = lookahead;
                by.apply(&mut p, false);
                let ch = 2;
                let mut bp = processor(fs, ch, &p);
                let lat = bp.latency_frames();
                assert_eq!(lat, BroadcastProcessor::latency_frames_for(fs, lookahead));
                // Same latency as the all-enabled configuration.
                let mut q = p;
                q.enabled = true;
                q.agc_enabled = true;
                q.multiband_enabled = true;
                q.limiter_enabled = true;
                bp.set_params(&q);
                assert_eq!(bp.latency_frames(), lat);
                bp.set_params(&p);

                let frames = lat + 4096;
                let at = 1000;
                let mut x = vec![0.0f32; frames * ch];
                x[at * ch] = 0.5;
                x[at * ch + 1] = -0.25;
                let y = run(&mut bp, &x, ch, 333);
                for (n, f) in y.chunks(ch).enumerate() {
                    let want: [f32; 2] = if n == at + lat {
                        [0.5, -0.25]
                    } else {
                        [0.0, 0.0]
                    };
                    for c in 0..ch {
                        assert!(
                            (f[c] - want[c]).abs() < 1e-6,
                            "fs {fs} lookahead {lookahead} {by:?}: frame {n} ch {c} = {} \
                             (impulse expected at {})",
                            f[c],
                            at + lat
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn latency_formula() {
    // Before configure: usable, and the same as after.
    for (fs, la) in [(48_000.0f32, 5.0f32), (44_100.0, 0.5), (96_000.0, 20.0)] {
        let mut p = BroadcastParams {
            limiter_lookahead_ms: la,
            ..BroadcastParams::default()
        };
        let mut bp = BroadcastProcessor::new();
        bp.set_params(&p);
        let want = BroadcastProcessor::latency_frames_for(fs, la);
        assert!(want >= (la as f64 * 0.001 * fs as f64).floor() as usize);
        bp.configure(fs, 2);
        assert_eq!(bp.latency_frames(), want);
        // The lookahead is structural: a change applies at the next configure.
        p.limiter_lookahead_ms = 10.0;
        bp.set_params(&p);
        assert_eq!(bp.latency_frames(), want);
        bp.configure(fs, 2);
        assert_eq!(
            bp.latency_frames(),
            BroadcastProcessor::latency_frames_for(fs, 10.0)
        );
    }
    // Lookahead plus the true-peak detector delay (24 + 8 frames).
    assert_eq!(
        BroadcastProcessor::latency_frames_for(48_000.0, 5.0),
        240 + 32
    );
    assert_eq!(
        BroadcastProcessor::latency_frames_for(44_100.0, 5.0),
        221 + 32
    );
    assert_eq!(
        BroadcastProcessor::latency_frames_for(96_000.0, 5.0),
        480 + 32
    );
    // Out-of-range lookaheads are clamped to 0.5 … 20 ms.
    assert_eq!(
        BroadcastProcessor::latency_frames_for(48_000.0, 0.0),
        BroadcastProcessor::latency_frames_for(48_000.0, 0.5)
    );
    assert_eq!(
        BroadcastProcessor::latency_frames_for(48_000.0, 100.0),
        BroadcastProcessor::latency_frames_for(48_000.0, 20.0)
    );
    assert_eq!(
        BroadcastProcessor::latency_frames_for(48_000.0, f32::NAN),
        BroadcastProcessor::latency_frames_for(48_000.0, 5.0)
    );
}

// ---------------------------------------------------------------------------
// Bypass toggles.

/// Peak of the sinusoid annihilator `r[n] = y[n] − 2cos(ω)y[n−1] + y[n−2]`
/// over frames `range`, per channel, maximum over channels.
fn annihilator_peak(y: &[f32], ch: usize, w: f64, range: std::ops::Range<usize>) -> f64 {
    let k = 2.0 * w.cos();
    let mut peak = 0.0f64;
    for n in range.start.max(2)..range.end {
        for c in 0..ch {
            let r =
                y[n * ch + c] as f64 - k * y[(n - 1) * ch + c] as f64 + y[(n - 2) * ch + c] as f64;
            peak = peak.max(r.abs());
        }
    }
    peak
}

fn abs_peak(y: &[f32], ch: usize, range: std::ops::Range<usize>) -> f64 {
    y[range.start * ch..range.end * ch]
        .iter()
        .fold(0.0f64, |a, &v| a.max(v.abs() as f64))
}

/// An active configuration: default 4 bands compressing a −12 dBFS tone,
/// the limiter limiting (ceiling −18 dBTP), the AGC riding.
fn active_params() -> BroadcastParams {
    BroadcastParams {
        true_peak_ceiling_dbtp: -18.0,
        ..BroadcastParams::default()
    }
}

/// Each toggle (off, then on again) is click-free; while switched off the
/// output settles to the bypassed signal.
#[test]
fn bypass_toggles_are_click_free() {
    const HARD_LIMIT_DB: f64 = -50.0;
    let ch = 2;
    let freq = 1000.0;
    let w = 2.0 * PI * freq / FS as f64;
    let toggles = [96_000usize, 120_000]; // off at 2.0 s, on at 2.5 s
    let total = 150_000;
    let x = sine(total, ch, FS as f64, freq, 0.25);
    for by in Bypass::ALL.into_iter().skip(1) {
        let base = active_params();
        let mut bp = processor(FS, ch, &base);
        let mut y = x.clone();
        let mut start = 0;
        for (i, &t) in toggles.iter().enumerate() {
            for c in y[start * ch..t * ch].chunks_mut(BLOCK * ch) {
                bp.process(c);
            }
            let mut p = base;
            by.apply(&mut p, i == 1);
            bp.set_params(&p);
            start = t;
        }
        for c in y[start * ch..].chunks_mut(BLOCK * ch) {
            bp.process(c);
        }
        for &t in &toggles {
            let settled =
                abs_peak(&y, ch, t - 1920..t - 960).max(abs_peak(&y, ch, t + 19_000..t + 20_000));
            let r = annihilator_peak(&y, ch, w, t - 960..t + 9600);
            let rel = db(r / settled);
            assert!(
                rel < HARD_LIMIT_DB,
                "{by:?} toggle at {t}: residual {rel:.1} dB (settled amplitude {settled})"
            );
        }
    }
}

/// After switching off (chain or all stages), the output is the input
/// delayed by exactly the latency, bit for bit; switching back on returns
/// to the processed signal.
#[test]
fn bypass_is_bit_exact_after_fade() {
    let ch = 2;
    let mut rng = Rng::new(7);
    let total = 96_000;
    let x: Vec<f32> = (0..total * ch)
        .map(|_| (0.7 * rng.bipolar()) as f32)
        .collect();
    for by in [Bypass::Chain, Bypass::AllStages] {
        let mut p = active_params();
        p.input_gain_db = 0.0;
        let mut bp = processor(FS, ch, &p);
        let lat = bp.latency_frames();
        let mut y = x.clone();
        let off = 24_000;
        for c in y[..off * ch].chunks_mut(BLOCK * ch) {
            bp.process(c);
        }
        by.apply(&mut p, false);
        bp.set_params(&p);
        for c in y[off * ch..].chunks_mut(BLOCK * ch) {
            bp.process(c);
        }
        // Fade: 10 ms (480 frames); check from 20 ms after the switch.
        let from = off + 960;
        for n in from..total {
            for c in 0..ch {
                assert_eq!(
                    y[n * ch + c].to_bits(),
                    x[(n - lat) * ch + c].to_bits(),
                    "{by:?}: frame {n} ch {c}"
                );
            }
        }
    }
}

/// With the whole chain switched off, input gain is bypassed too (unity).
#[test]
fn disabled_chain_ignores_input_gain() {
    let ch = 1;
    let mut p = flat_params();
    p.enabled = false;
    p.input_gain_db = 12.0;
    let mut bp = processor(FS, ch, &p);
    let lat = bp.latency_frames();
    let x = sine(10_000, ch, FS as f64, 440.0, 0.5);
    let y = run(&mut bp, &x, ch, BLOCK);
    for n in lat..10_000 {
        assert_eq!(y[n].to_bits(), x[n - lat].to_bits());
    }
}

/// The input gain glides (no click) and settles at the set value.
#[test]
fn input_gain_change_is_click_free() {
    let ch = 2;
    let freq = 1000.0;
    let w = 2.0 * PI * freq / FS as f64;
    let mut p = flat_params();
    p.agc_enabled = false;
    p.limiter_enabled = false;
    let x = sine(48_000, ch, FS as f64, freq, 0.05);
    let mut bp = processor(FS, ch, &p);
    let mut y = x.clone();
    let t = 24_000;
    for c in y[..t * ch].chunks_mut(BLOCK * ch) {
        bp.process(c);
    }
    p.input_gain_db = 24.0;
    bp.set_params(&p);
    for c in y[t * ch..].chunks_mut(BLOCK * ch) {
        bp.process(c);
    }
    let after = abs_peak(&y, ch, 40_000..48_000);
    let before = abs_peak(&y, ch, 20_000..23_000);
    assert!(
        (db(after / before) - 24.0).abs() < 0.05,
        "{}",
        db(after / before)
    );
    let r = annihilator_peak(&y, ch, w, t - 960..t + 9600);
    assert!(db(r / after) < -50.0, "residual {:.1} dB", db(r / after));
}

// ---------------------------------------------------------------------------
// Meters.

fn assert_no_nan(m: &BroadcastMeters) {
    let all = [
        m.input_momentary_lufs,
        m.input_short_term_lufs,
        m.output_momentary_lufs,
        m.output_short_term_lufs,
        m.output_integrated_lufs,
        m.input_true_peak_dbtp,
        m.output_true_peak_dbtp,
        m.agc_gain_db,
        m.limiter_gain_reduction_db,
    ];
    for v in all.iter().chain(&m.band_gain_reduction_db) {
        assert!(v.is_finite() && *v >= SILENCE_DB, "{m:?}");
    }
    assert!(m.limiter_gain_reduction_db >= 0.0);
    assert!(m.band_gain_reduction_db.iter().all(|&v| v >= 0.0));
}

#[test]
fn meters_read_silence() {
    // Unconfigured.
    let mut bp = BroadcastProcessor::new();
    assert_eq!(bp.take_meters(), BroadcastMeters::default());
    // Configured, nothing processed, then digital silence.
    let ch = 2;
    let mut bp = processor(FS, ch, &BroadcastParams::default());
    let silent = |m: &BroadcastMeters| {
        assert_no_nan(m);
        for v in [
            m.input_momentary_lufs,
            m.input_short_term_lufs,
            m.output_momentary_lufs,
            m.output_short_term_lufs,
            m.output_integrated_lufs,
            m.input_true_peak_dbtp,
            m.output_true_peak_dbtp,
        ] {
            assert_eq!(v, SILENCE_DB, "{m:?}");
        }
        assert_eq!(m.agc_gain_db, 0.0);
        assert_eq!(m.limiter_gain_reduction_db, 0.0);
        assert!(m.band_gain_reduction_db.iter().all(|&v| v == 0.0));
    };
    silent(&bp.take_meters());
    let mut buf = vec![0.0f32; BLOCK * ch];
    for _ in 0..400 {
        bp.process(&mut buf);
        silent(&bp.take_meters());
    }
}

/// Signs: a quiet programme is boosted (AGC gain > 0), a loud one cut
/// (< 0); gain reductions are positive; the true peak is held until read.
#[test]
fn meter_sign_conventions() {
    let ch = 2;
    let src = pink(1 << 16, ch, FS as f64, 3);
    for (lufs, boost) in [(-33.0, true), (-13.0, false)] {
        let mut x = src.clone();
        scale_to_lufs(&mut x, ch, FS as f64, lufs);
        // Thresholds and ceiling low enough that every band and the
        // limiter work on the levelled (−23 LUFS) programme.
        let mut p = BroadcastParams {
            agc_rate_up_db_s: 10.0,
            agc_rate_down_db_s: 10.0,
            true_peak_ceiling_dbtp: -20.0,
            ..BroadcastParams::default()
        };
        for b in &mut p.bands {
            b.threshold_db = -40.0;
        }
        let mut bp = processor(FS, ch, &p);
        let mut looper = Looper::new(&x);
        let mut buf = vec![0.0f32; BLOCK * ch];
        let mut last = BroadcastMeters::default();
        for _ in 0..600 {
            looper.fill(&mut buf);
            bp.process(&mut buf);
            last = bp.take_meters();
            assert_no_nan(&last);
        }
        if boost {
            assert!(last.agc_gain_db > 5.0, "{last:?}");
        } else {
            assert!(last.agc_gain_db < -5.0, "{last:?}");
            // Every band compresses, and the limiter works.
            assert!(last.limiter_gain_reduction_db > 0.0, "{last:?}");
            assert!(
                last.band_gain_reduction_db[..4].iter().all(|&v| v > 0.0),
                "{last:?}"
            );
            assert!(last.band_gain_reduction_db[4] == 0.0);
        }
        assert!(last.output_true_peak_dbtp > -40.0);
        // Peak hold: after a read and silence, the next read is silence.
        // (The buffer is refilled every time: `process` works in place.)
        for _ in 0..11 {
            buf.fill(0.0);
            bp.process(&mut buf);
            let _ = bp.take_meters();
        }
        buf.fill(0.0);
        bp.process(&mut buf);
        let m = bp.take_meters();
        assert_eq!(m.input_true_peak_dbtp, SILENCE_DB);
        assert_eq!(m.output_true_peak_dbtp, SILENCE_DB);

        // Bypassed stages read neutral.
        let mut q = p;
        q.agc_enabled = false;
        q.multiband_enabled = false;
        q.limiter_enabled = false;
        bp.set_params(&q);
        let m = bp.take_meters();
        assert_eq!(m.agc_gain_db, 0.0);
        assert_eq!(m.limiter_gain_reduction_db, 0.0);
        assert!(m.band_gain_reduction_db.iter().all(|&v| v == 0.0));
    }
}

#[test]
fn limiter_gain_reduction_is_positive_when_limiting() {
    let ch = 2;
    let mut p = flat_params();
    p.agc_enabled = false;
    p.true_peak_ceiling_dbtp = -6.0;
    let mut bp = processor(FS, ch, &p);
    let x = sine(48_000, ch, FS as f64, 200.0, 1.0);
    let mut y = x.clone();
    let mut max_gr = 0.0f64;
    for c in y.chunks_mut(BLOCK * ch) {
        bp.process(c);
        let m = bp.take_meters();
        assert_no_nan(&m);
        max_gr = max_gr.max(m.limiter_gain_reduction_db);
    }
    assert!((max_gr - 6.0).abs() < 0.5, "limiter GR {max_gr}");
}

/// The meters agree with `meter::LoudnessMeter` (`ebur128`) on the same
/// signals: input and output momentary/short-term within 0.2 LU, output
/// integrated exactly (it is a `LoudnessMeter`), true peaks within 0.2 dB.
#[test]
fn meters_agree_with_loudness_meter() {
    let ch = 2;
    let fs = FS as f64;
    let mut x = pink(1 << 17, ch, fs, 11);
    scale_to_lufs(&mut x, ch, fs, -20.0);
    // Add a tone with inter-sample peaks (fs/4 at 45°) so true and sample
    // peak differ.
    for n in 0..(1 << 17) {
        let v = (0.3 * (2.0 * PI * (fs / 4.0) * n as f64 / fs + PI / 4.0).sin()) as f32;
        x[n * ch] += v;
        x[n * ch + 1] += v;
    }
    let p = BroadcastParams::default();
    let mut bp = processor(FS, ch, &p);
    let cfg = LoudnessConfig {
        sample_rate: FS,
        channels: ch,
        layout: ChannelLayout::Unknown,
        true_peak: true,
    };
    let mut ref_in = LoudnessMeter::new(&cfg).unwrap();
    let mut ref_out = LoudnessMeter::new(&cfg).unwrap();
    let mut looper = Looper::new(&x);
    let mut buf = vec![0.0f32; BLOCK * ch];
    let mut checked = 0;
    // True peaks: maxima over the whole run (the oversampler delays its
    // peaks by a few frames, so single-block intervals do not line up).
    let (mut tp_in_max, mut tp_out_max) = (SILENCE_DB, SILENCE_DB);
    let (mut ref_in_max, mut ref_out_max) = (SILENCE_DB, SILENCE_DB);
    for i in 0..1000 {
        looper.fill(&mut buf);
        ref_in.process(&buf);
        bp.process(&mut buf);
        ref_out.process(&buf);
        let m = bp.take_meters();
        assert_no_nan(&m);
        let tp_in = db((0..ch)
            .map(|c| ref_in.interval_true_peak(c))
            .fold(0.0, f64::max));
        let tp_out = db((0..ch)
            .map(|c| ref_out.interval_true_peak(c))
            .fold(0.0, f64::max));
        ref_in.clear_interval_peaks();
        ref_out.clear_interval_peaks();
        tp_in_max = tp_in_max.max(m.input_true_peak_dbtp);
        tp_out_max = tp_out_max.max(m.output_true_peak_dbtp);
        ref_in_max = ref_in_max.max(tp_in);
        ref_out_max = ref_out_max.max(tp_out);
        if i < 400 {
            continue; // let the 3 s windows fill
        }
        checked += 1;
        let close = |a: f64, b: f64, tol: f64, what: &str| {
            assert!(
                (a - b).abs() <= tol,
                "block {i} {what}: {a} vs reference {b}"
            );
        };
        close(
            m.input_momentary_lufs,
            ref_in.momentary(),
            0.2,
            "input momentary",
        );
        close(
            m.input_short_term_lufs,
            ref_in.short_term(),
            0.2,
            "input short-term",
        );
        close(
            m.output_momentary_lufs,
            ref_out.momentary(),
            0.2,
            "output momentary",
        );
        close(
            m.output_short_term_lufs,
            ref_out.short_term(),
            0.2,
            "output short-term",
        );
        close(
            m.output_integrated_lufs,
            ref_out.integrated(),
            1e-9,
            "output integrated",
        );
    }
    assert!(checked > 500);
    assert!(
        (tp_in_max - ref_in_max).abs() <= 0.2,
        "input true peak {tp_in_max} vs {ref_in_max}"
    );
    assert!(
        (tp_out_max - ref_out_max).abs() <= 0.2,
        "output true peak {tp_out_max} vs {ref_out_max}"
    );

    // reset_integrated restarts the integrated value only.
    bp.reset_integrated();
    let m = bp.take_meters();
    assert_eq!(m.output_integrated_lufs, SILENCE_DB);
    assert!(m.output_short_term_lufs > -40.0);
    for _ in 0..100 {
        looper.fill(&mut buf);
        bp.process(&mut buf);
    }
    assert!(bp.take_meters().output_integrated_lufs > -40.0);
}

/// The input meters are measured before the input gain: with +12 dB of
/// input gain they read the raw input (same as with 0 dB, and as
/// `ebur128` on the raw signal), while the output meters see the gain.
#[test]
fn input_meters_are_before_input_gain() {
    let ch = 2;
    let fs = FS as f64;
    let mut x = pink(1 << 17, ch, fs, 5);
    scale_to_lufs(&mut x, ch, fs, -35.0);
    let cfg = LoudnessConfig {
        sample_rate: FS,
        channels: ch,
        layout: ChannelLayout::Unknown,
        true_peak: true,
    };
    // Flat chain without AGC and limiter, so the output is the input
    // times the input gain (−35 + 12 = −23 LUFS stays below the ceiling).
    let mut flat = flat_params();
    flat.agc_enabled = false;
    flat.limiter_enabled = false;
    let mut boosted = flat;
    boosted.input_gain_db = 12.0;
    let mut bp0 = processor(FS, ch, &flat);
    let mut bp12 = processor(FS, ch, &boosted);
    let mut reference = LoudnessMeter::new(&cfg).unwrap();
    let mut looper = Looper::new(&x);
    let mut buf = vec![0.0f32; BLOCK * ch];
    let (mut tp0, mut tp12) = (SILENCE_DB, SILENCE_DB);
    let mut ref_tp = SILENCE_DB;
    let mut checked = 0;
    for i in 0..800 {
        looper.fill(&mut buf);
        reference.process(&buf);
        let mut y0 = buf.clone();
        let mut y12 = buf.clone();
        bp0.process(&mut y0);
        bp12.process(&mut y12);
        let m0 = bp0.take_meters();
        let m12 = bp12.take_meters();
        assert_no_nan(&m12);
        tp0 = tp0.max(m0.input_true_peak_dbtp);
        tp12 = tp12.max(m12.input_true_peak_dbtp);
        ref_tp = ref_tp.max(db((0..ch)
            .map(|c| reference.interval_true_peak(c))
            .fold(0.0, f64::max)));
        reference.clear_interval_peaks();
        if i < 400 {
            continue; // let the 3 s windows fill
        }
        checked += 1;
        // Identical to the unboosted chain, and to the raw-input reference.
        assert_eq!(
            m12.input_momentary_lufs, m0.input_momentary_lufs,
            "block {i}"
        );
        assert_eq!(
            m12.input_short_term_lufs, m0.input_short_term_lufs,
            "block {i}"
        );
        assert!(
            (m12.input_momentary_lufs - reference.momentary()).abs() <= 0.2,
            "block {i}: input momentary {} vs raw {}",
            m12.input_momentary_lufs,
            reference.momentary()
        );
        assert!(
            (m12.input_short_term_lufs - reference.short_term()).abs() <= 0.2,
            "block {i}: input short-term {} vs raw {}",
            m12.input_short_term_lufs,
            reference.short_term()
        );
        assert!(
            (m12.input_short_term_lufs + 35.0).abs() <= 1.0,
            "block {i}: input short-term {}",
            m12.input_short_term_lufs
        );
        // The output does carry the +12 dB.
        let d = m12.output_short_term_lufs - m0.output_short_term_lufs;
        assert!((d - 12.0).abs() <= 0.1, "block {i}: output difference {d}");
    }
    assert!(checked > 300);
    assert_eq!(tp12, tp0);
    assert!(
        (tp12 - ref_tp).abs() <= 0.2,
        "input true peak {tp12} vs raw {ref_tp}"
    );
}

// ---------------------------------------------------------------------------
// Robustness and real-time safety.

fn random_params(rng: &mut Rng, base: &BroadcastParams) -> BroadcastParams {
    let mut r = |lo: f64, hi: f64| (lo + (hi - lo) * rng.uniform()) as f32;
    let mut p = *base;
    p.enabled = r(0.0, 1.0) > 0.2;
    p.input_gain_db = r(-30.0, 30.0);
    p.target_loudness_lufs = r(-45.0, 0.0);
    p.true_peak_ceiling_dbtp = r(-15.0, 3.0);
    p.agc_enabled = r(0.0, 1.0) > 0.3;
    p.agc_max_gain_db = r(-5.0, 35.0);
    p.agc_max_cut_db = r(-5.0, 35.0);
    p.agc_rate_up_db_s = r(0.0, 25.0);
    p.agc_rate_down_db_s = r(0.0, 45.0);
    p.agc_gate_lufs = r(-90.0, -10.0);
    p.multiband_enabled = r(0.0, 1.0) > 0.3;
    p.num_bands = (r(0.0, 7.0) as usize).min(6);
    p.multiband_drive_db = r(-15.0, 15.0);
    for f in &mut p.crossover_hz {
        *f = r(0.0, 25_000.0);
    }
    for b in &mut p.bands {
        b.threshold_db = r(-70.0, 5.0);
        b.ratio = r(0.5, 25.0);
        b.attack_ms = r(0.0, 600.0);
        b.release_ms = r(0.0, 6000.0);
        b.makeup_db = r(-15.0, 30.0);
        b.knee_db = r(-1.0, 30.0);
    }
    p.limiter_enabled = r(0.0, 1.0) > 0.3;
    p.limiter_release_ms = r(0.0, 2500.0);
    p.limiter_lookahead_ms = r(0.0, 25.0);
    p
}

#[test]
fn params_are_clamped() {
    let mut p = BroadcastParams {
        input_gain_db: f32::NAN,
        target_loudness_lufs: -100.0,
        true_peak_ceiling_dbtp: 5.0,
        num_bands: 0,
        crossover_hz: [5000.0, 100.0, f32::INFINITY, 30_000.0],
        limiter_lookahead_ms: 0.0,
        ..BroadcastParams::default()
    };
    let mut bp = BroadcastProcessor::new();
    bp.set_params(&p);
    let q = bp.params();
    assert_eq!(q.input_gain_db, 0.0);
    assert_eq!(q.target_loudness_lufs, -40.0);
    assert_eq!(q.true_peak_ceiling_dbtp, 0.0);
    assert_eq!(q.num_bands, 1);
    assert_eq!(q.crossover_hz, [5000.0, 5000.0, 20_000.0, 20_000.0]);
    assert_eq!(q.limiter_lookahead_ms, 0.5);
    p.num_bands = 99;
    bp.set_params(&p);
    assert_eq!(bp.params().num_bands, MAX_BANDS);
}

/// Random parameter jumps every buffer, odd buffer sizes, loud and tiny
/// signals, several rates and channel counts: output and meters finite.
#[test]
fn robust_under_random_changes() {
    let mut rng = Rng::new(99);
    for (fs, ch) in [
        (8_000.0f32, 1usize),
        (44_100.0, 2),
        (96_000.0, 6),
        (192_000.0, 2),
        (48_000.0, 70),
    ] {
        let base = BroadcastParams::default();
        let mut bp = processor(fs, ch, &random_params(&mut rng, &base));
        let sizes = [1usize, 480, 7, 2000, 64, 513];
        let mut buf = vec![0.0f32; 2000 * ch];
        for i in 0..120 {
            let n = sizes[i % sizes.len()] * ch;
            let level = match i % 4 {
                0 => 10.0,
                1 => 1e-30,
                2 => 0.0,
                _ => 0.5,
            };
            for v in &mut buf[..n] {
                *v = (level * rng.bipolar()) as f32;
            }
            bp.set_params(&random_params(&mut rng, &base));
            bp.process(&mut buf[..n]);
            assert!(buf[..n].iter().all(|v| v.is_finite()), "fs {fs} ch {ch}");
            assert_no_nan(&bp.take_meters());
            if i % 50 == 49 {
                bp.reset();
            }
        }
    }
    // Zero channels / unconfigured: no-ops.
    let mut bp = BroadcastProcessor::new();
    let mut buf = [0.5f32; 64];
    bp.process(&mut buf);
    assert!(buf.iter().all(|&v| v == 0.5));
    bp.configure(48_000.0, 0);
    bp.process(&mut buf);
    assert!(buf.iter().all(|&v| v == 0.5));
}

/// `set_params`, `process`, `take_meters`, `reset_integrated` and `reset`
/// never allocate.
#[test]
fn no_allocation() {
    let ch = 2;
    let mut rng = Rng::new(5);
    let base = BroadcastParams::default();
    let mut bp = processor(FS, ch, &base);
    let mut buf = vec![0.0f32; 3000 * ch];
    let sizes = [480usize, 1, 3000, 37, 1024];
    let before = allocs();
    for i in 0..200 {
        let n = sizes[i % sizes.len()] * ch;
        for v in &mut buf[..n] {
            *v = (0.8 * rng.bipolar()) as f32;
        }
        let p = random_params(&mut rng, &base);
        bp.set_params(&p);
        bp.process(&mut buf[..n]);
        let _ = bp.take_meters();
        if i % 40 == 13 {
            bp.reset_integrated();
        }
        if i % 70 == 69 {
            bp.reset();
        }
    }
    assert_eq!(allocs() - before, 0, "allocations in the processing path");
}

// ---------------------------------------------------------------------------
// Acceptance: integrated loudness and true peak per preset.

/// 60 s programmes per preset (`strom_dsp::presets`, at the default 4 bands)
/// at three input levels: the integrated loudness (excluding the first
/// 10 s) lands within ±1 LU of the preset target, and the output true peak
/// (BS.1770, `ebur128`) stays at or below the ceiling + 0.1 dB.
///
/// With these gentle presets this programme's output true peak stays 2 …
/// 9 dB below the ceiling, so it does not stress the true-peak limiter (its
/// own tests in `tests/truepeak.rs` do). TODO: add a hot, peaky programme
/// through the full chain.
mod acceptance {
    use super::*;
    use strom_dsp::presets::BroadcastPreset;

    const SECONDS: usize = 60;
    const SKIP_SECONDS: usize = 10;
    /// Allowed true-peak overshoot (0.1 dB, the true-peak limiter's guarantee).
    const TP_MARGIN_DB: f64 = 0.1;

    /// The preset with this nick, at the default band count.
    fn preset(name: &str) -> BroadcastParams {
        BroadcastParams::preset(BroadcastPreset::from_nick(name).unwrap(), 4)
    }

    /// A music-like programme: pink noise with a 4 Hz level modulation, two
    /// tones, and short speech-like gaps (a 10.9 s loop).
    fn programme(ch: usize, fs: f64) -> Vec<f32> {
        let frames = 1 << 19; // ~10.9 s at 48 kHz
        let mut x = pink(frames, ch, fs, 21);
        for n in 0..frames {
            let t = n as f64 / fs;
            // 4 Hz syllable-like envelope, with a 300 ms gap every 2.7 s.
            let syl = 0.6 + 0.4 * (2.0 * PI * 4.0 * t).sin();
            let gap = if (t % 2.7) < 0.3 { 0.05 } else { 1.0 };
            let tones = 0.4 * (2.0 * PI * 110.0 * t).sin() + 0.2 * (2.0 * PI * 1_320.0 * t).sin();
            for c in 0..ch {
                let v = &mut x[n * ch + c];
                *v = ((*v as f64 * 0.5 + tones) * syl * gap) as f32;
            }
        }
        x
    }

    /// Run the three input levels; assert ±1 LU and true peak ≤ ceiling +
    /// `tp_margin_db`.
    fn check(name: &str, tp_margin_db: f64) {
        let ch = 2;
        let fs = FS as f64;
        let p = preset(name);
        let src = programme(ch, fs);
        for input_lufs in [-35.0, -23.0, -10.0] {
            let mut x = src.clone();
            scale_to_lufs(&mut x, ch, fs, input_lufs);
            let mut bp = processor(FS, ch, &p);
            let mut meter = LoudnessMeter::new(&LoudnessConfig {
                sample_rate: FS,
                channels: ch,
                layout: ChannelLayout::Unknown,
                true_peak: true,
            })
            .unwrap();
            let mut looper = Looper::new(&x);
            let mut buf = vec![0.0f32; BLOCK * ch];
            let blocks = SECONDS * FS as usize / BLOCK;
            let skip = SKIP_SECONDS * FS as usize / BLOCK;
            for i in 0..blocks {
                looper.fill(&mut buf);
                bp.process(&mut buf);
                if i >= skip {
                    meter.process(&buf);
                }
            }
            let integrated = meter.integrated();
            let tp = db((0..ch).map(|c| meter.true_peak(c)).fold(0.0, f64::max));
            println!(
                "{name} at {input_lufs} LUFS: integrated {integrated:.2} LUFS (target {}), \
                 true peak {tp:.2} dBTP (ceiling {})",
                p.target_loudness_lufs, p.true_peak_ceiling_dbtp
            );
            assert!(
                (integrated - p.target_loudness_lufs as f64).abs() <= 1.0,
                "{name} at {input_lufs} LUFS: integrated {integrated:.2} LUFS, target {}",
                p.target_loudness_lufs
            );
            assert!(
                tp <= p.true_peak_ceiling_dbtp as f64 + tp_margin_db,
                "{name} at {input_lufs} LUFS: true peak {tp:.2} dBTP, ceiling {}",
                p.true_peak_ceiling_dbtp
            );
        }
    }

    #[test]
    fn preset_ebu_r128() {
        check("ebu-r128", TP_MARGIN_DB);
    }

    #[test]
    fn preset_atsc_a85() {
        check("atsc-a85", TP_MARGIN_DB);
    }

    #[test]
    fn preset_streaming() {
        check("streaming", TP_MARGIN_DB);
    }

    #[test]
    fn preset_speech() {
        check("speech", TP_MARGIN_DB);
    }
}
