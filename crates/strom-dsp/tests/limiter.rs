//! Limiter tests: ceiling guarantee, latency, safety-clip idleness, linking,
//! release behaviour, robustness and real-time safety.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use proptest::prelude::*;
use strom_dsp::limiter::GAIN_HOLD_MS;
use strom_dsp::{Limiter, LimiterParams, Processor, MAX_LOOKAHEAD_MS};

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

const RATES: [f32; 5] = [8_000.0, 44_100.0, 48_000.0, 96_000.0, 192_000.0];

fn db(x: f64) -> f64 {
    10f64.powf(x / 20.0)
}

/// Small deterministic PRNG (xorshift64*), uniform in [-1, 1).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

fn params(thr: f32, attack: f32, release: f32, lookahead: f32) -> LimiterParams {
    LimiterParams {
        enabled: true,
        threshold_db: thr,
        attack_ms: attack,
        release_ms: release,
        lookahead_ms: lookahead,
    }
}

fn make(fs: f32, ch: usize, p: &LimiterParams) -> Limiter {
    let mut l = Limiter::new();
    l.set_params(p);
    l.configure(fs, ch);
    l
}

/// Process in uneven block sizes to exercise block boundaries.
fn run(l: &mut Limiter, buf: &mut [f32], ch: usize) {
    let sizes = [1usize, 7, 64, 333, 1024];
    let mut i = 0;
    let mut k = 0;
    while i < buf.len() {
        let n = (sizes[k % sizes.len()] * ch).min(buf.len() - i);
        l.process(&mut buf[i..i + n]);
        i += n;
        k += 1;
    }
}

fn assert_ceiling(out: &[f32], thr_db: f32, what: &str) {
    let c = db(thr_db as f64);
    for (i, &y) in out.iter().enumerate() {
        assert!(y.is_finite(), "{what}: non-finite at {i}");
        assert!(
            (y.abs() as f64) <= c,
            "{what}: sample {i} = {y} exceeds ceiling {c}"
        );
    }
}

// Signal generators (interleaved, `ch` channels, all channels independent).

fn white_noise(frames: usize, ch: usize, amp: f64, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..frames * ch).map(|_| (amp * r.next()) as f32).collect()
}

fn square(frames: usize, ch: usize, amp: f64, period: usize) -> Vec<f32> {
    (0..frames * ch)
        .map(|i| {
            let n = i / ch;
            if (n % period) < period / 2 {
                amp as f32
            } else {
                -amp as f32
            }
        })
        .collect()
}

fn impulses(frames: usize, ch: usize, amp: f64, spacing: usize) -> Vec<f32> {
    (0..frames * ch)
        .map(|i| {
            let n = i / ch;
            if n % spacing == spacing / 3 {
                amp as f32
            } else {
                0.0
            }
        })
        .collect()
}

fn sine(frames: usize, ch: usize, amp: f64, freq: f64, fs: f64) -> Vec<f32> {
    (0..frames * ch)
        .map(|i| {
            let n = (i / ch) as f64;
            (amp * (2.0 * std::f64::consts::PI * freq * n / fs).sin()) as f32
        })
        .collect()
}

/// Bursts of noise with random lengths and levels up to +20 dBFS, separated by
/// silence or quiet tones.
fn bursts(frames: usize, ch: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    let mut out = vec![0.0f32; frames * ch];
    let mut n = 0;
    while n < frames {
        let len = 1 + ((r.next() + 1.0) * 400.0) as usize;
        let level = db((r.next() + 1.0) * 30.0 - 40.0); // -40 … +20 dBFS
        let loud = r.next() > -0.3;
        for f in n..(n + len).min(frames) {
            for c in 0..ch {
                out[f * ch + c] = if loud { (level * r.next()) as f32 } else { 0.0 };
            }
        }
        n += len;
    }
    out
}

/// "Music-like": a few decaying partials with random onsets plus light noise.
fn music_like(frames: usize, ch: usize, fs: f64, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    let mut out = vec![0.0f64; frames];
    let mut onset = 0usize;
    while onset < frames {
        let f0 = 80.0 + (r.next() + 1.0) * 400.0;
        let amp = 0.3 + (r.next() + 1.0) * 0.6;
        let decay = 1.0 / (0.05 + (r.next() + 1.0) * 0.2) / fs;
        for (k, o) in out.iter_mut().enumerate().skip(onset) {
            let t = (k - onset) as f64;
            let env = amp * (-t * decay).exp();
            if env < 1e-4 {
                break;
            }
            let ph = 2.0 * std::f64::consts::PI * f0 * t / fs;
            *o += env * (ph.sin() + 0.5 * (2.0 * ph).sin() + 0.25 * (3.0 * ph).sin());
        }
        onset += (fs * (0.05 + (r.next() + 1.0) * 0.15)) as usize;
    }
    let mut v = Vec::with_capacity(frames * ch);
    for s in out {
        for _ in 0..ch {
            v.push((s + 0.01 * r.next()) as f32);
        }
    }
    v
}

// ---------------------------------------------------------------------------
// Ceiling guarantee.

#[test]
fn ceiling_holds_on_hard_signals_all_rates() {
    for &fs in &RATES {
        let frames = (fs as usize) / 4;
        for &la in &[0.0f32, 0.1, 1.5, 5.0, 20.0] {
            for &(thr, att, rel) in &[
                (-1.0f32, 1.0f32, 100.0f32),
                (-20.0, 50.0, 1.0),
                (0.0, 0.01, 2000.0),
                (-60.0, 5.0, 10.0),
            ] {
                let p = params(thr, att, rel, la);
                let signals: Vec<(&str, Vec<f32>)> = vec![
                    ("impulses", impulses(frames, 2, 10.0, 97)),
                    ("square_fullscale", square(frames, 2, 1.0, 100)),
                    ("square_+20", square(frames, 2, 10.0, 37)),
                    ("noise_+20", white_noise(frames, 2, 10.0, 0x1234)),
                    ("bursts", bursts(frames, 2, 0x5678)),
                ];
                for (name, mut sig) in signals {
                    let mut l = make(fs, 2, &p);
                    run(&mut l, &mut sig, 2);
                    let what = format!("{name} fs={fs} la={la} {p:?}");
                    assert_ceiling(&sig, thr, &what);
                    assert_eq!(l.safety_clip_count(), 0, "{what}: safety clip used");
                }
            }
        }
    }
}

#[test]
fn single_full_scale_impulse_is_limited_exactly_to_ceiling() {
    for &la in &[0.0f32, 5.0] {
        let fs = 48_000.0;
        let p = params(-6.0, 1.0, 100.0, la);
        let mut l = make(fs, 1, &p);
        let mut x = vec![0.0f32; 2000];
        x[100] = 1.0;
        l.process(&mut x);
        let lat = l.latency_frames();
        let peak = x.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!((x[100 + lat] - l.ceiling_linear()).abs() < 1e-6);
        assert!(peak as f64 <= db(-6.0));
        assert_eq!(l.safety_clip_count(), 0);
    }
}

#[derive(Debug, Clone)]
enum Kind {
    Noise,
    Square(usize),
    Impulses(usize),
    Bursts,
    Raw(Vec<f32>),
}

fn kind_strategy() -> impl Strategy<Value = Kind> {
    prop_oneof![
        Just(Kind::Noise),
        (2usize..400).prop_map(Kind::Square),
        (1usize..300).prop_map(Kind::Impulses),
        Just(Kind::Bursts),
        prop::collection::vec(-10.0f32..10.0, 1..600).prop_map(Kind::Raw),
    ]
}

fn param_strategy() -> impl Strategy<Value = LimiterParams> {
    (
        prop::bool::weighted(0.9),
        -60.0f32..=0.0,
        0.01f32..=50.0,
        1.0f32..=2000.0,
        prop_oneof![Just(0.0f32), 0.0f32..=20.0],
    )
        .prop_map(|(enabled, thr, att, rel, la)| LimiterParams {
            enabled,
            threshold_db: thr,
            attack_ms: att,
            release_ms: rel,
            lookahead_ms: la,
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// For arbitrary signals up to +20 dBFS and arbitrary parameter changes
    /// between blocks (threshold, attack, release, lookahead, enable), every
    /// output sample stays below the ceiling that was in force for the input
    /// frames it can stem from, and the safety clip never triggers.
    #[test]
    fn prop_ceiling_guarantee(
        fs in prop::sample::select(RATES.to_vec()),
        ch in 1usize..4,
        kind in kind_strategy(),
        amp in 0.0f64..10.0,
        seed in any::<u64>(),
        blocks in prop::collection::vec((param_strategy(), 1usize..900), 1..8),
    ) {
        let total: usize = blocks.iter().map(|b| b.1).sum();
        let mut sig = match &kind {
            Kind::Noise => white_noise(total, ch, amp, seed),
            Kind::Square(p) => square(total, ch, amp, *p),
            Kind::Impulses(s) => impulses(total, ch, amp, *s),
            Kind::Bursts => bursts(total, ch, seed),
            Kind::Raw(v) => (0..total * ch).map(|i| v[i % v.len()]).collect(),
        };
        let mut l = make(fs, ch, &blocks[0].0);
        // Ceiling in force for each input frame (infinite when disabled).
        let mut ceil_in = Vec::with_capacity(total);
        let mut pos = 0;
        for (p, n) in &blocks {
            l.set_params(p);
            let c = if p.enabled { db(p.threshold_db as f64) } else { f64::INFINITY };
            ceil_in.extend(std::iter::repeat_n(c, *n));
            l.process(&mut sig[pos * ch..(pos + n) * ch]);
            pos += n;
        }
        let lmax = (MAX_LOOKAHEAD_MS as f64 * fs as f64 / 1000.0).round() as usize;
        for n in 0..total {
            let lo = n.saturating_sub(lmax);
            let c = ceil_in[lo..=n].iter().cloned().fold(0.0, f64::max);
            for k in 0..ch {
                let y = sig[n * ch + k];
                prop_assert!(y.is_finite());
                prop_assert!((y.abs() as f64) <= c, "frame {} ch {} y={} c={}", n, k, y, c);
            }
        }
        prop_assert_eq!(l.safety_clip_count(), 0);
    }
}

// ---------------------------------------------------------------------------
// Latency.

#[test]
fn delay_equals_lookahead() {
    for &fs in &RATES {
        for &la in &[0.0f32, 0.05, 1.0, 5.0, 12.34, 20.0] {
            let p = params(0.0, 1.0, 100.0, la);
            let mut l = make(fs, 2, &p);
            let expect = (la as f64 * fs as f64 / 1000.0).round() as usize;
            assert_eq!(l.latency_frames(), expect, "fs={fs} la={la}");
            let frames = expect + 64;
            let mut x = vec![0.0f32; frames * 2];
            x[0] = 0.5;
            x[1] = -0.25;
            l.process(&mut x);
            for n in 0..frames {
                let (a, b) = (x[2 * n], x[2 * n + 1]);
                if n == expect {
                    assert_eq!((a, b), (0.5, -0.25), "fs={fs} la={la}");
                } else {
                    assert_eq!((a, b), (0.0, 0.0), "fs={fs} la={la} n={n}");
                }
            }
        }
    }
}

#[test]
fn disabled_is_exact_pure_delay() {
    for &fs in &RATES {
        for &la in &[0.0f32, 3.0, 20.0] {
            let mut p = params(-30.0, 1.0, 100.0, la);
            p.enabled = false;
            let mut l = make(fs, 3, &p);
            let frames = 5000;
            let input = white_noise(frames, 3, 10.0, 99);
            let mut out = input.clone();
            run(&mut l, &mut out, 3);
            let d = l.latency_frames();
            assert_eq!(d, (la as f64 * fs as f64 / 1000.0).round() as usize);
            for i in 0..frames * 3 {
                let expect = if i >= d * 3 { input[i - d * 3] } else { 0.0 };
                assert_eq!(out[i].to_bits(), expect.to_bits(), "fs={fs} la={la} i={i}");
            }
            assert_eq!(l.safety_clip_count(), 0);
        }
    }
}

#[test]
fn below_threshold_is_exact_delay() {
    let fs = 48_000.0;
    let p = params(-1.0, 1.0, 100.0, 5.0);
    let mut l = make(fs, 2, &p);
    let input = sine(10_000, 2, 0.5, 440.0, fs as f64);
    let mut out = input.clone();
    run(&mut l, &mut out, 2);
    let d = l.latency_frames() * 2;
    for i in d..out.len() {
        assert_eq!(out[i], input[i - d]);
    }
}

// ---------------------------------------------------------------------------
// Safety clip idle on normal material.

#[test]
fn safety_clip_idle_on_normal_material() {
    for &fs in &[44_100.0f32, 48_000.0, 96_000.0] {
        let frames = fs as usize; // 1 s
        let f = fs as f64;
        for &(thr, att, rel, la) in &[
            (-1.0f32, 1.0f32, 100.0f32, 5.0f32),
            (-6.0, 0.5, 50.0, 2.0),
            (-3.0, 5.0, 300.0, 10.0),
            (-1.0, 1.0, 100.0, 0.0),
        ] {
            let p = params(thr, att, rel, la);
            let signals: Vec<(&str, Vec<f32>)> = vec![
                ("sine_1k_+6", sine(frames, 2, 2.0, 1000.0, f)),
                ("sine_50_+12", sine(frames, 2, 4.0, 50.0, f)),
                ("sine_10k_0", sine(frames, 2, 1.0, 10_000.0, f)),
                ("music", music_like(frames, 2, f, 7)),
                ("noise_0", white_noise(frames, 2, 1.0, 3)),
                ("square_0", square(frames, 2, 1.0, 200)),
            ];
            for (name, mut sig) in signals {
                let mut l = make(fs, 2, &p);
                run(&mut l, &mut sig, 2);
                let what = format!("{name} fs={fs} {p:?}");
                assert_ceiling(&sig, thr, &what);
                assert_eq!(l.safety_clip_count(), 0, "{what}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Linking.

#[test]
fn linked_gain_applies_to_all_channels() {
    let fs = 48_000.0;
    for &ch in &[2usize, 8] {
        for &la in &[0.0f32, 5.0] {
            let p = params(-6.0, 1.0, 50.0, la);
            let mut l = make(fs, ch, &p);
            let frames = 20_000;
            let loud_ch = ch - 1;
            // Loud bursting square in one channel, quiet distinct sines elsewhere.
            let mut input = vec![0.0f32; frames * ch];
            for n in 0..frames {
                for c in 0..ch {
                    input[n * ch + c] = if c == loud_ch {
                        if (n / 3000) % 2 == 0 {
                            if (n / 50) % 2 == 0 {
                                2.0
                            } else {
                                -2.0
                            }
                        } else {
                            0.0
                        }
                    } else {
                        let ph = 2.0 * std::f64::consts::PI * (200.0 + 100.0 * c as f64) * n as f64
                            / fs as f64;
                        (0.1 * ph.sin() + 0.02) as f32
                    };
                }
            }
            let mut out = input.clone();
            run(&mut l, &mut out, ch);
            let d = l.latency_frames();
            let mut reduced = 0;
            for n in d..frames {
                let ref_c = 0; // a quiet channel
                let xr = input[(n - d) * ch + ref_c] as f64;
                if xr.abs() < 1e-3 {
                    continue;
                }
                let g = out[n * ch + ref_c] as f64 / xr;
                if g < 0.99 {
                    reduced += 1;
                }
                for c in 0..ch {
                    let x = input[(n - d) * ch + c] as f64;
                    if x.abs() < 1e-3 {
                        continue;
                    }
                    let gc = out[n * ch + c] as f64 / x;
                    assert!((gc - g).abs() < 1e-5, "ch={ch} n={n} c={c} g={g} gc={gc}");
                }
            }
            assert!(reduced > 1000, "quiet channels should be reduced too");
        }
    }
}

// ---------------------------------------------------------------------------
// Smoothness: steady gain on sustained material, documented release time.

#[test]
fn sustained_sine_has_steady_gain() {
    let fs = 48_000.0;
    let p = params(-1.0, 1.0, 100.0, 5.0);
    let mut l = make(fs, 1, &p);
    let frames = 48_000;
    let input = sine(frames, 1, 2.0, 1000.0, fs as f64);
    let mut out = input.clone();
    l.process(&mut out);
    let d = l.latency_frames();
    let target = db(-1.0) / 2.0;
    for n in 4800..frames {
        let x = input[n - d] as f64;
        if x.abs() < 0.2 {
            continue;
        }
        let g = out[n] as f64 / x;
        assert!((g / target).log10().abs() * 20.0 < 0.01, "n={n} g={g}");
    }
}

#[test]
fn release_follows_documented_time_constant() {
    // Release is a one-pole rise in linear gain: after `release` ms the gap
    // to unity has fallen to 1/e.
    let fs = 48_000.0;
    for &rel in &[10.0f32, 100.0, 500.0] {
        let p = params(-20.0, 0.01, rel, 2.0);
        let mut l = make(fs, 1, &p);
        let loud = 4800;
        let tail = (rel as f64 * 0.001 * fs as f64 * 4.0) as usize + 2000;
        let mut input = vec![0.0f32; loud + tail];
        for (n, v) in input.iter_mut().enumerate() {
            *v = if n < loud { 1.0 } else { 0.01 };
        }
        let mut out = input.clone();
        l.process(&mut out);
        let d = l.latency_frames();
        let gains: Vec<f64> = (d..out.len())
            .map(|n| out[n] as f64 / input[n - d] as f64)
            .collect();
        let g0 = db(-20.0);
        // First frame of release. The gain is held for
        // H = min(GAIN_HOLD_MS, release) frames after the last loud frame
        // (the bass THD fix); the time constant applies from there.
        let start = gains.iter().position(|&g| g > g0 * 1.0001).unwrap();
        let hold = (f64::from(rel).min(GAIN_HOLD_MS) * 0.001 * fs as f64).round() as usize;
        assert_eq!(
            start,
            loud + hold,
            "rel={rel}: release starts after the hold"
        );
        assert!((gains[start - 1] - g0).abs() < 1e-6);
        let tau = (rel as f64 * 0.001 * fs as f64).round() as usize;
        let gap0 = 1.0 - g0;
        let gap = 1.0 - gains[start - 1 + tau];
        let ratio = gap / gap0;
        assert!(
            (ratio - (-1.0f64).exp()).abs() < 0.01,
            "rel={rel} ratio={ratio}"
        );
        // Monotonic, no overshoot.
        for w in gains[start..].windows(2) {
            assert!(w[1] >= w[0] - 1e-6 && w[1] <= 1.0 + 1e-6);
        }
    }
}

#[test]
fn longer_hold_does_not_drop_a_releasing_gain() {
    // Release 5 ms (hold 5 ms) → 100 ms (hold 20 ms) while the gain is
    // releasing after a loud passage: the longer hold must not reach back to
    // the loud frames and pull the gain down again. It grows frame by frame.
    let fs = 48_000.0;
    let mut l = make(fs, 1, &params(-20.0, 1.0, 5.0, 2.0));
    let d = l.latency_frames();
    let loud = 4800;
    let mut input = vec![1.0f32; loud];
    input.extend(std::iter::repeat_n(0.01f32, 4800));
    let mut out = input.clone();
    // Switch 10 ms after the loud part has left the output: hold (5 ms)
    // over, release under way.
    let split = loud + d + 480;
    l.process(&mut out[..split]);
    let g_before = out[split - 1] as f64 / input[split - 1 - d] as f64;
    assert!(g_before > db(-20.0) * 1.01, "release under way: {g_before}");
    l.set_params(&params(-20.0, 1.0, 100.0, 2.0));
    l.process(&mut out[split..]);
    let gains: Vec<f64> = (split - 1..out.len())
        .map(|n| out[n] as f64 / input[n - d] as f64)
        .collect();
    for w in gains.windows(2) {
        assert!(w[1] >= w[0] - 1e-9, "gain fell {} → {}", w[0], w[1]);
    }
    assert_eq!(l.safety_clip_count(), 0);
}

#[test]
fn attack_ramp_completes_exactly_at_peak() {
    // Gain is still unity `attack + 1` frames before the step reaches the
    // output and is fully reduced when it arrives.
    let fs = 48_000.0;
    let p = params(-6.0, 2.0, 100.0, 5.0);
    let mut l = make(fs, 1, &p);
    let mut x = vec![0.1f32; 2000];
    for v in x.iter_mut().skip(1000) {
        *v = 1.0;
    }
    let input = x.clone();
    l.process(&mut x);
    let d = l.latency_frames();
    let a = 96; // 2 ms
    let arrive = 1000 + d;
    assert_eq!(x[arrive - a - 1], 0.1);
    assert!(x[arrive - a / 2] < 0.1 && x[arrive - a / 2] > 0.05);
    let c = l.ceiling_linear();
    assert_eq!(x[arrive], c);
    for n in arrive..2000 {
        assert!((x[n] / input[n - d] - c).abs() < 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Robustness.

#[test]
fn extreme_params_rates_channels_no_nan() {
    let mins = params(-60.0, 0.01, 1.0, 0.0);
    let maxs = params(0.0, 50.0, 2000.0, 20.0);
    for &fs in &RATES {
        for &ch in &[1usize, 2, 8, 64] {
            for combo in 0..16u32 {
                let pick = |bit: u32, a: f32, b: f32| if combo & (1 << bit) != 0 { b } else { a };
                let p = LimiterParams {
                    enabled: true,
                    threshold_db: pick(0, mins.threshold_db, maxs.threshold_db),
                    attack_ms: pick(1, mins.attack_ms, maxs.attack_ms),
                    release_ms: pick(2, mins.release_ms, maxs.release_ms),
                    lookahead_ms: pick(3, mins.lookahead_ms, maxs.lookahead_ms),
                };
                let lmax = (20.0 * fs / 1000.0) as usize;
                let frames = lmax + 600;
                let mut sig = white_noise(frames, ch, 10.0, combo as u64 + 1);
                let mut l = make(fs, ch, &p);
                run(&mut l, &mut sig, ch);
                let what = format!("fs={fs} ch={ch} {p:?}");
                assert_ceiling(&sig, p.threshold_db, &what);
                assert_eq!(l.safety_clip_count(), 0, "{what}");
            }
        }
    }
}

#[test]
fn out_of_range_and_non_finite_params_are_sanitized() {
    let mut l = Limiter::new();
    l.configure(48_000.0, 2);
    for p in [
        params(f32::NAN, f32::NAN, f32::NAN, f32::NAN),
        params(
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::INFINITY,
        ),
        params(100.0, -5.0, 0.0, -1.0),
        params(-1000.0, 1e9, 1e9, 1e9),
    ] {
        l.set_params(&p);
        let q = *l.params();
        assert!((-60.0..=0.0).contains(&q.threshold_db));
        assert!((0.01..=50.0).contains(&q.attack_ms));
        assert!((1.0..=2000.0).contains(&q.release_ms));
        assert!((0.0..=20.0).contains(&q.lookahead_ms));
        let mut sig = white_noise(3000, 2, 10.0, 5);
        l.process(&mut sig);
        assert!(sig.iter().all(|v| v.is_finite()));
        assert!(sig
            .iter()
            .all(|v| v.abs() as f64 <= db(q.threshold_db as f64).max(1.0)));
    }
    assert_eq!(l.safety_clip_count(), 0);
}

#[test]
fn unconfigured_and_odd_configurations() {
    // Unconfigured: no-op.
    let mut l = Limiter::new();
    let mut x = vec![5.0f32; 16];
    l.process(&mut x);
    assert!(x.iter().all(|&v| v == 5.0));
    assert_eq!(l.latency_frames(), 0);
    // Zero channels: still a no-op.
    l.configure(48_000.0, 0);
    l.process(&mut x);
    assert!(x.iter().all(|&v| v == 5.0));
    // Odd sample rates.
    for fs in [1.0f32, 1000.0, 11_025.0, 384_000.0, f32::NAN, -5.0] {
        let mut l = Limiter::new();
        l.configure(fs, 3);
        let mut sig = white_noise(4000, 3, 10.0, 11);
        l.process(&mut sig);
        assert_ceiling(&sig, -1.0, &format!("fs={fs}"));
        assert_eq!(l.safety_clip_count(), 0);
    }
    // Buffer not a multiple of channels: trailing partial frame untouched.
    let mut l = Limiter::new();
    l.configure(48_000.0, 2);
    let mut y = vec![9.0f32; 5];
    l.process(&mut y);
    assert_eq!(y[4], 9.0);
}

#[test]
fn reset_clears_state() {
    let fs = 48_000.0;
    let p = params(-6.0, 1.0, 500.0, 5.0);
    let mut fresh = make(fs, 2, &p);
    let mut used = make(fs, 2, &p);
    let mut junk = white_noise(10_000, 2, 10.0, 1);
    used.process(&mut junk);
    used.reset();
    assert_eq!(used.safety_clip_count(), 0);
    let mut a = music_like(20_000, 2, fs as f64, 4);
    let mut b = a.clone();
    fresh.process(&mut a);
    used.process(&mut b);
    assert_eq!(a, b);
}

/// Jump bound for parameter changes while running.
///
/// Documented bound: with a 1 kHz sine of amplitude 0.5 at 48 kHz and the
/// parameters below changed every 20 ms, the largest sample-to-sample output
/// step exceeds the input's own largest step by at most 0.01 (−40 dBFS).
#[test]
fn param_changes_do_not_jump() {
    let fs = 48_000.0f32;
    let frames = 48_000 * 2;
    let amp = 0.5;
    let input = sine(frames, 2, amp, 1000.0, fs as f64);
    let natural = 2.0 * amp * (std::f64::consts::PI * 1000.0 / fs as f64).sin();
    let settings = [
        params(-1.0, 1.0, 100.0, 5.0),
        params(-12.0, 1.0, 100.0, 5.0),
        params(-12.0, 3.0, 20.0, 5.0),
        params(-3.0, 0.5, 500.0, 10.0),
        params(-12.0, 2.0, 50.0, 2.0),
        params(-6.0, 2.0, 50.0, 8.0),
        params(-20.0, 1.0, 10.0, 8.0),
        LimiterParams {
            enabled: false,
            ..params(-20.0, 1.0, 10.0, 8.0)
        },
        params(-20.0, 1.0, 10.0, 8.0),
        params(-9.0, 4.0, 200.0, 4.0),
    ];
    let mut l = make(fs, 2, &settings[0]);
    let mut out = input.clone();
    let block = 960; // 20 ms
    for (k, chunk) in out.chunks_mut(block * 2).enumerate() {
        l.set_params(&settings[k % settings.len()]);
        l.process(chunk);
    }
    let mut worst: f64 = 0.0;
    for n in 1..frames {
        let step = (out[n * 2] as f64 - out[(n - 1) * 2] as f64).abs();
        worst = worst.max(step);
    }
    assert!(
        worst <= natural + 0.01,
        "worst step {worst}, natural {natural}"
    );
    assert_eq!(l.safety_clip_count(), 0);
}

// ---------------------------------------------------------------------------
// Real-time safety.

#[test]
fn no_allocation_in_process_or_set_params() {
    for &fs in &[8_000.0f32, 48_000.0, 192_000.0] {
        let mut l = Limiter::new();
        l.configure(fs, 8);
        let mut buf = white_noise(1024, 8, 10.0, 2);
        let variations = [
            params(-1.0, 1.0, 100.0, 5.0),
            params(-30.0, 50.0, 2000.0, 20.0),
            params(0.0, 0.01, 1.0, 0.0),
            LimiterParams {
                enabled: false,
                ..params(-6.0, 3.0, 10.0, 13.0)
            },
            params(-6.0, 20.0, 10.0, 1.0),
        ];
        let before = allocs();
        for _ in 0..20 {
            for p in &variations {
                l.set_params(p);
                l.process(&mut buf);
                for v in buf.iter_mut() {
                    *v *= 10.0;
                }
            }
        }
        l.reset();
        l.process(&mut buf);
        let after = allocs();
        assert_eq!(
            after - before,
            0,
            "fs={fs}: allocations in process/set_params"
        );
    }
}

// ---------------------------------------------------------------------------
// Call order and latency helper.

#[test]
fn set_params_before_or_after_configure_is_identical() {
    let fs = 48_000.0f32;
    for p in [
        params(-6.0, 2.0, 50.0, 3.0),
        params(-12.0, 0.5, 20.0, 0.0),
        params(-3.0, 10.0, 200.0, 17.0),
        LimiterParams {
            enabled: false,
            ..params(-6.0, 2.0, 50.0, 12.0)
        },
    ] {
        let input = white_noise(20_000, 2, 4.0, 9);
        let a = make(fs, 2, &p);
        let mut b = Limiter::new();
        b.configure(fs, 2);
        b.set_params(&p);
        let mut c = Limiter::new();
        c.configure(fs, 2);
        c.process(&mut input.clone());
        c.reset();
        c.set_params(&p);
        assert_eq!(a.latency_frames(), b.latency_frames());
        let mut outs = Vec::new();
        for mut l in [a, b, c] {
            let mut y = input.clone();
            run(&mut l, &mut y, 2);
            outs.push(y);
        }
        for y in &outs[1..] {
            assert!(
                outs[0]
                    .iter()
                    .zip(y)
                    .all(|(u, v)| u.to_bits() == v.to_bits()),
                "{p:?}"
            );
        }
    }
}

#[test]
fn lookahead_frames_matches_reported_latency() {
    for fs in RATES
        .into_iter()
        .chain([22_050.0, 11_025.0, 1.0e6, f32::NAN])
    {
        for ms in [
            0.0,
            0.01,
            1.0,
            2.5,
            3.3,
            5.0,
            12.345,
            20.0,
            25.0,
            -1.0,
            f32::NAN,
        ] {
            let mut l = Limiter::new();
            l.configure(fs, 1);
            l.set_params(&params(-1.0, 1.0, 100.0, ms));
            assert_eq!(
                strom_dsp::limiter::lookahead_frames(ms, fs),
                l.latency_frames(),
                "fs={fs} ms={ms}"
            );
        }
    }
    assert_eq!(strom_dsp::limiter::lookahead_frames(5.0, 48_000.0), 240);
    assert_eq!(strom_dsp::limiter::lookahead_frames(3.0, 44_100.0), 132); // 132.3
}
