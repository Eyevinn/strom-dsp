//! Limiter tests: ceiling guarantee, latency, safety-clip idleness, linking,
//! release behaviour, robustness and real-time safety.

use proptest::prelude::*;
use strom_dsp::limiter::GAIN_HOLD_MS;
use strom_dsp::{Limiter, LimiterParams, Processor, MAX_LOOKAHEAD_MS};

use crate::counting_alloc::allocs;

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
    #![proptest_config(ProptestConfig::with_cases(crate::proptest_cases(200)))]

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

// ---------------------------------------------------------------------------
// Live lookahead changes (ELEMENT_GUIDELINES, "Latency" and "Live parameter
// changes"): `lookahead` is mutable while playing, so after a change and its
// cross-fade the audio must be delayed by exactly the new reported latency.

/// Length of the lookahead cross-fade (module docs: raised cosine, 5 ms).
fn xfade_frames(fs: f32) -> usize {
    ((5.0 * fs as f64 / 1000.0).round() as usize).max(1)
}

/// Process `frames` frames of silence in uneven blocks.
fn silence(l: &mut Limiter, frames: usize, ch: usize) {
    let mut z = vec![0.0f32; frames * ch];
    run(l, &mut z, ch);
}

/// Feed an impulse after silence and return the frame at which it comes out
/// (checking that it is the only non-zero frame and is bit-exact).
fn impulse_delay(l: &mut Limiter, ch: usize, what: &str) -> usize {
    let frames = 2 * l.latency_frames() + 200;
    let mut x = vec![0.0f32; frames * ch];
    x[0] = 0.5;
    if ch > 1 {
        x[1] = -0.25;
    }
    run(l, &mut x, ch);
    let hits: Vec<usize> = (0..frames)
        .filter(|&n| x[n * ch..(n + 1) * ch].iter().any(|&v| v != 0.0))
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "{what}: impulse smeared over frames {hits:?}"
    );
    let n = hits[0];
    assert_eq!(x[n * ch], 0.5, "{what}: impulse not bit-exact");
    if ch > 1 {
        assert_eq!(x[n * ch + 1], -0.25, "{what}: impulse not bit-exact");
    }
    n
}

/// After a live lookahead change (longer, shorter, to and from 0, a value
/// that is not a whole number of frames, a change queued during a running
/// cross-fade) and the cross-fade that follows it, an impulse comes out
/// delayed by exactly the new `latency_frames()`, which is
/// `lookahead_frames(new)`. Also when disabled: the delay is kept.
#[test]
fn live_lookahead_change_delays_by_the_new_latency() {
    for &fs in &[48_000.0f32, 44_100.0] {
        let ch = 2;
        let xf = xfade_frames(fs);
        for enabled in [true, false] {
            let p0 = LimiterParams {
                enabled,
                ..params(0.0, 1.0, 100.0, 5.0)
            };
            let mut l = make(fs, ch, &p0);
            silence(&mut l, 1000, ch); // no longer fresh: changes cross-fade
            let what = format!("fs={fs} enabled={enabled} initial");
            assert_eq!(impulse_delay(&mut l, ch, &what), l.latency_frames());
            // Lookahead values set in turn; a second value is set while the
            // cross-fade to the first is running (queued).
            let steps: [&[f32]; 7] = [
                &[12.0],
                &[0.0],
                &[20.0],
                &[3.3],
                &[0.0],
                &[7.0, 2.0],
                &[2.0],
            ];
            for seq in steps {
                for (k, &ms) in seq.iter().enumerate() {
                    l.set_params(&LimiterParams {
                        lookahead_ms: ms,
                        ..p0
                    });
                    assert_eq!(
                        l.latency_frames(),
                        strom_dsp::limiter::lookahead_frames(ms, fs),
                        "reported latency follows the property at once"
                    );
                    if k + 1 < seq.len() {
                        silence(&mut l, xf / 4, ch);
                    }
                }
                // Both cross-fades (if one was queued), and enough silence
                // that a longer tap no longer reaches the previous impulse
                // (a longer delay replays what is already in the line).
                let lmax = strom_dsp::limiter::lookahead_frames(MAX_LOOKAHEAD_MS, fs);
                silence(&mut l, 2 * xf + lmax + 64, ch);
                let ms = *seq.last().unwrap();
                let what = format!("fs={fs} enabled={enabled} → {seq:?}");
                let d = impulse_delay(&mut l, ch, &what);
                assert_eq!(d, l.latency_frames(), "{what}");
                assert_eq!(d, strom_dsp::limiter::lookahead_frames(ms, fs), "{what}");
            }
            assert_eq!(l.safety_clip_count(), 0);
        }
    }
}

/// The cross-fade between the old and the new delay tap is the documented
/// raised cosine over 5 ms. On a linear ramp the two taps differ by a known
/// constant, so the weight `w` of the new tap can be read from each output
/// frame. Before the change the output is the old tap and after the
/// cross-fade the new one, bit for bit. In between there are exactly
/// `N = round(5 ms · fs)` mixed frames with weights strictly rising inside
/// (0, 1) and point-symmetric about the middle (`w[k] + w[N − 1 − k] = 1`,
/// the defining property of a raised-cosine fade). Click bound: no output
/// step exceeds the ramp's own step plus `ΔL·b·π / (2(N + 1))`, the largest
/// per-frame change of a raised cosine of `N + 1` steps times the tap
/// difference.
#[test]
fn lookahead_crossfade_is_a_raised_cosine_over_5_ms() {
    for &fs in &[48_000.0f32, 44_100.0] {
        let n_x = xfade_frames(fs);
        for &(from, to) in &[(2.0f32, 7.0f32), (7.0, 2.0), (0.0, 5.0), (5.0, 0.0)] {
            let (la, lb) = (
                strom_dsp::limiter::lookahead_frames(from, fs),
                strom_dsp::limiter::lookahead_frames(to, fs),
            );
            let total = 4000;
            let b = 0.8 / total as f64; // ramp slope per frame, stays below 1
            let input: Vec<f32> = (0..total).map(|n| (0.05 + b * n as f64) as f32).collect();
            let mut l = make(fs, 1, &params(0.0, 1.0, 100.0, from));
            let mut out = input.clone();
            let split = 1500;
            l.process(&mut out[..split]);
            l.set_params(&params(0.0, 1.0, 100.0, to));
            run(&mut l, &mut out[split..], 1);
            let what = format!("fs={fs} {from} → {to} ms");
            for n in la..split {
                assert_eq!(out[n], input[n - la], "{what}: old tap at {n}");
            }
            let dx = input[split - lb] as f64 - input[split - la] as f64;
            assert!(dx.abs() > 0.01, "{what}: taps too close to measure");
            let w: Vec<f64> = (split..split + n_x)
                .map(|n| {
                    let (xa, xb) = (input[n - la] as f64, input[n - lb] as f64);
                    (out[n] as f64 - xa) / (xb - xa)
                })
                .collect();
            for n in split + n_x..total {
                assert_eq!(out[n], input[n - lb], "{what}: new tap at {n}");
            }
            // Weights are recovered to about 1e-6 (f32 rounding of a sample
            // below 1 over a tap difference of at least 0.04).
            let tol = 1e-5;
            assert!(
                w[0] > 0.0 && w[n_x - 1] < 1.0,
                "{what}: {:?}",
                (w[0], w[n_x - 1])
            );
            for k in 1..n_x {
                assert!(w[k] > w[k - 1], "{what}: weight not rising at {k}");
            }
            for k in 0..n_x {
                let sym = w[k] + w[n_x - 1 - k];
                assert!(
                    (sym - 1.0).abs() < tol,
                    "{what}: w[{k}] + w[N−1−{k}] = {sym}"
                );
            }
            let bound = b + dx.abs() * std::f64::consts::PI / (2.0 * (n_x + 1) as f64) + 1e-6;
            for n in split..total {
                let step = (out[n] as f64 - out[n - 1] as f64).abs();
                assert!(step <= bound, "{what}: step {step} > {bound} at {n}");
            }
        }
    }
}

/// While the required gain is constant (a steady DC level 12 dB over the
/// ceiling), changes of attack, release (and so the hold) and lookahead
/// leave the gain where it is: every rebuild replays the base window exactly
/// and the cross-fade mixes two equal samples. The output stays at its
/// settled value (within 1e-6, two f32 rounding steps), never above the
/// ceiling, and the safety clip stays idle.
#[test]
fn steady_limiting_is_unchanged_by_shape_and_lookahead_changes() {
    let fs = 48_000.0f32;
    let thr = -6.0;
    let changes = [
        params(thr, 5.0, 100.0, 5.0),
        params(thr, 0.01, 100.0, 5.0),
        params(thr, 20.0, 5.0, 5.0),
        params(thr, 20.0, 2000.0, 20.0),
        params(thr, 1.0, 1.0, 20.0),
        params(thr, 50.0, 300.0, 0.0),
        params(thr, 3.0, 10.0, 8.0),
        params(thr, 19.0, 100.0, 20.0),
        params(thr, 20.0, 100.0, 20.0),
        params(thr, 2.0, 50.0, 1.0),
    ];
    let mut l = make(fs, 1, &params(thr, 1.0, 100.0, 5.0));
    let mut x = vec![2.0f32; 9600];
    l.process(&mut x);
    let y0 = x[9599];
    let c = l.ceiling_linear();
    assert!(
        y0 <= c && y0 as f64 >= c as f64 * (1.0 - 1e-6),
        "settled at {y0}, ceiling {c}"
    );
    for (k, p) in changes.iter().enumerate() {
        l.set_params(p);
        let mut x = vec![2.0f32; 480]; // 10 ms, longer than a cross-fade
        run(&mut l, &mut x, 1);
        for (n, &y) in x.iter().enumerate() {
            assert!(y <= c, "change {k}: {y} over the ceiling at {n}");
            assert!(
                (y - y0).abs() <= 1e-6,
                "change {k} {p:?}: {y} vs {y0} at {n}"
            );
        }
    }
    assert_eq!(l.safety_clip_count(), 0);
}

/// A lookahead increase larger than the old window (lookahead + hold)
/// makes the new, longer tap output audio again that left through the old
/// tap before the change. The module docs' cross-fade argument still
/// applies: the window is widened to cover both taps and its base part is
/// replayed from the stored history, so the old loud burst is limited the
/// second time too. The ceiling holds with no help from the safety clip.
#[test]
fn lookahead_increase_limits_the_replayed_audio() {
    let fs = 48_000.0f32;
    let thr = -6.0;
    for &(rel, from, to) in &[
        (1.0f32, 2.0f32, 20.0f32),
        (1.0, 0.0, 15.0),
        (3.0, 1.0, 12.0),
    ] {
        for after in [50usize, 200, 400, 600] {
            let mut l = make(fs, 1, &params(thr, 1.0, rel, from));
            let mut x = vec![0.01f32; 4000];
            for v in &mut x[1000..1300] {
                *v = 2.0;
            }
            let split = 1300 + after;
            l.process(&mut x[..split]);
            l.set_params(&params(thr, 1.0, rel, to));
            l.process(&mut x[split..]);
            let what = format!("release {rel}, {from} → {to} ms, {after} after the burst");
            assert_ceiling(&x, thr, &what);
            assert_eq!(l.safety_clip_count(), 0, "{what}: safety clip used");
        }
    }
}

/// A rebuild of the gain pipeline (on an attack or hold change) replays it
/// from the stored history. When the geometry ends up as it was (attack or
/// release changed and changed back before the next buffer), the replay
/// must restore the exact state: the output is bit-identical to a limiter
/// that saw no change. One setting uses the largest pipeline (lookahead
/// 20 ms, attack capped at the lookahead, hold 20 ms), where the replay
/// reaches back `a + L + H + 1` frames, the history's full documented depth.
#[test]
fn rebuild_with_unchanged_geometry_is_invisible() {
    for &fs in &[48_000.0f32, 44_100.0, 8_000.0] {
        for base in [
            params(-6.0, 20.0, 100.0, 20.0),
            params(-3.0, 1.0, 100.0, 5.0),
        ] {
            let wiggles = [
                LimiterParams {
                    attack_ms: base.attack_ms * 0.5,
                    ..base
                },
                LimiterParams {
                    release_ms: 5.0,
                    ..base
                },
            ];
            let frames = (fs as usize) / 2;
            let input = bursts(frames, 2, 0x9e37);
            let mut plain = make(fs, 2, &base);
            let mut wiggled = make(fs, 2, &base);
            let (mut a, mut b) = (input.clone(), input);
            let block = 200;
            for (k, (ca, cb)) in a
                .chunks_mut(block * 2)
                .zip(b.chunks_mut(block * 2))
                .enumerate()
            {
                if k % 3 == 1 {
                    wiggled.set_params(&wiggles[k % 2]);
                    wiggled.set_params(&base);
                }
                plain.process(ca);
                wiggled.process(cb);
            }
            let first = a
                .iter()
                .zip(&b)
                .position(|(u, v)| u.to_bits() != v.to_bits());
            assert_eq!(first, None, "fs={fs} {base:?}: outputs differ");
        }
    }
}

/// With the limiter switched off while it is limiting, incoming frames
/// require unity gain and, once the limited frames have left, the gain
/// returns to exactly 1.0 by a linear ramp of 10 ms for the full scale
/// (module docs). Read frame by frame: the gain never rises by more than
/// `1/(10 ms · fs)` per frame, never exceeds 1, reaches exactly 1.0 after
/// `(1 − g0) · 10 ms · fs` frames of ramp (plus at most the attack length,
/// where the box ramp is still the slower of the two), and from then on the
/// output is the input delayed by the latency, bit for bit.
#[test]
fn disabling_ramps_the_gain_linearly_to_exact_unity() {
    for &fs in &[48_000.0f32, 44_100.0] {
        let p = params(-20.0, 1.0, 50.0, 5.0);
        let mut l = make(fs, 1, &p);
        let mut x = vec![1.0f32; 4800];
        l.process(&mut x);
        let g0 = l.current_gain();
        assert!((g0 - db(-20.0)).abs() < 1e-6, "settled gain {g0}");
        l.set_params(&LimiterParams {
            enabled: false,
            ..p
        });
        let step = 1.0 / (0.010 * fs as f64);
        let attack = (fs as f64 * 0.001).round();
        let d = l.latency_frames();
        let n_total = 4800;
        let input: Vec<f32> = (0..n_total).map(|n| 0.9 - 0.1 * ((n % 7) as f32)).collect();
        let mut out = input.clone();
        let mut gains = Vec::with_capacity(n_total);
        for f in out.chunks_mut(1) {
            l.process(f);
            gains.push(l.current_gain());
        }
        let what = format!("fs={fs}");
        let rise_start = gains.iter().position(|&g| g > g0).expect("gain never rose");
        let unity = gains
            .iter()
            .position(|&g| g == 1.0)
            .expect("gain never reached 1.0");
        for w in gains.windows(2) {
            let up = w[1] - w[0];
            assert!(up <= step + 1e-12, "{what}: gain rose by {up} > {step}");
            assert!(w[1] <= 1.0, "{what}: gain {} over unity", w[1]);
        }
        let ramp = (unity - rise_start + 1) as f64;
        let expect = (1.0 - g0) / step;
        assert!(
            ramp >= expect - 1.0 && ramp <= expect + attack + 2.0,
            "{what}: ramp of {ramp} frames, expected {expect}"
        );
        for n in unity + 1..n_total {
            assert_eq!(gains[n], 1.0, "{what}: gain left unity at {n}");
            if n >= d {
                assert_eq!(
                    out[n].to_bits(),
                    input[n - d].to_bits(),
                    "{what}: not a pure delay at {n}"
                );
            }
        }
    }
}

/// The safety clip is a counted guard (module docs): a NaN output sample,
/// only possible from NaN input, is replaced by 0 and counted. Every NaN
/// input sample comes out as one zeroed, counted sample in the single-tap
/// path. Inside a lookahead cross-fade a sample is read from both taps, so
/// it is counted once per tap it leaves through.
#[test]
fn safety_clip_replaces_and_counts_nan() {
    let fs = 48_000.0f32;
    for ch in [1usize, 2, 3] {
        let mut l = make(fs, ch, &params(-1.0, 1.0, 100.0, 2.0));
        let mut x = vec![0.25f32; 3000 * ch];
        let nans = [100usize, 101, 2000];
        for &n in &nans {
            x[n * ch + ch - 1] = f32::NAN;
        }
        run(&mut l, &mut x, ch);
        let d = l.latency_frames();
        for &n in &nans {
            assert_eq!(x[(n + d) * ch + ch - 1], 0.0);
        }
        assert!(x.iter().all(|v| v.is_finite()));
        assert_eq!(l.safety_clip_count(), nans.len() as u64, "ch={ch}");

        // Cross-fade 2 → 7 ms starting at input frame `m0`: a NaN 50 frames
        // earlier leaves through the old tap inside the fade and through the
        // new tap after it: two counted samples.
        let mut l = make(fs, ch, &params(-1.0, 1.0, 100.0, 2.0));
        let m0 = 1000;
        let mut x = vec![0.25f32; 2000 * ch];
        x[(m0 - 50) * ch] = f32::NAN;
        l.process(&mut x[..m0 * ch]);
        assert_eq!(l.safety_clip_count(), 0);
        l.set_params(&params(-1.0, 1.0, 100.0, 7.0));
        l.process(&mut x[m0 * ch..]);
        assert!(x.iter().all(|v| v.is_finite()));
        assert_eq!(l.safety_clip_count(), 2, "ch={ch} cross-fade");
    }
}

/// The enforced ceiling is the largest `f32` not above `10^(dB/20)`
/// (`ceiling_linear` docs): not above it, and the next `f32` up is.
#[test]
fn ceiling_is_the_largest_f32_not_above_the_threshold() {
    let mut l = Limiter::new();
    l.configure(48_000.0, 1);
    for k in 0..=240 {
        let thr = -(k as f32) * 0.25;
        for t in [thr, (thr - 0.0206).max(-60.0)] {
            l.set_params(&params(t, 1.0, 100.0, 5.0));
            let c = l.ceiling_linear();
            let exact = 10f64.powf(t as f64 / 20.0);
            let up = f32::from_bits(c.to_bits() + 1);
            assert!(c as f64 <= exact, "thr={t}: {c} above {exact}");
            assert!(up as f64 > exact, "thr={t}: {c} is not the largest");
        }
    }
    l.set_params(&params(0.0, 1.0, 100.0, 5.0));
    assert_eq!(l.ceiling_linear(), 1.0);
}

/// `params()` returns the sanitised parameters last set, not the defaults.
#[test]
fn params_returns_what_was_set() {
    let mut l = Limiter::new();
    l.configure(48_000.0, 2);
    for p in [
        params(-12.5, 3.25, 640.0, 7.5),
        LimiterParams {
            enabled: false,
            ..params(-0.5, 0.02, 1.5, 0.0)
        },
    ] {
        l.set_params(&p);
        assert_eq!(*l.params(), p);
    }
}

/// A rate the limiter cannot use (0, negative, infinite, NaN) falls back to
/// the same 48 kHz in `lookahead_frames` and in `configure`, so the reported
/// latency is the one at 48 kHz.
#[test]
fn unusable_rates_fall_back_to_48_khz() {
    let at48 = strom_dsp::limiter::lookahead_frames(5.0, 48_000.0);
    for fs in [0.0f32, -5.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        assert_eq!(
            strom_dsp::limiter::lookahead_frames(5.0, fs),
            at48,
            "fs={fs}"
        );
        let mut l = Limiter::new();
        l.configure(fs, 1);
        assert_eq!(l.latency_frames(), at48, "fs={fs}");
    }
}

/// A SIMD level forced with `set_simd_level` is kept across `configure`;
/// without one, `configure` detects the best level.
#[test]
fn simd_level_forced_or_detected() {
    use strom_dsp::simd::{detect, Level};
    let mut l = Limiter::new();
    l.set_simd_level(Level::fallback());
    l.configure(48_000.0, 2);
    assert!(l.simd_level().is_fallback());
    l.configure(44_100.0, 1);
    assert!(l.simd_level().is_fallback());
    let mut l = Limiter::new();
    l.configure(48_000.0, 2);
    assert_eq!(format!("{:?}", l.simd_level()), format!("{:?}", detect()));
}
