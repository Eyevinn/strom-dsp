//! True-peak limiter tests: the BS.1770 true-peak guarantee (measured with
//! the `ebur128` crate in true-peak mode), latency, bypass, safety-stage
//! idleness, bass THD, robustness and real-time safety.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::f64::consts::PI;

use ebur128::{EbuR128, Mode};
use proptest::prelude::*;
use strom_dsp::truepeak::{
    latency_frames_for, TruePeakLimiter, TruePeakLimiterParams, DETECTOR_DELAY_FRAMES, GUARD_FRAMES,
};
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

/// The guarantee's sample rates.
const RATES: [f32; 3] = [44_100.0, 48_000.0, 96_000.0];
/// Allowed true-peak excess over the ceiling (dB).
const TOLERANCE_DB: f64 = 0.1;

fn lin(db: f64) -> f64 {
    10f64.powf(db / 20.0)
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

fn params(ceiling: f32, release: f32, lookahead: f32) -> TruePeakLimiterParams {
    TruePeakLimiterParams {
        enabled: true,
        ceiling_dbtp: ceiling,
        release_ms: release,
        lookahead_ms: lookahead,
    }
}

fn make(fs: f32, ch: usize, p: &TruePeakLimiterParams) -> TruePeakLimiter {
    let mut l = TruePeakLimiter::new();
    l.set_params(p);
    l.configure(fs, ch);
    l
}

/// Process in uneven block sizes to exercise block boundaries.
fn run(l: &mut TruePeakLimiter, buf: &mut [f32], ch: usize) {
    let sizes = [1usize, 7, 64, 333, 1024, 2, 480];
    let mut i = 0;
    let mut k = 0;
    while i < buf.len() {
        let n = (sizes[k % sizes.len()] * ch).min(buf.len() - i);
        l.process(&mut buf[i..i + n]);
        i += n;
        k += 1;
    }
}

/// BS.1770 true peak (dBTP) of interleaved `out`, the largest over channels,
/// measured by the `ebur128` crate.
fn true_peak_dbtp(out: &[f32], ch: usize, fs: f32) -> f64 {
    let mut m = EbuR128::new(ch as u32, fs as u32, Mode::TRUE_PEAK).unwrap();
    m.add_frames_f32(out).unwrap();
    let tp = (0..ch as u32)
        .map(|c| m.true_peak(c).unwrap())
        .fold(0.0f64, f64::max);
    20.0 * tp.max(1e-20).log10()
}

/// Run `sig` through a fresh limiter and return (true-peak excess over the
/// ceiling in dB, the limiter).
fn excess(
    sig: &[f32],
    ch: usize,
    fs: f32,
    p: &TruePeakLimiterParams,
) -> (f64, TruePeakLimiter, Vec<f32>) {
    let mut l = make(fs, ch, p);
    let mut buf = sig.to_vec();
    run(&mut l, &mut buf, ch);
    // Flush the latency so the tail of the signal is output too.
    let mut tail = vec![0.0f32; (l.latency_frames() + 16) * ch];
    l.process(&mut tail);
    buf.extend_from_slice(&tail);
    assert!(buf.iter().all(|v| v.is_finite()));
    let tp = true_peak_dbtp(&buf, ch, fs);
    (tp - p.ceiling_dbtp as f64, l, buf)
}

fn white_noise(frames: usize, ch: usize, amp: f64, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed);
    (0..frames * ch).map(|_| (amp * r.next()) as f32).collect()
}

/// First difference of white noise: energy rising towards Nyquist.
fn hf_noise(frames: usize, ch: usize, amp: f64, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed);
    let mut prev = vec![0.0f64; ch];
    let mut out = Vec::with_capacity(frames * ch);
    for _ in 0..frames {
        for p in prev.iter_mut() {
            let v = r.next();
            out.push((0.5 * amp * (v - *p)) as f32);
            *p = v;
        }
    }
    out
}

fn square(frames: usize, ch: usize, amp: f64, period: usize) -> Vec<f32> {
    (0..frames)
        .flat_map(|n| {
            let v = if (n % period) < period / 2 { amp } else { -amp };
            vec![v as f32; ch]
        })
        .collect()
}

fn impulses(frames: usize, ch: usize, amp: f64, spacing: usize) -> Vec<f32> {
    (0..frames)
        .flat_map(|n| {
            let v = if n % spacing == spacing / 2 { amp } else { 0.0 };
            vec![v as f32; ch]
        })
        .collect()
}

/// Sine of `freq` Hz, amplitude `amp`, start phase `phase` (radians).
fn sine(frames: usize, ch: usize, amp: f64, freq: f64, phase: f64, fs: f64) -> Vec<f32> {
    (0..frames)
        .flat_map(|n| {
            let v = amp * (2.0 * PI * freq * n as f64 / fs + phase).sin();
            vec![v as f32; ch]
        })
        .collect()
}

/// The classic inter-sample-peak signal: fs/4 at 45° phase, whose samples
/// are all ±amp/√2 while its true peak is amp. Scaled so the *samples* are
/// at full scale (true peak +3 dBTP).
fn quarter_rate_45(frames: usize, ch: usize, fs: f64) -> Vec<f32> {
    sine(frames, ch, 2f64.sqrt(), fs / 4.0, PI / 4.0, fs)
}

/// Loud high-frequency material right at the ceiling, then a single huge
/// impulse, then the same material again: the gain drops fast while the
/// interpolator sees high-frequency content, the worst case for the
/// gain-modulation term (module docs).
fn hf_then_spike(frames: usize, ch: usize, fs: f64, freq: f64, ceiling_db: f64) -> Vec<f32> {
    let mut s = sine(frames, ch, lin(ceiling_db + 6.0), freq, 0.3, fs);
    for k in 1..4 {
        let at = (k * frames / 4) * ch;
        for c in 0..ch {
            s[at + c] = 30.0;
        }
    }
    s
}

/// Speech/music-like multitone with a slow envelope (normal material).
fn music_like(frames: usize, ch: usize, fs: f64, seed: u64, amp: f64) -> Vec<f32> {
    let mut r = Rng(seed);
    let tones: Vec<(f64, f64)> = (0..8)
        .map(|_| {
            (
                80.0 + 4000.0 * (0.5 + 0.5 * r.next()).powi(2),
                0.5 + 0.5 * r.next(),
            )
        })
        .collect();
    (0..frames)
        .flat_map(|n| {
            let t = n as f64 / fs;
            let env = 0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin();
            let v: f64 = tones
                .iter()
                .map(|(f, a)| a * (2.0 * PI * f * t).sin())
                .sum::<f64>()
                / 8.0;
            vec![(amp * env * v) as f32; ch]
        })
        .collect()
}

/// The explicit nasty cases (signals that provoke inter-sample peaks) at `fs`.
fn nasty_cases(fs: f32, ch: usize) -> Vec<(String, Vec<f32>)> {
    let f = fs as f64;
    let n = (0.25 * f) as usize;
    let mut v = vec![
        ("square full scale".into(), square(n, ch, 1.0, 100)),
        ("square +20 dBFS".into(), square(n, ch, 10.0, 37)),
        ("square fast".into(), square(n, ch, 1.0, 4)),
        ("impulses".into(), impulses(n, ch, 1.0, 1000)),
        ("impulses dense".into(), impulses(n, ch, 10.0, 3)),
        ("fs/4 at 45°".into(), quarter_rate_45(n, ch, f)),
        ("noise +20 dBFS".into(), white_noise(n, ch, 10.0, 7)),
        ("hf noise +20 dBFS".into(), hf_noise(n, ch, 10.0, 11)),
    ];
    for (i, freq) in [15_000.0, 17_000.0, 18_500.0, 19_500.0, 20_500.0, 21_500.0]
        .into_iter()
        .chain([0.47 * f, 0.49 * f, 0.499 * f])
        .enumerate()
    {
        if freq >= 0.5 * f {
            continue;
        }
        for (j, ph) in [0.0, 0.7, PI / 4.0, 1.9].into_iter().enumerate() {
            v.push((
                format!("sine {freq:.0} Hz phase {j}"),
                sine(n, ch, 4.0, freq, ph + i as f64, f),
            ));
        }
        v.push((
            format!("hf then spike {freq:.0} Hz"),
            hf_then_spike(n, ch, f, freq, -1.0),
        ));
    }
    v
}

// ---------------------------------------------------------------------------
// The guarantee.

/// The explicit nasty cases at 44.1, 48 and 96 kHz, mono and stereo, at
/// several ceilings, the shortest and default lookaheads and fast/slow
/// release.
#[test]
fn true_peak_ceiling_holds_on_nasty_signals() {
    let mut worst = (f64::NEG_INFINITY, String::new());
    for fs in RATES {
        for ch in [1usize, 2] {
            for p in [
                params(-1.0, 100.0, 5.0),
                params(-1.0, 1.0, 0.5),
                params(0.0, 50.0, 1.5),
                params(-6.0, 1000.0, 20.0),
            ] {
                for (name, sig) in nasty_cases(fs, ch) {
                    let (e, l, _) = excess(&sig, ch, fs, &p);
                    let what = format!("{name} @ {fs} Hz ch {ch} {p:?}");
                    assert!(e <= TOLERANCE_DB, "{what}: excess {e:.4} dB");
                    assert_eq!(l.safety_clip_count(), 0, "{what}");
                    if e > worst.0 {
                        worst = (e, what);
                    }
                }
            }
        }
    }
    eprintln!("worst true-peak excess: {:.4} dB ({})", worst.0, worst.1);
}

#[derive(Debug, Clone)]
enum Kind {
    Noise(f64),
    HfNoise(f64),
    Square(f64, usize),
    Impulses(f64, usize),
    Sine(f64, f64, f64),
    Spike(f64),
    Music(f64),
}

fn kind_strategy() -> impl Strategy<Value = Kind> {
    prop_oneof![
        (0.1f64..10.0).prop_map(Kind::Noise),
        (0.1f64..10.0).prop_map(Kind::HfNoise),
        (0.1f64..10.0, 2usize..400).prop_map(|(a, p)| Kind::Square(a, p)),
        (0.1f64..30.0, 1usize..2000).prop_map(|(a, s)| Kind::Impulses(a, s)),
        (0.1f64..10.0, 0.0f64..0.5, 0.0f64..6.3).prop_map(|(a, f, ph)| Kind::Sine(a, f, ph)),
        (0.05f64..0.499).prop_map(Kind::Spike),
        (0.1f64..10.0).prop_map(Kind::Music),
    ]
}

fn signal(kind: &Kind, frames: usize, ch: usize, fs: f64, seed: u64) -> Vec<f32> {
    match *kind {
        Kind::Noise(a) => white_noise(frames, ch, a, seed),
        Kind::HfNoise(a) => hf_noise(frames, ch, a, seed),
        Kind::Square(a, p) => square(frames, ch, a, p),
        Kind::Impulses(a, s) => impulses(frames, ch, a, s),
        Kind::Sine(a, f, ph) => sine(frames, ch, a, f * fs, ph, fs),
        Kind::Spike(f) => hf_then_spike(frames, ch, fs, f * fs, -1.0),
        Kind::Music(a) => music_like(frames, ch, fs, seed, a),
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: if cfg!(debug_assertions) { 24 } else { 160 },
        .. ProptestConfig::default()
    })]

    /// Property: for any signal and parameters, at 44.1/48/96 kHz, the
    /// output's BS.1770 true peak is at most ceiling + 0.1 dB, and the safety
    /// stage never acts.
    #[test]
    fn prop_true_peak_guarantee(
        kind in kind_strategy(),
        rate in 0usize..3,
        ch in 1usize..3,
        ceiling in -20.0f32..0.0,
        release in 1.0f32..500.0,
        lookahead in 0.5f32..20.0,
        seed in 1u64..u64::MAX,
    ) {
        let fs = RATES[rate];
        let sig = signal(&kind, (0.2 * fs) as usize, ch, fs as f64, seed);
        let p = params(ceiling, release, lookahead);
        let (e, l, _) = excess(&sig, ch, fs, &p);
        prop_assert!(e <= TOLERANCE_DB, "{kind:?} @ {fs}: excess {e:.4} dB");
        prop_assert_eq!(l.safety_clip_count(), 0);
    }
}

/// Excess of a single click (short snippet of arbitrary samples, in
/// silence) through the fastest configuration.
fn click_excess(snippet: &[f32], fs: f32, lookahead: f32) -> f64 {
    let mut sig = vec![0.0f32; 200];
    sig.extend_from_slice(snippet);
    sig.resize(sig.len() + 200, 0.0);
    excess(&sig, 1, fs, &params(-1.0, 1.0, lookahead)).0
}

/// Short full-band clicks are where a BS.1770 meter's short interpolator
/// reads most above ours (module docs, "Interpolator mismatch"): random
/// clicks, then an adversarial search that hill-climbs click samples to
/// maximise the measured excess.
#[test]
fn true_peak_ceiling_holds_on_random_and_adversarial_clicks() {
    let (randoms, restarts, iters) = if cfg!(debug_assertions) {
        (100, 1, 60)
    } else {
        (1500, 3, 600)
    };
    let mut worst = f64::NEG_INFINITY;
    let mut r = Rng(0x5eed);
    for fs in RATES {
        for _ in 0..randoms {
            let len = 1 + ((r.next() + 1.0) * 20.0) as usize;
            let amp = 0.5 + 5.0 * (r.next() + 1.0);
            let snippet: Vec<f32> = (0..len).map(|_| (amp * r.next()) as f32).collect();
            let e = click_excess(&snippet, fs, 5.0);
            assert!(e <= TOLERANCE_DB, "random click @ {fs}: {e:.4} dB");
            worst = worst.max(e);
        }
        for k in 0..restarts {
            let la = if k % 2 == 0 { 0.5 } else { 5.0 };
            let len = 8 + ((r.next() + 1.0) * 25.0) as usize;
            let mut x: Vec<f32> = (0..len).map(|_| (2.0 * r.next()) as f32).collect();
            let mut best = click_excess(&x, fs, la);
            for it in 0..iters {
                let mut y = x.clone();
                let i = ((r.next() + 1.0) * 0.5 * len as f64) as usize % len;
                let step = if it < iters / 2 { 1.0 } else { 0.2 };
                y[i] = (y[i] as f64 + step * r.next()).clamp(-4.0, 4.0) as f32;
                let e = click_excess(&y, fs, la);
                if e > best {
                    best = e;
                    x = y;
                }
            }
            assert!(
                best <= TOLERANCE_DB,
                "adversarial click @ {fs}: {best:.4} dB"
            );
            worst = worst.max(best);
        }
    }
    eprintln!("clicks: worst true-peak excess {worst:.4} dB");
}

// ---------------------------------------------------------------------------
// Latency and bypass.

#[test]
fn latency_is_lookahead_plus_detector_delay() {
    for (fs, la, want) in [
        (44_100.0f32, 5.0f32, 221 + 32),
        (48_000.0, 5.0, 240 + 32),
        (96_000.0, 5.0, 480 + 32),
        (48_000.0, 0.5, 24 + 32),
        (48_000.0, 20.0, 960 + 32),
    ] {
        let l = make(fs, 2, &params(-1.0, 100.0, la));
        assert_eq!(l.latency_frames(), want, "{fs} Hz, {la} ms");
        assert_eq!(latency_frames_for(la, fs), want);
    }
    assert_eq!(DETECTOR_DELAY_FRAMES + GUARD_FRAMES, 32);
    let mut os = strom_dsp::oversample::Oversampler::new();
    os.configure(strom_dsp::truepeak::OVERSAMPLE, 1);
    assert_eq!(os.up_latency_frames(), DETECTOR_DELAY_FRAMES);
    // Lookahead is clamped to 0.5 … 20 ms.
    assert_eq!(latency_frames_for(0.0, 48_000.0), 24 + 32);
    assert_eq!(latency_frames_for(f32::NAN, 48_000.0), 240 + 32);
}

/// An impulse well below the ceiling comes out exactly at the reported
/// latency, unchanged, with `enabled` true and false; all other output
/// samples are exactly 0.
#[test]
fn impulse_delay_equals_reported_latency_enabled_and_disabled() {
    for fs in RATES {
        for enabled in [true, false] {
            let p = TruePeakLimiterParams {
                enabled,
                ..params(-1.0, 100.0, 5.0)
            };
            let mut l = make(fs, 2, &p);
            let lat = l.latency_frames();
            let n = lat + 500;
            let mut buf = vec![0.0f32; n * 2];
            buf[100 * 2] = 0.25;
            buf[100 * 2 + 1] = -0.25;
            run(&mut l, &mut buf, 2);
            for (i, f) in buf.chunks_exact(2).enumerate() {
                if i == 100 + lat {
                    assert_eq!(f, [0.25, -0.25], "{fs} enabled {enabled}");
                } else {
                    assert_eq!(f, [0.0, 0.0], "{fs} enabled {enabled}: frame {i}");
                }
            }
        }
    }
}

/// After `enabled = false` (and the bypass ramp), the output is the input
/// delayed by exactly the latency, bit for bit, even for input far above the
/// ceiling.
#[test]
fn disabled_is_exact_delay_of_same_latency() {
    let fs = 48_000.0;
    let mut l = make(fs, 2, &params(-6.0, 50.0, 5.0));
    let lat = l.latency_frames();
    let sig = white_noise(48_000, 2, 3.0, 3);
    let mut a = sig[..24_000 * 2].to_vec();
    run(&mut l, &mut a, 2);
    l.set_params(&TruePeakLimiterParams {
        enabled: false,
        ..params(-6.0, 50.0, 5.0)
    });
    assert_eq!(l.latency_frames(), lat);
    let mut b = sig[24_000 * 2..].to_vec();
    run(&mut l, &mut b, 2);
    // After the frames limited before the switch have left (latency + 20 ms
    // hold), the 10 ms ramp to unity and a margin: an exact delay.
    let start = lat + 960 + 480 + 200;
    for i in start..24_000 {
        assert_eq!(b[i * 2], sig[(24_000 + i - lat) * 2], "frame {i}");
        assert_eq!(b[i * 2 + 1], sig[(24_000 + i - lat) * 2 + 1], "frame {i}");
    }
}

// ---------------------------------------------------------------------------
// Safety stage, linking, behaviour on normal material.

/// On normal material (multitone with an envelope, noise, speech-like
/// bursts) driven well into limiting, the safety stage never acts, and the
/// detector does the work: the true peak is at the ceiling (within the
/// tolerance) and gain reduction actually happened.
#[test]
fn safety_stage_idle_on_normal_material() {
    for fs in RATES {
        for (i, sig) in [
            music_like(fs as usize, 2, fs as f64, 1, 4.0),
            white_noise(fs as usize, 2, 2.0, 5),
            music_like(fs as usize, 2, fs as f64, 9, 1.5),
        ]
        .into_iter()
        .enumerate()
        {
            let (e, l, _) = excess(&sig, 2, fs, &params(-1.0, 100.0, 5.0));
            assert_eq!(l.safety_clip_count(), 0, "{fs} case {i}");
            assert!(e <= TOLERANCE_DB, "{fs} case {i}: {e:.4}");
            // White noise (case 1) is full-band: the high-frequency term
            // deliberately limits it further (up to ~2 dB at 44.1 kHz).
            let want = (true_peak_dbtp(&sig, 2, fs) + 1.0).min(0.0);
            if i != 1 {
                assert!(e > want - 0.3, "{fs} case {i}: over-limited, {e:.4} dB");
            }
            eprintln!("normal material {fs} case {i}: true peak {e:+.3} dB re ceiling");
        }
    }
}

/// Programme below ~5 kHz pays (almost) nothing for the high-frequency
/// term: a 1 kHz tone 6 dB over the ceiling is limited to within 0.05 dB of
/// it.
#[test]
fn low_frequency_material_is_limited_to_the_ceiling() {
    for fs in RATES {
        let n = fs as usize;
        let mut sig = sine(n, 1, lin(5.0), 1000.0, 0.0, fs as f64);
        for (k, v) in sig.iter_mut().enumerate().take(1000) {
            *v *= k as f32 / 1000.0;
        }
        let (e, l, out) = excess(&sig, 1, fs, &params(-1.0, 100.0, 5.0));
        assert!(e <= TOLERANCE_DB && e > -0.05, "{fs}: {e:.4} dB");
        assert_eq!(l.safety_clip_count(), 0);
        // Sample peak never above the ceiling (exact).
        assert!(out.iter().all(|v| v.abs() <= l.ceiling_linear()));
    }
}

/// One loud channel pulls the gain of all channels down (linked).
#[test]
fn channels_are_linked() {
    let fs = 48_000.0;
    let n = 9600;
    let mut sig = vec![0.0f32; n * 2];
    for i in 0..n {
        sig[i * 2] = (4.0 * (2.0 * PI * 200.0 * i as f64 / fs as f64).sin()) as f32;
        sig[i * 2 + 1] = (0.1 * (2.0 * PI * 300.0 * i as f64 / fs as f64).sin()) as f32;
    }
    let mut l = make(fs, 2, &params(-1.0, 100.0, 5.0));
    let mut out = sig.clone();
    run(&mut l, &mut out, 2);
    let lat = l.latency_frames();
    let i = n - 100;
    let g = out[i * 2 + 1] as f64 / sig[(i - lat) * 2 + 1] as f64;
    assert!((g - l.current_gain()).abs() < 0.02, "gain {g}");
    assert!(l.gain_reduction_db() > 12.0);
}

// ---------------------------------------------------------------------------
// Bass THD at fast settings (as `limiter_thd.rs`).

fn dft_amplitude(x: &[f64], freq: f64, fs: f64) -> f64 {
    let w = 2.0 * PI * freq / fs;
    let (mut re, mut im) = (0.0, 0.0);
    for (n, &v) in x.iter().enumerate() {
        let ph = w * n as f64;
        re += v * ph.cos();
        im -= v * ph.sin();
    }
    2.0 * (re * re + im * im).sqrt() / x.len() as f64
}

fn thd(x: &[f64], freq: f64, fs: f64) -> f64 {
    let a1 = dft_amplitude(x, freq, fs);
    let h: f64 = (2..=10)
        .map(|k| dft_amplitude(x, k as f64 * freq, fs).powi(2))
        .sum();
    h.sqrt() / a1
}

/// −13 dBFS sines into a −25 dBTP ceiling, lookahead 5 ms, release 50 and
/// 100 ms (the limiter settings used when tuning it by ear): ≤ 1 % at 50 Hz, ≤ 0.3 %
/// at 100 Hz, ≤ 0.1 % at 200 Hz, as for the sample-peak limiter.
#[test]
fn bass_thd_at_fast_settings_is_low() {
    let fs = 48_000.0f32;
    for release in [50.0f32, 100.0] {
        for (f, limit) in [(50.0, 0.01), (100.0, 0.003), (200.0, 0.001)] {
            let frames = 3 * fs as usize;
            let sig = sine(frames, 2, lin(-13.0), f, 0.0, fs as f64);
            let mut l = make(fs, 2, &params(-25.0, release, 5.0));
            let mut out = sig.clone();
            for c in out.chunks_mut(1024 * 2) {
                l.process(c);
            }
            let last: Vec<f64> = out[(frames - fs as usize) * 2..]
                .iter()
                .step_by(2)
                .map(|&v| v as f64)
                .collect();
            let t = thd(&last, f, fs as f64);
            eprintln!(
                "true-peak limiter {f} Hz, release {release} ms: THD {:.5} %",
                100.0 * t
            );
            assert!(t <= limit, "{f} Hz release {release}: THD {t}");
            assert_eq!(l.safety_clip_count(), 0);
        }
    }
}

// ---------------------------------------------------------------------------
// Parameter changes, robustness, real-time safety.

/// Every parameter jumps between extremes while a loud signal flows: no
/// NaN, the sample peak stays under the ceiling in force, and the gain never
/// jumps up by more than the documented bypass/release slope allows.
#[test]
fn parameter_jumps_are_smooth_and_safe() {
    let fs = 48_000.0;
    let mut l = make(fs, 2, &params(-1.0, 100.0, 5.0));
    let sig = music_like(fs as usize * 2, 2, fs as f64, 4, 3.0);
    let mut prev_gain = 1.0f64;
    let mut max_rise = 0.0f64;
    let mut r = Rng(17);
    for (k, chunk) in sig.chunks(480 * 2).enumerate() {
        let p = TruePeakLimiterParams {
            enabled: k % 23 != 7,
            ceiling_dbtp: if k % 3 == 0 {
                -12.0
            } else {
                (r.next() * 6.0 - 6.0) as f32
            },
            release_ms: if k % 2 == 0 { 1.0 } else { 2000.0 },
            lookahead_ms: if k % 11 < 5 { 0.5 } else { 20.0 },
        };
        l.set_params(&p);
        let mut buf = chunk.to_vec();
        l.process(&mut buf);
        assert!(buf.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
        let g = l.current_gain();
        max_rise = max_rise.max(g - prev_gain);
        prev_gain = g;
    }
    // Over one 10 ms buffer the gain can rise at most by the 10 ms bypass
    // ramp (full scale) or a 1 ms release: never a step to unity.
    assert!(max_rise <= 1.0 + 1e-12);
    assert_eq!(l.safety_clip_count(), 0);
}

#[test]
fn extreme_params_rates_channels_no_nan() {
    for fs in [8_000.0f32, 22_050.0, 44_100.0, 96_000.0, 192_000.0] {
        for ch in [1usize, 2, 6] {
            for p in [
                params(-60.0, 1.0, 0.5),
                params(0.0, 2000.0, 20.0),
                params(-12.0, 1.0, 20.0),
            ] {
                let mut l = make(fs, ch, &p);
                let mut buf = white_noise(4096, ch, 100.0, 1);
                buf.extend(vec![0.0f32; 4096 * ch]);
                buf.extend(white_noise(1000, ch, 1e-30, 2));
                run(&mut l, &mut buf, ch);
                assert!(buf.iter().all(|v| v.is_finite()), "{fs} {ch} {p:?}");
                assert!(buf.iter().all(|v| v.abs() <= l.ceiling_linear()));
                assert_eq!(l.safety_clip_count(), 0);
                assert!(l.current_gain().is_finite());
                // No subnormal output after the loud part.
                assert!(buf.iter().all(|v| *v == 0.0 || v.is_normal()));
            }
        }
    }
}

#[test]
fn non_finite_input_and_params_are_handled() {
    let mut l = make(48_000.0, 2, &params(-1.0, 100.0, 5.0));
    let mut buf = white_noise(2000, 2, 0.5, 1);
    buf[100] = f32::NAN;
    buf[501] = f32::INFINITY;
    buf[900] = f32::NEG_INFINITY;
    run(&mut l, &mut buf, 2);
    let mut tail = white_noise(4000, 2, 0.5, 2);
    run(&mut l, &mut tail, 2);
    assert!(buf.iter().chain(&tail).all(|v| v.is_finite()));
    assert!(buf
        .iter()
        .chain(&tail)
        .all(|v| v.abs() <= l.ceiling_linear()));
    // The detector recovers: the last part is limited normally again.
    assert!(tail[6000..].iter().any(|v| v.abs() > 0.1));

    let mut l = TruePeakLimiter::new();
    l.set_params(&TruePeakLimiterParams {
        enabled: true,
        ceiling_dbtp: f32::NAN,
        release_ms: f32::INFINITY,
        lookahead_ms: -5.0,
    });
    let d = TruePeakLimiterParams::default();
    assert_eq!(l.params().ceiling_dbtp, d.ceiling_dbtp);
    assert_eq!(l.params().release_ms, d.release_ms);
    assert_eq!(l.params().lookahead_ms, 0.5);
    l.set_params(&params(10.0, 0.0, 100.0));
    assert_eq!(*l.params(), params(0.0, 1.0, 20.0));
}

#[test]
fn unconfigured_and_reset() {
    let mut l = TruePeakLimiter::new();
    let mut buf = vec![2.0f32; 64];
    l.process(&mut buf);
    assert!(buf.iter().all(|&v| v == 2.0));
    l.configure(48_000.0, 0);
    l.process(&mut buf);
    assert!(buf.iter().all(|&v| v == 2.0));

    let mut l = make(48_000.0, 2, &params(-1.0, 100.0, 5.0));
    let sig = white_noise(10_000, 2, 3.0, 8);
    let mut a = sig.clone();
    run(&mut l, &mut a, 2);
    l.reset();
    assert_eq!(l.current_gain(), 1.0);
    let mut b = sig.clone();
    run(&mut l, &mut b, 2);
    assert_eq!(a, b, "reset restores the initial state");
}

#[test]
fn no_allocation_in_process_set_params_reset() {
    for (fs, ch) in [(48_000.0f32, 2usize), (44_100.0, 1), (96_000.0, 6)] {
        let mut l = make(fs, ch, &params(-1.0, 100.0, 5.0));
        let sig = white_noise(20_000, ch, 4.0, 3);
        let mut buf = sig.clone();
        let mut r = Rng(9);
        let before = allocs();
        for (k, c) in buf.chunks_mut(997 * ch).enumerate() {
            l.set_params(&params(
                (r.next() * 6.0 - 6.0) as f32,
                (r.next() * 500.0 + 501.0) as f32,
                if k % 4 == 0 { 0.5 } else { 10.0 },
            ));
            l.process(c);
            if k == 10 {
                l.reset();
            }
        }
        let _ = l.gain_reduction_db();
        assert_eq!(allocs() - before, 0, "{fs} {ch}");
    }
}
