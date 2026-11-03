//! Sidechain (key input) for the gate and compressor, and the gate's expander
//! mode: the gain follows the key and not the programme, the key filter
//! works, the no-key path is unchanged, the expander's static curve matches
//! its formula, every SIMD level gives the same output, and nothing allocates.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use strom_dsp::biquad::uses_fma;
use strom_dsp::dynamics::{KeyDetector, KeyFilter};
use strom_dsp::gate::{GateMode, DEFAULT_EXPANDER_RATIO};
use strom_dsp::simd::{detect, Level};
use strom_dsp::util::{db_to_lin, lin_to_db};
use strom_dsp::{Compressor, CompressorParams, Gate, GateParams, Processor};

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

fn alloc_count() -> usize {
    ALLOCS.with(|c| c.get())
}

// ---------------------------------------------------------------------------
// Helpers.

const FS: f32 = 48_000.0;
/// Constant probe level, far below any threshold, used to read the gain.
const PROBE: f32 = 1e-5;

struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn ms(n_ms: f64) -> usize {
    (n_ms * 0.001 * FS as f64).round() as usize
}

/// Interleaved sine of `ch` channels at `freq` Hz and peak `amp`.
fn sine(amp: f32, freq: f64, frames: usize, ch: usize) -> Vec<f32> {
    (0..frames)
        .flat_map(|i| {
            let v = amp * (2.0 * std::f64::consts::PI * freq * i as f64 / FS as f64).sin() as f32;
            std::iter::repeat_n(v, ch)
        })
        .collect()
}

/// Constant-magnitude signal `±amp` (alternating sign), so a peak detector
/// reads exactly `amp` on every frame.
fn square_nyquist(amp: f32, frames: usize) -> Vec<f32> {
    (0..frames)
        .map(|i| if i % 2 == 0 { amp } else { -amp })
        .collect()
}

fn gate(ch: usize, p: GateParams) -> Gate {
    let mut g = Gate::new();
    g.set_params(&p);
    g.configure(FS, ch);
    g
}

fn compressor(ch: usize, p: CompressorParams) -> Compressor {
    let mut c = Compressor::new();
    c.set_params(&p);
    c.configure(FS, ch);
    c
}

fn comp_params() -> CompressorParams {
    CompressorParams {
        threshold: 0.1, // −20 dBFS
        ratio: 4.0,
        knee: 1.0, // hard knee
        attack_ms: 1.0,
        release_ms: 20.0,
        makeup_gain: 1.0,
    }
}

fn assert_bits_equal(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: sample {i}: {x} vs {y}");
    }
}

// ---------------------------------------------------------------------------
// The gain follows the key, not the programme.

#[test]
fn gate_follows_key_not_programme() {
    let p = GateParams::default(); // open −20, close −30, reduction −60
    let n = ms(300.0);

    // Loud programme (−3 dBFS), silent key: stays closed at `reduction`.
    let mut g = gate(2, p);
    let mut x = sine(0.7, 1000.0, n, 2);
    let key = vec![0.0f32; n * 2];
    g.process_with_key(&mut x, &key, 2);
    assert!(!g.is_open());
    assert!((lin_to_db(g.current_gain()) - p.reduction_db as f64).abs() < 1e-6);

    // Quiet programme (−70 dBFS), loud key (−6 dBFS): opens to unity.
    let mut g = gate(2, p);
    let mut x = sine(db_to_lin(-70.0) as f32, 1000.0, n, 2);
    let input = x.clone();
    let key = sine(0.5, 1000.0, n, 2);
    g.process_with_key(&mut x, &key, 2);
    assert!(g.is_open());
    assert_eq!(g.current_gain(), 1.0);
    // Once fully open the programme passes unchanged.
    let tail = ms(100.0) * 2;
    assert_eq!(x[tail..], input[tail..]);
}

#[test]
fn compressor_follows_key_not_programme() {
    let p = comp_params();
    let n = ms(300.0);

    // Loud programme (0 dBFS, 20 dB over threshold), silent key: untouched.
    let mut c = compressor(2, p);
    let input = sine(1.0, 1000.0, n, 2);
    let mut x = input.clone();
    c.process_with_key(&mut x, &vec![0.0; n * 2], 2);
    assert_eq!(c.gain_reduction_db(), 0.0);
    assert_bits_equal(&x, &input, "loud programme, silent key");

    // Quiet programme, key 12 dB over threshold (constant magnitude): the
    // reduction settles at the static curve of the key level, S·(x − T).
    let mut c = compressor(2, p);
    let mut x = sine(0.01, 1000.0, n, 2);
    let key: Vec<f32> = square_nyquist(db_to_lin(-8.0) as f32, n)
        .into_iter()
        .flat_map(|v| [v, 0.0])
        .collect();
    c.process_with_key(&mut x, &key, 2);
    let want = 0.75 * 12.0;
    assert!(
        (c.gain_reduction_db() - want).abs() < 0.01,
        "{}",
        c.gain_reduction_db()
    );
}

#[test]
fn key_with_other_channel_count_uses_max_over_key_channels() {
    let n = ms(200.0);
    for key_ch in [1usize, 3, 6] {
        // Loud only in the last key channel.
        let key: Vec<f32> = (0..n)
            .flat_map(|i| {
                (0..key_ch).map(move |c| {
                    if c == key_ch - 1 {
                        if i % 2 == 0 {
                            0.5
                        } else {
                            -0.5
                        }
                    } else {
                        0.0
                    }
                })
            })
            .collect();

        let mut g = gate(2, GateParams::default());
        g.set_key_channels(key_ch);
        assert_eq!(g.key_channels(), key_ch);
        let mut x = vec![1e-4f32; n * 2];
        g.process_with_key(&mut x, &key, key_ch);
        assert!(g.is_open(), "gate key_ch={key_ch}");

        let mut c = compressor(2, comp_params());
        c.set_key_channels(key_ch);
        let mut x = vec![1e-4f32; n * 2];
        c.process_with_key(&mut x, &key, key_ch);
        // 0.5 = −6.02 dBFS: 13.98 dB over, reduction 0.75 · 13.98.
        let want = 0.75 * (lin_to_db(0.5) + 20.0);
        assert!(
            (c.gain_reduction_db() - want).abs() < 0.01,
            "compressor key_ch={key_ch}: {}",
            c.gain_reduction_db()
        );

        // The key channel count survives a reconfigure.
        c.configure(44_100.0, 4);
        assert_eq!(c.key_channels(), key_ch);
    }
}

#[test]
fn mismatched_key_reads_as_silence() {
    let n = ms(100.0);
    let mut c = compressor(2, comp_params());
    let input = sine(1.0, 1000.0, n, 2);
    let mut x = input.clone();
    // Configured for a 2-channel key, called with 1 channel: no compression.
    c.process_with_key(&mut x, &sine(1.0, 1000.0, n, 1), 1);
    assert_eq!(c.gain_reduction_db(), 0.0);
    assert_eq!(x, input);
}

// ---------------------------------------------------------------------------
// Key filter.

/// Gate with a −10 dBFS key at `freq` Hz through `filter`: open?
fn gate_opens_on_key(filter: KeyFilter, freq: f64) -> bool {
    let mut g = gate(2, GateParams::default());
    g.set_key_channels(1);
    g.set_key_filter(filter);
    let n = ms(500.0);
    let mut x = vec![1e-4f32; n * 2];
    let key = sine(db_to_lin(-10.0) as f32, freq, n, 1);
    g.process_with_key(&mut x, &key, 1);
    g.is_open()
}

/// Settled compressor reduction with a 0 dBFS key at `freq` Hz through `filter`.
fn comp_reduction_on_key(filter: KeyFilter, freq: f64) -> f64 {
    let mut c = compressor(1, comp_params());
    c.set_key_filter(filter);
    let n = ms(500.0);
    let mut x = vec![1e-4f32; n];
    let key = sine(1.0, freq, n, 1);
    c.process_with_key(&mut x, &key, 1);
    c.gain_reduction_db()
}

#[test]
fn high_pass_key_filter_ignores_low_frequencies() {
    let hpf = KeyFilter::HighPass {
        frequency_hz: 2000.0,
        q: 0.707,
    };
    // Without the filter, a 60 Hz key opens the gate; with a 2 kHz HPF
    // (−12 dB/oct: about −60 dB at 60 Hz) it does not.
    assert!(gate_opens_on_key(KeyFilter::Off, 60.0));
    assert!(!gate_opens_on_key(hpf, 60.0));
    assert!(gate_opens_on_key(hpf, 8000.0));

    // Compressor: no reduction from a 60 Hz key through the HPF; a key well
    // above the corner compresses (12 dB above the −20 dB threshold region).
    assert!(comp_reduction_on_key(KeyFilter::Off, 60.0) > 10.0);
    assert_eq!(comp_reduction_on_key(hpf, 60.0), 0.0);
    assert!(comp_reduction_on_key(hpf, 10_000.0) > 10.0);
}

#[test]
fn band_pass_key_filter_selects_its_band() {
    let bpf = KeyFilter::BandPass {
        frequency_hz: 6000.0,
        q: 2.0,
    };
    assert!(gate_opens_on_key(bpf, 6000.0));
    assert!(!gate_opens_on_key(bpf, 100.0));
    // 0 dB peak gain: at the centre the reduction equals the unfiltered one.
    let centre = comp_reduction_on_key(bpf, 6000.0);
    let plain = comp_reduction_on_key(KeyFilter::Off, 6000.0);
    assert!((centre - plain).abs() < 0.2, "{centre} vs {plain}");
}

#[test]
fn key_detector_is_reusable_on_its_own() {
    let mut d = KeyDetector::new();
    d.configure(FS, 2);
    d.set_filter(KeyFilter::HighPass {
        frequency_hz: 1000.0,
        q: 0.707,
    });
    let n = ms(200.0);
    let mut out = vec![0.0f32; n];
    d.linked_peaks(&sine(0.5, 50.0, n, 2), 2, &mut out);
    assert!(out[n / 2..].iter().all(|&p| p < 0.01));
    d.reset();
    d.linked_peaks(&sine(0.5, 12_000.0, n, 2), 2, &mut out);
    let max = out[n / 2..].iter().fold(0.0f32, |m, &p| m.max(p));
    assert!((max - 0.5).abs() < 0.02, "{max}");
}

// ---------------------------------------------------------------------------
// No key: unchanged.

/// `process` with every new setting touched gives the same bits as a plain
/// gate/compressor; and keying with the programme itself (no filter) gives
/// the same bits as `process`.
#[test]
fn without_key_the_path_is_unchanged() {
    let mut rng = Rng(5);
    for ch in [1usize, 2, 3, 8] {
        let input: Vec<f32> = (0..ch * 20_000)
            .map(|i| rng.next_f32() * [0.001f32, 0.6, 0.05, 1.5][(i / (ch * 3000)) % 4])
            .collect();

        let gp = GateParams {
            open_threshold_db: -30.0,
            close_threshold_db: -40.0,
            hold_ms: 5.0,
            release_ms: 20.0,
            ..GateParams::default()
        };
        let mut plain = gate(ch, gp);
        let mut touched = gate(ch, gp);
        touched.set_key_filter(KeyFilter::HighPass {
            frequency_hz: 500.0,
            q: 0.707,
        });
        touched.set_expander_ratio(7.0);
        let mut self_keyed = gate(ch, gp);
        let (mut a, mut b, mut c) = (input.clone(), input.clone(), input.clone());
        for ((x, y), z) in a
            .chunks_mut(ch * 333)
            .zip(b.chunks_mut(ch * 333))
            .zip(c.chunks_mut(ch * 333))
        {
            plain.process(x);
            touched.process(y);
            let key = z.to_vec();
            self_keyed.process_with_key(z, &key, ch);
        }
        assert_bits_equal(&a, &b, &format!("gate settings ch={ch}"));
        assert_bits_equal(&a, &c, &format!("gate self-keyed ch={ch}"));

        let cp = comp_params();
        let mut plain = compressor(ch, cp);
        let mut touched = compressor(ch, cp);
        touched.set_key_filter(KeyFilter::BandPass {
            frequency_hz: 3000.0,
            q: 1.0,
        });
        let mut self_keyed = compressor(ch, cp);
        let (mut a, mut b, mut c) = (input.clone(), input.clone(), input.clone());
        for ((x, y), z) in a
            .chunks_mut(ch * 333)
            .zip(b.chunks_mut(ch * 333))
            .zip(c.chunks_mut(ch * 333))
        {
            plain.process(x);
            touched.process(y);
            let key = z.to_vec();
            self_keyed.process_with_key(z, &key, ch);
        }
        assert_bits_equal(&a, &b, &format!("compressor settings ch={ch}"));
        assert_bits_equal(&a, &c, &format!("compressor self-keyed ch={ch}"));
    }
}

// ---------------------------------------------------------------------------
// Expander mode.

#[test]
fn expander_defaults() {
    let g = Gate::new();
    assert_eq!(g.mode(), GateMode::Gate);
    assert_eq!(g.expander_ratio(), DEFAULT_EXPANDER_RATIO);
    let mut g = Gate::new();
    g.set_expander_ratio(f32::NAN);
    assert_eq!(g.expander_ratio(), 2.0);
    g.set_expander_ratio(100.0);
    assert_eq!(g.expander_ratio(), 20.0);
    g.set_expander_ratio(0.0);
    assert_eq!(g.expander_ratio(), 1.0);
}

/// Settled gain (dB) of an expander fed a constant-magnitude level `x_db`
/// on channel 0, read from a probe on channel 1.
fn expander_gain_db(p: GateParams, ratio: f32, x_db: f64) -> f64 {
    let mut g = gate(2, p);
    g.set_mode(GateMode::Expander);
    g.set_expander_ratio(ratio);
    let n = ms(1000.0);
    let mut buf: Vec<f32> = square_nyquist(db_to_lin(x_db) as f32, n)
        .into_iter()
        .flat_map(|v| [v, PROBE])
        .collect();
    g.process(&mut buf);
    lin_to_db(buf[buf.len() - 1] as f64 / PROBE as f64)
}

#[test]
fn expander_static_curve_matches_formula() {
    let p = GateParams {
        open_threshold_db: -20.0,
        close_threshold_db: -30.0,
        attack_ms: 2.0,
        hold_ms: 10.0,
        release_ms: 20.0,
        reduction_db: -80.0,
    };
    let t = p.open_threshold_db as f64;
    for ratio in [1.0f32, 1.5, 2.0, 4.0, 10.0, 20.0] {
        for x in [-10.0, -19.5, -21.0, -25.0, -30.0, -40.0, -60.0] {
            let expected = if x >= t {
                0.0
            } else {
                ((ratio as f64 - 1.0) * (x - t)).max(p.reduction_db as f64)
            };
            let got = expander_gain_db(p, ratio, x);
            assert!(
                (got - expected).abs() < 0.1,
                "ratio {ratio}, level {x} dB: got {got:.3} dB, expected {expected:.3} dB"
            );
        }
    }
    // A shallower floor caps the expansion.
    let shallow = GateParams {
        reduction_db: -12.0,
        ..p
    };
    assert!((expander_gain_db(shallow, 4.0, -40.0) + 12.0).abs() < 0.1);
}

#[test]
fn expander_hysteresis_and_hold() {
    let p = GateParams {
        open_threshold_db: -20.0,
        close_threshold_db: -30.0,
        attack_ms: 1.0,
        hold_ms: 50.0,
        release_ms: 5.0,
        reduction_db: -80.0,
    };
    let mut g = gate(2, p);
    g.set_mode(GateMode::Expander);
    g.set_expander_ratio(3.0);
    let feed = |g: &mut Gate, x_db: f64, n: usize| -> Vec<f64> {
        let mut buf: Vec<f32> = square_nyquist(db_to_lin(x_db) as f32, n)
            .into_iter()
            .flat_map(|v| [v, PROBE])
            .collect();
        g.process(&mut buf);
        buf.chunks_exact(2)
            .map(|f| lin_to_db(f[1] as f64 / PROBE as f64))
            .collect()
    };
    // Open above T.
    feed(&mut g, -10.0, ms(100.0));
    assert!(g.is_open());
    // Between close and open: hysteresis keeps unity gain.
    let gains = feed(&mut g, -25.0, ms(300.0));
    assert!(gains.iter().all(|&d| d.abs() < 1e-6));
    // Below close: unity for the hold time, then the curve (−20 dB·2 at −35).
    let gains = feed(&mut g, -35.0, ms(300.0));
    let hold_end = ms(50.0) + ms(20.0); // + the detector's 20 ms decay
    assert!(gains[..ms(50.0)].iter().all(|&d| d.abs() < 1e-6));
    assert!(!g.is_open());
    assert!((gains[hold_end + ms(100.0)] + 30.0).abs() < 0.1);
}

#[test]
fn mode_and_ratio_changes_do_not_jump() {
    let p = GateParams {
        attack_ms: 5.0,
        release_ms: 50.0,
        ..GateParams::default()
    };
    let mut g = gate(2, p);
    // Steady level below T, so gate and expander targets differ.
    let n = ms(50.0);
    let mut prev = None;
    for k in 0..20 {
        match k % 3 {
            0 => g.set_mode(GateMode::Expander),
            1 => g.set_expander_ratio(1.0 + (k % 7) as f32 * 3.0),
            _ => g.set_mode(GateMode::Gate),
        }
        let mut buf: Vec<f32> = square_nyquist(db_to_lin(-26.0) as f32, n)
            .into_iter()
            .flat_map(|v| [v, PROBE])
            .collect();
        g.process(&mut buf);
        for f in buf.chunks_exact(2) {
            let gain = f[1] as f64 / PROBE as f64;
            if let Some(q) = prev {
                let q: f64 = q;
                // One-pole with τ ≥ 5 ms at 48 kHz moves ≤ 1/240 of the
                // distance (≤ 1) per sample.
                assert!((gain - q).abs() < 1.0 / 200.0, "jump {q} → {gain}");
            }
            prev = Some(gain);
        }
    }
}

// ---------------------------------------------------------------------------
// SIMD levels.

/// Buffer sizes cycling through odd and block-crossing lengths.
const BUFFERS: [usize; 6] = [480, 1, 63, 65, 1024, 7];

fn signal(ch: usize, frames: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed | 1);
    let levels = [0.001f32, 0.5, 4.0, 0.0, 0.05, 1.0, 0.2, 0.02];
    (0..frames * ch)
        .map(|i| levels[(i / ch / 900) % levels.len()] * rng.next_f32())
        .collect()
}

/// Run `f(proc, buffer, key, k)` over `input`/`key` in varying buffer sizes.
fn run_keyed<P>(
    p: &mut P,
    input: &[f32],
    ch: usize,
    key: &[f32],
    kc: usize,
    mut f: impl FnMut(&mut P, &mut [f32], &[f32], usize),
) -> Vec<f32> {
    let mut out = input.to_vec();
    let (mut pos, mut k) = (0, 0);
    while pos < input.len() / ch {
        let frames = BUFFERS[k % BUFFERS.len()].min(input.len() / ch - pos);
        f(
            p,
            &mut out[pos * ch..(pos + frames) * ch],
            &key[pos * kc..(pos + frames) * kc],
            k,
        );
        pos += frames;
        k += 1;
    }
    out
}

/// Fallback vs detected level for the keyed paths and expander mode. With a
/// key filter the biquad's FMA policy applies: compared bit for bit when both
/// levels share it (always on aarch64), otherwise only checked to be finite.
#[test]
fn new_paths_are_bit_identical_across_simd_levels() {
    let (lo, hi) = (Level::fallback(), detect());
    let filtered_exact = uses_fma(lo) == uses_fma(hi);
    eprintln!("detected {hi:?}, key filter bit-exact: {filtered_exact}");
    for (ch, kc) in [(1usize, 1usize), (2, 1), (2, 2), (3, 8), (8, 3), (64, 2)] {
        let input = signal(ch, 9000, 11 + ch as u64);
        let key = signal(kc, 9000, 99 + kc as u64);
        for filtered in [false, true] {
            let filter = |k: usize| match (filtered, k % 3) {
                (false, _) => KeyFilter::Off,
                (true, 0) => KeyFilter::HighPass {
                    frequency_hz: 200.0 + k as f32 * 50.0,
                    q: 0.707,
                },
                (true, _) => KeyFilter::BandPass {
                    frequency_hz: 4000.0,
                    q: 1.5,
                },
            };
            let run_gate = |level: Level| {
                let mut g = Gate::new();
                g.set_simd_level(level);
                g.configure(FS, ch);
                g.set_key_channels(kc);
                g.set_params(&GateParams {
                    open_threshold_db: -30.0,
                    close_threshold_db: -40.0,
                    hold_ms: 3.0,
                    release_ms: 10.0,
                    ..GateParams::default()
                });
                run_keyed(&mut g, &input, ch, &key, kc, |g, x, key, k| {
                    if k % 4 == 1 {
                        g.set_key_filter(filter(k));
                        g.set_mode(if k % 8 == 1 {
                            GateMode::Gate
                        } else {
                            GateMode::Expander
                        });
                        g.set_expander_ratio(1.0 + (k % 5) as f32 * 4.0);
                    }
                    if k % 2 == 0 {
                        g.process_with_key(x, key, kc);
                    } else {
                        g.process(x);
                    }
                })
            };
            let run_comp = |level: Level| {
                let mut c = Compressor::new();
                c.set_simd_level(level);
                c.configure(FS, ch);
                c.set_key_channels(kc);
                c.set_params(&comp_params());
                run_keyed(&mut c, &input, ch, &key, kc, |c, x, key, k| {
                    if k % 4 == 1 {
                        c.set_key_filter(filter(k));
                    }
                    c.process_with_key(x, key, kc);
                })
            };
            let (ga, gb) = (run_gate(lo), run_gate(hi));
            let (ca, cb) = (run_comp(lo), run_comp(hi));
            let what = format!("ch={ch} kc={kc} filtered={filtered}");
            if !filtered || filtered_exact {
                assert_bits_equal(&ga, &gb, &format!("gate {what}"));
                assert_bits_equal(&ca, &cb, &format!("compressor {what}"));
            } else {
                assert!(ga
                    .iter()
                    .chain(&gb)
                    .chain(&ca)
                    .chain(&cb)
                    .all(|v| v.is_finite()));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Real-time safety and robustness.

#[test]
fn no_allocation_in_keyed_processing_and_setters() {
    let (ch, kc) = (2usize, 3usize);
    let mut g = Gate::new();
    g.configure(FS, ch);
    g.set_key_channels(kc);
    let mut c = Compressor::new();
    c.configure(FS, ch);
    c.set_key_channels(kc);
    let mut d = KeyDetector::new();
    d.configure(FS, kc);

    let mut buf = signal(ch, 1024, 3);
    let key = signal(kc, 1024, 4);
    let mut peaks = vec![0.0f32; 1024];
    let filters = [
        KeyFilter::Off,
        KeyFilter::HighPass {
            frequency_hz: 120.0,
            q: 0.707,
        },
        KeyFilter::BandPass {
            frequency_hz: 6000.0,
            q: 2.0,
        },
    ];

    let before = alloc_count();
    for i in 0..60 {
        let f = filters[i % 3];
        g.set_key_filter(f);
        g.set_mode(if i % 2 == 0 {
            GateMode::Expander
        } else {
            GateMode::Gate
        });
        g.set_expander_ratio(1.0 + i as f32 * 0.3);
        g.process_with_key(&mut buf, &key, kc);
        g.process(&mut buf);
        c.set_key_filter(f);
        c.process_with_key(&mut buf, &key, kc);
        // Mismatched and short keys take the silent path, also allocation-free.
        c.process_with_key(&mut buf, &key[..10], 1);
        d.set_filter(f);
        d.linked_peaks(&key, kc, &mut peaks);
        if i % 17 == 16 {
            g.reset();
            c.reset();
            d.reset();
        }
    }
    assert_eq!(alloc_count() - before, 0);
}

#[test]
fn extremes_stay_finite() {
    for fs in [8_000.0f32, 44_100.0, 192_000.0] {
        for (ch, kc) in [(1usize, 1usize), (2, 6), (8, 1)] {
            let mut g = Gate::new();
            g.configure(fs, ch);
            g.set_key_channels(kc);
            g.set_mode(GateMode::Expander);
            let mut c = Compressor::new();
            c.configure(fs, ch);
            c.set_key_channels(kc);
            for (i, f) in [
                KeyFilter::HighPass {
                    frequency_hz: 0.0,
                    q: 0.0,
                },
                KeyFilter::HighPass {
                    frequency_hz: 1e9,
                    q: 1e9,
                },
                KeyFilter::BandPass {
                    frequency_hz: f32::INFINITY,
                    q: f32::NAN,
                },
                KeyFilter::BandPass {
                    frequency_hz: 20.0,
                    q: 10.0,
                },
            ]
            .into_iter()
            .enumerate()
            {
                g.set_key_filter(f);
                g.set_expander_ratio([1.0, 20.0, f32::NAN, 5.0][i]);
                c.set_key_filter(f);
                let mut x = signal(ch, 2000, 1 + i as u64);
                let mut key = signal(kc, 2000, 2 + i as u64);
                key[0] = f32::NAN;
                key[kc] = f32::INFINITY;
                g.process_with_key(&mut x, &key, kc);
                c.process_with_key(&mut x, &key, kc);
                assert!(
                    x.iter().all(|v| v.is_finite()),
                    "fs={fs} ch={ch} kc={kc} {f:?}"
                );
            }
        }
    }
}
