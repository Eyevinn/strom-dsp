//! Intentional delay (`strom_dsp::delay`): impulses are delayed by exactly the
//! expected number of frames per channel, changes while running stay within
//! the documented discontinuity bound, delay 0 / disabled are bit-exact,
//! extremes and odd inputs are handled, and nothing allocates after setup.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use strom_dsp::delay::{
    crossfade_frames, max_crossfade_step, Delay, DelayParams, MAX_DELAY_MS, MAX_OFFSET_CHANNELS,
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

fn alloc_count() -> usize {
    ALLOCS.with(|c| c.get())
}

// ---------------------------------------------------------------------------
// Helpers.

const RATES: [f32; 5] = [8_000.0, 44_100.0, 48_000.0, 96_000.0, 192_000.0];
/// Irregular buffer sizes, so block boundaries fall everywhere.
const SIZES: [usize; 7] = [1, 480, 7, 1024, 63, 256, 257];

struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// Expected delay in frames, computed independently from the definition:
/// clamp(delay + offset, 0, 2000) ms, rounded to the nearest frame.
fn expected_frames(delay_ms: f32, offset_ms: f32, rate: f32) -> usize {
    let ms = f64::from((delay_ms + offset_ms).clamp(0.0, MAX_DELAY_MS));
    (ms * f64::from(rate) / 1000.0).round() as usize
}

fn params(delay_ms: f32, offsets: &[f32]) -> DelayParams {
    let mut p = DelayParams {
        delay_ms,
        ..DelayParams::default()
    };
    p.set_offsets(offsets);
    p
}

/// Run `input` through `d` in irregular buffer sizes.
fn run(d: &mut Delay, input: &[f32], channels: usize) -> Vec<f32> {
    let mut out = input.to_vec();
    let mut pos = 0;
    let mut k = 0;
    while pos < out.len() {
        let n = (SIZES[k % SIZES.len()] * channels).min(out.len() - pos);
        d.process(&mut out[pos..pos + n]);
        pos += n;
        k += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// Exact delays.

#[test]
fn impulse_is_delayed_by_exact_frames_per_channel() {
    let offsets = [0.0f32, 3.3, -2.0, 7.77];
    let channels = 5; // the last channel has no offset entry: 0
    for rate in RATES {
        for delay_ms in [0.0f32, 1.0, 12.5, 100.0] {
            let mut d = Delay::new();
            d.set_params(&params(delay_ms, &offsets));
            d.configure(rate, channels);
            assert_eq!(d.latency_frames(), 0, "intentional delay is not latency");
            let frames = (0.12 * rate) as usize;
            let mut input = vec![0.0f32; frames * channels];
            input[..channels].fill(1.0);
            let out = run(&mut d, &input, channels);
            for c in 0..channels {
                let off = offsets.get(c).copied().unwrap_or(0.0);
                let want = expected_frames(delay_ms, off, rate);
                assert_eq!(d.target_delay_frames(c), want);
                for f in 0..frames {
                    let v = out[f * channels + c];
                    let e = if f == want { 1.0 } else { 0.0 };
                    assert_eq!(
                        v, e,
                        "rate {rate} delay {delay_ms} ch {c} frame {f} (want {want})"
                    );
                }
            }
        }
    }
}

#[test]
fn delayed_signal_is_bit_exact_copy() {
    let mut rng = Rng(42);
    let channels = 3;
    let rate = 48_000.0;
    let input: Vec<f32> = (0..rate as usize / 4 * channels)
        .map(|_| rng.next_f32())
        .collect();
    let mut d = Delay::new();
    d.set_params(&params(5.0, &[1.0, 0.0, 2.5]));
    d.configure(rate, channels);
    let out = run(&mut d, &input, channels);
    for c in 0..channels {
        let k = d.target_delay_frames(c);
        for f in 0..input.len() / channels {
            let want = if f >= k {
                input[(f - k) * channels + c]
            } else {
                0.0
            };
            assert_eq!(out[f * channels + c].to_bits(), want.to_bits());
        }
    }
}

#[test]
fn zero_delay_and_disabled_are_bit_exact() {
    let mut rng = Rng(7);
    let channels = 2;
    let mut input: Vec<f32> = (0..20_000).map(|_| rng.next_f32() * 3.0).collect();
    input[100] = f32::NAN;
    input[101] = f32::INFINITY;
    input[102] = -0.0;
    input[103] = f32::MIN_POSITIVE / 8.0; // subnormal
    let mut d = Delay::new();
    d.configure(48_000.0, channels);
    let out = run(&mut d, &input, channels);
    for (a, b) in out.iter().zip(&input) {
        assert_eq!(a.to_bits(), b.to_bits(), "delay 0 must be bit-exact");
    }

    // Disabled with a delay set: bit-exact from the start (no frame processed yet).
    let mut p = params(50.0, &[10.0, 20.0]);
    p.enabled = false;
    d.set_params(&p);
    d.reset();
    let out = run(&mut d, &input, channels);
    for (a, b) in out.iter().zip(&input) {
        assert_eq!(a.to_bits(), b.to_bits(), "disabled must be bit-exact");
    }

    // Disabling while running: bit-exact once the crossfade is over.
    let mut d = Delay::new();
    d.set_params(&params(50.0, &[]));
    d.configure(48_000.0, channels);
    let mut buf = input.clone();
    d.process(&mut buf[..4800 * channels]);
    d.set_params(&p);
    d.process(&mut buf[4800 * channels..]);
    let start = (4800 + crossfade_frames(48_000.0)) * channels;
    for i in start..input.len() {
        assert_eq!(buf[i].to_bits(), input[i].to_bits(), "sample {i}");
    }
    // Re-enabling fades back to the delayed signal; its history was kept.
    let mut p2 = p;
    p2.enabled = true;
    d.set_params(&p2);
    let mut tail: Vec<f32> = (0..9600 * channels).map(|_| rng.next_f32()).collect();
    let src = tail.clone();
    d.process(&mut tail);
    for c in 0..channels {
        let k = d.target_delay_frames(c);
        for f in crossfade_frames(48_000.0) + k..9600 {
            assert_eq!(tail[f * channels + c], src[(f - k) * channels + c]);
        }
    }
}

#[test]
fn offsets_shorter_or_longer_than_channel_count() {
    let rate = 8_000.0;
    // Shorter: missing channels get offset 0. Longer: extra entries ignored.
    for offsets in [vec![1.0f32], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] {
        let mut d = Delay::new();
        d.set_params(&params(10.0, &offsets));
        d.configure(rate, 3);
        for c in 0..3 {
            let off = offsets.get(c).copied().unwrap_or(0.0);
            assert_eq!(d.target_delay_frames(c), expected_frames(10.0, off, rate));
        }
    }
    // More than MAX_OFFSET_CHANNELS entries are truncated.
    let long: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let p = params(0.0, &long);
    assert_eq!(p.offsets().len(), MAX_OFFSET_CHANNELS);
    assert_eq!(p.channel_delay_ms(80), 0.0);
}

#[test]
fn extremes_zero_and_max_delay() {
    for rate in [8_000.0f32, 48_000.0] {
        let channels = 2;
        let mut d = Delay::new();
        // 2000 ms on the left; 2000 + 500 clamps to 2000; -2500 clamps to 0.
        d.set_params(&params(2000.0, &[500.0, -2500.0]));
        d.configure(rate, channels);
        let max = expected_frames(MAX_DELAY_MS, 0.0, rate);
        assert_eq!(d.max_delay_frames(), max);
        assert_eq!(d.target_delay_frames(0), max);
        assert_eq!(d.target_delay_frames(1), 0);
        let frames = max + 2000;
        let mut input = vec![0.0f32; frames * channels];
        input[0] = 1.0;
        input[1] = 1.0;
        let out = run(&mut d, &input, channels);
        let hits: Vec<(usize, usize)> = out
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0.0)
            .map(|(i, _)| (i / channels, i % channels))
            .collect();
        assert_eq!(hits, vec![(0, 1), (max, 0)], "rate {rate}");
    }
}

// ---------------------------------------------------------------------------
// Changes while running.

/// Every output step stays within `D + Δg·2A` (module docs), the output never
/// exceeds the input peak, and after the last crossfade the output is the
/// input delayed by the final delay, bit-exact.
#[test]
fn changes_while_running_respect_discontinuity_bound() {
    for rate in [44_100.0f32, 48_000.0, 96_000.0] {
        let channels = 3;
        let amp = 0.9f32;
        let frames = rate as usize * 2;
        let freqs = [997.0f32, 2_500.0, 6_100.0];
        let input: Vec<f32> = (0..frames * channels)
            .map(|i| {
                let (f, c) = (i / channels, i % channels);
                amp * (2.0 * std::f32::consts::PI * freqs[c] * f as f32 / rate).sin()
            })
            .collect();
        let mut d_in = 0.0f32;
        let mut a_in = 0.0f32;
        for f in 1..frames {
            for c in 0..channels {
                d_in = d_in.max((input[f * channels + c] - input[(f - 1) * channels + c]).abs());
                a_in = a_in.max(input[f * channels + c].abs());
            }
        }
        let bound = d_in + (max_crossfade_step(rate) * 2.0 * f64::from(a_in)) as f32 + 1e-6;
        // The extra term is small compared to a hard switch (2A).
        assert!(bound - d_in < 0.01 * 2.0 * a_in);

        let mut d = Delay::new();
        d.set_params(&params(10.0, &[]));
        d.configure(rate, channels);
        let mut out = input.clone();
        // Changes at buffer boundaries, some inside running crossfades.
        let changes: [(usize, f32, [f32; 3]); 6] = [
            (4_000, 37.3, [0.0, 0.0, 0.0]),
            (4_500, 3.0, [1.0, 0.0, -1.0]),
            (20_000, 0.0, [0.0, 0.0, 0.0]),
            (30_000, 1999.0, [0.0, 5.0, 0.0]),
            (30_100, 250.0, [0.0, 0.0, 9.0]),
            (50_000, 20.0, [0.0, 0.5, 0.25]),
        ];
        let mut pos = 0;
        for (at, delay, offs) in changes {
            d.process(&mut out[pos * channels..at * channels]);
            d.set_params(&params(delay, &offs));
            pos = at;
        }
        d.process(&mut out[pos * channels..]);
        assert!(!d.is_changing());

        let mut prev = [0.0f32; 3];
        for f in 0..frames {
            for c in 0..channels {
                let y = out[f * channels + c];
                assert!(y.abs() <= a_in, "rate {rate} frame {f}: |y| {y} > A");
                let step = (y - prev[c]).abs();
                // The first frames step from silence into the delayed signal
                // exactly as the input steps from its (zero) start.
                assert!(
                    step <= bound,
                    "rate {rate} ch {c} frame {f}: step {step} > bound {bound}"
                );
                prev[c] = y;
            }
        }
        // Settled: bit-exact delayed copy.
        let last = 50_000 + 4 * crossfade_frames(rate);
        for f in last..frames {
            for c in 0..channels {
                let k = d.current_delay_frames(c);
                assert_eq!(out[f * channels + c], input[(f - k) * channels + c]);
            }
        }
    }
}

#[test]
fn reset_clears_history_and_finishes_crossfade() {
    let mut d = Delay::new();
    d.set_params(&params(10.0, &[]));
    d.configure(48_000.0, 1);
    let mut buf = vec![1.0f32; 1000];
    d.process(&mut buf);
    d.set_params(&params(20.0, &[]));
    d.process(&mut buf[..10]);
    assert!(d.is_changing());
    d.reset();
    assert!(!d.is_changing());
    assert_eq!(d.current_delay_frames(0), 960);
    let mut z = vec![0.5f32; 2000];
    d.process(&mut z);
    assert!(z[..960].iter().all(|&v| v == 0.0));
    assert!(z[960..].iter().all(|&v| v == 0.5));
}

// ---------------------------------------------------------------------------
// Robustness.

#[test]
fn robust_for_channel_counts_extremes_and_odd_input() {
    let mut rng = Rng(99);
    for channels in [1usize, 2, 8, 64] {
        let rate = if channels == 64 { 8_000.0 } else { 48_000.0 };
        let mut d = Delay::new();
        d.configure(rate, channels);
        let frames = 3 * rate as usize;
        let mut input: Vec<f32> = (0..frames * channels)
            .map(|_| rng.next_f32() * f32::MAX)
            .collect();
        input[10 * channels] = f32::NAN;
        input[11 * channels] = f32::INFINITY;
        input[12 * channels] = f32::NEG_INFINITY;
        let mut out = input.clone();
        let mut pos = 0;
        let mut k = 0;
        let params_list = [
            params(0.0, &[]),
            params(MAX_DELAY_MS, &[]),
            params(f32::NAN, &[f32::INFINITY, -1e30, 1e30]),
            params(1e9, &[-1e9]),
            params(MAX_DELAY_MS, &vec![-MAX_DELAY_MS; channels]),
            params(1000.0, &vec![MAX_DELAY_MS; channels]),
        ];
        while pos < out.len() {
            let n = (SIZES[k % SIZES.len()] * channels).min(out.len() - pos);
            if k % 5 == 0 {
                d.set_params(&params_list[(k / 5) % params_list.len()]);
            }
            d.process(&mut out[pos..pos + n]);
            pos += n;
            k += 1;
        }
        // The NaN/Inf samples come out (once per tap, delayed) but never
        // spread: everything else stays finite and within the input peak.
        // (A tap moving back in time may replay them, so count only where.)
        for (i, v) in out.iter().enumerate() {
            assert!(
                v.is_finite() || i % channels == 0,
                "channels {channels}: non-finite output {v} at {i} spread to another channel"
            );
        }
        // Partial frames are ignored.
        if channels > 1 {
            let mut odd = vec![1.0f32; channels + 1];
            d.process(&mut odd);
            assert_eq!(odd[channels], 1.0);
        }
    }
    // Unconfigured: no-op.
    let mut d = Delay::new();
    let mut x = [1.0f32, 2.0];
    d.process(&mut x);
    assert_eq!(x, [1.0, 2.0]);
}

#[test]
fn no_allocation_after_configure() {
    let mut d = Delay::new();
    d.configure(48_000.0, 8);
    let mut rng = Rng(5);
    let mut buf: Vec<f32> = (0..1024 * 8).map(|_| rng.next_f32()).collect();
    let offsets: Vec<f32> = (0..8).map(|i| i as f32 * 1.5).collect();
    let mut p = params(0.0, &offsets);
    let before = alloc_count();
    for i in 0..400 {
        p.delay_ms = (i % 7) as f32 * 30.0;
        p.enabled = i % 11 != 0;
        d.set_params(&p);
        let n = SIZES[i % SIZES.len()] * 8;
        d.process(&mut buf[..n]);
        if i % 50 == 49 {
            d.reset();
        }
    }
    assert_eq!(alloc_count() - before, 0, "process/set_params allocated");
}
