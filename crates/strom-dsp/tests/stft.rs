//! STFT tests (phase 1b spec 2.5 and 4): perfect reconstruction with an
//! identity callback at every frame/hop combination (error ≤ −100 dB),
//! latency by impulse, spectral modifications behave as theory says, the
//! callback sees the right channel, and zero allocations in `process`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::f64::consts::PI;

use strom_dsp::stft::{Complex32, Stft, StftHop};

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAlloc;

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

const FRAMES: [usize; 5] = [256, 512, 1024, 2048, 4096];
const HOPS: [StftHop; 2] = [StftHop::Half, StftHop::Quarter];
/// Buffer sizes, cycled (hop-unaligned on purpose).
const BLOCKS: [usize; 7] = [1, 480, 7, 1024, 65, 333, 4096];

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-30).log10()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn noise(len: usize, seed: u64, a: f32) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..len).map(|_| a * r.next()).collect()
}

/// Run `stft` over `x` in place, in the varying buffer sizes of `BLOCKS`.
fn run(stft: &mut Stft, x: &mut [f32], ch: usize, proc: &mut dyn FnMut(usize, &mut [Complex32])) {
    let frames = x.len() / ch;
    let mut pos = 0;
    for b in BLOCKS.iter().cycle() {
        if pos >= frames {
            break;
        }
        let n = (*b).min(frames - pos);
        stft.process(&mut x[pos * ch..(pos + n) * ch], proc);
        pos += n;
    }
}

fn identity(_: usize, _: &mut [Complex32]) {}

/// Identity callback: output = input delayed by `latency_frames`, error
/// (RMS and peak, relative to the signal) ≤ −100 dB, for every frame size
/// and hop, stereo, with hop-unaligned buffer sizes.
#[test]
fn identity_reconstructs_perfectly() {
    let ch = 2;
    let mut worst = -400.0f64;
    for &n in &FRAMES {
        for &hop in &HOPS {
            let mut s = Stft::new();
            s.configure(n, hop, ch).unwrap();
            let lat = s.latency_frames();
            let frames = 6 * n + 3000;
            let x = noise(frames * ch, n as u64, 0.5);
            let mut y = x.clone();
            run(&mut s, &mut y, ch, &mut identity);
            let (mut e2, mut s2, mut emax) = (0.0f64, 0.0f64, 0.0f64);
            for i in lat..frames {
                for c in 0..ch {
                    let r = x[(i - lat) * ch + c] as f64;
                    let e = y[i * ch + c] as f64 - r;
                    e2 += e * e;
                    s2 += r * r;
                    emax = emax.max(e.abs());
                }
            }
            let rms = db((e2 / s2).sqrt());
            let peak = db(emax / 0.5);
            println!(
                "N={n} hop={}: error RMS {rms:.1} dB, peak {peak:.1} dB",
                s.hop_size()
            );
            worst = worst.max(rms).max(peak);
            assert!(
                rms <= -100.0 && peak <= -100.0,
                "N={n} {hop:?}: {rms:.1} / {peak:.1} dB"
            );
            // Before the latency the output is silence.
            assert!(y[..lat * ch].iter().all(|v| v.abs() < 1e-6));
        }
    }
    println!("worst reconstruction error {worst:.1} dB");
}

/// An impulse comes out exactly `latency_frames` later, at unit gain.
#[test]
fn latency_is_exact_by_impulse() {
    for &n in &FRAMES {
        for &hop in &HOPS {
            let mut s = Stft::new();
            s.configure(n, hop, 1).unwrap();
            assert_eq!(s.latency_frames(), n - 1);
            for n0 in [0usize, 1, 37, n / 2 + 5] {
                s.reset();
                let mut x = vec![0.0f32; 3 * n];
                x[n0] = 1.0;
                run(&mut s, &mut x, 1, &mut identity);
                let at = n0 + s.latency_frames();
                assert!((x[at] - 1.0).abs() < 1e-5, "N={n} n0={n0}: {}", x[at]);
                let rest = x
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != at)
                    .fold(0.0f32, |m, (_, v)| m.max(v.abs()));
                assert!(rest < 1e-5, "N={n} n0={n0}: leakage {rest}");
            }
        }
    }
}

/// DFT amplitude at an exact bin of channel `c`.
fn bin_amp(y: &[f32], ch: usize, c: usize, start: usize, n: usize, bin: usize) -> f64 {
    let (mut re, mut im) = (0.0, 0.0);
    for k in 0..n {
        let ph = 2.0 * PI * ((bin * k) % n) as f64 / n as f64;
        let v = y[(start + k) * ch + c] as f64;
        re += v * ph.cos();
        im -= v * ph.sin();
    }
    2.0 * (re * re + im * im).sqrt() / n as f64
}

/// Zeroing every bin above a cut-off removes a high tone and keeps a low one:
/// 1 kHz and 15 kHz at 48 kHz, bins above 5 kHz zeroed.
#[test]
fn zeroing_high_bins_removes_high_tone() {
    let fs = 48_000.0;
    let ch = 1;
    for &n in &[512usize, 1024, 2048] {
        for &hop in &HOPS {
            let mut s = Stft::new();
            s.configure(n, hop, ch).unwrap();
            let cut = (5_000.0 * n as f64 / fs) as usize;
            let mut lowpass = |_: usize, sp: &mut [Complex32]| {
                sp[cut..]
                    .iter_mut()
                    .for_each(|b| *b = Complex32::new(0.0, 0.0));
            };
            // Analysis window: 4800 samples, so both tones are exact bins.
            let an = 4800;
            let (lo, hi) = (100usize, 1500usize); // 1 kHz, 15 kHz
            let frames = 4 * n + an + 1000;
            let mut x: Vec<f32> = (0..frames)
                .map(|i| {
                    let t = i as f64 / an as f64;
                    (0.4 * (2.0 * PI * lo as f64 * t).sin()
                        + 0.4 * (2.0 * PI * hi as f64 * t).sin()) as f32
                })
                .collect();
            run(&mut s, &mut x, ch, &mut lowpass);
            let start = frames - an;
            let a_lo = db(bin_amp(&x, ch, 0, start, an, lo) / 0.4);
            let a_hi = db(bin_amp(&x, ch, 0, start, an, hi) / 0.4);
            assert!(a_lo.abs() < 0.01, "N={n}: low tone changed by {a_lo:.3} dB");
            assert!(a_hi < -90.0, "N={n}: high tone only {a_hi:.1} dB down");
        }
    }
}

/// A uniform spectral gain is a gain on the output; the callback receives
/// each channel's own spectrum, with the right index and length.
#[test]
fn per_channel_gain_and_callback_contract() {
    let ch = 3;
    let n = 1024;
    let mut s = Stft::new();
    s.configure(n, StftHop::Quarter, ch).unwrap();
    let gains = [0.5f32, 0.0, 2.0];
    let mut calls = [0usize; 3];
    let mut proc = |c: usize, sp: &mut [Complex32]| {
        assert_eq!(sp.len(), n / 2 + 1);
        calls[c] += 1;
        sp.iter_mut().for_each(|b| *b *= gains[c]);
    };
    let frames = 8 * n;
    let x = noise(frames * ch, 42, 0.3);
    let mut y = x.clone();
    run(&mut s, &mut y, ch, &mut proc);
    assert_eq!(calls, [frames / (n / 4); 3]);
    let lat = s.latency_frames();
    for i in lat..frames {
        for c in 0..ch {
            let want = gains[c] * x[(i - lat) * ch + c];
            assert!((y[i * ch + c] - want).abs() < 1e-5);
        }
    }
}

/// Non-finite values from the input or the callback flush out after one
/// frame plus latency; unconfigured `process` is a no-op.
#[test]
fn robustness() {
    let n = 256;
    let mut s = Stft::new();
    let mut buf = [1.0f32; 16];
    s.process(&mut buf, &mut identity);
    assert_eq!(buf, [1.0; 16]);

    s.configure(n, StftHop::Half, 2).unwrap();
    let mut x = vec![0.0f32; 20 * n * 2];
    x[10] = f32::NAN;
    x[11] = f32::INFINITY;
    run(&mut s, &mut x, 2, &mut identity);
    assert!(x[4 * n * 2..].iter().all(|v| *v == 0.0));

    let mut bad = |_: usize, sp: &mut [Complex32]| sp[3] = Complex32::new(f32::NAN, 0.0);
    let mut x = vec![0.0f32; 4 * n * 2];
    run(&mut s, &mut x, 2, &mut bad);
    s.reset();
    let mut x = noise(10 * n * 2, 3, 0.5);
    run(&mut s, &mut x, 2, &mut identity);
    assert!(x.iter().all(|v| v.is_finite()));
    // Empty and odd-length buffers.
    s.process(&mut [], &mut identity);
    s.process(&mut [0.5], &mut identity);
}

#[test]
fn no_allocation_in_process() {
    for &n in &FRAMES {
        for &hop in &HOPS {
            let ch = 2;
            let mut s = Stft::new();
            s.configure(n, hop, ch).unwrap();
            let mut x = noise(3 * n * ch, 9, 0.5);
            let mut gain = |_: usize, sp: &mut [Complex32]| {
                sp.iter_mut().for_each(|b| *b *= 0.9);
            };
            let before = allocs();
            run(&mut s, &mut x, ch, &mut gain);
            s.process(&mut x[..2 * n * ch], &mut identity);
            s.reset();
            let _ = s.latency_frames();
            assert_eq!(allocs() - before, 0, "N={n} {hop:?} allocated");
        }
    }
}
