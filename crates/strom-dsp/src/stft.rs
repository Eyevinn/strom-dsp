//! Short-time Fourier transform framework: streaming analysis, a per-frame
//! spectrum callback, and weighted overlap-add resynthesis.
//!
//! # Method
//!
//! Weighted overlap-add (WOLA) after J. O. Smith III, *Spectral Audio Signal
//! Processing* (STFT, overlap-add, the constant-overlap-add condition), and
//! Crochiere & Rabiner, *Multirate Digital Signal Processing* (ch. 7, filter
//! bank view of the STFT):
//!
//! 1. Every `hop` input samples, the newest `N` samples (`N` = frame size) are
//!    multiplied by the analysis window `wa` and transformed with a real FFT
//!    (`N/2 + 1` bins).
//! 2. The callback ([`SpectrumProcessor`]) may modify the bins in place, per
//!    channel.
//! 3. The inverse real FFT is multiplied by the synthesis window `ws` and
//!    overlap-added into the output.
//!
//! Both windows are the square root of the **periodic** Hann window
//! `hann[n] = ½ − ½·cos(2πn/N)`, so `wa·ws = hann`. Periodic Hann satisfies the
//! constant-overlap-add condition for hops `N/2` and `N/4`:
//! `Σ_m hann[n − mH] = N/(2H)` for every `n`. The synthesis window is scaled
//! by `2H/N` for that sum and by `1/N` for the unnormalised inverse FFT, so an
//! unmodified spectrum reconstructs the input exactly, up to rounding
//! (measured: RMS error below −135 dB, peak below −125 dB, see
//! `tests/stft.rs`).
//!
//! # Latency
//!
//! An output sample is final once the last frame that overlaps it has been
//! processed, and that frame ends up to `N − 1` samples later. Because
//! `process` accepts any buffer size (not just whole hops), the delay is a
//! fixed **`N − 1`** samples ([`Stft::latency_frames`]). The often-quoted
//! `N − hop` counts only whole-hop blocks and ignores the `hop − 1` samples of
//! gathering before the first block can be transformed.
//!
//! # Real-time safety
//!
//! The FFT plans (from `realfft`, which wraps `rustfft`), all buffers and the
//! FFT scratch space are created in [`Stft::configure`]. `process` calls only
//! `process_with_scratch`, which does not allocate (checked by the counting
//! allocator and RTSan tests).

use std::fmt;
use std::sync::Arc;

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// One complex FFT bin (`re`, `im`), re-exported from `realfft`/`num-complex`.
pub use realfft::num_complex::Complex32;

/// Smallest supported frame size.
pub const MIN_FRAME: usize = 256;
/// Largest supported frame size.
pub const MAX_FRAME: usize = 4096;

/// Modifies one channel's spectrum of one frame, in place.
///
/// Called from [`Stft::process`] (the real-time path), so implementations
/// must not allocate or block. The slice has `frame_size / 2 + 1` bins: bin
/// `k` is at `k · sample_rate / frame_size` Hz; bin 0 is DC and the last bin
/// is Nyquist (their imaginary parts are ignored by the inverse transform).
/// Bins are unnormalised FFT values of the windowed frame.
pub trait SpectrumProcessor {
    /// Modify `spectrum` (the current frame of `channel`) in place.
    fn process_spectrum(&mut self, channel: usize, spectrum: &mut [Complex32]);
}

impl<F: FnMut(usize, &mut [Complex32]) + ?Sized> SpectrumProcessor for F {
    fn process_spectrum(&mut self, channel: usize, spectrum: &mut [Complex32]) {
        self(channel, spectrum)
    }
}

/// Hop size relative to the frame size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StftHop {
    /// `frame / 2` (50 % overlap).
    Half,
    /// `frame / 4` (75 % overlap).
    Quarter,
}

/// Invalid STFT configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StftError {
    /// The frame size is not a power of two in [`MIN_FRAME`]`..=`[`MAX_FRAME`].
    FrameSize(usize),
}

impl fmt::Display for StftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FrameSize(n) => write!(
                f,
                "STFT frame size {n} is not a power of two in {MIN_FRAME}..={MAX_FRAME}"
            ),
        }
    }
}

impl std::error::Error for StftError {}

/// Streaming multichannel STFT with a per-frame spectrum callback and
/// constant-overlap-add resynthesis (see the module documentation).
///
/// Works in place on interleaved `f32` frames; the output is the input
/// delayed by [`latency_frames`](Self::latency_frames) when the callback does
/// nothing.
pub struct Stft {
    frame: usize,
    hop: usize,
    channels: usize,
    /// Position within the current hop (0 … hop − 1).
    slot: usize,
    fwd: Option<Arc<dyn RealToComplex<f32>>>,
    inv: Option<Arc<dyn ComplexToReal<f32>>>,
    /// Analysis and (scaled) synthesis windows, `frame` each.
    wa: Vec<f32>,
    ws: Vec<f32>,
    /// Per channel: last `frame` inputs (`channels × frame`).
    input: Vec<f32>,
    /// Per channel: overlap-add accumulator (`channels × frame`).
    acc: Vec<f32>,
    /// Per channel: finished output samples of the current hop (`channels × hop`).
    ready: Vec<f32>,
    /// Work buffers shared by all channels.
    work: Vec<f32>,
    spectrum: Vec<Complex32>,
    scratch: Vec<Complex32>,
}

impl fmt::Debug for Stft {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stft")
            .field("frame", &self.frame)
            .field("hop", &self.hop)
            .field("channels", &self.channels)
            .finish_non_exhaustive()
    }
}

impl Default for Stft {
    fn default() -> Self {
        Self::new()
    }
}

impl Stft {
    /// Unconfigured STFT: [`process`](Self::process) is a no-op until
    /// [`configure`](Self::configure).
    pub fn new() -> Self {
        Self {
            frame: 0,
            hop: 0,
            channels: 0,
            slot: 0,
            fwd: None,
            inv: None,
            wa: Vec::new(),
            ws: Vec::new(),
            input: Vec::new(),
            acc: Vec::new(),
            ready: Vec::new(),
            work: Vec::new(),
            spectrum: Vec::new(),
            scratch: Vec::new(),
        }
    }

    /// Plan the FFTs and allocate every buffer for `frame_size` (power of two,
    /// 256 … 4096), `hop` and `channels` interleaved channels (≥ 1). The only
    /// method that allocates; clears all state. On error the previous
    /// configuration is kept.
    pub fn configure(
        &mut self,
        frame_size: usize,
        hop: StftHop,
        channels: usize,
    ) -> Result<(), StftError> {
        if !frame_size.is_power_of_two() || !(MIN_FRAME..=MAX_FRAME).contains(&frame_size) {
            return Err(StftError::FrameSize(frame_size));
        }
        let n = frame_size;
        let h = match hop {
            StftHop::Half => n / 2,
            StftHop::Quarter => n / 4,
        };
        let channels = channels.max(1);
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(n);
        let scratch_len = fwd.get_scratch_len().max(inv.get_scratch_len());

        // sqrt of the periodic Hann window (computed in f64).
        let root_hann: Vec<f64> = (0..n)
            .map(|i| (std::f64::consts::PI * i as f64 / n as f64).sin())
            .collect();
        let scale = (2.0 * h as f64 / n as f64) / n as f64;

        self.frame = n;
        self.hop = h;
        self.channels = channels;
        self.slot = 0;
        self.wa = root_hann.iter().map(|&w| w as f32).collect();
        self.ws = root_hann.iter().map(|&w| (w * scale) as f32).collect();
        self.input = vec![0.0; channels * n];
        self.acc = vec![0.0; channels * n];
        self.ready = vec![0.0; channels * h];
        self.work = fwd.make_input_vec();
        self.spectrum = fwd.make_output_vec();
        self.scratch = vec![Complex32::new(0.0, 0.0); scratch_len];
        self.fwd = Some(fwd);
        self.inv = Some(inv);
        Ok(())
    }

    /// Frame size `N` in samples (0 before `configure`).
    pub fn frame_size(&self) -> usize {
        self.frame
    }

    /// Hop size in samples.
    pub fn hop_size(&self) -> usize {
        self.hop
    }

    /// Number of spectrum bins per frame, `N/2 + 1`.
    pub fn bins(&self) -> usize {
        if self.frame == 0 {
            0
        } else {
            self.frame / 2 + 1
        }
    }

    /// Configured channel count (0 before `configure`).
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Delay from input to output in frames: `N − 1` (see the module
    /// documentation).
    pub fn latency_frames(&self) -> usize {
        self.frame.saturating_sub(1)
    }

    /// Clear all signal state (input history, overlap-add accumulator).
    pub fn reset(&mut self) {
        self.slot = 0;
        self.input.fill(0.0);
        self.acc.fill(0.0);
        self.ready.fill(0.0);
    }

    /// Process interleaved `samples` in place (any length; a trailing partial
    /// frame is ignored). `proc` is called once per channel for every
    /// completed hop, channel 0 first. Real-time safe; a no-op if unconfigured.
    pub fn process<P: SpectrumProcessor + ?Sized>(&mut self, samples: &mut [f32], proc: &mut P) {
        if self.frame == 0 {
            return;
        }
        let ch = self.channels;
        let (n, h) = (self.frame, self.hop);
        let frames = samples.len() / ch;
        let mut pos = 0;
        while pos < frames {
            let k = self.slot;
            let m = (frames - pos).min(h - k);
            let block = &mut samples[pos * ch..(pos + m) * ch];
            for c in 0..ch {
                let inp = &mut self.input[c * n + n - h + k..c * n + n - h + k + m];
                for (d, s) in inp.iter_mut().zip(block.iter().skip(c).step_by(ch)) {
                    *d = *s;
                }
                // Slot `s` outputs `ready[s + 1]`, finished one hop ago; the
                // last slot outputs `ready[0]` of the frame completed below.
                let emit = m.min(h - 1 - k);
                let rdy = &self.ready[c * h + k + 1..c * h + k + 1 + emit];
                for (d, s) in block.iter_mut().skip(c).step_by(ch).zip(rdy) {
                    *d = *s;
                }
            }
            if k + m == h {
                for c in 0..ch {
                    self.frame_for(c, proc);
                    block[(m - 1) * ch + c] = self.ready[c * h];
                }
                self.slot = 0;
            } else {
                self.slot = k + m;
            }
            pos += m;
        }
    }

    /// Transform, modify and overlap-add the current frame of channel `c`,
    /// then advance its buffers by one hop.
    fn frame_for<P: SpectrumProcessor + ?Sized>(&mut self, c: usize, proc: &mut P) {
        let (n, h) = (self.frame, self.hop);
        let (Some(fwd), Some(inv)) = (&self.fwd, &self.inv) else {
            return;
        };
        let input = &mut self.input[c * n..(c + 1) * n];
        for ((d, x), w) in self.work.iter_mut().zip(input.iter()).zip(&self.wa) {
            *d = x * w;
        }
        // Lengths are fixed at configure, so these cannot fail.
        let _ = fwd.process_with_scratch(&mut self.work, &mut self.spectrum, &mut self.scratch);
        proc.process_spectrum(c, &mut self.spectrum);
        // DC and Nyquist must be real for the inverse real FFT.
        let last = self.spectrum.len() - 1;
        self.spectrum[0].im = 0.0;
        self.spectrum[last].im = 0.0;
        let _ = inv.process_with_scratch(&mut self.spectrum, &mut self.work, &mut self.scratch);

        let acc = &mut self.acc[c * n..(c + 1) * n];
        for ((a, y), w) in acc.iter_mut().zip(&self.work).zip(&self.ws) {
            *a += y * w;
        }
        self.ready[c * h..(c + 1) * h].copy_from_slice(&acc[..h]);
        acc.copy_within(h.., 0);
        acc[n - h..].fill(0.0);
        input.copy_within(h.., 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The window product satisfies COLA with the documented scale: the
    /// shifted sum of `wa·ws` is exactly `1/N` (the inverse-FFT scale).
    #[test]
    fn windows_are_cola() {
        for n in [256usize, 1024, 4096] {
            for hop in [StftHop::Half, StftHop::Quarter] {
                let mut s = Stft::new();
                s.configure(n, hop, 1).unwrap();
                let h = s.hop_size();
                for i in 0..h {
                    let sum: f64 = (0..n / h)
                        .map(|m| (s.wa[i + m * h] as f64) * (s.ws[i + m * h] as f64))
                        .sum();
                    assert!(
                        (sum * n as f64 - 1.0).abs() < 1e-6,
                        "n={n} h={h} i={i}: {sum}"
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_bad_frame_sizes() {
        let mut s = Stft::new();
        for n in [0usize, 128, 300, 8192] {
            assert_eq!(
                s.configure(n, StftHop::Half, 2),
                Err(StftError::FrameSize(n))
            );
        }
        assert_eq!(s.frame_size(), 0);
        assert!(s.configure(512, StftHop::Quarter, 2).is_ok());
        assert_eq!((s.frame_size(), s.hop_size(), s.bins()), (512, 128, 257));
    }
}
