//! Polyphase oversampling (2×, 4×, 8×) with a fixed integer latency.
//!
//! [`Oversampler`] converts interleaved `f32` audio up to 2, 4 or 8 times the
//! base rate and back down, as a cascade of 2× stages. Every stage is a
//! linear-phase **half-band FIR** run in polyphase form: every other tap of a
//! half-band filter is zero except the centre tap, so one polyphase branch is a
//! pure delay and the other a short symmetric FIR (Crochiere & Rabiner,
//! *Multirate Digital Signal Processing*, ch. 4–5; Lyons, *Understanding DSP*,
//! half-band filters).
//!
//! # Design (all computed in [`Oversampler::configure`])
//!
//! Windowed-sinc half-bands with a Kaiser window (Oppenheim & Schafer,
//! *Discrete-Time Signal Processing*, §7.5–7.6). The passband edge is
//! [`PASSBAND_EDGE`] · base rate (20 kHz at 48 kHz, 18.4 kHz at 44.1 kHz,
//! 40 kHz at 96 kHz), so the filters are the same at every sample rate:
//!
//! | stage | rate | side taps (unique) | Kaiser β | stopband |
//! |---|---|---|---|---|
//! | 1 | 1× → 2× | 40 (20) | 10.45 | ≥ 102 dB |
//! | 2 | 2× → 4× | 16 (8) | 10.40 | ≥ 98 dB |
//! | 3 | 4× → 8× | 10 (5) | 11.15 | ≥ 106 dB |
//!
//! Stage 1 has the narrow transition band (from the passband edge to its
//! mirror image around half the base rate). Later stages only have to reject
//! the images of what the stages before them left, so they are much shorter.
//! The passband ripple of every stage is below 0.001 dB.
//!
//! # Latency
//!
//! A half-band stage delays by an odd number `M` of high-rate samples. In the
//! polyphase form the delay can be made `M + 1` (upsampling) or `M − 1`
//! (downsampling) at no cost, by swapping which branch is the pure delay. The
//! stages pick the parity that makes the total delay of the up path and of the
//! down path each an integer number of base-rate frames. The round trip delay,
//! [`Oversampler::latency_frames`], is their sum:
//!
//! | factor | up | down | round trip |
//! |---|---|---|---|
//! | 2× | 20 | 19 | 39 |
//! | 4× | 24 | 23 | 47 |
//! | 8× | 25 | 24 | 49 |
//!
//! # SIMD
//!
//! The FIRs are vectorised along time (one output sample per lane, the
//! native `f32` width of the detected level), with the symmetric taps folded
//! (`c·(x[n−i] + x[n−M+i])`), on per-channel contiguous history buffers.
//! Multiply-adds follow the crate's FMA policy ([`crate::biquad::uses_fma`]),
//! so the fallback level is bit-identical to the vector path whenever both
//! have the same policy.

use std::f64::consts::PI;

use fearless_simd::{dispatch, Level, Simd, SimdBase, SimdFloat};

use crate::lanes::{fma_mode, Fma};

/// Passband edge as a fraction of the base sample rate (20 kHz at 48 kHz).
/// Ripple up to here is far below 0.1 dB; images and aliases of signals up to
/// here are attenuated by at least 90 dB.
pub const PASSBAND_EDGE: f64 = 20.0 / 48.0;

/// Oversampling factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OversampleFactor {
    /// 2× (one half-band stage).
    X2,
    /// 4× (two stages).
    X4,
    /// 8× (three stages).
    X8,
}

impl OversampleFactor {
    /// The factor as a number: 2, 4 or 8.
    pub fn ratio(self) -> usize {
        1 << self.stages()
    }

    /// Number of 2× stages.
    pub fn stages(self) -> usize {
        match self {
            Self::X2 => 1,
            Self::X4 => 2,
            Self::X8 => 3,
        }
    }
}

/// Unique side taps (`P`) and Kaiser β of each stage (stage 1 first). Found by
/// scanning β for the least `P` with ≥ 95 dB stopband (see the module docs);
/// checked by the unit tests and by `tests/oversample.rs`.
const STAGE_DESIGN: [(usize, f64); 3] = [(20, 10.45), (8, 10.40), (5, 11.15)];

/// Base frames per inner block.
const CHUNK: usize = 64;
/// Extra room after each buffer, for the whole-vector overrun of the kernels
/// (each kernel works in blocks of two native vectors, up to 2 × 16 lanes).
const SLACK: usize = 128;

/// Zeroth-order modified Bessel function of the first kind, `I0(x)`, from its
/// power series `Σ ((x/2)^k / k!)²`, summed until the terms stop mattering.
pub(crate) fn bessel_i0(x: f64) -> f64 {
    let q = 0.25 * x * x;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut k = 1.0;
    while term > sum * 1e-17 {
        term *= q / (k * k);
        sum += term;
        k += 1.0;
    }
    sum
}

/// Kaiser-windowed half-band low-pass (cut-off at a quarter of its sample
/// rate, DC gain 1): returns the `p` non-zero side taps `h(d)` at odd offsets
/// `d = 1, 3, …, 2p − 1` from the centre tap, which is exactly ½.
///
/// `h(d) = sin(πd/2)/(πd) · w(d)` with the Kaiser window
/// `w(d) = I0(β·√(1 − (d/2p)²)) / I0(β)`. The window spans `±2p`, so its
/// (smallest) end values fall on the zero taps at even offsets `±2p`. The side
/// taps are then scaled so that they sum to exactly ¼: `H(0) = 1` and
/// `H(½) = 0`, since a half-band response satisfies `H(f) + H(½ − f) = 1`.
pub(crate) fn halfband_taps(p: usize, beta: f64) -> Vec<f64> {
    let span = (2 * p) as f64;
    let i0b = bessel_i0(beta);
    let mut h: Vec<f64> = (1..=p)
        .map(|k| {
            let d = (2 * k - 1) as f64;
            let ideal = (PI * d / 2.0).sin() / (PI * d);
            let r = d / span;
            ideal * bessel_i0(beta * (1.0 - r * r).sqrt()) / i0b
        })
        .collect();
    let g = 0.25 / h.iter().sum::<f64>();
    h.iter_mut().for_each(|v| *v *= g);
    h
}

/// Zero-phase response `H(f) = ½ + 2·Σ h(d)·cos(2πfd)` of a half-band given
/// by its side taps, `f` in cycles per sample.
#[cfg(test)]
pub(crate) fn halfband_response(h: &[f64], f: f64) -> f64 {
    h.iter().enumerate().fold(0.5, |acc, (k, c)| {
        acc + 2.0 * c * (2.0 * PI * f * (2 * k + 1) as f64).cos()
    })
}

/// One 2× stage: coefficients, latency choices and per-channel buffers.
#[derive(Debug, Clone)]
struct Stage {
    /// Unique side taps.
    p: usize,
    /// FIR-branch coefficients for interpolation, outermost first, scaled by 2
    /// (the zero-stuffing gain): `up[i] = 2·h(2(p − i) − 1)`.
    up: Vec<f32>,
    /// The same for decimation (gain 1).
    down: Vec<f32>,
    /// Interpolation: delay branch first in each output pair (delay `M + 1`)
    /// rather than the FIR branch first (delay `M`).
    up_delay_first: bool,
    /// Decimation: FIR on the even input samples and delay on the odd ones
    /// (delay `M`) rather than the opposite (delay `M − 1`).
    down_fir_even: bool,
    /// Input history plus block, per channel (`channels × up_stride`).
    up_buf: Vec<f32>,
    up_stride: usize,
    /// Even and odd high-rate input samples for decimation, per channel.
    dn_even: Vec<f32>,
    dn_odd: Vec<f32>,
    dn_stride: usize,
}

impl Stage {
    /// History length `2p − 1` (samples at the stage's input rate, or pairs).
    fn hist(&self) -> usize {
        2 * self.p - 1
    }
}

/// Polyphase 2×/4×/8× up- and down-sampler for interleaved `f32` audio (see the
/// module documentation for the design, latency and SIMD notes).
///
/// Up and down paths keep separate state, so a processor can upsample, work at
/// the high rate, and downsample in the same call sequence. Allocation-free
/// after [`configure`](Self::configure).
#[derive(Debug, Clone)]
pub struct Oversampler {
    factor: OversampleFactor,
    channels: usize,
    stages: Vec<Stage>,
    /// Final-stage output of one block (one channel).
    tmp: Vec<f32>,
    up_latency: usize,
    down_latency: usize,
    level: Level,
    configured: bool,
}

impl Default for Oversampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Oversampler {
    /// Unconfigured oversampler: every call is a no-op until
    /// [`configure`](Self::configure).
    pub fn new() -> Self {
        Self {
            factor: OversampleFactor::X2,
            channels: 0,
            stages: Vec::new(),
            tmp: Vec::new(),
            up_latency: 0,
            down_latency: 0,
            level: crate::simd::detect(),
            configured: false,
        }
    }

    /// Design the filters for `factor` and allocate state for `channels`
    /// interleaved channels (≥ 1). The only method that allocates; clears all
    /// state. The design does not depend on the sample rate (see
    /// [`PASSBAND_EDGE`]).
    pub fn configure(&mut self, factor: OversampleFactor, channels: usize) {
        let channels = channels.max(1);
        let k = factor.stages();
        let mut stages: Vec<Stage> = (0..k)
            .map(|s| {
                let (p, beta) = STAGE_DESIGN[s];
                let h = halfband_taps(p, beta);
                let down: Vec<f32> = (0..p).map(|i| h[p - 1 - i] as f32).collect();
                let up: Vec<f32> = (0..p).map(|i| (2.0 * h[p - 1 - i]) as f32).collect();
                // Block length at this stage's input rate (up) / output rate (down).
                let block = CHUNK << s;
                let up_stride = 2 * p - 1 + block + SLACK;
                let dn_stride = 2 * p - 1 + block + SLACK;
                Stage {
                    p,
                    up,
                    down,
                    up_delay_first: false,
                    down_fir_even: false,
                    up_buf: vec![0.0; channels * up_stride],
                    up_stride,
                    dn_even: vec![0.0; channels * dn_stride],
                    dn_odd: vec![0.0; channels * dn_stride],
                    dn_stride,
                }
            })
            .collect();

        // Delay parities, innermost stage first: `t` is the delay of the
        // stages after this one, in samples at this stage's output rate (an
        // integer by construction); this stage's delay `d` must have the same
        // parity so that `(d + t) / 2` is an integer at its input rate.
        let (mut t_up, mut t_dn) = (0usize, 0usize);
        for st in stages.iter_mut().rev() {
            let m = 2 * st.p - 1;
            st.up_delay_first = t_up % 2 == 0;
            let d_up = if st.up_delay_first { m + 1 } else { m };
            t_up = (d_up + t_up) / 2;
            st.down_fir_even = t_dn % 2 == 1;
            let d_dn = if st.down_fir_even { m } else { m - 1 };
            t_dn = (d_dn + t_dn) / 2;
        }

        self.factor = factor;
        self.channels = channels;
        self.stages = stages;
        self.tmp = vec![0.0; CHUNK * factor.ratio() + SLACK];
        self.up_latency = t_up;
        self.down_latency = t_dn;
        self.configured = true;
    }

    /// Override the SIMD level (default: detected at construction). Tests use
    /// `Level::fallback()` to compare with the scalar path.
    pub fn set_simd_level(&mut self, level: Level) {
        self.level = level;
    }

    /// The configured factor.
    pub fn factor(&self) -> OversampleFactor {
        self.factor
    }

    /// The configured channel count (0 before `configure`).
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Delay of [`upsample`](Self::upsample) in base-rate frames: an impulse at
    /// base frame `n` peaks at high-rate frame `(n + up_latency) · factor`.
    pub fn up_latency_frames(&self) -> usize {
        self.up_latency
    }

    /// Delay of [`downsample`](Self::downsample) in base-rate frames: an
    /// impulse at high-rate frame `n · factor` peaks at base frame
    /// `n + down_latency`.
    pub fn down_latency_frames(&self) -> usize {
        self.down_latency
    }

    /// Round-trip delay (upsample then downsample) in base-rate frames.
    pub fn latency_frames(&self) -> usize {
        self.up_latency + self.down_latency
    }

    /// Clear all filter history (both paths).
    pub fn reset(&mut self) {
        for st in &mut self.stages {
            st.up_buf.fill(0.0);
            st.dn_even.fill(0.0);
            st.dn_odd.fill(0.0);
        }
    }

    /// Upsample interleaved base-rate `input` into `output`
    /// (`output.len() = input.len() · factor`, same channel interleaving).
    /// Processes the frames both slices can hold; a no-op if unconfigured.
    pub fn upsample(&mut self, input: &[f32], output: &mut [f32]) {
        if !self.configured {
            return;
        }
        let level = self.level;
        dispatch!(level, simd => self.run_up(simd, input, Sink::Samples(output)));
    }

    /// Upsample `input` and store, per base frame, the largest absolute value
    /// of the oversampled signal over all sub-samples and channels (an
    /// inter-sample peak estimate, delayed by
    /// [`up_latency_frames`](Self::up_latency_frames)). `peaks.len()` is the
    /// frame count. Shares the up-path state with [`upsample`](Self::upsample).
    pub fn upsample_peaks(&mut self, input: &[f32], peaks: &mut [f32]) {
        if !self.configured {
            return;
        }
        let level = self.level;
        dispatch!(level, simd => self.run_up(simd, input, Sink::Peaks(peaks)));
    }

    /// Downsample interleaved high-rate `input` into `output`
    /// (`input.len() = output.len() · factor`). Processes the frames both
    /// slices can hold; a no-op if unconfigured.
    pub fn downsample(&mut self, input: &[f32], output: &mut [f32]) {
        if !self.configured {
            return;
        }
        let level = self.level;
        dispatch!(level, simd => self.run_down(simd, input, output));
    }

    fn run_up<S: Simd>(&mut self, simd: S, input: &[f32], mut sink: Sink<'_>) {
        let ch = self.channels;
        let l = self.factor.ratio();
        let frames = match &sink {
            Sink::Samples(out) => (input.len() / ch).min(out.len() / (ch * l)),
            Sink::Peaks(p) => (input.len() / ch).min(p.len()),
        };
        let k = self.stages.len();
        let mut f0 = 0;
        while f0 < frames {
            let n0 = CHUNK.min(frames - f0);
            if let Sink::Peaks(p) = &mut sink {
                p[f0..f0 + n0].fill(0.0);
            }
            for c in 0..ch {
                // Gather the channel into the first stage's buffer.
                {
                    let st = &mut self.stages[0];
                    let h = st.hist();
                    let buf = &mut st.up_buf[c * st.up_stride + h..];
                    for (i, d) in buf[..n0].iter_mut().enumerate() {
                        *d = input[(f0 + i) * ch + c];
                    }
                }
                for s in 0..k {
                    let n = n0 << s;
                    let (cur, rest) = self.stages.split_at_mut(s + 1);
                    let st = &cur[s];
                    let src = &st.up_buf[c * st.up_stride..(c + 1) * st.up_stride];
                    let dst: &mut [f32] = match rest.first_mut() {
                        Some(nx) => {
                            let h = nx.hist();
                            &mut nx.up_buf[c * nx.up_stride + h..(c + 1) * nx.up_stride]
                        }
                        None => &mut self.tmp,
                    };
                    let shift = if st.up_delay_first { st.p } else { st.p - 1 };
                    let args = UpArgs {
                        coefs: &st.up,
                        src,
                        n,
                        shift,
                        delay_first: st.up_delay_first,
                        dst,
                    };
                    match st.p {
                        20 => up_kernel::<S, 20>(simd, args),
                        8 => up_kernel::<S, 8>(simd, args),
                        5 => up_kernel::<S, 5>(simd, args),
                        _ => unreachable!("stage designs are fixed"),
                    }
                }
                let y = &self.tmp[..n0 * l];
                match &mut sink {
                    Sink::Samples(out) => {
                        let base = f0 * l * ch + c;
                        for (i, v) in y.iter().enumerate() {
                            out[base + i * ch] = *v;
                        }
                    }
                    Sink::Peaks(p) => {
                        for (pk, sub) in p[f0..f0 + n0].iter_mut().zip(y.chunks_exact(l)) {
                            let m = sub.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                            *pk = pk.max(m);
                        }
                    }
                }
                // Keep the newest `2p − 1` inputs of every stage as history.
                for (s, st) in self.stages.iter_mut().enumerate() {
                    let n = n0 << s;
                    let h = st.hist();
                    let b = c * st.up_stride;
                    st.up_buf.copy_within(b + n..b + n + h, b);
                }
            }
            f0 += n0;
        }
    }

    fn run_down<S: Simd>(&mut self, simd: S, input: &[f32], output: &mut [f32]) {
        let ch = self.channels;
        let l = self.factor.ratio();
        let frames = (output.len() / ch).min(input.len() / (ch * l));
        let k = self.stages.len();
        let mut f0 = 0;
        while f0 < frames {
            let n0 = CHUNK.min(frames - f0);
            for c in 0..ch {
                // Split the channel's high-rate samples into even/odd streams
                // of the last (highest-rate) stage.
                {
                    let st = &mut self.stages[k - 1];
                    let h = st.hist();
                    let b = c * st.dn_stride + h;
                    let src = &input[f0 * l * ch..];
                    let pairs = n0 * l / 2;
                    for i in 0..pairs {
                        st.dn_even[b + i] = src[(2 * i) * ch + c];
                        st.dn_odd[b + i] = src[(2 * i + 1) * ch + c];
                    }
                }
                for s in (0..k).rev() {
                    // Output samples of this stage = input pairs.
                    let n = n0 << s;
                    let (lower, upper) = self.stages.split_at_mut(s);
                    let st = &upper[0];
                    let b = c * st.dn_stride;
                    let (fir, dly) = if st.down_fir_even {
                        (&st.dn_even, &st.dn_odd)
                    } else {
                        (&st.dn_odd, &st.dn_even)
                    };
                    let shift = if st.down_fir_even { st.p } else { st.p - 1 };
                    let dst = match lower.last_mut() {
                        Some(nx) => {
                            let h = nx.hist();
                            let r = c * nx.dn_stride + h..(c + 1) * nx.dn_stride;
                            DownDst::Split(&mut nx.dn_even[r.clone()], &mut nx.dn_odd[r])
                        }
                        None => DownDst::Flat(&mut self.tmp),
                    };
                    let args = DownArgs {
                        coefs: &st.down,
                        fir: &fir[b..b + st.dn_stride],
                        dly: &dly[b..b + st.dn_stride],
                        n,
                        shift,
                        dst,
                    };
                    match st.p {
                        20 => down_kernel::<S, 20>(simd, args),
                        8 => down_kernel::<S, 8>(simd, args),
                        5 => down_kernel::<S, 5>(simd, args),
                        _ => unreachable!("stage designs are fixed"),
                    }
                }
                for (i, v) in self.tmp[..n0].iter().enumerate() {
                    output[(f0 + i) * ch + c] = *v;
                }
                for (s, st) in self.stages.iter_mut().enumerate() {
                    let n = n0 << s;
                    let h = st.hist();
                    let b = c * st.dn_stride;
                    st.dn_even.copy_within(b + n..b + n + h, b);
                    st.dn_odd.copy_within(b + n..b + n + h, b);
                }
            }
            f0 += n0;
        }
    }
}

/// Where the up path's final stage goes.
enum Sink<'a> {
    Samples(&'a mut [f32]),
    Peaks(&'a mut [f32]),
}

/// `a·b + c` per the crate's FMA policy (see [`crate::lanes`]), for `f32`.
#[inline(always)]
fn madd32<S: Simd, V: SimdFloat<S, Element = f32>>(fma: Fma, a: V, b: V, c: V) -> V {
    match fma {
        Fma::Vector => a.mul_add(b, c),
        Fma::Scalar => V::from_fn(a.token(), |i| a[i].mul_add(b[i], c[i])),
        Fma::Unfused => a * b + c,
    }
}

/// Load one native vector from `src[at..]`.
#[inline(always)]
fn ld<S: Simd>(simd: S, src: &[f32], at: usize) -> S::f32s {
    S::f32s::from_slice(simd, &src[at..at + S::f32s::LEN])
}

/// The folded symmetric FIR branch for the `W` outputs starting at `j`:
/// `Σ_i c[i]·(x[h + j − i] + x[j + i])` over a buffer whose first `h = 2P − 1`
/// samples are history. Two accumulators (even and odd `i`) halve the
/// dependency chain.
#[inline(always)]
fn fir_branch<S: Simd, const P: usize>(
    simd: S,
    fma: Fma,
    c: &[S::f32s; P],
    x: &[f32],
    j: usize,
) -> S::f32s {
    let h = 2 * P - 1;
    let mut a0 = S::f32s::splat(simd, 0.0);
    let mut a1 = S::f32s::splat(simd, 0.0);
    let mut i = 0;
    while i + 1 < P {
        let s0 = ld(simd, x, h + j - i) + ld(simd, x, j + i);
        a0 = madd32(fma, s0, c[i], a0);
        let s1 = ld(simd, x, h + j - i - 1) + ld(simd, x, j + i + 1);
        a1 = madd32(fma, s1, c[i + 1], a1);
        i += 2;
    }
    if i < P {
        let s0 = ld(simd, x, h + j - i) + ld(simd, x, j + i);
        a0 = madd32(fma, s0, c[i], a0);
    }
    a0 + a1
}

struct UpArgs<'a> {
    coefs: &'a [f32],
    /// History (`2P − 1`) then `n` new input samples.
    src: &'a [f32],
    n: usize,
    /// Delay-branch tap: output uses `src[h + j − shift]`.
    shift: usize,
    delay_first: bool,
    /// Receives `2n` samples (plus whole-vector overrun).
    dst: &'a mut [f32],
}

/// One interpolation stage: for each input sample, an output pair (FIR branch,
/// delay branch) or (delay, FIR), written contiguously.
fn up_kernel<S: Simd, const P: usize>(simd: S, a: UpArgs<'_>) {
    simd.vectorize(
        #[inline(always)]
        || {
            let w = S::f32s::LEN;
            let h = 2 * P - 1;
            let fma = fma_mode(simd.level());
            let c: [S::f32s; P] = core::array::from_fn(|i| S::f32s::splat(simd, a.coefs[i]));
            let mut j = 0;
            while j < a.n {
                for jj in [j, j + w] {
                    let fir = fir_branch::<S, P>(simd, fma, &c, a.src, jj);
                    let dly = ld(simd, a.src, h + jj - a.shift);
                    let (lo, hi) = if a.delay_first {
                        dly.interleave(fir)
                    } else {
                        fir.interleave(dly)
                    };
                    lo.store_slice(&mut a.dst[2 * jj..2 * jj + w]);
                    hi.store_slice(&mut a.dst[2 * jj + w..2 * jj + 2 * w]);
                }
                j += 2 * w;
            }
        },
    )
}

/// Destination of a decimation stage: the even/odd streams of the next
/// (lower-rate) stage, or a flat buffer for the last one.
enum DownDst<'a> {
    Split(&'a mut [f32], &'a mut [f32]),
    Flat(&'a mut [f32]),
}

struct DownArgs<'a> {
    coefs: &'a [f32],
    /// Stream through the FIR branch (history, then `n` samples).
    fir: &'a [f32],
    /// Stream through the delay branch (gain ½).
    dly: &'a [f32],
    n: usize,
    shift: usize,
    dst: DownDst<'a>,
}

/// One decimation stage: `y[j] = FIR(fir)[j] + ½·dly[h + j − shift]`.
fn down_kernel<S: Simd, const P: usize>(simd: S, a: DownArgs<'_>) {
    simd.vectorize(
        #[inline(always)]
        || {
            let w = S::f32s::LEN;
            let h = 2 * P - 1;
            let fma = fma_mode(simd.level());
            let c: [S::f32s; P] = core::array::from_fn(|i| S::f32s::splat(simd, a.coefs[i]));
            let half = S::f32s::splat(simd, 0.5);
            let DownArgs {
                fir, dly, n, shift, ..
            } = a;
            let mut dst = a.dst;
            let mut j = 0;
            while j < n {
                let y = |jj: usize| {
                    let f = fir_branch::<S, P>(simd, fma, &c, fir, jj);
                    madd32(fma, ld(simd, dly, h + jj - shift), half, f)
                };
                let y0 = y(j);
                let y1 = y(j + w);
                match &mut dst {
                    DownDst::Flat(out) => {
                        y0.store_slice(&mut out[j..j + w]);
                        y1.store_slice(&mut out[j + w..j + 2 * w]);
                    }
                    DownDst::Split(even, odd) => {
                        let (e, o) = y0.deinterleave(y1);
                        let k = j / 2;
                        e.store_slice(&mut even[k..k + w]);
                        o.store_slice(&mut odd[k..k + w]);
                    }
                }
                j += 2 * w;
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Largest |H| (dB) in the stopband and largest |20·log10 H| in the
    /// passband of a stage whose passband edge is `fp` (cycles/sample at the
    /// stage's output rate).
    fn measure(h: &[f64], fp: f64) -> (f64, f64) {
        let n = 4000;
        let (mut sb, mut pb) = (0.0f64, 0.0f64);
        for i in 0..=n {
            let t = i as f64 / n as f64;
            sb = sb.max(halfband_response(h, 0.5 - fp + fp * t).abs());
            pb = pb.max((20.0 * halfband_response(h, fp * t).log10()).abs());
        }
        (20.0 * sb.log10(), pb)
    }

    /// Analytic check of each stage design against the edges it is
    /// responsible for (module documentation).
    #[test]
    fn stage_designs_meet_spec() {
        let r = PASSBAND_EDGE;
        let edges = [r / 2.0, (1.0 - r) / 4.0, (1.0 - r) / 8.0];
        for ((p, beta), fp) in STAGE_DESIGN.iter().zip(edges) {
            let h = halfband_taps(*p, *beta);
            let (sb, pb) = measure(&h, fp);
            assert!(sb < -95.0, "p={p}: stopband {sb:.1} dB");
            assert!(pb < 0.001, "p={p}: passband ripple {pb:.2e} dB");
            assert!((halfband_response(&h, 0.0) - 1.0).abs() < 1e-14);
            assert!(halfband_response(&h, 0.5).abs() < 1e-14);
        }
    }

    #[test]
    fn bessel_i0_values() {
        assert_eq!(bessel_i0(0.0), 1.0);
        // I0(1) = 1.2660658777520082, I0(10) = 2815.716628466254.
        assert!((bessel_i0(1.0) - 1.266_065_877_752_008_2).abs() < 1e-14);
        assert!((bessel_i0(10.0) / 2_815.716_628_466_254 - 1.0).abs() < 1e-13);
    }

    #[test]
    fn latencies_are_as_documented() {
        let mut o = Oversampler::new();
        for (f, up, dn) in [
            (OversampleFactor::X2, 20, 19),
            (OversampleFactor::X4, 24, 23),
            (OversampleFactor::X8, 25, 24),
        ] {
            o.configure(f, 1);
            assert_eq!((o.up_latency_frames(), o.down_latency_frames()), (up, dn));
        }
    }
}
