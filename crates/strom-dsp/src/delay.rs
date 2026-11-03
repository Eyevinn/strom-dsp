//! Intentional multichannel delay with per-channel offsets and click-free changes.
//!
//! [`Delay`] delays every channel of an interleaved buffer by a whole number
//! of frames: a common delay ([`DelayParams::delay_ms`]) plus an optional
//! per-channel offset ([`DelayParams::channel_offsets_ms`]). It is meant for
//! content alignment (microphone alignment, lip sync), so it is **not**
//! processing latency: [`Processor::latency_frames`] stays 0.
//!
//! # Delay per channel
//!
//! `total_c = clamp(delay_ms + offset_c, 0, MAX_DELAY_MS)`, rounded to the
//! nearest whole frame: `round(total_c · fs / 1000)`. Offsets may be negative
//! (advance one channel relative to the others, as long as the total stays
//! ≥ 0). Channels without an offset entry (the list is shorter than the
//! channel count, or the channel index is ≥ [`MAX_OFFSET_CHANNELS`]) use
//! offset 0; entries beyond the channel count are ignored.
//!
//! # Memory
//!
//! [`configure`](Processor::configure) allocates one interleaved ring for
//! [`MAX_DELAY_MS`] at the configured rate and channel count (plus one block
//! of headroom), whatever the current delay is. Nothing allocates afterwards.
//!
//! # Changes while running: crossfade
//!
//! A change of any channel's delay crossfades from the old read tap to the new
//! one over [`XFADE_MS`] with an equal-gain raised-cosine fade:
//!
//! `y[n] = (1 − g_k)·x[n − d_old] + g_k·x[n − d_new]`,
//! `g_k = ½ − ½·cos(π·k / (N + 1))`, `k = 1 … N`, `N = round(XFADE_MS · fs / 1000)`.
//!
//! Channels whose delay does not change are copied unchanged. A change that
//! arrives during a crossfade is queued: the next crossfade starts from the
//! reached delay when the current one ends (the latest target wins).
//!
//! **Discontinuity bound.** With `D = max |x[n] − x[n−1]|` (the largest step
//! already in the input) and `A = max |x|`, every output step during a
//! crossfade satisfies
//!
//! `|y[n] − y[n−1]| ≤ D + Δg · 2A`, `Δg = max_k (g_k − g_{k−1}) ≤ π / (2(N + 1))`,
//!
//! because `y[n] − y[n−1] = (1−g)(a[n] − a[n−1]) + g(b[n] − b[n−1]) +
//! (g_k − g_{k−1})(b[n−1] − a[n−1])`. At 48 kHz (`N = 960`) the extra term is at
//! most 0.0016·2A, i.e. about −50 dB relative to a hard switch (`2A`). Outside
//! crossfades the output is the input delayed, so its steps are the input's.
//! [`max_crossfade_step`] returns `π / (2(N + 1))` for a sample rate. The output
//! never exceeds `A` (equal-gain fades of two values stay between them).
//!
//! # Enabled
//!
//! `enabled = false` crossfades every channel to delay 0. Once that crossfade
//! has finished, the output is a **bit-exact** copy of the input. The ring keeps
//! being written, so re-enabling crossfades back to the delayed signal without
//! a gap. Delay 0 (the default) is bit-exact too.
//!
//! As everywhere in this crate, between [`configure`](Processor::configure) or
//! [`reset`](Processor::reset) and the first processed frame, new parameters
//! apply instantly (no crossfade), so the call order does not change the output.

use crate::dynamics::DelayLine;
use crate::Processor;

/// Largest total delay per channel (ms).
pub const MAX_DELAY_MS: f32 = 2000.0;

/// Number of channels that can carry a per-channel offset. Channels at a
/// higher index use offset 0.
pub const MAX_OFFSET_CHANNELS: usize = 64;

/// Duration of the crossfade between the old and new read taps (ms).
pub const XFADE_MS: f32 = 20.0;

/// Frames processed per inner block; the ring has this much headroom so a
/// block can be written before it is read.
const BLOCK: usize = 256;

/// Parameters of [`Delay`]. `Copy`, so hosts can hand them over without allocating.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DelayParams {
    /// `false`: crossfade to delay 0 on every channel, then bit-exact copy.
    pub enabled: bool,
    /// Common delay for every channel (ms, 0 … [`MAX_DELAY_MS`]).
    pub delay_ms: f32,
    /// Per-channel offsets (ms, −[`MAX_DELAY_MS`] … [`MAX_DELAY_MS`]); only
    /// the first [`num_offsets`](Self::num_offsets) entries are used.
    pub channel_offsets_ms: [f32; MAX_OFFSET_CHANNELS],
    /// Number of valid entries in `channel_offsets_ms` (≤ [`MAX_OFFSET_CHANNELS`]).
    pub num_offsets: usize,
}

impl Default for DelayParams {
    fn default() -> Self {
        Self {
            enabled: true,
            delay_ms: 0.0,
            channel_offsets_ms: [0.0; MAX_OFFSET_CHANNELS],
            num_offsets: 0,
        }
    }
}

impl DelayParams {
    /// Set the per-channel offsets from a slice (ms). Entries beyond
    /// [`MAX_OFFSET_CHANNELS`] are dropped; missing channels use 0.
    pub fn set_offsets(&mut self, offsets_ms: &[f32]) {
        let n = offsets_ms.len().min(MAX_OFFSET_CHANNELS);
        self.channel_offsets_ms = [0.0; MAX_OFFSET_CHANNELS];
        self.channel_offsets_ms[..n].copy_from_slice(&offsets_ms[..n]);
        self.num_offsets = n;
    }

    /// The valid per-channel offsets (ms).
    pub fn offsets(&self) -> &[f32] {
        &self.channel_offsets_ms[..self.num_offsets.min(MAX_OFFSET_CHANNELS)]
    }

    /// Total delay of `channel` in ms, clamped to `0 … MAX_DELAY_MS`
    /// (0 when disabled). NaN parameters count as 0.
    pub fn channel_delay_ms(&self, channel: usize) -> f32 {
        if !self.enabled {
            return 0.0;
        }
        let base = finite_or_zero(self.delay_ms);
        let off = self
            .offsets()
            .get(channel)
            .copied()
            .map_or(0.0, finite_or_zero);
        (base + off).clamp(0.0, MAX_DELAY_MS)
    }
}

fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// `ms` at `sample_rate` Hz as whole frames, rounded to nearest.
pub fn ms_to_frames(ms: f32, sample_rate: f32) -> usize {
    let f = f64::from(ms.max(0.0)) * f64::from(sample_rate.max(0.0)) / 1000.0;
    if f.is_finite() {
        f.round() as usize
    } else {
        0
    }
}

/// Crossfade length in frames at `sample_rate` (≥ 1).
pub fn crossfade_frames(sample_rate: f32) -> usize {
    ms_to_frames(XFADE_MS, sample_rate).max(1)
}

/// Upper bound of the per-sample crossfade gain increment, `π / (2(N + 1))`
/// (see the module docs for the discontinuity bound it gives).
pub fn max_crossfade_step(sample_rate: f32) -> f64 {
    std::f64::consts::PI / (2.0 * (crossfade_frames(sample_rate) + 1) as f64)
}

/// Intentional multichannel delay (see the module documentation).
#[derive(Debug, Clone, Default)]
pub struct Delay {
    params: DelayParams,
    sample_rate: f32,
    channels: usize,
    line: DelayLine,
    max_frames: usize,
    /// Current read delay per channel (frames).
    cur: Vec<usize>,
    /// Target read delay per channel (frames), from the parameters.
    target: Vec<usize>,
    /// Delay each channel fades to during an active crossfade.
    next: Vec<usize>,
    xfade_frames: usize,
    /// Position in the active crossfade (`0` = none active).
    xfade_pos: usize,
    /// No frame processed since configure/reset: parameter changes snap.
    fresh: bool,
    /// Every channel's current delay is 0 and no crossfade is active.
    all_zero: bool,
}

impl Delay {
    /// Unconfigured delay (process is a no-op until [`Processor::configure`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// Current parameters.
    pub fn params(&self) -> &DelayParams {
        &self.params
    }

    /// Set parameters. Never allocates. A changed delay starts a crossfade
    /// (or is queued behind the running one).
    pub fn set_params(&mut self, p: &DelayParams) {
        self.params = *p;
        self.params.num_offsets = p.num_offsets.min(MAX_OFFSET_CHANNELS);
        self.update_targets();
    }

    /// Delay currently applied to `channel`, in frames (during a crossfade:
    /// the delay it fades from).
    pub fn current_delay_frames(&self, channel: usize) -> usize {
        self.cur.get(channel).copied().unwrap_or(0)
    }

    /// Delay `channel` is heading for, in frames.
    pub fn target_delay_frames(&self, channel: usize) -> usize {
        self.target.get(channel).copied().unwrap_or(0)
    }

    /// Whether a crossfade is running or queued.
    pub fn is_changing(&self) -> bool {
        self.xfade_pos > 0 || self.cur != self.target
    }

    /// Largest supported delay in frames at the configured rate.
    pub fn max_delay_frames(&self) -> usize {
        self.max_frames
    }

    fn update_targets(&mut self) {
        for (c, t) in self.target.iter_mut().enumerate() {
            *t = ms_to_frames(self.params.channel_delay_ms(c), self.sample_rate)
                .min(self.max_frames);
        }
        if self.fresh {
            self.cur.copy_from_slice(&self.target);
            self.next.copy_from_slice(&self.target);
            self.xfade_pos = 0;
        }
        self.refresh_all_zero();
    }

    fn refresh_all_zero(&mut self) {
        self.all_zero = self.xfade_pos == 0 && self.cur.iter().all(|&d| d == 0);
    }

    /// Start a crossfade toward the targets if they differ from the current delays.
    fn start_crossfade_if_needed(&mut self) {
        if self.xfade_pos == 0 && self.cur != self.target {
            self.next.copy_from_slice(&self.target);
            self.xfade_pos = 1;
            self.all_zero = false;
        }
    }

    /// Process `n` frames of `block` (already written to the ring) without crossfade.
    fn read_steady(&mut self, block: &mut [f32], n: usize) {
        let ch = self.channels;
        let d0 = self.cur[0];
        if self.cur.iter().all(|&d| d == d0) {
            self.line.read_frames(d0, block);
            return;
        }
        for (i, frame) in block.chunks_exact_mut(ch).enumerate() {
            let back = n - 1 - i;
            for (c, s) in frame.iter_mut().enumerate() {
                *s = self.line.read(self.cur[c] + back)[c];
            }
        }
    }

    /// Process frames during a crossfade; returns the number of frames handled
    /// (stops when the crossfade ends).
    fn read_xfade(&mut self, block: &mut [f32], n: usize, start: usize) -> usize {
        let ch = self.channels;
        let denom = (self.xfade_frames + 1) as f64;
        let mut i = start;
        while i < n && self.xfade_pos > 0 {
            let g = 0.5 - 0.5 * (std::f64::consts::PI * self.xfade_pos as f64 / denom).cos();
            let back = n - 1 - i;
            let frame = &mut block[i * ch..(i + 1) * ch];
            for (c, s) in frame.iter_mut().enumerate() {
                let a = self.line.read(self.cur[c] + back)[c];
                if self.cur[c] == self.next[c] {
                    *s = a;
                } else {
                    let b = self.line.read(self.next[c] + back)[c];
                    *s = ((1.0 - g) * f64::from(a) + g * f64::from(b)) as f32;
                }
            }
            i += 1;
            if self.xfade_pos >= self.xfade_frames {
                self.xfade_pos = 0;
                self.cur.copy_from_slice(&self.next);
                // A target that changed meanwhile starts the next crossfade.
                self.start_crossfade_if_needed();
                self.refresh_all_zero();
            } else {
                self.xfade_pos += 1;
            }
        }
        i
    }
}

impl Processor for Delay {
    fn configure(&mut self, sample_rate: f32, channels: usize) {
        let channels = channels.max(1);
        self.sample_rate = sample_rate;
        self.channels = channels;
        self.max_frames = ms_to_frames(MAX_DELAY_MS, sample_rate);
        self.line.allocate(self.max_frames + BLOCK, channels);
        self.cur = vec![0; channels];
        self.target = vec![0; channels];
        self.next = vec![0; channels];
        self.xfade_frames = crossfade_frames(sample_rate);
        self.reset();
    }

    fn reset(&mut self) {
        self.line.clear();
        self.xfade_pos = 0;
        self.fresh = true;
        self.update_targets();
    }

    fn process(&mut self, samples: &mut [f32]) {
        let ch = self.channels;
        if ch == 0 {
            return;
        }
        let usable = samples.len() / ch * ch;
        if usable == 0 {
            return;
        }
        if self.fresh {
            self.fresh = false;
        }
        self.start_crossfade_if_needed();
        for block in samples[..usable].chunks_mut(BLOCK * ch) {
            let n = block.len() / ch;
            self.line.write_frames(block);
            if self.all_zero {
                // Delay 0 everywhere: the output is the input (bit-exact).
                continue;
            }
            let mut i = 0;
            while i < n {
                if self.xfade_pos > 0 {
                    i = self.read_xfade(block, n, i);
                } else {
                    let rest = &mut block[i * ch..];
                    let m = n - i;
                    if !self.all_zero {
                        self.read_steady(rest, m);
                    }
                    i = n;
                }
            }
        }
    }
}
