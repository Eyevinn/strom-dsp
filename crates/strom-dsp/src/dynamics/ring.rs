//! Preallocated ring structures: a multichannel delay line and a sliding-window
//! minimum (monotonic deque, the standard ascending-minima technique, and the
//! van Herk / Gil–Werman segment algorithm).

/// Multichannel delay line for interleaved `f32` frames.
///
/// Storage is a power-of-two number of frames, allocated once by
/// [`allocate`](Self::allocate). [`write`](Self::write) appends one frame;
/// [`read`](Self::read)`(d)` returns the frame written `d` frames before the most
/// recent one (`d = 0` is the frame just written), for any `d ≤ max_delay()`.
/// Before anything is written (or after [`clear`](Self::clear)) the line holds
/// silence.
#[derive(Debug, Clone, Default)]
pub struct DelayLine {
    buf: Vec<f32>,
    channels: usize,
    /// Frame mask (capacity − 1).
    mask: usize,
    /// Frame index of the most recent write.
    pos: usize,
}

impl DelayLine {
    /// Allocate for delays up to `max_delay_frames` frames of `channels` channels
    /// (≥ 1) and clear. The only allocating method.
    pub fn allocate(&mut self, max_delay_frames: usize, channels: usize) {
        let channels = channels.max(1);
        let frames = (max_delay_frames + 1).next_power_of_two();
        self.buf = vec![0.0; frames * channels];
        self.channels = channels;
        self.mask = frames - 1;
        self.pos = 0;
    }

    /// Largest supported delay in frames (0 before `allocate`).
    pub fn max_delay(&self) -> usize {
        if self.buf.is_empty() {
            0
        } else {
            self.mask
        }
    }

    /// Channel count (0 before `allocate`).
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Fill with silence.
    pub fn clear(&mut self) {
        self.buf.fill(0.0);
    }

    /// Append one frame (`frame.len()` must equal the channel count).
    #[inline]
    pub fn write(&mut self, frame: &[f32]) {
        self.pos = (self.pos + 1) & self.mask;
        let i = self.pos * self.channels;
        self.buf[i..i + self.channels].copy_from_slice(frame);
    }

    /// The frame written `delay` frames before the most recent one
    /// (`delay ≤ max_delay()`; larger values wrap).
    #[inline]
    pub fn read(&self, delay: usize) -> &[f32] {
        let i = (self.pos.wrapping_sub(delay) & self.mask) * self.channels;
        &self.buf[i..i + self.channels]
    }

    /// Append several interleaved frames (`frames.len()` a multiple of the
    /// channel count), equivalent to calling [`write`](Self::write) per frame
    /// but with bulk copies.
    #[inline]
    pub fn write_frames(&mut self, frames: &[f32]) {
        let ch = self.channels;
        if ch == 0 {
            return;
        }
        let cap = self.mask + 1;
        let mut src = &frames[..frames.len() / ch * ch];
        while !src.is_empty() {
            let start = (self.pos + 1) & self.mask;
            let n = (src.len() / ch).min(cap - start);
            let (head, tail) = src.split_at(n * ch);
            self.buf[start * ch..(start + n) * ch].copy_from_slice(head);
            self.pos = (self.pos + n) & self.mask;
            src = tail;
        }
    }

    /// Copy `out.len() / channels` consecutive frames into `out`, oldest
    /// first, ending with the frame [`read`](Self::read)`(delay)` returns.
    /// `delay + frames − 1` must not exceed [`max_delay`](Self::max_delay).
    #[inline]
    pub fn read_frames(&self, delay: usize, out: &mut [f32]) {
        let ch = self.channels;
        if ch == 0 {
            return;
        }
        let cap = self.mask + 1;
        let frames = out.len() / ch;
        let mut dst = &mut out[..frames * ch];
        let mut idx = self.pos.wrapping_sub(delay + frames).wrapping_add(1) & self.mask;
        while !dst.is_empty() {
            let n = (dst.len() / ch).min(cap - idx);
            let (head, tail) = dst.split_at_mut(n * ch);
            head.copy_from_slice(&self.buf[idx * ch..(idx + n) * ch]);
            idx = (idx + n) & self.mask;
            dst = tail;
        }
    }
}

/// Sliding-window minimum over time-stamped values, O(1) amortised per push.
///
/// Values are pushed with non-decreasing time stamps; [`expire`](Self::expire)
/// drops entries older than a given time, and [`min`](Self::min) is the minimum
/// of the remaining ones. Internally a monotonic deque (values non-decreasing
/// from front to back) in a ring preallocated by [`allocate`](Self::allocate).
/// The capacity must be at least the number of values that can be live at once
/// (the window length); if it is exceeded, the oldest entry is dropped.
///
/// For a sliding maximum, push negated values.
#[derive(Debug, Clone, Default)]
pub struct SlidingMin {
    ring: Vec<(u64, f64)>,
    mask: usize,
    head: usize,
    len: usize,
}

impl SlidingMin {
    /// Allocate for at least `capacity` live entries (rounded up to a power of
    /// two) and clear. The only allocating method.
    pub fn allocate(&mut self, capacity: usize) {
        let cap = capacity.max(1).next_power_of_two();
        self.ring = vec![(0, 0.0); cap];
        self.mask = cap - 1;
        self.clear();
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// Number of live entries (the deque, not the window).
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no live entries.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Add value `v` at time `t` (`t` not older than earlier pushes).
    #[inline(always)]
    pub fn push(&mut self, t: u64, v: f64) {
        if self.ring.is_empty() {
            return;
        }
        // Entries ≥ v can never be the minimum again.
        while self.len > 0 && self.ring[(self.head + self.len - 1) & self.mask].1 >= v {
            self.len -= 1;
        }
        if self.len == self.ring.len() {
            self.head = (self.head + 1) & self.mask;
            self.len -= 1;
        }
        self.ring[(self.head + self.len) & self.mask] = (t, v);
        self.len += 1;
    }

    /// Drop entries with time stamp `< oldest`.
    #[inline(always)]
    pub fn expire(&mut self, oldest: u64) {
        while self.len > 0 && self.ring[self.head].0 < oldest {
            self.head = (self.head + 1) & self.mask;
            self.len -= 1;
        }
    }

    /// Minimum of the live entries, or `None` if empty.
    #[inline(always)]
    pub fn min(&self) -> Option<f64> {
        if self.len > 0 {
            Some(self.ring[self.head].1)
        } else {
            None
        }
    }
}

/// Sliding-window minimum over the last `w` pushed values with a fixed,
/// branch-light cost: the van Herk / Gil–Werman algorithm.
///
/// References: M. van Herk, "A fast algorithm for local minimum and maximum
/// filters on rectangular and octagonal kernels", Pattern Recognition
/// Letters 13(7), 1992; J. Gil and M. Werman, "Computing 2-D min, median,
/// and max filters", IEEE Trans. PAMI 15(5), 1993.
///
/// The input stream is cut into segments of `w` consecutive values. A window
/// of `w` values either is one whole segment, or covers the tail of one
/// segment and the head of the next, so its minimum is
/// `min(suffix_min(previous segment from the window start), prefix_min(current
/// segment up to the newest value))`. The prefix minimum is a running value;
/// the suffix minima of a segment are computed backwards once it is complete.
/// That is about three comparisons per value, independent of the data,
/// where the monotonic deque of [`SlidingMin`] has a data-dependent loop.
///
/// The result is the exact minimum (comparisons only, no arithmetic), so it
/// equals [`SlidingMin`] for the same window. Before `w` values have been
/// pushed after [`reset`](Self::reset), it is the minimum of the values pushed
/// so far. NaN inputs are not supported (they are never the minimum).
#[derive(Debug, Clone, Default)]
pub struct SegmentedMin {
    /// Values of the current (incomplete) segment, by position.
    current: Vec<f64>,
    /// Suffix minima of the previous segment, by position (`+∞` after reset).
    suffix: Vec<f64>,
    window: usize,
    /// Position of the next value within its segment (`0 .. window`).
    pos: usize,
    /// Minimum of the current segment so far.
    prefix: f64,
}

impl SegmentedMin {
    /// Allocate for windows of up to `max_window` values and reset to a
    /// window of 1. The only allocating method.
    pub fn allocate(&mut self, max_window: usize) {
        let cap = max_window.max(1);
        self.current = vec![f64::INFINITY; cap];
        // One spare slot: the window start's suffix index can be `window`.
        self.suffix = vec![f64::INFINITY; cap + 1];
        self.reset(1);
    }

    /// Window length in values (≥ 1).
    pub fn window(&self) -> usize {
        self.window
    }

    /// Forget all values and set the window length (clamped to
    /// `1 ..= capacity`). O(window).
    pub fn reset(&mut self, window: usize) {
        self.window = window.clamp(1, self.current.len().max(1));
        self.pos = 0;
        self.prefix = f64::INFINITY;
        let w = self.window;
        if let Some(s) = self.suffix.get_mut(..=w) {
            s.fill(f64::INFINITY);
        }
    }

    /// Push the newest value and return the minimum of the last `window`
    /// values (of fewer, right after a reset).
    #[inline(always)]
    pub fn push(&mut self, v: f64) -> f64 {
        let w = self.window;
        let j = self.pos;
        let (Some(c), Some(&s)) = (self.current.get_mut(j), self.suffix.get(j + 1)) else {
            return v; // not allocated
        };
        *c = v;
        // Running prefix minimum of the current segment (restarts at j = 0).
        let p = if j == 0 || v < self.prefix {
            v
        } else {
            self.prefix
        };
        self.prefix = p;
        // The window starts at position j + 1 of the previous segment
        // (`suffix[w]` is +∞: then the window is exactly the current segment).
        let out = if s < p { s } else { p };
        if j + 1 == w {
            // Segment complete: its suffix minima serve the next segment.
            let mut m = f64::INFINITY;
            for (s, &c) in self.suffix[..w].iter_mut().zip(&self.current[..w]).rev() {
                if c < m {
                    m = c;
                }
                *s = m;
            }
            self.pos = 0;
        } else {
            self.pos = j + 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_line_delays_each_channel() {
        let mut d = DelayLine::default();
        d.allocate(5, 2);
        assert!(d.max_delay() >= 5);
        for n in 0..20 {
            d.write(&[n as f32, -(n as f32)]);
            for k in 0..=5usize {
                let want = if n >= k { (n - k) as f32 } else { 0.0 };
                assert_eq!(d.read(k), &[want, -want][..], "n={n} k={k}");
            }
        }
        d.clear();
        assert_eq!(d.read(0), &[0.0, 0.0][..]);
    }

    #[test]
    fn delay_line_block_access_matches_per_frame() {
        for ch in [1usize, 2, 3] {
            let mut a = DelayLine::default();
            let mut b = DelayLine::default();
            a.allocate(20, ch);
            b.allocate(20, ch);
            let cap = a.max_delay() + 1;
            let mut t = 0.0f32;
            for (k, n) in [1usize, 5, 7, 31, 32, 3, 40, 2, 64].into_iter().enumerate() {
                let frames: Vec<f32> = (0..n * ch)
                    .map(|_| {
                        t += 1.0;
                        t
                    })
                    .collect();
                a.write_frames(&frames);
                for f in frames.chunks_exact(ch) {
                    b.write(f);
                }
                for d in 0..cap {
                    assert_eq!(a.read(d), b.read(d), "ch={ch} k={k} d={d}");
                }
                let m = n.min(cap);
                for delay in 0..=cap - m {
                    let mut out = vec![0.0; m * ch];
                    a.read_frames(delay, &mut out);
                    for (i, f) in out.chunks_exact(ch).enumerate() {
                        assert_eq!(f, b.read(delay + m - 1 - i), "ch={ch} delay={delay}");
                    }
                }
            }
        }
    }

    #[test]
    fn segmented_min_matches_brute_force_and_deque() {
        let mut x = 777u64;
        let mut seg = SegmentedMin::default();
        seg.allocate(70);
        for w in [1usize, 2, 3, 7, 16, 63, 64, 70] {
            seg.reset(w);
            let mut deque = SlidingMin::default();
            deque.allocate(w);
            let mut vals = Vec::new();
            for t in 0..3000u64 {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Runs of equal values and ramps exercise ties and monotone input.
                let v = match (t / 200) % 3 {
                    0 => (x >> 40) as f64,
                    1 => (t % 17) as f64,
                    _ => -(t as f64),
                };
                vals.push(v);
                let got = seg.push(v);
                let lo = vals.len().saturating_sub(w);
                let brute = vals[lo..].iter().cloned().fold(f64::INFINITY, f64::min);
                assert_eq!(got, brute, "w={w} t={t}");
                deque.push(t, v);
                deque.expire((t + 1).saturating_sub(w as u64));
                assert_eq!(Some(got), deque.min());
            }
        }
        // Window larger than the capacity is clamped; unallocated passes through.
        seg.reset(1000);
        assert_eq!(seg.window(), 70);
        assert_eq!(SegmentedMin::default().push(3.0), 3.0);
    }

    #[test]
    fn sliding_min_matches_brute_force() {
        let mut s = SlidingMin::default();
        let w = 7u64;
        s.allocate(w as usize);
        let mut x = 12345u64;
        let mut vals = Vec::new();
        for t in 0..2000u64 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = (x >> 40) as f64;
            vals.push(v);
            s.push(t, v);
            s.expire((t + 1).saturating_sub(w));
            let lo = (t + 1).saturating_sub(w) as usize;
            let brute = vals[lo..].iter().cloned().fold(f64::INFINITY, f64::min);
            assert_eq!(s.min(), Some(brute));
        }
        s.clear();
        assert_eq!(s.min(), None);
    }

    #[test]
    fn sliding_min_ascending_input_stays_within_capacity() {
        let mut s = SlidingMin::default();
        s.allocate(4);
        for t in 0..100u64 {
            s.push(t, t as f64);
            s.expire(t.saturating_sub(3));
            assert!(s.len() <= 4);
            assert_eq!(s.min(), Some(t.saturating_sub(3) as f64));
        }
    }
}
