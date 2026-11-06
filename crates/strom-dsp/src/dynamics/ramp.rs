//! Exact integer moving averages and the two-box ramp smoother used by the
//! limiter (moving-average basics: J. O. Smith III, *Introduction to Digital
//! Filters*; cascading two boxes gives a trapezoidal/triangular kernel).

/// Fractional bits of the fixed-point representation used by [`RampSmoother`].
pub const RAMP_FRAC_BITS: u32 = 44;
const ONE: u64 = 1 << RAMP_FRAC_BITS;
const SCALE: f64 = ONE as f64;

/// Unsigned division by an invariant divisor `d ≥ 1` as a multiply and a
/// shift (T. Granlund and P. L. Montgomery, "Division by invariant integers
/// using multiplication", PLDI 1994).
///
/// With `l = ⌈log2 d⌉`, `s = 63 + l` and `m = ⌈2^s / d⌉` (which fits in a
/// `u64`), write `m·d = 2^s + e` with `0 ≤ e < d ≤ 2^l`. For `n < 2^63`,
/// `n·m / 2^s = n/d + n·e/(d·2^s)`, and the second term is below `1/d`
/// because `n·e < 2^63·2^l = 2^s`. Since the fractional part of `n/d` is at
/// most `(d − 1)/d`, the floor is unchanged: `⌊n·m / 2^s⌋ = ⌊n/d⌋` exactly.
/// Larger `n` fall back to a hardware division.
///
/// On x86-64 a 64-bit `div` has a latency of tens of cycles, and the
/// limiter's ramp chains two of them per frame on its serial path (issue #9).
#[derive(Debug, Clone, Copy, Default)]
struct Divisor {
    d: u64,
    m: u64,
    s: u32,
}

impl Divisor {
    /// Largest dividend handled by the multiply path.
    const MUL_MAX: u64 = (1 << 63) - 1;

    fn new(d: u64) -> Self {
        let d = d.max(1);
        let l = if d == 1 {
            0
        } else {
            64 - (d - 1).leading_zeros()
        };
        let s = 63 + l;
        let m = (1u128 << s).div_ceil(d as u128);
        debug_assert!(m <= u64::MAX as u128);
        Self { d, m: m as u64, s }
    }

    /// `⌊n / d⌋`, exactly.
    #[inline(always)]
    fn div(self, n: u64) -> u64 {
        if n <= Self::MUL_MAX {
            ((n as u128 * self.m as u128) >> self.s) as u64
        } else {
            n / self.d
        }
    }
}

/// Moving average of the last `len` integer inputs, rounded **down**:
/// `y[n] = ⌊(x[n] + … + x[n−len+1]) / len⌋`.
///
/// The running sum is exact, so there is no drift and `y[n]` never exceeds the
/// true mean (and so never exceeds the largest input in the window). The sum of
/// `len` inputs must fit in a `u64`.
#[derive(Debug, Clone, Default)]
pub struct BoxAverage {
    ring: Vec<u64>,
    mask: usize,
    pos: usize,
    len: usize,
    sum: u64,
    /// Division by the invariant `len` as a multiplication (see [`Divisor`]).
    div: Divisor,
}

impl BoxAverage {
    /// Allocate for window lengths up to `max_len` and set the window to 1
    /// (filled with 0). The only allocating method.
    pub fn allocate(&mut self, max_len: usize) {
        let cap = max_len.max(1).next_power_of_two();
        self.ring = vec![0; cap];
        self.mask = cap - 1;
        self.pos = 0;
        self.len = 1;
        self.sum = 0;
        self.div = Divisor::new(1);
    }

    /// Window length in samples.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always false: the window holds at least one sample.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Set the window length (clamped to `1 ..= capacity`) and fill the window
    /// as if `value` had been the input forever. O(len).
    pub fn reset(&mut self, len: usize, value: u64) {
        if self.ring.is_empty() {
            return;
        }
        self.len = len.clamp(1, self.ring.len());
        self.div = Divisor::new(self.len as u64);
        for k in 0..self.len {
            self.ring[self.pos.wrapping_sub(k) & self.mask] = value;
        }
        self.sum = value * self.len as u64;
    }

    /// Push one input and return the floor of the window mean.
    #[inline(always)]
    pub fn step(&mut self, v: u64) -> u64 {
        self.pos = (self.pos + 1) & self.mask;
        let old_idx = self.pos.wrapping_sub(self.len) & self.mask;
        let old = self.ring[old_idx];
        self.ring[self.pos] = v;
        self.sum = self.sum - old + v;
        self.div.div(self.sum)
    }
}

/// Smooth S-shaped ramp: two cascaded [`BoxAverage`]s of lengths `⌈A/2⌉ + 1`
/// and `⌊A/2⌋ + 1` on inputs in `[0, 1]`, quantised down to
/// [`RAMP_FRAC_BITS`] fractional bits.
///
/// The combined kernel spans `A + 1` inputs (the current one and the `A`
/// before it) with non-negative weights summing to 1, so a step in the input
/// becomes a ramp of `A` samples, and — because every rounding is downward —
/// the output is **never larger than any upper bound that holds for all of the
/// last `A + 1` inputs**. The limiter relies on this for its ceiling guarantee.
/// `A = 0` passes the (quantised) input through.
#[derive(Debug, Clone, Default)]
pub struct RampSmoother {
    first: BoxAverage,
    second: BoxAverage,
    length: usize,
}

/// Quantise a value in `[0, 1]` down to fixed point (NaN → 0).
#[inline(always)]
fn quantize(x: f64) -> u64 {
    // `as` truncates toward zero (floor for x ≥ 0), saturates (negative → 0)
    // and maps NaN to 0; `min` caps values above 1.
    ((x * SCALE) as u64).min(ONE)
}

impl RampSmoother {
    /// Allocate for ramp lengths up to `max_length` samples and set length 0.
    pub fn allocate(&mut self, max_length: usize) {
        let m = max_length / 2 + 2;
        self.first.allocate(m);
        self.second.allocate(m);
        self.reset(0, 1.0);
    }

    /// Ramp length `A` in samples.
    pub fn length(&self) -> usize {
        self.length
    }

    /// Set the ramp length `A` and fill the state as if `value` had been the
    /// input forever. After `A + 1` further steps the state depends only on
    /// those inputs. O(A).
    pub fn reset(&mut self, length: usize, value: f64) {
        let q = quantize(value);
        self.first.reset(length.div_ceil(2) + 1, q);
        self.second.reset(length / 2 + 1, q);
        self.length = (self.first.len() - 1) + (self.second.len() - 1);
    }

    /// Push one input (clamped to `[0, 1]`) and return the smoothed output.
    #[inline(always)]
    pub fn step(&mut self, x: f64) -> f64 {
        let a = self.first.step(quantize(x));
        let b = self.second.step(a);
        b as f64 / SCALE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divisor_matches_hardware_division() {
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let ds = (1..=4100u64).chain([
            (1 << 19) - 1,
            1 << 19,
            (1 << 31) + 1,
            u32::MAX as u64,
            (1 << 62) + 3,
            u64::MAX,
        ]);
        for d in ds {
            let v = Divisor::new(d);
            let mut ns = vec![
                0,
                1,
                d - 1,
                d,
                d.wrapping_add(1),
                Divisor::MUL_MAX,
                Divisor::MUL_MAX + 1,
                u64::MAX,
            ];
            // Values just below and at multiples of d, across the range.
            for k in [1u64, 2, 3, (1 << 44) - 1, 1 << 44, Divisor::MUL_MAX / d] {
                let q = k.saturating_mul(d);
                ns.extend([q, q.saturating_sub(1), q.saturating_add(1)]);
            }
            for _ in 0..200 {
                let r = next();
                ns.extend([r, r >> 1, r >> 19, r >> 40]);
            }
            for n in ns {
                assert_eq!(v.div(n), n / d, "n={n} d={d}");
            }
        }
    }

    #[test]
    fn box_average_floor_mean() {
        let mut b = BoxAverage::default();
        b.allocate(8);
        b.reset(3, 0);
        assert_eq!(b.step(3), 1);
        assert_eq!(b.step(4), 2); // 7/3
        assert_eq!(b.step(5), 4);
        assert_eq!(b.step(0), 3);
        b.reset(4, 10);
        assert_eq!(b.step(10), 10);
        assert_eq!(b.step(0), 7); // 30/4
    }

    #[test]
    fn ramp_step_response_is_monotone_s_curve_of_length_a() {
        let mut r = RampSmoother::default();
        r.allocate(64);
        for a in [0usize, 1, 2, 7, 10, 64] {
            r.reset(a, 1.0);
            assert_eq!(r.length(), a);
            let mut prev = 1.0;
            let mut out = Vec::new();
            for _ in 0..a + 5 {
                let y = r.step(0.25);
                assert!(y <= prev && y >= 0.25 - 1e-12);
                prev = y;
                out.push(y);
            }
            // Fully settled after A + 1 inputs, not before (for A > 0).
            assert!((out[a] - 0.25).abs() < 1e-12, "a={a}");
            if a > 0 {
                assert!(out[a - 1] > 0.25 + 1e-6);
            }
            // Symmetric S-shape: halfway point at the middle of the ramp.
            if a >= 2 && a % 2 == 0 {
                let mid = out[a / 2 - 1] + out[a / 2];
                assert!((mid / 2.0 - 0.625).abs() < 0.2);
            }
        }
    }

    #[test]
    fn ramp_never_exceeds_window_bound() {
        let mut r = RampSmoother::default();
        r.allocate(32);
        let a = 13;
        r.reset(a, 1.0);
        let mut x = 99u64;
        let mut hist = vec![1.0f64; a];
        for _ in 0..20_000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            let v = (x >> 11) as f64 / (1u64 << 53) as f64;
            hist.push(v);
            let y = r.step(v);
            let bound = hist[hist.len() - a - 1..]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            // The output is a weighted mean, so it is at most the max and
            // at most any common bound; check the max (the tightest valid bound
            // is per-input, see the limiter tests) and the floor rounding.
            let max = hist[hist.len() - a - 1..]
                .iter()
                .cloned()
                .fold(0.0, f64::max);
            assert!(y <= max);
            assert!(y >= bound - 1e-9);
        }
    }

    #[test]
    fn ramp_output_bounded_by_common_upper_bound() {
        // If the last A+1 inputs are all ≤ c, the output is ≤ c exactly.
        let mut r = RampSmoother::default();
        r.allocate(16);
        r.reset(9, 1.0);
        let c = 0.123_456_789_f64;
        for _ in 0..10 {
            r.step(c);
        }
        for k in 0..30 {
            let y = r.step(if k % 3 == 0 { c } else { c * 0.5 });
            assert!(y <= c);
        }
        assert!(r.step(f64::NAN) >= 0.0);
    }
}
