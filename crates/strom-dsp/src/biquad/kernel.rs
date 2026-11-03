//! Explicit-SIMD Direct Form I kernels (crate-internal; shared by
//! [`super::Biquad`] and [`crate::Equalizer`]).
//!
//! Every entry point takes a `fearless_simd` token `S` (obtained once per
//! `process` call with `dispatch!`). Kernels are generic over `S` and over the
//! vector type, and run their body inside `Simd::vectorize`, so each one is
//! compiled once per SIMD level with that level's target features.
//!
//! # Lanes
//!
//! Channels are lanes. They are processed in groups of the native `f64` width
//! of the level (2 on NEON/SSE/fallback, 4 on AVX2, 8 on AVX-512). The
//! remaining channels form one last group: an `f64x2` if at most two are left,
//! otherwise a native-width vector. Unused lanes of a group are zero-padded,
//! computed and discarded. Every lane executes exactly the same operations, so
//! a channel's output does not depend on the group it lands in or on the
//! vector width; only the FMA policy ([`crate::lanes::uses_fma`]) changes the
//! rounding.
//!
//! # State layout
//!
//! Each state bank holds `STATE_PER_CHANNEL · channels` values,
//! structure-of-arrays: `[x1; channels][x2; channels][y1; channels][y2;
//! channels]`. The layout does not depend on the level, so the level may change
//! between `process` calls.

use fearless_simd::{f64x2, Simd, SimdBase, SimdFloat};

use super::BiquadCoefficients;
use crate::lanes::{blend, flush_denormal, fma_mode, load, load_f32, madd, store, store_f32, Fma};

/// Maximum number of biquads run in series by one kernel call.
pub(crate) const MAX_STAGES: usize = 8;

/// Number of `f64` values of DF-I state per channel.
pub(crate) const STATE_PER_CHANNEL: usize = 4;

/// Frames between denormal flushes of the filter state in the steady kernels.
pub(crate) const FLUSH_INTERVAL: usize = 32;

/// Coefficients of one stage, broadcast to every lane, with the feedback
/// coefficients negated (`na1 = −a1`, `na2 = −a2`; exact).
#[derive(Clone, Copy)]
struct LaneCoefs<V> {
    b0: V,
    b1: V,
    b2: V,
    na1: V,
    na2: V,
}

impl<V> LaneCoefs<V> {
    #[inline(always)]
    fn splat<S: Simd>(simd: S, c: &BiquadCoefficients) -> Self
    where
        V: SimdFloat<S, Element = f64>,
    {
        Self {
            b0: V::splat(simd, c.b0),
            b1: V::splat(simd, c.b1),
            b2: V::splat(simd, c.b2),
            na1: V::splat(simd, -c.a1),
            na2: V::splat(simd, -c.a2),
        }
    }

    /// Add a per-sample increment (built with [`LaneCoefs::splat`] from a
    /// [`BiquadCoefficients::ramp_step`]). Lane-wise this is exactly
    /// [`BiquadCoefficients::add_step`]: `(−a) + (−d) = −(a + d)` in IEEE
    /// arithmetic.
    #[inline(always)]
    fn add_step<S: Simd>(&mut self, d: &Self)
    where
        V: SimdFloat<S, Element = f64>,
    {
        self.b0 += d.b0;
        self.b1 += d.b1;
        self.b2 += d.b2;
        self.na1 += d.na1;
        self.na2 += d.na2;
    }
}

/// DF-I state of one lane group.
#[derive(Clone, Copy)]
struct LaneState<V> {
    x1: V,
    x2: V,
    y1: V,
    y2: V,
}

impl<V> LaneState<V> {
    #[inline(always)]
    fn zero<S: Simd>(simd: S) -> Self
    where
        V: SimdFloat<S, Element = f64>,
    {
        let z = V::splat(simd, 0.0);
        Self {
            x1: z,
            x2: z,
            y1: z,
            y2: z,
        }
    }

    /// Load lanes `c0 .. c0 + valid` of a bank (see the module documentation).
    #[inline(always)]
    fn load<S: Simd>(simd: S, bank: &[f64], channels: usize, c0: usize, valid: usize) -> Self
    where
        V: SimdFloat<S, Element = f64>,
    {
        let part = |k: usize| &bank[k * channels + c0..k * channels + c0 + valid];
        Self {
            x1: load(simd, part(0)),
            x2: load(simd, part(1)),
            y1: load(simd, part(2)),
            y2: load(simd, part(3)),
        }
    }

    #[inline(always)]
    fn store<S: Simd>(&self, bank: &mut [f64], channels: usize, c0: usize, valid: usize)
    where
        V: SimdFloat<S, Element = f64>,
    {
        let r = |k: usize| k * channels + c0..k * channels + c0 + valid;
        store(self.x1, &mut bank[r(0)]);
        store(self.x2, &mut bank[r(1)]);
        store(self.y1, &mut bank[r(2)]);
        store(self.y2, &mut bank[r(3)]);
    }

    /// One DF-I step, `y = b0·x + b1·x1 + b2·x2 − a1·y1 − a2·y2`, evaluated
    /// as the chain
    ///
    /// ```text
    /// s = x2·b2;  s = x1·b1 + s;  s = y2·(−a2) + s;  t = x·b0 + s;  y = y1·(−a1) + t
    /// ```
    ///
    /// with every `·+` a [`madd`] (fused or not, per level). The terms that
    /// only depend on the past are summed first, so the path from the stage
    /// input `x` to `y` is two multiply-adds and the loop-carried path from
    /// `y1` is one.
    ///
    /// No denormal flush here (it would lengthen that dependency); callers
    /// flush with [`LaneState::flush`] at least every [`FLUSH_INTERVAL`]
    /// frames.
    #[inline(always)]
    fn tick<S: Simd>(&mut self, fma: Fma, c: &LaneCoefs<V>, x: V) -> V
    where
        V: SimdFloat<S, Element = f64>,
    {
        let s = self.x2 * c.b2;
        let s = madd(fma, self.x1, c.b1, s);
        let s = madd(fma, self.y2, c.na2, s);
        let t = madd(fma, x, c.b0, s);
        let y = madd(fma, self.y1, c.na1, t);
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    /// Flush every state value below [`crate::util::DENORMAL_FLUSH`] to zero.
    #[inline(always)]
    fn flush<S: Simd>(&mut self, simd: S)
    where
        V: SimdFloat<S, Element = f64>,
    {
        self.x1 = flush_denormal(simd, self.x1);
        self.x2 = flush_denormal(simd, self.x2);
        self.y1 = flush_denormal(simd, self.y1);
        self.y2 = flush_denormal(simd, self.y2);
    }
}

/// One lane group: first channel, number of real channels, and whether it uses
/// the native vector width (`true`) or `f64x2` (`false`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Group {
    pub c0: usize,
    pub valid: usize,
    pub native: bool,
}

/// Split `channels` into lane groups for a level whose native `f64` width is
/// `native` (see the module documentation).
pub(crate) fn channel_groups(channels: usize, native: usize) -> impl Iterator<Item = Group> {
    let full = channels / native;
    let rem = channels % native;
    (0..full)
        .map(move |g| Group {
            c0: g * native,
            valid: native,
            native: true,
        })
        .chain((rem > 0).then_some(Group {
            c0: full * native,
            valid: rem,
            native: rem > 2,
        }))
}

/// One steady (fixed-coefficient) biquad in a series chain.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SteadyStage {
    pub coefs: BiquadCoefficients,
    /// Start of this stage's state bank (`STATE_PER_CHANNEL · channels` values).
    pub state: usize,
}

/// Number of lanes a group actually loads and stores, as a compile-time
/// constant wherever possible: `V::LEN` for full groups, 1 for a padded
/// `f64x2` (the only padded `f64x2` holds a single channel), else `valid`.
#[inline(always)]
fn group_width<S: Simd, V: SimdFloat<S>, const FULL: bool>(valid: usize) -> usize {
    if FULL {
        V::LEN
    } else if V::LEN == 2 {
        1
    } else {
        valid
    }
}

/// Run `stages` in series over interleaved `data` (whole frames of `channels`),
/// sample by sample. `states` holds every bank referenced by the stages.
///
/// `ramps` is either empty (fixed coefficients: the steady-state loop) or has
/// one entry per stage; a stage with `Some(delta)` adds `delta` to its
/// coefficients before each sample (per-sample linear interpolation), the
/// others keep theirs.
pub(crate) fn run_chain<S: Simd>(
    simd: S,
    data: &mut [f32],
    channels: usize,
    stages: &[SteadyStage],
    ramps: &[Option<BiquadCoefficients>],
    states: &mut [f64],
) {
    debug_assert!(stages.len() <= MAX_STAGES);
    debug_assert!(ramps.is_empty() || ramps.len() == stages.len());
    if stages.is_empty() {
        return;
    }
    let ramp = !ramps.is_empty();
    for g in channel_groups(channels, S::f64s::LEN) {
        let args = ChainArgs {
            data: &mut *data,
            channels,
            group: g,
            stages,
            ramps,
            states: &mut *states,
        };
        let full = g.valid == if g.native { S::f64s::LEN } else { 2 };
        match (g.native, full, ramp) {
            (true, true, false) => chain_group::<S, S::f64s, true, false>(simd, args),
            (true, true, true) => chain_group::<S, S::f64s, true, true>(simd, args),
            (true, false, false) => chain_group::<S, S::f64s, false, false>(simd, args),
            (true, false, true) => chain_group::<S, S::f64s, false, true>(simd, args),
            (false, true, false) => chain_group::<S, f64x2<S>, true, false>(simd, args),
            (false, true, true) => chain_group::<S, f64x2<S>, true, true>(simd, args),
            (false, false, false) => chain_group::<S, f64x2<S>, false, false>(simd, args),
            (false, false, true) => chain_group::<S, f64x2<S>, false, true>(simd, args),
        }
    }
}

struct ChainArgs<'a> {
    data: &'a mut [f32],
    channels: usize,
    group: Group,
    stages: &'a [SteadyStage],
    ramps: &'a [Option<BiquadCoefficients>],
    states: &'a mut [f64],
}

fn chain_group<S: Simd, V: SimdFloat<S, Element = f64>, const FULL: bool, const RAMP: bool>(
    simd: S,
    args: ChainArgs<'_>,
) {
    // A fixed stage count lets the compiler unroll the chain and keep every
    // state in registers.
    match args.stages.len() {
        1 => chain_n::<S, V, 1, FULL, RAMP>(simd, args),
        2 => chain_n::<S, V, 2, FULL, RAMP>(simd, args),
        3 => chain_n::<S, V, 3, FULL, RAMP>(simd, args),
        4 => chain_n::<S, V, 4, FULL, RAMP>(simd, args),
        5 => chain_n::<S, V, 5, FULL, RAMP>(simd, args),
        6 => chain_n::<S, V, 6, FULL, RAMP>(simd, args),
        7 => chain_n::<S, V, 7, FULL, RAMP>(simd, args),
        _ => chain_n::<S, V, 8, FULL, RAMP>(simd, args),
    }
}

/// One frame through `N` stages.
#[inline(always)]
fn chain_tick<S: Simd, V: SimdFloat<S, Element = f64>, const N: usize, const RAMP: bool>(
    fma: Fma,
    coefs: &mut [LaneCoefs<V>; N],
    deltas: &[Option<LaneCoefs<V>>; N],
    st: &mut [LaneState<V>; N],
    mut x: V,
) -> V {
    for ((c, d), s) in coefs.iter_mut().zip(deltas).zip(st.iter_mut()) {
        if RAMP {
            if let Some(d) = d {
                c.add_step::<S>(d);
            }
        }
        x = s.tick::<S>(fma, c, x);
    }
    x
}

fn chain_n<
    S: Simd,
    V: SimdFloat<S, Element = f64>,
    const N: usize,
    const FULL: bool,
    const RAMP: bool,
>(
    simd: S,
    args: ChainArgs<'_>,
) {
    simd.vectorize(
        #[inline(always)]
        || {
            let ChainArgs {
                data,
                channels,
                group: Group { c0, valid, .. },
                stages,
                ramps,
                states,
            } = args;
            let w = group_width::<S, V, FULL>(valid);
            let fma = fma_mode(simd.level());
            let mut coefs: [LaneCoefs<V>; N] =
                core::array::from_fn(|i| LaneCoefs::splat(simd, &stages[i].coefs));
            let deltas: [Option<LaneCoefs<V>>; N] = core::array::from_fn(|i| {
                if RAMP {
                    ramps[i].map(|d| LaneCoefs::splat(simd, &d))
                } else {
                    None
                }
            });
            let mut st: [LaneState<V>; N] = core::array::from_fn(|i| {
                LaneState::load(simd, &states[stages[i].state..], channels, c0, w)
            });
            for block in data.chunks_mut(FLUSH_INTERVAL * channels) {
                if channels == w {
                    // One group covers the frame (mono, stereo, …): no offset.
                    for io in block.chunks_exact_mut(w) {
                        let x = load_f32(simd, io);
                        let y = chain_tick::<S, V, N, RAMP>(fma, &mut coefs, &deltas, &mut st, x);
                        store_f32(y, io);
                    }
                } else {
                    for frame in block.chunks_exact_mut(channels) {
                        let io = &mut frame[c0..c0 + w];
                        let x = load_f32(simd, io);
                        let y = chain_tick::<S, V, N, RAMP>(fma, &mut coefs, &deltas, &mut st, x);
                        store_f32(y, io);
                    }
                }
                flush_all(simd, &mut st);
            }
            for (s, stage) in st.iter().zip(stages) {
                s.store::<S>(&mut states[stage.state..], channels, c0, w);
            }
        },
    )
}

#[inline(always)]
fn flush_all<S: Simd, V: SimdFloat<S, Element = f64>, const N: usize>(
    simd: S,
    st: &mut [LaneState<V>; N],
) {
    st.iter_mut().for_each(|s| s.flush(simd));
}

/// The realisation being faded out during a crossfade.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FadeOut {
    /// `None`: the old realisation is the dry input.
    pub old: Option<SteadyStage>,
    /// Crossfade frames already done before the first frame of this call.
    pub done: usize,
    /// `1 / crossfade length`.
    pub inv_len: f64,
}

/// One stage of a chain whose coefficients and/or realisation are changing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ChangingStage {
    /// `None`: the new realisation is the dry input (fading out to "off").
    pub new: Option<SteadyStage>,
    /// Per-sample coefficient increment of `new`, added before each sample.
    pub delta: Option<BiquadCoefficients>,
    /// Crossfade from an old realisation, if one is running.
    pub fade: Option<FadeOut>,
}

/// Like [`run_chain`], with crossfades between two realisations per stage
/// (and per-sample coefficient interpolation). Used only while a crossfade
/// is running.
pub(crate) fn run_changing<S: Simd>(
    simd: S,
    data: &mut [f32],
    channels: usize,
    stages: &[ChangingStage],
    states: &mut [f64],
) {
    debug_assert!(stages.len() <= MAX_STAGES);
    if stages.is_empty() {
        return;
    }
    for g in channel_groups(channels, S::f64s::LEN) {
        let args = (&mut *data, channels, g, stages, &mut *states);
        match (g.native, g.valid == if g.native { S::f64s::LEN } else { 2 }) {
            (true, true) => changing_group::<S, S::f64s, true>(simd, args),
            (true, false) => changing_group::<S, S::f64s, false>(simd, args),
            (false, true) => changing_group::<S, f64x2<S>, true>(simd, args),
            (false, false) => changing_group::<S, f64x2<S>, false>(simd, args),
        }
    }
}

type ChangingArgs<'a> = (
    &'a mut [f32],
    usize,
    Group,
    &'a [ChangingStage],
    &'a mut [f64],
);

fn changing_group<S: Simd, V: SimdFloat<S, Element = f64>, const FULL: bool>(
    simd: S,
    (data, channels, group, stages, states): ChangingArgs<'_>,
) {
    simd.vectorize(
        #[inline(always)]
        || {
            let c0 = group.c0;
            let valid = group_width::<S, V, FULL>(group.valid);
            let fma = fma_mode(simd.level());
            let load_bank = |s: Option<SteadyStage>, states: &[f64]| match s {
                Some(s) => LaneState::<V>::load(simd, &states[s.state..], channels, c0, valid),
                None => LaneState::zero(simd),
            };
            let zero_coefs = LaneCoefs::splat(simd, &BiquadCoefficients::default());
            let mut coefs = [zero_coefs; MAX_STAGES];
            let mut deltas = [None; MAX_STAGES];
            let mut old_coefs = [zero_coefs; MAX_STAGES];
            let mut cur = [LaneState::<V>::zero(simd); MAX_STAGES];
            let mut old = [LaneState::<V>::zero(simd); MAX_STAGES];
            for (k, s) in stages.iter().enumerate() {
                if let Some(n) = s.new {
                    coefs[k] = LaneCoefs::splat(simd, &n.coefs);
                }
                deltas[k] = s.delta.map(|d| LaneCoefs::splat(simd, &d));
                if let Some(o) = s.fade.and_then(|f| f.old) {
                    old_coefs[k] = LaneCoefs::splat(simd, &o.coefs);
                }
                cur[k] = load_bank(s.new, states);
                old[k] = load_bank(s.fade.and_then(|f| f.old), states);
            }
            for (i, frame) in data.chunks_exact_mut(channels).enumerate() {
                let io = &mut frame[c0..c0 + valid];
                let mut x: V = load_f32(simd, io);
                for (k, s) in stages.iter().enumerate() {
                    if let Some(d) = &deltas[k] {
                        coefs[k].add_step::<S>(d);
                    }
                    let y_new = match s.new {
                        Some(_) => {
                            let y = cur[k].tick::<S>(fma, &coefs[k], x);
                            cur[k].flush(simd);
                            y
                        }
                        None => x,
                    };
                    x = match &s.fade {
                        None => y_new,
                        Some(f) => {
                            let y_old = match &f.old {
                                Some(_) => {
                                    let y = old[k].tick::<S>(fma, &old_coefs[k], x);
                                    old[k].flush(simd);
                                    y
                                }
                                None => x,
                            };
                            let g = ((f.done + i + 1) as f64 * f.inv_len).min(1.0);
                            blend(simd, fma, y_old, y_new, g)
                        }
                    };
                }
                store_f32(x, io);
            }
            for (k, s) in stages.iter().enumerate() {
                if let Some(st) = s.new {
                    cur[k].store::<S>(&mut states[st.state..], channels, c0, valid);
                }
                if let Some(o) = s.fade.and_then(|f| f.old) {
                    old[k].store::<S>(&mut states[o.state..], channels, c0, valid);
                }
            }
        },
    )
}

/// Zero the whole state region if any value in it is non-finite (a NaN/Inf
/// input poisoned it).
pub(crate) fn sanitize_bank(states: &mut [f64]) {
    if states.iter().any(|v| !v.is_finite()) {
        states.iter_mut().for_each(|v| *v = 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_groups_cover_every_channel_once() {
        for native in [2usize, 4, 8] {
            for ch in 1..=70 {
                let mut seen = vec![0; ch];
                for g in channel_groups(ch, native) {
                    assert!(g.valid >= 1 && g.valid <= native);
                    if !g.native {
                        assert!(g.valid <= 2);
                    }
                    seen[g.c0..g.c0 + g.valid].iter_mut().for_each(|s| *s += 1);
                }
                assert!(seen.iter().all(|&s| s == 1), "{native} {ch}");
            }
        }
    }
}
