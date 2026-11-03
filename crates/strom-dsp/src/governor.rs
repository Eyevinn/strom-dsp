//! Clip budget governor: a slow outer loop of the broadcast chain that keeps
//! the work of the final clipper and the true-peak limiter within one
//! operator-facing number, the **clip budget** (dB), and trades loudness
//! for cleanliness when both cannot be had.
//!
//! # Measurement
//!
//! Every 10 ms output sub-block the chain hands over the final clipper's
//! [`ClipStats`] of that sub-block, the limiter's mean gain reduction, and
//! whether the sub-block is *active* (its 400 ms output loudness above
//! `agc-gate`, the same gating as the output-referenced trim). Over the
//! newest 3 s of active sub-blocks the governor forms:
//!
//! - the **clip depth**: removed energy / clipper input energy, in dB;
//! - the limiter's **mean gain reduction**, in dB;
//! - the **touched share**: high-rate samples the clipper's curve acted on.
//!
//! # Budget mapping
//!
//! Each measure is converted to the budget (in dB, 0 … 6) that would allow
//! it exactly ("budget used"); the largest of the three is compared with the
//! clip budget `B`:
//!
//! | Measure | Allowed at budget `B` | Budget used by a reading |
//! |---|---|---|
//! | clip depth | `−40 + 5·B` dB (−40 at 0, −30 at 2, −10 at 6) | `(depth + 40)/5` |
//! | limiter mean gain reduction | `B/2` dB (0 at 0, 1 at 2, 3 at 6) | `2·gr` |
//! | touched share | `0.2 % · 2^(B/2)` (0.4 % at 2, 1.6 % at 6) | `2·log2(share / 0.2 %)` |
//!
//! The excess `e = used − B` (dB of budget) drives the loop.
//!
//! # Control law (every active, unfrozen 10 ms sub-block)
//!
//! - **Over budget** (`e > 0`): the output-referenced trim may not add gain
//!   (it may still take gain away). Any density is given back first (its
//!   makeup raised the peaks too), then a **cut** lowers the AGC target by
//!   `e·dt/8 s`, at most `min(0.5 dB/s, ½·slower AGC rate)`, up to 12 dB.
//!   The loudness deficit this causes is accepted and reported instead of
//!   distortion.
//! - **While a cut is applied** the trim stays held; the cut is released
//!   (at half its rate) only well under budget (`e < −0.4`, a hysteresis
//!   band), so it does not chatter around the budget.
//! - **Budget left but the target out of reach** (no cut, `e < −0.2`, and
//!   the gated output more than 0.5 LU under target, i.e. the trim and the
//!   AGC are at a bound): **density** rises: every band's threshold is
//!   lowered by `density` dB, with makeup `density·(1 − 1/ratio)`, so
//!   passages well above the thresholds keep their level and everything
//!   below them gets louder: more loudness per peak. It moves
//!   `(deficit − 0.5)·dt/10 s`, at most 0.2 dB/s, up to 6 dB, and is
//!   released (at half its rate) once the output reaches the target without
//!   it, well under budget.
//! - **Frozen** (nothing moves) in pauses and below `agc-gate` (the trim's
//!   rule: the newest 100 ms below the gate, or less than 400 ms of active
//!   output in the window), and while the AGC, the clipper or the chain is
//!   bypassed or fading.
//!
//! **Why density is not the first resort when over budget.** The order
//! "hold the trim, raise density, then cut" was tried first. On percussive
//! programme the peaks that reach the clipper are transients faster than
//! the band compressors' attack: density's makeup raised them as much as
//! the rest, so the cut had to take it back, and the clip depth did not
//! move (streaming preset, sport-like programme, budget 1 dB: clip depth
//! −30.8 dB with 6 dB of density and a growing cut, the same as with the
//! governor off). Density now only buys loudness where budget is left.
//!
//! The **loudness deficit** reported is `target − gated output loudness`
//! (3 s, ≥ 0) while the governor holds the trim or applies density;
//! otherwise 0.
//!
//! # Stability
//!
//! The cut acts on the AGC target directly: like the trim, an integrator
//! (τ = 8 s) behind the 3 s measurement window (group delay 1.5 s), so the
//! trim's margins apply (phase margin ≥ 79°). Density acts only while the
//! trim is at a bound (not moving): again a single integrator (τ = 10 s)
//! on the loudness error. Hysteresis (cut released below −0.4, density
//! raised below −0.2, both in budget dB) and the rate limits stop small
//! limit cycles; the 10-minute step test in `tests/governor.rs` checks for
//! them.

use crate::clipper::ClipStats;
use crate::meter::{History, WindowSum, SUB_BLOCK_MS};

/// Range of the clip budget (dB).
pub const BUDGET_RANGE_DB: (f32, f32) = (0.0, 6.0);
/// Measurement window (ms), the same 3 s as the trim.
const WINDOW_MS: f64 = 3000.0;
/// Hysteresis band (budget dB) below the budget: the cut and the density
/// are released only below it.
const RELEASE_BAND: f64 = 0.4;
/// Density rises only with at least this much budget left (budget dB).
const DENSITY_HEADROOM: f64 = 0.2;
/// Loudness deficit (LU) above which density rises near the budget.
const DEFICIT_TOL_LU: f64 = 0.5;
/// Density: range, integral time constant, rate limit.
pub const DENSITY_MAX_DB: f64 = 6.0;
const DENSITY_TAU_S: f64 = 10.0;
const DENSITY_RATE_DB_S: f64 = 0.2;
/// Cut: range, integral time constant, rate limit.
pub const CUT_MAX_DB: f64 = 12.0;
const CUT_TAU_S: f64 = 8.0;
const CUT_RATE_DB_S: f64 = 0.5;
/// Budget mapping (see the module docs).
const DEPTH_AT_ZERO_DB: f64 = -40.0;
const DEPTH_PER_BUDGET_DB: f64 = 5.0;
const GR_AT_ZERO_DB: f64 = 0.0;
const GR_PER_BUDGET_DB: f64 = 0.5;
const TOUCH_AT_ZERO: f64 = 0.002;

/// What the governor is doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GovernorState {
    /// Switched off: no density, no cut, the trim is free.
    #[default]
    Off,
    /// Holding everything (pause, below the gate, a stage bypassed).
    Frozen,
    /// Well under budget, nothing applied: the trim is free.
    Free,
    /// Near or over budget: the trim may not add gain; nothing else moves.
    Hold,
    /// Raising density.
    Density,
    /// Cutting the AGC target (a loudness deficit is accepted).
    Cut,
    /// Well under budget: releasing the cut or the density.
    Release,
}

impl GovernorState {
    /// Short name, used in the element's meter message.
    pub fn nick(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Frozen => "frozen",
            Self::Free => "free",
            Self::Hold => "hold",
            Self::Density => "density",
            Self::Cut => "cut",
            Self::Release => "release",
        }
    }
}

/// Clip depth (dB) a budget allows.
pub fn depth_target_db(budget_db: f64) -> f64 {
    DEPTH_AT_ZERO_DB + DEPTH_PER_BUDGET_DB * budget_db
}

/// Limiter mean gain reduction (dB) a budget allows.
pub fn limiter_target_db(budget_db: f64) -> f64 {
    GR_AT_ZERO_DB + GR_PER_BUDGET_DB * budget_db
}

/// Touched share (0 … 1) a budget allows.
pub fn touched_target(budget_db: f64) -> f64 {
    TOUCH_AT_ZERO * (budget_db / 2.0).exp2()
}

/// Everything the governor needs for one output sub-block.
#[derive(Debug, Clone, Copy, Default)]
pub struct GovernorInputs {
    /// The sub-block counts (output 400 ms loudness at or above the gate).
    pub active: bool,
    /// Hold every value (see the module docs).
    pub frozen: bool,
    /// The clipper's work over the sub-block.
    pub clip: ClipStats,
    /// Mean limiter gain reduction over the sub-block (dB, ≥ 0).
    pub limiter_gr_db: f64,
    /// Target loudness (LUFS).
    pub target_lufs: f64,
    /// Gated output loudness over the trim's 3 s window (LUFS), if known.
    pub output_lufs: Option<f64>,
    /// Density reaches the output (multiband in use).
    pub density_available: bool,
    /// Slower of the AGC's rates (dB/s); the cut moves at most half of it.
    pub slower_agc_rate_db_s: f64,
}

/// The clip budget governor. See the module documentation.
#[derive(Debug, Default)]
pub struct ClipBudgetGovernor {
    enabled: bool,
    budget_db: f64,
    window: usize,
    removed: History,
    input: History,
    touched: History,
    samples: History,
    gr: History,
    active: History,
    removed_sum: WindowSum,
    input_sum: WindowSum,
    touched_sum: WindowSum,
    samples_sum: WindowSum,
    gr_sum: WindowSum,
    active_sum: WindowSum,
    density_db: f64,
    cut_db: f64,
    state: GovernorState,
    hold_trim: bool,
    deficit_lu: f64,
    used_db: f64,
}

impl ClipBudgetGovernor {
    /// Allocate the 3 s window. The only allocating method.
    pub fn allocate(&mut self) {
        self.window = (WINDOW_MS / SUB_BLOCK_MS).round() as usize;
        for h in [
            &mut self.removed,
            &mut self.input,
            &mut self.touched,
            &mut self.samples,
            &mut self.gr,
            &mut self.active,
        ] {
            h.allocate(self.window + 1);
        }
        self.reset();
    }

    /// Clear the window and every output (density, cut, state).
    pub fn reset(&mut self) {
        for h in [
            &mut self.removed,
            &mut self.input,
            &mut self.touched,
            &mut self.samples,
            &mut self.gr,
            &mut self.active,
        ] {
            h.clear();
        }
        self.removed_sum.set_len(self.window, &self.removed);
        self.input_sum.set_len(self.window, &self.input);
        self.touched_sum.set_len(self.window, &self.touched);
        self.samples_sum.set_len(self.window, &self.samples);
        self.gr_sum.set_len(self.window, &self.gr);
        self.active_sum.set_len(self.window, &self.active);
        self.density_db = 0.0;
        self.cut_db = 0.0;
        self.hold_trim = false;
        self.deficit_lu = 0.0;
        self.used_db = f64::NEG_INFINITY;
        self.state = if self.enabled {
            GovernorState::Free
        } else {
            GovernorState::Off
        };
    }

    /// Switch on or off, and set the clip budget (dB, 0 … 6). Switching off
    /// drops density and cut at once (the multiband thresholds and the AGC
    /// target glide on their own).
    pub fn set_params(&mut self, enabled: bool, budget_db: f32) {
        let b = if budget_db.is_finite() {
            budget_db.clamp(BUDGET_RANGE_DB.0, BUDGET_RANGE_DB.1)
        } else {
            2.0
        };
        self.budget_db = b as f64;
        if enabled != self.enabled {
            self.enabled = enabled;
            self.density_db = 0.0;
            self.cut_db = 0.0;
            self.hold_trim = false;
            self.deficit_lu = 0.0;
            self.state = if enabled {
                GovernorState::Free
            } else {
                GovernorState::Off
            };
        }
    }

    /// Record one output sub-block and update the outputs.
    pub fn on_sub_block(&mut self, inp: &GovernorInputs) {
        let a = inp.active;
        let c = &inp.clip;
        let pick = |v: f64| if a && v.is_finite() { v } else { 0.0 };
        self.removed.push(pick(c.removed_energy));
        self.input.push(pick(c.input_energy));
        self.touched.push(pick(c.touched as f64));
        self.samples.push(pick(c.samples as f64));
        self.gr.push(pick(inp.limiter_gr_db.max(0.0)));
        self.active.push(if a { 1.0 } else { 0.0 });
        self.removed_sum.pushed(&self.removed);
        self.input_sum.pushed(&self.input);
        self.touched_sum.pushed(&self.touched);
        self.samples_sum.pushed(&self.samples);
        self.gr_sum.pushed(&self.gr);
        self.active_sum.pushed(&self.active);

        if !self.enabled {
            self.state = GovernorState::Off;
            return;
        }
        self.used_db = self.budget_used();
        if inp.frozen {
            self.state = GovernorState::Frozen;
            return;
        }
        let e = self.used_db - self.budget_db;
        let deficit = inp
            .output_lufs
            .map_or(0.0, |o| (inp.target_lufs - o).max(0.0));
        let dt = SUB_BLOCK_MS * 0.001;
        let cut_rate = CUT_RATE_DB_S.min(0.5 * inp.slower_agc_rate_db_s);
        let density_room = inp.density_available && self.density_db < DENSITY_MAX_DB;
        let cut_step = |drive: f64| (drive * dt / CUT_TAU_S).clamp(0.0, cut_rate * dt);
        let density_step =
            |drive: f64| (drive * dt / DENSITY_TAU_S).clamp(0.0, DENSITY_RATE_DB_S * dt);
        self.hold_trim = e > 0.0 || self.cut_db > 0.0;
        self.state = if e > 0.0 {
            // Over budget: give back density (its makeup raised the peaks
            // too), then cut.
            if self.density_db > 0.0 {
                self.density_db = (self.density_db - density_step(e)).max(0.0);
                GovernorState::Release
            } else {
                self.cut_db = (self.cut_db + cut_step(e)).min(CUT_MAX_DB);
                GovernorState::Cut
            }
        } else if e < -RELEASE_BAND && self.cut_db > 0.0 {
            // Well under budget: release the cut first.
            let under = -(e + RELEASE_BAND);
            self.cut_db = (self.cut_db - 0.5 * cut_step(under)).max(0.0);
            GovernorState::Release
        } else if self.cut_db > 0.0 {
            GovernorState::Hold
        } else if e < -DENSITY_HEADROOM && deficit > DEFICIT_TOL_LU && density_room {
            // Budget left, but the trim and the AGC cannot reach the target
            // (at a bound): get loudness per peak from density.
            self.density_db =
                (self.density_db + density_step(deficit - DEFICIT_TOL_LU)).min(DENSITY_MAX_DB);
            GovernorState::Density
        } else if e < -RELEASE_BAND && self.density_db > 0.0 && deficit <= 0.0 {
            // On target without it: release the density slowly.
            let under = -(e + RELEASE_BAND);
            self.density_db = (self.density_db - 0.5 * density_step(under)).max(0.0);
            GovernorState::Release
        } else if e > -RELEASE_BAND || self.density_db > 0.0 {
            GovernorState::Hold
        } else {
            GovernorState::Free
        };
        self.deficit_lu = if self.hold_trim || self.density_db > 0.0 {
            deficit
        } else {
            0.0
        };
    }

    /// Budget used (dB of budget) by the measures over the window; −∞ when
    /// nothing was measured.
    fn budget_used(&self) -> f64 {
        let mut used = f64::NEG_INFINITY;
        let input = self.input_sum.sum();
        let removed = self.removed_sum.sum();
        if input > 0.0 && removed > 0.0 {
            let depth = 10.0 * (removed / input).log10();
            used = used.max((depth - DEPTH_AT_ZERO_DB) / DEPTH_PER_BUDGET_DB);
        }
        let n = self.active_sum.sum().round();
        if n >= 1.0 {
            let gr = self.gr_sum.sum() / n;
            if gr > 0.0 {
                used = used.max((gr - GR_AT_ZERO_DB) / GR_PER_BUDGET_DB);
            }
        }
        let samples = self.samples_sum.sum();
        let touched = self.touched_sum.sum();
        if samples > 0.0 && touched > 0.0 {
            used = used.max(2.0 * (touched / samples / TOUCH_AT_ZERO).log2());
        }
        used
    }

    /// Density (dB): how far every band threshold is lowered.
    pub fn density_db(&self) -> f64 {
        self.density_db
    }

    /// Cut of the AGC target (dB, ≥ 0).
    pub fn cut_db(&self) -> f64 {
        self.cut_db
    }

    /// The output-referenced trim may not add gain.
    pub fn holds_trim(&self) -> bool {
        self.enabled && self.hold_trim
    }

    /// What the governor is doing.
    pub fn state(&self) -> GovernorState {
        self.state
    }

    /// Loudness deficit (LU, ≥ 0) the governor accepts; 0 while the trim is
    /// free.
    pub fn deficit_lu(&self) -> f64 {
        if self.enabled {
            self.deficit_lu
        } else {
            0.0
        }
    }

    /// Budget used (dB of budget) over the window, as last computed; −∞
    /// before anything was clipped or limited.
    pub fn budget_used_db(&self) -> f64 {
        self.used_db
    }

    /// Clip depth over the window (dB; −200 when nothing was removed).
    pub fn window_depth_db(&self) -> f64 {
        let input = self.input_sum.sum();
        let removed = self.removed_sum.sum();
        if input > 0.0 && removed > 0.0 {
            10.0 * (removed / input).log10()
        } else {
            -200.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(depth_db: f64, share: f64) -> ClipStats {
        let input = 1000.0;
        ClipStats {
            input_energy: input,
            removed_energy: input * 10f64.powf(depth_db / 10.0),
            touched: (share * 1e6) as u64,
            samples: 1_000_000,
        }
    }

    fn inputs(depth_db: f64) -> GovernorInputs {
        GovernorInputs {
            active: true,
            frozen: false,
            clip: stats(depth_db, 0.0),
            limiter_gr_db: 0.0,
            target_lufs: -16.0,
            output_lufs: Some(-16.0),
            density_available: true,
            slower_agc_rate_db_s: 3.0,
        }
    }

    #[test]
    fn mapping_is_consistent() {
        for b in [0.0, 2.0, 6.0] {
            let mut g = ClipBudgetGovernor::default();
            g.allocate();
            g.set_params(true, b as f32);
            let i = inputs(depth_target_db(b));
            for _ in 0..400 {
                g.on_sub_block(&i);
            }
            assert!((g.budget_used_db() - b).abs() < 1e-6);
        }
        assert!((limiter_target_db(6.0) - 3.0).abs() < 1e-12);
        assert!((touched_target(2.0) - 0.004).abs() < 1e-12);
    }

    #[test]
    fn over_budget_holds_trim_and_cuts() {
        let mut g = ClipBudgetGovernor::default();
        g.allocate();
        g.set_params(true, 2.0);
        let i = inputs(-20.0);
        for _ in 0..300 {
            g.on_sub_block(&i);
        }
        assert_eq!(g.state(), GovernorState::Cut);
        assert!(g.holds_trim() && g.cut_db() > 0.0 && g.density_db() == 0.0);
        for _ in 0..30_000 {
            g.on_sub_block(&i);
        }
        assert_eq!(g.cut_db(), CUT_MAX_DB);
    }

    #[test]
    fn deficit_with_budget_left_raises_density() {
        let mut g = ClipBudgetGovernor::default();
        g.allocate();
        g.set_params(true, 2.0);
        let mut i = inputs(-40.0);
        i.output_lufs = Some(-19.0);
        for _ in 0..3000 {
            g.on_sub_block(&i);
        }
        assert_eq!(g.state(), GovernorState::Density);
        assert!(g.density_db() > 0.0 && !g.holds_trim());
        assert!((g.deficit_lu() - 3.0).abs() < 1e-9);
        // Over budget: the density goes back first.
        let d = g.density_db();
        for _ in 0..400 {
            g.on_sub_block(&inputs(-20.0));
        }
        assert!(g.density_db() < d && g.cut_db() == 0.0);
    }

    #[test]
    fn frozen_and_off_hold_everything() {
        let mut g = ClipBudgetGovernor::default();
        g.allocate();
        g.set_params(true, 2.0);
        for _ in 0..2000 {
            g.on_sub_block(&inputs(-20.0));
        }
        let (d, c) = (g.density_db(), g.cut_db());
        assert!(c > 0.0);
        let mut f = inputs(-20.0);
        f.frozen = true;
        for _ in 0..2000 {
            g.on_sub_block(&f);
        }
        assert_eq!((g.density_db(), g.cut_db()), (d, c));
        assert_eq!(g.state(), GovernorState::Frozen);
        g.set_params(false, 2.0);
        assert_eq!((g.density_db(), g.cut_db()), (0.0, 0.0));
        assert!(!g.holds_trim() && g.deficit_lu() == 0.0);
    }

    #[test]
    fn well_under_budget_releases() {
        let mut g = ClipBudgetGovernor::default();
        g.allocate();
        g.set_params(true, 2.0);
        for _ in 0..3000 {
            g.on_sub_block(&inputs(-20.0));
        }
        assert!(g.cut_db() > 0.0);
        let mut q = inputs(-60.0);
        q.clip = ClipStats::default();
        for _ in 0..100_000 {
            g.on_sub_block(&q);
        }
        assert_eq!((g.density_db(), g.cut_db()), (0.0, 0.0));
        assert_eq!(g.state(), GovernorState::Free);
        assert!(!g.holds_trim());
    }
}
