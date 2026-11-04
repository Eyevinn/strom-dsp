//! Least-squares AR interpolation of missing samples, with box constraints.
//!
//! The model (Janssen, Veldhuis, Vries 1986; Godsill & Rayner 1998, ch. 5,
//! "least squares AR interpolation"): the signal in a short window
//! `w[0] … w[N−1]` is an autoregressive process of order `p`,
//!
//! `e[k] = Σ_{j=0..p} a_j · w[k−j]`, `a_0 = 1`, `k = p … N−1`,
//!
//! with a small, white prediction error `e`. The unknown samples are the ones
//! that minimise `Σ_k e[k]²` given the known ones. Splitting the window into
//! the unknown samples `x_U` and the rest, `e = A_U·x_U + e₀` (`e₀` is the
//! prediction error with the unknowns set to zero), so the minimiser solves
//! the normal equations
//!
//! `(A_Uᵀ A_U) · x_U = −A_Uᵀ e₀`.
//!
//! `(A_Uᵀ A_U)[i][j] = Σ_k a_{k−i}·a_{k−j}` over the rows `k` that contain
//! both samples. Away from the window edges this is the autocorrelation of
//! the coefficients, `R_a(|i−j|)`; near an edge the sum is taken over the rows
//! that exist. The matrix is symmetric positive definite whenever every
//! unknown sits in at least one row; it is solved by Cholesky.
//!
//! **AR estimate.** The coefficients come from the whole window (with the
//! current values of the unknowns) by the autocorrelation method: Hann
//! window, biased autocorrelation, a white-noise correction of
//! `1 + WHITE_NOISE` on lag 0, Levinson–Durbin recursion. The correction keeps
//! the recursion stable on near-singular windows (a pure tone). Janssen et al.
//! alternate the AR estimate and the interpolation; the caller chooses the
//! number of rounds.
//!
//! With [`ArMethod::Burg`] the first estimate uses Burg's method (J. P.
//! Burg, *Maximum entropy spectral analysis*, PhD thesis, Stanford 1975;
//! S. M. Kay, *Modern Spectral Estimation*, 1988, ch. 7) on the known samples
//! only: the forward and backward prediction errors run over each stretch of
//! known samples between the unknowns, and the reflection coefficient of
//! each order is `−2·Σ f·b / Σ (f² + b²)` summed over all stretches. The
//! clipped values never enter the model, and Burg needs no data window,
//! which biases short-window estimates towards a flatter spectrum. Later
//! rounds use the autocorrelation method on the whole window with the
//! interpolated values in it.
//!
//! **Constraints** (Abel & Smith 1991: a clipped sample's true value lies
//! beyond the clip level, on the same side). Each unknown has a box
//! `lo_i ≤ x_i ≤ hi_i`. The constrained least-squares problem is approximated
//! by a bounded active-set loop: solve with all unknowns free; every free
//! unknown outside its box is fixed at the bound it crossed and becomes
//! known; solve again for the rest; at most [`MAX_ACTIVE_SET_ROUNDS`] times.
//! A bound is never released again (a full active-set QP would test the
//! Lagrange multipliers); after the last round every unknown is projected
//! into its box. The cost is bounded and the result always satisfies the
//! constraints.

/// White-noise correction on lag 0 of the autocorrelation (−50 dB).
pub const WHITE_NOISE: f64 = 1e-5;

/// Most solves in the active-set loop.
pub const MAX_ACTIVE_SET_ROUNDS: usize = 4;

/// How the AR coefficients are estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ArMethod {
    /// Autocorrelation method on the whole window (Hann window,
    /// Levinson–Durbin), alternated with the interpolation.
    #[default]
    Autocorrelation,
    /// Burg's method on the known samples only (the stretches between the
    /// unknowns), summed over the stretches.
    Burg,
}

/// Preallocated scratch for one interpolation problem.
#[derive(Debug, Clone, Default)]
pub struct LsarSolver {
    /// Estimation method.
    pub method: ArMethod,
    /// Most known samples before the first unknown that the Burg estimate
    /// uses (the rest of the window still serves the solve and the later
    /// rounds). Default: all.
    pub burg_before: usize,
    /// With [`ArMethod::Autocorrelation`]: when the first round leaves every
    /// unknown at its starting value (the model has learnt the clipped
    /// plateau), try the first round again with Burg on the known samples,
    /// if the work limit allows. Default `false`.
    pub burg_retry: bool,
    /// Burg forward and backward prediction errors.
    fwd: Vec<f64>,
    bwd: Vec<f64>,
    /// Stretches of known samples `(first, last)` for Burg.
    segs: Vec<(usize, usize)>,
    p_max: usize,
    n_max: usize,
    m_max: usize,
    /// AR coefficients `a[0..=p]`.
    a: Vec<f64>,
    /// The first round's coefficients.
    a0: Vec<f64>,
    /// Signal autocorrelation, then coefficient autocorrelation `R_a`.
    r: Vec<f64>,
    ra: Vec<f64>,
    /// Levinson scratch.
    tmp: Vec<f64>,
    /// Hann-windowed copy of the window.
    win: Vec<f64>,
    /// Hann window of the current length.
    hann: Vec<f64>,
    hann_len: usize,
    /// Prediction error with the free unknowns at zero.
    e0: Vec<f64>,
    /// Normal-equation matrix (row-major, `m × m`) and right-hand side.
    mat: Vec<f64>,
    rhs: Vec<f64>,
    /// Free unknowns (window indices) and their slot in the caller's list.
    free: Vec<usize>,
    free_slot: Vec<usize>,
    /// Values of the unknowns after the last round that finished.
    kept: Vec<f64>,
    /// Prediction gain of the last estimate, of the first round's model, and
    /// of the model behind the kept values (0: not known).
    gain_est: f64,
    gain0: f64,
    gain: f64,
    /// Work of the last [`interpolate`](Self::interpolate) call, also when
    /// it returned [`LsarError::Budget`].
    spent: f64,
}

/// Why an interpolation was not done.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LsarError {
    /// The window is too short for the model order.
    WindowTooShort,
    /// More unknowns, or a longer window, than the solver was sized for.
    TooLarge,
    /// The window holds a non-finite value, or the solve broke down.
    Numerical,
    /// The first round did not fit the work limit: not even its estimate
    /// and one solve, or its active-set loop had to stop with unknowns
    /// still outside their boxes. [`LsarSolver::spent`] gives the work
    /// done before it stopped.
    Budget,
}

impl LsarSolver {
    /// Scratch for windows of up to `n_max` samples, `m_max` unknowns and AR
    /// order up to `p_max`. Allocates.
    pub fn new(n_max: usize, m_max: usize, p_max: usize) -> Self {
        Self {
            method: ArMethod::Autocorrelation,
            burg_before: usize::MAX,
            burg_retry: false,
            fwd: vec![0.0; n_max],
            bwd: vec![0.0; n_max],
            segs: Vec::with_capacity(m_max + 1),
            p_max,
            n_max,
            m_max,
            a: vec![0.0; p_max + 1],
            a0: vec![0.0; p_max + 1],
            r: vec![0.0; p_max + 1],
            ra: vec![0.0; p_max + 1],
            tmp: vec![0.0; p_max + 1],
            win: vec![0.0; n_max],
            hann: vec![0.0; n_max],
            hann_len: 0,
            e0: vec![0.0; n_max],
            mat: vec![0.0; m_max * m_max],
            rhs: vec![0.0; m_max],
            free: Vec::with_capacity(m_max),
            free_slot: Vec::with_capacity(m_max),
            kept: vec![0.0; m_max],
            gain_est: 0.0,
            gain0: 0.0,
            gain: 0.0,
            spent: 0.0,
        }
    }

    /// Work of the last [`interpolate`](Self::interpolate) call, including
    /// one that returned [`LsarError::Budget`].
    pub fn spent(&self) -> f64 {
        self.spent
    }

    /// Work of the Burg estimate on a window of `n` with these unknowns.
    fn burg_units(&self, n: usize, unknown: &[usize], p: usize) -> f64 {
        let skip = unknown
            .first()
            .map_or(0, |&u| u.saturating_sub(self.burg_before));
        let (n, nb, p) = (n as f64, (n - skip) as f64, p as f64);
        n * (p + 2.0) + 1.5 * p * p + 5.0 * p * nb
    }

    /// Largest window length.
    pub fn n_max(&self) -> usize {
        self.n_max
    }

    /// Largest number of unknowns.
    pub fn m_max(&self) -> usize {
        self.m_max
    }

    /// Estimate AR(`p`) coefficients from `w` into `self.a`. Returns the order used.
    fn estimate(&mut self, w: &[f64], p: usize) -> Result<usize, LsarError> {
        let n = w.len();
        if self.hann_len != n {
            // h[i] = (1 − cos((i + 1)·δ))/2, δ = 2π/(n + 1); the cosine by a
            // rotation recurrence (the window length changes from run to run).
            let delta = 2.0 * std::f64::consts::PI / (n + 1) as f64;
            let (sd, cd) = delta.sin_cos();
            let (mut c, mut s) = (cd, sd);
            for h in self.hann[..n].iter_mut() {
                *h = 0.5 - 0.5 * c;
                let c2 = c * cd - s * sd;
                s = s * cd + c * sd;
                c = c2;
            }
            self.hann_len = n;
        }
        for ((d, &x), &h) in self.win[..n].iter_mut().zip(w).zip(&self.hann[..n]) {
            *d = x * h;
        }
        let win = &self.win[..n];
        for lag in 0..=p {
            self.r[lag] = dot(&win[lag..], &win[..n - lag]);
        }
        if self.r[0] <= 0.0 || !self.r[0].is_finite() {
            // Silence (or non-finite): the trivial predictor.
            self.a[..=p].fill(0.0);
            self.a[0] = 1.0;
            self.gain_est = 0.0;
            return if self.r[0] == 0.0 {
                Ok(p)
            } else {
                Err(LsarError::Numerical)
            };
        }
        self.r[0] *= 1.0 + WHITE_NOISE;
        // Levinson–Durbin.
        self.a[..=p].fill(0.0);
        self.a[0] = 1.0;
        let mut err = self.r[0];
        for i in 1..=p {
            let mut acc = self.r[i];
            for j in 1..i {
                acc += self.a[j] * self.r[i - j];
            }
            let k = -acc / err;
            if !k.is_finite() || k.abs() >= 1.0 {
                // Numerically at the edge of stability: stop at the order reached.
                for c in &mut self.a[i..=p] {
                    *c = 0.0;
                }
                break;
            }
            self.tmp[..i].copy_from_slice(&self.a[..i]);
            for j in 1..i {
                self.a[j] = self.tmp[j] + k * self.tmp[i - j];
            }
            self.a[i] = k;
            err *= 1.0 - k * k;
            let positive = err > 0.0;
            if !positive {
                break;
            }
        }
        self.gain_est = if err > 0.0 { self.r[0] / err } else { 0.0 };
        Ok(p)
    }

    /// Burg estimate of AR(`p`) coefficients into `self.a` from the known
    /// samples of `w` (every index not in `unknown`), the reflection
    /// coefficients summed over the stretches between the unknowns.
    fn estimate_burg(
        &mut self,
        w: &[f64],
        unknown: &[usize],
        p: usize,
    ) -> Result<usize, LsarError> {
        let n = w.len();
        self.segs.clear();
        let mut start = unknown
            .first()
            .map_or(0, |&u| u.saturating_sub(self.burg_before));
        for &u in unknown.iter().chain(std::iter::once(&n)) {
            if u > start && u - start > p {
                self.segs.push((start, u - 1));
            }
            start = u + 1;
        }
        self.a[..=p].fill(0.0);
        self.a[0] = 1.0;
        self.fwd[..n].copy_from_slice(w);
        self.bwd[..n].copy_from_slice(w);
        let mut energy = 0.0;
        for &(a, e) in &self.segs {
            for v in &w[a..=e] {
                energy += v * v;
            }
        }
        self.gain_est = 0.0;
        if energy == 0.0 || self.segs.is_empty() {
            return Ok(p);
        }
        // Prediction error power relative to the signal: Π (1 − k²).
        let mut rel_err = 1.0;
        if !energy.is_finite() {
            return Err(LsarError::Numerical);
        }
        for m in 1..=p {
            let (mut num, mut den) = (0.0, 0.0);
            for &(a, e) in &self.segs {
                let f = &self.fwd[a + m..=e];
                let b = &self.bwd[a + m - 1..e];
                num += dot(f, b);
                den += sum_sq2(f, b);
            }
            let positive = den > 0.0;
            if !positive {
                break;
            }
            let k = -2.0 * num / den;
            if !k.is_finite() || k.abs() >= 1.0 {
                break;
            }
            for &(a, e) in &self.segs {
                // New errors from the old ones: the backward errors shift by
                // one sample (copied first, so the loop has no dependency).
                let len = e + 1 - (a + m);
                let old_b = &mut self.win[..len];
                old_b.copy_from_slice(&self.bwd[a + m - 1..e]);
                let f = &mut self.fwd[a + m..=e];
                let b = &mut self.bwd[a + m..=e];
                for ((fj, bj), &bp) in f.iter_mut().zip(b.iter_mut()).zip(old_b.iter()) {
                    let f_old = *fj;
                    *fj = f_old + k * bp;
                    *bj = bp + k * f_old;
                }
            }
            self.tmp[..m].copy_from_slice(&self.a[..m]);
            for j in 1..m {
                self.a[j] = self.tmp[j] + k * self.tmp[m - j];
            }
            self.a[m] = k;
            rel_err *= 1.0 - k * k;
        }
        self.gain_est = if rel_err > 0.0 { 1.0 / rel_err } else { 0.0 };
        Ok(p)
    }

    /// Interpolate the unknowns of `w` in place.
    ///
    /// `unknown[i]` are window indices (ascending, distinct), `lo[i] ≤ hi[i]`
    /// their boxes; `w` holds their current values on entry (the first AR
    /// estimate uses them). `order` is the AR order (≤ `p_max`, and the window
    /// must be at least `2·order + 1` long); `rounds` the number of
    /// alternations of AR estimate and interpolation (≥ 1).
    ///
    /// `limit` caps the work, in the units of [`work_units`]: no step starts
    /// that would take the work past it. The first round uses
    /// [`method`](Self::method), or the cheaper autocorrelation estimate when
    /// that and one solve do not fit, or the coefficients in `first` when
    /// given (they must have `order + 1` entries, `first[0] = 1`; with their
    /// prediction gain, see [`prediction_gain`](Self::prediction_gain)). A later
    /// round that does not fit, or whose active-set loop has to stop for
    /// the limit, is dropped and the last finished round's values are kept.
    /// When the first round does not fit (not even its estimate and one
    /// solve, or its active-set loop stops for the limit), it returns
    /// [`LsarError::Budget`]; the unknowns then hold their values from
    /// entry, and [`spent`](Self::spent) the work done.
    ///
    /// Returns the work done. On another error the contents of `w` are
    /// undefined.
    #[allow(clippy::too_many_arguments)]
    pub fn interpolate(
        &mut self,
        w: &mut [f64],
        unknown: &[usize],
        lo: &[f64],
        hi: &[f64],
        order: usize,
        rounds: usize,
        limit: f64,
        first: Option<(&[f64], f64)>,
    ) -> Result<f64, LsarError> {
        let n = w.len();
        let m = unknown.len();
        let p = order.min(self.p_max);
        if n > self.n_max || m > self.m_max {
            return Err(LsarError::TooLarge);
        }
        if p == 0 || n < 2 * p + 1 {
            return Err(LsarError::WindowTooShort);
        }
        if first.is_some_and(|(a, _)| a.len() != p + 1) {
            return Err(LsarError::TooLarge);
        }
        if w.iter().any(|v| !v.is_finite()) {
            return Err(LsarError::Numerical);
        }
        if m == 0 {
            return Ok(0.0);
        }
        self.spent = 0.0;
        self.gain = 0.0;
        let mut work = 0.0;
        let first_solve = solve_units(m, p);
        for (slot, &i) in unknown.iter().enumerate() {
            self.kept[slot] = w[i];
        }
        // First round: the estimate (or the given model), then the
        // active-set loop. With `burg_retry`, a first round that leaves
        // every unknown at its starting value is tried again with Burg.
        let mut burg = self.method == ArMethod::Burg;
        let mut retry = false;
        let mut use_first = first;
        loop {
            let auto = estimate_units(n, p, false);
            if let Some((a, g)) = use_first {
                self.a[..=p].copy_from_slice(a);
                self.gain_est = g;
                work += 1.5 * (p * p) as f64;
            } else if burg && work + self.burg_units(n, unknown, p) + first_solve <= limit {
                work += self.burg_units(n, unknown, p);
                self.estimate_burg(w, unknown, p)?;
            } else if retry {
                // A retry that does not fit: keep the first attempt.
                break;
            } else if work + auto + first_solve <= limit {
                work += auto;
                self.estimate(w, p)?;
            } else {
                self.spent = work;
                return Err(LsarError::Budget);
            }
            self.a0[..=p].copy_from_slice(&self.a[..=p]);
            self.gain0 = self.gain_est;
            if self.active_set(w, unknown, lo, hi, p, &mut work, limit)? {
                for (slot, &i) in unknown.iter().enumerate() {
                    w[i] = self.kept[slot];
                }
                self.spent = work;
                if retry {
                    // The retry stopped: keep the first attempt, which
                    // left the unknowns at their starting values.
                    break;
                }
                return Err(LsarError::Budget);
            }
            let moved = unknown
                .iter()
                .enumerate()
                .any(|(slot, &i)| w[i] != self.kept[slot]);
            if moved || burg || !self.burg_retry {
                break;
            }
            // Nothing moved: the model has learnt the plateau. The unknowns
            // are back at their starting values, so `w` is as on entry.
            burg = true;
            retry = true;
            use_first = None;
        }
        for (slot, &i) in unknown.iter().enumerate() {
            self.kept[slot] = w[i];
        }
        self.gain = self.gain_est;
        for _ in 1..rounds.max(1) {
            let est = estimate_units(n, p, false);
            if work + est + first_solve > limit {
                break;
            }
            work += est;
            self.estimate(w, p)?;
            if self.active_set(w, unknown, lo, hi, p, &mut work, limit)? {
                // Stopped for the work limit with unknowns that were last
                // solved outside their boxes: projecting them can land
                // anywhere in the box. Keep the round before.
                for (slot, &i) in unknown.iter().enumerate() {
                    w[i] = self.kept[slot];
                }
                break;
            }
            for (slot, &i) in unknown.iter().enumerate() {
                self.kept[slot] = w[i];
            }
            self.gain = self.gain_est;
        }
        self.spent = work;
        Ok(work)
    }

    /// The bounded active-set loop for the current model (`self.a`): solve
    /// with every unknown free, fix those outside their box at the bound
    /// they crossed, solve again, at most [`MAX_ACTIVE_SET_ROUNDS`] times;
    /// then every unknown into its box. Returns `true` when it stopped for
    /// the work limit before it finished (the unknowns are then undefined).
    #[allow(clippy::too_many_arguments)]
    fn active_set(
        &mut self,
        w: &mut [f64],
        unknown: &[usize],
        lo: &[f64],
        hi: &[f64],
        p: usize,
        work: &mut f64,
        limit: f64,
    ) -> Result<bool, LsarError> {
        self.coef_autocorr(p);
        self.free.clear();
        self.free_slot.clear();
        for (slot, &i) in unknown.iter().enumerate() {
            self.free.push(i);
            self.free_slot.push(slot);
            // Start from the box (the clipped value on the first round):
            // only the free unknowns are re-solved.
            w[i] = w[i].clamp(lo[slot], hi[slot]);
        }
        for _ in 0..MAX_ACTIVE_SET_ROUNDS {
            if self.free.is_empty() {
                break;
            }
            let cost = solve_units(self.free.len(), p);
            if *work + cost > limit {
                return Ok(true);
            }
            *work += cost;
            self.solve_free(w, p)?;
            // Fix every free unknown outside its box at the bound it crossed.
            let mut kept = 0;
            for f in 0..self.free.len() {
                let i = self.free[f];
                let slot = self.free_slot[f];
                if w[i] < lo[slot] {
                    w[i] = lo[slot];
                } else if w[i] > hi[slot] {
                    w[i] = hi[slot];
                } else {
                    self.free[kept] = i;
                    self.free_slot[kept] = slot;
                    kept += 1;
                }
            }
            if kept == self.free.len() {
                break;
            }
            self.free.truncate(kept);
            self.free_slot.truncate(kept);
        }
        for (slot, &i) in unknown.iter().enumerate() {
            w[i] = w[i].clamp(lo[slot], hi[slot]);
        }
        Ok(false)
    }

    /// Prediction gain (signal power over prediction error power, linear)
    /// of the AR model behind the values the last successful
    /// [`interpolate`](Self::interpolate) left in the unknowns: how well the
    /// model predicts the window. For a model passed in as `first`, the gain
    /// given with it; 0 for a silent window.
    pub fn prediction_gain(&self) -> f64 {
        self.gain
    }

    /// The first round's AR coefficients of the last successful
    /// [`interpolate`](Self::interpolate) (`order + 1` entries, `a[0] = 1`).
    pub fn first_model(&self, order: usize) -> &[f64] {
        &self.a0[..=order.min(self.p_max)]
    }

    /// Prediction gain of [`first_model`](Self::first_model) on its window
    /// (see [`prediction_gain`](Self::prediction_gain)).
    pub fn first_model_gain(&self) -> f64 {
        self.gain0
    }

    /// `R_a(d) = Σ_t a_t·a_{t+d}`.
    fn coef_autocorr(&mut self, p: usize) {
        for d in 0..=p {
            let mut s = 0.0;
            for t in 0..=p - d {
                s += self.a[t] * self.a[t + d];
            }
            self.ra[d] = s;
        }
    }

    /// Least-squares solve for the free unknowns; writes them into `w`.
    fn solve_free(&mut self, w: &mut [f64], p: usize) -> Result<(), LsarError> {
        let n = w.len();
        let m = self.free.len();
        // e₀ with the free unknowns at zero.
        let saved = &mut self.rhs;
        for (f, &i) in self.free.iter().enumerate() {
            saved[f] = w[i];
            w[i] = 0.0;
        }
        // Only the rows that contain a free unknown are needed: the union of
        // [max(i, p), min(i + p, n − 1)] over the free unknowns (ascending).
        for j in 0..=p {
            self.tmp[j] = self.a[p - j];
        }
        let mut next = p;
        for &i in &self.free {
            let k1 = (i + p).min(n - 1);
            for k in next.max(i)..=k1 {
                self.e0[k] = dot(&self.tmp[..=p], &w[k - p..=k]);
            }
            next = next.max(k1 + 1);
        }
        for (f, &i) in self.free.iter().enumerate() {
            w[i] = saved[f];
        }
        // Right-hand side b_i = −Σ_k a_{k−i}·e₀[k], rows k ∈ [max(i,p), min(i+p,n−1)].
        for f in 0..m {
            let i = self.free[f];
            let (k0, k1) = (i.max(p), (i + p).min(n - 1));
            self.rhs[f] = -dot(&self.a[k0 - i..=k1 - i], &self.e0[k0..=k1]);
        }
        // Normal matrix.
        for f in 0..m {
            let i = self.free[f];
            for g in f..m {
                let j = self.free[g];
                let d = j - i;
                let v = if d > p {
                    0.0
                } else if i >= p && j + p < n {
                    self.ra[d]
                } else {
                    let (k0, k1) = (j.max(p), (i + p).min(n - 1));
                    if k0 > k1 {
                        0.0
                    } else {
                        dot(&self.a[k0 - i..=k1 - i], &self.a[k0 - j..=k1 - j])
                    }
                };
                self.mat[f * m + g] = v;
                self.mat[g * m + f] = v;
            }
        }
        // Diagonal loading, relative to the mean diagonal, for safety.
        let mut tr = 0.0;
        for f in 0..m {
            tr += self.mat[f * m + f];
        }
        let jitter = 1e-10 * tr / m as f64 + 1e-300;
        for f in 0..m {
            self.mat[f * m + f] += jitter;
        }
        cholesky_solve(&mut self.mat[..m * m], &mut self.rhs[..m], m)?;
        for f in 0..m {
            w[self.free[f]] = self.rhs[f];
        }
        Ok(())
    }
}

/// `Σ a[i]·b[i]` over the shorter length, with four partial sums (so the
/// compiler can vectorise it).
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0f64; 4];
    let (ac, ar) = a.as_chunks::<4>();
    let (bc, br) = b.as_chunks::<4>();
    for (x, y) in ac.iter().zip(bc) {
        for l in 0..4 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (x, y) in ar.iter().zip(br) {
        s += x * y;
    }
    s
}

/// `Σ a[i]² + b[i]²` over the shorter length, with four partial sums.
#[inline]
fn sum_sq2(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0f64; 4];
    let (ac, ar) = a.as_chunks::<4>();
    let (bc, br) = b.as_chunks::<4>();
    for (x, y) in ac.iter().zip(bc) {
        for l in 0..4 {
            acc[l] += x[l] * x[l] + y[l] * y[l];
        }
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (x, y) in ar.iter().zip(br) {
        s += x * x + y * y;
    }
    s
}

/// Work of one AR estimate on a window of `n` samples at order `p`
/// (multiply-adds, rounded up).
fn estimate_units(n: usize, p: usize, burg: bool) -> f64 {
    let (n, p) = (n as f64, p as f64);
    let base = n * (p + 2.0) + 1.5 * p * p;
    if burg {
        base + 5.0 * p * n
    } else {
        base
    }
}

/// Work of one least-squares solve with `m` free unknowns at order `p`
/// (multiply-adds, rounded up): prediction errors, right-hand side, normal
/// matrix and Cholesky.
fn solve_units(m: usize, p: usize) -> f64 {
    let (m, p) = (m as f64, p as f64);
    (m + p) * (p + 1.0) + 3.0 * m * (p + 1.0) + m * m * m / 6.0 + 2.0 * m * m
}

/// Upper bound of the work of [`LsarSolver::interpolate`] on a window of
/// `n` samples with `m` unknowns at order `p` and `rounds` rounds, in
/// multiply-adds (each step is counted at this rate when it runs).
pub fn work_units(n: usize, m: usize, p: usize, rounds: usize, method: ArMethod) -> f64 {
    let rounds = rounds.max(1);
    let mut w = 0.0;
    for r in 0..rounds {
        w += estimate_units(n, p, method == ArMethod::Burg && r == 0);
        w += MAX_ACTIVE_SET_ROUNDS as f64 * solve_units(m, p);
    }
    w
}

/// Solve `M·x = b` in place (`M` symmetric positive definite, row-major
/// `m × m`, overwritten by its Cholesky factor; `b` overwritten by `x`).
fn cholesky_solve(mat: &mut [f64], b: &mut [f64], m: usize) -> Result<(), LsarError> {
    for j in 0..m {
        let row_j = &mat[j * m..j * m + j];
        let d = mat[j * m + j] - dot(row_j, row_j);
        let positive = d > 0.0;
        if !positive || !d.is_finite() {
            return Err(LsarError::Numerical);
        }
        let d = d.sqrt();
        mat[j * m + j] = d;
        let inv = 1.0 / d;
        for i in j + 1..m {
            let (upper, lower) = mat.split_at_mut(i * m);
            let s = lower[j] - dot(&lower[..j], &upper[j * m..j * m + j]);
            lower[j] = s * inv;
        }
    }
    // L·y = b
    for i in 0..m {
        let s = b[i] - dot(&mat[i * m..i * m + i], &b[..i]);
        b[i] = s / mat[i * m + i];
    }
    // Lᵀ·x = y
    for i in (0..m).rev() {
        let mut s = b[i];
        for k in i + 1..m {
            s -= mat[k * m + i] * b[k];
        }
        b[i] = s / mat[i * m + i];
    }
    if b.iter().any(|v| !v.is_finite()) {
        return Err(LsarError::Numerical);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sum of two sines is an AR(4) process: a gap is filled almost exactly.
    #[test]
    fn fills_gap_in_two_tones() {
        let n = 200;
        let truth: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                (0.07 * t).sin() + 0.5 * (0.31 * t + 1.0).sin()
            })
            .collect();
        let mut w = truth.clone();
        let unknown: Vec<usize> = (95..105).collect();
        for &i in &unknown {
            w[i] = 0.0;
        }
        let lo = vec![-10.0; unknown.len()];
        let hi = vec![10.0; unknown.len()];
        let mut s = LsarSolver::new(n, 16, 16);
        s.interpolate(&mut w, &unknown, &lo, &hi, 16, 3, f64::INFINITY, None)
            .unwrap();
        for &i in &unknown {
            assert!(
                (w[i] - truth[i]).abs() < 1e-2,
                "{i}: {} vs {}",
                w[i],
                truth[i]
            );
        }
    }

    /// Burg on the known samples alone (one round) fills the same gap.
    #[test]
    fn burg_fills_gap_in_two_tones() {
        let n = 200;
        let truth: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                (0.07 * t).sin() + 0.5 * (0.31 * t + 1.0).sin()
            })
            .collect();
        let mut w = truth.clone();
        let unknown: Vec<usize> = (95..105).collect();
        for &i in &unknown {
            w[i] = 0.0;
        }
        let lo = vec![-10.0; unknown.len()];
        let hi = vec![10.0; unknown.len()];
        let mut s = LsarSolver::new(n, 16, 16);
        s.method = ArMethod::Burg;
        s.interpolate(&mut w, &unknown, &lo, &hi, 16, 1, f64::INFINITY, None)
            .unwrap();
        for &i in &unknown {
            assert!(
                (w[i] - truth[i]).abs() < 1e-2,
                "{i}: {} vs {}",
                w[i],
                truth[i]
            );
        }
    }

    #[test]
    fn respects_boxes() {
        let n = 120;
        let mut w: Vec<f64> = (0..n).map(|i| (0.2 * i as f64).sin()).collect();
        let unknown: Vec<usize> = (50..60).collect();
        let lo = vec![0.3; 10];
        let hi = vec![0.4; 10];
        let mut s = LsarSolver::new(n, 16, 8);
        s.interpolate(&mut w, &unknown, &lo, &hi, 8, 2, f64::INFINITY, None)
            .unwrap();
        for &i in &unknown {
            assert!((0.3..=0.4).contains(&w[i]));
        }
    }
}
