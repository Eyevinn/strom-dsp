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

/// Preallocated scratch for one interpolation problem.
#[derive(Debug, Clone, Default)]
pub struct LsarSolver {
    p_max: usize,
    n_max: usize,
    m_max: usize,
    /// AR coefficients `a[0..=p]`.
    a: Vec<f64>,
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
}

impl LsarSolver {
    /// Scratch for windows of up to `n_max` samples, `m_max` unknowns and AR
    /// order up to `p_max`. Allocates.
    pub fn new(n_max: usize, m_max: usize, p_max: usize) -> Self {
        Self {
            p_max,
            n_max,
            m_max,
            a: vec![0.0; p_max + 1],
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
        }
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
            let denom = (n + 1) as f64;
            for (i, h) in self.hann[..n].iter_mut().enumerate() {
                *h = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * (i + 1) as f64 / denom).cos();
            }
            self.hann_len = n;
        }
        for ((d, &x), &h) in self.win[..n].iter_mut().zip(w).zip(&self.hann[..n]) {
            *d = x * h;
        }
        let win = &self.win[..n];
        for lag in 0..=p {
            let mut s = 0.0;
            for k in lag..n {
                s += win[k] * win[k - lag];
            }
            self.r[lag] = s;
        }
        if self.r[0] <= 0.0 || !self.r[0].is_finite() {
            // Silence (or non-finite): the trivial predictor.
            self.a[..=p].fill(0.0);
            self.a[0] = 1.0;
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
    /// On error `w` is left as on entry.
    pub fn interpolate(
        &mut self,
        w: &mut [f64],
        unknown: &[usize],
        lo: &[f64],
        hi: &[f64],
        order: usize,
        rounds: usize,
    ) -> Result<(), LsarError> {
        let n = w.len();
        let m = unknown.len();
        let p = order.min(self.p_max);
        if n > self.n_max || m > self.m_max {
            return Err(LsarError::TooLarge);
        }
        if p == 0 || n < 2 * p + 1 {
            return Err(LsarError::WindowTooShort);
        }
        if w.iter().any(|v| !v.is_finite()) {
            return Err(LsarError::Numerical);
        }
        if m == 0 {
            return Ok(());
        }
        for _ in 0..rounds.max(1) {
            let p = self.estimate(w, p)?;
            self.coef_autocorr(p);
            self.free.clear();
            self.free_slot.clear();
            for (slot, &i) in unknown.iter().enumerate() {
                self.free.push(i);
                self.free_slot.push(slot);
                // Start each round from the box (the clipped value on the
                // first round): only the free unknowns are re-solved.
                w[i] = w[i].clamp(lo[slot], hi[slot]);
            }
            for _ in 0..MAX_ACTIVE_SET_ROUNDS {
                if self.free.is_empty() {
                    break;
                }
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
        }
        Ok(())
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
        let mut next = p;
        for &i in &self.free {
            let k1 = (i + p).min(n - 1);
            for k in next.max(i)..=k1 {
                let mut s = 0.0;
                for j in 0..=p {
                    s += self.a[j] * w[k - j];
                }
                self.e0[k] = s;
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
            let mut s = 0.0;
            for k in k0..=k1 {
                s += self.a[k - i] * self.e0[k];
            }
            self.rhs[f] = -s;
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
                    let mut s = 0.0;
                    let mut k = k0;
                    while k <= k1 {
                        s += self.a[k - i] * self.a[k - j];
                        k += 1;
                    }
                    s
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

/// Solve `M·x = b` in place (`M` symmetric positive definite, row-major
/// `m × m`, overwritten by its Cholesky factor; `b` overwritten by `x`).
fn cholesky_solve(mat: &mut [f64], b: &mut [f64], m: usize) -> Result<(), LsarError> {
    for j in 0..m {
        let mut d = mat[j * m + j];
        for k in 0..j {
            d -= mat[j * m + k] * mat[j * m + k];
        }
        let positive = d > 0.0;
        if !positive || !d.is_finite() {
            return Err(LsarError::Numerical);
        }
        let d = d.sqrt();
        mat[j * m + j] = d;
        let inv = 1.0 / d;
        for i in j + 1..m {
            let mut s = mat[i * m + j];
            for k in 0..j {
                s -= mat[i * m + k] * mat[j * m + k];
            }
            mat[i * m + j] = s * inv;
        }
    }
    // L·y = b
    for i in 0..m {
        let mut s = b[i];
        for k in 0..i {
            s -= mat[i * m + k] * b[k];
        }
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
        s.interpolate(&mut w, &unknown, &lo, &hi, 16, 3).unwrap();
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
        s.interpolate(&mut w, &unknown, &lo, &hi, 8, 2).unwrap();
        for &i in &unknown {
            assert!((0.3..=0.4).contains(&w[i]));
        }
    }
}
