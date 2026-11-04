# Provenance

`strom-dsp` and `gst-plugin-stromdsp` are a **clean-room implementation**. Every
line of code in this repository was written new, from public descriptions of
algorithms (textbooks, papers, public-domain notes) and from the documentation
of permissively licensed dependencies.

## Clean-room rules followed

- **Not accessed, in any form:** LSP Plugins (and any fork, mirror or quotation
  of it), lsp-plugins-rs (source, checkouts or built plugin), the Strom
  repository, and source code under copyleft or unclear licences (GPL, LGPL,
  AGPL, MPL, CC-BY-SA, unlicensed), including other DSP plugin collections,
  GStreamer's own C audio filters and Stack Overflow code listings.
- **Allowed sources only:** algorithm descriptions and equations from
  textbooks, papers and standards; public-domain material (RBJ Audio EQ
  Cookbook); docs and source of permissively licensed dependencies; the
  gstreamer-rs bindings and GStreamer API docs.
- **No transcription from memory:** each algorithm was derived from the
  references listed below, in our own structure and naming.
- **Crates only for what is hard to get right** (FFT, standards-compliant
  measurement, resampling), and only under a licence on the `deny.toml`
  allowlist.
- **No comparison** of output, behaviour, performance or code with the plugin
  this project replaces, and no tuning to match the output of any product.
  Correctness is verified against theory (analytic gain curves, transfer
  functions, time constants) and invariants. Measuring against an
  implementation of a standard (for example the `ebur128` crate for
  ITU-R BS.1770) is allowed, in tests only.
- The only thing taken from the replaced plugin is the *interface* Strom
  relies on: property names, GLib types, units and ranges. The defaults are
  strom-dsp's own choice (see "Element defaults").

## Algorithms

### Shared helpers (`crates/strom-dsp/src/util.rs`)

- dB/linear conversion: `20·log10(x)` and `10^(dB/20)`, with a −200 dB floor so
  results are always finite.
- One-pole smoother `y[n] = a·y[n−1] + (1−a)·x[n]`, `a = exp(−1/(τ·fs))`: the
  standard discretisation of a first-order RC low-pass (impulse-invariant
  mapping of `e^(−t/τ)`); the step response reaches 1 − 1/e ≈ 63.2 % after τ.
  Reference: J. O. Smith III, *Introduction to Digital Filters*, one-pole
  filters; Giannoulis/Massberg/Reiss 2012, section on level detection.
- Denormals: recursive states are flushed to exactly zero below 1e-15
  (≈ −300 dB). Own design choice.

<!-- Per-processor sections follow. -->

### Gate (`crates/strom-dsp/src/gate.rs`)

Linked downward noise gate with hysteresis and hold, from the
textbook description of a gate (Zölzer, *DAFX*, dynamics chapter).

- **Detector (own choice):** key = max |x| over the channels of a frame, into a
  peak envelope follower with instant attack and a fixed 20 ms exponential
  release (`e = p` if `p > e`, else `e = a·e + (1−a)·p`). Thresholds compare
  against this peak level in dBFS.
- **State machine:** closed → open when `e > open`. While open, a hold counter
  of `round(hold·fs/1000)` frames is re-armed on every frame with `e ≥ close`;
  below close it counts down, and the gate closes when it reaches zero.
  `close > open` is treated as `close = open`.
- **Gain fades:** a linear-gain one-pole toward 1 (open) or `10^(reduction/20)`
  (closed), with the attack coefficient when rising and the release
  coefficient when falling. Time constant = 63.2 % of a step after τ.
- Parameter changes only move the fade target, so they are always smoothed.
  The envelope is denormal-flushed, the gain snaps to its target within 1e-12,
  and the gate starts closed after `configure`/`reset`.
- **Superseded:** two parts of this section were replaced after problems
  found by ear on speech and programme material, see "Sidechain, expander
  mode and gate detector" below:
  - the peak detector, now an RMS detector with a 1 dB minimum hysteresis;
  - the one-pole release, now a full-range dB ramp (`GateFade`).

### Compressor (`crates/strom-dsp/src/compressor.rs`)

Feed-forward downward compressor following D. Giannoulis, M. Massberg,
J. D. Reiss, "Digital Dynamic Range Compressor Design — A Tutorial and
Analysis", JAES 60(6), 2012.

- **Detector:** `x = 20·log10(max |x| over channels)` (peak, −200 dB floor).
- **Gain computer** (knee given as a linear amplitude factor `k`):
  `T = 20·log10(threshold)`, `k ∈ [0.001, 1]`, `W = −40·log10(k)`,
  `S = 1 − 1/ratio`. The gain reduction is 0 below `T − W/2`, `S·(x − T + W/2)²/(2W)` inside the knee (the paper's
  quadratic interpolation), and `S·(x − T)` above it. `W = 0` is a hard knee.
- **Smoothing:** a branching one-pole on the gain reduction in dB (paper's
  "smooth branching" variant): the attack coefficient while the reduction
  rises, release otherwise. Time constant = 63.2 % of a dB step after τ.
- **Output:** `10^(−y/20) · makeup`. Ratio 1 gives S = 0, so it is bit-exact unity.
- **Parameter smoothing (own choice):** threshold (dB), S, W and makeup glide
  with a 10 ms one-pole; values snap on the first block after
  `configure`/`reset`. States are f64 and denormal-flushed.

**Later changes:**

- **Detector peak hold (bass distortion fix, found by ear on bass-heavy
  programme).** At fast attack/release (1 ms /
  50 ms, 10:1) the per-frame peak detector followed every half-cycle. The gain
  rippled with the waveform: 3.0 % THD at 50 Hz, 0.77 % at 200 Hz.
  - The detector level is now the sliding maximum of the linked peak over
    H = min(20 ms, release) frames. It uses the van Herk / Gil–Werman segment
    algorithm (`SegmentedMin` on −p; references as for the limiter).
  - Reason: Giannoulis et al. 2012 on fast-time-constant artefacts. The
    detector must not follow individual cycles, so its release must be long
    compared with the period of the lowest frequency of interest.
  - Own choices: a window, not a slower exponential release, so attack stays
    instant. H is tied to the release.
  - Result: THD below 0.00001 % from 50 to 200 Hz. The release starts up to
    20 ms later.
  - Guard test: `tests/compressor_thd.rs`, which computes THD by DFT over an
    integer number of cycles.
- **Detector mode `peak` | `rms`.** The `rms` mode is a one-pole mean square
  over `rms-window` (Zölzer, *DAFX*, RMS level measurement), the gate's
  `MeanSquareEnvelope`.
  - Own choices: sine calibration ×2 (+3.01 dB) in both modes, so a threshold
    means the same in each, and the same as the gate; the peak hold comes after
    the mean square; switching modes seeds the new detector from the held level.
- **Auto-makeup (programme-dependent).** The first version used the static reduction at −18 dB. It under-compensated
  speech that peaks near −3 dBFS by 2–6 LU, so it was replaced.
  - Starting point: automatic make-up gain from the gain computer (Giannoulis,
    Massberg & Reiss, JAES 60(6), 2012). Related work on automating compressor
    parameters from signal statistics: Giannoulis, Massberg & Reiss, "Parameter
    Automation in a Dynamic Range Compressor", JAES 61(10), 716–726, 2013. That
    is cited as context only; the method below is our own.
  - Own design: makeup M = clamp(M_s + C, 0, 24 dB), using only frames whose
    detector level is ≥ −50 dB (it freezes in pauses).
    - M_s applies the current static curve to an energy-weighted level
      histogram (61 × 1 dB bins), so setting changes take effect at once.
    - C is a measured correction for attack and release, bounded to ±6 dB.
    - Both age with τ = 10 s of gated frames. A short prior (the static
      reduction at −12 dB, 0.5 s) avoids start-up jumps.
    - Energy weighting is used because BS.1770 loudness is a mean square.
  - Test: integrated output loudness stays within ±2 LU of the input (measured
    ±1.0 LU) over thresholds −30 … −10 dB, ratios 2 … 10, both detectors and
    three speech-like programmes. Measured with `meter::LoudnessMeter`. The old
    reference failed 87 of 150 grid points.
- **Peak hold on speech.** The hold reads the true peak of every cycle, so peak
  mode compresses speech about 3 dB harder than a detector without hold. A
  shorter hold barely changes this but breaks the THD guard. The hold stays at
  20 ms; use `rms` for gentler speech compression.


### Equalizer (`crates/strom-dsp/src/equalizer.rs`)

Up to 8 second-order bands in series. Coefficients from the equations of
R. Bristow-Johnson, "Audio EQ Cookbook" (W3C Note, public domain):
`w0 = 2π·f0/fs`, `α = sin(w0)/(2Q)`, `A = 10^(gain/40)`, for LPF, HPF, BPF
(constant skirt gain, b0 = sin(w0)/2), BPF (constant 0 dB peak, b0 = α),
notch, APF, peakingEQ, lowShelf and highShelf (cookbook Q form, not S).

- **Numerics (own choice):** `1 − cos w0` and `1 + cos w0` are evaluated as
  `2·sin²(w0/2)` and `2·cos²(w0/2)`, and the shelf terms are rewritten in terms
  of `1 − cos w0`, to avoid cancellation at very low w0. All coefficients are
  normalised by a0. A test checks them against the literal cookbook formulas.
- **Structure:** Direct Form I (J. O. Smith III, *Introduction to Digital
  Filters*), f64 coefficients and state per band and channel. DF-I keeps only
  past inputs/outputs as state, so coefficient changes never leave a state
  that is inconsistent with the coefficients.
- **Clamping:** frequency to `[1 Hz, min(24 kHz, 0.499·fs)]`, gain ±60 dB,
  Q 0.01…100; NaN/Inf parameters fall back to defaults.
- **Parameter changes (own design):** frequency, gain and Q glide through a
  10 ms one-pole in ln f, dB and ln Q; coefficients are recomputed every 32
  frames and linearly interpolated per sample in between. Every intermediate
  filter is stable, because the stability triangle in (a1, a2) is convex.
  Type changes, enable/disable and `num_bands` changes use a 10 ms linear
  crossfade between the old and new realisation (a second state bank).
- Off, disabled and unused bands are skipped entirely (bit-exact, zero cost).
  Output is clamped to the finite f32 range.
- **Hot path (own design):** channels are processed as lanes in pairs
  (structure-of-arrays state, plus a single lane for an odd channel), and all
  bands run in series per sample with the signal kept in f64 between bands.
  There are specialised loops for 1–8 bands: a plain steady-state loop, a
  ramp loop with a per-stage coefficient step, and a general loop used only
  during crossfades. The DF-I sum is ordered so the loop-carried dependency is
  one multiply-add (fused where the target has FMA). Steady-state loops flush
  filter state below 1e-15 every 32 frames; the crossfade loop every sample.

### Biquad building blocks (`crates/strom-dsp/src/biquad.rs`)

Public, reusable primitives; the equalizer is built from them.

- `BiquadCoefficients`: normalised coefficients (a0 = 1) with one design
  function per RBJ cookbook type (as above), `magnitude`/`magnitude_at`
  (analytic |H(e^jω)|), `is_stable` (stability triangle, J. O. Smith III) and
  linear per-sample interpolation steps.
- `Biquad`: Direct Form I filter with f64 state for mono or interleaved
  multichannel data, with `set_coefficients` and `ramp_to` (stable throughout,
  because the stability triangle is convex). Never allocates after `new`.
- Internal helpers in `lanes.rs`: thin wrappers over `fearless_simd` vectors,
  used by the explicit kernels below.

### Explicit SIMD: biquad/EQ kernels (`src/biquad/kernel.rs`, `src/lanes.rs`)

Written from the documentation, source and examples of `fearless_simd` 1.2
(Apache-2.0 OR MIT): `dispatch!`, `Simd::vectorize`, the `f64x2/f64x4/f64x8`
types, and the per-level `mul_add` semantics.

- **Dispatch:** the level is detected once in `configure`/`new` and dispatched
  once per `process` call. `set_simd_level` lets hosts and tests override it.
- **Lanes:** channels are lanes at the level's native f64 width (f64x2 on
  NEON/SSE, f64x4 on AVX2, f64x8 on AVX-512), with padded remainder groups. The
  structure-of-arrays state layout does not depend on the level, so the level
  can change between calls.
- **Evaluation order:** `s = x2·b2 + x1·b1 − y2·a2`, `t = x·b0 + s`,
  `y = t − y1·a1`. The `y[n−1]` recurrence is a single multiply-add.
- **FMA policy:** fused on NEON/AVX2/AVX-512, unfused on SSE. The scalar
  fallback follows the platform baseline. Levels with the same policy are
  bit-identical; tests check this for 1–64 channels through ramps,
  crossfades and type changes.
- **Rejected after measurement:** a skewed (software-pipelined) schedule and
  bands-as-lanes for mono, both textbook pipelining ideas implemented from
  first principles; both were measured slower on NEON (2–6 % and about 2.5×).

### Explicit SIMD: dynamics (`src/dynamics/frame.rs`, `vmath.rs`, `SegmentedMin`)

The gate, compressor and limiter dispatch at runtime through `fearless_simd`
(written from its docs and source).

- **Exactness:** vector kernels use only IEEE-exact operations: add, sub, mul,
  div, compare/select, round-to-integer and bit operations, with no FMA. Every
  level, including the scalar fallback, is therefore bit-identical, which
  `tests/simd_levels.rs` verifies. Linked peaks use a compare-select maximum
  that ignores NaN.
- **`20·log10(x)`** (`vmath.rs`): split exponent and mantissa from the IEEE
  fields, centre the mantissa in [√½, √2), then ln m = 2·atanh(s) with
  s = (m−1)/(m+1), as a power series through s¹⁵/15 (Abramowitz & Stegun,
  *Handbook of Mathematical Functions*, 4.1.27).
- **`10^(−y/20)`:** computed as 2^t with t = n + f and n = round(t). 2^f =
  e^(f·ln 2) is a Taylor series through u¹¹/11!, and 2^n goes directly into
  the exponent field (textbook range reduction).
- **Accuracy:** tested against `f64::log10`/`exp`. Maximum errors are 2.3e-13
  dB (log) and 4.3e-13 dB (gain); the test asserts < 1e-9 dB.
- **Sliding minimum (`SegmentedMin`):** prefix and suffix minima over segments
  of the window length, following M. van Herk, *Pattern Recognition Letters*
  13(7), 1992, and J. Gil & M. Werman, *IEEE Trans. PAMI* 15(5), 1993. It uses
  comparisons only, so it is the exact minimum and the limiter's ceiling proof
  is unchanged. Chosen for its flat worst-case cost: about 28 % faster than
  the deque under heavy limiting and about 7 % slower on light material.
- **`AttackRelease::step_block`:** the same one-pole written as `a·y + k·x`, to
  shorten the loop-carried dependency (own design).
- **Considered and rejected:** division by invariant integers via
  multiplication (Granlund & Montgomery, PLDI 1994); no measurable gain.

### Dynamics building blocks (`crates/strom-dsp/src/dynamics/`)

Public, reusable primitives; the gate, compressor and limiter are built from
them. Time constants everywhere: `a = exp(−1/(τ·fs))` (63.2 % after τ).

| Block | What | Reference |
|---|---|---|
| `OnePole` | one-pole smoother toward a target, snaps exactly when close | J. O. Smith III, one-pole low-pass (impulse-invariant RC) |
| `AttackRelease` | one-pole with separate rise/fall coefficients | Giannoulis et al. 2012, "branching" smoother |
| `PeakEnvelope` | peak follower, instant attack, exponential release, denormal flush | Giannoulis et al. 2012, level detection |
| `GainComputer` | static soft-knee curve, knee as a linear amplitude factor | Giannoulis et al. 2012 |
| `GateStateMachine` | hysteresis thresholds + hold counter | Zölzer, *DAFX*, dynamics chapter |
| `DelayLine` | multichannel interleaved ring buffer | standard circular buffer, own design |
| `SlidingMin` | sliding-window minimum, monotonic deque in a preallocated ring | standard ascending-minima technique |
| `BoxAverage` | exact integer moving average, floor-rounded | J. O. Smith III, moving average |
| `RampSmoother` | two cascaded `BoxAverage`s in fixed point; output never exceeds any bound holding for its last A+1 inputs | own derivation (limiter ceiling proof) |
| `frame` | linked-channel peak / gain-apply helpers with mono/stereo paths | own design |

Gate and compressor work in 64-frame blocks (peaks → serial envelope chain →
apply gains). While parameters are steady, the compressor skips `log10` below
the knee and `exp` at zero reduction. Between `configure`/`reset` and the first
processed frame, `set_params` applies instantly (no glide, no cross-fade), so
the call order never changes the output.

### Limiter (`crates/strom-dsp/src/limiter.rs`)

Linked lookahead sample-peak limiter, derived from first principles from the
requirement "the gain applied to output frame n is ≤ the gain required by input
frame n − L". General lookahead/gain-smoothing ideas from Zölzer (*DAFX*,
dynamics chapter) and Giannoulis et al. 2012; moving-average and one-pole
basics from J. O. Smith III. No code listing was used.

- **Required gain:** `r[j] = min(1, c / max_ch |x_ch[j]|)`, where `c` is the
  largest f32 not above `10^(threshold/20)`. Audio is delayed by L frames.
- **Gain shape (own design):** (1) a sliding minimum of r over
  `[m−L, m−L+A]` (A = attack frames ≤ L), using a monotonic deque in a
  preallocated ring (O(1) amortised); (2) two cascaded moving averages of
  lengths ⌈A/2⌉+1 and ⌊A/2⌋+1, giving an S-shaped ramp of A frames that ends at
  the peak; (3) release: the gain follows the ramp down immediately and rises
  toward it with a one-pole in linear gain (63 % after `release` ms).
- **Guarantee:** every term averaged in (2) has a window containing n − L, so
  the gain is ≤ r[n−L]. Lookahead 0 is the case A = 0 (instant drop). The
  averages run in 44-bit fixed point with floor rounding, so every rounding
  lowers the gain, and the ceiling is itself representable in f32.
- **Parameter changes:** threshold and `enabled` are stored per input frame
  (take effect with the audio's latency). An attack change rebuilds the
  pipeline from stored history (no allocation). A lookahead change cross-fades
  the delay tap over 5 ms, with the min window widened to cover both taps.
  `enabled = false` ramps the gain back to exactly 1 and leaves a bit-exact
  delay of the same latency.
- **Safety clip:** a guard at the ceiling that counts each intervention
  (`safety_clip_count()`); tests show it stays at 0. Release state is snapped
  to its target below 1e-15 (denormal-safe).
- **Gain hold (bass distortion fix, found by ear on bass-heavy programme).**
  - **Problem:** at fast settings (1 ms / 50 ms) the sliding-min window covered
    only A + 1 frames, so the gain released between waveform peaks. THD was
    3.4 % at 50 Hz, and 9.3 % with a 10 ms release.
  - **Fix:** the window is widened backwards by H = min(20 ms, release), to
    `[m − L − H, m − L + A]`. This follows Giannoulis et al. 2012 on artefacts
    of fast time constants.
  - **Proof:** unchanged in substance. The base part of every averaged window
    still contains n − L.
  - **Own choice:** on a rebuild (attack, hold or lookahead change), the hold
    part never reaches further back than the previous pipeline did, so a longer
    hold grows frame by frame instead of dropping a releasing gain.
  - **Result:** THD below 0.00005 % from 50 to 440 Hz. The release starts H
    frames later.
  - **Guard test:** `tests/limiter_thd.rs`.
- **Peak control (crate-internal hook, off unless set; used by the
  broadcast chain's true-peak limiter).** The compressor gain computer
  (Giannoulis et al. 2012: static curve with soft knee, branching one-pole
  attack/release in dB) on the frame peak, with its threshold under the
  ceiling and its slope scaled by a per-frame drive; the static reduction
  held for 20 ms (own choice: longer than a pitch period, so the smoothed
  reduction rises continuously on speech; the same reasoning as the gain
  hold above, after Giannoulis et al. on fast time constants). The required
  gain becomes `min(r[j], g_pc[j])`, so the proof above is unchanged (own
  derivation; a two-stage limiter is textbook, the min form keeps the
  stages from adding up). Guard tests: `tests/clipper_adaptive.rs`.

### Loudness detector (`src/meter/kweighting.rs`)

- **K-weighting:** follows ITU-R BS.1770-4/-5 Annex 1 §2.1, i.e. the
  pre-filter and RLB stages, with coefficients from tables 1 and 2 at 48 kHz.
- **Other sample rates:** an analog second-order prototype is recovered from
  the 48 kHz tables by inverting the pre-warped bilinear transform. It is then
  transformed again at the target rate (Oppenheim & Schafer; J. O. Smith III).
  - The closed-form inverse is our own derivation.
  - At 48 kHz it reproduces the tables to < 1e-12.
  - Up to 16 kHz, the response deviates from the 48 kHz response by ≤ 0.008 dB
    at 44.1–192 kHz.
- **Loudness:** BS.1770 eq. 2 (−0.691 offset, channel weights G). The
  momentary (400 ms) and short-term (3 s) windows follow EBU Tech 3341.
- **Running sums (own design):** 10 ms sub-block running sums, with periodic
  exact re-summation and re-summation triggered by a rounding-error bound.
  Readings are O(1).
- **Verification:** against the `ebur128` crate in tests only, within
  0.022 LU on noise.
- **Why not `meter::LoudnessMeter`:** fed and read every block, it costs about
  231 µs (2.3 % of a core). This detector costs about 1.3 µs.

### Wideband AGC (`src/agc.rs`)

A feed-forward loudness rider: a slow, gated, rate-limited gain that moves
the programme toward a target loudness.
- **Target gain:** `clamp(target − measured, −max_cut, +max_gain)`, updated
  every 10 ms. The applied gain ramps linearly in dB at separate up and down
  rates and stops exactly at the target, so it never overshoots.
- **Measurement:** BS.1770-4 Annex 1 §5.1 absolute gating on 400 ms blocks,
  with the AGC gate as the threshold.
- **Own choices:**
  - Freeze when the newest 100 ms is below the gate, or when there is less
    than 400 ms of gated programme.
  - A documented convergence bound.
- **Tests:**
  - Pink noise from −40 to −6 LUFS converges to target ±1 dB within the bound.
  - 10 s pauses move the gain by ≤ 0.1 dB, with no overshoot afterwards.
- No commercial processor material was used.

### True-peak limiter (`src/truepeak.rs`)

**Measurement concept.** ITU-R BS.1770-4/-5 Annex 2 defines the true peak as
the maximum absolute value of the signal oversampled 4×.

**Detector.**
- Interpolation: our polyphase half-band `Oversampler` at 4×, taking
  the max |·| over sub-samples and channels (linked).
- High-frequency term (own design): κ·|Δ⁶x|/64. The finite difference has gain
  sin⁶(πf/fs) (J. O. Smith III).
- Each term passes through a sliding maximum over ±8 frames (van Herk /
  Gil–Werman, `SegmentedMin`).

**Gain stage.** The sample-peak `Limiter`'s gain pipeline, reused through
crate-internal hooks. `Limiter` is unchanged and bit-identical. Attack equals
the lookahead.

**Latency.** round(lookahead·fs/1000) + 24 + 8 frames, constant. With
`enabled = false` the output is a bit-exact delay of the same length.

**Guarantee.**
- Proven: every output frame near an interpolated peak has gain ≤ c/|x̂|, and
  the output sample peak is ≤ c.
- Bounded and tested, not proven: gain modulation between neighbouring
  samples.

**Why the high-frequency term (own design choice).**
- A BS.1770 meter's interpolator (tested with the `ebur128` crate) reads
  near-Nyquist content higher than our half-band does. Without the term,
  clicks exceeded the ceiling by up to 0.78 dB.
- κ = 0.45 gives about 30 % headroom over the largest value an adversarial
  search needed. Measuring against an implementation of a standard is allowed
  by the clean-room rules above.
- Cost: extra limiting on HF-heavy material. About 1 dB at 15 kHz; full-band
  white noise ends about 1.9 dB under the ceiling.
- The term is a conservative detector margin. It is not a clipping or
  distortion-cancellation method.

**Measured.** The worst true-peak excess over the ceiling is +0.004 dB (meter
ripple), within the +0.1 dB requirement.

No commercial-processor material and no patent-driven method was used.

### Final clipper (`src/clipper.rs`)

**References.**
- Differential clipping, where only the clipped-off part is resampled and
  filtered and then subtracted from the delayed programme: US 6,337,999
  (oversampled clipping with filtered clippings) and US 4,208,548 (filtering
  the clipped-off part so the clipper's distortion stays out of a band).
  *Expired; status not legally reviewed.*
- Overshoot compensation after band-limiting the clippings: US 4,460,871.
  *Expired; status not legally reviewed.*
- Repeated clip-and-filter to control peak regrowth: J. Armstrong,
  Electronics Letters 38(5), 2002.
- First-order antiderivative antialiasing: Bilbao, Esqueda, Parker &
  Välimäki, IEEE SPL 24(7), 2017.
- Window-method FIR design (Kaiser window): Oppenheim & Schafer §7.5–7.6.
- Delay-complementary filter pairs `H = 1 − L`: the complementary-filter
  literature (e.g. Higgins, IEEE Trans. AES 11(3), 1975), with the bilinear
  transform pre-warped at the corner.

**Own design choices.**
- Curve: identity to `T − knee`, then `k + w·tanh(e/w)`; closed-form
  antiderivative `e²/2 − w²·ln cosh(e/w)` for the antialiasing.
- Latency settings: standard 8×, one pass, 41-tap shaping FIR (69 frames);
  low 4×, one pass, 3-tap FIR (48 frames); two passes (138 frames). At 4×
  a 15 kHz tone 6 dB over the threshold left a −75 … −80 dBFS alias, at 8×
  −94 dBFS. The shaping FIR is normalised to unit DC gain.
- The shaping FIR's passband `1 + a·(f/f_b)²`, `a = 1`, found by measuring
  sample-peak overshoot on our own synthetic signals for a ∈ {0, 0.3, 0.6,
  1, 1.5}. The FIR is centred half a high-rate sample early to undo the
  antialiasing's delay.
- Low cut `H(s) = s²(s + 3)/(s + 1)³` (complement of
  `L = (1 + 3s)/(1 + s)³`), −3 dB at the bass crossover: 12 dB/octave below
  it, no latency, near-zero phase error well above it.
- One pass by default (the limiter after the clipper takes the regrowth),
  two as an option; early-out on a hard bound of the interpolator's gain
  (exact, with the interpolator primed from a short input history on
  return).
- Activity statistics (clip depth as removed/input energy, share of
  touched high-rate samples) for the clip budget and for stages before the
  clipper.

**Depth (both parts off by default; on in the broadcast chain).** The
clippings are scaled by a depth before they are subtracted.
- Content-adaptive depth: less clipping on more predictable material, the
  idea of expired US 4,241,266 (*expired; status not legally reviewed*).
  One control value per 64-frame block from the linear-prediction gain
  (`src/predictability.rs`, below); thresholds 2 and 5 dB chosen on the
  test corpus. No masking model, no scoring of alternatives.
- Bandwidth guard: no clippings while the programme is band-limited
  (`src/bandwidth_guard.rs`, below). Own choice, by measurement: shaping
  the clippings below the edge with precomputed filters (the window method
  above, per edge) stopped about 50 dB within the clipper's latency, too
  little against an empty band (corpus phone clips +5.6 … +27 dB
  worst-frame NMR at `streaming`, against +0.2 … +1.7 dB with no
  clippings).
- Both decide on a fixed grid of input frames and ramp linearly (64 frames;
  the guard 15 ms), so the output does not depend on the buffer size.

**Measured** (`tests/clipper.rs`): spurs of the clipped 15 kHz tone −94.0
dBFS (standard), −75.8 dBFS (low); worst sample peak over the threshold on
band-limited signals driven 2 and 4 dB +0.77 dB (standard), +2.17 dB
(low), +0.27 dB (two passes; +1.6 dB without the compensation); a lower
NMR and
equal or higher loudness than the true-peak limiter alone and than a
base-rate hard clip at drives 0 … 4 dB. The NMR is a test metric only (own
Bark-band model after Brandenburg 1987 and Zwicker & Fastl); nothing in the
processing path uses a masking model.

### Linear-prediction gain (`src/predictability.rs`)

- Autocorrelation method of linear prediction and the Levinson–Durbin
  recursion: J. Makhoul, "Linear prediction: a tutorial review", Proc.
  IEEE 63(4), 1975; the prediction gain `r₀/E_P`.
- Recursive (exponential) autocorrelation window: T. P. Barnwell,
  "Recursive windowing for generating autocorrelation coefficients for
  LPC analysis", IEEE Trans. ASSP 29(5), 1981.
- Pre-emphasis `1 − 0.9 z⁻¹`, order 10 at an 8 kHz analysis rate, white-noise
  correction of r₀: the textbook narrowband speech LPC setup (Rabiner &
  Schafer, *Digital Processing of Speech Signals*, ch. 8).
- Own choices: analysis below 3.5 kHz (4th-order Butterworth, decimated),
  so the reading does not depend on the programme's upper band edge; a
  reflection coefficient at or over 1 (the recursive window is not always
  positive definite) reads as fully predictable (a sum of tones).

### Bandwidth guard (`src/bandwidth_guard.rs`)

- Short-time band energies from a filter bank (4th-order Butterworth
  high-passes at 4, 8 and 12 kHz) against the full band, one-pole
  averages per block: textbook spectral analysis (Oppenheim & Schafer;
  Zölzer, *DAFX*, filter banks). Hysteresis thresholds (−40 / −46 dB),
  the faster fall (0.25 s against 1 s) and the hold in quiet blocks are
  own choices, set on the test corpus (see the module docs).

### Clip budget governor (`src/governor.rs`)

**References.**
- Slow integrating loops with conditional integration (anti-windup),
  freeze and hysteresis: Åström & Murray, *Feedback Systems*; Åström &
  Hägglund, *PID Controllers*, ch. 3 (as for the output-referenced trim).
- Allowing more clipping on some material than on other, as a control
  input rather than a fixed setting, is in expired US 4,241,266. *Expired;
  status not legally reviewed.* The governor uses only the general idea of
  a budget; its measures and law are our own.

**Own design choices.**
- One operator number, the clip budget in dB (0 … 6), mapped to three
  allowances: clip depth `−40 + 5·B` dB, limiter mean gain reduction
  `B/2` dB, touched share `0.5 %·2^(B/2)`. The largest "budget used" of the
  three, over a gated 3 s window, is compared with `B`.
- Law (a product decision: loudness compliance comes first): the governor
  never lowers the output below the target. Over budget with the output on
  target it holds the trim from adding gain and gives back density, and
  reports the excess; below the target the trim stays free. Density (band
  thresholds lowered with compensating makeup, τ = 10 s, rate-limited) rises
  whenever the trim and the AGC cannot reach the target. A loudness deficit
  is reported only once all density is used. Frozen in pauses and below the
  gate, like the trim.
- An earlier law cut the AGC target when over budget, accepting a loudness
  deficit instead of more clipping; it cost 1.4 … 2.0 LU on our own
  percussive test programme and was replaced (documented in the module).

No masking model and no candidate gain curves are used anywhere in the
governor or the clipper.

### Broadcast chain and presets (`src/broadcast.rs`, `src/presets.rs`, element `stromdsp-broadcastprocessor`)

**The chain** is our own composition of existing parts, not a new algorithm:
input gain → spike guard (`spike_guard.rs`) → wideband AGC (`agc.rs`) →
multiband compressor (`multiband.rs`, with the per-band expander and the
bass clipper) → final clipper (`clipper.rs`) → true-peak limiter
(`truepeak.rs`), with the clip budget governor (`governor.rs`) around the
clipper and the limiter. The spike guard sits ahead of the AGC so clicks
neither pump it nor count toward the loudness it measures. Levelling comes
first, so the band dynamics work on programme at the target loudness, and
the true-peak ceiling comes last, so no later stage can push a peak over it.

The clipper's depth (content-adaptive and bandwidth guard) and the
limiter's peak control are coupled per frame (own design): the peak
control's drive is what the clipper's depth leaves, or with the clipper
off the clipper's predictability. Measured on the corpus speech and phone
clips (crackle check), each part is needed: without the peak control,
the adaptive depth or the guard the worst frame reaches +12.6, +17.8 and
+32.9 dB.

With the clipper taking the peaks, the limiter's default lookahead dropped
from 5 ms to 1.5 ms, justified by measurement: the true-peak ceiling holds
within +0.008 dB on the nasty-signal set at 44.1, 48 and 96 kHz, and the
limiter's mean gain reduction on the hot peaky test programme is lower than
before (0.82 dB against 1.68 dB with no clipper and 5 ms). A ceiling change
glides at 1 dB per 512 frames, so the shorter attack ramp does not make
a large jump audible.

- **Latency modes** (own design, following the element guidelines on
  latency budgets): `LatencyMode::Standard` (spike guard 0.5 ms, bass
  clipper 1 ms, standard clipper, limiter 1.5 ms; budget 6 ms) and
  `LatencyMode::Low` (none, none, low clipper, 0.6 ms; budget 3 ms). Each
  setting's cost is measured in the tests (overshoot, gain reduction,
  aliasing, onsets); the low limiter lookahead is 0.6 ms because 0.75 ms
  put 44.1 kHz over the 3 ms budget.
- **Speech preset without the final clipper** (own choice, measured): at
  −18 LUFS the clipper clipped speech peaks, worst-frame NMR up to +35 dB
  on phone speech (+2.7 dB without it), so the `speech` preset leaves the
  peaks to the limiter.
- **Crackle acceptance check** (own test design, `strom-dsp-harness`
  `crackle`): worst per-frame NMR (the harness's own NMR model, see
  Measurement harness) under 0 dB over the corpus speech and phone clips
  for every preset and latency mode, known failures listed with their
  measured cause.
- **IM feedback wiring:** the final clipper's touched share is handed to
  the bass clipper once per 512 frames (one chunk late, since the clipper
  runs after the multiband).
- **Internal chunk grid:** the chain cuts its work into 512-frame chunks
  on a fixed grid from configure/reset, so per-chunk work (the ceiling
  glide, the IM hand-over) does not depend on buffer sizes.

Own choices:
- **Input gain:** glides in dB through two cascaded 5 ms one-poles.
- **Bypass:** each stage's bypass, and the whole-chain `enabled`, is a 10 ms
  raised-cosine crossfade between the stage output and its input delayed by
  exactly the stage's latency. Latency therefore never depends on bypass
  state, and a settled bypass is bit-exact.
- **Bypassed stages keep running.**
- **Output-referenced trim:** the multiband and the limiter after the AGC
  take level away (most on peaky programme), so a slow integral loop on the
  gated output loudness (the AGC's own gating and freeze rules, on the
  output detector) adds a trim to the AGC target. Textbook integral control
  with conditional-integration anti-windup (Åström & Hägglund, *PID
  Controllers*): τ = 8 s, slew ≤ 0.5 dB/s and ≤ half the slower AGC rate,
  range ±6 dB, frozen in pauses, while the AGC ramps or is bypassed, and at
  the AGC's gain bounds, which still apply to the total. Stability margins
  (phase ≥ 79°, gain ≥ 18 dB, against the 3 s window as a 1.5 s delay) are
  derived in `broadcast.rs`.
- **Meters:**
  - momentary and short-term loudness from the BS.1770 `LoudnessDetector`;
  - integrated loudness from the `ebur128`-based `LoudnessMeter` in histogram
    mode;
  - true peaks from the 4× `Oversampler`.

**Presets** are our own.
- **Targets and ceilings** come from EBU R 128 (−23 LUFS / −1 dBTP), ATSC A/85
  (−24 LKFS / −2 dBTP) and the AES TD1004/TD1008 streaming range (−16 LUFS;
  −18 LUFS / −1 dBTP for speech, our own choice inside that range).
- **Dynamics settings** are textbook: low ratios, band thresholds relative to
  the target, and attack/release of at least one or ten periods of the
  band's lowest frequency (Giannoulis et al. 2012; Zölzer, *DAFX*).
- **Crossover choices:** a 200 Hz bass split, and 3 kHz at the ear's most
  sensitive region (ISO 226).
- **Expander and bass clipper in the chain** (own values): every band's
  expander threshold at target − 24 dB (speech − 22 dB), ratio 1.5 (speech
  2), floor 4 dB (speech 6 dB), hold 100 ms, knee 6 dB, open 3 ms, close
  150 ms; the bass clip threshold at target + 9 dB. The bass lookahead is
  1 ms (standard) or none (low latency): measured, a longer one buys little
  because the analytic envelope reaches an onset later than a few ms of
  lookahead can cover. Reasoning: studio
  noise levels relative to R 68 / R 128 alignment and speech timing, the
  same textbook sources as the gate's defaults; the floor is kept small so
  a misjudged threshold can do little on quiet music.

The values were checked with the chain acceptance test in
`tests/broadcast.rs` (60 s programme at −35, −23 and −10 LUFS input:
integrated loudness within ±1 LU of target after the first 10 s, true peak
≤ ceiling + 0.1 dB, measured with `ebur128`), and with a hot, peaky,
sport-like programme (hits, clicks, inter-sample peaks) under the same
bounds plus no short-term overshoot after a 10 s pause.

No commercial broadcast-processor manuals, presets or tuning were used.

### Spike guard (`src/spike_guard.rs`)

Removes short, isolated peaks (clicks) ahead of the AGC, so they neither
pump the gain control nor count toward the measured loudness.

- **Detection:** impulsive-noise detection against a signal envelope
  (S. Godsill, P. Rayner, *Digital Audio Restoration*, Springer 1998): the
  4× interpolated per-frame peak (`Oversampler`, as in ITU-R BS.1770
  Annex 2) against a peak envelope (`PeakEnvelope`, 40 ms release) fed
  0.33 ms late and only with frames that are not part of an event.
- **Look-ahead gain with a band-limited control signal** (US 5,737,434,
  look-ahead clipping with a band-limited gain control signal; expired;
  status not legally reviewed): the gain is written ahead of the output
  (0.5 ms standard lookahead, none in the low-latency setting) as a dip
  with raised-cosine ramps, so it creates no harmonics.
- **Soft knee:** the event's peak goes through a static curve
  `a + (L − a)·tanh((P − a)/(L − a))` (identity below the knee `a`, never
  above the cap `L`), as in the soft-knee gain computers of Giannoulis,
  Massberg & Reiss (JAES 2012) and Zölzer (*DAFX*, dynamics chapter).
- **Reused:** `Oversampler` (detector), `PeakEnvelope`, `DelayLine` (main
  path).

Own design choices:
- **Causal event rule:** a frame over the cap starts an event at once. It
  is abandoned as programme if its core (frames within 20 dB of its peak
  and above the knee) reappears more than 0.25 ms after its start within
  the next 0.15 ms; otherwise it is a spike. Tones, noise bursts and drum
  onsets are abandoned because their peaks recur; a click is not. With the
  standard 0.5 ms lookahead the decision is made before the output reads
  any of the dip, so an abandoned event leaves the programme bit-exact;
  with no lookahead the output keeps up to 0.4 ms of the dip (an onset
  "nip").
- **Rise gate** (own refinement of the envelope detection above, not from
  the textbook): an excursion over the cap starts an event only if, in
  the base-rate input the 4× interpolator already buffers (24 frames, no
  added latency), (1) the one-sample step into its peak exceeds the cap,
  (2) the two samples before that step are at or under the cap (it rises
  from the programme level), and (3) nothing above its core comes back in
  the isolation window (0.25 … 0.4 ms after its start; at 88.2 kHz and
  above, where the buffer reaches only 0.25 ms, the last 7 buffered
  frames). A refused excursion goes into the envelope at once. Measured
  thresholds: the step at −3 dB re the cap let plosives through on corpus
  speech (+2.1 dB worst-frame NMR in the low mode), 0 and +3 dB were
  clean, and +6 dB lost harness clicks (58 of 78 caught instead of 69);
  0 dB is used. Each of the three conditions is needed (worst-frame NMR
  of the low mode on corpus speech, −9.3 dB with all three): without (1)
  +14.1 dB, without (3) +7.6 dB, without (2) at least −0.7 dB (a
  high-frequency word onset was dipped). With it the low mode needs no lookahead: corpus speech measures
  the same with the guard as without it, and the test clicks are caught
  as before.
- **Envelope hygiene:** frames up to 0.33 ms ahead of an event (the
  interpolator's ringing before a large click) and the ramp after a spike
  stay out of the envelope; an abandoned event's peak goes into it at once.
- **A gain dip, not a waveform clipper:** a 4× soft clipper whose
  clippings are band-limited back to the base rate (US 6,337,999; expired;
  status not legally reviewed) was built first and measured: on a
  single-sample click the band-limited clippings are much smaller than the
  clippings, so the peak grew back (a 1.0 click over a 0.1 tone came out
  at 0.40 against a cap of 0.25; a 10.0 click at 0.87 against 0.13). The
  gain dip lands the click on the cap and adds no harmonics.
- Bit-exact outside dips: the gain is exactly 1 there and is not applied.

### Multiband compressor (`src/multiband.rs`)

This is our own composition of existing parts, not a new algorithm. The
LR4 `Crossover` (Linkwitz, JAES 1976, all-pass compensated) feeds one
`Compressor` per band (Giannoulis et al. 2012), and the bands are then
summed.

Own choices:
- The interface is in dB/ms. Values are converted to the compressor's linear
  units: threshold and makeup as 10^(dB/20), knee factor as 10^(−W/40).
- A global drive is applied before the crossover and is not compensated at the
  output. It glides in dB through two cascaded 5 ms one-poles.
- Band-count changes go through the crossover's crossfade, and newly active
  bands start from a reset compressor.
- Default band settings are gentle: threshold −18 dB, ratio 2, knee 6 dB, with
  shorter time constants toward the high bands.

Tests check the static curve per band against the gain-computer formula, and
the analytic LR4 band magnitudes `1/(1+r⁴)` and `r⁴/(1+r⁴)` with pre-warped
`r`. No commercial broadcast-processor material was used.

### Downward expander per band (`src/dynamics/expander.rs`, `Compressor::set_expander`, `MultibandCompressor::set_expanders`)

- **Static curve:** the gain computer of Giannoulis, Massberg & Reiss
  (JAES 60(6), 2012), mirrored below its own threshold `E`: slope
  `S = ratio − 1`, the same quadratic soft knee of width `W` centred on `E`,
  and a floor (largest attenuation) as `min(F, e)`. It is added to the
  compressor's static reduction of the same detector level, so one gain
  computer has compression, a unity region and expansion, with a knee at
  each threshold. The idea of a downward expander below a threshold, with a
  hold time and a release that is slower than the attack, is the textbook
  noise gate / expander of Zölzer (ed.), *DAFX*, dynamics chapter.
- **Own choices:**
  - The expander part is smoothed separately from the compression part
    (open fast, close slowly), because an expander needs the opposite of a
    compressor's attack/release.
  - Anti-chatter is a **hold on the attenuation** (the target may rise only
    after it has stayed above the smoothed value for `hold`) plus the
    slower close time, not hysteresis. The guard test measures the gain's
    total movement per second on a level hovering at the threshold
    (`tests/multiband_expander.rs::no_chatter_at_threshold`).
  - Threshold, slope, knee and floor glide over 10 ms like the compressor's
    curve. At ratio 1 the stage turns itself off once open, so the
    compressor is bit-identical to one without it (guard:
    `ratio_one_is_bit_exact`).
  - Defaults: off (ratio 1); knee 6 dB, open 3 ms, close 150 ms, hold
    100 ms, floor 12 dB.

### All-pass phase-difference pair (`src/filter/hilbert.rs`)

- **What:** two chains of first-order all-pass sections whose phases
  differ by 90° over a 600:1 band; `√(a² + b²)` of their outputs is the
  analytic-signal magnitude, used only in a control path (bass clipper).
- **References:** S. Darlington, "Realization of a constant phase
  difference", Bell System Technical Journal 29, 1950, and the general
  all-pass phase-difference (Hilbert pair) literature used for Weaver
  single-sideband modulation: alternate the poles of one increasing
  sequence between the two chains. Bilinear transform with pre-warping
  (Oppenheim & Schafer).
- **Own design:** the pole positions were found by our own numerical
  minimax optimisation (Gauss–Newton steps with Lawson-style reweighting,
  i.e. iteratively reweighted least squares, written for this project) over
  the phase error in `ln` of the pre-warped frequency: 8 + 8 sections,
  0.0091° maximum error over 600:1. The module docs give the derivation of
  why equal spacing is not enough (error at the ends of the sequence decays
  only like `e^{−distance}`). Darlington's elliptic-function solution was not
  used.

### Bass clipper (`src/bassclip.rs`, on band 0 of the multiband)

- **Envelope:** the analytic-signal magnitude of band 0 from the all-pass
  pair above (expired US 4,495,643: gain control of a low band from its
  envelope, without generating harmonics — expired; status not legally
  reviewed). For a steady sine it is flat, so the gain is constant.
- **Gain:** the limiter's gain core reused (sliding minimum over the
  lookahead, cascaded-box ramp of the lookahead's length, one-pole release;
  see the limiter entry), with attack equal to the lookahead and no hold:
  look-ahead gain control with a band-limited gain signal (expired
  US 5,737,434 — expired; status not legally reviewed). Required gain
  `min(1, T/env)`.
- **IM feedback:** a slow value lowers the bass threshold by at most 3 dB
  while bass peaks (envelope within 6 dB of the threshold) coincide with
  activity of the upper clipper, and releases it (1 s) otherwise (expired
  US 5,737,432 — expired; status not legally reviewed). Own choices: the
  activity input is the share of samples the upper clipper touched per
  block, with the full reduction at 1 %; attack 30 ms.
- **Own choices:** release 50 ms (the analytic envelope is flat, so a
  short release does not ripple; with the rectified waveform it would —
  that is the guard test), lookahead set by the host (1 ms or none in the
  broadcast processor), a gain ramp of at least 0.5 ms (with a shorter
  lookahead the gain arrives late by the difference rather than stepping),
  threshold changes gliding through two cascaded 5 ms one-poles (as the
  chain's input gain), raised-cosine enable fade of 10 ms over a constant
  delay, control path stopped while off.
- **Tests:** `tests/bass_clipper.rs` (harmonics with a rectified-envelope
  guard, IM sidebands with and without feedback against a stand-in upper
  clipper, band alignment, click-free changes at 1 ms and with no
  lookahead, the pinned cost of 0, 1 and 3 ms, no allocation),
  `tests/broadcast_expander_bass.rs` (IM sidebands end to end through the
  real final clipper) and `tests/rtsan_multiband_expander_bass.rs`.

### Butterworth filters and Linkwitz-Riley crossover (`src/filter*`)

- **Butterworth design** (Oppenheim & Schafer, *Discrete-Time Signal
  Processing*; J. O. Smith III, *Introduction to Digital Filters*).
  - Analog prototype poles sit at ψ_k = (2k+1)π/(2n) from the imaginary axis.
    Each conjugate pair gives `s² + s/Q_k + 1` with `Q_k = 1/(2 sin ψ_k)`; odd
    orders add a real pole.
  - The bilinear transform is pre-warped with `K = tan(π fc/fs)`, so −3.01 dB
    lands on fc at any rate.
  - Pair sections are algebraically identical to the RBJ LPF/HPF at Q_k, so
    the biquad designs and SIMD kernels are reused.
- **Crossover** (S. Linkwitz, JAES 24(1), 1976; LR2n = Butterworth n squared).
  - From `B(s)B(−s) = 1 + (−1)ⁿs²ⁿ`: `LP² + HP² = B(−s)/B(s)` for n = 2, 4,
    and `LP² − HP² = B(−s)/B(s)` for n = 1.
  - `B(−s)/B(s)` is an all-pass: one cookbook APF per pole pair, plus
    `(1−s)/(1+s)`.
  - The topology (own choice) is a cascade of high-passes, where band s is
    `LP_s` of the remainder. Each band then passes through the all-passes of
    every higher split, so the band sum is the product of all split all-passes
    (flat, zero latency).
  - Own choice: the LR2 high-pass is defined as `−HP²`, so the plain band sum
    is all-pass for every order.
  - Frequency changes glide as in the EQ. Order and band-count changes
    crossfade over 10 ms.

### Oversampler (`src/oversample.rs`)

- **Structure:** a cascade of 2× half-band FIR stages in polyphase form
  (Crochiere & Rabiner, *Multirate Digital Signal Processing*, ch. 4–5; Lyons,
  *Understanding DSP*, half-band filters).
- **Filter design:** each stage is a windowed sinc with a Kaiser window
  (Oppenheim & Schafer §7.5–7.6), with I0 computed from its power series.
- **Own design choices:**
  - The passband edge is 20/48 of the base rate.
  - Side taps are normalised so H(0) = 1 and H(½) = 0.
  - Tap counts and β (20/10.45, 8/10.40, 5/11.15) were found by scanning for a
    stopband of at least 95 dB. Later stages are designed only against the
    images the earlier stages leave.
  - The pure-delay polyphase branch is chosen per stage so that the up and down
    delays are whole base-rate frames.
- **Measured:** passband error ≤ 0.0001 dB, images and aliases ≤ −98.4 dB.

### STFT (`src/stft.rs`)

- **Method:** weighted overlap-add after J. O. Smith III, *Spectral Audio
  Signal Processing* (STFT, OLA, the COLA condition).
- **Windows:** analysis and synthesis are both √(periodic Hann). Their product
  satisfies COLA for hops N/2 and N/4, with sum N/(2H).
- **Normalisation:** that sum and the 1/N inverse-FFT scale are folded into
  the synthesis window.
- **Latency (own analysis):** N − 1 samples for arbitrary buffer sizes. The
  often-quoted N − hop only holds for hop-aligned blocks; an impulse test
  verifies N − 1 exactly.

### Meters (`src/meter/`)

- **`LoudnessMeter`** wraps `ebur128`, always in histogram mode.
  - Why histogram mode: the default queue mode reallocates after 500 s and
    allocates on every LRA read (verified with a counting allocator).
    Histogram mode is allocation-free on our path; verified with the counting
    allocator and RTSan.
  - Channel weights follow ITU-R BS.1770: 1.0 front, 1.41 surround, LFE
    excluded (EBU Tech 3341 §2.10). Own choice: an unknown layout weights
    every channel 1.0.
  - Tests generate EBU Tech 3341 (v4, 2023) cases 1–6 and 9–19, and Tech 3342
    (2023) cases 1–4, in code from the published descriptions. No test files
    were downloaded.
- **`CorrelationMeter`:** normalised cross-correlation at lag 0, with one-pole
  averages. The side/mid ratio is derived from the same averages. Own choice:
  r = 0 when either channel is below −100 dBFS RMS.
- **`MonoDetector`:** own design. Two-threshold hysteresis on the S/M ratio and
  the correlation, a hold time, and freezing below a silence level (the same
  idea as the gate, Zölzer, *DAFX*).

### Sidechain, expander mode and gate detector (`dynamics/key.rs`, `gate.rs`)

- **`KeyDetector`:** an optional key filter (RBJ HPF, or BPF with 0 dB peak,
  using the `Biquad`), then the linked peak over the key's channels.
  The detector-path structure follows Giannoulis et al. 2012. Own choice: a
  key with the wrong channel count, or too short, reads as silence (fail-safe:
  nothing is processed).
- **Gate detector (fix for chatter found by ear on speech in studio
  noise):** `MeanSquareEnvelope` is a
  one-pole average of x² with τ = 10 ms (Zölzer, *DAFX*: RMS level
  measurement).
  - Own calibration: ×2 (+3.01 dB), so a steady sine reads its peak level.
  - Own choice: a 1 dB minimum hysteresis.
  - These replace the instant-attack peak follower, which let single noise
    peaks open the gate when open == close. The guard test is
    `tests/gate_chatter.rs`.
- **Gate release (fix for a gate that stayed open in short speech pauses,
  found by ear):** `dynamics::GateFade`.
  - Attack stays a one-pole on the linear gain.
  - Release is the **time to close**: a straight line in dB from 0 dB to
    `reduction` in exactly `release` ms, i.e. a constant per-sample gain
    factor `10^(−range/(20·N))` with range at least 1 dB.
  - This follows the common operator definition of gate release, a full-range
    traversal time.
  - Why: with the earlier one-pole release, the gate needed ≈ 6.9 τ to reach
    −60 dB and never closed in short pauses.
  - Closing time after the key stops is `t_detector + hold + release`.
  - Guard test: `tests/gate_chatter.rs::gate_closes_within_hold_plus_release_after_burst`
    (the old code took 1462 ms against a 140 ms budget).
  - The earlier test `release_time_constant` was replaced by
    `release_closes_over_full_range_in_release_time`, because it measured the
    old definition.
- **Expander mode:** a downward expander after Zölzer, *DAFX*. Below
  T = open threshold, the static gain is `max(D, (R−1)(x−T))` dB. Own choice:
  hysteresis and hold act as in the gate, and only the closed target changes.
  The element exposes it as `mode`/`ratio`.
- **Interface tests:** `tests/interface.rs` asserts the exact set of gate
  properties and allows only `mode` and `ratio`, which were added later, as
  extra ones. Full checks of those two are in `tests/interface_gate_1b.rs`.
- **Test signals:** pink noise is generated with the Voss method (M. Gardner,
  *Scientific American*, 1978).

### `BandCorrelation` (`src/meter/band.rs`)

- **Structure:** our own composition of existing parts. The LR4 `Crossover`
  (Linkwitz 1976) is configured once for L and R, so both get identical
  filters, and feeds one `CorrelationMeter` per band. Because L and R are
  filtered identically, the filter phase cancels in the correlation.
- **Own choice:** default splits at 250 Hz and 4 kHz.

### Ducker (`src/ducker.rs`, `src/resample.rs`, element `stromdsp-ducker`)

- **Detector:** the existing sidechain blocks (`KeyDetector` with an RBJ HPF;
  `MeanSquareEnvelope`, a J. O. Smith III one-pole averager and the RMS level
  measurement of Zölzer, *DAFX*), run at the key's own rate.
- **Rate conversion (own design):** the detector envelope is converted to the
  programme rate, not the key audio. It is linearly interpolated across
  64-frame programme blocks (J. O. Smith III, linear interpolation).
  - Why: a 10 ms RMS envelope has about 16 Hz of bandwidth. A band-limited
    resampler would add cost, group delay and chunking constraints for no
    gain.
  - `rubato` was therefore dropped as a dependency.
- **Static curve** (Zölzer, *DAFX*: ducking as a keyed downward gain). Own
  definitions:
  - `fixed`: depth applied while the key is above the threshold.
  - `proportional`: 1 dB of reduction per dB above the threshold, capped at
    the depth.
- **Hold (own design):** a peak hold with a timer, as in peak-programme meters.
- **Fades (own choice):** linear in dB at constant rates `max(depth, 1 dB)/attack`
  and `/release`, so attack and release are fade durations.
- **Element:** written from the gstreamer-rs `gstreamer-base` Aggregator
  bindings (MIT/Apache-2.0) and the GstAggregator API documentation
  (`next_time`, `ignore-inactive-pads`, `update_segment`,
  `AggregatorPad::is_inactive`).
  - Own design: the key FIFO (2 s), its running-time alignment (1 ms jitter
    tolerance) and the live deadline logic.
  - No GStreamer C source was read.

### De-esser (`src/deesser.rs`)

- **Composition:** built only from existing blocks: `KeyDetector`,
  `GainComputer` and `AttackRelease` (Giannoulis, Massberg & Reiss, JAES 2012),
  and the LR4 `Crossover` (Linkwitz, JAES 1976).
- **Structure:** a sibilance-band detector with split-band or wideband gain,
  after the textbook description in Zölzer, *DAFX*, dynamics chapter.
- **Own choices:**
  - slope 1 (∞:1) with a fixed 6 dB knee, capped at `max-reduction`;
  - a 12 dB/octave detector high-pass at the split frequency;
  - both modes share the all-pass split, so mode changes glide (10 ms mix
    factor), at the cost of wideband output being the crossover's all-pass of
    the input;
  - 10 ms glides for threshold and max-reduction.

### Input cleanup (`src/cleanup.rs`)

- **DC blocker:** the order-1 Butterworth high-pass at 5 Hz. It equals the
  one-zero/one-pole DC blocker (J. O. Smith III, *Introduction to Digital
  Filters*, "DC blocker"; Oppenheim & Schafer, bilinear transform).
- **High-pass:** `filter::Butterworth`.
- **Hum notches:** R. Bristow-Johnson, "Audio EQ Cookbook" (W3C Note), notch
  filter. The relation between α and the −3 dB bandwidth, `tan(Δω/2) = α`, is
  the standard second-order notch written as `(1 + A(z))/2` with a
  second-order all-pass (P. A. Regalia & S. K. Mitra, IEEE Trans. ASSP 35(1),
  1987).
- **Own choices:**
  - constant bandwidth f0/`hum-q` for every harmonic, instead of constant Q;
  - a `hum-q` lower limit of 6, from an analytic flatness computation (worst
    case 0.40 dB at hum-q 6; 0.57 dB at hum-q 5 fails the 0.5 dB target);
  - harmonics at or above 0.49·fs are skipped;
  - stages that are off are skipped, with 10 ms dry/wet crossfades when
    switching and two-bank crossfades for hum changes.

### Delay and stereo (`src/delay.rs`, `src/stereo.rs`)

- **Delay:** an integer-frame read tap on the shared `DelayLine` circular
  buffer (J. O. Smith III, *Physical Audio Signal Processing*, delay lines).
  - Tap changes use an equal-gain raised-cosine crossfade between two taps
    (Zölzer, *DAFX*, delay-based effects; Oppenheim & Schafer, window
    functions).
  - The step bound `D + Δg·2A` with `Δg ≤ π/(2(N+1))` is our own derivation.
  - The delay is intentional and never reported as latency.
  - Own choices: 20 ms crossfade; the latest target is queued behind a running
    crossfade; negative offsets are allowed; at most 64 offsets;
    `enabled=false` crossfades to delay 0.
- **Stereo:** textbook 2×2 linear matrixing.
  - M/S as `M = (L+R)/2`, `S = (L−R)/2`, with inverse `L = M+S`, `R = M−S`
    (sum/difference stereo, Blumlein; e.g. Rumsey, *Spatial Audio*).
  - Width as side scaling.
  - A sine/cosine constant-power pan law (Rumsey; Pulkki's amplitude-panning
    literature), normalised to unity at the centre.
  - Mono sum at −3 or −6 dB.
  - Own choices: polarity applied to the inputs; the order B·W·M·P; a linear
    10 ms matrix ramp, which equals a crossfade because the processor is
    linear; signed-permutation matrices applied as copies/negations for
    bit-exactness.
  - The element was scaffolded as `stromdsp-stereotools` and renamed to the
    shorter, more generic `stromdsp-stereo`.

### Meter element (`stromdsp-meter`)

- **Sources:** written from the gstreamer-rs documentation and the GStreamer
  API docs (BaseTransform passthrough, element messages,
  `GstAudioChannelPosition`).
- **Message format:** our own design. GStreamer's `ebur128level` was not read.
- **Channel weights:** the BS.1770 role mapping of channel positions follows
  the azimuth zones of ITU-R BS.1770 and EBU Tech 3341 §2.10.
- **RMS:** a plain mean square over the interval.
- **Silence:** −200.0 (never NaN or −∞).
- **Cost control:** loudness readings, which are expensive (short-term about
  150 µs), happen only when a message is due.

### GStreamer elements (`crates/gst-plugin-stromdsp`)

Written from the gstreamer-rs bindings documentation and crate sources
(MIT/Apache-2.0) and the GStreamer API docs.

- Each element keeps its settings as a `Copy` struct behind a `Mutex`, plus an
  atomic generation counter. Once per buffer, the streaming thread compares
  generations (one atomic load) and copies the settings only if they changed,
  using `try_lock`. If a setter holds the lock, the change is picked up on the
  next buffer, so the streaming thread never blocks on the application thread.
- `set_caps` reads `rate` and `channels` from the caps (not `AudioInfo`, which
  rejects more than 2 channels without a channel mask) and reconfigures the
  processor; this is the only place that allocates. DISCONT buffers and
  FLUSH_STOP call `reset()`.
- Samples are converted from F32LE in fixed-size chunks through a scratch
  buffer, so no `unsafe` code and no alignment assumptions are needed.
- Toggling `enabled` fades instead of switching (own design). Gate,
  compressor, EQ, de-esser, cleanup and ducker share a raised-cosine
  crossfade between the processed and the dry signal (the raised cosine is
  the textbook Hann window shape, Oppenheim & Schafer); once fully dry, the
  BaseTransform elements switch to passthrough and the ducker copies the
  programme. Delay and stereo fade through their own parameter ramps (to
  delay 0, or to the identity matrix) and end in a bit-exact copy. The
  limiter and the broadcast processor are never passthrough: they keep
  their lookahead delay, report it in the LATENCY query, and post a Latency
  message when it changes.

### Element defaults (`crates/strom-dsp/src/presets.rs`)

The property defaults of every element are strom-dsp's own choice, including
the decision to keep `stromdsp-equalizer`, `-cleanup`, `-delay` and `-stereo`
neutral (bit-exact until an operator changes them) and the broadcast presets
(see "Broadcast chain and presets"). They are chosen for live broadcast
speech and programme from textbook and common broadcast practice, never from
any product's presets. Each value and its reason is documented on the
element's page under [docs/elements/](docs/elements/README.md)
("Why these defaults"), and for the gate, compressor, limiter and broadcast
presets also in `presets.rs`. The references are:

- EBU R68, the −18 dBFS alignment level. This sets the compressor threshold
  and the level assumptions for the gate.
- EBU R128 and ITU-R BS.1770: −23 LUFS programme loudness and −1 dBTP maximum
  true peak. These set the limiter ceiling and where speech sits relative to
  the compressor threshold.
- Giannoulis, Massberg & Reiss, JAES 60(6), 2012, and Zölzer, *DAFX* (dynamics
  chapter): the soft knee, the RMS detector for programme levelling, and time
  constants that must be long compared with the lowest signal period.
- General speech properties: syllable rate about 4–5 per second, consonant
  onsets of a few ms, and intra-word gaps under about 100 ms. These set the
  attack, hold and release times.

The library's `*Params::default()` values are unchanged, because the theory
tests are written against them. The elements start from
`GateParams::broadcast()`, `CompressorParams::broadcast()`,
`LimiterParams::broadcast()` and the `*_BROADCAST_*` constants for the
setter-based properties. The ducker, de-esser, cleanup, delay, stereo and
meter elements start from their library defaults. Strom sets every property
it relies on explicitly, so these defaults do not change Strom's sound. The
compressor's `auto-makeup` default is `false`, so inserting the element never
raises the level by itself.

### Measurement harness (`crates/strom-dsp-harness`, tests only)

A test-support crate, never linked into the plugin. It runs fixed synthetic
signals through the broadcast chain per preset and compares a fixed set of
numbers with a checked-in baseline (`baseline.json`). All signals are
generated in code with fixed seeds. The programme generators of the chain
tests (`tests/broadcast.rs`, `tests/governor.rs`,
`tests/broadcast_latency.rs`) live in one place, `tests/programme_support/`,
and are shared with the harness.

- **Signals:** sine; exponential sweep (A. Farina, "Simultaneous
  measurement of impulse response and distortion with a swept-sine
  technique", AES 108th Convention, 2000: the phase law); the SMPTE
  RP 120 pair (60 Hz + 7 kHz, 4:1); the CCIF twin tone (19 + 20 kHz); a
  bass-into-highs pair (50 Hz + 3 kHz at −20 dB); a band-limited square
  wave through a first-order high-pass (tilted tops), and a hard-clipped
  sine synthesised from its Fourier series (no aliasing in the source);
  speech-like noise (band-limited pink noise under an own word and
  syllable envelope, with room tone between words), with and without
  clicks; the music-like and sport-like programmes of the chain tests.
- **Metrics:**
  - loudness, LRA and true peak through `ebur128` (ITU-R BS.1770,
    EBU Tech 3342);
  - spectra with a 4-term Blackman-Harris window (F. J. Harris, "On the use
    of windows for harmonic analysis with the discrete Fourier transform",
    Proc. IEEE 66(1), 1978);
  - SMPTE RP 120 and CCIF intermodulation, sidebands around a carrier, and
    non-harmonic spurs of a tone (aliasing);
  - clip depth (removed energy per 10 ms block relative to the programme,
    own definition) and gain reduction split into "on peaks" and "on
    programme" around known event frames (own definition);
  - the chain's own meters per case: final clipper depth, bass clipper
    gain reduction and IM feedback, expander gain reduction, governor
    budget excess, density and deficit (reported as the chain reports
    them);
  - noise-to-mask ratio after K. Brandenburg, "Evaluation of quality for
    audio encoding at low bit rates", AES 82nd Convention, 1987, with our
    own simple masking model: integer Bark bands (Zwicker & Terhardt's
    critical-band-rate formula, in Zwicker & Fastl, *Psychoacoustics*),
    per-band least-squares gain matching, a spreading function of
    −10/−25 dB per Bark and a 10 dB offset. **The masking model is a test
    metric only**; nothing in the processing or control path uses it.
    `metrics::nmr_frames` exposes the per-frame, per-band numbers for
    diagnosis (the crackle example's `diag:` mode).

### Listening kit and test corpus (`crates/strom-dsp-harness/src/listen/`, `scripts/`, `corpus/`)

Test and listening support only, never linked into the plugin.

- **Corpus.** Real recordings and generated test signals under CC0, public
  domain, CC BY 3.0 or CC BY 4.0 only. The audio ships as a release asset,
  never in git. `corpus/manifest.tsv` is the provenance record of every clip
  (source URL, licence and licence URL, attribution, extract, processing,
  SHA-256) and `corpus/ATTRIBUTION.md` the attribution the licences require;
  `scripts/make-corpus-manifest.sh` generates both from the curated corpus
  and refuses any other licence. Material whose terms forbid redistribution
  or commercial use (e.g. the EBU SQAM and loudness test sets) is not in the
  corpus. The corpus is data for listening and measurement: nothing in the
  processing is tuned to it, and it is never compared with another
  product's output.
- **Archive format** (own choice): 24-bit FLAC, lossless from the curated
  24-bit clips (checked by comparing decoded samples); the few clips with
  sample peaks above 0 dBFS stay 32-bit float WAV, because FLAC holds only
  integers and a gain to fit them would change the level the processors see.
- **Render and page** (own design): every configuration processes the clip
  as stereo at −23 LUFS (broadcast configurations at their preset target, as
  in the harness), outputs are time-aligned by removing the reported latency,
  and each set shares one listening gain. The numbers come from the harness
  metrics above. The page uses the Web Audio API (W3C) with all versions of
  a set playing in sync and a short gain ramp to switch between them.
  Encoding and decoding go through the `ffmpeg` command-line tool, run as a
  separate program (not linked).

### Performance and real-time-safety tooling

The `tests/rtsan*.rs` files use RealtimeSanitizer through `rtsan-standalone`, written
from that crate's README and API docs; a canary test that must abort proves
the sanitizer is really active. `benches/strip_gungraun.rs` uses gungraun,
written from its crate documentation. Both are optional features used only
by tests and benches, never by library code. `tests/denormals.rs` (own
design) verifies the explicit-flush policy: no subnormal output or gain state
after loud input followed by long silence at extreme parameters. FTZ/DAZ is
deliberately not set (see "Engineering policies").

### Engineering policies

- **No FTZ/DAZ.** We do not set flush-to-zero or denormals-are-zero in the
  CPU, for three reasons:
  - Rust treats changing MXCSR (x86) or FPCR (aarch64) outside inline assembly
    as undefined behaviour.
  - The FP mode is per-thread state owned by the host, so changing it on a
    GStreamer streaming thread would affect other elements.
  - It is not needed. Every recursive state is flushed to exactly zero below
    1e-15, or snapped to its target. `tests/denormals.rs` enforces this.
- **No thread pools in the streaming path.** Processing stays on the caller's
  thread, with no rayon or worker pools, for three reasons:
  - Waking and joining workers costs about as much as the microseconds of
    work per buffer, and the worst case is unbounded.
  - Real-time-priority streaming threads would wait on normal-priority
    workers (priority inversion).
  - Strom already runs each strip on its own streaming thread.
  - Speed-ups come from algorithms and SIMD instead.
- **Portable binaries.** We never build with `target-cpu=native`. SIMD levels
  are dispatched at runtime.
- **`panic = unwind`.** gstreamer-rs turns a panic in an element into an
  element error. `abort` would take down the host.

## Incident log

- **2026-11-03, research only, no code affected.** During online research of
  Rust SIMD and denormal handling, a research assistant (which wrote no
  project code) viewed `src/denormal.rs` of the crate `truce-core`. That crate
  turned out to carry a proprietary licence (`LicenseRef-TruceLicense-1.0`),
  i.e. an "unclear licence" in the sense of the clean-room rules. The file only
  sets the CPU flush-to-zero flag. Remediation: nothing from it is used. FTZ
  handling, if added, comes from the MIT-licensed `no_denormals` crate or is
  written from the Intel SDM (MXCSR) and Arm ARM (FPCR.FZ) by a contributor
  who has not seen that file. The project's correctness never relies on FTZ:
  all recursive states are flushed explicitly (see above).

## Dependencies

| Crate | Used by | Licence |
|---|---|---|
| `fearless_simd` 1.2 | `strom-dsp` (SIMD with runtime dispatch) | Apache-2.0 OR MIT |
| `ebur128` 0.1 | `strom-dsp` meter (BS.1770 loudness, gating, LRA, sample and true peak) | MIT |
| `realfft` 3.5 (with `rustfft` 6.4) | `strom-dsp` STFT (real FFT); harness spectra | MIT / MIT OR Apache-2.0 |
| `gstreamer`, `gstreamer-base`, `gstreamer-audio` 0.25 | plugin | MIT OR Apache-2.0 |
| `gst-plugin-version-helper` 0.8 (build) | plugin | MIT OR Apache-2.0 |
| `gstreamer-check` 0.25 (dev) | plugin tests | MIT OR Apache-2.0 |
| `proptest` 1 (dev) | library tests | MIT OR Apache-2.0 |
| `criterion` 0.8 (dev) | benchmark | MIT OR Apache-2.0 |
| `serde_json` 1 | measurement harness only (baseline file; already in the tree through `criterion`) | MIT OR Apache-2.0 |
| `rtsan-standalone` 0.3 (optional, feature `rtsan`) | real-time-safety tests | Apache-2.0 |
| RTSan runtime `libclang_rt.rtsan` (LLVM 20, binary downloaded at build time, test binary only) | rtsan tests | Apache-2.0 WITH LLVM-exception |
| `gungraun` =0.20.0 (optional, feature `gungraun`) | instruction-count benchmark | MIT OR Apache-2.0 |

**Why these are reused (rule: reuse a crate only for what is hard to get
right):**
- **`ebur128`:** the standard has many edge cases: K-weighting at any rate,
  75 %-overlap gating, absolute and relative gates, Tech 3342 percentiles, and
  true-peak oversampling. The crate is a mature port of libebur128 and is used
  by GStreamer's `ebur128level`.
- **`realfft`/`rustfft`:** fast, correct FFTs (mixed radix, SIMD). Only
  `process_with_scratch` runs on the real-time path; planning happens at
  configure.

Both are wrapped behind our own types and verified real-time safe on our path
with the counting allocator and RTSan.

**Considered and dropped:** `rubato` 5 (MIT OR Apache-2.0). It was declared
for resampling the ducker key, but it was removed unused, because the ducker
converts its detector envelope instead (see "Ducker"). If a later element needs
key audio, not just its level, at another rate, `rubato`'s documented
allocation-free `process_into_buffer` should be wrapped behind our own type and
verified with the counting allocator and RTSan.

External developer/CI tools, not linked or distributed: `gungraun-runner`
(MIT OR Apache-2.0), Valgrind (GPL-2.0, run as a separate program by the
gungraun CI job only), `ffmpeg` (LGPL/GPL, run as a separate program by the listening
kit scripts and render tool only) and `cargo-show-asm` (MIT OR Apache-2.0).

Transitive dependencies are checked against the allowlist in `deny.toml` by
`cargo deny check licenses`. The GStreamer C libraries (LGPL) are system
libraries linked dynamically; they are not crates and not part of this check.
