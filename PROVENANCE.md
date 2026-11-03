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
- **No comparison** of output, behaviour, performance or code with the plugin
  this project replaces. Correctness is verified against theory (analytic gain
  curves, transfer functions, time constants) and invariants only.
- The only thing taken from the replaced plugin is the *interface* Strom
  relies on: property names, GLib types, units, ranges and defaults.

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

Linked downward noise gate with hysteresis and hold (spec 3.2), from the
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
- **Superseded after listening tests (phase 1b):** two parts of this section
  were replaced, see "Sidechain, expander mode and gate detector" below:
  - the peak detector, now an RMS detector with a 1 dB minimum hysteresis;
  - the one-pole release, now a full-range dB ramp (`GateFade`).

### Compressor (`crates/strom-dsp/src/compressor.rs`)

Feed-forward downward compressor following D. Giannoulis, M. Massberg,
J. D. Reiss, "Digital Dynamic Range Compressor Design — A Tutorial and
Analysis", JAES 60(6), 2012.

- **Detector:** `x = 20·log10(max |x| over channels)` (peak, −200 dB floor).
- **Gain computer** (knee per spec 3.3): `T = 20·log10(threshold)`,
  `k ∈ [0.001, 1]`, `W = −40·log10(k)`, `S = 1 − 1/ratio`. The gain reduction
  is 0 below `T − W/2`, `S·(x − T + W/2)²/(2W)` inside the knee (the paper's
  quadratic interpolation), and `S·(x − T)` above it. `W = 0` is a hard knee.
- **Smoothing:** a branching one-pole on the gain reduction in dB (paper's
  "smooth branching" variant): the attack coefficient while the reduction
  rises, release otherwise. Time constant = 63.2 % of a dB step after τ.
- **Output:** `10^(−y/20) · makeup`. Ratio 1 gives S = 0, so it is bit-exact unity.
- **Parameter smoothing (own choice):** threshold (dB), S, W and makeup glide
  with a 10 ms one-pole; values snap on the first block after
  `configure`/`reset`. States are f64 and denormal-flushed.

**Changes after listening tests (phase 1b):**

- **Detector peak hold (bass distortion fix).** At fast attack/release (1 ms /
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
- **Auto-makeup.** Makeup equals the static gain reduction at a reference level
  (Giannoulis et al. 2012, automatic make-up gain from the gain computer).
  - Own choice: the reference is −18 dB sine-calibrated (the EBU R68 alignment
    level) instead of 0 dBFS. It glides with the existing 10 ms makeup smoother.
  - Test: integrated loudness stays within ±1.5 LU over thresholds −30 … −10 dB
    and ratios 2 … 10, measured with `meter::LoudnessMeter`.


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
  first principles (see `docs/PERFORMANCE.md`).

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
| `GainComputer` | static soft-knee curve, knee per spec 3.3 | Giannoulis et al. 2012 |
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

### Butterworth filters and Linkwitz-Riley crossover (`src/filter*`, phase 1b)

- **Butterworth design** (Oppenheim & Schafer, *Discrete-Time Signal
  Processing*; J. O. Smith III, *Introduction to Digital Filters*).
  - Analog prototype poles sit at ψ_k = (2k+1)π/(2n) from the imaginary axis.
    Each conjugate pair gives `s² + s/Q_k + 1` with `Q_k = 1/(2 sin ψ_k)`; odd
    orders add a real pole.
  - The bilinear transform is pre-warped with `K = tan(π fc/fs)`, so −3.01 dB
    lands on fc at any rate.
  - Pair sections are algebraically identical to the RBJ LPF/HPF at Q_k, so
    the phase 1 biquad designs and SIMD kernels are reused.
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

### Oversampler (`src/oversample.rs`, phase 1b)

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

### STFT (`src/stft.rs`, phase 1b)

- **Method:** weighted overlap-add after J. O. Smith III, *Spectral Audio
  Signal Processing* (STFT, OLA, the COLA condition).
- **Windows:** analysis and synthesis are both √(periodic Hann). Their product
  satisfies COLA for hops N/2 and N/4, with sum N/(2H).
- **Normalisation:** that sum and the 1/N inverse-FFT scale are folded into
  the synthesis window.
- **Latency (own analysis):** N − 1 samples for arbitrary buffer sizes. The
  spec says N − hop, which only holds for hop-aligned blocks; an impulse test
  verifies N − 1 exactly.

### Meters (`src/meter/`, phase 1b)

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

### Sidechain, expander mode and gate detector (`dynamics/key.rs`, `gate.rs`, phase 1b)

- **`KeyDetector`:** an optional key filter (RBJ HPF, or BPF with 0 dB peak,
  using the phase 1 `Biquad`), then the linked peak over the key's channels.
  The detector-path structure follows Giannoulis et al. 2012. Own choice: a
  key with the wrong channel count, or too short, reads as silence (fail-safe:
  nothing is processed).
- **Gate detector (fix after listening tests):** `MeanSquareEnvelope` is a
  one-pole average of x² with τ = 10 ms (Zölzer, *DAFX*: RMS level
  measurement).
  - Own calibration: ×2 (+3.01 dB), so a steady sine reads its peak level.
  - Own choice: a 1 dB minimum hysteresis.
  - These replace the instant-attack peak follower, which let single noise
    peaks open the gate when open == close. The guard test is
    `tests/gate_chatter.rs`.
- **Gate release (second fix after listening tests):** `dynamics::GateFade`.
  - Attack stays a one-pole on the linear gain.
  - Release is the **time to close**: a straight line in dB from 0 dB to
    `reduction` in exactly `release` ms, i.e. a constant per-sample gain
    factor `10^(−range/(20·N))` with range at least 1 dB.
  - This follows the common operator definition of gate release, a full-range
    traversal time.
  - Why: with the phase 1 one-pole release, the gate needed ≈ 6.9 τ to reach
    −60 dB and never closed in short pauses.
  - Closing time after the key stops is `t_detector + hold + release`.
  - Guard test: `tests/gate_chatter.rs::gate_closes_within_hold_plus_release_after_burst`
    (the old code took 1462 ms against a 140 ms budget).
  - The phase 1 test `release_time_constant` was replaced by
    `release_closes_over_full_range_in_release_time`, because it measured the
    old definition.
- **Expander mode:** a downward expander after Zölzer, *DAFX*. Below
  T = open threshold, the static gain is `max(D, (R−1)(x−T))` dB. Own choice:
  hysteresis and hold act as in the gate, and only the closed target changes.
  The element exposes it as `mode`/`ratio`.
- **Interface-test deviation:** phase 1 `tests/interface.rs` asserts the exact
  set of gate properties, so the additive `mode`/`ratio` would fail it by
  construction.
  - Minimal change: every phase 1 assertion is kept, and only these two names
    are allowed as extra gate properties.
  - Full checks of the two new properties are in `tests/interface_gate_1b.rs`.
- **Test signals:** pink noise is generated with the Voss method (M. Gardner,
  *Scientific American*, 1978).

### `BandCorrelation` (`src/meter/band.rs`, phase 1b)

- **Structure:** our own composition of existing parts. The LR4 `Crossover`
  (Linkwitz 1976) is configured once for L and R, so both get identical
  filters, and feeds one `CorrelationMeter` per band. Because L and R are
  filtered identically, the filter phase cancels in the correlation.
- **Own choice:** default splits at 250 Hz and 4 kHz.

### Ducker (`src/ducker.rs`, `src/resample.rs`, element `stromdsp-ducker`, phase 1b)

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

### De-esser (`src/deesser.rs`, phase 1b)

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

### Input cleanup (`src/cleanup.rs`, phase 1b)

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

### Delay and stereo (`src/delay.rs`, `src/stereo.rs`, phase 1b)

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
  - The element was scaffolded as `stromdsp-stereotools` and renamed to
    `stromdsp-stereo`, because the old name was too close to a commercial
    broadcast product.

### Meter element (`stromdsp-meter`, phase 1b)

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
- `enabled = false` uses BaseTransform passthrough (gate, compressor, EQ). The
  limiter is never passthrough: it keeps its lookahead delay, adds the
  lookahead to upstream's min/max in the LATENCY query, and posts a Latency
  message when the lookahead changes.

### Performance and real-time-safety tooling

`tests/rtsan.rs` uses RealtimeSanitizer through `rtsan-standalone`, written
from that crate's README and API docs; a canary test that must abort proves
the sanitizer is really active. `benches/strip_gungraun.rs` uses gungraun,
written from its crate documentation. Both are optional features used only
by tests and benches, never by library code. `tests/denormals.rs` (own
design) verifies the explicit-flush policy: no subnormal output or gain state
after loud input followed by long silence at extreme parameters. FTZ/DAZ is
deliberately not set (see `docs/PERFORMANCE.md`).

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
| `realfft` 3.5 (with `rustfft` 6.4) | `strom-dsp` STFT (real FFT) | MIT / MIT OR Apache-2.0 |
| `gstreamer`, `gstreamer-base`, `gstreamer-audio` 0.25 | plugin | MIT OR Apache-2.0 |
| `gst-plugin-version-helper` 0.8 (build) | plugin | MIT OR Apache-2.0 |
| `gstreamer-check` 0.25 (dev) | plugin tests | MIT OR Apache-2.0 |
| `proptest` 1 (dev) | library tests | MIT OR Apache-2.0 |
| `criterion` 0.8 (dev) | benchmark | MIT OR Apache-2.0 |
| `rtsan-standalone` 0.3 (optional, feature `rtsan`) | real-time-safety tests | Apache-2.0 |
| RTSan runtime `libclang_rt.rtsan` (LLVM 20, binary downloaded at build time, test binary only) | rtsan tests | Apache-2.0 WITH LLVM-exception |
| `gungraun` =0.20.0 (optional, feature `gungraun`) | instruction-count benchmark | MIT OR Apache-2.0 |

**Why these are reused (phase 1b rule: hard to get right):**
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
gungraun CI job only) and `cargo-show-asm` (MIT OR Apache-2.0).

Transitive dependencies are checked against the allowlist in `deny.toml` by
`cargo deny check licenses`. The GStreamer C libraries (LGPL) are system
libraries linked dynamically; they are not crates and not part of this check.
