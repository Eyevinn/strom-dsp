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
  Biquad outputs are denormal-flushed below 1e-15, and output is clamped to the
  finite f32 range.

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
| (none at runtime) | `strom-dsp` | |
| `gstreamer`, `gstreamer-base`, `gstreamer-audio` 0.25 | plugin | MIT OR Apache-2.0 |
| `gst-plugin-version-helper` 0.8 (build) | plugin | MIT OR Apache-2.0 |
| `gstreamer-check` 0.25 (dev) | plugin tests | MIT OR Apache-2.0 |
| `proptest` 1 (dev) | library tests | MIT OR Apache-2.0 |
| `criterion` 0.8 (dev) | benchmark | MIT OR Apache-2.0 |

Transitive dependencies are checked against the allowlist in `deny.toml` by
`cargo deny check licenses`. The GStreamer C libraries (LGPL) are system
libraries linked dynamically; they are not crates and not part of this check.
