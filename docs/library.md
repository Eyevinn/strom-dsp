# Using the library directly

**`strom-dsp`** (`crates/strom-dsp`) is the DSP library behind the GStreamer
plugin. It has no GStreamer dependency and builds with Rust 1.89.

## The `Processor` contract

- Processors work on interleaved `f32` in place, with any channel count and
  8–192 kHz.
- `configure(sample_rate, channels)` allocates: call it off the audio thread.
- After `configure`, processing is real-time safe: no allocation, locking,
  logging or I/O.
- SIMD is used with runtime dispatch (NEON, SSE4.2/AVX2/AVX-512), so one
  binary runs on every CPU of its architecture.

```rust
use strom_dsp::{Compressor, CompressorParams, Processor};

let mut comp = Compressor::new();
comp.configure(48_000.0, 2); // allocates; do this off the audio thread
comp.set_params(&CompressorParams { threshold: 0.25, ratio: 4.0, ..Default::default() });
comp.process(&mut interleaved_f32); // real-time safe
```

How real-time safety is tested (counting allocator, RealtimeSanitizer,
denormals) is in [PERFORMANCE.md](PERFORMANCE.md#real-time-safety). The
references behind each algorithm, the SIMD design and the denormal policy are
in [PROVENANCE.md](../PROVENANCE.md).

## Modules and building blocks

Reusable building blocks are public too:
- `strom_dsp::biquad`: RBJ coefficient design, analytic magnitude, a DF-I
  `Biquad` with coefficient ramps.
- `strom_dsp::dynamics`, the building blocks of the dynamics processors:
  - smoothers and envelope followers: `OnePole`, `AttackRelease`,
    `PeakEnvelope`, `MeanSquareEnvelope` (RMS);
  - the soft-knee `GainComputer`;
  - the `GateStateMachine` and `GateFade` (one-pole attack, full-range dB
    release);
  - the sidechain `KeyDetector` (key filter, linked peak);
  - `DelayLine`, `SlidingMin`/`SegmentedMin` and the `RampSmoother`.
- `strom_dsp::filter`: Butterworth high-/low-pass (order 1–8) and the
  Linkwitz-Riley `Crossover` (LR2/4/8, 1–5 bands, all-pass compensated).
- `strom_dsp::oversample`: polyphase half-band oversampler, 2×/4×/8×, with
  integer latency.
- `strom_dsp::stft`: STFT framework (√Hann WOLA over `realfft`) with a
  per-frame spectrum callback.
- `strom_dsp::meter`: `LoudnessMeter` (BS.1770 via `ebur128`), the
  low-cost BS.1770 `LoudnessDetector` (momentary and short-term),
  `CorrelationMeter`, `BandCorrelation` and `MonoDetector`.
- `strom_dsp::agc`: `Agc`, the gated, rate-limited loudness-targeting AGC.
- `strom_dsp::multiband`: `MultibandCompressor`, an LR4 crossover with one
  compressor per band (1–5 bands), a downward expander per band
  (`strom_dsp::dynamics::expander`) and the bass clipper on band 0
  (`strom_dsp::bassclip`, with its IM-feedback input).
- `strom_dsp::truepeak`: `TruePeakLimiter`, a lookahead limiter with a 4×
  oversampled true-peak detector (BS.1770 Annex 2).
- `strom_dsp::spike_guard`: `SpikeGuard`, removes short, isolated peaks
  (clicks) with a look-ahead gain dip; an exact delay outside its dips.
- `strom_dsp::clipper`: `FinalClipper`, oversampled soft clipping of
  shaped clippings, with latency settings and per-block activity
  (`ClipStats`).
- `strom_dsp::governor`: `ClipBudgetGovernor`, the clip budget loop
  (loudness first: it reports a budget excess, it never lowers the
  loudness).
- `strom_dsp::broadcast`: `BroadcastProcessor`, the chain behind
  `stromdsp-broadcastprocessor` (input gain → spike guard → AGC →
  multiband → final clipper → true-peak limiter, clip budget governor),
  with its meters, and `LatencyMode` (standard, low), which sets every
  stage's latency setting at once.
- `strom_dsp::presets`: the element defaults (`GateParams::broadcast()` and
  friends) and the broadcast presets (`BroadcastPreset`).
- Processors behind the utility elements: `ducker`, `deesser`, `cleanup`,
  `delay` and `stereo`.
- `strom_dsp::declipper`: `Declipper` (library only, no element yet), with
  its `ClipDetector` and `LsarSolver`; see below.

The behaviour of each processor, as seen through its element, is described in
the [element pages](elements/README.md).

## Declipper (library only)

> Code is the source of truth — this may have drifted; read the code for the
> current implementation.

`strom_dsp::declipper::Declipper` repairs hard-clipped audio (an overdriven
microphone, a phone contribution that clipped before it reached us) in real
time. It finds clipped runs on the waveform and rebuilds them by AR
interpolation, kept beyond the clip level. There is no GStreamer element
for it yet.

```rust
use strom_dsp::declipper::{Declipper, DeclipperParams};
use strom_dsp::Processor;

let mut d = Declipper::new();
d.set_params(&DeclipperParams::default());
d.configure(48_000.0, 2); // allocates
d.process(&mut interleaved_f32); // real-time safe, 10 ms latency
let stats = d.take_stats(); // runs repaired, too long, max overshoot
```

- **Latency:** `round(10 ms · fs)` frames at every rate (480 at 48 kHz).
- **Unclipped audio** comes out bit-exact, delayed by the latency. Across
  51 unclipped test signals (full-scale tones at every peak phase,
  band-limited squares, soft-saturated tones, speech-, music- and noise-like
  material) as float, 16-bit and 24-bit, nothing is detected.
- **What it finds:** two or more consecutive samples on a flat plateau with
  sharp corners, above −20 dBFS. Gain, DC and requantisation after the clip
  do not matter. A high-pass after the clip (DC blocker) tilts the plateau;
  `tilt_hz` (default 50 Hz) covers part of that.
- **What it cannot do:** single clipped samples (no plateau), soft clipping
  or saturation, clipping followed by a lossy codec (the plateau turns into
  ripple), and runs longer than `max_run_ms` (default 2 ms; passed through
  and counted). A naive (not band-limited) square wave is indistinguishable
  from clipping and is "repaired".
- **Parameters:** `enabled` (crossfade to the delayed dry signal),
  `min_level_db` (−20), `tolerance` (1e-5 relative), `max_run_ms` (2),
  `max_overshoot_db` (12), `tilt_hz` (50).

**Headroom.** Rebuilt peaks lie beyond the old clip level, so the output is
hotter than the input: by up to `max_overshoot_db` over the clip level, in
practice up to about the clip depth (3 dB of clipping gives about 3 dB of
overshoot, 10 dB gives 7–12 dB). In a float chain that is harmless. Before
any stage with a ceiling (integer conversion, an encoder, a transmission
limit) put a limiter after the declipper, e.g. the true-peak limiter at the
end of the broadcast chain. Lowering the gain before the declipper does not
help: the detector needs the plateau as it is, and the overshoot is
relative to it.

**Cost** grows with the amount of clipping: about 4 µs per 10 ms stereo
buffer on clean audio, 140 µs on speech and music clipped 10 dB, 0.7 ms
(worst buffer about 1.6 ms) on dense tones clipped 10 dB (Apple M5 Pro).

`cargo run --release --example declip_render -- IN.wav OUT.wav` declips a
WAV file and writes a report; `--demo DIR` renders a synthetic A/B set.
