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
- `strom_dsp::declipper`: `Declipper`, with its `ClipDetector` and
  `LsarSolver`; see below.

The behaviour of each processor, as seen through its element, is described in
the [element pages](elements/README.md).

## Declipper

> Code is the source of truth — this may have drifted; read the code for the
> current implementation.

`strom_dsp::declipper::Declipper` repairs hard-clipped audio (an overdriven
microphone, a phone contribution that clipped before it reached us) in real
time. It finds clipped runs on the waveform and rebuilds them by AR
interpolation, kept beyond the clip level. The element is
[`stromdsp-declipper`](elements/declipper.md); its page has the behaviour,
the latency per setting and the measured quality.

```rust
use strom_dsp::declipper::{Declipper, DeclipperLatency, DeclipperParams};
use strom_dsp::Processor;

let mut d = Declipper::new();
d.set_params(&DeclipperParams {
    latency: DeclipperLatency::Low, // 3 ms; Normal (default) 5, High 10
    ..Default::default()
});
d.configure(48_000.0, 2); // allocates
d.process(&mut interleaved_f32); // real-time safe
let stats = d.take_stats(); // runs repaired, too long, fallbacks, overshoot
```

- **Latency:** `floor(ms · fs)` frames (at least 16), fixed at `configure`.
- **Unclipped audio** comes out bit-exact, delayed by the latency.
- **Parameters:** `enabled` (crossfade to the delayed dry signal),
  `latency`, `min_level_db` (−20), `tolerance` (1e-5 relative),
  `max_run_ms` (2), `max_overshoot_db` (12), `tilt_hz` (50).
- **Headroom:** rebuilt peaks lie beyond the old clip level, so the output
  is hotter than the input; put a limiter after it before any ceiling.
- **Work budget:** each channel spends at most 200 000 multiply-adds of
  saved-up work plus 50·10⁶ per second of audio; past that, runs get a
  cheap constrained fallback (`DeclipperStats::fallbacks`). The output does
  not depend on the machine or the buffer size.

`cargo run --release --example declip_render -- IN.wav OUT.wav` declips a
WAV file and writes a report; `--demo DIR` renders a synthetic A/B set.
`cargo run --release --example declip_eval -- FILE.wav...` clips recordings
3, 6 and 10 dB below their peak and reports recall, false detections, the
SDR improvement on the clipped samples and the time per buffer.
