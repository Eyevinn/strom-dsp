# strom-dsp

A small, clean-room audio DSP library in Rust, and a GStreamer plugin on top of
it, built for the [Strom](https://github.com/Eyevinn/strom) live media engine.

- **`strom-dsp`** (`crates/strom-dsp`): the DSP library. It has no GStreamer
  dependency and is real-time safe: after `configure` it does no allocation,
  locking, logging or I/O. It processes interleaved `f32` in place, with any
  channel count and 8–192 kHz. It uses SIMD with runtime dispatch (NEON,
  SSE4.2/AVX2/AVX-512).
- **`gst-plugin-stromdsp`** (`crates/gst-plugin-stromdsp`): the GStreamer
  plugin `stromdsp` with ten elements: the four phase 1 channel-strip
  elements (gate, compressor, equalizer, limiter) and six phase 1b elements
  (ducker, de-esser, cleanup, delay, stereo, meter). It builds as a `cdylib` (for
  `GST_PLUGIN_PATH`) and as an `rlib` for static linking
  (`gststromdsp::plugin_register_static()`).

Licence: MIT OR Apache-2.0. See [PROVENANCE.md](PROVENANCE.md) for the
clean-room record: which reference each algorithm follows, and every
dependency with its licence.

## Elements

The four phase 1 elements are in-place `BaseTransform`s:
- caps `audio/x-raw, format=F32LE, layout=interleaved, rate=[1,MAX], channels=[1,MAX]`
- klass `Filter/Effect/Audio`
- every property controllable while PLAYING, except `num-bands`
- `enabled=false` is bit-exact passthrough, except for the limiter (see below)

Gate, compressor and limiter use **linked channels**: one gain is computed from
the loudest channel and applied to all, so the stereo image never moves.

### `stromdsp-gate`: noise gate with hysteresis and hold

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `open-threshold` | float | dB | −80 … 0 | −20 |
| `close-threshold` | float | dB | −80 … 0 | −30 |
| `attack` | float | ms | 0.01 … 200 | 5 |
| `hold` | float | ms | 0 … 1000 | 50 |
| `release` | float | ms | 1 … 2000 | 100 |
| `reduction` | float | dB | −80 … 0 | −60 |

Phase 1b additions (additive; every phase 1 property is unchanged):

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `mode` | enum `StromDspGateMode` | | `gate` (0), `expander` (1) | `gate` |
| `ratio` | float | expander ratio | 1 … 20 | 2 |

**Detector.** The level is a sine-calibrated RMS:
- a 10 ms one-pole mean square of the loudest channel, read as `sqrt(2·m)`;
- a steady sine reads its peak level in dBFS, and noise reads its RMS + 3 dB;
- after the signal stops, the level decays with 20 ms.

Single peaks of noise below the threshold therefore do not open the gate.

**Opening and closing.**
- **Opening:** the gate opens when the level goes above `open-threshold`, with
  the `attack` time constant (63 % after `attack`).
- **Hysteresis:** the effective close threshold is
  `min(close-threshold, open-threshold − 1 dB)`. If
  `close-threshold` > `open-threshold`, it is treated as equal, and a built-in
  1 dB minimum hysteresis still applies.
- **Hold:** once the level is below the close threshold, the gate stays open
  for `hold` ms.
- **Release** is the **time to close**: the gain falls in a straight line in
  dB from 0 dB to `reduction` in exactly `release` ms.

After the signal stops, the gate is fully closed after
`t_detector + hold + release`. Here `t_detector` ≈ 2.3 ms per dB that the
signal was above the close threshold.

**Expander mode.** `mode=expander` replaces the fixed closed gain with downward
expansion:
- below `open-threshold` the gain is `max(reduction, (ratio − 1)·(level − open-threshold))` dB;
- hysteresis, hold and the fades work as in gate mode;
- `mode=gate` is bit-identical to the phase 1 behaviour.

### `stromdsp-compressor`: feed-forward compressor with soft knee

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `threshold` | float | linear amplitude | 0 … 1 | 0.5 |
| `ratio` | float | ratio (4 = 4:1) | 1 … 100 | 1 |
| `attack` | float | ms | 0.01 … 10000 | 20 |
| `release` | float | ms | 0.01 … 10000 | 100 |
| `makeup-gain` | float | linear multiplier | 0 … 100 | 1 |
| `knee` | float | linear factor | 0 … 1 | 0.0625 |

**Knee definition.**
- `knee` is a linear amplitude factor `k`.
- The soft-knee region spans from `threshold · k` to `threshold / k`, so the
  knee width in dB is `W = −40·log10(k)`. The default 0.0625 gives about 48 dB.
- `k = 1` is a hard knee. Values below 0.001 are treated as 0.001.
- Inside the knee the gain follows the quadratic interpolation of Giannoulis,
  Massberg & Reiss (JAES 2012).
- Ratio 1 means unity (apart from makeup).
- `knee` reads back exactly as written.

### `stromdsp-equalizer`: parametric EQ, up to 8 biquad bands

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `num-bands` | uint | | 1 … 8 (NULL/READY only) | 4 |
| `bandN-enabled` | bool | | | `true` |
| `bandN-type` | int | see below | 0 … 9 | 0 |
| `bandN-frequency` | float | Hz | 1 … 24000 | 1000 |
| `bandN-gain` | float | dB | −60 … 60 | 0 |
| `bandN-q` | float | | 0.01 … 100 | 0.707 |

All eight `bandN-*` sets (N = 0 … 7) always exist. Bands at index ≥
`num-bands` are inactive.

`bandN-type` values:

| Value | Type |
|---|---|
| 0 | off |
| 1 | low-pass |
| 2 | high-pass |
| 3 | band-pass, constant skirt gain |
| 4 | band-pass, 0 dB peak |
| 5 | notch |
| 6 | all-pass |
| 7 | peaking |
| 8 | low shelf |
| 9 | high shelf |

Coefficients come from the RBJ Audio EQ Cookbook, using the Q form for the
shelves as well. `gain` applies to peaking and the shelves only. Frequency is
clamped just below Nyquist. Parameter changes glide smoothly, and type changes
crossfade, so there are no clicks.

### `stromdsp-limiter`: lookahead peak limiter

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `threshold` | float | dB (ceiling) | −60 … 0 | −1 |
| `attack` | float | ms | 0.01 … 50 | 1 |
| `release` | float | ms | 1 … 2000 | 100 |
| `lookahead` | float | ms | 0 … 20 | 5 |

**Ceiling guarantee.** No output sample exceeds `threshold`, for any input and
also with `lookahead = 0`. The gain ramp is built so that it reaches the
required gain before the peak arrives, and it rounds only downwards. A final
safety clip exists as a guard only; tests show it never triggers on normal
material.

**Latency.**
- The audio is delayed by `lookahead`, and the element adds it to the LATENCY
  query.
- A change of `lookahead` posts a `Latency` message.
- Timestamps pass through unchanged.

**Disabled behaviour.** With `enabled=false` the limiter applies unity gain but
**keeps the delay**, so the latency does not jump when an operator toggles it.

## Phase 1b elements

These are new interfaces: units are dB / ms / Hz. Every property is mutable
while PLAYING unless noted.

### `stromdsp-ducker`: lower the programme while a key is present

A `GstAggregator` with three pads:
- an always pad `sink` for the programme;
- a request pad `sidechain` for the key (F32LE interleaved, any rate and
  channel count, at most one);
- an always pad `src`.

The output has the programme's caps, segment and timestamps. The key detector
runs at the key's own rate, so the key is never resampled:
- an optional high-pass;
- the linked peak over the key's channels;
- a 10 ms sine-calibrated RMS.

**Live pipelines.** A missing, stalled or unlinked key never blocks. The
element waits for the key at most the pipeline latency; raise it with the
inherited `latency` property if the key has more jitter. Missing key audio
reads as silence, so the programme holds and then fades back to unity.

**Non-live pipelines.** Request `sidechain` only if it will be fed or reach
EOS.

`enabled=false` is a bit-exact passthrough, and the key is still consumed.

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `threshold` | float | dB (key level, sine-calibrated RMS) | −80 … 0 | −30 |
| `depth` | float | dB (programme gain at full ducking) | −80 … 0 | −12 |
| `attack` | float | ms (unity to full depth, linear in dB) | 0.1 … 2000 | 20 |
| `hold` | float | ms | 0 … 5000 | 300 |
| `release` | float | ms (full depth back to unity, linear in dB) | 1 … 10000 | 500 |
| `key-highpass` | float | Hz, 0 = off | 0 … 20000 | 100 |
| `range-mode` | enum `StromDspDuckerRangeMode` | | `fixed` (0), `proportional` (1) | `fixed` |
| `meter-interval` | uint | ms of running time, 0 = off (not controllable) | 0 … 10000 | 100 |

**Range modes.**
- `fixed`: duck by `depth` while the key is above `threshold`.
- `proportional`: 1 dB of reduction per dB the key is above `threshold`, up to
  `depth`.

**Meter message.** The element posts an element message with structure name
`stromdsp-ducker` at the first buffer, and then at most once per
`meter-interval`:

| Field | Type | Meaning |
|---|---|---|
| `running-time`, `stream-time` | guint64 | ns, end of the triggering buffer |
| `reduction` | gdouble | dB (≤ 0), gain applied |
| `key-level` | gdouble | dBFS, sine-calibrated RMS after the key filter; −200 for silence |
| `key-active` | gboolean | key above threshold or within hold |
| `key-present` | gboolean | real key samples covered part of that buffer |

```sh
gst-launch-1.0 -m \
  audiotestsrc is-live=true wave=sine freq=220 volume=0.5 \
    ! audio/x-raw,format=F32LE,rate=48000,channels=2 \
    ! stromdsp-ducker name=duck threshold=-30 depth=-12 attack=20 hold=300 release=500 \
    ! audioconvert ! autoaudiosink \
  audiotestsrc is-live=true wave=pink-noise volume=0.2 \
    ! audio/x-raw,format=F32LE,rate=16000,channels=1 ! duck.sidechain
```

### `stromdsp-deesser`: sibilance reduction

A linked-channel de-esser.
- A 12 dB/octave high-pass detector at `frequency` drives a gain reduction.
  Sibilance is held at `threshold`, and the reduction is at most
  `max-reduction` dB.
- `split-band` reduces only the band above `frequency`, split by an LR4
  crossover. With no reduction the bands sum to a flat all-pass.
- `wideband` reduces the whole signal. It goes through the same all-pass
  split, so mode changes glide.
- No latency. `enabled=false` is a bit-exact passthrough.

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `frequency` | float | Hz (limited to 0.45·fs) | 1000 … 20000 | 6000 |
| `threshold` | float | dBFS (detector peak) | −60 … 0 | −30 |
| `max-reduction` | float | dB (positive = attenuation) | 0 … 40 | 10 |
| `attack` | float | ms (63 % time constant) | 0.01 … 200 | 1 |
| `release` | float | ms (63 % time constant) | 1 … 2000 | 50 |
| `mode` | enum `StromDspDeesserMode` | | `split-band` (0), `wideband` (1) | `split-band` |

```sh
gst-launch-1.0 pulsesrc ! audioconvert ! audio/x-raw,format=F32LE,layout=interleaved ! \
  stromdsp-deesser frequency=6500 threshold=-32 max-reduction=8 mode=split-band ! \
  audioconvert ! autoaudiosink
```

### `stromdsp-cleanup`: DC, rumble and hum removal

Input cleanup for contributions and field recordings. The stages run in this
order:
1. **DC blocker:** first-order high-pass, −3 dB at 5 Hz.
2. **High-pass:** Butterworth, order 1–8.
3. **Hum notches:** at 50 or 60 Hz plus harmonics. Every notch has the same
   bandwidth, f0/`hum-q`.
   - Each enabled harmonic is attenuated by ≥ 40 dB.
   - The passband between notches stays within 0.5 dB over the whole `hum-q`
     range.
   - Notches at or above 0.49·fs are skipped.

Every stage is off by default, so the default element is a bit-exact
passthrough. Switching a stage on or off crossfades over 10 ms. No latency.
`enabled=false` is a bit-exact passthrough.

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `dc-block` | bool | | | `false` |
| `highpass` | float | Hz (−3 dB), 0 = off | 0 … 1000 | 0 |
| `highpass-order` | uint | 6 dB/oct per order | 1 … 8 | 2 |
| `hum` | enum `StromDspCleanupHum` | | `off` (0), `50` (50), `60` (60) | `off` |
| `hum-harmonics` | uint | number of notches | 1 … 10 | 5 |
| `hum-q` | float | Q of the fundamental; bandwidth f0/`hum-q` | 6 … 100 | 10 |

```sh
gst-launch-1.0 filesrc location=field.wav ! wavparse ! audioconvert ! \
  audio/x-raw,format=F32LE,layout=interleaved ! \
  stromdsp-cleanup dc-block=true highpass=70 highpass-order=4 hum=50 hum-harmonics=8 ! \
  audioconvert ! autoaudiosink
```

### `stromdsp-delay`: intentional delay for alignment

Content alignment, such as mic alignment and lip sync. There is a common
`delay` plus per-channel `channel-delays`. The delay per channel is
`clamp(delay + offset, 0, 2000)` ms, rounded to whole frames.

> **This delay is NOT latency.** It is not reported in the LATENCY query, and
> timestamps pass through unchanged, so the audio content moves later relative
> to its timestamps. If it were reported, a latency-compensating pipeline would
> undo it.

**Changes and toggling.**
- Changes crossfade over 20 ms (raised cosine between the old and new taps).
- `enabled=false` crossfades to delay 0, after which the output is a bit-exact
  copy. The delay line keeps recording, so toggling is click-free.
- Delay 0, the default, is bit-exact.
- Memory for 2000 ms is allocated at caps negotiation.

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `delay` | float | ms | 0 … 2000 | 0 |
| `channel-delays` | GstValueArray of float | ms per channel, added to `delay` | −2000 … 2000 each | `<>` (empty) |

**`channel-delays` details.**
- Missing entries are 0, and extra entries are ignored.
- At most 64 entries are used.
- It is mutable while PLAYING, but not controllable, because arrays cannot be
  interpolated.

```sh
gst-launch-1.0 audiotestsrc ! audio/x-raw,format=F32LE,channels=2,rate=48000 ! \
  stromdsp-delay delay=40 'channel-delays=<0.0, 2.5>' ! autoaudiosink
```

### `stromdsp-stereo`: stereo utility

A stereo utility for exactly 2 channels (caps `channels=2`). It applies one
2×2 matrix: balance · width · mode · polarity.
- **Polarity:** applied to the inputs.
- **M/S:** `M = (L+R)/2` and `S = (L−R)/2`. Decode is `L = M+S`, `R = M−S`, so
  encode followed by decode is exact.
- **Width:** scales S. 0 is mono, 1 is unchanged, 2 is wide. There is no gain
  compensation.
- **Balance:** constant power, with `θ = (b+1)·π/4`, `gL = √2·cos θ` and
  `gR = √2·sin θ`. The centre is unity; at ±1 the remaining side is +3 dB.

Changes ramp over 10 ms. Pure copy, swap and invert are applied bit-exactly.
The defaults, and `enabled=false` once its ramp ends, are bit-exact
passthrough. No latency.

| Property | Type | Range / values | Default |
|---|---|---|---|
| `enabled` | bool | | `true` |
| `mode` | enum `StromDspStereoMode` | `stereo`, `mono`, `left-to-both`, `right-to-both`, `swap`, `ms-encode`, `ms-decode` | `stereo` |
| `mono-law` | enum `StromDspStereoMonoLaw` | `minus-3db` (1/√2), `minus-6db` (1/2) | `minus-3db` |
| `invert-left`, `invert-right` | bool | | `false` |
| `balance` | float | −1 … 1 | 0 |
| `width` | float | 0 … 2 | 1 |

```sh
gst-launch-1.0 audiotestsrc ! audio/x-raw,format=F32LE,channels=2,rate=48000 ! \
  stromdsp-stereo mode=mono mono-law=minus-6db ! autoaudiosink
```

### `stromdsp-meter`: loudness, peak, RMS and correlation meter

A passthrough meter: samples are never changed. It has klass
`Filter/Analyzer/Audio` and accepts F32LE interleaved audio with 1–64 channels.
Every `interval` ms of stream time it posts an element message.

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `interval` | uint | ms | 10 … 60000 | 100 |
| `true-peak` | bool | (changeable in NULL/READY only; applies at the next caps) | | `true` |

**Action signal `reset`** (no arguments). It restarts integrated loudness and
loudness range. It also restarts every other loudness value and the
programme peaks.

**Message format.** The structure is named `stromdsp-meter`
(`gststromdsp::meter::MESSAGE_NAME`).

| Field | Type | Unit / meaning |
|---|---|---|
| `stream-time`, `running-time` | guint64 | ns, end (PTS + duration) of the last measured buffer; `GST_CLOCK_TIME_NONE` if unknown |
| `duration` | guint64 | ns of audio covered by the interval values |
| `channels` | gint | |
| `momentary`, `short-term`, `integrated` | gdouble | LUFS |
| `loudness-range` | gdouble | LU (0 without data) |
| `sample-peak` | GstValueArray of gdouble | dBFS per channel, this interval |
| `sample-peak-max` | gdouble | dBFS, highest channel |
| `true-peak` | GstValueArray of gdouble | dBTP per channel, this interval (omitted when `true-peak=false`) |
| `true-peak-max` | gdouble | dBTP (omitted when `true-peak=false`) |
| `programme-sample-peak` | gdouble | dBFS since start or `reset` |
| `programme-true-peak` | gdouble | dBTP since start or `reset` (omitted when `true-peak=false`) |
| `rms` | GstValueArray of gdouble | dBFS per channel over the interval, unweighted (a full-scale sine reads −3.01) |
| `correlation` | gdouble | −1 … +1, broadband (300 ms integration); only for exactly 2 channels |
| `correlation-low`, `correlation-mid`, `correlation-high` | gdouble | −1 … +1 below 250 Hz, 250 Hz–4 kHz, above 4 kHz; only for 2 channels |
| `stereo-state` | gchararray | `"mono"`, `"stereo"` or `"inverted"`; only for 2 channels |

**Silence and timing.**
- Silence is reported as **−200.0** (`gststromdsp::meter::SILENCE_DB`). No
  value is ever NaN or −∞, and a silent channel or band reads correlation 0.
- At most one message is posted per buffer.
- FLUSH_STOP and DISCONT keep the integrated measurement.

**Channel weights.** These follow BS.1770 when the caps carry a channel mask:
- LFE is excluded;
- side/surround positions, and rear positions in 5.x, are weighted 1.41;
- everything else is weighted 1.0.

Without a mask, every channel is weighted 1.0.

**Cost.** Measuring costs about 7 µs per 10 ms of stereo (10 µs with true peak).
The loudness readings are taken only when a message is due, about 170 µs per
message.

```sh
gst-launch-1.0 -m audiotestsrc wave=pink-noise ! audioconvert ! \
  audio/x-raw,format=F32LE,channels=2 ! stromdsp-meter interval=100 ! autoaudiosink
```

## Build and test

You need Rust 1.92 or newer and GStreamer ≥ 1.22 development files:
- Debian/Ubuntu: `libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev`
- macOS: `brew install gstreamer`
- Windows: the official MSVC runtime and devel installers

The library alone builds with Rust 1.89.

```sh
cargo build --release
cargo test --all
cargo clippy --all-targets -- -D warnings
cargo deny check licenses
```

Optional checks, described in [docs/PERFORMANCE.md](docs/PERFORMANCE.md):
- RealtimeSanitizer on Linux/macOS, run once per `tests/rtsan*.rs` file:
  `RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test rtsan`
  (also `rtsan_filter`, `rtsan_meter`, `rtsan_sampling`, `rtsan_sidechain`,
  `rtsan_ducker`, `rtsan_deesser_cleanup`, `rtsan_delay_stereo`).
- Instruction-count benchmarks with gungraun on Linux.

## Use with GStreamer

```sh
cargo build --release
export GST_PLUGIN_PATH="$PWD/target/release"
gst-inspect-1.0 stromdsp

gst-launch-1.0 audiotestsrc wave=pink-noise ! audioconvert ! \
  stromdsp-gate open-threshold=-40 close-threshold=-50 ! \
  stromdsp-compressor threshold=0.25 ratio=4 knee=0.25 makeup-gain=1.5 ! \
  stromdsp-equalizer num-bands=2 band0-type=2 band0-frequency=80 \
                     band1-type=7 band1-frequency=3000 band1-gain=3 band1-q=1 ! \
  stromdsp-limiter threshold=-1 lookahead=5 ! \
  audioconvert ! autoaudiosink
```

The plugin file is `libgststromdsp.so` on Linux, `libgststromdsp.dylib` on
macOS and `gststromdsp.dll` on Windows.

To link statically into an application, depend on the crate and call:

```rust
gststromdsp::plugin_register_static()?;
let comp = gst::ElementFactory::make("stromdsp-compressor").build()?;
```

## Using the library directly

```rust
use strom_dsp::{Compressor, CompressorParams, Processor};

let mut comp = Compressor::new();
comp.configure(48_000.0, 2); // allocates; do this off the audio thread
comp.set_params(&CompressorParams { threshold: 0.25, ratio: 4.0, ..Default::default() });
comp.process(&mut interleaved_f32); // real-time safe
```

Reusable building blocks are public too:
- `strom_dsp::biquad`: RBJ coefficient design, analytic magnitude, a DF-I
  `Biquad` with coefficient ramps.
- `strom_dsp::dynamics`:
  - envelope followers (peak, RMS);
  - a soft-knee gain computer;
  - the gate state machine and `GateFade` (one-pole attack, full-range dB release);
  - the sidechain `KeyDetector` (key filter, linked peak);
  - delay line, sliding minimum and ramp smoother.
- `strom_dsp::filter`: Butterworth high-/low-pass (order 1–8) and the
  Linkwitz-Riley `Crossover` (LR2/4/8, 1–5 bands, all-pass compensated).
- `strom_dsp::oversample`: polyphase half-band oversampler, 2×/4×/8×, with
  integer latency.
- `strom_dsp::stft`: STFT framework (√Hann WOLA over `realfft`) with a
  per-frame spectrum callback.
- `strom_dsp::meter`: `LoudnessMeter` (BS.1770 via `ebur128`),
  `CorrelationMeter`, `BandCorrelation` and `MonoDetector`.
- Processors behind the phase 1b elements: `ducker`, `deesser`, `cleanup`,
  `delay` and `stereo`.

## Performance

Run the benchmark with `cargo bench -p strom-dsp --bench strip` (criterion).
These numbers are criterion medians on an Apple M5 Pro (aarch64, NEON level),
release profile with fat LTO, 48 kHz.
- **% of one core** = time per buffer ÷ buffer duration. 480 frames are 10 ms;
  1024 frames are 21.3 ms.
- **The strip** is gate → compressor → 4-band EQ (HPF 80 Hz, bell 300 Hz
  −3 dB, bell 3 kHz +2.5 dB, high shelf 10 kHz +2 dB) → limiter. The input is a
  synthetic programme, so every stage does real work.

| Benchmark | 480 frames | 1024 frames | % of one core |
|---|---|---|---|
| **Channel strip, stereo** | **6.83 µs** | **14.33 µs** | **0.068 %** |
| Gate, stereo | 0.79 µs | 1.65 µs | 0.008 % |
| Compressor, stereo | 2.54 µs | 5.38 µs | 0.025 % |
| EQ 4-band, mono | 0.98 µs | 2.01 µs | 0.010 % |
| EQ 4-band, stereo | 1.02 µs | 2.08 µs | 0.010 % |
| EQ 4-band, 8 channels | 3.91 µs | 8.15 µs | 0.039 % |
| EQ 4-band, stereo, gain automated every buffer | 2.69 µs | 5.63 µs | 0.027 % |
| Limiter, stereo | 3.04 µs | 6.46 µs | 0.030 % |
| Limiter, stereo, +20 dB input (heavy limiting) | 3.08 µs | 6.58 µs | 0.031 % |

The spec target is "well under 1 % of one core per strip". One strip costs
about 0.07 %, so one core can run more than 1000 stereo strips.

### Phase 1b components

These numbers were reported by the agents that built each component: criterion
medians on the same Apple M-series machine, 48 kHz stereo, one 480-frame
(10 ms) buffer, taken while the machine was under other load. Expect about
±10 %. Run them with `cargo bench -p strom-dsp --bench <name>`, using the names
`filters`, `sampling`, `meter`, `sidechain`, `ducker`, `dynamics_1b`
(de-esser), `cleanup` and `channel_tools` (delay, stereo).

| Benchmark | Time per buffer | % of one core |
|---|---|---|
| Butterworth order 4 / order 8 | 0.86 / 1.02 µs | 0.009 / 0.010 % |
| LR4 crossover, 3 / 5 bands | 3.48 / 7.98 µs | 0.035 / 0.08 % |
| Oversampler 4× up+down | 11.7 µs | 0.12 % |
| STFT 1024/256, 2048/512 (identity) | 7.0 / 8.6 µs | 0.07 / 0.09 % |
| Loudness meter, without / with true peak | 7.2 / 10.4 µs | 0.07 / 0.10 % |
| Band correlation (3 bands) | 5.6 µs | 0.056 % |
| Compressor keyed + HPF key filter | 3.55 µs | 0.036 % |
| Gate (RMS detector), expander | 1.53 / 1.79 µs | 0.015 / 0.018 % |
| Ducker (48 kHz mono key, HPF) | 3.35 µs | 0.034 % |
| De-esser, split-band | 8.42 µs | 0.084 % |
| Cleanup, DC + HP4 + 10 hum notches | 4.71 µs | 0.047 % |
| Delay 20 ms, stereo | 0.15 µs | 0.0015 % |
| Stereo, width + balance | 0.33 µs | 0.003 % |

The gate became about 1.8× slower with the RMS detector, which fixes chatter on
noise peaks. It is still 0.015 % of a core.

x86-64 timings (SSE4.2/AVX2/AVX-512 paths) have not been measured yet. CI runs
the tests on those paths. For instruction-count benchmarks (gungraun), the
RealtimeSanitizer tests, SIMD design notes and the denormal policy, see
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Contributing

Every element, present and future, follows the
[element guidelines](docs/ELEMENT_GUIDELINES.md): `enabled` bypass that fades
and ends bit-exact, click-free live changes of every PLAYING-mutable property,
real-time safety, reported and constant latency, sensible standalone defaults,
a test per rule, and clean-room provenance. strom-dsp is a clean-room project:
keep [PROVENANCE.md](PROVENANCE.md) current with the reference for every
algorithm and every reused crate.
