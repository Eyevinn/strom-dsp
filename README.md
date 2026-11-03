# strom-dsp

A small, clean-room audio DSP library in Rust, and a GStreamer plugin on top of
it, built for the [Strom](https://github.com/Eyevinn/strom) live media engine.

- **`strom-dsp`** (`crates/strom-dsp`): the DSP library. It has no GStreamer
  dependency and is real-time safe: after `configure` it does no allocation,
  locking, logging or I/O. It processes interleaved `f32` in place, with any
  channel count and 8–192 kHz. It uses SIMD with runtime dispatch (NEON,
  SSE4.2/AVX2/AVX-512).
- **`gst-plugin-stromdsp`** (`crates/gst-plugin-stromdsp`): the GStreamer
  plugin `stromdsp` with four elements. It builds as a `cdylib` (for
  `GST_PLUGIN_PATH`) and as an `rlib` for static linking
  (`gststromdsp::plugin_register_static()`).

Licence: MIT OR Apache-2.0. See [PROVENANCE.md](PROVENANCE.md) for the
clean-room record: which reference each algorithm follows, and every
dependency with its licence.

## Elements

All four are in-place `BaseTransform`s:
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

The gate opens above `open-threshold`, and starts closing once the level has
stayed below `close-threshold` for `hold` ms. It fades to `reduction` with the
`release` time constant and opens with the `attack` time constant. If
`close-threshold` > `open-threshold`, it is treated as equal.

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
- RealtimeSanitizer on Linux/macOS:
  `RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test rtsan`
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
- `strom_dsp::dynamics`: envelope followers, soft-knee gain computer, gate
  state machine, delay line, sliding minimum, ramp smoother.

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

x86-64 timings (SSE4.2/AVX2/AVX-512 paths) have not been measured yet. CI runs
the tests on those paths. For instruction-count benchmarks (gungraun), the
RealtimeSanitizer tests, SIMD design notes and the denormal policy, see
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).
