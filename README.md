# strom-dsp

A small, clean-room audio DSP library in Rust, and a GStreamer plugin on top of
it, built for the [Strom](https://github.com/Eyevinn/strom) live media engine.

- **`strom-dsp`** (`crates/strom-dsp`): the DSP library. It has no GStreamer
  dependency and is real-time safe: after `configure` it does no allocation,
  locking, logging or I/O. It processes interleaved `f32` in place, with any
  channel count and 8–192 kHz. It uses SIMD with runtime dispatch (NEON,
  SSE4.2/AVX2/AVX-512).
- **`gst-plugin-stromdsp`** (`crates/gst-plugin-stromdsp`): the GStreamer
  plugin `stromdsp` with eleven elements. It builds as a `cdylib` (for
  `GST_PLUGIN_PATH`) and as an `rlib` for static linking
  (`gststromdsp::plugin_register_static()`).

Licence: MIT OR Apache-2.0. See [PROVENANCE.md](PROVENANCE.md) for the
clean-room record: which reference each algorithm follows, and every
dependency with its licence.

## Elements

| Element | What it does |
|---|---|
| [`stromdsp-gate`](docs/elements/gate.md) | Noise gate with hysteresis and hold, or downward expander |
| [`stromdsp-compressor`](docs/elements/compressor.md) | Feed-forward compressor with soft knee, peak or RMS detector |
| [`stromdsp-equalizer`](docs/elements/equalizer.md) | Parametric EQ, up to 8 biquad bands |
| [`stromdsp-limiter`](docs/elements/limiter.md) | Lookahead peak limiter with a guaranteed ceiling |
| [`stromdsp-ducker`](docs/elements/ducker.md) | Lowers the programme while a sidechain key is present |
| [`stromdsp-deesser`](docs/elements/deesser.md) | Sibilance reduction, split-band or wideband |
| [`stromdsp-cleanup`](docs/elements/cleanup.md) | DC, rumble and hum removal |
| [`stromdsp-delay`](docs/elements/delay.md) | Intentional delay for alignment and lip sync |
| [`stromdsp-stereo`](docs/elements/stereo.md) | Balance, width, mono, swap, M/S and polarity |
| [`stromdsp-meter`](docs/elements/meter.md) | Loudness (BS.1770), peak, true peak, RMS and correlation meter |
| [`stromdsp-broadcastprocessor`](docs/elements/broadcastprocessor.md) | Final processor: AGC → multiband → true-peak limiter, with presets |

The first four form the channel strip. Every processing element has an
`enabled` property that crossfades to a bit-exact bypass, every
PLAYING-mutable property changes without clicks, and only the limiter and the
broadcast processor add latency. The shared conventions (bypass, latency,
caps, linked channels, defaults) are in the
[element overview](docs/elements/README.md).

## The library

```rust
use strom_dsp::{Compressor, CompressorParams, Processor};

let mut comp = Compressor::new();
comp.configure(48_000.0, 2); // allocates; do this off the audio thread
comp.set_params(&CompressorParams { threshold: 0.25, ratio: 4.0, ..Default::default() });
comp.process(&mut interleaved_f32); // real-time safe
```

Filters, crossovers, oversampling, STFT, envelope followers, loudness meters,
AGC, multiband and true-peak limiting are public building blocks too. See
[docs/library.md](docs/library.md).

## Quick start

You need Rust 1.92 or newer (the library alone: 1.89) and the GStreamer ≥ 1.22
development files. Details, platforms and the optional checks are in
[docs/building.md](docs/building.md).

```sh
cargo build --release
cargo test --all

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

`cargo test --all` includes the measurement harness
(`crates/strom-dsp-harness`, test support only): fixed synthetic signals
through every broadcast preset, reduced to loudness, true-peak,
gain-reduction, clip-depth, aliasing, intermodulation and noise-to-mask
numbers, and checked against `crates/strom-dsp-harness/baseline.json`.

To link statically into an application, depend on the crate and call:

```rust
gststromdsp::plugin_register_static()?;
let comp = gst::ElementFactory::make("stromdsp-compressor").build()?;
```

## Performance

At 48 kHz stereo with 10 ms buffers, on an Apple M5 Pro (criterion medians,
measured under heavy unrelated load, so upper bounds):

| Benchmark | Time per 10 ms buffer | % of one core |
|---|---|---|
| Channel strip (gate → compressor → 4-band EQ → limiter) | 18.3 µs | 0.18 % |
| Broadcast processor (AGC → 4-band multiband → true-peak limiter, with meters) | 78.3 µs | 0.78 % |

[docs/PERFORMANCE.md](docs/PERFORMANCE.md) lists every benchmark, and covers
the instruction-count benchmarks (gungraun) and the RealtimeSanitizer tests.

## Contributing

Every element, present and future, follows the
[element guidelines](docs/ELEMENT_GUIDELINES.md): `enabled` bypass that fades
and ends bit-exact, click-free live changes of every PLAYING-mutable property,
real-time safety, reported and constant latency, sensible standalone defaults,
a test per rule, and clean-room provenance. strom-dsp is a clean-room project:
keep [PROVENANCE.md](PROVENANCE.md) current with the reference for every
algorithm and every reused crate.
