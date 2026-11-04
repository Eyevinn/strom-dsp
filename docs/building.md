# Building and testing

## Requirements

You need Rust 1.92 or newer and GStreamer ≥ 1.22 development files:
- Debian/Ubuntu: `libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev`
- macOS: `brew install gstreamer`
- Windows: the official MSVC runtime and devel installers

The library alone (`cargo build -p strom-dsp`) builds with Rust 1.89. CI
checks both minimum versions: `strom-dsp` with Rust 1.89 and
`gst-plugin-stromdsp` with Rust 1.92. CI runs on Linux, macOS and Windows.

## Build and test

```sh
cargo build --release
cargo test --all
cargo clippy --all-targets -- -D warnings
cargo deny check licenses
```

`cargo deny check licenses` checks every dependency against the licence
allowlist in `deny.toml`.

## Using the plugin

```sh
cargo build --release
export GST_PLUGIN_PATH="$PWD/target/release"
gst-inspect-1.0 stromdsp
```

The plugin file is `libgststromdsp.so` on Linux, `libgststromdsp.dylib` on
macOS and `gststromdsp.dll` on Windows.

To link statically into an application, depend on the crate and call:

```rust
gststromdsp::plugin_register_static()?;
let comp = gst::ElementFactory::make("stromdsp-compressor").build()?;
```

## Optional checks

These are described in [PERFORMANCE.md](PERFORMANCE.md):
- RealtimeSanitizer on Linux/macOS: every `crates/strom-dsp/tests/rtsan*.rs`
  file runs under RealtimeSanitizer, all in one run:
  `RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test suite rtsan`.
  See [Real-time safety](PERFORMANCE.md#real-time-safety).
- Instruction-count benchmarks with gungraun on Linux. See
  [Instruction counts](PERFORMANCE.md#instruction-counts-gungraun-linux).
- Wall-clock benchmarks: `cargo bench -p strom-dsp`. See
  [Wall-clock benchmarks](PERFORMANCE.md#wall-clock-benchmarks-criterion).
