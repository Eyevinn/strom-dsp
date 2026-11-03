# Performance and real-time safety

How to measure `strom-dsp`, how to check that the processing path is
real-time safe, and the policies behind both.

## Build profile

`[profile.release]` (and `bench`, which inherits it) uses `opt-level = 3`,
`lto = "fat"`, `codegen-units = 1` and `debug = "line-tables-only"`. Builds
target the generic CPU of the platform; never set `target-cpu=native`, since
the plugin ships as one binary for many machines. `panic` stays at `unwind`:
gstreamer-rs turns a panic in an element vfunc into an element error, while
`panic = "abort"` would take down the host (Strom).

## Wall-clock benchmark (criterion)

```sh
cargo bench -p strom-dsp --bench strip
```

Runs on any platform. Results land in `target/criterion/`. Wall-clock numbers
depend on the machine and its load, so compare runs only on the same machine.

## Instruction counts (gungraun)

`benches/strip_gungraun.rs` counts instructions with Valgrind Callgrind, which
is deterministic and therefore usable on shared CI runners. It measures one
1024-frame, 48 kHz stereo buffer through the channel strip (gate → compressor
→ 4-band EQ: HPF + 2 peaking + high shelf → limiter) and through each
processor alone, after a warm-up. Setup is not counted.

Linux only (Valgrind). The runner binary must match the `gungraun` crate
version pinned in `crates/strom-dsp/Cargo.toml`:

```sh
sudo apt-get install valgrind
cargo install --locked gungraun-runner --version 0.20.0
cargo bench -p strom-dsp --features gungraun --bench strip_gungraun
```

The bench is declared with `required-features = ["gungraun"]`, so plain
`cargo bench`/`cargo test` never build it. Gungraun compares each run with the
previous one in `target/gungraun/` and prints the difference. No regression
limits are set yet, so the CI job is report-only. In Docker, Valgrind needs
`--security-opt seccomp=unconfined` (gungraun disables ASLR with `setarch`).

## Real-time safety (RealtimeSanitizer)

`tests/rtsan.rs` runs every processor and a full strip over many buffers of
varying size, with new parameters before every buffer. `set_params`,
`process`, `latency_frames` and periodic `reset` run inside a
`ScopedSanitizeRealtime` scope. RTSan aborts with a stack trace on any
allocation, lock, sleep, blocking system call or other unsafe call in that
scope.

```sh
RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test rtsan
```

Linux and macOS (x86-64 and aarch64). `RTSAN_ENABLE` is read at **build** time
by `rtsan-standalone`, which then downloads the pre-built sanitizer runtime
with `curl` and links it. Without the variable the scopes compile to nothing,
and the test is only a smoke test.

To confirm the sanitizer is really active, run the canary. It allocates
inside a scope and **must abort**:

```sh
RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test rtsan \
  -- --ignored --exact canary_allocation_is_detected
```

CI runs both and fails if the canary passes. The counting-allocator
tests in plain `cargo test` cover allocation on every platform. RTSan also
catches locks and system calls.

## Checking vectorisation (cargo-show-asm)

```sh
cargo install --locked cargo-show-asm
cargo asm -p strom-dsp --lib                       # list functions
cargo asm -p strom-dsp --lib "<strom_dsp::gate::Gate as strom_dsp::Processor>::process" --rust
```

`--rust` interleaves source lines (needs `rustup component add rust-src`).
Small `#[inline]` helpers (e.g. in `dynamics::frame`) get no symbol of their
own. Look at the `process` functions they are inlined into. Packed
instructions (`fmul v0.4s` on aarch64, `mulps`/`vmulps` on x86-64) show a
loop was vectorised; scalar forms (`fmul s0`, `mulss`) show it was not. The
listing follows the release profile, so it shows what ships.

## Explicit SIMD in the biquad/EQ kernels

The biquad and equalizer kernels use `fearless_simd` vectors with runtime
dispatch (see the `strom_dsp::biquad` module docs). The level is detected in
`configure`/`Biquad::new` and dispatched once per `process` call. Channels
are lanes at the native `f64` width: 2 on NEON/SSE, 4 on AVX2 and 8 on
AVX-512. Hosts and tests can force a level with `set_simd_level`.

FMA policy: fused on NEON/AVX2/AVX-512 and unfused on SSE. The scalar
fallback matches the platform baseline (fused on aarch64). Levels with the
same policy are bit-identical. `tests/simd_equivalence.rs` checks this
against `Level::fallback()`, and `tests/equalizer_fallback.rs` reruns the
whole EQ suite on the fallback.

Measured on an Apple M5 Pro (NEON), 4-band EQ, 480 frames, criterion median
(the machine was under moderate load, so expect about ±5 % run to run):

| Case | Before (autovectorised `[f64; 2]`) | Explicit SIMD |
|---|---|---|
| mono | 1.16 µs | 0.98 µs |
| stereo | 1.30 µs | 1.02 µs |
| 8 channels | 5.03 µs | 3.93 µs |
| stereo, gain automated | 3.07 µs | 2.68 µs |

Most of the gain comes from two changes:

- **Compile-time group widths:** loads and stores have a fixed size, with no
  per-lane loops.
- **DF-I evaluation order:** the stage input enters one FMA before the `y1`
  term, so the input-to-output path of a stage is 2 FMAs instead of
  mul + FMA + add + FMA. The loop-carried path stays 1 FMA.

Variants measured on NEON and rejected:

- **Skewed (software-pipelined) schedule** with channels as lanes (stage k
  processes frame t−k): 2–6 % slower.
- **Bands as lanes ("wavefront") for mono**, one lane shift per sample:
  about 2.5× slower in the prototype. Even with ideal codegen, the
  loop-carried chain (lane shift → FMA → FMA, ≥ 10 cycles) is longer than the
  ~9 cycles/frame the per-channel path already reaches.
- **Input term last** (`y = x·b0 + (y1·(−a1) + …)`): 11 % slower, because the
  `y1` recurrence becomes 2 FMAs.
- **`f64x4` groups on NEON** (two registers per state): 60 % slower for 8
  channels (register pressure).
- **Unrolling by two frames:** no gain.

The x86 paths (SSE2/SSE4.2/AVX2/AVX-512) are verified for correctness by the
same tests on x86-64 CI, but no x86 timings have been taken. `f64x8` on
AVX-512 is chosen on the instruction-count argument (one register per state,
32 registers), not on a measurement.

## Denormal policy

We do **not** set flush-to-zero / denormals-are-zero (FTZ/DAZ) in the CPU:

- Rust's documentation treats changing MXCSR (x86) or FPCR (aarch64) outside
  inline assembly as undefined behaviour: the compiler assumes the default
  floating-point environment.
- The FP mode is per-thread state owned by the host. A library that changes
  it on a GStreamer streaming thread would affect other elements too.
- It is not needed. Every recursive state (envelopes, smoothers, biquad
  states, limiter release) is flushed to exactly zero below 1e-15 (about
  −300 dB) or snapped to its target, so it never enters the subnormal range.

`tests/denormals.rs` enforces this. Each processor gets a loud signal and
then 6 s of exact silence at 8/48/192 kHz, with extreme parameters: longest
releases, 1 Hz and Q 100 EQ bands, ±60 dB gains, maximum makeup. The test
asserts that no output sample and no exposed state (`current_gain`,
`gain_reduction_db`) is ever subnormal. It checks values, not timing.

## No thread pools (rayon) in the streaming path

Processing stays on the caller's thread. We do not fan out to rayon or any
other worker pool, for three reasons:

- **Wake-up latency.** A 1024-frame buffer at 48 kHz is about 21 ms of audio,
  and one strip processes it in microseconds. Waking pool threads and joining
  them costs about as much as the work itself, and the worst case is
  unbounded. An occasional late wake-up means a missed deadline.
- **Priority inversion.** GStreamer streaming threads may run with raised
  (real-time) priority. Pool workers run at normal priority, so the
  real-time thread would wait on lower-priority threads. Those can be
  preempted by anything else on the system.
- **Parallelism already exists.** Strom runs dozens of strips, each on its
  own streaming thread. The OS spreads those threads over the cores. Adding a
  second level of parallelism inside each strip would only add contention.

Speed-ups therefore come from per-thread work: better algorithms, fewer
transcendental calls, and SIMD.
