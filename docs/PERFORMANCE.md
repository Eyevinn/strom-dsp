# Performance and real-time safety

How to run and read the benchmarks and the real-time-safety checks. Design
decisions and their reasons are in [PROVENANCE.md](../PROVENANCE.md) and
in the commit messages.

## Build profile

`[profile.release]` (and `bench`, which inherits it) uses `opt-level = 3`,
`lto = "fat"`, `codegen-units = 1` and `debug = "line-tables-only"`.

- Never set `target-cpu=native`. The plugin ships as one binary, and SIMD
  levels are chosen at runtime.
- `panic` stays `unwind`. A panic in an element then becomes an element error;
  `abort` would take down the host.

## Wall-clock benchmarks (criterion)

```sh
cargo bench -p strom-dsp                       # everything
cargo bench -p strom-dsp --bench strip         # channel strip and its processors
cargo bench -p strom-dsp --bench strip -- eq   # filter by name
```

| Bench | Covers |
|---|---|
| `strip` | gate, compressor, EQ, limiter, and the full strip |
| `filters` | Butterworth and crossover |
| `sampling` | oversampler and STFT |
| `meter` | loudness, correlation, band correlation, mono detector |
| `sidechain` | keyed compressor and expander |
| `ducker` | ducker |
| `dynamics_1b` | de-esser |
| `channel_tools` | delay and stereo |
| `cleanup` | input cleanup |
| `agc` | loudness detector and wideband AGC |
| `multiband` | multiband compressor (4 bands) |
| `truepeak` | true-peak limiter |
| `broadcast` | full broadcast chain with all meters |

All benches use 48 kHz. Most use 480-frame (10 ms) buffers; some also run
1024 frames.

**Reading the results:** the share of one core is time per buffer ÷ buffer
duration. For example, 5 µs per 10 ms buffer is 0.05 %. Results land in
`target/criterion/`, and criterion prints the change against the previous run
on the same machine. Wall-clock numbers depend on the machine and its load:
compare runs only on the same, otherwise idle, machine.

## Instruction counts (gungraun, Linux)

| Bench | Covers |
|---|---|
| `strip_gungraun` | gate, compressor, EQ, limiter, and the full strip |

`benches/strip_gungraun.rs` counts instructions for one 1024-frame, 48 kHz
stereo buffer through the strip and through each processor, using Valgrind
Callgrind. The count is deterministic, so it also works on shared CI runners.
The runner version must match the `gungraun` version pinned in
`crates/strom-dsp/Cargo.toml`:

```sh
sudo apt-get install valgrind
cargo install --locked gungraun-runner --version 0.20.0
cargo bench -p strom-dsp --features gungraun --bench strip_gungraun
```

- Each run is compared with the previous one in `target/gungraun/`.
- CI runs it report-only, with no regression limits.
- In Docker, Valgrind needs `--security-opt seccomp=unconfined`.

## Real-time safety

**Counting allocator.** Plain `cargo test` includes tests that run every
processor with a counting `#[global_allocator]`. They assert zero allocations
in `process`/`set_params` after setup, on every platform.

**RealtimeSanitizer (Linux and macOS).** Every `crates/strom-dsp/tests/rtsan*.rs`
file runs the processing path inside a sanitizer scope. The sanitizer aborts
with a stack trace on any allocation, lock, blocking call or other unsafe call.

```sh
RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test rtsan
# the canary allocates inside a scope and MUST abort (proves RTSan is active):
RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test rtsan \
  -- --ignored --exact canary_allocation_is_detected
```

`RTSAN_ENABLE` is read at build time; it makes `rtsan-standalone` download and
link the sanitizer runtime. Without it, the scopes compile to nothing. CI runs
every `rtsan*.rs` file and fails if the canary passes.

**Denormals.** `tests/denormals.rs` feeds loud input followed by long silence
at extreme parameters. It asserts that no output sample and no exposed state
is ever subnormal.

## Checking vectorisation (cargo-show-asm)

```sh
cargo install --locked cargo-show-asm
cargo asm -p strom-dsp --lib                       # list functions
cargo asm -p strom-dsp --lib "<strom_dsp::gate::Gate as strom_dsp::Processor>::process" --rust
```

- Packed instructions show that a loop was vectorised: `fmul v0.2d`/`v0.4s` on
  aarch64, `vmulpd`/`mulps` on x86-64.
- Scalar forms (`fmul d0`, `mulsd`) show that it was not.
- Small inlined helpers have no symbol of their own; look at the `process`
  function they are inlined into.
