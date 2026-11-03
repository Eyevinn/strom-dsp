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
| `multiband` | multiband compressor (4 bands); with per-band expander and bass clipper |
| `truepeak` | true-peak limiter |
| `broadcast` | full broadcast chain with all meters, presets' defaults in both latency modes; final clipper alone |
| `spike_guard` | spike guard, without events and with a click every 50 ms |

All benches use 48 kHz. Most use 480-frame (10 ms) buffers; some also run
1024 frames.

**Reading the results:** the share of one core is time per buffer ÷ buffer
duration. For example, 5 µs per 10 ms buffer is 0.05 %. Results land in
`target/criterion/`, and criterion prints the change against the previous run
on the same machine. Wall-clock numbers depend on the machine and its load:
compare runs only on the same, otherwise idle, machine.

**Current numbers.** These are criterion medians on an Apple M5 Pro (NEON),
48 kHz stereo, 480 frames (10 ms). The machine was under heavy unrelated load
(load average about 12), so treat them as upper bounds.

| Bench | Case | Time |
|---|---|---|
| `strip` | gate / compressor / limiter / strip | 2.19 / 7.91 / 5.72 / 18.3 µs |
| `strip` | EQ 4-band mono / stereo / 8 ch / stereo automated | 2.06 / 2.16 / 8.26 / 5.91 µs |
| `broadcast` (complete chain, separate run, load average 3 … 4) | full chain defaults `ebu-r128`, `standard` / `low` latency | 82.6 / 82.0 µs |
| `broadcast` (same run) | full chain defaults `streaming`, `standard` / `low` latency | 93.4 / 88.6 µs |
| `broadcast` (same run) | full chain, library defaults (expanders and bass clipper off) | 68.7 µs |
| `broadcast` | final clipper alone, standard: idle / clipping | 1.27 / 46.7 µs |
| `broadcast` | final clipper alone, low latency: idle / clipping | 1.27 / 25.4 µs |
| `truepeak` | 5 ms lookahead | 18.8 µs |
| `multiband` | 4 bands | 46.7 µs |
| `spike_guard` | no events / a click every 50 ms (separate run, load average about 4) | 12.1 / 13.9 µs |
| `multiband` | 4 bands, plain / with expander on every band and bass clipper (3 ms lookahead at the time; now 1 ms) | 18.2 / 33.3 µs (separate run, machine not idle: other builds ran in parallel) |
| `agc` | detector / AGC / ebur128 read every block | 1.55 / 1.95 / 263 µs |
| `filters` | Butterworth 4 / 8, LR4 crossover 3 / 5 bands | 1.46 / 2.18 / 7.37 / 13.6 µs |
| `sampling` | oversampler 4× up+down, 4× peaks; STFT 1024/256 | 17.6 / 9.44 / 13.3 µs |
| `meter` | loudness / + true peak / correlation / bands / mono detector | 10.7 / 17.4 / 1.23 / 12.2 / 1.70 µs |
| `sidechain` | keyed compressor / expander / keyed expander | 9.32 / 3.42 / 4.53 µs |
| `ducker` | mono key / proportional / 16 kHz key / no key | 4.52 / 5.29 / 2.52 / 3.04 µs |
| `dynamics_1b` | de-esser split / wide | 12.7 / 12.6 µs |
| `channel_tools` | delay 0 / 20 ms / 8 ch / crossfading; stereo default / width+balance | 0.16 / 0.22 / 3.42 / 3.24; 0.08 / 0.55 µs |
| `cleanup` | off / DC + HP4 / 10 hum notches / all | 0.10 / 2.28 / 5.20 / 7.19 µs |

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

**Complete broadcast chain.** Measured on the same Apple M5 Pro, not idle
(load average 3 … 4 from other work), criterion with 1 s warm-up and 3 s
measurement, so treat them as upper bounds:

```sh
cargo bench -p strom-dsp --bench broadcast -- full_chain --warm-up-time 1 --measurement-time 3
```

The full chain (spike guard, AGC, 4-band multiband with expanders and bass
clipper, final clipper, limiter, governor, all meters) with the `ebu-r128`
defaults takes 82.6 µs per 10 ms stereo buffer in the `standard` latency
mode (0.83 % of one core) and 82.0 µs in `low` (0.82 %); with the
`streaming` defaults 93.4 µs (0.93 %) and 88.6 µs (0.89 %). The low mode
is no more expensive: its clipper runs at 4× instead of 8×. The clipper's
early-out makes it nearly free on programme that does not reach its knee
(1.27 µs); driven 6 dB over its threshold on every buffer it takes 46.7 µs
(standard, 8×) or 25.4 µs (low latency, 4×), so even a chain that clips
all the time stays under the 2 % budget.

