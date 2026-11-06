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
| `declipper` | declipper: clean, speech+music clipped 6 / 10 dB, tones clipped 10 dB, runs too long; first a per-buffer table (mean and worst 10 ms buffer per latency setting) |
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

**Current numbers.** These are criterion medians from one full
`cargo bench -p strom-dsp` run at v0.2.3 (2026-11-06), on an Apple M5 Pro
(NEON), 48 kHz stereo, 480 frames (10 ms). The machine ran on mains power in
High Power mode, not idle (load average about 2 from unrelated work), so
treat them as upper bounds.

On a laptop, measure on mains power with Low Power Mode off: on battery in
Low Power Mode the same code reads about 1.5× slower (the full broadcast
chain 137 µs instead of 89 µs), and the numbers are not comparable.

| Bench | Case | Time |
|---|---|---|
| `strip` | gate / compressor / limiter / strip | 1.45 / 3.59 / 3.15 / 8.75 µs |
| `strip` | EQ 4-band mono / stereo / 8 ch / stereo automated | 0.98 / 1.01 / 3.90 / 2.86 µs |
| `broadcast` | full chain defaults `ebu-r128`, `standard` / `low` latency | 89.0 / 89.1 µs |
| `broadcast` | full chain defaults `streaming`, `standard` / `low` latency | 102.1 / 96.6 µs |
| `broadcast` | full chain, library defaults (expanders and bass clipper off) | 75.7 µs |
| `broadcast` | final clipper alone, standard: idle / clipping | 1.65 / 47.9 µs |
| `broadcast` | final clipper alone, low latency: idle / clipping | 1.64 / 26.2 µs |
| `truepeak` | 5 ms lookahead | 12.2 µs |
| `spike_guard` | no events / a click every 50 ms | 12.5 / 12.7 µs |
| `multiband` | 4 bands, plain / with expander on every band and bass clipper | 18.3 / 34.0 µs |
| `agc` | detector / AGC / ebur128 read every block | 1.01 / 1.26 / 170 µs |
| `filters` | Butterworth 4 / 8, LR4 crossover 3 / 5 bands | 0.78 / 0.96 / 3.36 / 7.20 µs |
| `sampling` | oversampler 2× / 4× / 8× up+down, 4× peaks; STFT 1024/256 / 2048/512 | 7.01 / 11.1 / 18.9 / 5.77; 6.84 / 7.44 µs |
| `meter` | loudness / + true peak / correlation / bands / mono detector | 7.00 / 10.3 / 0.66 / 5.47 / 0.68 µs |
| `sidechain` | keyed compressor / expander / keyed expander | 4.34 / 1.75 / 2.26 µs |
| `ducker` | mono key / proportional / 16 kHz key / no key | 2.88 / 3.38 / 1.67 / 1.98 µs |
| `dynamics_1b` | de-esser split / wide | 8.35 / 8.38 µs |
| `channel_tools` | delay 0 / 20 ms / 8 ch / crossfading; stereo default / width+balance | 0.11 / 0.15 / 2.25 / 2.15; 0.05 / 0.34 µs |
| `cleanup` | off / DC + HP4 / 10 hum notches / all | 0.06 / 1.49 / 3.11 / 4.55 µs |
| `declipper` | clean / speech+music 6 dB / 10 dB / tones 10 dB / runs too long (cost scales with the amount of clipping; worst single buffer 3.8 % of 10 ms) | 9.25 / 33.7 / 81.0 / 257 / 16.4 µs |

**x86-64.** A full run on a cloud VM (CPU model AMD EPYC-Milan, 6 vCPU;
the guest reports FMA, AVX2 and AVX-512; Linux 6.8, load average about 1),
at the M5 Pro run's commit plus the oversampler fix of
[#8](https://github.com/Eyevinn/strom-dsp/pull/8). A VM shares its host,
so treat these as upper bounds too.

| Bench | Case | Time |
|---|---|---|
| `strip` | gate / compressor / limiter / strip | 1.87 / 4.08 / 21.0 / 30.9 µs |
| `strip` | EQ 4-band mono / stereo / 8 ch | 4.04 / 4.11 / 6.64 µs |
| `broadcast` | full chain defaults `ebu-r128`, `standard` / `low` latency | 190 / 190 µs |
| `broadcast` | full chain defaults `streaming`, `standard` / `low` latency | 216 / 205 µs |
| `broadcast` | final clipper alone, standard: idle / clipping | 3.03 / 117 µs |
| `broadcast` | final clipper alone, low latency: idle / clipping | 2.99 / 60.5 µs |
| `truepeak` | 5 ms lookahead | 48.0 µs |
| `multiband` | 4 bands, plain / with expander on every band and bass clipper | 32.6 / 67.1 µs |
| `sampling` | oversampler 2× / 4× / 4× scalar fallback / 8× up+down | 9.80 / 17.8 / 45.6 / 32.1 µs |
| `declipper` | clean / speech+music 6 dB / 10 dB / tones 10 dB (worst single buffer 6.2 % of 10 ms) | 22.1 / 54.3 / 139 / 370 µs |

Most processors take about 1 … 3.5× their M5 Pro time here. The limiter,
the true-peak limiter and the 4-band EQ take 4 … 7×; this is not yet
investigated. Forced to each level, the limiter takes the same time at
AVX-512, AVX2, SSE4.2 and the scalar fallback (21.5 … 22.4 µs), so its
extra cost lies outside its vector kernels; the stereo EQ is fastest on the
scalar fallback (3.57 µs against 4.03 µs at AVX-512).

Before #8 the oversampler's vector levels were slower than its scalar
fallback on x86 (4×: 62.7 µs at AVX-512, 97.8 µs at AVX2, 47.0 µs
fallback). A local closure in the decimation kernel was not inlined, and
code outlined from `vectorize` is compiled without its target features, so
every vector operation in it became a call. After a change to a kernel,
check on x86 that none of its closures appears as a symbol of its own in
the disassembly (`objdump -d -C … | grep 'kernel.*closure'`): aarch64
does not show the problem, because NEON is part of its baseline.

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
RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test suite rtsan
# the canary allocates inside a scope and MUST abort (proves RTSan is active):
RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test suite \
  -- --ignored --exact rtsan::canary_allocation_is_detected
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
(mains power, High Power mode, load average about 2 from other work), with
criterion's default warm-up and measurement, so treat them as upper bounds:

```sh
cargo bench -p strom-dsp --bench broadcast -- full_chain
```

The full chain (spike guard, AGC, 4-band multiband with expanders and bass
clipper, final clipper, limiter, governor, all meters) with the `ebu-r128`
defaults takes 89.0 µs per 10 ms stereo buffer in the `standard` latency
mode (0.89 % of one core) and 89.1 µs in `low` (0.89 %); with the
`streaming` defaults 102.1 µs (1.02 %) and 96.6 µs (0.97 %). The low mode
costs the same or less: its clipper runs at 4× instead of 8×. The
clipper's early-out keeps it cheap on programme that does not reach its
knee (1.65 µs); driven 6 dB over its threshold on every buffer it takes
47.9 µs (standard, 8×) or 26.2 µs (low latency, 4×). A chain that clipped
on every buffer would therefore come to roughly 1.1 … 1.5 % of one core,
under the 2 % budget.

