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
| `simd_levels` | oversampler 4× / 8× up+down, EQ 4-band stereo / 8 ch, limiter, true-peak limiter, each forced to every SIMD level the CPU supports (x86-64: AVX-512, AVX2, SSE4.2; aarch64: NEON; and the scalar fallback) |

All benches use 48 kHz. Most use 480-frame (10 ms) buffers; some also run
1024 frames.

**Reading the results:** the share of one core is time per buffer ÷ buffer
duration. For example, 5 µs per 10 ms buffer is 0.05 %. Results land in
`target/criterion/`, and criterion prints the change against the previous run
on the same machine. Wall-clock numbers depend on the machine and its load:
compare runs only on the same, otherwise idle, machine.

**Current numbers.** These are criterion medians from one full
`cargo bench -p strom-dsp` run (2026-11-06, with the fixes of #8 and #13),
on an Apple M5 Pro (NEON), 48 kHz stereo, 480 frames (10 ms). The machine
ran on mains power in High Power mode, not idle (load average about 1 from
unrelated work), so treat them as upper bounds.

On a laptop, measure on mains power with Low Power Mode off: on battery in
Low Power Mode the same code reads about 1.5× slower (the full broadcast
chain 137 µs instead of 89 µs), and the numbers are not comparable.

| Bench | Case | Time |
|---|---|---|
| `strip` | gate / compressor / limiter / strip | 1.43 / 3.55 / 3.14 / 8.75 µs |
| `strip` | EQ 4-band mono / stereo / 8 ch / stereo automated | 0.97 / 1.01 / 3.87 / 2.86 µs |
| `broadcast` | full chain defaults `ebu-r128`, `standard` / `low` latency | 88.4 / 88.4 µs |
| `broadcast` | full chain defaults `streaming`, `standard` / `low` latency | 99.5 / 95.9 µs |
| `broadcast` | full chain, library defaults (expanders and bass clipper off) | 75.0 µs |
| `broadcast` | final clipper alone, standard: idle / clipping | 1.63 / 45.0 µs |
| `broadcast` | final clipper alone, low latency: idle / clipping | 1.61 / 25.6 µs |
| `truepeak` | 5 ms lookahead | 12.1 µs |
| `spike_guard` | no events / a click every 50 ms | 12.4 / 12.5 µs |
| `multiband` | 4 bands, plain / with expander on every band and bass clipper | 17.5 / 32.7 µs |
| `agc` | detector / AGC / ebur128 read every block | 1.02 / 1.26 / 168 µs |
| `filters` | Butterworth 4 / 8, LR4 crossover 3 / 5 bands | 0.77 / 0.95 / 3.30 / 7.07 µs |
| `sampling` | oversampler 2× / 4× / 8× up+down, 4× peaks; STFT 1024/256 / 2048/512 | 6.82 / 10.7 / 15.3 / 5.71; 6.81 / 7.42 µs |
| `meter` | loudness / + true peak / correlation / bands / mono detector | 6.79 / 10.1 / 0.64 / 5.35 / 0.67 µs |
| `sidechain` | keyed compressor / expander / keyed expander | 4.21 / 1.70 / 2.22 µs |
| `ducker` | mono key / proportional / 16 kHz key / no key | 2.76 / 3.30 / 1.62 / 1.95 µs |
| `dynamics_1b` | de-esser split / wide | 8.06 / 8.07 µs |
| `channel_tools` | delay 0 / 20 ms / 8 ch / crossfading; stereo default / width+balance | 0.10 / 0.15 / 2.23 / 2.12; 0.05 / 0.33 µs |
| `cleanup` | off / DC + HP4 / 10 hum notches / all | 0.05 / 1.45 / 3.06 / 4.47 µs |
| `declipper` | clean / speech+music 6 dB / 10 dB / tones 10 dB / runs too long (cost scales with the amount of clipping; worst single buffer 3.7 % of 10 ms) | 8.96 / 32.3 / 78.8 / 251 / 16.0 µs |

**x86-64.** A full run on a cloud VM (CPU model AMD EPYC-Milan, Zen 3,
6 vCPU; the guest reports FMA, AVX2 and AVX-512; Linux 6.8, load average
about 1) at the same commit. A VM shares its host, so treat these as upper
bounds too.

| Bench | Case | Time |
|---|---|---|
| `strip` | gate / compressor / limiter / strip | 1.87 / 4.06 / 21.6 / 28.7 µs |
| `strip` | EQ 4-band mono / stereo / 8 ch | 1.82 / 1.90 / 3.53 µs |
| `broadcast` | full chain defaults `ebu-r128`, `standard` / `low` latency | 165 / 164 µs |
| `broadcast` | full chain defaults `streaming`, `standard` / `low` latency | 193 / 180 µs |
| `broadcast` | final clipper alone, standard: idle / clipping | 3.05 / 115 µs |
| `broadcast` | final clipper alone, low latency: idle / clipping | 3.02 / 58.3 µs |
| `truepeak` | 5 ms lookahead | 28.9 µs |
| `multiband` | 4 bands, plain / with expander on every band and bass clipper | 22.8 / 58.0 µs |
| `sampling` | oversampler 2× / 4× / 4× scalar fallback / 8× up+down | 9.76 / 17.8 / 45.6 / 32.2 µs |
| `declipper` | clean / speech+music 6 dB / 10 dB / tones 10 dB (worst single buffer 6.0 % of 10 ms) | 22.1 / 54.1 / 139 / 368 µs |

Most processors take about 1 … 3.5× their M5 Pro time here. The limiter
takes about 7× ([#9](https://github.com/Eyevinn/strom-dsp/issues/9)); on an
Intel Core Ultra 7 165H it takes about 2×, so the gap is specific to this
CPU or VM.

Two fixes brought x86 closer.
[#8](https://github.com/Eyevinn/strom-dsp/pull/8) and
[#13](https://github.com/Eyevinn/strom-dsp/pull/13) each removed a closure
that the compiler did not inline from a vector kernel: the oversampler's
decimation (before: 4× at 62.7 µs on AVX-512 and 97.8 µs on AVX2, slower
than the 47.0 µs scalar fallback) and the biquad chain's `flush_all`
(before: EQ stereo 4.10 µs). Code outlined from `vectorize` is compiled
without its target features, so every vector operation in it becomes a
call, and state passed to it lives in memory instead of registers. After a
change to a kernel, check on x86 that none of its closures appears as a
symbol of its own in the disassembly (`objdump -d -C … | grep
'kernel.*closure'`): aarch64 does not show the problem, because NEON is
part of its baseline. The weekly extended workflow catches both cases:
`simd_gungraun` (see "Instruction counts" below) fails when a vector
kernel's instruction count grows by more than 20 %. Pull-request CI does
not run it. `cargo bench --bench simd_levels` shows each kernel's time at
every level ([#12](https://github.com/Eyevinn/strom-dsp/issues/12)). #13 also
replaced a 64-bit division per frame in the ramp's box average with a
multiply: free on Apple and recent Intel cores, but it took the true-peak
limiter on Zen 3 from 48.1 to 28.9 µs.

## Instruction counts (gungraun, Linux)

| Bench | Covers |
|---|---|
| `strip_gungraun` | gate, compressor, EQ, limiter, and the full strip (report-only) |
| `simd_gungraun` | oversampler 4× / 8× up+down, EQ 4-band stereo / 8 ch, true-peak limiter, full broadcast chain (`ebu-r128` defaults), at AVX2, with hard limits |

Both count instructions with Valgrind Callgrind. The count is
deterministic, so it also works on shared CI runners. The runner version
must match the `gungraun` version pinned in `crates/strom-dsp/Cargo.toml`:

```sh
sudo apt-get install valgrind
cargo install --locked gungraun-runner --version 0.20.0
cargo bench -p strom-dsp --features gungraun --bench strip_gungraun
cargo bench -p strom-dsp --features gungraun --bench simd_gungraun
```

- Each run is compared with the previous one in `target/gungraun/`.
- In Docker, Valgrind needs `--security-opt seccomp=unconfined`.

Both run in the weekly extended workflow (and on demand), not on pull
requests. `strip_gungraun` counts one 1024-frame stereo buffer and is
report-only. `simd_gungraun` counts one 480-frame buffer per case at
AVX2 (Valgrind does not implement AVX-512, so forcing AVX2 makes the count
independent of the runner) and fails when a count exceeds its **hard
limit**, 20 % over the count measured when the limit was set. Both known
closure regressions break the limits: #8 doubles the oversampler's counts,
and #11 raises the EQ's by 30–54 %.

The limits are absolute, so a new `rustc` or a change to a kernel can move
the counts. To update them:

1. Run `cargo bench -p strom-dsp --features gungraun --bench simd_gungraun`
   on x86-64 Linux with the stable toolchain CI uses.
2. Put each case's measured `Instructions` value into its
   `config = limit(…)` in `crates/strom-dsp/benches/simd_gungraun.rs`
   (`limit` adds the 20 %).
3. Say in the commit message why the counts changed (toolchain, an
   intended change to a kernel, …). Raising a limit to make a regression
   pass is not a reason.

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
defaults takes 88.4 µs per 10 ms stereo buffer in the `standard` latency
mode (0.88 % of one core) and 88.4 µs in `low` (0.88 %); with the
`streaming` defaults 99.5 µs (0.99 %) and 95.9 µs (0.96 %). The low mode
costs the same or less: its clipper runs at 4× instead of 8×. The
clipper's early-out keeps it cheap on programme that does not reach its
knee (1.63 µs); driven 6 dB over its threshold on every buffer it takes
45.0 µs (standard, 8×) or 25.6 µs (low latency, 4×). A chain that clipped
on every buffer would therefore come to roughly 1.1 … 1.4 % of one core,
under the 2 % budget.

