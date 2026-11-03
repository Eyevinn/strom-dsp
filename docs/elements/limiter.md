# `stromdsp-limiter`: lookahead peak limiter

A channel-strip element: an in-place `BaseTransform` on F32LE interleaved
audio, any rate and channel count, with linked channels. It is the only
channel-strip element with latency (`lookahead`, see below). The bypass, caps
and latency rules shared by all elements are in the
[element overview](README.md#shared-conventions).

## Properties

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `threshold` | float | dB (ceiling) | −60 … 0 | −1 |
| `attack` | float | ms | 0.01 … 50 | 1 |
| `release` | float | ms | 1 … 2000 | 200 |
| `lookahead` | float | ms | 0 … 20 | 5 |

## Behaviour

**Ceiling guarantee.** No output sample exceeds `threshold`, for any input and
also with `lookahead = 0`. The gain ramp is built so that it reaches the
required gain before the peak arrives, and it rounds only downwards. A final
safety clip exists as a guard only; tests show it never triggers on normal
material.

**Release hold.** After the last sample that needed gain reduction, the
limiter holds the gain for min(20 ms, release) before the release starts. This
keeps bass undistorted at fast attack and release, because the gain no longer
rises between the waveform peaks of low notes. The ceiling guarantee is
unaffected.

**Latency.**
- The audio is delayed by `lookahead`, and the element adds it to the LATENCY
  query.
- A change of `lookahead` posts a `Latency` message.
- Timestamps pass through unchanged.

**Disabled behaviour.** With `enabled=false` the limiter applies unity gain but
**keeps the delay**, so the latency does not jump when an operator toggles it.

## Why these defaults

The defaults give a programme-safe peak ceiling. They
are strom-dsp's own choice from textbook and broadcast practice
(`strom_dsp::presets`, see [PROVENANCE.md](../../PROVENANCE.md)).
- `threshold` −1 dBFS: the EBU R128 maximum true-peak level (−1 dBTP). This is
  a sample-peak ceiling, so inter-sample peaks can still exceed it slightly.
- `attack` 1 ms: a smooth gain ramp instead of a step on each peak. It fits
  well inside the lookahead, so the ceiling still holds.
- `release` 200 ms: suited to programme material. It is long compared with
  bass periods, so there is no low-frequency distortion, and short enough to
  recover between phrases without audible pumping.
- `lookahead` 5 ms: 5 ms of latency, which Strom relies on. It covers the
  attack with margin.

## Example

```sh
gst-launch-1.0 autoaudiosrc ! audioconvert ! audio/x-raw,format=F32LE,layout=interleaved ! \
  stromdsp-limiter threshold=-1 lookahead=5 release=200 ! audioconvert ! autoaudiosink
```
