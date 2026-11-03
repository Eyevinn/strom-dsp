# `stromdsp-deesser`: sibilance reduction

A utility element with no latency. The bypass, caps and latency rules shared
by all elements are in the [element overview](README.md#shared-conventions).

## Behaviour

A linked-channel de-esser.
- A 12 dB/octave high-pass detector at `frequency` drives a gain reduction.
  Sibilance is held at `threshold`, and the reduction is at most
  `max-reduction` dB.
- `split-band` reduces only the band above `frequency`, split by an LR4
  crossover. With no reduction the bands sum to a flat all-pass.
- `wideband` reduces the whole signal. It goes through the same all-pass
  split, so mode changes glide.
- No latency. Toggling `enabled` crossfades over 20 ms, then the output is
  bit-exact.

## Properties

Units are dB, ms and Hz. Every property is mutable while PLAYING unless
noted.

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `frequency` | float | Hz (limited to 0.45·fs) | 1000 … 20000 | 6000 |
| `threshold` | float | dBFS (detector peak) | −60 … 0 | −30 |
| `max-reduction` | float | dB (positive = attenuation) | 0 … 40 | 10 |
| `attack` | float | ms (63 % time constant) | 0.01 … 200 | 1 |
| `release` | float | ms (63 % time constant) | 1 … 2000 | 50 |
| `mode` | enum `StromDspDeesserMode` | | `split-band` (0), `wideband` (1) | `split-band` |

## Why these defaults

The defaults tame sibilance on live speech without
dulling the voice.
- `frequency` 6 kHz: the energy of "s" and "sh" sounds sits mostly from
  about 5 to 10 kHz, while the body and most consonants of the voice lie
  below.
- `threshold` −30 dBFS: at EBU R68/R128 alignment, sharp sibilants exceed it
  in the high-passed detector, while normal high-frequency content mostly
  stays below.
- `max-reduction` 10 dB: enough to tame harsh sibilance, not so much that
  the talker starts to sound lisping.
- `attack` 1 ms: sibilants start abruptly, so the reduction must be in place
  within their first milliseconds.
- `release` 50 ms: recovers before the following vowel, so the next syllable
  is not dulled.
- `mode` `split-band`: only the band above `frequency` is reduced, so the
  level and body of the voice are untouched.

## Example

```sh
gst-launch-1.0 pulsesrc ! audioconvert ! audio/x-raw,format=F32LE,layout=interleaved ! \
  stromdsp-deesser frequency=6500 threshold=-32 max-reduction=8 mode=split-band ! \
  audioconvert ! autoaudiosink
```
