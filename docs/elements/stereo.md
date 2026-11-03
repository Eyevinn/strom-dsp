# `stromdsp-stereo`: stereo utility

A utility element with no latency. The bypass and latency rules shared by all
elements are in the [element overview](README.md#shared-conventions).

## Behaviour

A stereo utility for exactly 2 channels (caps `channels=2`). It applies one
2×2 matrix: balance · width · mode · polarity.
- **Polarity:** applied to the inputs.
- **M/S:** `M = (L+R)/2` and `S = (L−R)/2`. Decode is `L = M+S`, `R = M−S`, so
  encode followed by decode is exact.
- **Width:** scales S. 0 is mono, 1 is unchanged, 2 is wide. There is no gain
  compensation.
- **Balance:** constant power, with `θ = (b+1)·π/4`, `gL = √2·cos θ` and
  `gR = √2·sin θ`. The centre is unity; at ±1 the remaining side is +3 dB.

Changes ramp over 10 ms. Pure copy, swap and invert are applied bit-exactly.
The defaults, and `enabled=false` once its ramp ends, are bit-exact
passthrough. No latency.

## Properties

Units are dB, ms and Hz. Every property is mutable while PLAYING unless
noted.

| Property | Type | Range / values | Default |
|---|---|---|---|
| `enabled` | bool | | `true` |
| `mode` | enum `StromDspStereoMode` | `stereo`, `mono`, `left-to-both`, `right-to-both`, `swap`, `ms-encode`, `ms-decode` | `stereo` |
| `mono-law` | enum `StromDspStereoMonoLaw` | `minus-3db` (1/√2), `minus-6db` (1/2) | `minus-3db` |
| `invert-left`, `invert-right` | bool | | `false` |
| `balance` | float | −1 … 1 | 0 |
| `width` | float | 0 … 2 | 1 |

## Why these defaults

`mode` `stereo`, `width` 1, `balance` 0 and no
inversion: the element is a bit-exact passthrough until an operator changes
it, so inserting it never alters the stereo image. `mono-law` `minus-3db`:
when an operator folds to mono, uncorrelated material (wide stereo, room
sound) keeps its power, and centred material rises only 3 dB. This is the
usual compromise for mixed programme; `minus-6db` keeps centred material
(a mono voice) at its level instead.

## Example

```sh
gst-launch-1.0 audiotestsrc ! audio/x-raw,format=F32LE,channels=2,rate=48000 ! \
  stromdsp-stereo mode=mono mono-law=minus-6db ! autoaudiosink
```
