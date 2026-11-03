# `stromdsp-equalizer`: parametric EQ, up to 8 biquad bands

A channel-strip element: an in-place `BaseTransform` on F32LE interleaved
audio, any rate and channel count, with no latency. Every property is
controllable while PLAYING, except `num-bands`. The bypass, caps and latency
rules shared by all elements are in the
[element overview](README.md#shared-conventions).

## Properties

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `num-bands` | uint | | 1 … 8 (NULL/READY only) | 4 |
| `bandN-enabled` | bool | | | `true` |
| `bandN-type` | int | see below | 0 … 9 | 0 |
| `bandN-frequency` | float | Hz | 1 … 24000 | 1000 |
| `bandN-gain` | float | dB | −60 … 60 | 0 |
| `bandN-q` | float | | 0.01 … 100 | 0.707 |

All eight `bandN-*` sets (N = 0 … 7) always exist. Bands at index ≥
`num-bands` are inactive.

`bandN-type` values:

| Value | Type |
|---|---|
| 0 | off |
| 1 | low-pass |
| 2 | high-pass |
| 3 | band-pass, constant skirt gain |
| 4 | band-pass, 0 dB peak |
| 5 | notch |
| 6 | all-pass |
| 7 | peaking |
| 8 | low shelf |
| 9 | high shelf |

## Behaviour

Coefficients come from the RBJ Audio EQ Cookbook, using the Q form for the
shelves as well. `gain` applies to peaking and the shelves only. Frequency is
clamped just below Nyquist. Parameter changes glide smoothly, and type changes
crossfade, so there are no clicks.

## Why these defaults

The equalizer is neutral by default: every band is
`off` (type 0) with 0 dB gain, so the element is a transparent pass-through
until an operator shapes it. Any built-in curve would be wrong for most sources,
and equalisation is always a decision for a specific microphone, room or
programme. The parked band settings make a first edit easy: 1000 Hz is the
middle of the speech band, Q 0.707 is the Butterworth value (a moderate width
for shelves and peaks), and 4 bands is a common size for a channel-strip EQ.

## Example

An 80 Hz high-pass (type 2) and a 3 dB presence peak at 3 kHz (type 7):

```sh
gst-launch-1.0 autoaudiosrc ! audioconvert ! audio/x-raw,format=F32LE,layout=interleaved ! \
  stromdsp-equalizer num-bands=2 band0-type=2 band0-frequency=80 \
                     band1-type=7 band1-frequency=3000 band1-gain=3 band1-q=1 ! \
  audioconvert ! autoaudiosink
```
