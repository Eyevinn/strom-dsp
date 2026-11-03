# `stromdsp-cleanup`: DC, rumble and hum removal

A utility element with no latency. The bypass, caps and latency rules shared
by all elements are in the [element overview](README.md#shared-conventions).

## Behaviour

Input cleanup for contributions and field recordings. The stages run in this
order:
1. **DC blocker:** first-order high-pass, −3 dB at 5 Hz.
2. **High-pass:** Butterworth, order 1–8.
3. **Hum notches:** at 50 or 60 Hz plus harmonics. Every notch has the same
   bandwidth, f0/`hum-q`.
   - Each enabled harmonic is attenuated by ≥ 40 dB.
   - The passband between notches stays within 0.5 dB over the whole `hum-q`
     range.
   - Notches at or above 0.49·fs are skipped.

Every stage is off by default, so the default element is a bit-exact
passthrough. Switching a stage on or off crossfades over 10 ms. No latency.
Toggling `enabled` crossfades over 20 ms, then the output is bit-exact.

## Properties

Units are dB, ms and Hz. Every property is mutable while PLAYING unless
noted.

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `dc-block` | bool | | | `false` |
| `highpass` | float | Hz (−3 dB), 0 = off | 0 … 1000 | 0 |
| `highpass-order` | uint | 6 dB/oct per order | 1 … 8 | 2 |
| `hum` | enum `StromDspCleanupHum` | | `off` (0), `50` (50), `60` (60) | `off` |
| `hum-harmonics` | uint | number of notches | 1 … 10 | 5 |
| `hum-q` | float | Q of the fundamental; bandwidth f0/`hum-q` | 6 … 100 | 10 |

## Why these defaults

Every stage is off, so inserting the element never
changes the sound: whether a source needs a low cut, and whether the local
mains hum is at 50 or 60 Hz, depends on the source and the site. The parked
values make switching a stage on safe:
- `highpass-order` 2 (12 dB/octave): a gentle microphone-style low cut that
  removes rumble without thinning the voice above the corner.
- `hum-harmonics` 5: the fundamental up to 250 Hz (50 Hz mains) or 300 Hz
  (60 Hz), where most hum energy lies.
- `hum-q` 10: notches 5 or 6 Hz wide, narrow enough to leave the programme
  between them intact and wide enough for normal mains frequency drift.

## Example

```sh
gst-launch-1.0 filesrc location=field.wav ! wavparse ! audioconvert ! \
  audio/x-raw,format=F32LE,layout=interleaved ! \
  stromdsp-cleanup dc-block=true highpass=70 highpass-order=4 hum=50 hum-harmonics=8 ! \
  audioconvert ! autoaudiosink
```
