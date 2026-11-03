# `stromdsp-ducker`: lower the programme while a key is present

A utility element with no latency. The bypass and latency rules shared by
all elements are in the [element overview](README.md#shared-conventions).

## Pads and behaviour

A `GstAggregator` with three pads:
- an always pad `sink` for the programme;
- a request pad `sidechain` for the key (F32LE interleaved, any rate and
  channel count, at most one);
- an always pad `src`.

The output has the programme's caps, segment and timestamps. The key detector
runs at the key's own rate, so the key is never resampled:
- an optional high-pass;
- the linked peak over the key's channels;
- a 10 ms sine-calibrated RMS.

**Live pipelines.** A missing, stalled or unlinked key never blocks. The
element waits for the key at most the pipeline latency; raise it with the
inherited `latency` property if the key has more jitter. Missing key audio
reads as silence, so the programme holds and then fades back to unity.

**Non-live pipelines.** Request `sidechain` only if it will be fed or reach
EOS.

Toggling `enabled` crossfades over 20 ms, then the programme passes bit-exact.
The key is still consumed while disabled.

## Properties

Units are dB, ms and Hz. Every property is mutable while PLAYING unless
noted.

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `threshold` | float | dB (key level, sine-calibrated RMS) | −80 … 0 | −30 |
| `depth` | float | dB (programme gain at full ducking) | −80 … 0 | −12 |
| `attack` | float | ms (unity to full depth, linear in dB) | 0.1 … 2000 | 20 |
| `hold` | float | ms | 0 … 5000 | 300 |
| `release` | float | ms (full depth back to unity, linear in dB) | 1 … 10000 | 500 |
| `key-highpass` | float | Hz, 0 = off | 0 … 20000 | 100 |
| `range-mode` | enum `StromDspDuckerRangeMode` | | `fixed` (0), `proportional` (1) | `fixed` |
| `meter-interval` | uint | ms of running time, 0 = off (not controllable) | 0 … 10000 | 100 |

## Range modes

- `fixed`: duck by `depth` while the key is above `threshold`.
- `proportional`: 1 dB of reduction per dB the key is above `threshold`, up to
  `depth`.

## Meter message

The element posts an element message with structure name
`stromdsp-ducker` at the first buffer, and then at most once per
`meter-interval`:

| Field | Type | Meaning |
|---|---|---|
| `running-time`, `stream-time` | guint64 | ns, end of the triggering buffer |
| `reduction` | gdouble | dB (≤ 0), gain applied |
| `key-level` | gdouble | dBFS, sine-calibrated RMS after the key filter; −200 for silence |
| `key-active` | gboolean | key above threshold or within hold |
| `key-present` | gboolean | real key samples covered part of that buffer |

## Why these defaults

The defaults duck music or programme under a
presenter's microphone ("talk-over") out of the box. The levels assume EBU
R68/R128 alignment, where speech averages around −20 dBFS.
- `threshold` −30 dB: well below a talker at alignment level, so speech
  triggers it reliably, and well above the noise floor of an open studio
  microphone (< −50 dBFS), so the mic alone does not duck.
- `depth` −12 dB: the programme drops clearly below the voice for
  intelligibility but stays audible as a bed, without the hole of a full mute.
- `attack` 20 ms: the programme makes room within the first syllable, as a
  fade rather than a step.
- `hold` 300 ms: longer than the gap between syllables and short words
  (200–250 ms), so the programme does not pump up between words.
- `release` 500 ms: a smooth fade back after the talker stops, slower than
  the attack so the return is not abrupt.
- `key-highpass` 100 Hz: rumble, handling noise and plosive pops on the key
  mic do not trigger ducking, while the speech band above it does.
- `range-mode` `fixed`: a constant, predictable depth whatever the talker's
  level.
- `meter-interval` 100 ms: ten updates per second, smooth enough for an
  operator display; the same as `stromdsp-meter`.

## Example

```sh
gst-launch-1.0 -m \
  audiotestsrc is-live=true wave=sine freq=220 volume=0.5 \
    ! audio/x-raw,format=F32LE,rate=48000,channels=2 \
    ! stromdsp-ducker name=duck threshold=-30 depth=-12 attack=20 hold=300 release=500 \
    ! audioconvert ! autoaudiosink \
  audiotestsrc is-live=true wave=pink-noise volume=0.2 \
    ! audio/x-raw,format=F32LE,rate=16000,channels=1 ! duck.sidechain
```
