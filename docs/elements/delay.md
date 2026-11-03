# `stromdsp-delay`: intentional delay for alignment

A utility element. The bypass and latency rules shared by all elements are in
the [element overview](README.md#shared-conventions).

## Behaviour

Content alignment, such as mic alignment and lip sync. There is a common
`delay` plus per-channel `channel-delays`. The delay per channel is
`clamp(delay + offset, 0, 2000)` ms, rounded to whole frames.

> **This delay is NOT latency.** It is not reported in the LATENCY query, and
> timestamps pass through unchanged, so the audio content moves later relative
> to its timestamps. If it were reported, a latency-compensating pipeline would
> undo it.

**Changes and toggling.**
- Changes crossfade over 20 ms (raised cosine between the old and new taps).
- `enabled=false` crossfades to delay 0, after which the output is a bit-exact
  copy. The delay line keeps recording, so toggling is click-free.
- Delay 0, the default, is bit-exact.
- Memory for 2000 ms is allocated at caps negotiation.

## Properties

Units are dB, ms and Hz. Every property is mutable while PLAYING unless
noted.

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `delay` | float | ms | 0 … 2000 | 0 |
| `channel-delays` | GstValueArray of float | ms per channel, added to `delay` | −2000 … 2000 each | `<>` (empty) |

**`channel-delays` details.**
- Missing entries are 0, and extra entries are ignored.
- At most 64 entries are used.
- It is mutable while PLAYING, but not controllable, because arrays cannot be
  interpolated.

## Why these defaults

`delay` 0 and no `channel-delays`: the element is a
bit-exact passthrough until an operator sets the delay, because every
alignment value (microphone distance, video pipeline delay) is specific to
the setup. A wrong default delay would put audio out of sync.

## Example

```sh
gst-launch-1.0 audiotestsrc ! audio/x-raw,format=F32LE,channels=2,rate=48000 ! \
  stromdsp-delay delay=40 'channel-delays=<0.0, 2.5>' ! autoaudiosink
```
