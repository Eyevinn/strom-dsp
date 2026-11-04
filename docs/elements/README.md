# Elements

The GStreamer plugin `stromdsp` has twelve elements. Each page below has the
full property table, the behaviour, the latency, the reasons for the defaults,
any message format and a `gst-launch-1.0` example.

| Element | Group | What it does | Latency |
|---|---|---|---|
| [`stromdsp-gate`](gate.md) | channel strip | noise gate with hysteresis and hold, or downward expander | none |
| [`stromdsp-compressor`](compressor.md) | channel strip | feed-forward compressor with soft knee | none |
| [`stromdsp-equalizer`](equalizer.md) | channel strip | parametric EQ, up to 8 biquad bands | none |
| [`stromdsp-limiter`](limiter.md) | channel strip | lookahead peak limiter | `lookahead` (default 5 ms) |
| [`stromdsp-ducker`](ducker.md) | utility | lowers the programme while a sidechain key is present | none |
| [`stromdsp-deesser`](deesser.md) | utility | sibilance reduction, split-band or wideband | none |
| [`stromdsp-cleanup`](cleanup.md) | utility | DC, rumble and hum removal | none |
| [`stromdsp-declipper`](declipper.md) | utility | repairs hard-clipped audio; opt-in per input, needs a limiter after it | `latency-mode`: `low` 3 ms, `standard` 5 ms, `high` 10 ms |
| [`stromdsp-delay`](delay.md) | utility | intentional delay for alignment (not reported as latency) | none |
| [`stromdsp-stereo`](stereo.md) | utility | balance, width, mono, swap, M/S, polarity | none |
| [`stromdsp-meter`](meter.md) | measurement | loudness, peak, RMS and correlation meter | none |
| [`stromdsp-broadcastprocessor`](broadcastprocessor.md) | broadcast | spike guard → AGC → multiband (expander per band, bass clipper) → final clipper → true-peak limiter final processor | `latency-mode`: `standard` 5.60 ms, `low` 2.77 ms at 48 kHz (formula on its page) |

The rules every element follows, and the tests that enforce them, are in
[ELEMENT_GUIDELINES.md](../ELEMENT_GUIDELINES.md).

## Shared conventions

### Bypass and the `enabled` crossfade

Every processing element has an `enabled` property, mutable while PLAYING.
Toggling `enabled` crossfades between processed and dry (20 ms for most
elements, 10 ms for the limiter, stereo, declipper and broadcast processor), then the
output is bit-exact, delayed by the element's reported latency (zero for most
elements; the limiter, the declipper and the broadcast processor keep their
delay).

`stromdsp-meter` has no `enabled` property: it never changes the samples.

### Latency

None of the elements adds latency, except:
- `stromdsp-limiter`: the audio is delayed by `lookahead`, which is added to
  the LATENCY query; a change posts a `Latency` message. The delay is kept
  with `enabled=false`.
- `stromdsp-broadcastprocessor`: set by `latency-mode` (READY-only):
  `standard` 269 frames (5.60 ms) at 48 kHz, at most 6 ms at 44.1, 48 and
  96 kHz; `low` 133 frames (2.77 ms), at most 3 ms. The formula is on its
  page. The same with `enabled=false` and with any stage bypassed.
- `stromdsp-declipper`: set by `latency-mode` (READY-only): `low` 3 ms,
  `standard` 5 ms, `high` 10 ms (`floor(ms · fs)` frames, at most the
  setting at any rate). The same with `enabled=false`.

The content delay of `stromdsp-delay` is not latency: it is not reported, and
timestamps pass through unchanged.

### Caps

Every element works on `audio/x-raw, format=F32LE, layout=interleaved`, with
any rate (`rate=[1,MAX]`), except where its page says otherwise:
- channels `[1,MAX]` for all elements, except `stromdsp-stereo` (exactly 2)
  and `stromdsp-meter` (1–64);
- `stromdsp-ducker` takes its programme on `sink` and a key with its own rate
  and channel count on the request pad `sidechain`.

### Channel strip

The four channel-strip elements (gate, compressor, equalizer, limiter) are
in-place `BaseTransform`s:
- caps `audio/x-raw, format=F32LE, layout=interleaved, rate=[1,MAX], channels=[1,MAX]`
- klass `Filter/Effect/Audio`
- every property controllable while PLAYING, except `num-bands`
- latency: none, except the limiter (`lookahead`, see
  [limiter](limiter.md#behaviour))

### Utility and measurement elements

Units are dB, ms and Hz. Every property is mutable while PLAYING unless
noted. None of these elements adds latency, except the declipper.

### Linked channels

Gate, compressor and limiter use **linked channels**: one gain is computed from
the loudest channel and applied to all, so the stereo image never moves. The
de-esser and the ducker's key detector are linked the same way.

### Defaults

Every property has a default that is sensible on its own for live broadcast
speech and programme audio, and every page explains each default under
"Why these defaults".
- The defaults are strom-dsp's own choice from textbook and broadcast
  practice (`strom_dsp::presets`), never from any product's presets. The
  references are in [PROVENANCE.md](../../PROVENANCE.md).
- Levels assume EBU R68/R128 alignment, where speech averages around
  −20 dBFS.
- `stromdsp-equalizer`, `-cleanup`, `-delay` and `-stereo` are neutral by
  default: bit-exact passthrough until an operator changes them.
- `stromdsp-broadcastprocessor` starts as its `ebu-r128` preset.
