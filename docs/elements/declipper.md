# `stromdsp-declipper`: repairs hard-clipped audio

An in-place `BaseTransform` on F32LE interleaved audio, any rate and channel
count; channels are processed independently. It finds runs of clipped
samples on the waveform and rebuilds them, and passes everything else
bit-exact, delayed by its latency. The bypass, caps and latency rules shared
by all elements are in the [element overview](README.md#shared-conventions).

> Code is the source of truth — this may have drifted; read the code for the
> current implementation.

**Opt-in, one per input.** Put it on an input that is known or likely to
clip: an overdriven microphone, a phone or remote contribution that clipped
before it reached you. No chain or preset turns it on; the broadcast
processor does not contain it.

**Put a limiter after it.** The rebuilt peaks lie beyond the old clip level,
so the output can be louder than the input's peak: by up to `max-overshoot`
(6 dB) over the clip level, in practice about the clip depth (3 dB of
clipping gives about 3 dB more peak). In a float chain that is harmless. Before anything
with a ceiling (integer conversion, an encoder, a transmission limit) put
`stromdsp-limiter` or the broadcast processor after it. Lowering the gain
before the declipper does not help: the overshoot is relative to the clip
level.

## Properties

| Property | Type | Unit | Range | Default | Mutable |
|---|---|---|---|---|---|
| `enabled` | bool | | | `true` | PLAYING |
| `latency-mode` | enum | | `low` (3 ms), `standard` (5 ms), `high` (10 ms) | `standard` | READY |
| `max-overshoot` | float | dB | 0 … 24 | 6 | PLAYING |
| `min-level` | float | dBFS | −60 … 0 | −20 | PLAYING |
| `meter-interval` | uint | ms | 0 … 10000 (0 = off) | 1000 | PLAYING |

`latency-mode` written while a stream runs reads back at once and takes
effect at the next caps negotiation (a warning is logged), because the delay
line and the solver are sized for it. `max-overshoot` and `min-level` apply to
runs found after the change; a change that stops repairs (for example
`min-level` above the clip level) takes effect at the next run, never inside
one.

## Behaviour

**What it finds.**
- Two or more consecutive samples on a flat plateau, entered and left by a
  sharp corner from the same side, at or above `min-level`. The level is the
  plateau's own value, so clipping below 0 dBFS, gain, DC or requantisation
  after the clip are covered. A plateau that droops like the output of a
  DC blocker after the clip (up to 50 Hz) counts too.
- Once the input has shown its clip level (flat runs of four or more
  samples), single samples and pairs at that level, with sharp corners, count
  as clipped as well. The level is forgotten 5 s after it was last
  confirmed, and single samples never confirm it.
- **Single-sample repair.** A peak that only just reaches a confirmed clip
  level is clipped on one sample and has no plateau of its own. When that
  sample sits at the confirmed level (per direction, top and bottom) and
  both of its neighbours lie clearly inside it, it is rebuilt like any
  other run, as a run of one sample, and counted in `runs-repaired`. It is
  found one sample later than a longer run, which the latency already
  covers. Audio that has never shown a clip level gets no single-sample
  repair.
- Unclipped audio is never touched: across the test signals (full-scale
  tones at every peak phase, band-limited squares, soft-saturated tones,
  speech-, music- and noise-like material, as float, 16-bit and 24-bit) and
  on clean programme it is bit-exact.

**What it cannot repair.** Soft clipping or saturation, clipping followed by
a lossy codec (the plateau becomes ripple), and runs longer than about 2 ms
(passed through unchanged and counted in `runs-too-long`).

**How it rebuilds.** Least-squares AR interpolation from the context around
the run (10 ms before, up to 5 ms after), each rebuilt sample kept beyond its
clipped value and within `max-overshoot`. The work per channel is bounded:
when it is spent, a run is rebuilt by a cheap curve that keeps the same
limits, and counted in `fallbacks`.

**Never worse than the input.** A rebuild is kept only when it is
plausible; otherwise the run keeps its clipped samples and is counted in
`runs-rejected`. It is rejected when it rises further above the clip level
than the slope of the samples around the run allows for a run of its
length (a spike), or when it reaches `max-overshoot` while the signal
around the run is too noise-like to predict (music with dense
high-frequency content, where long runs were otherwise filled with humps
at the cap). On the test corpus (speech and music clipped 3, 6 and 10 dB)
no clip comes out further from the original on its clipped samples than
the clipped input, and no output peak lies more than 2.1 dB above the
original's (none above it at 6 and 10 dB). Heavy clipping stays heavily
distorted: on speech clipped 10 dB the repair takes back about 9 dB of
the error on the clipped samples.

**CPU.** Grows with the amount of clipping. On an Apple M5 Pro, 48 kHz
stereo in 10 ms buffers: about 0.1 % of the buffer duration on clean audio,
0.3–0.8 % on speech and music clipped 6–10 dB, 2.6 % on dense tones clipped
10 dB; the worst single buffer stays under 5 % on any input.

**Disabled behaviour.** `enabled=false` crossfades (10 ms raised cosine) to
the input delayed by the same latency; after the fade the output is the
input sample for sample. The latency does not change. Detection keeps
running, so re-enabling fades back without a gap.

## Latency

`max(16, floor(ms · fs))` frames, with `ms` from `latency-mode`, added to the
LATENCY query (rounded up to whole nanoseconds). The setting is the budget:
the test fails if any rate gives more.

| `latency-mode` | 44.1 kHz | 48 kHz | 96 kHz |
|---|---|---|---|
| `low` | 132 frames (2.99 ms) | 144 (3.00 ms) | 288 (3.00 ms) |
| `standard` | 220 (4.99 ms) | 240 (5.00 ms) | 480 (5.00 ms) |
| `high` | 441 (10.0 ms) | 480 (10.0 ms) | 960 (10.0 ms) |

What a shorter setting costs (recorded speech from the test corpus, nine
clips, clipped 3, 6 and 10 dB below their peak; mean SDR improvement on the
clipped samples): `low` 16.6 / 13.9 / 9.0 dB, `standard` 16.5 / 13.7 / 8.9 dB,
`high` 16.4 / 13.9 / 8.9 dB. The detection is the same in every setting;
`low` gives a long run less context after it.

## Meter message

Every `meter-interval` ms of stream time an element message named
`stromdsp-declipper` is posted (counts since the previous message):

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `running-time` | `guint64` | ns | running time of the end of the buffer (`GST_CLOCK_TIME_NONE` without timestamps) |
| `stream-time` | `guint64` | ns | stream time of the end of the buffer (likewise) |
| `runs-repaired` | `guint64` | | clipped runs rebuilt |
| `samples-repaired` | `guint64` | | samples rebuilt |
| `runs-too-long` | `guint64` | | runs too long to repair, passed through |
| `fallbacks` | `guint64` | | runs rebuilt by the cheap fallback because the work budget was spent (included in `runs-repaired`) |
| `runs-rejected` | `guint64` | | runs whose rebuild was not plausible, left at their clipped samples (not in `runs-repaired`) |
| `runs-failed` | `guint64` | | runs left at their clipped samples because the rebuild could not be computed (non-finite input, a numerical breakdown) or the run queue was full (not in `runs-repaired`) |
| `largest-overshoot` | `gdouble` | dB | largest rebuilt peak over its clip level (0 when nothing was rebuilt) |

## Why these defaults

- `latency-mode` `standard` (5 ms): the context after a run that a speech
  peak needs, at a latency that fits a live chain; `low` costs little and
  suits commentary, talkback and calls; `high` only helps long runs.
- `max-overshoot` 6 dB: covers the overshoot of clipping up to about 6 dB
  deep and limits a rebuild that runs away. Measured on the recorded speech
  of the test corpus it repairs as well as 12 dB at 3 dB of clipping and
  better at 6 and 10 dB (13.7 against 12.8 dB, 8.9 against 8.7 dB of SDR
  improvement on the clipped samples): a deeper clip is rebuilt to 6 dB
  rather than guessed further. Its cap also feeds the plausibility check.
- `min-level` −20 dBFS: covers up to 20 dB of gain applied after the clip,
  and keeps quiet flat stretches (digital silence steps, test tones) out.
- `meter-interval` 1000 ms: the fields are event counts, not levels; one
  message a second shows an operator that an input clips without flooding
  the bus.
- `enabled` `true`: inserting the element is the opt-in.

## Example

```sh
gst-launch-1.0 autoaudiosrc ! audioconvert ! audio/x-raw,format=F32LE,layout=interleaved ! \
  stromdsp-declipper latency-mode=low ! stromdsp-limiter threshold=-1 ! \
  audioconvert ! autoaudiosink
```
