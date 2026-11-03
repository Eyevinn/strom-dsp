# `stromdsp-gate`: noise gate with hysteresis and hold

A channel-strip element: an in-place `BaseTransform` on F32LE interleaved
audio, any rate and channel count, with linked channels and no latency. The
bypass, caps and latency rules shared by all elements are in the
[element overview](README.md#shared-conventions).

## Properties

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `open-threshold` | float | dB | −80 … 0 | −45 |
| `close-threshold` | float | dB | −80 … 0 | −50 |
| `attack` | float | ms | 0.01 … 200 | 3 |
| `hold` | float | ms | 0 … 1000 | 120 |
| `release` | float | ms | 1 … 2000 | 200 |
| `reduction` | float | dB | −80 … 0 | −35 |

Expander properties:

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `mode` | enum `StromDspGateMode` | | `gate` (0), `expander` (1) | `gate` |
| `ratio` | float | expander ratio | 1 … 20 | 2 |

## Behaviour

**Detector.** The level is a sine-calibrated RMS:
- a 10 ms one-pole mean square of the loudest channel, read as `sqrt(2·m)`;
- a steady sine reads its peak level in dBFS, and noise reads its RMS + 3 dB;
- after the signal stops, the level decays with 20 ms.

Single peaks of noise below the threshold therefore do not open the gate.

**Opening and closing.**
- **Opening:** the gate opens when the level goes above `open-threshold`, with
  the `attack` time constant (63 % after `attack`).
- **Hysteresis:** the effective close threshold is
  `min(close-threshold, open-threshold − 1 dB)`. If
  `close-threshold` > `open-threshold`, it is treated as equal, and a built-in
  1 dB minimum hysteresis still applies.
- **Hold:** once the level is below the close threshold, the gate stays open
  for `hold` ms.
- **Release** is the **time to close**: the gain falls in a straight line in
  dB from 0 dB to `reduction` in exactly `release` ms.

After the signal stops, the gate is fully closed after
`t_detector + hold + release`. Here `t_detector` ≈ 2.3 ms per dB that the
signal was above the close threshold.

**Expander mode.** `mode=expander` replaces the fixed closed gain with downward
expansion:
- below `open-threshold` the gain is `max(reduction, (ratio − 1)·(level − open-threshold))` dB;
- hysteresis, hold and the fades work as in gate mode;
- in `mode=gate`, `ratio` has no effect.

## Why these defaults

The defaults make the gate close cleanly in speech
pauses on its own. They are strom-dsp's own choice from textbook and broadcast
practice (`strom_dsp::presets`, see [PROVENANCE.md](../../PROVENANCE.md)). The levels assume EBU
R68/R128 alignment, where speech averages around −20 dBFS.
- `open-threshold` −45 / `close-threshold` −50: well below speech, so quiet
  word endings keep the gate open, and above a typical studio noise floor
  (< −50 dBFS). The 5 dB hysteresis stops breaths or room tone near the
  threshold from making the gate chatter.
- `attack` 3 ms: faster than a consonant or plosive onset (about 5–10 ms), so
  word starts are not clipped, but not so fast that the gain rise clicks.
- `hold` 120 ms: bridges the short gaps inside and between words (under about
  100 ms), so the gate does not chatter mid-sentence.
- `release` 200 ms: closes the full 35 dB range in 200 ms, a gentle fade that
  follows a natural room decay and is done well inside a speech pause.
- `reduction` −35 dB: a −50 dBFS noise floor drops to −85 dBFS, which is
  inaudible, without the "dead air" contrast and pumping of a full mute.
- `mode` `gate`, `ratio` 2: a plain gate is the predictable choice out of the
  box. 1:2 is the gentlest textbook expansion slope, ready for anyone who
  switches to `expander`.

## Example

```sh
gst-launch-1.0 autoaudiosrc ! audioconvert ! audio/x-raw,format=F32LE,layout=interleaved ! \
  stromdsp-gate open-threshold=-45 close-threshold=-50 hold=120 release=200 ! \
  audioconvert ! autoaudiosink
```

As a gentle 1:2 expander instead of a gate:

```sh
gst-launch-1.0 autoaudiosrc ! audioconvert ! audio/x-raw,format=F32LE,layout=interleaved ! \
  stromdsp-gate mode=expander ratio=2 reduction=-20 ! audioconvert ! autoaudiosink
```
