# `stromdsp-broadcastprocessor`: broadcast final processor

A broadcast/streaming final processor: the last stage before a programme
leaves Strom. Its job is consistent, standards-compliant loudness under a
guaranteed true-peak ceiling. The chain is:

```
input gain → wideband AGC → multiband compressor (LR4) → true-peak limiter
```

- **AGC:** slow and loudness-targeting. Its BS.1770 detector freezes in
  pauses and below `agc-gate`.
- **Multiband:** LR4 crossover, one compressor per band.
- **True-peak limiter:** 4× oversampled detector, ceiling in dBTP per BS.1770
  Annex 2.
- **Stage bypass:** each stage has `*-enabled`, and the whole element has
  `enabled`. Toggles crossfade over 10 ms to a dry path delayed by the same
  latency, so latency never changes when stages are toggled.

The conventions shared by all elements are in the
[element overview](README.md#shared-conventions).

## Latency

`round(limiter-lookahead · fs / 1000) + 32` frames: 272 frames at
48 kHz with 5 ms. It is reported in the LATENCY query, is the same with
`enabled=false` and with any stage bypassed, and a `Latency` message is posted
when it changes (only at caps negotiation). With `enabled=false` the output is
the input delayed by exactly that latency. The element never uses
BaseTransform passthrough.

## Properties

Units are dB, LUFS, dBTP, dB/s, ms and Hz. Every property is controllable and
mutable in PLAYING unless noted.

| Property | Type | Range | Default |
|---|---|---|---|
| `enabled` | bool | | true |
| `preset` | enum `StromDspBroadcastPreset` (`ebu-r128`=0, `atsc-a85`=1, `streaming`=2, `speech`=3, `custom`=4) | | `ebu-r128` |
| `input-gain` | float, dB | −24…24 | 0 |
| `target-loudness` | float, LUFS | −40…−5 | −23 |
| `true-peak-ceiling` | float, dBTP | −12…0 | −1 |
| `agc-enabled` | bool | | true |
| `agc-max-gain` | float, dB | 0…30 | 12 |
| `agc-max-cut` | float, dB | 0…30 | 12 |
| `agc-rate-up` | float, dB/s | 0.1…20 | 1 |
| `agc-rate-down` | float, dB/s | 0.1…40 | 3 |
| `agc-gate` | float, LUFS | −80…−20 | −50 |
| `multiband-enabled` | bool | | true |
| `num-bands` | uint (READY only, not controllable) | 1…5 | 4 |
| `multiband-drive` | float, dB | −12…12 | 0 |
| `crossoverN-frequency` (N = 0…3) | float, Hz | 20…20000 | 120, 800, 4000, 10000 |
| `crossover-frequencies` | GstValueArray of gfloat, Hz (not controllable) | 20…20000 each | <120, 800, 4000, 10000> |
| `bandN-threshold` (N = 0…4) | float, dB | −60…0 | −18 |
| `bandN-ratio` | float | 1…20 | 2 |
| `bandN-attack` | float, ms | 0.1…500 | 30, 15, 8, 5, 3 |
| `bandN-release` | float, ms | 5…5000 | 300, 200, 150, 100, 80 |
| `bandN-makeup` | float, dB | −12…24 | 0 |
| `limiter-enabled` | bool | | true |
| `limiter-release` | float, ms | 1…2000 | 200 |
| `limiter-lookahead` | float, ms (READY only, not controllable) | 0.5…20 | 5 |
| `meter-interval` | uint, ms (0 = off) | 0…10000 | 100 |

The defaults are the `ebu-r128` preset at 4 bands. READY-only properties
written while streaming are stored and read back at once, but take effect at
the next caps negotiation.

**Crossovers** are kept strictly ascending, at least 1 Hz apart. A write that
would cross a neighbour is clamped to just inside it and reads back clamped
(one `debug!` line). When moving several individually, write in the direction
of travel: when lowering, crossover 0 first; when raising, crossover 3 first.
Alternatively, set them all at once with `crossover-frequencies`, whose
entries are sorted. GObject validates each entry, so an out-of-range or NaN
entry rejects the whole write.

## Presets

Setting `preset` sets every property except `enabled`, `meter-interval`,
`num-bands` and `limiter-lookahead`, using the current `num-bands`.
- **`limiter-lookahead`:** it sets the element's latency, so it is READY-only
  and not part of a preset. All presets assume 5 ms.
- **Writing another property afterwards:** a write that changes a value
  changes only that property, and `preset` then reads `custom`. Writing
  `custom` changes nothing.
- **Changing `num-bands`** while a named preset is selected re-applies that
  preset for the new band count.

Presets are our own, based on the standards' targets and textbook dynamics,
and deliberately gentle:

| Nick | Target | Ceiling | AGC max gain/cut | AGC rate up/down | `agc-gate` | Band threshold | Band ratio | `limiter-release` |
|---|---|---|---|---|---|---|---|---|
| `ebu-r128` (default) | −23 LUFS | −1 dBTP | 12 / 12 dB | 1 / 3 dB/s | −50 LUFS | −18 dB | 2 | 200 ms |
| `atsc-a85` | −24 LKFS | −2 dBTP | 24 / 24 | 3 / 6 | −50 | −19 | 2 | 200 |
| `streaming` | −16 LUFS | −1 dBTP | 24 / 24 | 3 / 6 | −50 | −11 | 2 | 200 |
| `speech` | −18 LUFS | −1 dBTP | 24 / 24 | 3 / 6 | −40 | −11 | 1.5 | 150 |

All presets: `input-gain` 0, all stages on, `multiband-drive` 0, band makeup 0.

**Why these values:**
- **Targets and ceilings:** EBU R 128, ATSC A/85, and the AES TD1004/TD1008
  streaming range (`speech` sits inside that range at −18 LUFS).
- **Band thresholds:** target + 5 dB (speech: + 7 dB). They follow the target
  because the multiband comes after the AGC.
- **AGC range and rates:** `ebu-r128` uses ±12 dB, which covers programme
  delivered between −35 and −11 LUFS, and 1 dB/s up, 3 dB/s down: a slow
  fader, not a compressor, that brings a loud item down faster than it lifts
  a quiet one. The others sit further from typical delivery levels, so they
  need ±24 dB and 3/6 dB/s to reach target within 10 s from −35 or −10 LUFS.
- **Speech:** a gentler 1.5:1 multiband, a faster AGC gate at −40 LUFS, and a
  150 ms limiter release.

Crossovers and band time constants depend only on the band count:

| Bands | Crossovers (Hz) | Attack / release per band, low → high (ms) |
|---|---|---|
| 1 | (none) | 30/300 |
| 2 | 200 | 30/300, 10/150 |
| 3 | 200, 3000 | 30/300, 10/150, 5/100 |
| 4 | 120, 800, 4000 | 30/300, 15/200, 8/150, 5/100 |
| 5 | 120, 800, 4000, 10000 | 30/300, 15/200, 8/150, 5/100, 3/80 |

## Why these defaults

The element starts as the `ebu-r128` preset at
4 bands, so it delivers to the European broadcast loudness standard with no
property set; the preset reasons above apply. The settings outside the
presets:
- `enabled` `true`: a final processor is meant to be on.
- `num-bands` 4 (crossovers 120, 800 and 4000 Hz): bass, body, presence and
  top are controlled separately, without the extra phase shift and cost of a
  fifth band.
- `limiter-lookahead` 5 ms: the limiter's attack equals the lookahead, and
  5 ms catches a peak with a smooth ramp; it matches `stromdsp-limiter`.
- `meter-interval` 100 ms: ten updates per second, the same as
  `stromdsp-meter`.

## Measured result and limitations

**Measured result:** a 60 s synthetic programme at −35, −23 and −10 LUFS input
lands within ±1 LU of each preset's target (integrated, after the first
10 s), and true peak stays ≤ ceiling + 0.1 dB (both measured with `ebur128`).

**Known limitation:** very peaky material, such as frequent clicks or hard
percussive hits, can land 1–3 LU below target. The multiband stage and the
limiter come after the AGC, so the level they remove is not made up. True peak
still holds.

## Signals

- **`get-preset-values(gchararray nick) -> GstStructure`:** returns a
  structure named after the preset, with one field per property the preset
  sets, for the current `num-bands`. It returns NULL for `custom` or an
  unknown nick.
- **`reset()`:** restarts the output integrated loudness.

## Meter message

The element posts an element message named `stromdsp-broadcastprocessor`
every `meter-interval` ms of stream time, at most once per buffer. Silence
reads −200.0, never NaN.

| Field | Type | Meaning |
|---|---|---|
| `running-time`, `stream-time` | guint64, ns | end of the buffer |
| `input-momentary`, `input-short-term` | gdouble, LUFS | measured before `input-gain`: the signal as it arrives |
| `output-momentary`, `output-short-term` | gdouble, LUFS | |
| `output-integrated` | gdouble, LUFS | gated, since start or `reset` |
| `input-true-peak`, `output-true-peak` | gdouble, dBTP | max since the last message; input is measured before `input-gain` |
| `agc-gain` | gdouble, dB | signed; + is boost |
| `band-gain-reduction` | GstValueArray of gdouble | positive dB of reduction, one per active band |
| `limiter-gain-reduction` | gdouble | positive dB of reduction; max of values sampled every 512 frames |

## Example

```sh
GST_PLUGIN_PATH=target/release gst-launch-1.0 -m audiotestsrc wave=pink-noise ! \
  audio/x-raw,format=F32LE,rate=48000,channels=2 ! \
  stromdsp-broadcastprocessor preset=streaming meter-interval=500 ! autoaudiosink
```
