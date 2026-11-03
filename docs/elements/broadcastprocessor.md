# `stromdsp-broadcastprocessor`: broadcast final processor

A broadcast/streaming final processor: the last stage before a programme
leaves Strom. Its job is consistent, standards-compliant loudness under a
guaranteed true-peak ceiling. The chain is:

```
input gain → wideband AGC → multiband compressor (LR4) → true-peak limiter
                             (downward expander per band,
                              bass clipper on band 0)
```

- **AGC:** slow and loudness-targeting. Its BS.1770 detector freezes in
  pauses and below `agc-gate`.
- **Multiband:** LR4 crossover, one compressor per band.
- **Downward expander per band:** in each band's gain computer, below its
  own threshold: compression above the band threshold, unity between the
  two thresholds, expansion below the expander threshold, a 6 dB soft knee
  at each, and a floor (largest attenuation). It lowers room tone and noise
  between words after the AGC has lifted them. Against chatter it uses a
  hold (`expander-hold`) and a slower close (150 ms) than open (3 ms), not
  hysteresis. Ratio 1 turns it off exactly. With `num-bands` 1 it is a
  wideband expander.
- **Bass clipper on band 0:** lookahead gain control of the bass band from
  its analytic-signal envelope (a 90° all-pass pair in the control path
  only), so a loud steady bass is held at `bass-clip-threshold` without
  harmonics (measured below −130 dBc on a 60 Hz sine 6 dB over). The gain
  ramps over the 3 ms lookahead and releases with 50 ms. The other bands
  are delayed by the same 3 ms. It needs a separate bass band: with
  `num-bands` 1 it does nothing (the 3 ms delay stays). Its IM feedback
  (lowering the bass threshold by up to 3 dB while bass peaks coincide with
  clipping after the multiband) has no source yet and stays idle.
- **True-peak limiter:** 4× oversampled detector, ceiling in dBTP per BS.1770
  Annex 2.
- **Stage bypass:** each stage has `*-enabled`, and the whole element has
  `enabled`. Toggles crossfade over 10 ms to a dry path delayed by the same
  latency, so latency never changes when stages are toggled.

The conventions shared by all elements are in the
[element overview](README.md#shared-conventions).

## Latency

`round(3 · fs / 1000) + round(limiter-lookahead · fs / 1000) + 32`
frames: 144 + 240 + 32 = 416 frames (8.67 ms) at 48 kHz with 5 ms. The
first term is the bass clipper's fixed 3 ms lookahead; it is there with
the bass clipper off or the multiband bypassed too. It is reported in the LATENCY query, is the same with
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
| `agc-max-gain` | float, dB | 0…30 | 24 |
| `agc-max-cut` | float, dB | 0…30 | 24 |
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
| `bandN-expander-threshold` | float, dB (band level, like `bandN-threshold`) | −80…0 | −47 |
| `bandN-expander-ratio` | float (1 = off) | 1…10 | 1.5 |
| `bandN-expander-floor` | float, dB (largest attenuation) | 0…40 | 4 |
| `expander-hold` | float, ms | 0…1000 | 100 |
| `bass-clipper-enabled` | bool | | true |
| `bass-clip-threshold` | float, dB (band 0 analytic envelope) | −40…0 | −14 |
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
| `ebu-r128` (default) | −23 LUFS | −1 dBTP | 24 / 24 dB | 1 / 3 dB/s | −50 LUFS | −18 dB | 2 | 200 ms |
| `atsc-a85` | −24 LKFS | −2 dBTP | 24 / 24 | 3 / 6 | −50 | −19 | 2 | 200 |
| `streaming` | −16 LUFS | −1 dBTP | 24 / 24 | 3 / 6 | −50 | −11 | 2 | 200 |
| `speech` | −18 LUFS | −1 dBTP | 24 / 24 | 3 / 6 | −40 | −11 | 1.5 | 150 |

All presets: `input-gain` 0, all stages on, `multiband-drive` 0, band makeup 0,
`expander-hold` 100 ms, `bass-clipper-enabled` true,
`bass-clip-threshold` = target + 9 dB.

| Nick | `bandN-expander-threshold` (every band) | `bandN-expander-ratio` | `bandN-expander-floor` | `bass-clip-threshold` |
|---|---|---|---|---|
| `ebu-r128` | −47 dB | 1.5 | 4 dB | −14 dB |
| `atsc-a85` | −48 | 1.5 | 4 | −15 |
| `streaming` | −40 | 1.5 | 4 | −7 |
| `speech` | −40 | 2 | 6 | −9 |

**Why these values:**
- **Targets and ceilings:** EBU R 128, ATSC A/85, and the AES TD1004/TD1008
  streaming range (`speech` sits inside that range at −18 LUFS).
- **Band thresholds:** target + 5 dB (speech: + 7 dB). They follow the target
  because the multiband comes after the AGC.
- **AGC range:** ±24 dB for every preset. The AGC must cover the distance
  from input to target (12 dB for −35 LUFS input to −23 LUFS) plus what the
  multiband and limiter take away after it, which the output-referenced trim
  adds to the AGC target (up to 6 dB). With ±12 dB, −35 LUFS input to
  `ebu-r128` used the whole range just to reach the target.
- **AGC rates:** `ebu-r128` rides at 1 dB/s up, 3 dB/s down: a slow fader,
  not a compressor, that brings a loud item down faster than it lifts a
  quiet one. The others sit further from typical delivery levels, so they
  use 3/6 dB/s to reach target within 10 s from −35 or −10 LUFS.
- **Speech:** a gentler 1.5:1 multiband, a faster AGC gate at −40 LUFS, and a
  150 ms limiter release.
- **Expander:** target − 24 dB, 1:1.5, at most 4 dB for programme; target
  − 22 dB, 1:2, at most 6 dB for speech. After the AGC, room tone and noise
  between words sit near or below target − 25 dB, while quiet word endings
  stay above target − 20 dB. The floor is small on purpose: a quiet passage
  of music that falls below the threshold loses at most 4 dB, slowly.
  Measured: speech-like bursts with noise 40 dB below them through the
  `speech` preset: the noise between the bursts drops by 5.8 dB, the bursts
  change by less than 0.01 dB.
- **Expander hold** 100 ms: bridges gaps inside words, so word endings are
  not chopped.
- **Bass clip threshold** target + 9 dB: 4 dB above the band threshold. A
  steady bass at target + 13 dB comes out of the 2:1 band compressor at
  target + 9 dB, so the clipper catches only louder bass peaks and the ones
  the 30 ms band-0 attack lets through.

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

A hot, peaky, sport-like programme (dense percussive hits, clicks and
inter-sample peaks) meets the same bounds, and its short-term loudness stays
within 1 LU of target after a 10 s pause.

**Output-referenced trim:** the multiband stage and the limiter come after the
AGC and take level away, most on peaky material. A slow loop measures the
gated loudness at the output and trims the AGC target to make that up
(time constant 8 s, at most 0.5 dB/s, within ±6 dB). It freezes in pauses and
below `agc-gate`, while the AGC is ramping or bypassed, and when the AGC is at
`agc-max-gain` or `agc-max-cut`. Those bounds and the AGC rates still apply.

**Limitation:** the trim cannot exceed the AGC range or its own ±6 dB.
Programme that needs most of `agc-max-gain` just to reach the target gets
little make-up, and can land below target by what the multiband and limiter
remove. Material the limiter already holds at the ceiling cannot get louder
either; the trim then stops at its 6 dB range.

Measured case: the peaky programme above with a click every 300 ms instead
of every 1.1 s, `streaming` preset. The limiter takes about 13 dB off the
clicks, the trim reaches +6 dB, and at −35 LUFS input the AGC also reaches
its 24 dB maximum gain. Integrated output (10 … 60 s, target −16 LUFS):

| Input | Output | Off target |
|---|---|---|
| −35 LUFS | −17.69 LUFS | −1.69 LU |
| −23 LUFS | −17.15 LUFS | −1.15 LU |
| −10 LUFS | −17.11 LUFS | −1.11 LU |

All three miss ±1 LU. Short-term output settles near −16.9 LUFS (−17.4 at
−35 LUFS input); the rest is the time the trim takes to rise.

## Signals

- **`get-preset-values(gchararray nick) -> GstStructure`:** returns a
  structure named after the preset, with one field per property the preset
  sets, for the current `num-bands`. It returns NULL for `custom` or an
  unknown nick.
- **`get-preset-values-for-bands(gchararray nick, guint num_bands) -> GstStructure`:**
  the same structure, with the same fields, for `num_bands` bands instead of
  the current `num-bands`: band settings and crossovers follow that band
  count. Use it to preview a preset at another band count; it changes
  nothing on the element. It returns NULL for `custom`, an unknown nick, or
  `num_bands` outside 1–5.
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
| `band-expander-gain-reduction` | GstValueArray of gdouble | positive dB of expander attenuation, one per active band |
| `bass-clip-gr` | gdouble | positive dB of bass-clipper reduction, largest since the last message |

## Example

```sh
GST_PLUGIN_PATH=target/release gst-launch-1.0 -m audiotestsrc wave=pink-noise ! \
  audio/x-raw,format=F32LE,rate=48000,channels=2 ! \
  stromdsp-broadcastprocessor preset=streaming meter-interval=500 ! autoaudiosink
```
