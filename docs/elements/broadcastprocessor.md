# `stromdsp-broadcastprocessor`: broadcast final processor

A broadcast/streaming final processor: the last stage before a programme
leaves Strom. Its job is consistent, standards-compliant loudness under a
guaranteed true-peak ceiling. The chain is:

```
input gain → wideband AGC → multiband compressor (LR4) → final clipper → true-peak limiter
                                   ↑                                │
                                   └────── clip budget governor ←───┘
```

- **AGC:** slow and loudness-targeting. Its BS.1770 detector freezes in
  pauses and below `agc-gate`.
- **Multiband:** LR4 crossover, one compressor per band.
- **Final clipper:** soft clipping at 8× in which only the clippings (what
  the curve removes) are resampled, band-limited and subtracted from the
  programme, so the programme itself never passes through resampling
  filters and is untouched where nothing reaches the knee. Its threshold
  sits 0.5 dB under the true-peak ceiling; it leaves bass (below the lowest
  crossover) to the limiter. It takes the peaks so the limiter mostly
  catches residue.
- **True-peak limiter:** 4× oversampled detector, ceiling in dBTP per BS.1770
  Annex 2. It stays the guaranteed ceiling. A ceiling change glides at
  1 dB per 512 frames.
- **Clip budget governor:** a slow loop (seconds) that keeps the clipper's
  clip depth, the share of samples it touches and the limiter's mean gain
  reduction within `clip-budget`. Over budget it stops the
  output-referenced trim from adding gain and cuts the AGC target,
  reporting the shortfall as `loudness-deficit` rather than distorting.
  Where budget is left but the target is out of reach (AGC and trim at a
  bound) it lowers the band thresholds with matching makeup ("density").
  It freezes in pauses and below `agc-gate`.
- **Stage bypass:** each stage has `*-enabled`, and the whole element has
  `enabled`. Toggles crossfade over 10 ms to a dry path delayed by the same
  latency, so latency never changes when stages are toggled.

The conventions shared by all elements are in the
[element overview](README.md#shared-conventions).

## Latency

`69 + round(limiter-lookahead · fs / 1000) + 32` frames: the final clipper
(69 frames at every rate) plus the limiter. With the default 1.5 ms
lookahead that is 173 frames, **3.60 ms at 48 kHz** (it was 272 frames,
5.67 ms, before the clipper: the clipper's 1.44 ms is more than paid for
by the shorter lookahead). It is reported in the LATENCY query, is the same with
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
| `clipper-enabled` | bool | | true |
| `clipper-bandwidth` | float, Hz | 8000…22000 | 20000 |
| `governor-enabled` | bool | | true |
| `clip-budget` | float, dB | 0…6 | 2 |
| `limiter-enabled` | bool | | true |
| `limiter-release` | float, ms | 1…2000 | 200 |
| `limiter-lookahead` | float, ms (READY only, not controllable) | 0.5…20 | 1.5 |
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
  and not part of a preset. All presets assume 1.5 ms.
- **Writing another property afterwards:** a write that changes a value
  changes only that property, and `preset` then reads `custom`. Writing
  `custom` changes nothing.
- **Changing `num-bands`** while a named preset is selected re-applies that
  preset for the new band count.

Presets are our own, based on the standards' targets and textbook dynamics,
and deliberately gentle:

| Nick | Target | Ceiling | AGC max gain/cut | AGC rate up/down | `agc-gate` | Band threshold | Band ratio | `limiter-release` | `clip-budget` |
|---|---|---|---|---|---|---|---|---|---|
| `ebu-r128` (default) | −23 LUFS | −1 dBTP | 24 / 24 dB | 1 / 3 dB/s | −50 LUFS | −18 dB | 2 | 200 ms | 2 dB |
| `atsc-a85` | −24 LKFS | −2 dBTP | 24 / 24 | 3 / 6 | −50 | −19 | 2 | 200 | 2 |
| `streaming` | −16 LUFS | −1 dBTP | 24 / 24 | 3 / 6 | −50 | −11 | 2 | 200 | 3 |
| `speech` | −18 LUFS | −1 dBTP | 24 / 24 | 3 / 6 | −40 | −11 | 1.5 | 150 | 2 |

All presets: `input-gain` 0, all stages on (clipper and governor too),
`multiband-drive` 0, band makeup 0, `clipper-bandwidth` 20000 Hz.

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
- **Clip budget:** 2 dB allows a clip depth of −30 dB, 1 dB of mean limiter
  gain reduction and 1 % of samples touched (see the mapping below). At
  −18 … −24 LUFS programme rarely reaches the clipper, so the budget
  matters only for hot, peaky material; the peaky test programme stays
  within ±1 LU at every preset. Speech keeps 2 dB: with 1 dB the governor
  took 1 … 2 LU off the peaky programme. Streaming gets 3 dB (−25 dB,
  1.5 dB, 1.4 %): at −16 LUFS peaks reach the clipper more often, and 2 dB
  cost the peaky programme 0.6 … 1.5 LU.
- **Clipper bandwidth:** 20 kHz, the top of the audio band: the clipper adds
  nothing above it.

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
- `limiter-lookahead` 1.5 ms: the final clipper takes the peaks, so the
  limiter handles residue only. Measured on the peaky programme
  (streaming, −23 LUFS input): limiter mean gain reduction 1.72 dB with no
  clipper and 5 ms, 0.71 dB with the clipper and 5 ms, 0.69 dB with the
  clipper and 1.5 ms; true peak within the ceiling in all cases, and
  within +0.008 dB on nasty signals (squares and impulses at +20 dBFS,
  fs/4 at 45°, tones up to 0.45·fs, white noise, clicks) at 44.1, 48 and
  96 kHz. A 0.75 ms lookahead with the clipper's low-latency setting
  holds the same ceiling (0.97 dB mean gain reduction).
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
of every 1.1 s, `streaming` preset, integrated output 10 … 60 s (target
−16 LUFS). Before the final clipper the limiter took about 13 dB off the
clicks and every input level missed ±1 LU. Now:

| Input | No clipper (before) | Clipper, governor off | Defaults (clipper and governor) |
|---|---|---|---|
| −35 LUFS | −17.69 LUFS | −16.52 LUFS | −18.72 LUFS |
| −23 LUFS | −17.15 LUFS | −16.37 LUFS | −19.13 LUFS |
| −10 LUFS | −17.11 LUFS | −16.33 LUFS | −19.68 LUFS |

The clipper alone lands within ±1 LU. With the governor on, the clipping
of the clicks and the limiting of what the clipper leaves of them use more
than the streaming budget (3 dB), so the governor holds the trim, cuts
the AGC target, and reports the shortfall in `loudness-deficit`. Clicks
like these should be removed before the AGC by a stage made for that
(they are defects, not programme peaks); the budget is meant for
programme.

**Clip budget mapping.** Each measure, over a gated 3 s window, is turned
into the budget that would allow it; the largest is compared with
`clip-budget` `B`:

| Measure | Allowed at budget `B` |
|---|---|
| clip depth (removed / input energy of the clipper) | `−40 + 5·B` dB |
| limiter mean gain reduction | `B/2` dB |
| share of samples the clipper touches | `0.5 % · 2^(B/2)` |

Holding a budget has a loudness cost that depends on the programme. On
the sport-like peaky programme (transients the band compressors cannot
catch) with `clip-budget` 1.5 at `streaming`, the governor holds the
clip depth under −32.5 dB (−31.2 dB with it off) and the output lands
1.4 … 2.0 LU under the target, reported as `loudness-deficit`.

## Final clipper latency

The clipper's structure is a library setting (`clipper_latency`, chosen
by the element's coming latency mode; the element uses `standard` today):

| Setting | Structure | Latency (48 kHz) | 15 kHz alias | Worst overshoot | Limiter lookahead | Element latency |
|---|---|---|---|---|---|---|
| `standard` | 8×, one pass, 41-tap shaping | 69 frames, 1.44 ms | −94.0 dBFS | +0.77 dB | 1.5 ms | 173 frames, 3.60 ms |
| `low` | 4×, one pass, 3-tap shaping | 48 frames, 1.00 ms | −75.8 dBFS | +2.17 dB | 0.75 ms | 116 frames, 2.42 ms |
| `two-pass` | 8×, two passes | 138 frames, 2.88 ms | −94.0 dBFS | +0.27 dB | | |

Alias: largest spur of a 15 kHz tone 6 dB over the threshold. Overshoot:
worst sample peak over the threshold on band-limited signals (tones,
19 + 20 kHz, pink and speech-like noise) driven 2 and 4 dB into the
clipper; the true-peak limiter after the clipper takes it. Every setting
beats both the limiter alone and a plain hard clip on loudness against
noise-to-mask ratio at drives 0 … 4 dB (`tests/clipper.rs`).

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
| `clip-depth` | gdouble, dB | energy the final clipper removed relative to its input since the last message; −200.0 when nothing |
| `loudness-deficit` | gdouble, LU | how far the governor lets the output fall short of the target (≥ 0) |
| `governor-state` | gchararray | `off`, `frozen` (pause, below the gate, a stage bypassed), `free`, `hold`, `density`, `cut` or `release` |

## Example

```sh
GST_PLUGIN_PATH=target/release gst-launch-1.0 -m audiotestsrc wave=pink-noise ! \
  audio/x-raw,format=F32LE,rate=48000,channels=2 ! \
  stromdsp-broadcastprocessor preset=streaming meter-interval=500 ! autoaudiosink
```
