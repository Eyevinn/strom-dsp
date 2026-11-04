# `stromdsp-broadcastprocessor`: broadcast final processor

A broadcast/streaming final processor: the last stage before a programme
leaves Strom. Its job is consistent, standards-compliant loudness under a
guaranteed true-peak ceiling. The chain is:

```
input gain → spike guard → wideband AGC → multiband compressor (LR4) → final clipper → true-peak limiter
                                          (downward expander per band,  ↑                │
                                           bass clipper on band 0)      └─ governor ←────┘
```

- **Spike guard:** removes clicks and other short, isolated peaks before
  they pump the AGC or count toward the loudness it measures. A peak more
  than `spike-threshold` over the programme's own peak envelope starts an
  event only if it looks like a click in the input already buffered: it
  jumps from the programme level to its peak in one sample, by more than
  the threshold, and does not come back within about 0.4 ms (the rise
  gate). Speech, tones, noise bursts and drum onsets build up over
  several samples or recur, so they never start one. An event whose peak
  still recurs is programme and is lifted; otherwise a short gain dip with
  smooth ramps brings it down to the threshold through a soft knee. Each
  channel on its own; outside its dips the stage is an exact delay.
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
  harmonics (2nd and 3rd harmonic of a 60 Hz sine 6 dB over: below
  −125 dBc at every lookahead). The gain ramps over the lookahead (1 ms
  standard; at least 0.5 ms, arriving late by the difference, with no
  lookahead) and releases with 50 ms; a threshold change glides. The other
  bands are delayed by the same lookahead. It needs a separate bass band:
  with `num-bands` 1 it does nothing (the delay stays).
- **IM feedback:** once per 512 frames the final clipper's share of
  touched samples goes to the bass clipper, which lowers its threshold by
  up to 3 dB while bass peaks coincide with clipping (30 ms attack, 1 s
  release), so a loud bass stops chopping the voice in the final clipper.
  The clipper comes after the multiband, so the feedback works with the
  previous 512 frames (about 10 ms late at 48 kHz).
- **Final clipper:** soft clipping at 8× (4× in the low latency mode) in
  which only the clippings (what the curve removes) are resampled,
  band-limited and subtracted from the programme, so the programme itself
  never passes through resampling filters and is untouched where nothing
  reaches the knee. Its threshold sits 0.5 dB under the true-peak ceiling;
  it leaves bass (below the lowest crossover) to the bass clipper and the
  limiter. It takes the peaks so the limiter mostly catches residue.
- **True-peak limiter:** 4× oversampled detector, ceiling in dBTP per BS.1770
  Annex 2. It stays the guaranteed ceiling. A ceiling change glides at
  1 dB per 512 frames.
- **Clip budget governor:** a slow loop (seconds) that watches the
  clipper's clip depth, the share of samples it touches and the limiter's
  mean gain reduction against `clip-budget`. **Loudness comes first:** it
  never lowers the output below the target to stay within the budget.
  Over budget with the output on target it stops the output-referenced
  trim from adding gain and gives back density, and reports the excess
  (`budget-excess`, state `over-budget`). When the target is out of reach
  (AGC and trim at a bound) it lowers the band thresholds with matching
  makeup ("density"), over budget too. It freezes in pauses and below
  `agc-gate`.
- **Stage bypass:** each stage has `*-enabled`, and the whole element has
  `enabled`. Toggles crossfade over 10 ms to a dry path delayed by the same
  latency, so latency never changes when stages are toggled.

The conventions shared by all elements are in the
[element overview](README.md#shared-conventions).

## Latency

The latency is set by `latency-mode` (READY-only):

`24 + round(spike lookahead · fs) + round(bass lookahead · fs) + C +
round(limiter-lookahead · fs) + 32` frames, lookaheads in seconds: the
spike guard's 4× interpolator (24) and lookahead, the bass clipper's
lookahead (there with the bass clipper off or the multiband bypassed too),
the final clipper (`C` frames at every rate), the limiter's lookahead and
its true-peak detector delay (32).

| `latency-mode` | Spike guard | Bass clipper | Final clipper `C` | Limiter | 44.1 kHz | 48 kHz | 96 kHz | Budget |
|---|---|---|---|---|---|---|---|---|
| `standard` (default) | 0.5 ms | 1 ms | 69 | 1.5 ms | 257 frames, 5.83 ms | 269 frames, 5.60 ms | 413 frames, 4.30 ms | 6 ms |
| `low` | 0 | 0 | 48 | 0.6 ms | 130 frames, 2.95 ms | 133 frames, 2.77 ms | 162 frames, 1.69 ms | 3 ms |

A test fails if either mode exceeds its budget at 44.1, 48 or 96 kHz.
The stages share the work: the spike guard takes clicks before the AGC,
the bass clipper takes bass peaks, the final clipper takes the remaining
peaks, so the true-peak limiter only catches residue and needs a short
lookahead. The low mode's limiter lookahead is 0.6 ms rather than 0.75:
the fixed frame counts (24, 48, 32) cost the most time at 44.1 kHz,
where 0.75 ms gave 137 frames, 3.11 ms. Writing `limiter-lookahead` after
`latency-mode` changes the limiter's share (see Properties).

The latency is reported in the LATENCY query, is the same with
`enabled=false` and with any stage bypassed, and a `Latency` message is
posted when it changes (only at caps negotiation). With `enabled=false`
the output is the input delayed by exactly that latency. The element
never uses BaseTransform passthrough.

**What the low mode costs** (measured; `tests/broadcast_latency.rs`,
`tests/bass_clipper.rs`, `tests/spike_guard.rs`, the harness):

- **True peak:** the ceiling holds in both modes, within +0.008 dB on
  nasty signals (squares and impulses at +20 dBFS, fs/4 at 45°, tones up to
  0.45·fs, white noise, clicks) at 44.1, 48 and 96 kHz.
- **Limiter and clipper:** on the peaky programme driven 6 dB hot (AGC
  off, `streaming`), the limiter's mean / largest gain reduction is
  0.82 / 2.55 dB in `standard` and 0.94 / 2.83 dB in `low` (0.6 and
  0.75 ms measure the same); the low clipper's 15 kHz alias is −75.8 dBFS
  against −94.0, and its worst sample-peak overshoot +2.17 dB against
  +0.77 dB, which the limiter takes.
- **Bass clipper:** steady bass stays clean at any lookahead; onsets pass
  at every setting, because the analytic envelope reaches an onset later
  than a few ms of lookahead can cover. An onset 12 dB over the threshold
  comes through 11.3 dB over with 3 ms, 12.0 dB with 1 ms and with none;
  decaying 60 Hz thumps peak 10.4 / 11.4 / 11.4 dB over. In the chain the
  difference does not show: limiter mean reduction 0.82 dB with 3, 1 and
  0 ms on the hot peaky programme.
- **Spike guard:** with no lookahead the guard cannot lift an event once
  it has started, so the rise gate decides before one starts, from the
  input the 4× interpolator already buffers (0.5 ms at 48 kHz; at 88.2 kHz
  and above the isolation check is shortened to what is buffered). On the
  corpus speech and phone clips every preset in the low mode measures the
  same with the guard as without it, clip for clip (crackle check below);
  before the rise gate it reached +37 dB, from word onsets after pauses
  capped for up to 0.4 ms. The clicky programme lands within
  0.02 LU of the standard mode (click table below). Harness NMR of the
  music-like programme: −42.8 dB standard, −43.3 dB low (−24.3 dB before
  the rise gate). The alternative, a 0.5 ms lookahead in the low mode,
  costs 24 frames (157 frames, 3.27 ms at 48 kHz, over the 3 ms budget),
  and with the old event rule still took plosives for clicks, as the
  standard mode did.

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
| `spike-guard-enabled` | bool | | true |
| `spike-threshold` | float, dB | 6…30 | 8 |
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
| `clipper-enabled` | bool | | true |
| `clipper-bandwidth` | float, Hz | 8000…22000 | 20000 |
| `governor-enabled` | bool | | true |
| `clip-budget` | float, dB | 0…6 | 2 |
| `limiter-enabled` | bool | | true |
| `limiter-release` | float, ms | 1…2000 | 200 |
| `latency-mode` | enum `StromDspBroadcastLatencyMode` (`standard`=0, `low`=1; READY only, not controllable) | | `standard` |
| `limiter-lookahead` | float, ms (READY only, not controllable) | 0.5…20 | 1.5 |
| `meter-interval` | uint, ms (0 = off) | 0…10000 | 100 |

The defaults are the `ebu-r128` preset at 4 bands. READY-only properties
written while streaming are stored and read back at once, but take effect at
the next caps negotiation.

**`latency-mode` and `limiter-lookahead`.** Writing `latency-mode` sets the
spike guard and bass clipper lookaheads, the final clipper's structure and
`limiter-lookahead` to the mode's values (see Latency); `limiter-lookahead`
then reads 1.5 or 0.6. A later write of `limiter-lookahead` changes only
the limiter's lookahead, and `latency-mode` keeps reading the mode that
was written; writing `latency-mode` again sets all four again. Set
`latency-mode` first and `limiter-lookahead` after it, if at all. Presets
change neither.

**Crossovers** are kept strictly ascending, at least 1 Hz apart. A write that
would cross a neighbour is clamped to just inside it and reads back clamped
(one `debug!` line). When moving several individually, write in the direction
of travel: when lowering, crossover 0 first; when raising, crossover 3 first.
Alternatively, set them all at once with `crossover-frequencies`, whose
entries are sorted. GObject validates each entry, so an out-of-range or NaN
entry rejects the whole write.

## Presets

Setting `preset` sets every property except `enabled`, `meter-interval`,
`num-bands`, `latency-mode` and `limiter-lookahead`, using the current
`num-bands`.
- **`latency-mode` and `limiter-lookahead`:** they set the element's
  latency, so they are READY-only and not part of a preset. Every preset
  works in both modes.
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

All presets: `input-gain` 0, all stages on (governor too; the clipper
except at `speech`),
`spike-threshold` 8 dB, `multiband-drive` 0, band makeup 0,
`clipper-bandwidth` 20000 Hz, `expander-hold` 100 ms,
`bass-clipper-enabled` true, `bass-clip-threshold` = target + 9 dB.

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
- **Spike threshold 8 dB:** gentle. Only peaks more than 8 dB over the
  programme's own recent peaks, reached in one sample and not recurring
  within about 0.4 ms, count; they come down to that level, not to the
  programme's.
- **Speech:** a gentler 1.5:1 multiband, a faster AGC gate at −40 LUFS, a
  150 ms limiter release, and **no final clipper**. At −18 LUFS the
  clipper clips the peaks of loud syllables and plosives, audibly on
  phone and codec speech: over the corpus speech and phone clips the
  worst-frame NMR at `speech` reached +35 dB with the clipper and
  +2.7 dB without it (standard mode; what remains is the limiter, see the
  crackle check). The governor stays on but frozen (it budgets the
  clipper's work).
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
- **Clip budget:** 2 dB allows a clip depth of −30 dB, 1 dB of mean limiter
  gain reduction and 1 % of samples touched (see the mapping below). At
  −18 … −24 LUFS programme rarely reaches the clipper, so the budget
  matters only for hot, peaky material. Streaming gets 3 dB (−25 dB,
  1.5 dB, 1.4 %): at −16 LUFS peaks reach the clipper more often. The
  budget never costs loudness; it decides when the trim holds and when the
  excess is reported. On the presets' defaults no test programme goes
  over it (see the measured results).
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
- `latency-mode` `standard`: within the 6 ms budget, with the margins the
  stages' guarantees need at little cost. `low` is for two-way use
  (commentary, talkback, calls), where every millisecond counts.
- `limiter-lookahead` 1.5 ms (the standard mode's): the final clipper
  takes the peaks, so the limiter handles residue only. On the peaky
  programme driven 6 dB hot (AGC off, `streaming`) the limiter's mean gain
  reduction is 1.68 dB with no clipper and 5 ms, 0.83 dB with the clipper
  and 5 ms, 0.82 dB with the clipper and 1.5 ms; the true peak stays within
  the ceiling in all cases.
- `meter-interval` 100 ms: ten updates per second, the same as
  `stromdsp-meter`.

## Measured result and limitations

**Measured result:** each preset at 4 bands, 60 s at −35, −23 and −10 LUFS
input, integrated output from 10 s (strom-dsp-harness
`tests/programme_table.rs`, run on purpose). On the peaky programme (a click
every 1.1 s), the clicky programme (a click every 300 ms), the music-like
programme and speech-like noise (7 kHz), every case lands within
−0.26 … +0.07 LU of the target, the governor reports no budget excess and
no loudness deficit, and the true peak stays at least 0.4 dB under the
ceiling. The final clipper works only at `streaming` and `speech`, and
lightly (clip depth −50 … −87 dB on the peaky, clicky and speech-like
programmes, −106 dB on music); the limiter's mean gain reduction stays at or under
0.13 dB. A 60 s synthetic programme, and the hot, peaky, sport-like one,
also meet ±1 LU and the ceiling + 0.1 dB in the chain tests
(`tests/broadcast.rs`, `ebur128`), and the short-term loudness is back
within 1 LU of target after a 10 s pause.

**Output-referenced trim:** the multiband stage and the limiter come after the
AGC and take level away, most on peaky material. A slow loop measures the
gated loudness at the output and trims the AGC target to make that up
(time constant 8 s, at most 0.5 dB/s, within ±6 dB). It freezes in pauses and
below `agc-gate`, while the AGC is ramping or bypassed, and when the AGC is at
`agc-max-gain` or `agc-max-cut`. Those bounds and the AGC rates still apply.

**Limitation:** the trim cannot exceed the AGC range or its own ±6 dB.
Programme that needs most of `agc-max-gain` just to reach the target gets
little make-up; the governor then adds density (up to 6 dB), and what is
still missing is reported as `loudness-deficit`. Material the limiter
already holds at the ceiling cannot get louder either: a brickwalled,
square-like input with a −5 LUFS target ends at −6.6 LUFS with all
density used and reports a 1.6 LU deficit.

**Measured case: clicks.** The peaky programme with a click every 300 ms
instead of every 1.1 s, `streaming` preset, integrated output from 10 to
60 s (target −16 LUFS). Before the spike guard and the final clipper the
limiter took about 13 dB off the clicks, the trim ran into its +6 dB range
and every input level missed ±1 LU (−17.69, −17.15, −17.11 LUFS). Now:

| Input | Spike guard off | On (`standard`) | On (`low`) | Limiter GR on clicks, mean (off → on) |
|---|---|---|---|---|
| −35 LUFS | −16.44 LUFS | −16.14 LUFS | −16.13 LUFS | 4.0 → 0.2 dB |
| −23 LUFS | −16.34 LUFS | −16.13 LUFS | −16.12 LUFS | 4.1 → 0.2 dB |
| −10 LUFS | −16.32 LUFS | −16.09 LUFS | −16.09 LUFS | 4.1 → 0.2 dB |

With the spike guard off the final clipper takes most of each click and
the output is on target, but the limiter still takes 4 dB off the clicks.
With it on, the clicks are removed before the AGC; the largest reduction
around a single click is 2.3 dB (a click that lands on a loud hit gets the
hit's own reduction).

**Spike guard limits:** a spike is at most 0.25 ms long at its core
and jumps to its peak in one sample (single-sample clicks and glitches;
a longer microphone bump, a plosive, or a click spread over several
samples by a filter or a codec is programme to it). A click that lands
on a loud passage, where the samples just before it are already over
the threshold, is left to the clipper and the limiter.

**Measured case: crackle on speech.** The corpus speech and phone clips
(18 clips, 30 s each, at the preset's target) through every preset and
latency mode, worst per-frame NMR against the linear reference over all
clips (`strom-dsp-harness` `tests/crackle.rs`, extended workflow). The
bound is 0 dB, where the error reaches the masked threshold; the dynamics
alone (AGC, multiband, expander, bass clipper) stay under −8 dB.

| Preset | Mode | Before | Now | Cause of what remains |
|---|---|---|---|---|
| `ebu-r128` | standard | +10.8 dB | +10.8 dB (−8.3 without the shortwave clip) | clipper on the shortwave clip's static crashes |
| `ebu-r128` | low | +37.6 | +5.6 (−9.3) | the same |
| `atsc-a85` | standard | +17.5 | +17.5 (−5.6) | the same |
| `atsc-a85` | low | +37.0 | +14.4 (−7.3) | the same |
| `streaming` | standard | +35.7 | +35.7 | clipper on speech peaks at −16 LUFS; limiter +6.3 without it |
| `streaming` | low | +43.1 | +35.7 | the same; limiter (0.6 ms) +13.9 without it |
| `speech` | standard | +35.4 | +2.7 | limiter on one clip's peaks at −18 LUFS |
| `speech` | low | +41.2 | +7.6 | limiter (0.6 ms) on three clips |

Before: the spike guard without the rise gate and the clipper at
`speech`. In the standard mode the old event rule also took plosive
bursts and mouth clicks for clicks (+0.8 … +4.1 dB on clean speech); the
rise gate leaves them too. What remains is listed in the check as known
failures, with these causes.

**Measured case: bass into highs.** A loud 50 Hz with a 3 kHz tone 20 dB
under it, the band compressors neutral, the bass clip threshold at the
final clipper's and the bass 3 dB over it, at a fixed drive (AGC and
governor off): without the IM feedback the clipper chops the 3 kHz at
every bass peak, and the sidebands at 3 kHz ± 50/100 Hz are −13.8 dBc;
with it the bass threshold comes down by its full 3 dB and the sidebands
fall to −53.8 dBc. With the presets the multiband and the bass clipper
keep the bass under the final clipper: at `streaming`, driven up to 9 dB
over the ceiling, the sidebands stay under −148 dBc.

**Measured case: noise between words.** Speech-like bursts with noise
40 dB below them through the `speech` preset: the noise between the bursts
drops by 5.8 dB, the bursts change by less than 0.01 dB.

**Clip budget mapping.** Each measure, over a gated 3 s window, is turned
into the budget that would allow it; the largest is compared with
`clip-budget` `B`, and the difference over it is `budget-excess`:

| Measure | Allowed at budget `B` |
|---|---|
| clip depth (removed / input energy of the clipper) | `−40 + 5·B` dB |
| limiter mean gain reduction | `B/2` dB |
| share of samples the clipper touches | `0.5 % · 2^(B/2)` |

**Over budget, on target.** The sport-like peaky programme with the spike
guard off (so its clicks reach the clipper) and `clip-budget` 1.5 at
`streaming` needs about 2.2 dB of budget at the target. The output lands
on target (−16.06 / −16.05 / −16.02 LUFS at −35 / −23 / −10 LUFS input)
and the governor reports an excess of about 0.65 dB. (An earlier version
cut the AGC target to hold the budget, which left the output 1.4 … 2.0 LU
under the target.)

**For a listening test:** speech band-limited to 3.4 kHz at `streaming`
(−16 LUFS) reaches the final clipper (clip depth −61 dB) and its NMR in the
harness rises from −27 dB (clipper off) to −11 dB; and two tones at 19 and
20 kHz driven 6 dB over the ceiling at `streaming` give a 1 kHz
difference tone at −74 dB through the clipper, where the limiter alone
left none.

## Final clipper latency

The clipper's structure is a library setting (`clipper_latency`) that
`latency-mode` chooses:

| Setting | Structure | Latency | 15 kHz alias | Worst overshoot | `latency-mode` |
|---|---|---|---|---|---|
| `standard` | 8×, one pass, 41-tap shaping | 69 frames (1.44 ms at 48 kHz) | −94.0 dBFS | +0.77 dB | `standard` |
| `low` | 4×, one pass, 3-tap shaping | 48 frames (1.00 ms at 48 kHz) | −75.8 dBFS | +2.17 dB | `low` |
| `two-pass` | 8×, two passes | 138 frames (2.88 ms at 48 kHz) | −94.0 dBFS | +0.27 dB | (library only) |

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
| `spike-events` | guint64 | spikes removed since the last message; 0 while the spike guard is off |
| `agc-gain` | gdouble, dB | signed; + is boost |
| `band-gain-reduction` | GstValueArray of gdouble | positive dB of reduction, one per active band |
| `limiter-gain-reduction` | gdouble | positive dB of reduction; max of values sampled every 512 frames |
| `band-expander-gain-reduction` | GstValueArray of gdouble | positive dB of expander attenuation, one per active band |
| `bass-clip-gr` | gdouble | positive dB of bass-clipper reduction, largest since the last message |
| `clip-depth` | gdouble, dB | energy the final clipper removed relative to its input since the last message; −200.0 when nothing |
| `loudness-deficit` | gdouble, LU | how far the output stays under the target with all of the governor's density used: a shortfall from the AGC range, the trim's range or the ceiling, never from the budget (≥ 0) |
| `budget-excess` | gdouble, dB | how far the clipper's and the limiter's work is over `clip-budget` at the target loudness (≥ 0; dB of budget, see the mapping) |
| `governor-state` | gchararray | `off`, `frozen` (pause, below the gate, a stage bypassed), `free`, `hold` (near the budget, or holding density), `density` (raising it: the target is out of reach otherwise), `over-budget` (on target, trim held, excess reported) or `release` (giving density back) |

## Example

```sh
GST_PLUGIN_PATH=target/release gst-launch-1.0 -m audiotestsrc wave=pink-noise ! \
  audio/x-raw,format=F32LE,rate=48000,channels=2 ! \
  stromdsp-broadcastprocessor preset=streaming meter-interval=500 ! autoaudiosink
```
