# strom-dsp

A small, clean-room audio DSP library in Rust, and a GStreamer plugin on top of
it, built for the [Strom](https://github.com/Eyevinn/strom) live media engine.

- **`strom-dsp`** (`crates/strom-dsp`): the DSP library. It has no GStreamer
  dependency and is real-time safe: after `configure` it does no allocation,
  locking, logging or I/O. It processes interleaved `f32` in place, with any
  channel count and 8–192 kHz. It uses SIMD with runtime dispatch (NEON,
  SSE4.2/AVX2/AVX-512).
- **`gst-plugin-stromdsp`** (`crates/gst-plugin-stromdsp`): the GStreamer
  plugin `stromdsp` with eleven elements:
  - the channel strip: `stromdsp-gate`, `stromdsp-compressor`,
    `stromdsp-equalizer`, `stromdsp-limiter`;
  - utility and measurement elements: `stromdsp-ducker`, `stromdsp-deesser`,
    `stromdsp-cleanup`, `stromdsp-delay`, `stromdsp-stereo`, `stromdsp-meter`;
  - the broadcast final processor: `stromdsp-broadcastprocessor`.

  It builds as a `cdylib` (for `GST_PLUGIN_PATH`) and as an `rlib` for static
  linking (`gststromdsp::plugin_register_static()`).

Licence: MIT OR Apache-2.0. See [PROVENANCE.md](PROVENANCE.md) for the
clean-room record: which reference each algorithm follows, and every
dependency with its licence.

## Elements

**Bypass.** Every processing element has an `enabled` property, mutable while
PLAYING. Toggling `enabled` crossfades between processed and dry (20 ms for
most elements, 10 ms for the limiter, stereo and broadcast processor), then
the output is bit-exact,
delayed by the element's reported latency (zero for most elements; the
limiter and the broadcast processor keep their lookahead delay).

### Channel strip

The four channel-strip elements are in-place `BaseTransform`s:
- caps `audio/x-raw, format=F32LE, layout=interleaved, rate=[1,MAX], channels=[1,MAX]`
- klass `Filter/Effect/Audio`
- every property controllable while PLAYING, except `num-bands`
- latency: none, except the limiter (`lookahead`, see below)

Gate, compressor and limiter use **linked channels**: one gain is computed from
the loudest channel and applied to all, so the stereo image never moves.

#### `stromdsp-gate`: noise gate with hysteresis and hold

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

**Why these defaults.** The defaults make the gate close cleanly in speech
pauses on its own. They are strom-dsp's own choice from textbook and broadcast
practice (`strom_dsp::presets`, see `PROVENANCE.md`). The levels assume EBU
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

#### `stromdsp-compressor`: feed-forward compressor with soft knee

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `threshold` | float | linear amplitude | 0 … 1 | 0.125 (−18 dBFS) |
| `ratio` | float | ratio (4 = 4:1) | 1 … 100 | 3 |
| `attack` | float | ms | 0.01 … 10000 | 10 |
| `release` | float | ms | 0.01 … 10000 | 150 |
| `makeup-gain` | float | linear multiplier | 0 … 100 | 1 |
| `knee` | float | linear factor | 0 … 1 | 0.5 |

**Knee definition.**
- `knee` is a linear amplitude factor `k`.
- The soft-knee region spans from `threshold · k` to `threshold / k`, so the
  knee width in dB is `W = −40·log10(k)`. The default 0.5 gives about 12 dB.
- `k = 1` is a hard knee. Values below 0.001 are treated as 0.001.
- Inside the knee the gain follows the quadratic interpolation of Giannoulis,
  Massberg & Reiss (JAES 2012).
- Ratio 1 means unity (apart from makeup).
- `knee` reads back exactly as written.

**Detector and automatic makeup.** All three are controllable and mutable
in PLAYING.

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `detector` | enum `StromDspCompressorDetector` | `peak` (0), `rms` (1) | | `rms` |
| `rms-window` | float | ms | 1 … 100 | 10 |
| `auto-makeup` | bool | | | `false` |

- **`detector`:** `peak` follows the linked sample peak. `rms` averages power
  over `rms-window`, the classic choice for programme and speech. Both read a
  steady sine at its peak level, so `threshold` means the same in both modes;
  in `rms` mode, noise reads its RMS + 3 dB.
- **`auto-makeup`:** keeps the perceived level roughly where it was without
  compression. It makes up the gain reduction the programme actually gets.
  - The estimate is energy-weighted over about the last 10 s.
  - It freezes in pauses and silence, below −50 dB detector level.
  - It is capped at 24 dB and never rises on noise.
  - When threshold, ratio or knee change, the makeup follows at once.
  - Changes glide over 10 ms, and `makeup-gain` applies on top.
  - Trade-off: level differences that last much longer than 10 s are evened
    out less.
- **Peak detector on speech:** with the `peak` detector, speech is compressed
  about 3 dB harder than with `rms`, because the detector hold reads the true
  peak of every cycle. Use `rms` or a higher threshold for gentler speech
  compression.
- **Detector hold:** the detector holds its level for min(20 ms, release)
  after the level falls, so bass is not distorted at fast attack and release
  times. The release starts after that hold.

**Why these defaults.** The defaults make the compressor do gentle speech and
programme levelling out of the box. They are strom-dsp's own choice from
textbook and broadcast practice (`strom_dsp::presets`, see `PROVENANCE.md`).
- `threshold` 0.125 (−18.06 dBFS): the EBU R68 alignment level. Speech at
  −23 LUFS (EBU R128) sits around it, so loud syllables are compressed and
  quiet speech is not. The value 2⁻³ is exact in `f32`.
- `ratio` 3: a moderate ratio for speech and programme. It levels audibly
  without the flattened sound of 4:1 and above.
- `attack` 10 ms: lets the first few milliseconds of a transient through,
  which keeps articulation, while still catching each syllable.
- `release` 150 ms: shorter than the syllable spacing (200–250 ms), so the
  gain recovers between syllables. It is long compared with bass periods
  (20 ms at 50 Hz), so it neither distorts nor pumps.
- `knee` 0.5 (≈ 12 dB, from −24 to −12 dBFS): compression sets in gradually
  (Giannoulis et al. 2012).
- `makeup-gain` 1: inserting the element does not raise the level.
- `detector` `rms`, `rms-window` 10 ms: an average level follows perceived
  loudness better than the sample peak, the classic choice for speech and
  programme. 10 ms covers a period of 100 Hz and matches the gate's detector.
- `auto-makeup` `false`: the element never raises the level by itself.
  Automatic makeup is programme-dependent (see above): it learns the
  programme over about 10 s and, like any makeup, also lifts the noise in
  pauses (up to 24 dB after heavy compression). That is a choice for an
  operator who knows the source, not a safe default for an element inserted
  into a live chain.

#### `stromdsp-equalizer`: parametric EQ, up to 8 biquad bands

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `num-bands` | uint | | 1 … 8 (NULL/READY only) | 4 |
| `bandN-enabled` | bool | | | `true` |
| `bandN-type` | int | see below | 0 … 9 | 0 |
| `bandN-frequency` | float | Hz | 1 … 24000 | 1000 |
| `bandN-gain` | float | dB | −60 … 60 | 0 |
| `bandN-q` | float | | 0.01 … 100 | 0.707 |

All eight `bandN-*` sets (N = 0 … 7) always exist. Bands at index ≥
`num-bands` are inactive.

`bandN-type` values:

| Value | Type |
|---|---|
| 0 | off |
| 1 | low-pass |
| 2 | high-pass |
| 3 | band-pass, constant skirt gain |
| 4 | band-pass, 0 dB peak |
| 5 | notch |
| 6 | all-pass |
| 7 | peaking |
| 8 | low shelf |
| 9 | high shelf |

Coefficients come from the RBJ Audio EQ Cookbook, using the Q form for the
shelves as well. `gain` applies to peaking and the shelves only. Frequency is
clamped just below Nyquist. Parameter changes glide smoothly, and type changes
crossfade, so there are no clicks.

**Why these defaults.** The equalizer is neutral by default: every band is
`off` (type 0) with 0 dB gain, so the element is a transparent pass-through
until an operator shapes it. Any built-in curve would be wrong for most sources,
and equalisation is always a decision for a specific microphone, room or
programme. The parked band settings make a first edit easy: 1000 Hz is the
middle of the speech band, Q 0.707 is the Butterworth value (a moderate width
for shelves and peaks), and 4 bands is a common size for a channel-strip EQ.

#### `stromdsp-limiter`: lookahead peak limiter

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `threshold` | float | dB (ceiling) | −60 … 0 | −1 |
| `attack` | float | ms | 0.01 … 50 | 1 |
| `release` | float | ms | 1 … 2000 | 200 |
| `lookahead` | float | ms | 0 … 20 | 5 |

**Ceiling guarantee.** No output sample exceeds `threshold`, for any input and
also with `lookahead = 0`. The gain ramp is built so that it reaches the
required gain before the peak arrives, and it rounds only downwards. A final
safety clip exists as a guard only; tests show it never triggers on normal
material.

**Release hold.** After the last sample that needed gain reduction, the
limiter holds the gain for min(20 ms, release) before the release starts. This
keeps bass undistorted at fast attack and release, because the gain no longer
rises between the waveform peaks of low notes. The ceiling guarantee is
unaffected.

**Latency.**
- The audio is delayed by `lookahead`, and the element adds it to the LATENCY
  query.
- A change of `lookahead` posts a `Latency` message.
- Timestamps pass through unchanged.

**Disabled behaviour.** With `enabled=false` the limiter applies unity gain but
**keeps the delay**, so the latency does not jump when an operator toggles it.

**Why these defaults.** The defaults give a programme-safe peak ceiling. They
are strom-dsp's own choice from textbook and broadcast practice
(`strom_dsp::presets`, see `PROVENANCE.md`).
- `threshold` −1 dBFS: the EBU R128 maximum true-peak level (−1 dBTP). This is
  a sample-peak ceiling, so inter-sample peaks can still exceed it slightly.
- `attack` 1 ms: a smooth gain ramp instead of a step on each peak. It fits
  well inside the lookahead, so the ceiling still holds.
- `release` 200 ms: suited to programme material. It is long compared with
  bass periods, so there is no low-frequency distortion, and short enough to
  recover between phrases without audible pumping.
- `lookahead` 5 ms: 5 ms of latency, which Strom relies on. It covers the
  attack with margin.

### Utility and measurement elements

Units are dB, ms and Hz. Every property is mutable while PLAYING unless
noted. None of these elements adds latency.

#### `stromdsp-ducker`: lower the programme while a key is present

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

**Range modes.**
- `fixed`: duck by `depth` while the key is above `threshold`.
- `proportional`: 1 dB of reduction per dB the key is above `threshold`, up to
  `depth`.

**Meter message.** The element posts an element message with structure name
`stromdsp-ducker` at the first buffer, and then at most once per
`meter-interval`:

| Field | Type | Meaning |
|---|---|---|
| `running-time`, `stream-time` | guint64 | ns, end of the triggering buffer |
| `reduction` | gdouble | dB (≤ 0), gain applied |
| `key-level` | gdouble | dBFS, sine-calibrated RMS after the key filter; −200 for silence |
| `key-active` | gboolean | key above threshold or within hold |
| `key-present` | gboolean | real key samples covered part of that buffer |

**Why these defaults.** The defaults duck music or programme under a
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

```sh
gst-launch-1.0 -m \
  audiotestsrc is-live=true wave=sine freq=220 volume=0.5 \
    ! audio/x-raw,format=F32LE,rate=48000,channels=2 \
    ! stromdsp-ducker name=duck threshold=-30 depth=-12 attack=20 hold=300 release=500 \
    ! audioconvert ! autoaudiosink \
  audiotestsrc is-live=true wave=pink-noise volume=0.2 \
    ! audio/x-raw,format=F32LE,rate=16000,channels=1 ! duck.sidechain
```

#### `stromdsp-deesser`: sibilance reduction

A linked-channel de-esser.
- A 12 dB/octave high-pass detector at `frequency` drives a gain reduction.
  Sibilance is held at `threshold`, and the reduction is at most
  `max-reduction` dB.
- `split-band` reduces only the band above `frequency`, split by an LR4
  crossover. With no reduction the bands sum to a flat all-pass.
- `wideband` reduces the whole signal. It goes through the same all-pass
  split, so mode changes glide.
- No latency. Toggling `enabled` crossfades over 20 ms, then the output is
  bit-exact.

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `frequency` | float | Hz (limited to 0.45·fs) | 1000 … 20000 | 6000 |
| `threshold` | float | dBFS (detector peak) | −60 … 0 | −30 |
| `max-reduction` | float | dB (positive = attenuation) | 0 … 40 | 10 |
| `attack` | float | ms (63 % time constant) | 0.01 … 200 | 1 |
| `release` | float | ms (63 % time constant) | 1 … 2000 | 50 |
| `mode` | enum `StromDspDeesserMode` | | `split-band` (0), `wideband` (1) | `split-band` |

**Why these defaults.** The defaults tame sibilance on live speech without
dulling the voice.
- `frequency` 6 kHz: the energy of "s" and "sh" sounds sits mostly from
  about 5 to 10 kHz, while the body and most consonants of the voice lie
  below.
- `threshold` −30 dBFS: at EBU R68/R128 alignment, sharp sibilants exceed it
  in the high-passed detector, while normal high-frequency content mostly
  stays below.
- `max-reduction` 10 dB: enough to tame harsh sibilance, not so much that
  the talker starts to sound lisping.
- `attack` 1 ms: sibilants start abruptly, so the reduction must be in place
  within their first milliseconds.
- `release` 50 ms: recovers before the following vowel, so the next syllable
  is not dulled.
- `mode` `split-band`: only the band above `frequency` is reduced, so the
  level and body of the voice are untouched.

```sh
gst-launch-1.0 pulsesrc ! audioconvert ! audio/x-raw,format=F32LE,layout=interleaved ! \
  stromdsp-deesser frequency=6500 threshold=-32 max-reduction=8 mode=split-band ! \
  audioconvert ! autoaudiosink
```

#### `stromdsp-cleanup`: DC, rumble and hum removal

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

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `dc-block` | bool | | | `false` |
| `highpass` | float | Hz (−3 dB), 0 = off | 0 … 1000 | 0 |
| `highpass-order` | uint | 6 dB/oct per order | 1 … 8 | 2 |
| `hum` | enum `StromDspCleanupHum` | | `off` (0), `50` (50), `60` (60) | `off` |
| `hum-harmonics` | uint | number of notches | 1 … 10 | 5 |
| `hum-q` | float | Q of the fundamental; bandwidth f0/`hum-q` | 6 … 100 | 10 |

**Why these defaults.** Every stage is off, so inserting the element never
changes the sound: whether a source needs a low cut, and whether the local
mains hum is at 50 or 60 Hz, depends on the source and the site. The parked
values make switching a stage on safe:
- `highpass-order` 2 (12 dB/octave): a gentle microphone-style low cut that
  removes rumble without thinning the voice above the corner.
- `hum-harmonics` 5: the fundamental up to 250 Hz (50 Hz mains) or 300 Hz
  (60 Hz), where most hum energy lies.
- `hum-q` 10: notches 5 or 6 Hz wide, narrow enough to leave the programme
  between them intact and wide enough for normal mains frequency drift.

```sh
gst-launch-1.0 filesrc location=field.wav ! wavparse ! audioconvert ! \
  audio/x-raw,format=F32LE,layout=interleaved ! \
  stromdsp-cleanup dc-block=true highpass=70 highpass-order=4 hum=50 hum-harmonics=8 ! \
  audioconvert ! autoaudiosink
```

#### `stromdsp-delay`: intentional delay for alignment

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

**Why these defaults.** `delay` 0 and no `channel-delays`: the element is a
bit-exact passthrough until an operator sets the delay, because every
alignment value (microphone distance, video pipeline delay) is specific to
the setup. A wrong default delay would put audio out of sync.

```sh
gst-launch-1.0 audiotestsrc ! audio/x-raw,format=F32LE,channels=2,rate=48000 ! \
  stromdsp-delay delay=40 'channel-delays=<0.0, 2.5>' ! autoaudiosink
```

#### `stromdsp-stereo`: stereo utility

A stereo utility for exactly 2 channels (caps `channels=2`). It applies one
2×2 matrix: balance · width · mode · polarity.
- **Polarity:** applied to the inputs.
- **M/S:** `M = (L+R)/2` and `S = (L−R)/2`. Decode is `L = M+S`, `R = M−S`, so
  encode followed by decode is exact.
- **Width:** scales S. 0 is mono, 1 is unchanged, 2 is wide. There is no gain
  compensation.
- **Balance:** constant power, with `θ = (b+1)·π/4`, `gL = √2·cos θ` and
  `gR = √2·sin θ`. The centre is unity; at ±1 the remaining side is +3 dB.

Changes ramp over 10 ms. Pure copy, swap and invert are applied bit-exactly.
The defaults, and `enabled=false` once its ramp ends, are bit-exact
passthrough. No latency.

| Property | Type | Range / values | Default |
|---|---|---|---|
| `enabled` | bool | | `true` |
| `mode` | enum `StromDspStereoMode` | `stereo`, `mono`, `left-to-both`, `right-to-both`, `swap`, `ms-encode`, `ms-decode` | `stereo` |
| `mono-law` | enum `StromDspStereoMonoLaw` | `minus-3db` (1/√2), `minus-6db` (1/2) | `minus-3db` |
| `invert-left`, `invert-right` | bool | | `false` |
| `balance` | float | −1 … 1 | 0 |
| `width` | float | 0 … 2 | 1 |

**Why these defaults.** `mode` `stereo`, `width` 1, `balance` 0 and no
inversion: the element is a bit-exact passthrough until an operator changes
it, so inserting it never alters the stereo image. `mono-law` `minus-3db`:
when an operator folds to mono, uncorrelated material (wide stereo, room
sound) keeps its power, and centred material rises only 3 dB. This is the
usual compromise for mixed programme; `minus-6db` keeps centred material
(a mono voice) at its level instead.

```sh
gst-launch-1.0 audiotestsrc ! audio/x-raw,format=F32LE,channels=2,rate=48000 ! \
  stromdsp-stereo mode=mono mono-law=minus-6db ! autoaudiosink
```

#### `stromdsp-meter`: loudness, peak, RMS and correlation meter

A passthrough meter: samples are never changed. It has klass
`Filter/Analyzer/Audio` and accepts F32LE interleaved audio with 1–64 channels.
Every `interval` ms of stream time it posts an element message.

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `interval` | uint | ms | 10 … 60000 | 100 |
| `true-peak` | bool | (changeable in NULL/READY only; applies at the next caps) | | `true` |

**Action signal `reset`** (no arguments). It restarts integrated loudness and
loudness range. It also restarts every other loudness value and the
programme peaks.

**Message format.** The structure is named `stromdsp-meter`
(`gststromdsp::meter::MESSAGE_NAME`).

| Field | Type | Unit / meaning |
|---|---|---|
| `stream-time`, `running-time` | guint64 | ns, end (PTS + duration) of the last measured buffer; `GST_CLOCK_TIME_NONE` if unknown |
| `duration` | guint64 | ns of audio covered by the interval values |
| `channels` | gint | |
| `momentary`, `short-term`, `integrated` | gdouble | LUFS |
| `loudness-range` | gdouble | LU (0 without data) |
| `sample-peak` | GstValueArray of gdouble | dBFS per channel, this interval |
| `sample-peak-max` | gdouble | dBFS, highest channel |
| `true-peak` | GstValueArray of gdouble | dBTP per channel, this interval (omitted when `true-peak=false`) |
| `true-peak-max` | gdouble | dBTP (omitted when `true-peak=false`) |
| `programme-sample-peak` | gdouble | dBFS since start or `reset` |
| `programme-true-peak` | gdouble | dBTP since start or `reset` (omitted when `true-peak=false`) |
| `rms` | GstValueArray of gdouble | dBFS per channel over the interval, unweighted (a full-scale sine reads −3.01) |
| `correlation` | gdouble | −1 … +1, broadband (300 ms integration); only for exactly 2 channels |
| `correlation-low`, `correlation-mid`, `correlation-high` | gdouble | −1 … +1 below 250 Hz, 250 Hz–4 kHz, above 4 kHz; only for 2 channels |
| `stereo-state` | gchararray | `"mono"`, `"stereo"` or `"inverted"`; only for 2 channels |

**Silence and timing.**
- Silence is reported as **−200.0** (`gststromdsp::meter::SILENCE_DB`). No
  value is ever NaN or −∞, and a silent channel or band reads correlation 0.
- At most one message is posted per buffer.
- FLUSH_STOP and DISCONT keep the integrated measurement.

**Channel weights.** These follow BS.1770 when the caps carry a channel mask:
- LFE is excluded;
- side/surround positions, and rear positions in 5.x, are weighted 1.41;
- everything else is weighted 1.0.

Without a mask, every channel is weighted 1.0.

**Cost.** The loudness readings, the expensive part, are taken only when a
message is due (`cargo bench -p strom-dsp --bench meter`).

**Why these defaults.**
- `interval` 100 ms: ten messages per second, smooth for an operator's
  meter display and fine enough to follow the 400 ms momentary loudness,
  while keeping the cost of the readings low.
- `true-peak` `true`: EBU R128 and ATSC A/85 set their peak limits in dBTP,
  so the meter shows the value that delivery compliance is checked against.

```sh
gst-launch-1.0 -m audiotestsrc wave=pink-noise ! audioconvert ! \
  audio/x-raw,format=F32LE,channels=2 ! stromdsp-meter interval=100 ! autoaudiosink
```

### Broadcast final processor: `stromdsp-broadcastprocessor`

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

**Latency:** `round(limiter-lookahead · fs / 1000) + 32` frames: 272 frames at
48 kHz with 5 ms. It is reported in the LATENCY query, is the same with
`enabled=false` and with any stage bypassed, and a `Latency` message is posted
when it changes (only at caps negotiation). With `enabled=false` the output is
the input delayed by exactly that latency. The element never uses
BaseTransform passthrough.

#### Properties

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

#### Presets

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

**Why these defaults.** The element starts as the `ebu-r128` preset at
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

**Measured result:** a 60 s synthetic programme at −35, −23 and −10 LUFS input
lands within ±1 LU of each preset's target (integrated, after the first
10 s), and true peak stays ≤ ceiling + 0.1 dB (both measured with `ebur128`).

**Known limitation:** very peaky material, such as frequent clicks or hard
percussive hits, can land 1–3 LU below target. The multiband stage and the
limiter come after the AGC, so the level they remove is not made up. True peak
still holds.

#### Signals

- **`get-preset-values(gchararray nick) -> GstStructure`:** returns a
  structure named after the preset, with one field per property the preset
  sets, for the current `num-bands`. It returns NULL for `custom` or an
  unknown nick.
- **`reset()`:** restarts the output integrated loudness.

#### Meter message

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

```sh
GST_PLUGIN_PATH=target/release gst-launch-1.0 -m audiotestsrc wave=pink-noise ! \
  audio/x-raw,format=F32LE,rate=48000,channels=2 ! \
  stromdsp-broadcastprocessor preset=streaming meter-interval=500 ! autoaudiosink
```

## Build and test

You need Rust 1.92 or newer and GStreamer ≥ 1.22 development files:
- Debian/Ubuntu: `libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev`
- macOS: `brew install gstreamer`
- Windows: the official MSVC runtime and devel installers

The library alone builds with Rust 1.89.

```sh
cargo build --release
cargo test --all
cargo clippy --all-targets -- -D warnings
cargo deny check licenses
```

Optional checks, described in [docs/PERFORMANCE.md](docs/PERFORMANCE.md):
- RealtimeSanitizer on Linux/macOS: every `crates/strom-dsp/tests/rtsan*.rs`
  file runs under RealtimeSanitizer, one `--test` at a time, e.g.
  `RTSAN_ENABLE=1 cargo test -p strom-dsp --features rtsan --test rtsan`.
- Instruction-count benchmarks with gungraun on Linux.

## Use with GStreamer

```sh
cargo build --release
export GST_PLUGIN_PATH="$PWD/target/release"
gst-inspect-1.0 stromdsp

gst-launch-1.0 audiotestsrc wave=pink-noise ! audioconvert ! \
  stromdsp-gate open-threshold=-40 close-threshold=-50 ! \
  stromdsp-compressor threshold=0.25 ratio=4 knee=0.25 makeup-gain=1.5 ! \
  stromdsp-equalizer num-bands=2 band0-type=2 band0-frequency=80 \
                     band1-type=7 band1-frequency=3000 band1-gain=3 band1-q=1 ! \
  stromdsp-limiter threshold=-1 lookahead=5 ! \
  audioconvert ! autoaudiosink
```

The plugin file is `libgststromdsp.so` on Linux, `libgststromdsp.dylib` on
macOS and `gststromdsp.dll` on Windows.

To link statically into an application, depend on the crate and call:

```rust
gststromdsp::plugin_register_static()?;
let comp = gst::ElementFactory::make("stromdsp-compressor").build()?;
```

## Using the library directly

```rust
use strom_dsp::{Compressor, CompressorParams, Processor};

let mut comp = Compressor::new();
comp.configure(48_000.0, 2); // allocates; do this off the audio thread
comp.set_params(&CompressorParams { threshold: 0.25, ratio: 4.0, ..Default::default() });
comp.process(&mut interleaved_f32); // real-time safe
```

Reusable building blocks are public too:
- `strom_dsp::biquad`: RBJ coefficient design, analytic magnitude, a DF-I
  `Biquad` with coefficient ramps.
- `strom_dsp::dynamics`, the building blocks of the dynamics processors:
  - smoothers and envelope followers: `OnePole`, `AttackRelease`,
    `PeakEnvelope`, `MeanSquareEnvelope` (RMS);
  - the soft-knee `GainComputer`;
  - the `GateStateMachine` and `GateFade` (one-pole attack, full-range dB
    release);
  - the sidechain `KeyDetector` (key filter, linked peak);
  - `DelayLine`, `SlidingMin`/`SegmentedMin` and the `RampSmoother`.
- `strom_dsp::filter`: Butterworth high-/low-pass (order 1–8) and the
  Linkwitz-Riley `Crossover` (LR2/4/8, 1–5 bands, all-pass compensated).
- `strom_dsp::oversample`: polyphase half-band oversampler, 2×/4×/8×, with
  integer latency.
- `strom_dsp::stft`: STFT framework (√Hann WOLA over `realfft`) with a
  per-frame spectrum callback.
- `strom_dsp::meter`: `LoudnessMeter` (BS.1770 via `ebur128`), the
  low-cost BS.1770 `LoudnessDetector` (momentary and short-term),
  `CorrelationMeter`, `BandCorrelation` and `MonoDetector`.
- `strom_dsp::agc`: `Agc`, the gated, rate-limited loudness-targeting AGC.
- `strom_dsp::multiband`: `MultibandCompressor`, an LR4 crossover with one
  compressor per band (1–5 bands).
- `strom_dsp::truepeak`: `TruePeakLimiter`, a lookahead limiter with a 4×
  oversampled true-peak detector (BS.1770 Annex 2).
- `strom_dsp::broadcast`: `BroadcastProcessor`, the chain behind
  `stromdsp-broadcastprocessor` (input gain → AGC → multiband → true-peak
  limiter), with its meters.
- `strom_dsp::presets`: the element defaults (`GateParams::broadcast()` and
  friends) and the broadcast presets (`BroadcastPreset`).
- Processors behind the utility elements: `ducker`, `deesser`, `cleanup`,
  `delay` and `stereo`.

## Performance

`cargo bench -p strom-dsp` runs the criterion benchmarks: the channel strip
(gate → compressor → 4-band EQ → limiter), each processor on its own, and the
building blocks (filters, oversampler, STFT, meters, AGC, multiband,
true-peak limiter, broadcast chain). Run a single one with
`cargo bench -p strom-dsp --bench <name>`. "% of one core" is the time per
buffer divided by the buffer's duration (480 frames at 48 kHz are 10 ms).

<!-- BENCHMARK NUMBERS -->

For the full benchmark list, instruction-count benchmarks (gungraun), the
RealtimeSanitizer tests, SIMD design notes and the denormal policy, see
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Contributing

Every element, present and future, follows the
[element guidelines](docs/ELEMENT_GUIDELINES.md): `enabled` bypass that fades
and ends bit-exact, click-free live changes of every PLAYING-mutable property,
real-time safety, reported and constant latency, sensible standalone defaults,
a test per rule, and clean-room provenance. strom-dsp is a clean-room project:
keep [PROVENANCE.md](PROVENANCE.md) current with the reference for every
algorithm and every reused crate.
