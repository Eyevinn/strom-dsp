# `stromdsp-compressor`: feed-forward compressor with soft knee

A channel-strip element: an in-place `BaseTransform` on F32LE interleaved
audio, any rate and channel count, with linked channels and no latency. The
bypass, caps and latency rules shared by all elements are in the
[element overview](README.md#shared-conventions).

## Properties

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `enabled` | bool | | | `true` |
| `threshold` | float | linear amplitude | 0 … 1 | 0.125 (−18 dBFS) |
| `ratio` | float | ratio (4 = 4:1) | 1 … 100 | 3 |
| `attack` | float | ms | 0.01 … 10000 | 10 |
| `release` | float | ms | 0.01 … 10000 | 150 |
| `makeup-gain` | float | linear multiplier | 0 … 100 | 1 |
| `knee` | float | linear factor | 0 … 1 | 0.5 |

## Behaviour

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

## Why these defaults

The defaults make the compressor do gentle speech and
programme levelling out of the box. They are strom-dsp's own choice from
textbook and broadcast practice (`strom_dsp::presets`, see [PROVENANCE.md](../../PROVENANCE.md)).
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

## Example

`threshold` and `makeup-gain` are linear: 0.25 is about −12 dBFS, and 1.5
is about +3.5 dB.

```sh
gst-launch-1.0 autoaudiosrc ! audioconvert ! audio/x-raw,format=F32LE,layout=interleaved ! \
  stromdsp-compressor threshold=0.25 ratio=4 knee=0.25 makeup-gain=1.5 detector=rms ! \
  audioconvert ! autoaudiosink
```
