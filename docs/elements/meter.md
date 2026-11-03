# `stromdsp-meter`: loudness, peak, RMS and correlation meter

A measurement element with no latency. The conventions shared by all
elements are in the [element overview](README.md#shared-conventions).

A passthrough meter: samples are never changed. It has klass
`Filter/Analyzer/Audio` and accepts F32LE interleaved audio with 1–64 channels.
Every `interval` ms of stream time it posts an element message.

## Properties

Units are dB, ms and Hz. Every property is mutable while PLAYING unless
noted.

| Property | Type | Unit | Range | Default |
|---|---|---|---|---|
| `interval` | uint | ms | 10 … 60000 | 100 |
| `true-peak` | bool | (changeable in NULL/READY only; applies at the next caps) | | `true` |

## Action signal `reset`

No arguments. It restarts integrated loudness and loudness range. It also
restarts every other loudness value and the programme peaks.

## Message format

The structure is named `stromdsp-meter`
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

## Example

```sh
gst-launch-1.0 -m audiotestsrc wave=pink-noise ! audioconvert ! \
  audio/x-raw,format=F32LE,channels=2 ! stromdsp-meter interval=100 ! autoaudiosink
```
