# Listening kit

Hear every processor before and after on real material, with the harness
numbers next to each version. Code is the source of truth — this may have
drifted; read the code for the current implementation.

## Run it

```sh
scripts/listen.sh ~/strom-dsp-listen
open ~/strom-dsp-listen/index.html      # or xdg-open on Linux
```

One command fetches the test corpus (about 165 MB, once), renders every
configuration and writes the page. It needs bash, curl, tar, gzip,
`sha256sum` or `shasum`, ffmpeg and a Rust toolchain, on macOS or Linux.
On an Apple M5 Pro the render takes about 30 s at the default 4 threads
(after the first build of the render tool).

| Option | What it does |
|---|---|
| `--from FILE` | use a local copy of the corpus archive instead of downloading it |
| `--corpus DIR` | where the corpus is unpacked (default `corpus/data/`, git-ignored) |
| `--no-wav` | write only the MP4 files: the output drops from about 1.1 GB to about 180 MB |
| `--jobs N` | render threads (default 4) |
| `--max-seconds S` | cut longer clips (default 30) |

The page is static: no server, no external requests. Open `index.html`
from disk, or serve the output directory with any static web server
(`python3 -m http.server`, `npx serve`, …) to listen on a phone. Pick a
processor, a configuration and a clip; the version buttons switch between
the original and every configuration while playing, at the same position.
"Match loudness" turns louder versions down to the quietest, so a
comparison is not just "louder sounds better".

On an iPhone the silent switch mutes Web Audio unless the page sets its
audio session to playback. The page does that where Safari supports it;
otherwise turn the silent switch off.

## What it renders

`strom-dsp-listen list` prints the current groups, configurations and clips.
In short:

- **Broadcast processor, presets and latency modes:** `ebu-r128`,
  `streaming` and `speech`, each in the standard and the low latency mode,
  on speech, phone, field, sport, music and clipped clips.
- **Broadcast processor, each stage on and off:** preset `streaming`, all
  stages, then the spike guard, expander, bass clipper, final clipper and
  governor switched off one at a time.
- **Channel strip on speech:** gate, gate as expander, compressor,
  de-esser and cleanup at the element defaults (cleanup with a usual
  speech setting, since its default is all off).
- **Declipper:** clean speech and music hard-clipped 3, 6 and 10 dB
  below their peak (the clipped version is shown too), and the corpus
  recordings that clipped before they reached us, each through the
  declipper's three latency settings. For the clips clipped here the page
  shows the SDR gain on the clipped samples against the original; the
  counters (runs repaired, fallbacks, runs rejected, runs too long, largest
  overshoot) are shown for all.

Every original is the clip scaled to −23 LUFS. Broadcast configurations
get it at their preset's target loudness, as the harness cases do. All
versions of a set share one gain (loudest sample peak at −1 dBFS), so the
level differences between versions are real.

Output: `sets/<group>/<clip>/` holds `<version>.mp4`, `<version>.wav`
(24-bit), `metrics.json` and the page's `audio.js`; `listen-data.json`
has every number of every set. The numbers are those of the measurement
harness (`crates/strom-dsp-harness`): loudness, LRA, true peak, and for
the broadcast processor NMR, clip depth, limiter, expander and bass clipper
gain reduction, governor and spike guard numbers.

## Add a processor

Add a group to `groups()` in `crates/strom-dsp-harness/src/listen/mod.rs`:
its clips, the numbers to show, and one configuration per setting.
`strip_config` wraps any `Processor`; a group that needs a damaged input
(a declipper on clipped speech) sets `pre`, whose output becomes an extra
version and the input of every configuration.

## The corpus

The clips are freely licensed (CC0, public domain, CC BY 3.0 / 4.0) and
ship as a release asset of this repository (`corpus-v1`), never in git.
`corpus/manifest.tsv` records each clip's source, licence, attribution,
extract, processing and SHA-256; `corpus/ATTRIBUTION.md` is the attribution
list. `scripts/fetch-corpus.sh` verifies the archive and every clip before
unpacking.

Maintainers build a new archive with `scripts/make-corpus-manifest.sh`
(manifest and attribution from the curated corpus's provenance record) and
`scripts/build-corpus-archive.sh` (clips as 24-bit FLAC, or 32-bit float WAV
for the few with overs above 0 dBFS, which FLAC cannot hold; it pins the
archive's SHA-256 in `corpus/VERSION`). Publishing the release asset is a
manual step.
