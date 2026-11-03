#!/usr/bin/env bash
# Listening kit in one command: fetch the test corpus (if needed), render
# every processor configuration over it, and build the listening page.
#
# Usage: scripts/listen.sh [--from ARCHIVE] [--corpus DIR] [--jobs N]
#                          [--no-wav] [--max-seconds S] OUT_DIR
#   OUT_DIR          where the audio, the numbers and index.html go
#   --from ARCHIVE   use a local copy of the corpus archive (see fetch-corpus.sh)
#   --corpus DIR     the unpacked corpus (default: corpus/data)
#   --jobs N         render threads (default: 4)
#   --no-wav         write only the MP4 files (much smaller output)
#   --max-seconds S  cut clips longer than this (default: 30)
#
# Then open OUT_DIR/index.html in a browser (no server needed), or serve
# OUT_DIR with any static web server to listen on a phone.
#
# Needs: bash, curl, tar, gzip, sha256sum or shasum, ffmpeg, cargo.
set -euo pipefail

usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; }

repo=$(cd "$(dirname "$0")/.." && pwd)
from=()
corpus=$repo/corpus/data
render_args=()
out=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --from) from=(--from "${2:?--from needs a file}"); shift 2 ;;
        --corpus) corpus=${2:?--corpus needs a directory}; shift 2 ;;
        --jobs) render_args+=(--jobs "${2:?--jobs needs a number}"); shift 2 ;;
        --max-seconds) render_args+=(--max-seconds "${2:?--max-seconds needs a number}"); shift 2 ;;
        --no-wav) render_args+=(--no-wav); shift ;;
        -h | --help) usage; exit 0 ;;
        -*) usage >&2; exit 2 ;;
        *) [[ -z $out ]] || { usage >&2; exit 2; }; out=$1; shift ;;
    esac
done
[[ -n $out ]] || { usage >&2; exit 2; }
for t in ffmpeg cargo; do
    command -v "$t" >/dev/null || { echo "missing tool: $t" >&2; exit 1; }
done

start=$SECONDS
"$repo/scripts/fetch-corpus.sh" --dest "$corpus" ${from[@]+"${from[@]}"}

echo "building the render tool"
cargo build --release --quiet --manifest-path "$repo/Cargo.toml" --bin strom-dsp-listen
tool=$(cd "$repo" && cargo metadata --format-version 1 --no-deps |
    sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')/release/strom-dsp-listen

mkdir -p "$out"
"$tool" render --corpus "$corpus" --out "$out" ${render_args[@]+"${render_args[@]}"}
"$tool" page --out "$out"
cp "$corpus/ATTRIBUTION.md" "$out/ATTRIBUTION.md"

echo "done in $((SECONDS - start)) s: $(du -sh "$out" | cut -f1) in $out"
echo "open $out/index.html"
