#!/usr/bin/env bash
# Build the test corpus release archive from the curated clips. Maintainers
# only; publishing the archive as a release asset is a manual step.
#
# Usage: scripts/build-corpus-archive.sh CORPUS_DIR OUT_DIR
#   CORPUS_DIR  the curated corpus: <category>/<source>/<id>_48k.wav clips
#               (and <category>/generated/<id>.wav test signals)
#   OUT_DIR     where strom-dsp-corpus-<version>.tar.gz is written
#               (keep it outside the repository: audio is never committed)
#
# Reads corpus/manifest.tsv (make it with make-corpus-manifest.sh) and
# corpus/VERSION. For every manifest row it stores the clip under the
# row's `file` name:
#   - `.flac`: 24-bit FLAC, lossless from the 24-bit PCM clip (checked by
#     comparing the decoded samples with the source);
#   - `.wav`:  32-bit float WAV, for the few clips whose sample peak is
#     above 0 dBFS (overs from lossy decoding). FLAC holds only integer
#     samples, and a gain to fit them would change the level every
#     processor sees, so these stay float and bit-exact.
# Metadata is stripped and the encoders run in bit-exact mode, so the clip
# files (and their SHA-256) do not depend on the machine.
#
# Then it writes the SHA-256 of every clip into corpus/manifest.tsv, packs
# the clips, the manifest and corpus/ATTRIBUTION.md into
# strom-dsp-corpus-<version>.tar.gz (top directory
# strom-dsp-corpus-<version>/), writes the archive's SHA-256 into
# corpus/VERSION and prints it. Commit both files with the release.
#
# Needs: bash, ffmpeg/ffprobe, tar, gzip, sha256sum or shasum.
set -euo pipefail

usage() { sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; }
[[ $# -eq 2 ]] || { usage >&2; exit 2; }
corpus=$(cd "$1" && pwd)
mkdir -p "$2"
outdir=$(cd "$2" && pwd)
repo=$(cd "$(dirname "$0")/.." && pwd)
manifest=$repo/corpus/manifest.tsv
attribution=$repo/corpus/ATTRIBUTION.md
version_file=$repo/corpus/VERSION

for t in ffmpeg ffprobe tar gzip; do
    command -v "$t" >/dev/null || { echo "missing tool: $t" >&2; exit 1; }
done
sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
sha256_stdin() {
    if command -v sha256sum >/dev/null; then sha256sum | cut -d' ' -f1
    else shasum -a 256 | cut -d' ' -f1; fi
}

version=$(sed -n 's/^version=//p' "$version_file")
[[ -n $version ]] || { echo "no version= in $version_file" >&2; exit 1; }
name=strom-dsp-corpus-$version
archive=$outdir/$name.tar.gz

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
stage=$work/$name
mkdir -p "$stage"

# The curated clip for a manifest row: <category>/*/<id>_48k.wav, or
# <category>/*/<id>.wav for the generated signals.
find_clip() {
    local cat=$1 id=$2 f
    for f in "$corpus/$cat"/*/"${id}_48k.wav" "$corpus/$cat"/*/"$id.wav"; do
        [[ -f $f ]] && { printf '%s\n' "$f"; return 0; }
    done
    return 1
}

newmanifest=$work/manifest.tsv
head -n1 "$manifest" >"$newmanifest"
n=0
while IFS= read -r line; do
    # cut keeps empty fields (a tab is whitespace to `read`).
    id=$(cut -f1 <<<"$line"); cat=$(cut -f2 <<<"$line"); file=$(cut -f3 <<<"$line")
    src=$(find_clip "$cat" "$id") || { echo "clip not found for $cat/$id" >&2; exit 1; }
    dst=$stage/$file
    mkdir -p "$(dirname "$dst")"
    case $file in
        *.flac)
            bits=$(ffprobe -v error -select_streams a:0 -show_entries stream=bits_per_sample -of csv=p=0 "$src")
            [[ $bits == 24 ]] || { echo "$src: expected 24-bit PCM, got $bits" >&2; exit 1; }
            ffmpeg -nostdin -v error -y -i "$src" -map_metadata -1 -fflags +bitexact \
                -flags:a +bitexact -c:a flac -compression_level 8 "$dst"
            a=$(ffmpeg -nostdin -v error -i "$src" -f s32le - | sha256_stdin)
            b=$(ffmpeg -nostdin -v error -i "$dst" -f s32le - | sha256_stdin)
            [[ $a == "$b" ]] || { echo "$file: FLAC is not lossless" >&2; exit 1; }
            ;;
        *.wav)
            ffmpeg -nostdin -v error -y -i "$src" -map_metadata -1 -fflags +bitexact \
                -flags:a +bitexact -c:a pcm_f32le "$dst"
            a=$(ffmpeg -nostdin -v error -i "$src" -f f32le - | sha256_stdin)
            b=$(ffmpeg -nostdin -v error -i "$dst" -f f32le - | sha256_stdin)
            [[ $a == "$b" ]] || { echo "$file: float WAV differs from the source" >&2; exit 1; }
            ;;
        *) echo "unknown clip type: $file" >&2; exit 1 ;;
    esac
    sum=$(sha256 "$dst")
    awk -F'\t' -v OFS='\t' -v s="$sum" '{ $4 = s; print }' <<<"$line" >>"$newmanifest"
    n=$((n + 1))
done < <(tail -n +2 "$manifest")

cp "$newmanifest" "$manifest"
cp "$manifest" "$stage/manifest.tsv"
cp "$attribution" "$stage/ATTRIBUTION.md"

# A reproducible archive: fixed order, owner and modification time.
find "$stage" -exec touch -h -t 202601010000 {} +
list=$work/list
(cd "$work" && find "$name" -type f | LC_ALL=C sort) >"$list"
if tar --version 2>/dev/null | grep -q 'GNU tar'; then
    tar -C "$work" --owner=0 --group=0 --numeric-owner --format=ustar \
        -cf "$work/$name.tar" -T "$list"
else
    tar -C "$work" --uid 0 --gid 0 --uname '' --gname '' --format ustar \
        -cf "$work/$name.tar" -T "$list"
fi
gzip -n -9 -c "$work/$name.tar" >"$archive"

sum=$(sha256 "$archive")
tmpv=$work/VERSION
awk -v s="$sum" '/^sha256=/ { print "sha256=" s; next } { print }' "$version_file" >"$tmpv"
cp "$tmpv" "$version_file"

size=$(wc -c <"$archive" | tr -d ' ')
echo "clips:   $n"
echo "archive: $archive ($((size / 1048576)) MiB)"
echo "sha256:  $sum"
echo "updated: corpus/manifest.tsv (clip SHA-256) and corpus/VERSION (archive SHA-256)"
