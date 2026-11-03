#!/usr/bin/env bash
# Fetch the strom-dsp test corpus: download the release archive pinned in
# corpus/VERSION, verify it and every clip, and unpack it.
#
# Usage: scripts/fetch-corpus.sh [--from FILE] [--dest DIR] [--force]
#   --from FILE  use a local copy of the archive instead of downloading it
#   --dest DIR   where the clips go (default: corpus/data, git-ignored)
#   --force      unpack again even if DIR already holds this version
#
# The archive's SHA-256 must match corpus/VERSION, and every clip's SHA-256
# must match corpus/manifest.tsv; anything that differs is refused and
# nothing is left in DIR. Running it again when DIR already holds a
# verified copy of this version does nothing (it checks the clips again).
#
# Needs: bash, curl, tar, gzip, sha256sum or shasum. macOS and Linux.
set -euo pipefail

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; }

repo=$(cd "$(dirname "$0")/.." && pwd)
version_file=$repo/corpus/VERSION
manifest=$repo/corpus/manifest.tsv
from=""
dest=$repo/corpus/data
force=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --from) from=${2:?--from needs a file}; shift 2 ;;
        --dest) dest=${2:?--dest needs a directory}; shift 2 ;;
        --force) force=1; shift ;;
        -h | --help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
for t in tar gzip; do
    command -v "$t" >/dev/null || { echo "missing tool: $t" >&2; exit 1; }
done
command -v sha256sum >/dev/null || command -v shasum >/dev/null ||
    { echo "missing tool: sha256sum or shasum" >&2; exit 1; }

version=$(sed -n 's/^version=//p' "$version_file")
url=$(sed -n 's/^url=//p' "$version_file")
want=$(sed -n 's/^sha256=//p' "$version_file")
[[ -n $version && -n $url && ${#want} -eq 64 ]] ||
    { echo "corpus/VERSION needs version=, url= and sha256=" >&2; exit 1; }
name=strom-dsp-corpus-$version

# Every clip of the manifest present in `dir` with its SHA-256.
verify_clips() {
    local dir=$1 bad=0 n=0 file sum got
    while IFS=$'\t' read -r file sum; do
        n=$((n + 1))
        if [[ ! -f $dir/$file ]]; then
            echo "missing: $file" >&2; bad=$((bad + 1)); continue
        fi
        got=$(sha256 "$dir/$file")
        if [[ $got != "$sum" ]]; then
            echo "checksum differs: $file" >&2; bad=$((bad + 1))
        fi
    done < <(awk -F'\t' 'NR > 1 { print $3 "\t" $4 }' "$manifest")
    [[ $bad -eq 0 ]] || { echo "$bad of $n clips failed verification" >&2; return 1; }
    echo "$n clips verified"
}

stamp=$dest/.corpus-version
if [[ $force -eq 0 && -f $stamp && $(cat "$stamp") == "$version $want" ]]; then
    if verify_clips "$dest"; then
        echo "corpus $version already in $dest"
        exit 0
    fi
    echo "the copy in $dest is incomplete or changed; fetching again" >&2
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
archive=$work/$name.tar.gz
if [[ -n $from ]]; then
    [[ -f $from ]] || { echo "no such file: $from" >&2; exit 1; }
    cp "$from" "$archive"
else
    command -v curl >/dev/null || { echo "missing tool: curl" >&2; exit 1; }
    echo "downloading $url"
    curl --fail --location --retry 3 --proto '=https' --output "$archive" "$url"
fi

got=$(sha256 "$archive")
if [[ $got != "$want" ]]; then
    echo "archive SHA-256 is $got, expected $want: refusing it" >&2
    exit 1
fi

# Unpack into a scratch directory, refuse any path outside the top
# directory, verify every clip, then move into place.
if tar -tzf "$archive" | grep -v "^$name/" | grep -q .; then
    echo "archive holds paths outside $name/: refusing it" >&2; exit 1
fi
if tar -tzf "$archive" | grep -q '\(^\|/\)\.\.\(/\|$\)'; then
    echo "archive holds '..' paths: refusing it" >&2; exit 1
fi
mkdir -p "$work/unpack"
tar -xzf "$archive" -C "$work/unpack"
staged=$work/unpack/$name
cmp -s "$staged/manifest.tsv" "$manifest" ||
    { echo "the archive's manifest differs from corpus/manifest.tsv: refusing it" >&2; exit 1; }
verify_clips "$staged"

mkdir -p "$(dirname "$dest")"
if [[ -e $dest ]]; then
    # Replace only what this script put there.
    [[ -f $dest/manifest.tsv || -z $(ls -A "$dest") ]] ||
        { echo "$dest exists and is not a corpus directory: not touching it" >&2; exit 1; }
    rm -rf "$dest"
fi
mv "$staged" "$dest"
echo "$version $want" >"$dest/.corpus-version"
echo "corpus $version unpacked into $dest"
