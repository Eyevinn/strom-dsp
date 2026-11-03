#!/usr/bin/env bash
# Generate corpus/manifest.tsv and corpus/ATTRIBUTION.md from the curated
# corpus's provenance record (its PROVENANCE.md). Maintainers only: run it
# when the curated corpus changes, then run build-corpus-archive.sh, which
# fills in the SHA-256 column.
#
# Usage: scripts/make-corpus-manifest.sh CORPUS_DIR [OUT_DIR]
#   CORPUS_DIR  the curated corpus (holds PROVENANCE.md and the *_48k.wav clips)
#   OUT_DIR     where manifest.tsv and ATTRIBUTION.md go (default: corpus/)
#
# Only clips (the 48 kHz `*_48k.wav` files and the generated test signals)
# are listed, never the downloaded originals. A clip whose licence is not
# CC0, public domain, CC BY 3.0 or CC BY 4.0 stops the script.
#
# Existing SHA-256 values in OUT_DIR/manifest.tsv are kept for clips whose
# archive file name is unchanged.
set -euo pipefail

usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; }

[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
corpus=$1
here=$(cd "$(dirname "$0")/.." && pwd)
out=${2:-$here/corpus}
prov=$corpus/PROVENANCE.md
[[ -f $prov ]] || { echo "no PROVENANCE.md in $corpus" >&2; exit 1; }
mkdir -p "$out"

old_manifest=$out/manifest.tsv
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

# Channel count, frames and sample format of every clip, for the
# manifest (the format decides FLAC or float WAV in the archive).
info=$(mktemp)
trap 'rm -f "$tmp" "$info"' EXIT
while IFS= read -r -d '' f; do
    rel=${f#"$corpus"/}
    fmt=$(ffprobe -v error -select_streams a:0 \
        -show_entries stream=sample_fmt,channels,duration_ts -of csv=p=0 "$f")
    printf '%s\t%s\n' "$rel" "$fmt" >>"$info"
done < <(find "$corpus" -path "$corpus/_scripts" -prune -o \
    \( -name '*.wav' -not -path '*/original/*' -not -path '*/intermediate/*' -print0 \))

awk -F'\t' -v OFS='\t' -v info="$info" -v old="$old_manifest" '
function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t]+$/, "", s); return s }
function stripext(p) { sub(/\.[^.\/]*$/, "", p); return p }
function base(p) { sub(/.*\//, "", p); return p }
# "CC BY 4.0 (https://...)" -> name, url. Public-domain items under the
# MUSAN corpus carry both: the per-file status and the corpus licence.
function licname(l,   n) { n = l; sub(/ \(https?:[^)]*\)$/, "", n); return n }
function licurl(l,   u) { if (match(l, /\(https?:[^)]*\)$/)) { u = substr(l, RSTART + 1, RLENGTH - 2); return u } return "" }
function allowed(n) { return n ~ /^(CC0 1\.0|Public Domain|CC BY 3\.0|CC BY 4\.0)/ }
function srcurl(s,   u) { u = s; sub(/ .*/, "", u); return u }
BEGIN {
    while ((getline line < info) > 0) {
        split(line, a, "\t"); split(a[2], b, ",")
        fmt[a[1]] = b[1]; chans[a[1]] = b[2]; frames[a[1]] = b[3]
    }
    while ((getline line < old) > 0) {
        split(line, a, "\t"); if (a[1] != "id") oldsha[a[3]] = a[4]
    }
}
/^## Original files/ { sec = "orig"; next }
/^## Derived 48 kHz clips/ { sec = "clip"; next }
/^## / { sec = ""; next }
/^### `/ {
    p = $0; sub(/^### `/, "", p); sub(/`.*$/, "", p)
    if (sec == "orig") { origs[++no] = p } else if (sec == "clip") { clips[++nc] = p }
    cur = p; next
}
sec != "" && /^- [A-Za-z \/]+: / {
    k = $0; sub(/^- /, "", k); v = k; sub(/: .*/, "", k); sub(/^[^:]+: /, "", v)
    field[cur, k] = trim(v)
}
# The originals a clip was made from (keys into field[]).
function sources(c,   m, s, n, i, p, q, r, sp) {
    m = field[c, "Made from / processing"]; n = 0; delete srcs
    if (m ~ /^mix of /) {
        s = m; sub(/^mix of /, "", s)
        p = s; sub(/ \(.*/, "", p)
        q = s; sub(/^[^)]*\) and /, "", q); sub(/ \(.*/, "", q)
        n = sources_into(p, 0); n = sources_into(q, n); return n
    }
    return sources_into(c, 0)
}
function sources_into(c, n,   m, p, sp, i, cand) {
    m = field[c, "Made from / processing"]
    if (field[c, "Licence"] != "") { srcs[++n] = c; return n }
    if (match(m, /vctk(_|\/vctk_)?p2[0-9][0-9]/) || match(m, /\(p2[0-9][0-9]_/)) {
        sp = m; match(sp, /p2[0-9][0-9]/); sp = substr(sp, RSTART, RLENGTH)
        srcs[++n] = "1_speech_clean/vctk/original/" sp "_003_mic1.flac"; return n
    }
    if (match(m, /^from [^:]+:/)) {
        p = substr(m, 6, RLENGTH - 6)
        cand = stripext(p)
        for (i = 1; i <= no; i++) if (index(origs[i], cand) == 1) { srcs[++n] = origs[i]; return n }
    }
    print "cannot resolve the source of " c ": " m > "/dev/stderr"; exit 1
}
function extract(c, m, src,   s, d, a) {
    if (match(m, /excerpt start=[0-9.]+s dur=[0-9.]+s/)) {
        s = substr(m, RSTART + 8, RLENGTH - 8); return s
    }
    if (m ~ /^generated/) return "generated"
    if (m ~ /^mix of /) return "first " sprintf("%.0f", frames[c] / 48000) " s of each input"
    if (m ~ /concatenated/) return "10 utterances (003-012) concatenated with 0.3 s gaps, first 45 s"
    if (m ~ /^derived from/) return "whole clip it was derived from"
    if (match(src, /excerpt_[0-9]+-[0-9]+s/)) {
        a = substr(src, RSTART + 8); sub(/-.*/, "", a)
        return "start=" a "s dur=" sprintf("%.0f", frames[c] / 48000) "s"
    }
    if (m ~ /first [0-9]+ s/) return "start=0s dur=" sprintf("%.0f", frames[c] / 48000) "s"
    return "whole file"
}
END {
    print "id", "category", "file", "sha256", "channels", "seconds", "source_url", \
        "licence", "licence_url", "attribution", "extract", "processing"
    bad = 0
    for (i = 1; i <= nc; i++) {
        c = clips[i]
        if (!(c in fmt)) { print "clip listed but missing on disk: " c > "/dev/stderr"; exit 1 }
        cat = c; sub(/\/.*/, "", cat)
        id = base(c); sub(/_48k\.wav$/, "", id); sub(/\.wav$/, "", id)
        ext = (fmt[c] == "flt") ? "wav" : "flac"
        file = cat "/" id "." ext
        n = sources(c)
        urls = ""; lics = ""; lurls = ""; attr = ""
        for (j = 1; j <= n; j++) {
            s = srcs[j]
            if (field[s, "Licence"] == "") { print "no licence for " s > "/dev/stderr"; exit 1 }
            ln = licname(field[s, "Licence"]); lu = licurl(field[s, "Licence"])
            if (!allowed(ln)) { print "excluded (licence " ln "): " c > "/dev/stderr"; bad++; continue }
            u = srcurl(field[s, "Source"])
            if (u !~ /^https?:/) u = "(generated in the corpus)"
            urls = urls (j > 1 ? " ; " : "") u
            lics = lics (j > 1 ? " ; " : "") ln
            lurls = lurls (j > 1 ? " ; " : "") lu
            at = field[s, "Attribution"]
            if (at == "") at = field[s, "Source"]
            attr = attr (j > 1 ? " ; " : "") at
        }
        m = field[c, "Made from / processing"]
        proc = m; sub(/; STORAGE:.*/, "", proc)
        if (ext == "wav") proc = proc "; archive: 32-bit float WAV (sample peak above 0 dBFS, overs kept)"
        else proc = proc "; archive: 24-bit FLAC (lossless from the 24-bit PCM clip)"
        secs = sprintf("%.3f", frames[c] / 48000)
        print id, cat, file, (file in oldsha ? oldsha[file] : ""), chans[c], secs, urls, \
            lics, lurls, attr, extract(c, m, srcs[1]), proc
    }
    if (bad) exit 1
}
' "$prov" >"$tmp"

# Clip paths in the archive must be unique.
dups=$(cut -f3 "$tmp" | sort | uniq -d)
[[ -z $dups ]] || { echo "duplicate archive names: $dups" >&2; exit 1; }

{ head -n1 "$tmp"; tail -n +2 "$tmp" | sort -t$'\t' -k2,2 -k1,1; } >"$out/manifest.tsv"

# ATTRIBUTION.md: one entry per clip, grouped by category.
{
    echo "# strom-dsp test corpus: attribution"
    echo
    echo "The clips of the strom-dsp test corpus (release asset \`strom-dsp-corpus-v*.tar.gz\`)"
    echo "come from the sources below. Each is under CC0 1.0, in the public domain, or"
    echo "under CC BY 3.0 / 4.0; the licences require the attribution given here."
    echo "Clips were cut, resampled to 48 kHz and in some cases processed (codec,"
    echo "band-pass, gain and hard clipping, mixing); \`manifest.tsv\` records what was"
    echo "done to each. Generated test signals are CC0."
    echo
    echo "Licences: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/),"
    echo "[Public Domain Mark 1.0](https://creativecommons.org/publicdomain/mark/1.0/),"
    echo "[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/),"
    echo "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)."
    # awk, not `read`: a tab is whitespace to `read`, so empty fields
    # would shift.
    awk -F'\t' 'NR > 1 {
        if ($2 != prev) { printf "\n## %s\n\n", $2; prev = $2 }
        printf "- `%s`: %s (%s; source: %s)\n", $3, $10, $8, $7
    }' "$out/manifest.tsv"
} >"$out/ATTRIBUTION.md"

echo "wrote $out/manifest.tsv ($(($(wc -l <"$out/manifest.tsv") - 1)) clips) and $out/ATTRIBUTION.md"
