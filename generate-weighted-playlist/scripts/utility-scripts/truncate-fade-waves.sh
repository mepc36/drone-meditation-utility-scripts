#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="/Users/martinconnor/Desktop/x-drone-meditation-xvii/7-python/generate-weighted-playlist/audio-strings-kontakt"
OUTDIR="$INPUT_DIR/1-output"

# Easy-to-change vars
TRUNCATE_MS=2500
FADE_MS=200

# Use "tri" first so you can clearly hear the fade.
# Other good options later: qsin, hsin
FADE_CURVE="tri"

rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

find "$INPUT_DIR" -maxdepth 1 -type f -iname "*.wav" -print0 | while IFS= read -r -d '' f; do
  filename="$(basename "$f")"

  dur="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$f")"

  newdur="$(awk -v d="$dur" -v t="$TRUNCATE_MS" 'BEGIN {
    nd = d - (t / 1000)
    if (nd < 0.05) nd = 0.05
    print nd
  }')"

  fade_seconds="$(awk -v fm="$FADE_MS" 'BEGIN { print fm / 1000 }')"

  fadestart="$(awk -v nd="$newdur" -v fs="$fade_seconds" 'BEGIN {
    start = nd - fs
    if (start < 0) start = 0
    print start
  }')"

  echo "Processing: $filename"
  echo "  original duration: $dur"
  echo "  new duration:      $newdur"
  echo "  fade starts at:    $fadestart"
  echo "  fade length:       $fade_seconds"

  ffmpeg -y -hide_banner -loglevel error \
    -i "$f" \
    -af "atrim=0:$newdur,asetpts=PTS-STARTPTS,afade=t=out:st=$fadestart:d=$fade_seconds:curve=$FADE_CURVE" \
    "$OUTDIR/$filename"
done

echo "Done. Output written to:"
echo "$OUTDIR"