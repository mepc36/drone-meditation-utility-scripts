#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="/Users/martinconnor/Desktop/x-drone-meditation-xvii/7-python/generate-weighted-playlist/audio-samples/strings-kontakt-samples/audio-strings-kontakt"
OUTDIR="/Users/martinconnor/Desktop/x-drone-meditation-xvii/7-python/generate-weighted-playlist/audio-samples/strings-kontakt-samples/audio-strings-kontakt-truncated-faded"

# Easy-to-change vars
TRUNCATE_MS=2000
FADE_MS=100
FADE_CURVE="tri"

# Add exact filenames to copy unchanged
IGNORE_FILES=(
  "kaleidoscope-quartet_-motivic-inversion-padded_strings"
)

should_ignore() {
  local filename="$1"

  for ignored in "${IGNORE_FILES[@]}"; do
    if [[ "$filename" == "$ignored" ]]; then
      return 0
    fi
  done

  return 1
}

rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

find "$INPUT_DIR" -maxdepth 1 -type f -iname "*.wav" -print0 | while IFS= read -r -d '' f; do
  filename="$(basename "$f")"

  if should_ignore "$filename"; then
    echo "Copying ignored file unchanged: $filename"
    cp -f "$f" "$OUTDIR/$filename"
    continue
  fi

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

  padded_dur="$(awk -v nd="$newdur" 'BEGIN { print nd + 0.05 }')"

  echo "Processing: $filename"
  echo "  original duration: $dur"
  echo "  new duration:      $newdur"
  echo "  fade starts at:    $fadestart"
  echo "  fade length:       $fade_seconds"

  ffmpeg -y -hide_banner -loglevel error \
    -i "$f" \
    -af "atrim=0:$newdur,asetpts=PTS-STARTPTS,afade=t=out:st=$fadestart:d=$fade_seconds:curve=$FADE_CURVE,apad=pad_dur=0.05,atrim=0:$padded_dur" \
    "$OUTDIR/$filename"
done

echo "Done. Output written to:"
echo "$OUTDIR"