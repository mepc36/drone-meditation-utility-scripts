#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="/Users/martinconnor/Desktop/x-drone-meditation-xvii/7-python/generate-weighted-playlist/audio-samples/strings-kontakt-samples/audio-strings-kontakt/???.wav"

# Desired final file length in seconds
# Examples:
#   2     = 2 seconds
#   0.5   = half a second
#   7     = 7 seconds
TARGET_DURATION_SEC=2

dir="$(dirname "$INPUT_FILE")"
filename="$(basename "$INPUT_FILE")"
base="${filename%.wav}"

IFS='_' read -r part1 part2 part3 <<< "$base"

OUTFILE="$dir/${part1}_${part2}set-length_${part3}.wav"

ffmpeg -hide_banner -y \
  -i "$INPUT_FILE" \
  -af "apad" \
  -t "$TARGET_DURATION_SEC" \
  -c:a pcm_s16le \
  "$OUTFILE"

echo "Original untouched: $INPUT_FILE"
echo "Set-length file written: $OUTFILE"
echo "Target duration: ${TARGET_DURATION_SEC}s"