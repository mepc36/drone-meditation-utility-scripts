#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="/Users/martinconnor/Desktop/x-drone-meditation-xvii/7-python/generate-weighted-playlist/audio-samples/strings-kontakt-samples/audio-strings-kontakt/???.wav"

# audio to remove
TRIM_START_MS=0
TRIM_END_MS=1000

dir="$(dirname "$INPUT_FILE")"
filename="$(basename "$INPUT_FILE")"
base="${filename%.wav}"

IFS='_' read -r part1 part2 part3 <<< "$base"

OUTFILE="$dir/${part1}_${part2}trimmed_${part3}.wav"

dur="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$INPUT_FILE")"

start_sec="$(awk -v ms="$TRIM_START_MS" 'BEGIN { print ms / 1000 }')"
end_sec="$(awk -v d="$dur" -v ms="$TRIM_END_MS" 'BEGIN { print d - (ms / 1000) }')"

awk -v s="$start_sec" -v e="$end_sec" 'BEGIN { exit !(e > s) }' || {
  echo "Error: trim amount is longer than or equal to file duration." >&2
  exit 1
}

ffmpeg -hide_banner -y \
  -i "$INPUT_FILE" \
  -af "atrim=start=${start_sec}:end=${end_sec},asetpts=PTS-STARTPTS" \
  -c:a pcm_s16le \
  "$OUTFILE"

echo "Original untouched: $INPUT_FILE"
echo "Trimmed file written: $OUTFILE"