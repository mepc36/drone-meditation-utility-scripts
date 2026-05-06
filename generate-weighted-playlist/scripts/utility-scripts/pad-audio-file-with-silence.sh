#!/usr/bin/env bash
set -euo pipefail

# INPUT_FILE="/Users/martinconnor/Desktop/x-drone-meditation-xvii/7-python/generate-weighted-playlist/audio-samples/strings-kontakt-samples/audio-strings-kontakt/kaleidoscope-quartet_-silent-start-_strings.wav"
INPUT_FILE="/Users/martinconnor/Desktop/x-drone-meditation-xvii/7-python/generate-weighted-playlist/audio-samples/strings-kontakt-samples/audio-strings-kontakt/kaleidoscope-quartet_-motivic-inversion-_strings.wav"

# silence padding
PAD_START_MS=0
PAD_END_MS=12880

dir="$(dirname "$INPUT_FILE")"
filename="$(basename "$INPUT_FILE")"
base="${filename%.wav}"

IFS='_' read -r part1 part2 part3 <<< "$base"

OUTFILE="$dir/${part1}_${part2}padded_${part3}.wav"

ffmpeg -hide_banner -y \
  -i "$INPUT_FILE" \
  -af "adelay=${PAD_START_MS}:all=1,apad=pad_dur=$(awk -v ms="$PAD_END_MS" 'BEGIN{print ms/1000}')" \
  -c:a pcm_s16le \
  "$OUTFILE"

echo "Original untouched: $INPUT_FILE"
echo "Padded file written: $OUTFILE"