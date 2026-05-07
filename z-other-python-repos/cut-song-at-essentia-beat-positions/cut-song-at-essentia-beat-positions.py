#!/usr/bin/env python3
"""
cut-song-at-essentia-beat-positions.py

Reads:
  ./input/input.mp3
  ./input/essentia.json

Writes beat-sliced audio files to:
  ./output/

Each output file is the audio between consecutive values in:
  rhythm.beats_position

Requires ffmpeg to be installed and available on PATH.

Usage:
  python3 cut-song-at-essentia-beat-positions.py
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


INPUT_DIR = Path("./input")
OUTPUT_DIR = Path("./output")
INPUT_MP3 = INPUT_DIR / "input.mp3"
INPUT_JSON = INPUT_DIR / "essentia.json"


def fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        fail("ffmpeg is not installed or not on PATH.")


def load_beat_positions(json_path: Path) -> list[float]:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        fail(f"Missing JSON file: {json_path}")
    except json.JSONDecodeError as e:
        fail(f"Invalid JSON in {json_path}: {e}")

    try:
        beats = data["rhythm"]["beats_position"]
    except KeyError:
        fail("Could not find rhythm.beats_position in essentia.json")

    if not isinstance(beats, list):
        fail("rhythm.beats_position must be a list")

    cleaned: list[float] = []
    for i, value in enumerate(beats):
        try:
            beat = float(value)
        except (TypeError, ValueError):
            fail(f"Invalid beat value at index {i}: {value!r}")

        if math.isnan(beat) or math.isinf(beat):
            fail(f"Non-finite beat value at index {i}: {value!r}")

        cleaned.append(beat)

    if not cleaned:
        fail("rhythm.beats_position is empty")

    # Ensure ascending order and uniqueness
    cleaned = sorted(cleaned)
    deduped: list[float] = []
    for beat in cleaned:
        if not deduped or abs(beat - deduped[-1]) > 1e-9:
            deduped.append(beat)

    if len(deduped) < 2:
        fail("Need at least 2 beat positions to create slices")

    return deduped


def run_ffmpeg_cut(input_file: Path, start: float, duration: float, output_file: Path) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_file),
        "-ss",
        f"{start:.6f}",
        "-t",
        f"{duration:.6f}",
        "-vn",
        "-acodec",
        "copy",
        str(output_file),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Fallback to re-encoding if stream copy fails at cut points
        fallback_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_file),
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-vn",
            "-q:a",
            "2",
            str(output_file),
        ]
        fallback_result = subprocess.run(fallback_cmd, capture_output=True, text=True)
        if fallback_result.returncode != 0:
            fail(
                f"ffmpeg failed for segment {output_file.name}\n"
                f"copy mode stderr:\n{result.stderr}\n"
                f"re-encode mode stderr:\n{fallback_result.stderr}"
            )


def main() -> None:
    require_ffmpeg()

    if not INPUT_MP3.exists():
        fail(f"Missing audio file: {INPUT_MP3}")

    beats = load_beat_positions(INPUT_JSON)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    written = 0

    for i in range(len(beats) - 1):
        start = beats[i]
        end = beats[i + 1]
        duration = end - start

        if duration <= 0:
            continue

        output_file = OUTPUT_DIR / f"beat_{i + 1:04d}_{start:.6f}_to_{end:.6f}.mp3"
        run_ffmpeg_cut(INPUT_MP3, start, duration, output_file)
        written += 1

    print(f"Wrote {written} beat slices to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()