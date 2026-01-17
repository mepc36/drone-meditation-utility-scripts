#!/usr/bin/env python3
"""
0-detect-downbeat-offset.py  (Essentia/Docker version)

Goal:
- Handle tracks that start with *talking* (not silence) and find when the *beat* starts.
- For each song under ./input/*/audio/ (exactly 1 audio file), run Essentia (via Docker),
  infer a "beat start" time, then truncate the audio from that time onward.
- Writes truncated audio to:
    ./input/<song_slug>/audio-truncated/<same_filename>
- Writes analysis JSON to:
    ./output/<song_slug>/essentia/essentia-analysis-output.json
- ALWAYS overwrites outputs.
- ALWAYS prints the FULL docker + ffmpeg commands.

Assumed Essentia invocation (as you specified):
docker run -ti --rm -v `pwd`:/essentia mtgupf/essentia essentia_streaming_extractor_music input.mp3 essentia-analysis-output.json
"""

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# Configuration / constants (no magic numbers)
# ─────────────────────────────────────────────────────────────

# Tools
DOCKER_BIN = "docker"
FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"

# Docker / Essentia
ESSENTIA_IMAGE = "mtgupf/essentia"
ESSENTIA_EXTRACTOR_BIN = "essentia_streaming_extractor_music"
ESSENTIA_CONTAINER_MOUNT_PATH = "/essentia"

# Directory names
INPUT_DIR_NAME = "input"
OUTPUT_DIR_NAME = "output"
AUDIO_SUBDIR_NAME = "audio"
AUDIO_TRUNC_SUBDIR_NAME = "audio-truncated"
PROMPTS_DIR_NAME = "prompts"
MAC_DS_STORE_NAME = ".DS_Store"

# Audio search patterns (must find exactly one)
AUDIO_GLOB_PATTERNS = ["*.mp3", "*.wav", "*.flac", "*.m4a", "*.aac", "*.ogg", "*.wma"]

# ffmpeg flags
FFMPEG_HIDE_BANNER_FLAG = "-hide_banner"
FFMPEG_NO_STATS_FLAG = "-nostats"
FFMPEG_OVERWRITE_FLAG = "-y"
FFMPEG_IGNORE_VIDEO_FLAG = "-vn"
FFMPEG_INPUT_FLAG = "-i"

# Output formatting
SECONDS_DECIMALS = 6
TRIM_START_SECONDS_FORMAT = f"{{:.{SECONDS_DECIMALS}f}}"

# Beat-start detection logic (from Essentia beat positions)
# We pick the earliest time where the beat becomes "stable" for N consecutive intervals.
DEFAULT_STABLE_BEAT_WINDOW_COUNT = 8            # number of consecutive beat intervals to evaluate
DEFAULT_STABILITY_CV_THRESHOLD = 0.06           # coefficient of variation threshold (lower = stricter)
DEFAULT_MIN_BEAT_START_SECONDS = 0.0            # allow beat at start; you can set e.g. 2.0 to ignore immediate beats
DEFAULT_MAX_REASONABLE_BPM = 220.0
DEFAULT_MIN_REASONABLE_BPM = 55.0

# Fallback: if stability logic fails, use the first beat >= this time
DEFAULT_FALLBACK_MIN_SECONDS = 0.0

# Exit codes
EXIT_OK = 0
EXIT_FATAL = 2

# Output subdir names
ESSENTIA_OUTPUT_DIR_NAME = "essentia"
ESSENTIA_OUTPUT_JSON_NAME = "essentia-analysis-output.json"

BEAT_OFFSET_DIR_NAME = "downbeat_offset"
BEAT_OFFSET_JSON_NAME = "beat_start.json"
BEAT_OFFSET_TXT_NAME = "beat_start.txt"

# ─────────────────────────────────────────────────────────────


@dataclass
class BeatStartResult:
    beat_start_seconds: float
    method: str
    beats_count: int


def _print_cmd(label: str, cmd: List[str]) -> None:
    print(f"\n[CMD — {label}]")
    print(" ".join(cmd))


def find_single_audio_file(audio_dir: Path) -> Path:
    audio_files: List[Path] = []
    for pat in AUDIO_GLOB_PATTERNS:
        audio_files.extend(audio_dir.glob(pat))

    if not audio_files:
        supported = ", ".join(AUDIO_GLOB_PATTERNS)
        raise FileNotFoundError(f"No audio files found in {audio_dir}\nSupported patterns: {supported}")

    if len(audio_files) > 1:
        files_list = "\n    ".join([f.name for f in audio_files])
        raise ValueError(
            f"Found {len(audio_files)} audio files in {audio_dir}, but only 1 is allowed.\n"
            f"  Found files:\n    {files_list}"
        )

    return audio_files[0]


def run_essentia_docker(cwd: Path, input_audio: Path, output_json: Path) -> None:
    """
    Runs Essentia in Docker. Assumes current repo root is mounted at /essentia.
    Uses paths relative to cwd to work inside container.
    ALWAYS overwrites output JSON (we delete it first for clarity).
    """
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if output_json.exists():
        output_json.unlink()

    # Ensure the input path is relative to cwd (since we mount cwd -> /essentia)
    input_rel = input_audio.relative_to(cwd)
    output_rel = output_json.relative_to(cwd)

    cmd = [
        DOCKER_BIN,
        "run",
        "-ti",
        "--rm",
        "-v",
        f"{str(cwd)}:{ESSENTIA_CONTAINER_MOUNT_PATH}",
        ESSENTIA_IMAGE,
        ESSENTIA_EXTRACTOR_BIN,
        str(input_rel),
        str(output_rel),
    ]

    _print_cmd("docker essentia extractor", cmd)

    subprocess.run(cmd, check=True, cwd=str(cwd))


def load_essentia_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _try_get_beats_position(essentia_json: dict) -> Optional[List[float]]:
    """
    Essentia extractor JSON often contains:
      essentia_json["rhythm"]["beats_position"] = [ ... seconds ... ]
    We look for that, but also try a couple common variants.
    """
    if isinstance(essentia_json, dict):
        rhythm = essentia_json.get("rhythm")
        if isinstance(rhythm, dict):
            beats = rhythm.get("beats_position")
            if isinstance(beats, list) and beats and all(isinstance(x, (int, float)) for x in beats):
                return [float(x) for x in beats]

        # fallback attempts (some configs output top-level keys differently)
        beats = essentia_json.get("beats_position")
        if isinstance(beats, list) and beats and all(isinstance(x, (int, float)) for x in beats):
            return [float(x) for x in beats]

    return None


def _coefficient_of_variation(values: List[float]) -> float:
    if not values:
        return float("inf")
    mean = sum(values) / len(values)
    if mean <= 0:
        return float("inf")
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    return std / mean


def infer_beat_start_from_beats(
    beats_position: List[float],
    stable_window_count: int,
    stability_cv_threshold: float,
    min_reasonable_bpm: float,
    max_reasonable_bpm: float,
    min_beat_start_seconds: float,
    fallback_min_seconds: float,
) -> BeatStartResult:
    """
    Heuristic:
    - compute inter-beat intervals (IBIs)
    - scan for earliest index i where the next N IBIs are stable (low CV)
      and within reasonable BPM bounds.
    - choose beat_start = beats_position[i]
    """
    if len(beats_position) < (stable_window_count + 1):
        # Not enough beats; fallback to first beat >= fallback_min_seconds
        for b in beats_position:
            if b >= fallback_min_seconds:
                return BeatStartResult(b, "fallback_first_beat", len(beats_position))
        return BeatStartResult(beats_position[0], "fallback_first_beat", len(beats_position))

    ibis = [beats_position[i + 1] - beats_position[i] for i in range(len(beats_position) - 1)]

    # sanity filter: ignore non-positive IBIs
    ibis = [x for x in ibis if x > 0]
    if len(ibis) < stable_window_count:
        for b in beats_position:
            if b >= fallback_min_seconds:
                return BeatStartResult(b, "fallback_first_beat", len(beats_position))
        return BeatStartResult(beats_position[0], "fallback_first_beat", len(beats_position))

    for i in range(0, len(beats_position) - (stable_window_count + 1)):
        beat_time = beats_position[i]
        if beat_time < min_beat_start_seconds:
            continue

        window_ibis = [
            beats_position[i + j + 1] - beats_position[i + j]
            for j in range(stable_window_count)
        ]
        if any(x <= 0 for x in window_ibis):
            continue

        cv = _coefficient_of_variation(window_ibis)
        mean_ibi = sum(window_ibis) / len(window_ibis)
        bpm = 60.0 / mean_ibi if mean_ibi > 0 else 0.0

        if cv <= stability_cv_threshold and (min_reasonable_bpm <= bpm <= max_reasonable_bpm):
            return BeatStartResult(beat_time, "stable_beats_window", len(beats_position))

    # Fallback: first beat >= fallback_min_seconds
    for b in beats_position:
        if b >= fallback_min_seconds:
            return BeatStartResult(b, "fallback_first_beat", len(beats_position))

    return BeatStartResult(beats_position[0], "fallback_first_beat", len(beats_position))


def truncate_audio_with_ffmpeg(input_file: Path, output_file: Path, offset_seconds: float) -> None:
    """
    Truncate from offset_seconds onward. Always overwrites.
    Uses stream selection via -vn to ignore embedded artwork.
    Uses re-encode via 'aformat/aresample' to keep ffmpeg happy across formats.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    offset_str = TRIM_START_SECONDS_FORMAT.format(offset_seconds)

    # Keep it simple: output format follows extension; ffmpeg chooses encoder if possible.
    # If you want strict encoders per extension again, we can add that back.
    safe_filter = "aformat=sample_fmts=s16:channel_layouts=stereo,aresample=44100"

    cmd = [
        FFMPEG_PATH,
        FFMPEG_OVERWRITE_FLAG,
        FFMPEG_HIDE_BANNER_FLAG,
        FFMPEG_NO_STATS_FLAG,
        FFMPEG_IGNORE_VIDEO_FLAG,
        FFMPEG_INPUT_FLAG,
        str(input_file),
        "-ss",
        offset_str,  # after -i = accurate
        "-af",
        safe_filter,
        str(output_file),
    ]

    _print_cmd("ffmpeg truncate", cmd)

    subprocess.run(cmd, check=True)


def write_beat_outputs(output_dir: Path, audio_file: Path, result: BeatStartResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "ok": True,
        "audio_file": str(audio_file),
        "beat_start_seconds": round(result.beat_start_seconds, SECONDS_DECIMALS),
        "method": result.method,
        "beats_count": result.beats_count,
    }

    with open(output_dir / BEAT_OFFSET_JSON_NAME, "w") as f:
        json.dump(payload, f, indent=2)

    with open(output_dir / BEAT_OFFSET_TXT_NAME, "w") as f:
        f.write(f"{result.beat_start_seconds:.{SECONDS_DECIMALS}f}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Use Essentia (Docker) to find beat start, then truncate audio.")
    parser.add_argument("--stable-window", type=int, default=DEFAULT_STABLE_BEAT_WINDOW_COUNT)
    parser.add_argument("--cv", type=float, default=DEFAULT_STABILITY_CV_THRESHOLD)
    parser.add_argument("--min-bpm", type=float, default=DEFAULT_MIN_REASONABLE_BPM)
    parser.add_argument("--max-bpm", type=float, default=DEFAULT_MAX_REASONABLE_BPM)
    parser.add_argument("--min-beat-start", type=float, default=DEFAULT_MIN_BEAT_START_SECONDS)
    parser.add_argument("--fallback-min", type=float, default=DEFAULT_FALLBACK_MIN_SECONDS)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    cwd = script_dir.resolve()

    input_dir = cwd / INPUT_DIR_NAME
    output_base_dir = cwd / OUTPUT_DIR_NAME
    output_base_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Input directory not found at {input_dir}", file=sys.stderr)
        return EXIT_FATAL

    song_dirs = [
        d for d in input_dir.iterdir()
        if d.is_dir() and d.name not in {MAC_DS_STORE_NAME, PROMPTS_DIR_NAME}
    ]

    if not song_dirs:
        print(f"No song directories found in ./{INPUT_DIR_NAME}")
        return EXIT_OK

    print(f"\nFound {len(song_dirs)} song(s) to process")

    last_trunc_path: Optional[Path] = None
    last_beat_start: Optional[float] = None

    for song_dir in song_dirs:
        song_name = song_dir.name

        print(f"\n{'='*80}")
        print(f"Processing: {song_name}")
        print(f"{'='*80}")

        audio_dir = song_dir / AUDIO_SUBDIR_NAME
        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found at {audio_dir}")

        audio_file = find_single_audio_file(audio_dir)
        print(f"✓ Found audio file: {audio_file.name}")

        # 1) Essentia analysis
        essentia_out_dir = output_base_dir / song_name / ESSENTIA_OUTPUT_DIR_NAME
        essentia_json_path = essentia_out_dir / ESSENTIA_OUTPUT_JSON_NAME

        try:
            run_essentia_docker(cwd=cwd, input_audio=audio_file, output_json=essentia_json_path)
        except subprocess.CalledProcessError as e:
            print(f"✗ Essentia docker run failed (exit={e.returncode})", file=sys.stderr)
            continue

        if not essentia_json_path.exists():
            print(f"✗ Expected Essentia output JSON not found at {essentia_json_path}", file=sys.stderr)
            continue

        essentia_json = load_essentia_json(essentia_json_path)
        beats_position = _try_get_beats_position(essentia_json)

        if not beats_position:
            print("✗ Could not find rhythm.beats_position in Essentia output JSON.", file=sys.stderr)
            print(f"  JSON path: {essentia_json_path}", file=sys.stderr)
            continue

        # 2) Infer beat start
        result = infer_beat_start_from_beats(
            beats_position=beats_position,
            stable_window_count=args.stable_window,
            stability_cv_threshold=args.cv,
            min_reasonable_bpm=args.min_bpm,
            max_reasonable_bpm=args.max_bpm,
            min_beat_start_seconds=args.min_beat_start,
            fallback_min_seconds=args.fallback_min,
        )

        print(f"✓ Beat start detected: {result.beat_start_seconds:.{SECONDS_DECIMALS}f} seconds (method={result.method})")

        # 3) Write beat-start metadata
        beat_out_dir = output_base_dir / song_name / BEAT_OFFSET_DIR_NAME
        write_beat_outputs(beat_out_dir, audio_file, result)

        # 4) Truncate audio to ./input/<song>/audio-truncated/<same filename>
        trunc_dir = song_dir / AUDIO_TRUNC_SUBDIR_NAME
        trunc_out = trunc_dir / audio_file.name

        print(f"Truncating (overwrite ON) → {trunc_out}")
        try:
            truncate_audio_with_ffmpeg(audio_file, trunc_out, result.beat_start_seconds)
        except subprocess.CalledProcessError as e:
            print(f"✗ Truncation failed (ffmpeg exit={e.returncode})", file=sys.stderr)
            continue

        print("✓ Truncated file written")

        last_trunc_path = trunc_out
        last_beat_start = result.beat_start_seconds

    # Final summary
    if last_trunc_path is not None and last_beat_start is not None:
        print("\n" + "=" * 80)
        print(f"TRUNCATED AUDIO: {str(last_trunc_path)}")
        print(f"BEAT START (seconds): {last_beat_start:.{SECONDS_DECIMALS}f}")
        print("=" * 80)

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
