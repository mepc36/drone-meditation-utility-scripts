#!/usr/bin/env python3
"""
0-detect-downbeat-offset.py

Detect leading silence (seconds) before audible audio begins, for each song under ./input/*/audio/.
Then truncate each audio file by that amount and write it to:
  ./input/<song_slug>/audio-truncated/<output_filename>

Includes:
- Always overwrites truncated output (-y).
- Fix for libmp3lame "inadequate AVFrame plane padding" by forcing packed PCM + stereo + 44.1k before encoding.
- Always prints the FULL ffmpeg command lines for both:
  (1) silence detect
  (2) truncate/trim
- Ends by printing the full path to the truncated audio file and the detected silence amount.
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

# ─────────────────────────────────────────────────────────────
# Configuration / constants
# ─────────────────────────────────────────────────────────────

FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"

# Directory names
INPUT_DIR_NAME = "input"
OUTPUT_DIR_NAME = "output"
AUDIO_SUBDIR_NAME = "audio"
AUDIO_TRUNC_SUBDIR_NAME = "audio-truncated"
PROMPTS_DIR_NAME = "prompts"

# Filesystem ignores
MAC_DS_STORE_NAME = ".DS_Store"

# ffmpeg flags / args
FFMPEG_HIDE_BANNER_FLAG = "-hide_banner"
FFMPEG_NO_STATS_FLAG = "-nostats"
FFMPEG_OVERWRITE_FLAG = "-y"
FFMPEG_IGNORE_VIDEO_FLAG = "-vn"
FFMPEG_INPUT_FLAG = "-i"
FFMPEG_AUDIO_FILTER_FLAG = "-af"
FFMPEG_FORMAT_FLAG = "-f"
FFMPEG_NULL_FORMAT = "null"
FFMPEG_NULL_OUTPUT = "-"

# Silence detect defaults
DEFAULT_SILENCE_THRESHOLD_DB = -20.0
DEFAULT_MIN_SILENCE_SECONDS = 0.05

# Floating-point tolerance/constants
TIME_EPSILON_SECONDS = 1e-6
ZERO_SECONDS = 0.0

# Output formatting
SECONDS_DECIMALS = 6
TRIM_START_SECONDS_FORMAT = f"{{:.{SECONDS_DECIMALS}f}}"

# Exit codes
EXIT_OK = 0
EXIT_FATAL = 2

# Regex patterns
SILENCE_START_PATTERN = r"silence_start:\s*([0-9]*\.?[0-9]+)"
SILENCE_END_PATTERN = r"silence_end:\s*([0-9]*\.?[0-9]+)"
SILENCE_DETECT_FILTER_TEMPLATE = "silencedetect=noise={noise_db}dB:d={min_silence}"

# Output metadata paths
DOWNBEAT_OFFSET_DIR_NAME = "downbeat_offset"
OUTPUT_JSON_NAME = "leading_silence.json"
OUTPUT_TXT_NAME = "leading_silence.txt"

# Supported audio patterns
AUDIO_GLOB_PATTERNS = ["*.mp3", "*.wav", "*.flac", "*.m4a", "*.aac", "*.ogg", "*.wma"]

# Encoding choices
MP3_ENCODER = "libmp3lame"
MP3_QUALITY_ARG = "-q:a"
MP3_QUALITY_VALUE = "2"

AAC_ENCODER = "aac"
AAC_BITRATE_ARG = "-b:a"
AAC_BITRATE_VALUE = "256k"

FLAC_ENCODER = "flac"

VORBIS_ENCODER = "libvorbis"
VORBIS_QUALITY_ARG = "-q:a"
VORBIS_QUALITY_VALUE = "6"

WAV_ENCODER = "pcm_s16le"

# Encoder-safety audio conversion (fix for the error you hit)
SAFE_SAMPLE_FORMAT = "s16"
SAFE_CHANNEL_LAYOUT = "stereo"
SAFE_SAMPLE_RATE_HZ = 44100

# ─────────────────────────────────────────────────────────────


@dataclass
class SilenceEvent:
    kind: str  # "start" or "end"
    t: float


SILENCE_START_RE = re.compile(SILENCE_START_PATTERN)
SILENCE_END_RE = re.compile(SILENCE_END_PATTERN)


def _print_cmd(label: str, cmd: List[str]) -> None:
    print(f"\n[FFMPEG CMD — {label}]")
    print(" ".join(cmd))


def run_ffmpeg_silencedetect(path: Path, noise_db: float, min_silence: float) -> str:
    filter_str = SILENCE_DETECT_FILTER_TEMPLATE.format(noise_db=noise_db, min_silence=min_silence)

    cmd = [
        FFMPEG_PATH,
        FFMPEG_HIDE_BANNER_FLAG,
        FFMPEG_NO_STATS_FLAG,
        FFMPEG_IGNORE_VIDEO_FLAG,
        FFMPEG_INPUT_FLAG,
        str(path),
        FFMPEG_AUDIO_FILTER_FLAG,
        filter_str,
        FFMPEG_FORMAT_FLAG,
        FFMPEG_NULL_FORMAT,
        FFMPEG_NULL_OUTPUT,
    ]

    _print_cmd("silence detect", cmd)

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError(f"ffmpeg not found at {FFMPEG_PATH}")

    return proc.stderr


def parse_silence_events(ffmpeg_stderr: str) -> List[SilenceEvent]:
    events: List[SilenceEvent] = []
    for line in ffmpeg_stderr.splitlines():
        m = SILENCE_START_RE.search(line)
        if m:
            events.append(SilenceEvent(kind="start", t=float(m.group(1))))
            continue
        m = SILENCE_END_RE.search(line)
        if m:
            events.append(SilenceEvent(kind="end", t=float(m.group(1))))
            continue
    return events


def leading_silence_seconds(events: List[SilenceEvent], eps: float = TIME_EPSILON_SECONDS) -> Optional[float]:
    for i, ev in enumerate(events):
        if ev.kind == "start" and abs(ev.t - ZERO_SECONDS) <= eps:
            for ev2 in events[i + 1 :]:
                if ev2.kind == "end":
                    return max(ZERO_SECONDS, ev2.t)
            return None
    return ZERO_SECONDS


def find_single_audio_file(audio_dir: Path) -> Path:
    audio_files: List[Path] = []
    for pattern in AUDIO_GLOB_PATTERNS:
        audio_files.extend(audio_dir.glob(pattern))

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


def encoder_args_for_extension(ext: str) -> List[str]:
    ext_lower = ext.lower()
    if ext_lower == ".mp3":
        return ["-c:a", MP3_ENCODER, MP3_QUALITY_ARG, MP3_QUALITY_VALUE]
    if ext_lower in {".m4a", ".aac"}:
        return ["-c:a", AAC_ENCODER, AAC_BITRATE_ARG, AAC_BITRATE_VALUE]
    if ext_lower == ".flac":
        return ["-c:a", FLAC_ENCODER]
    if ext_lower == ".ogg":
        return ["-c:a", VORBIS_ENCODER, VORBIS_QUALITY_ARG, VORBIS_QUALITY_VALUE]
    if ext_lower == ".wav":
        return ["-c:a", WAV_ENCODER]
    if ext_lower == ".wma":
        return ["-c:a", WAV_ENCODER]
    return ["-c:a", WAV_ENCODER]


def output_path_for_trunc(input_audio: Path, trunc_dir: Path) -> Path:
    ext = input_audio.suffix.lower()
    known = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma"}
    if ext not in known:
        return trunc_dir / f"{input_audio.stem}.wav"
    return trunc_dir / input_audio.name


def build_safe_audio_filter() -> str:
    return (
        f"aformat=sample_fmts={SAFE_SAMPLE_FORMAT}:channel_layouts={SAFE_CHANNEL_LAYOUT},"
        f"aresample={SAFE_SAMPLE_RATE_HZ}"
    )


def truncate_audio_sample_accurate(input_file: Path, output_file: Path, offset_seconds: float) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    offset_str = TRIM_START_SECONDS_FORMAT.format(offset_seconds)
    encoder_args = encoder_args_for_extension(output_file.suffix)
    safe_filter = build_safe_audio_filter()

    cmd = [
        FFMPEG_PATH,
        FFMPEG_OVERWRITE_FLAG,
        FFMPEG_HIDE_BANNER_FLAG,
        FFMPEG_NO_STATS_FLAG,
        FFMPEG_IGNORE_VIDEO_FLAG,
        FFMPEG_INPUT_FLAG,
        str(input_file),
        "-ss",
        offset_str,
        FFMPEG_AUDIO_FILTER_FLAG,
        safe_filter,
        *encoder_args,
        str(output_file),
    ]

    _print_cmd("truncate/trim", cmd)

    subprocess.run(cmd, check=True)


def write_outputs(output_dir: Path, audio_file: Path, lead_seconds: float, noise_db: float, min_silence: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "ok": True,
        "audio_file": str(audio_file),
        "leading_silence_seconds": round(lead_seconds, SECONDS_DECIMALS),
        "silence_threshold_db": noise_db,
        "min_silence_seconds": min_silence,
    }

    with open(output_dir / OUTPUT_JSON_NAME, "w") as f:
        json.dump(payload, f, indent=2)

    with open(output_dir / OUTPUT_TXT_NAME, "w") as f:
        f.write(f"{lead_seconds:.{SECONDS_DECIMALS}f}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect leading silence then truncate audio under ./input/*/audio/")
    parser.add_argument("--noise", type=float, default=DEFAULT_SILENCE_THRESHOLD_DB)
    parser.add_argument("--min", dest="min_silence", type=float, default=DEFAULT_MIN_SILENCE_SECONDS)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    input_dir = script_dir / INPUT_DIR_NAME
    output_base_dir = script_dir / OUTPUT_DIR_NAME

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
    last_lead_seconds: Optional[float] = None

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

        stderr = run_ffmpeg_silencedetect(audio_file, args.noise, args.min_silence)
        events = parse_silence_events(stderr)
        lead = leading_silence_seconds(events)

        if lead is None:
            print("✗ Found silence_start: 0 but no silence_end. Try raising --noise.", file=sys.stderr)
            continue

        # Detection metadata
        song_output_dir = output_base_dir / song_name / DOWNBEAT_OFFSET_DIR_NAME
        write_outputs(song_output_dir, audio_file, lead, args.noise, args.min_silence)
        print(f"✓ Leading silence detected: {lead:.{SECONDS_DECIMALS}f} seconds")

        # Truncate (overwrite ON)
        trunc_dir = song_dir / AUDIO_TRUNC_SUBDIR_NAME
        trunc_out = output_path_for_trunc(audio_file, trunc_dir)

        print(f"Truncating (overwrite ON) → {trunc_out}")
        try:
            truncate_audio_sample_accurate(audio_file, trunc_out, lead)
        except subprocess.CalledProcessError as e:
            print(f"✗ Truncation failed (ffmpeg exit={e.returncode})", file=sys.stderr)
            continue

        print("✓ Truncated file written")

        last_trunc_path = trunc_out
        last_lead_seconds = lead

    # Final summary
    if last_trunc_path is not None and last_lead_seconds is not None:
        print("\n" + "=" * 80)
        print(f"TRUNCATED AUDIO: {str(last_trunc_path)}")
        print(f"LEADING SILENCE (seconds): {last_lead_seconds:.{SECONDS_DECIMALS}f}")
        print("=" * 80)

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
