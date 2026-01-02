#!/usr/bin/env python3
"""
generate-weighted-playlist.py

Generates an .m3u playlist from pre-existing weighted audio copies.
Assumes you've already run prepare-copies.py to create 100 copies of each file.

Two modes:
  1) Use weights defined in code (default ratio).
  2) Prompt for ratio Breathing : Each-Other : Living (e.g., 8:4:1).
"""

from dataclasses import dataclass
import json
from pathlib import Path
import random
import subprocess


# -------------------------------------------------------------------
# CONFIG: Load from input/config/config.json
# -------------------------------------------------------------------
CONFIG_PATH = Path("./input/config/config.json")
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Canonical file names (must match the stems used in prepare-copies.py)
BREATHING = config["canonical_files"][0]  # "Breathing"
LIVING = config["living_file"]  # "Living"
SILENCE = "Silence"  # Silence file
OTHERS = config["canonical_files"][1:]  # All except Breathing

# iTunes import location where files actually are
ITUNES_DIR = Path(config["itunes_dir"])

# Output locations (relative to where you run the script)
OUTPUT_DIR = Path("./output")
PLAYLIST_NAME = config["playlist_name"]
PLAYLIST_PATH = OUTPUT_DIR / "playlists" / f"{PLAYLIST_NAME}.m3u"

# Number of copies available per file (created by prepare-copies.py)
COPIES_AVAILABLE = config["copies_per_file"]

# Samples ratio
SAMPLES_RATIO = config["samples_ratio"]  # Breathing : Each-Other : Living


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def run_applescript(script: str) -> str:
    """Execute an AppleScript and return the output."""
    try:
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"


@dataclass(frozen=True)
class Ratio:
    breathing: int
    each_other: int
    living: int
    silence: int

    @staticmethod
    def parse(s: str) -> "Ratio":
        parts = s.strip().split(":")
        if len(parts) != 4:
            raise ValueError("Ratio must look like '8:4:1:4' (Breathing:Each-Other:Living:Silence).")
        b, o, l, si = (int(p.strip()) for p in parts)
        if b <= 0 or o <= 0 or l <= 0 or si < 0:
            raise ValueError("All ratio parts must be positive integers (silence can be 0).")
        return Ratio(breathing=b, each_other=o, living=l, silence=si)


def ensure_weighted_files_exist() -> None:
    """Verify the iTunes directory exists with imported files."""
    if not ITUNES_DIR.exists():
        raise FileNotFoundError(
            f"iTunes directory not found: {ITUNES_DIR}\n"
            "Please run prepare-copies.py first to import files into iTunes."
        )
    
    # Check if we have any files
    files = list(ITUNES_DIR.glob("*.wav"))
    if not files:
        raise FileNotFoundError(
            f"No audio files found in: {ITUNES_DIR}\n"
            "Please run prepare-copies.py first to import files into iTunes."
        )


def get_available_copies(stem: str) -> list[Path]:
    """
    Get all available copies for a given file stem (e.g., "Breathing").
    Returns list of Paths matching the pattern stem_NNN.wav
    Special cases: Living.wav and Silence.wav may have numbered copies.
    """
    # Check for both numbered copies and single file
    pattern = f"{stem}_*.wav"
    copies = sorted(ITUNES_DIR.glob(pattern))
    
    # If no numbered copies found, check for single file (Living or Silence might be single)
    if not copies:
        single_path = ITUNES_DIR / f"{stem}.wav"
        if single_path.exists():
            return [single_path]
        raise FileNotFoundError(f"No copies found for {stem} (pattern: {pattern} or {stem}.wav)")
    
    return copies


def select_copies(stem: str, count: int) -> list[Path]:
    """
    Select 'count' copies from the available copies, always using the lowest numbered files.
    If count > available, uses all available and wraps around from the beginning.
    """
    available = get_available_copies(stem)
    
    if count <= len(available):
        # Select the first 'count' files (lowest numbered)
        return available[:count]
    else:
        # Need more than available - use all and then repeat from the beginning
        selected = available.copy()
        remaining = count - len(available)
        # Repeat from the beginning to fill the remaining slots
        selected.extend(available[:remaining])
        return selected


def build_plan(ratio: Ratio) -> dict[str, int]:
    """
    Returns counts per canonical file stem based on ratio.
    Uses the ratio directly (no multiplier).
    """
    plan = {
        BREATHING: ratio.breathing,
        LIVING: ratio.living,
        SILENCE: ratio.silence,
    }
    for name in OTHERS:
        plan[name] = ratio.each_other
    return plan


def write_m3u(tracks: list[Path]) -> None:
    """Write playlist file pointing to selected tracks."""
    PLAYLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#EXTM3U", *[str(p) for p in tracks], ""]
    PLAYLIST_PATH.write_text("\n".join(lines), encoding="utf-8")


def reset_playlist_folder() -> None:
    """Remove and recreate the playlist folder."""
    playlist_folder = PLAYLIST_PATH.parent
    if playlist_folder.exists():
        import shutil
        shutil.rmtree(playlist_folder)
    playlist_folder.mkdir(parents=True, exist_ok=True)


def delete_playlist_from_itunes() -> bool:
    """Delete the playlist from iTunes/Music library if it exists."""
    script = f'''
tell application "Music"
    try
        set targetPlaylist to user playlist "{PLAYLIST_NAME}"
        delete targetPlaylist
        return "deleted"
    on error
        return "not_found"
    end try
end tell
'''
    result = run_applescript(script)
    return "deleted" in result


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    print("\nWeighted Playlist Builder\n")
    print("This script creates a playlist from pre-existing file copies.")
    print("Make sure you've run prepare-copies.py first!\n")

    # Clean playlist folder at start
    reset_playlist_folder()
    
    # Delete existing playlist from iTunes
    print("Checking for existing playlist in iTunes/Music...")
    if delete_playlist_from_itunes():
        print("  Removed existing playlist from iTunes/Music library")
    else:
        print("  No existing playlist found")
    print()

    ensure_weighted_files_exist()

    # Use ratio from config
    ratio = Ratio.parse(SAMPLES_RATIO)
    print(f"Using ratio from config: {SAMPLES_RATIO}\n")

    plan = build_plan(ratio)

    # Summary
    total = sum(plan.values())
    print("Plan (tracks to select):")
    print(f"  {BREATHING}: {plan[BREATHING]}")
    for name in OTHERS:
        print(f"  {name}: {plan[name]}")
    print(f"  {LIVING}: {plan[LIVING]}")
    print(f"  {SILENCE}: {plan[SILENCE]}")
    print(f"\nTotal tracks in playlist: {total}\n")

    # Select copies (using lowest numbered files)
    print("Building playlist (selecting lowest numbered files)...")
    selected_tracks: list[Path] = []

    # Select in order: Breathing, Others, Living, Silence (grouped blocks)
    selected_tracks.extend(select_copies(BREATHING, plan[BREATHING]))
    for name in OTHERS:
        selected_tracks.extend(select_copies(name, plan[name]))
    selected_tracks.extend(select_copies(LIVING, plan[LIVING]))
    if plan[SILENCE] > 0:
        selected_tracks.extend(select_copies(SILENCE, plan[SILENCE]))

    # Sort tracks: Breathing first, then everything else alphabetically
    selected_tracks.sort(key=lambda p: (0 if p.name.startswith('Breathing') else 1, p.name))
    
    write_m3u(selected_tracks)

    print("Done.")
    print(f"  Playlist written to: {PLAYLIST_PATH.resolve()}\n")
    
    # Auto-open playlist in Music
    print("Opening playlist in Music...")
    subprocess.run(['open', str(PLAYLIST_PATH.resolve())], check=False)
    
    print("\nNext:")
    print("  1) Playlist opened in Apple Music")
    print("  2) Tracks are grouped by type (Breathing, then others, then Living)")
    print("  3) Turn Shuffle ON if you want varied playback")
    print("  4) Turn Repeat (All) ON for infinite looping\n")


if __name__ == "__main__":
    main()
