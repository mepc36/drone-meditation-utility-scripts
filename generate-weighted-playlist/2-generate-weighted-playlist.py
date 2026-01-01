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
# CONFIG: Load from input/config.json
# -------------------------------------------------------------------
CONFIG_PATH = Path("./input/config.json")
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Canonical file names (must match the stems used in prepare-copies.py)
BREATHING = config["canonical_files"][0]  # "Breathing"
LIVING = config["living_file"]  # "Living"
OTHERS = config["canonical_files"][1:]  # All except Breathing

# iTunes import location where files actually are
ITUNES_DIR = Path(config["itunes_dir"])

# Output locations (relative to where you run the script)
OUTPUT_DIR = Path("./output")
PLAYLIST_PATH = OUTPUT_DIR / "playlists" / "Maestro — The Playlist.m3u"

# Number of copies available per file (created by prepare-copies.py)
COPIES_AVAILABLE = config["copies_per_file"]

# Default ratio
DEFAULT_RATIO = config["default_ratio"]  # Breathing : Each-Other : Living


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
@dataclass(frozen=True)
class Ratio:
    breathing: int
    each_other: int
    living: int

    @staticmethod
    def parse(s: str) -> "Ratio":
        parts = s.strip().split(":")
        if len(parts) != 3:
            raise ValueError("Ratio must look like '8:4:1' (Breathing:Each-Other:Living).")
        b, o, l = (int(p.strip()) for p in parts)
        if b <= 0 or o <= 0 or l <= 0:
            raise ValueError("All ratio parts must be positive integers.")
        return Ratio(breathing=b, each_other=o, living=l)


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
    Special case: Living.wav exists as a single file without numbering.
    """
    # Living is stored as a single file without numbering
    if stem == LIVING:
        living_path = ITUNES_DIR / f"{stem}.wav"
        if not living_path.exists():
            raise FileNotFoundError(f"Living file not found: {living_path}")
        return [living_path]
    
    # All other files have numbered copies
    pattern = f"{stem}_*.wav"
    copies = sorted(ITUNES_DIR.glob(pattern))
    if not copies:
        raise FileNotFoundError(f"No copies found for {stem} (pattern: {pattern})")
    return copies


def select_copies(stem: str, count: int) -> list[Path]:
    """
    Randomly select 'count' copies from the available copies of a file.
    If count > available, uses all available and wraps around with replacement.
    """
    available = get_available_copies(stem)
    
    if count <= len(available):
        # Select without replacement
        return random.sample(available, count)
    else:
        # Need more than available - use all and then sample with replacement
        selected = available.copy()
        remaining = count - len(available)
        selected.extend(random.choices(available, k=remaining))
        return selected


def build_plan(ratio: Ratio) -> dict[str, int]:
    """
    Returns counts per canonical file stem based on ratio.
    Uses the ratio directly (no multiplier).
    """
    plan = {
        BREATHING: ratio.breathing,
        LIVING: ratio.living,
    }
    for name in OTHERS:
        plan[name] = ratio.each_other
    return plan


def write_m3u(tracks: list[Path]) -> None:
    """Write playlist file pointing to selected tracks."""
    PLAYLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#EXTM3U", *[str(p) for p in tracks], ""]
    PLAYLIST_PATH.write_text("\n".join(lines), encoding="utf-8")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    print("\nWeighted Playlist Builder\n")
    print("This script creates a playlist from pre-existing file copies.")
    print("Make sure you've run prepare-copies.py first!\n")

    ensure_weighted_files_exist()

    print("Choose mode:")
    print("  1) Use default weights from code")
    print("  2) Enter weights interactively (ratio like 8:4:1)")
    mode = input("Enter 1 or 2 [1]: ").strip() or "1"
    if mode not in ("1", "2"):
        raise ValueError("Mode must be 1 or 2.")

    if mode == "1":
        ratio = Ratio.parse(DEFAULT_RATIO)
        print(f"\nUsing default ratio: {DEFAULT_RATIO}\n")
    else:
        ratio_str = input("\nEnter ratio Breathing:Each-Other:Living (e.g. 8:4:1): ").strip()
        ratio = Ratio.parse(ratio_str)
        print(f"\nUsing ratio: {ratio.breathing}:{ratio.each_other}:{ratio.living}\n")

    plan = build_plan(ratio)

    # Summary
    total = sum(plan.values())
    print("Plan (tracks to select):")
    print(f"  {BREATHING}: {plan[BREATHING]}")
    for name in OTHERS:
        print(f"  {name}: {plan[name]}")
    print(f"  {LIVING}: {plan[LIVING]}")
    print(f"\nTotal tracks in playlist: {total}\n")

    # Select copies randomly according to the plan
    print("Building playlist with random selection...")
    selected_tracks: list[Path] = []

    # Select in order: Breathing, Others, Living (grouped blocks)
    selected_tracks.extend(select_copies(BREATHING, plan[BREATHING]))
    for name in OTHERS:
        selected_tracks.extend(select_copies(name, plan[name]))
    selected_tracks.extend(select_copies(LIVING, plan[LIVING]))

    # Keep tracks grouped by type (no shuffle)
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
