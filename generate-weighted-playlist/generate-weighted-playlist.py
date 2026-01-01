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
from pathlib import Path
import random


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
# Canonical file names (must match the stems used in prepare-copies.py)
BREATHING = "Breathing"
LIVING = "Living"
OTHERS = [
    "Being",
    "Feeling",
    "Thinking",
    "Listening",
    "Faking",
    "Waiting",
]

# Output locations (relative to where you run the script)
OUTPUT_DIR = Path("./output")
WEIGHTED_DIR = OUTPUT_DIR / "weighted_audio"
PLAYLIST_PATH = OUTPUT_DIR / "playlists" / "Maestro — The Playlist.m3u"

# Number of copies available per file (created by prepare-copies.py)
COPIES_AVAILABLE = 100

# Default ratio
DEFAULT_RATIO = "8:4:1"  # Breathing : Each-Other : Living


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
    """Verify the weighted_audio directory exists with copies."""
    if not WEIGHTED_DIR.exists():
        raise FileNotFoundError(
            f"Weighted audio directory not found: {WEIGHTED_DIR}\n"
            "Please run prepare-copies.py first to create the file copies."
        )
    
    # Check if we have any files
    files = list(WEIGHTED_DIR.glob("*.wav"))
    if not files:
        raise FileNotFoundError(
            f"No audio files found in: {WEIGHTED_DIR}\n"
            "Please run prepare-copies.py first to create the file copies."
        )


def get_available_copies(stem: str) -> list[Path]:
    """
    Get all available copies for a given file stem (e.g., "Breathing").
    Returns list of Paths matching the pattern stem_NNN.wav
    """
    pattern = f"{stem}_*.wav"
    copies = sorted(WEIGHTED_DIR.glob(pattern))
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
    OUTPUT_DIR.mkdir(exist_ok=True)
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

    # Shuffle the entire playlist for varied playback
    random.shuffle(selected_tracks)

    write_m3u(selected_tracks)

    print("Done.")
    print(f"  Playlist written to: {PLAYLIST_PATH.resolve()}\n")
    print("Next:")
    print("  1) Double-click the .m3u to open in Apple Music")
    print("  2) The playlist will use your pre-imported copies")
    print("  3) Turn Shuffle ON for varied playback")
    print("  4) Turn Repeat (All) ON for infinite looping\n")


if __name__ == "__main__":
    main()
