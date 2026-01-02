#!/usr/bin/env python3
"""
prepare-duplicate-audio-files.py

Prep step: Creates 100 copies of each canonical audio file and imports them into iTunes/Music.

IMPORTANT: Living.wav is only copied once (no duplicates).
Living represents the end/completion state and should be rare in the playlist.
"""

import json
import shutil
import subprocess
from pathlib import Path


# -------------------------------------------------------------------
# CONFIG: Load from input/config/config.json
# -------------------------------------------------------------------
CONFIG_PATH = Path("./input/config/config.json")

# Ensure input directory structure exists
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

SOURCE_DIR = Path(config["source_dir"])
COPIES_PER_FILE = config["copies_per_file"]

# Canonical file names (must match exactly)
# Note: Living.wav is handled separately - only 1 copy, not 100
CANONICAL_FILES = [f"{name}.wav" for name in config["canonical_files"]]

# Living.wav - the end state, should only exist once
LIVING_FILE = f"{config['living_file']}.wav"

# Output locations (relative to where you run the script)
OUTPUT_DIR = Path("./output")
AUDIO_OUTPUT_DIR = OUTPUT_DIR / "audio"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def ensure_source_files_exist() -> None:
    """Verify all canonical source files exist."""
    missing = []
    for name in CANONICAL_FILES:
        if not (SOURCE_DIR / name).exists():
            missing.append(str(SOURCE_DIR / name))
    # Also check for Living.wav
    if not (SOURCE_DIR / LIVING_FILE).exists():
        missing.append(str(SOURCE_DIR / LIVING_FILE))
    if missing:
        raise FileNotFoundError("Missing expected source file(s):\n" + "\n".join(missing))


def reset_output_dir() -> None:
    """Clear and recreate the output directory."""
    # Remove entire output/audio directory if it exists
    audio_dir = OUTPUT_DIR / "audio"
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    
    # Recreate the weighted_audio directory
    if AUDIO_OUTPUT_DIR.exists():
        shutil.rmtree(AUDIO_OUTPUT_DIR)
    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_copies(src: Path, count: int) -> list[Path]:
    """
    Create 'count' copies of src in AUDIO_OUTPUT_DIR with unique names.
    Returns list of created Paths in deterministic order.
    """
    created = []
    stem = src.stem  # "Breathing" from "Breathing.wav"
    suffix = src.suffix  # ".wav"

    print(f"  Creating {count} copies of {src.name}...", end=" ", flush=True)
    
    for i in range(1, count + 1):
        # Example: Breathing_001.wav, Breathing_002.wav, etc.
        dst = AUDIO_OUTPUT_DIR / f"{stem}_{i:03d}{suffix}"
        shutil.copy2(src, dst)
        created.append(dst)

    print("✓")
    return created


def copy_single_file(src: Path) -> Path:
    """Copy a single file to the output directory without creating duplicates."""
    print(f"  Copying {src.name} (single copy only)...", end=" ", flush=True)
    dst = AUDIO_OUTPUT_DIR / src.name
    shutil.copy2(src, dst)
    print("✓")
    return dst


# -------------------------------------------------------------------
# iTunes Import Functions
# -------------------------------------------------------------------
def prompt_int(prompt: str, default: int, min_val: int = 1) -> int:
    """Prompt user for an integer value with a default."""
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        n = int(raw)
        if n < min_val:
            raise ValueError(f"Must be at least {min_val}.")
        return n
    except ValueError as e:
        print(f"Invalid input: {e}")
        return default


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


def import_folder_to_music(folder: Path) -> str:
    """Import entire folder to iTunes/Music in one operation."""
    script = f'''
tell application "Music"
    add (POSIX file "{folder.resolve()}/")
end tell
'''
    return run_applescript(script)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    print("\nPrepare Weighted Copies for iTunes Import\n")
    print("Source folder:")
    print(f"  {SOURCE_DIR}\n")
    
    ensure_source_files_exist()
    
    print(f"Creating {COPIES_PER_FILE} copies of each file ({len(CANONICAL_FILES)} files total)...")
    print(f"Plus 1 copy of {LIVING_FILE} (no duplicates - represents end state)")
    print(f"Total files to create: {COPIES_PER_FILE * len(CANONICAL_FILES) + 1}\n")
    
    reset_output_dir()
    
    total_created = 0
    all_files = []
    
    # Create 100 copies of each standard file
    for filename in CANONICAL_FILES:
        src = SOURCE_DIR / filename
        copies = make_copies(src, COPIES_PER_FILE)
        all_files.extend(copies)
        total_created += len(copies)
    
    # Copy Living.wav exactly once (no duplicates)
    # Living represents the end/completion state and should be rare
    print(f"\n  Note: {LIVING_FILE} is copied only once.")
    print(f"  This file represents the end state and should only appear rarely in playlists.\n")
    living_path = copy_single_file(SOURCE_DIR / LIVING_FILE)
    all_files.append(living_path)
    total_created += 1
    
    print(f"\nCreation complete! Created {total_created} files.")
    print(f"  - {len(CANONICAL_FILES)} files × {COPIES_PER_FILE} copies = {COPIES_PER_FILE * len(CANONICAL_FILES)} files")
    print(f"  - {LIVING_FILE} × 1 copy = 1 file")
    print(f"Output directory: {AUDIO_OUTPUT_DIR.resolve()}\n")
    
    # Import to iTunes
    print("="*60)
    print("Importing entire folder to iTunes/Music...")
    print("This may take a moment...\n")
    
    result = import_folder_to_music(AUDIO_OUTPUT_DIR)
    
    print(f"\nImport complete!")
    print(f"  Imported folder: {AUDIO_OUTPUT_DIR.resolve()}")
    print(f"  Total files: {total_created}")
    print(f"\nNext steps:")
    print(f"  1) Files are now in your iTunes/Music library")
    print(f"  2) Run generate-weighted-playlist.py to create playlists\n")


if __name__ == "__main__":
    main()
