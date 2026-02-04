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

# Extract config sections
pad_config = config["pad_samples_config"]
shared_config = config["shared_config"]

# Read from padded samples output directory
SOURCE_DIR = Path("./output/audio/padded-audio-samples")

# Parse samples_ratio (e.g., "8:4:1:4" means 8:4:1:4)
ratio_parts = [int(x) for x in shared_config["samples_ratio"].split(":")]
BREATHING_COPIES = ratio_parts[0]  # First number for Breathing
OTHER_CANONICAL_COPIES = ratio_parts[1]  # Second number for other 6 activities
LIVING_COPIES = ratio_parts[2]  # Third number for Living
SILENCE_COPIES = ratio_parts[3] if len(ratio_parts) > 3 else 0  # Fourth number for Silence

# Canonical file names
BREATHING_FILE = "Breathing.wav"
OTHER_CANONICAL_FILES = [f"{name}.wav" for name in pad_config["canonical_files"][1:]]  # Skip Breathing
LIVING_FILE = f"{pad_config['living_file']}.wav"
SILENCE_FILE = "Silence.wav"

# Output locations (relative to where you run the script)
OUTPUT_DIR = Path("./output")
AUDIO_OUTPUT_DIR = OUTPUT_DIR / "audio" / "final-sample-versions"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def ensure_source_files_exist() -> None:
    """Verify all canonical source files exist."""
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Padded samples directory not found: {SOURCE_DIR}\n"
            "Please run 1-pad-samples-with-silence.py first."
        )
    
    missing = []
    # Check Breathing
    if not (SOURCE_DIR / BREATHING_FILE).exists():
        missing.append(str(SOURCE_DIR / BREATHING_FILE))
    # Check other canonical files
    for name in OTHER_CANONICAL_FILES:
        if not (SOURCE_DIR / name).exists():
            missing.append(str(SOURCE_DIR / name))
    # Check Living
    if not (SOURCE_DIR / LIVING_FILE).exists():
        missing.append(str(SOURCE_DIR / LIVING_FILE))
    # Check Silence
    if not (SOURCE_DIR / SILENCE_FILE).exists():
        missing.append(str(SOURCE_DIR / SILENCE_FILE))
    
    if missing:
        raise FileNotFoundError("Missing expected source file(s):\n" + "\n".join(missing))


def reset_output_dir() -> None:
    """Clear and recreate the output directory for duplicates only (don't touch padded-audio-samples)."""
    # Only remove the AUDIO_OUTPUT_DIR (duplicates), not the entire audio directory
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
    
    print(f"Creating copies based on ratio {config['samples_ratio']}:")
    print(f"  - {BREATHING_FILE}: {BREATHING_COPIES} copies")
    print(f"  - Other activities (6 files): {OTHER_CANONICAL_COPIES} copies each")
    print(f"  - {LIVING_FILE}: {LIVING_COPIES} copy")
    print(f"  - {SILENCE_FILE}: {SILENCE_COPIES} copies")
    total_to_create = BREATHING_COPIES + (OTHER_CANONICAL_COPIES * len(OTHER_CANONICAL_FILES)) + LIVING_COPIES + SILENCE_COPIES
    print(f"Total files to create: {total_to_create}\n")
    
    reset_output_dir()
    
    total_created = 0
    all_files = []
    
    # Create copies of Breathing (first ratio number)
    src = SOURCE_DIR / BREATHING_FILE
    copies = make_copies(src, BREATHING_COPIES)
    all_files.extend(copies)
    total_created += len(copies)
    
    # Create copies of other canonical files (second ratio number)
    for filename in OTHER_CANONICAL_FILES:
        src = SOURCE_DIR / filename
        copies = make_copies(src, OTHER_CANONICAL_COPIES)
        all_files.extend(copies)
        total_created += len(copies)
    
    # Create copies of Living (third ratio number)
    src = SOURCE_DIR / LIVING_FILE
    if LIVING_COPIES == 1:
        living_path = copy_single_file(src)
        all_files.append(living_path)
        total_created += 1
    else:
        copies = make_copies(src, LIVING_COPIES)
        all_files.extend(copies)
        total_created += len(copies)
    
    # Create copies of Silence (fourth ratio number)
    src = SOURCE_DIR / SILENCE_FILE
    copies = make_copies(src, SILENCE_COPIES)
    all_files.extend(copies)
    total_created += len(copies)
    
    print(f"\nCreation complete! Created {total_created} files.")
    print(f"  - {BREATHING_FILE} × {BREATHING_COPIES} = {BREATHING_COPIES} files")
    print(f"  - Other activities × {OTHER_CANONICAL_COPIES} = {OTHER_CANONICAL_COPIES * len(OTHER_CANONICAL_FILES)} files")
    print(f"  - {LIVING_FILE} × {LIVING_COPIES} = {LIVING_COPIES} file")
    print(f"  - {SILENCE_FILE} × {SILENCE_COPIES} = {SILENCE_COPIES} files")
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
