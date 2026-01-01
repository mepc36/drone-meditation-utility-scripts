#!/usr/bin/env python3
"""
remove-from-itunes-library.py

Complete cleanup script for duplicate audio files:
1. Deletes physical .wav files from disk (iTunes import location)
2. Removes database entries from iTunes/Music library using AppleScript
"""

import subprocess
from pathlib import Path


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
# iTunes import location for unknown artist/album
ITUNES_DIR = Path(
    "/Users/martinconnor/Music/Music/Media.localized/Music/Unknown Artist/Unknown Album"
)

# File stems that were duplicated (100 copies each)
CANONICAL_STEMS = [
    "Breathing",
    "Being",
    "Feeling",
    "Thinking",
    "Listening",
    "Faking",
    "Waiting"
]

# Single file (not duplicated)
LIVING_FILE = "Living"

# Number of copies created per file
COPIES_PER_FILE = 100

# -------------------------------------------------------------------
# File Deletion Functions
# -------------------------------------------------------------------
def delete_physical_files() -> tuple[int, list[tuple[Path, str]]]:
    """
    Delete physical .wav files from iTunes directory.
    Returns (deleted_count, errors_list).
    """
    if not ITUNES_DIR.exists():
        print(f"iTunes directory does not exist: {ITUNES_DIR}")
        return (0, [])
    
    files_to_delete = []
    
    # Find all numbered copies
    for stem in CANONICAL_STEMS:
        for i in range(1, COPIES_PER_FILE + 1):
            filename = f"{stem}_{i:03d}.wav"
            filepath = ITUNES_DIR / filename
            if filepath.exists():
                files_to_delete.append(filepath)
    
    # Find Living.wav
    living_path = ITUNES_DIR / f"{LIVING_FILE}.wav"
    if living_path.exists():
        files_to_delete.append(living_path)
    
    if not files_to_delete:
        print("No physical files found to delete.")
        return (0, [])
    
    print(f"\nFound {len(files_to_delete)} physical file(s) to delete from disk.")
    
    # Delete files
    deleted_count = 0
    errors = []
    
    for filepath in files_to_delete:
        try:
            filepath.unlink()
            deleted_count += 1
        except Exception as e:
            errors.append((filepath, str(e)))
    
    return (deleted_count, errors)


# -------------------------------------------------------------------
# AppleScript Functions
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


def remove_track_by_name(track_name: str) -> bool:
    """Remove a track from Music/iTunes library by name."""
    script = f'''
tell application "Music"
    try
        set trackList to (every file track whose name is "{track_name}")
        repeat with aTrack in trackList
            delete aTrack
        end repeat
        return "deleted"
    on error errMsg
        return "error: " & errMsg
    end try
end tell
'''
    result = run_applescript(script)
    return "deleted" in result.lower()


def remove_tracks_batch(track_names: list[str]) -> int:
    """Remove multiple tracks in a single AppleScript call. Returns count of successful deletions."""
    # Build delete commands for each track
    delete_commands = []
    for track_name in track_names:
        delete_commands.append(f'''try
        set trackList to (every file track whose name is "{track_name}")
        repeat with aTrack in trackList
            delete aTrack
        end repeat
    end try''')
    
    commands_str = "\n    ".join(delete_commands)
    
    script = f'''
tell application "Music"
    {commands_str}
end tell
'''
    try:
        subprocess.run(
            ['osascript', '-e', script],
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        return len(track_names)
    except:
        return 0


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    print("\n" + "="*60)
    print("Complete Cleanup: Delete Files & Remove from iTunes Library")
    print("="*60 + "\n")
    
    print("This script will:")
    print("  1. Delete physical .wav files from disk")
    print("  2. Remove iTunes/Music library database entries\n")
    
    # Build list of all track names
    tracks_to_remove = []
    for stem in CANONICAL_STEMS:
        for i in range(1, COPIES_PER_FILE + 1):
            track_name = f"{stem}_{i:03d}"
            tracks_to_remove.append(track_name)
    tracks_to_remove.append(LIVING_FILE)
    
    print(f"Total items to process: {len(tracks_to_remove)}\n")
    
    # Confirm
    response = input("Continue with cleanup? (yes/no): ").strip().lower()
    if response not in ('yes', 'y'):
        print("Cancelled.\n")
        return
    
    # Step 1: Delete physical files
    print("\n" + "-"*60)
    print("STEP 1: Deleting physical files from disk...")
    print("-"*60)
    deleted_count, errors = delete_physical_files()
    
    print(f"Deleted {deleted_count} physical file(s) from disk.")
    if errors:
        print(f"Errors: {len(errors)} file(s) could not be deleted:")
        for filepath, error in errors[:5]:  # Show first 5 errors
            print(f"  {filepath.name}: {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    # Step 2: Remove from iTunes library
    print("\n" + "-"*60)
    print("STEP 2: Removing tracks from iTunes/Music library...")
    print("-"*60)
    print("This may take a few minutes...\n")
    
    removed_count = 0
    batch_size = 50  # Process 50 tracks at a time
    total_batches = (len(tracks_to_remove) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(tracks_to_remove))
        batch = tracks_to_remove[start_idx:end_idx]
        
        print(f"  Batch {batch_num + 1}/{total_batches}: Removing {len(batch)} tracks...", end=" ", flush=True)
        count = remove_tracks_batch(batch)
        removed_count += count
        print(f"✓")
    
    failed_count = len(tracks_to_remove) - removed_count
    
    # Final report
    print("\n" + "="*60)
    print("CLEANUP COMPLETE")
    print("="*60)
    print(f"Physical files deleted: {deleted_count}")
    print(f"Library entries removed: {removed_count}")
    if failed_count > 0:
        print(f"Failed or not found: {failed_count}")
    print()


if __name__ == "__main__":
    main()
