#!/usr/bin/env python3
"""
remove-from-itunes-library.py

Complete cleanup script for duplicate audio files:
1. Deletes physical .wav files from disk (iTunes import location)
2. Removes database entries from iTunes/Music library using AppleScript
"""

import json
import subprocess
from pathlib import Path


# -------------------------------------------------------------------
# CONFIG: Load from input/config/config.json
# -------------------------------------------------------------------
CONFIG_PATH = Path("./input/config/config.json")
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

ITUNES_DIR = Path(config["itunes_dir"])
CANONICAL_STEMS = config["canonical_files"]
LIVING_FILE = config["living_file"]
SILENCE_FILE = "Silence"
COPIES_PER_FILE = config["copies_per_file"]

# Parse samples_ratio to get actual copy counts
ratio_parts = [int(x) for x in config["samples_ratio"].split(":")]
BREATHING_COPIES = ratio_parts[0]  # First number for Breathing
OTHER_COPIES = ratio_parts[1]  # Second number for other 6 activities
LIVING_COPIES = ratio_parts[2]  # Third number for Living
SILENCE_COPIES = ratio_parts[3] if len(ratio_parts) > 3 else 0  # Fourth number for Silence

BREATHING_STEM = CANONICAL_STEMS[0]  # "Breathing"
OTHER_STEMS = CANONICAL_STEMS[1:]  # All except Breathing

# Playlist location and name
PLAYLIST_NAME = config["playlist_name"]
PLAYLIST_PATH = Path("./output/playlists") / f"{PLAYLIST_NAME}.m3u"

# -------------------------------------------------------------------
# File Deletion Functions
# -------------------------------------------------------------------
def delete_physical_files() -> tuple[int, list[tuple[Path, str]]]:
    """
    Delete physical .wav files from iTunes directory based on samples_ratio.
    Returns (deleted_count, errors_list).
    """
    if not ITUNES_DIR.exists():
        print(f"iTunes directory does not exist: {ITUNES_DIR}")
        return (0, [])
    
    files_to_delete = []
    
    # Find Breathing copies (first ratio number)
    for i in range(1, BREATHING_COPIES + 1):
        filename = f"{BREATHING_STEM}_{i:03d}.wav"
        filepath = ITUNES_DIR / filename
        if filepath.exists():
            files_to_delete.append(filepath)
    
    # Find other canonical files copies (second ratio number)
    for stem in OTHER_STEMS:
        for i in range(1, OTHER_COPIES + 1):
            filename = f"{stem}_{i:03d}.wav"
            filepath = ITUNES_DIR / filename
            if filepath.exists():
                files_to_delete.append(filepath)
    
    # Find Living files (third ratio number)
    if LIVING_COPIES == 1:
        living_path = ITUNES_DIR / f"{LIVING_FILE}.wav"
        if living_path.exists():
            files_to_delete.append(living_path)
    else:
        for i in range(1, LIVING_COPIES + 1):
            filename = f"{LIVING_FILE}_{i:03d}.wav"
            filepath = ITUNES_DIR / filename
            if filepath.exists():
                files_to_delete.append(filepath)
    
    # Find Silence files (fourth ratio number)
    if SILENCE_COPIES > 0:
        for i in range(1, SILENCE_COPIES + 1):
            filename = f"{SILENCE_FILE}_{i:03d}.wav"
            filepath = ITUNES_DIR / filename
            if filepath.exists():
                files_to_delete.append(filepath)
    
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
    print("  2. Remove iTunes/Music library database entries")
    print("  3. Delete playlist from iTunes/Music library")
    print("  4. Delete generated playlist file\n")
    
    # Build list of all track names based on samples_ratio
    tracks_to_remove = []
    
    # Add Breathing tracks (first ratio number)
    for i in range(1, BREATHING_COPIES + 1):
        tracks_to_remove.append(f"{BREATHING_STEM}_{i:03d}")
    
    # Add other canonical tracks (second ratio number)
    for stem in OTHER_STEMS:
        for i in range(1, OTHER_COPIES + 1):
            tracks_to_remove.append(f"{stem}_{i:03d}")
    
    # Add Living tracks (third ratio number)
    if LIVING_COPIES == 1:
        tracks_to_remove.append(LIVING_FILE)
    else:
        for i in range(1, LIVING_COPIES + 1):
            tracks_to_remove.append(f"{LIVING_FILE}_{i:03d}")
    
    # Add Silence tracks (fourth ratio number)
    if SILENCE_COPIES > 0:
        for i in range(1, SILENCE_COPIES + 1):
            tracks_to_remove.append(f"{SILENCE_FILE}_{i:03d}")
    
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
    
    # Step 3: Delete playlist from iTunes/Music library
    print("\n" + "-"*60)
    print("STEP 3: Deleting playlist from iTunes/Music library...")
    print("-"*60)
    
    playlist_removed_from_itunes = False
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
    if "deleted" in result:
        playlist_removed_from_itunes = True
        print(f"Removed playlist from iTunes/Music library: {PLAYLIST_NAME}")
    else:
        print(f"Playlist not found in iTunes/Music library (or already deleted)")
    
    # Step 4: Delete playlist file from disk
    print("\n" + "-"*60)
    print("STEP 4: Deleting playlist file from disk...")
    print("-"*60)
    
    playlist_deleted = False
    if PLAYLIST_PATH.exists():
        try:
            PLAYLIST_PATH.unlink()
            playlist_deleted = True
            print(f"Deleted playlist file: {PLAYLIST_PATH}")
        except Exception as e:
            print(f"Could not delete playlist file: {e}")
    else:
        print("Playlist file not found (already deleted or never created)")
    
    # Final report
    print("\n" + "="*60)
    print("CLEANUP COMPLETE")
    print("="*60)
    print(f"Physical files deleted: {deleted_count}")
    print(f"Library entries removed: {removed_count}")
    print(f"Playlist removed from iTunes: {'Yes' if playlist_removed_from_itunes else 'No'}")
    print(f"Playlist file deleted: {'Yes' if playlist_deleted else 'No'}")
    if failed_count > 0:
        print(f"Failed or not found: {failed_count}")
    print()


if __name__ == "__main__":
    main()
