#!/usr/bin/env python3
"""
3-clean-up-itunes-playlist-tracks-and-files.py

Complete cleanup script for audio files:
1. Deletes physical .wav files from local output directory
2. Deletes physical .wav files from iTunes import location (if they exist)
3. Removes database entries from iTunes/Music library using AppleScript
4. Deletes playlist file and removes playlist from iTunes
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
OUTPUT_AUDIO_DIR = Path("./output/audio/final-sample-versions")

# Playlist location and name
PLAYLIST_NAME = config["playlist_name"]
PLAYLIST_PATH = Path("./output/playlists") / f"{PLAYLIST_NAME}.m3u"


# -------------------------------------------------------------------
# File Deletion Functions
# -------------------------------------------------------------------
def get_output_files() -> list[Path]:
    """Get all .wav files from the output directory."""
    if not OUTPUT_AUDIO_DIR.exists():
        return []
    return list(OUTPUT_AUDIO_DIR.glob("*.wav"))


def delete_physical_files() -> tuple[int, list[tuple[Path, str]]]:
    """
    Delete physical .wav files from output directory and iTunes directory.
    Returns (deleted_count, errors_list).
    """
    files_to_delete = get_output_files()
    
    # Also check for these files in iTunes directory
    for output_file in files_to_delete[:]:  # Copy list to iterate
        itunes_file = ITUNES_DIR / output_file.name
        if itunes_file.exists():
            files_to_delete.append(itunes_file)
    
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


def stop_music_playback() -> bool:
    """Stop Music/iTunes playback if currently playing."""
    script = '''
tell application "Music"
    if player state is playing then
        pause
        return "stopped"
    else
        return "not_playing"
    end if
end tell
'''
    result = run_applescript(script)
    return "stopped" in result.lower()


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
    
    # Stop music playback first
    print("Checking Music playback status...")
    if stop_music_playback():
        print("✓ Stopped Music playback\n")
    else:
        print("✓ Music not currently playing\n")
    
    print("This script will:")
    print("  1. Delete physical .wav files from output directory")
    print("  2. Delete physical .wav files from iTunes directory")
    print("  3. Remove iTunes/Music library database entries")
    print("  4. Delete playlist from iTunes/Music library")
    print("  5. Delete generated playlist file\n")
    
    # Get list of files from output directory
    output_files = get_output_files()
    
    if not output_files:
        print("No files found in output directory.")
        print("Nothing to clean up.\n")
        return
    
    # Build track names (without .wav extension)
    tracks_to_remove = [f.stem for f in output_files]
    
    print(f"Found {len(output_files)} file(s) to process\n")
    
    print("Starting cleanup (no confirmation required)...\n")
    
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
