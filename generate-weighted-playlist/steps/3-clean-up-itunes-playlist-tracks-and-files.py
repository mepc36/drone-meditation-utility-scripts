#!/usr/bin/env python3
"""
3-clean-up-itunes-playlist-tracks-and-files.py

Cleanup script for audio files:
1. Deletes physical .wav files from local output directory
2. Deletes generated playlist file
"""

import subprocess
from pathlib import Path


# -------------------------------------------------------------------
# CONFIG: Load from input/config/config.json
# -------------------------------------------------------------------
OUTPUT_AUDIO_DIR = Path("./output/audio")
OUTPUT_RHYTHMICIZED_AUDIO_DIR = Path("./output/rhythmicized-audio")
PLAYLIST_PATH = Path("./output/playlists") / "playlist.m3u"


# -------------------------------------------------------------------
# File Deletion Functions
# -------------------------------------------------------------------
def get_output_files() -> list[Path]:
    """Get all .wav files from the output directories."""
    files: list[Path] = []
    for d in (OUTPUT_AUDIO_DIR, OUTPUT_RHYTHMICIZED_AUDIO_DIR):
        if d.exists():
            files.extend(d.glob("*.wav"))
    return files


def delete_physical_files() -> tuple[int, list[tuple[Path, str]]]:
    """
    Delete physical .wav files from output directory.
    Returns (deleted_count, errors_list).
    """
    files_to_delete = get_output_files()

    if not files_to_delete:
        print("No physical files found to delete.")
        return (0, [])

    print(f"\nFound {len(files_to_delete)} physical file(s) to delete from disk.")

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
# Main
# -------------------------------------------------------------------
def main() -> None:
    print("\n" + "="*60)
    print("Cleanup: Delete Files & Playlist")
    print("="*60 + "\n")

    # Quit mpv before deleting files so it doesn't log 'cannot open file' errors
    print("Quitting mpv (if running)...")
    subprocess.run(["pkill", "-x", "mpv"], capture_output=True)

    print("This script will:")
    print("  1. Delete physical .wav files from output directory")
    print("  2. Delete generated playlist file\n")

    output_files = get_output_files()

    if not output_files and not PLAYLIST_PATH.exists():
        print("Nothing to clean up.\n")
        return

    print(f"Found {len(output_files)} file(s) to process\n")

    # Step 1: Delete physical files
    print("-"*60)
    print("STEP 1: Deleting physical .wav files from disk...")
    print("-"*60)
    deleted_count, errors = delete_physical_files()
    print(f"Deleted {deleted_count} physical file(s) from disk.")
    if errors:
        print(f"Errors ({len(errors)}):")
        for filepath, error in errors[:5]:
            print(f"  {filepath.name}: {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    # Step 2: Delete playlist file
    print("\n" + "-"*60)
    print("STEP 2: Deleting playlist file from disk...")
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
    print(f"Playlist file deleted: {'Yes' if playlist_deleted else 'No'}")
    print()


if __name__ == "__main__":
    main()