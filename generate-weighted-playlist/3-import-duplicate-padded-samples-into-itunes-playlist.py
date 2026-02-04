#!/usr/bin/env python3
"""
3-import-duplicate-padded-samples-into-itunes-playlist.py

Imports audio files from ./output/audio/final-sample-versions/ to iTunes/Music
and generates an M3U playlist.
"""

import json
from pathlib import Path
import subprocess


# -------------------------------------------------------------------
# CONFIG: Load from input/config/config.json
# -------------------------------------------------------------------
CONFIG_PATH = Path("./input/config/config.json")
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Extract config sections
shared_config = config["shared_config"]

# Source directory with final samples
SOURCE_AUDIO_DIR = Path("./output/audio/final-sample-versions")

# iTunes import location where files will be copied
ITUNES_DIR = Path(shared_config["itunes_dir"])

# Output locations
OUTPUT_DIR = Path("./output")
PLAYLIST_NAME = shared_config["playlist_name"]
PLAYLIST_PATH = OUTPUT_DIR / "playlists" / f"{PLAYLIST_NAME}.m3u"


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


def ensure_source_files_exist() -> None:
    """Verify source audio directory exists with files."""
    if not SOURCE_AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Source directory not found: {SOURCE_AUDIO_DIR}\n"
            "Please run 1-pad + 2-duplicate OR 1-combine first to create files."
        )
    
    # Check if we have any files
    files = list(SOURCE_AUDIO_DIR.glob("*.wav"))
    if not files:
        raise FileNotFoundError(
            f"No .wav files found in: {SOURCE_AUDIO_DIR}\n"
            "Please run 1-pad + 2-duplicate OR 1-combine first to create files."
        )


def import_folder_to_music(folder: Path) -> str:
    """Import entire folder to iTunes/Music in one operation."""
    script = f'''
tell application "Music"
    add (POSIX file "{folder.resolve()}/")
end tell
'''
    return run_applescript(script)


def write_m3u(tracks: list[Path]) -> None:
    """Write playlist file pointing to selected tracks in iTunes directory."""
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
    print("\nPlaylist Builder\n")

    # Clean playlist folder at start
    reset_playlist_folder()
    
    # Delete existing playlist from iTunes
    print("Checking for existing playlist in iTunes/Music...")
    if delete_playlist_from_itunes():
        print("  Removed existing playlist from iTunes/Music library")
    else:
        print("  No existing playlist found")
    print()

    # Check source files exist
    ensure_source_files_exist()
    
    # Get all source files
    source_files = sorted(SOURCE_AUDIO_DIR.glob("*.wav"))
    print(f"Found {len(source_files)} file(s) in {SOURCE_AUDIO_DIR}")
    
    # Import to iTunes
    print("\n" + "="*60)
    print("Importing files to iTunes/Music...")
    print("This may take a moment...\n")
    
    result = import_folder_to_music(SOURCE_AUDIO_DIR)
    
    print("Import complete!")
    
    # Build playlist from iTunes directory (files should now be there)
    print("\nBuilding playlist from iTunes directory...")
    
    # Get the imported files from iTunes directory
    itunes_files = []
    for source_file in source_files:
        itunes_path = ITUNES_DIR / source_file.name
        if itunes_path.exists():
            itunes_files.append(itunes_path)
        else:
            print(f"  Warning: {source_file.name} not found in iTunes directory")
    
    if not itunes_files:
        print("Error: No files found in iTunes directory after import!")
        return
    
    print(f"Total tracks in playlist: {len(itunes_files)}\n")
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
    print("\nPlaylist Builder\n")

    # Clean playlist folder at start
    reset_playlist_folder()
    
    # Delete existing playlist from iTunes
    print("Checking for existing playlist in iTunes/Music...")
    if delete_playlist_from_itunes():
        print("  Removed existing playlist from iTunes/Music library")
    else:
        print("  No existing playlist found")
    print()

    # Check source files exist
    ensure_source_files_exist()
    
    # Get all source files
    source_files = sorted(SOURCE_AUDIO_DIR.glob("*.wav"))
    print(f"Found {len(source_files)} file(s) in {SOURCE_AUDIO_DIR}")
    
    # Import to iTunes
    print("\n" + "="*60)
    print("Importing files to iTunes/Music...")
    print("This may take a moment...\n")
    
    result = import_folder_to_music(SOURCE_AUDIO_DIR)
    
    print("Import complete!")
    
    # Build playlist from iTunes directory (files should now be there)
    print("\nBuilding playlist from iTunes directory...")
    
    # Get the imported files from iTunes directory
    itunes_files = []
    for source_file in source_files:
        itunes_path = ITUNES_DIR / source_file.name
        if itunes_path.exists():
            itunes_files.append(itunes_path)
        else:
            print(f"  Warning: {source_file.name} not found in iTunes directory")
    
    if not itunes_files:
        print("Error: No files found in iTunes directory after import!")
        return
    
    print(f"Total tracks in playlist: {len(itunes_files)}\n")
    
    # Write playlist
    write_m3u(itunes_files)

    print("Done.")
    print(f"  Playlist written to: {PLAYLIST_PATH.resolve()}\n")
    
    # Auto-open playlist in Music
    print("Opening playlist in Music...")
    subprocess.run(['open', str(PLAYLIST_PATH.resolve())], check=False)
    
    print("\nNext:")
    print("  1) Playlist opened in Apple Music")
    print("  2) Turn Shuffle ON if you want varied playback")
    print("  3) Turn Repeat (All) ON for infinite looping\n")


if __name__ == "__main__":
    main()
