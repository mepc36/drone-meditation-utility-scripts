#!/usr/bin/env python3
"""
2-import-duplicate-padded-samples-into-itunes-playlist.py

Generates an M3U playlist from ./output/rhythmicized-audio/ and plays it via mpv.
"""

from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import lib.config as cfg


# -------------------------------------------------------------------
# CONFIG: Load from input/config/config.json
# -------------------------------------------------------------------
_conf = cfg.load()

SOURCE_AUDIO_DIR = cfg.OUTPUT_RHYTHMICIZED_AUDIO_DIR

# Output locations
OUTPUT_DIR = Path("./output")
PLAYLIST_PATH = OUTPUT_DIR / "playlists" / "playlist.m3u"



# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def ensure_source_files_exist() -> None:
    """Verify source audio directory exists with files."""
    if not SOURCE_AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Source directory not found: {SOURCE_AUDIO_DIR}\n"
            "Please run 1-combine first to create files."
        )

    files = list(SOURCE_AUDIO_DIR.glob("*.wav"))
    if not files:
        raise FileNotFoundError(
            f"No .wav files found in: {SOURCE_AUDIO_DIR}\n"
            "Please run 1-combine first to create files."
        )


def write_m3u(tracks: list[Path]) -> None:
    """Write M3U playlist file pointing to the given tracks."""
    PLAYLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#EXTM3U", *[str(p) for p in tracks], ""]
    PLAYLIST_PATH.write_text("\n".join(lines), encoding="utf-8")


def reset_playlist_folder() -> None:
    """Remove and recreate the playlist folder."""
    import shutil
    playlist_folder = PLAYLIST_PATH.parent
    if playlist_folder.exists():
        shutil.rmtree(playlist_folder)
    playlist_folder.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    print("\nPlaylist Builder\n")

    # Clean playlist folder at start
    reset_playlist_folder()

    # Check source files exist
    ensure_source_files_exist()

    # Get all source files
    source_files = sorted(SOURCE_AUDIO_DIR.glob("*.wav"))
    print(f"Found {len(source_files)} file(s) in {SOURCE_AUDIO_DIR}")

    # Write M3U pointing directly to source files
    abs_tracks = [f.resolve() for f in source_files]
    write_m3u(abs_tracks)

    print(f"Total tracks in playlist: {len(abs_tracks)}\n")
    print(f"Playlist written to: {PLAYLIST_PATH.resolve()}\n")

    print("\nStarting playback with mpv (Ctrl+C to stop)...\n")
    subprocess.run(
        ["mpv", "--no-video", "--gapless-audio=yes", "--loop-playlist=inf", "--shuffle",
         str(PLAYLIST_PATH.resolve())],
    )


if __name__ == "__main__":
    main()
