#!/usr/bin/env python3
"""
2-import-duplicate-padded-samples-into-itunes-playlist.py

Generates an M3U playlist from ./output/rhythmicized-audio/ and plays it via mpv.
"""

from pathlib import Path
import shutil
import subprocess
import sys
import io

sys.path.insert(0, str(Path(__file__).parent.parent))

import lib.config as cfg


SOURCE_AUDIO_DIR = cfg.OUTPUT_RHYTHMICIZED_AUDIO_DIR

MUSIC_GROUPS = {'strings', 'kick', 'snare', 'kickstab', 'snarestab', 'acappella'}


def truncate_filename(path_str: str) -> str:
    """Return artist_sample_group.wav, stripping everything after the music group.
    For dualpan files, returns both sample names joined by ' + '."""
    name = Path(path_str).stem
    parts = name.split('_')
    group_indices = [i for i, part in enumerate(parts) if part in MUSIC_GROUPS]
    if len(group_indices) >= 2:
        first = '_'.join(parts[:group_indices[0] + 1])
        second = '_'.join(parts[group_indices[0] + 1:group_indices[1] + 1])
        return first + ' + ' + second
    for i, part in enumerate(parts):
        if part in MUSIC_GROUPS:
            return '_'.join(parts[:i + 1]) + '.wav'
    return Path(path_str).name  # silence files and unknowns — show basename as-is

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

    # Write M3U pointing directly to source files
    abs_tracks = [f.resolve() for f in source_files]
    write_m3u(abs_tracks)

    print(f"Total tracks in playlist: {len(abs_tracks)}")
    print(f"Playlist created.\n")

    proc = subprocess.Popen(
        ["mpv", "--no-video", "--gapless-audio=yes", "--loop-playlist=inf", "--shuffle",
         str(PLAYLIST_PATH.resolve())],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in io.TextIOWrapper(proc.stdout.detach(), errors='replace'):
        line = line.rstrip('\n')
        if line.startswith('Playing: '):
            print(f'\n{Path(truncate_filename(line[len("Playing: "):])).stem}')
        elif line.startswith('A:') or line.startswith('AO:') or line.startswith(' (+) ') or line.strip().startswith('●'):
            pass  # suppress progress and codec info lines
        else:
            print(line)
    proc.wait()


if __name__ == "__main__":
    main()
