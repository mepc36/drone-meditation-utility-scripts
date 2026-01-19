"""
Filter out quarter note samples with no lyrics.

This script:
1. Scans ./output/{SONG_NAME}/quarter-note-samples-labeled-with-lyrics directories
2. Identifies files with "no-lyrics" in the filename
3. Moves them to ./output/{SONG_NAME}/gentle-filtered-no-lyrics for safekeeping
"""

from pathlib import Path
import shutil
from datetime import datetime


def write_status_marker(output_dir: Path, status: str, error_msg: str = None) -> None:
    """Write status marker file (.success or .error) to output directory.
    
    Args:
        output_dir: Directory to write the marker file
        status: Either 'success' or 'error'
        error_msg: Optional error message for .error files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove old status markers
    for old_marker in ['.success', '.error']:
        old_file = output_dir / old_marker
        if old_file.exists():
            old_file.unlink()
    
    # Write new status marker
    marker_file = output_dir / f'.{status}'
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(marker_file, 'w') as f:
        f.write(f"timestamp: {timestamp}\n")
        if error_msg:
            f.write(f"error: {error_msg}\n")
from datetime import datetime


def filter_no_lyrics_samples_from_dir(source_dir: Path, filtered_dir: Path, track_type: str) -> int:
    """
    Filter out samples with no lyrics from a specific directory.
    
    Args:
        source_dir: Directory to scan for no-lyrics files
        filtered_dir: Directory to move no-lyrics files to
        track_type: Description of track type (for logging)
        
    Returns:
        Number of files moved
    """
    if not source_dir.exists():
        print(f"  ⚠️  {track_type} directory not found: {source_dir}")
        return 0
    
    # Find all files with "no-lyrics" in the name
    no_lyrics_files = [f for f in source_dir.iterdir() 
                       if f.is_file() and "no-lyrics" in f.name.lower()]
    
    if not no_lyrics_files:
        print(f"  ✓ {track_type}: No files with 'no-lyrics' found")
        return 0
    
    print(f"  {track_type}: Found {len(no_lyrics_files)} file(s) with 'no-lyrics'")
    
    # Create filtered directory
    filtered_dir.mkdir(parents=True, exist_ok=True)
    
    # Move files
    moved_count = 0
    for file in no_lyrics_files:
        destination = filtered_dir / file.name
        
        # If destination exists, skip to avoid overwriting
        if destination.exists():
            print(f"    ⏭  Skipping {file.name} (already exists in filtered directory)")
            continue
        
        shutil.move(str(file), str(destination))
        moved_count += 1
        
        if moved_count % 10 == 0:
            print(f"    Moved {moved_count}/{len(no_lyrics_files)} files...")
    
    if moved_count > 0:
        print(f"  ✓ {track_type}: Moved {moved_count} file(s)")
    
    return moved_count


def filter_no_lyrics_samples(output_dir: Path, song_name: str) -> None:
    """
    Filter out samples with no lyrics from a song's labeled directories.
    
    Args:
        output_dir: Base output directory
        song_name: Name of the song directory
    """
    filtered_dir = output_dir / song_name / "quarter-note-samples-with-no-lyrics"
    
    total_moved = 0
    
    # Process full song samples
    full_song_dir = output_dir / song_name / "quarter-note-samples-labeled-with-lyrics"
    total_moved += filter_no_lyrics_samples_from_dir(full_song_dir, filtered_dir, "Full song")
    
    # Process acappella samples
    acappella_dir = output_dir / song_name / "quarter-note-samples-acappella-labeled-with-lyrics"
    total_moved += filter_no_lyrics_samples_from_dir(acappella_dir, filtered_dir, "Acappella")
    
    if total_moved > 0:
        print(f"\n  ✓ Total files moved: {total_moved}")
        print(f"  Output: {filtered_dir}")


def main():
    """Main function to filter samples."""
    # Get directories
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    
    if not output_dir.exists():
        print("Output directory not found")
        return
    
    # Find all song directories
    song_dirs = [d for d in output_dir.iterdir() 
                 if d.is_dir() and d.name != '.DS_Store']
    
    if not song_dirs:
        print("No song directories found in ./output")
        return
    
    print(f"\nFound {len(song_dirs)} song(s) to process")
    
    # Process each song directory
    for song_dir in song_dirs:
        song_name = song_dir.name
        
        print(f"\n{'='*80}")
        print(f"Processing: {song_name}")
        print(f"{'='*80}")
        
        try:
            filter_no_lyrics_samples(output_dir, song_name)
            
            # Write success marker
            write_status_marker(song_dir, 'success')
        except Exception as e:
            # Write error marker
            write_status_marker(song_dir, 'error', str(e))
            print(f"  ✗ Error: {e}")
            continue
    
    print(f"\n✓ Complete!")


if __name__ == "__main__":
    main()
