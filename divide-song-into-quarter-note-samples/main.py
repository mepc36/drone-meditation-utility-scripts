#!/usr/bin/env python3
"""
Main pipeline script to process songs from lyrics fetching to filtered samples.

This script runs the following steps in order:
1. 0-fetch-lyrics.py - Fetch and clean lyrics from Genius (optional)
2. 1-separate-song-stems.py - Separate vocals using Demucs
3. 2-divide-song-into-quarter-note-samples.py - Divide audio into quarter note samples
4. 3-align-song-lyrics.py - Align lyrics with audio using Gentle
5. 4-label-quarter-note-samples-with-lyrics.py - Label samples with aligned lyrics
6. 5-filter-out-samples-with-no-lyrics.py - Filter samples with no lyrics

Note: Does NOT run 6-curate-lyrics-via-chatgpt.py or 7-overlay-two-quarter-note-samples.py

Usage:
  # Fetch lyrics and process a new song:
  python main.py "50 Cent" "If I Can't"
  
  # Process existing song(s) in ./input:
  python main.py song-name-here
  
  # Process all songs in ./input:
  python main.py
"""

import subprocess
import sys
import argparse
import traceback
from pathlib import Path
from datetime import datetime


def write_error_log(output_dir: Path, song_name: str, step: str, error: Exception) -> None:
    """Write error information to a log file.
    
    Args:
        output_dir: Base output directory
        song_name: Name of the song
        step: Description of the step that failed
        error: The exception that occurred
    """
    error_log_dir = output_dir / song_name
    error_log_dir.mkdir(parents=True, exist_ok=True)
    
    error_log_file = error_log_dir / "error-log.txt"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(error_log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"ERROR: {timestamp}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Step: {step}\n")
        f.write(f"Song: {song_name}\n")
        f.write(f"Error: {str(error)}\n")
        f.write(f"\nTraceback:\n")
        f.write(traceback.format_exc())
        f.write(f"\n{'='*80}\n")


def run_command(cmd: list, description: str, song_name: str, output_dir: Path) -> bool:
    """Run a command and return success status.
    
    Args:
        cmd: Command to run as list of strings
        description: Description of the step for logging
        song_name: Name of the song being processed
        output_dir: Base output directory for error logs
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        # Stream output in real-time instead of buffering
        result = subprocess.run(
            cmd, 
            check=True,
            text=True
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = f"{description} failed with exit code {e.returncode}"
        print(f"\n✗ {error_msg}", file=sys.stderr)
        
        # Write to error log
        write_error_log(output_dir, song_name, description, e)
        print(f"Error logged to: {output_dir / song_name / 'error-log.txt'}")
        return False
    except Exception as e:
        error_msg = f"{description} failed with error: {e}"
        print(f"\n✗ {error_msg}", file=sys.stderr)
        
        # Write to error log
        write_error_log(output_dir, song_name, description, e)
        print(f"Error logged to: {output_dir / song_name / 'error-log.txt'}")
        return False


def get_song_directories(input_dir: Path, specific_song: str = None) -> list:
    """Get list of song directories to process.
    
    Args:
        input_dir: Input directory path
        specific_song: Optional specific song name to filter by
        
    Returns:
        List of song directory paths
    """
    if specific_song:
        song_dir = input_dir / specific_song
        if not song_dir.exists():
            raise FileNotFoundError(f"Song directory not found: {song_dir}")
        return [song_dir]
    
    # Get all song directories
    song_dirs = [
        d for d in input_dir.iterdir()
        if d.is_dir() and d.name not in {".DS_Store", "prompts"}
    ]
    
    return song_dirs


def process_song(song_dir: Path, script_dir: Path, output_dir: Path, force_flag: list, skip_lyrics: bool = False) -> bool:
    """Process a single song through the pipeline.
    
    Args:
        song_dir: Path to the song directory
        script_dir: Path to the script directory
        output_dir: Path to the output directory
        force_flag: Force flag list for scripts
        skip_lyrics: Whether to skip lyrics fetching step
        
    Returns:
        True if all steps succeeded, False if any failed
    """
    song_name = song_dir.name
    all_succeeded = True
    
    print(f"\n{'#'*80}")
    print(f"# PROCESSING SONG: {song_name}")
    print(f"{'#'*80}")
    
    steps = []
    
    # Skip lyrics step if requested
    if not skip_lyrics:
        # Note: This would require artist and song to be known, skip for now
        pass
    
    # Steps 1-6 (or 2-6 if skipping lyrics)
    steps = [
        (str(script_dir / "1-separate-song-stems.py"), force_flag, "Step 1: Separate vocals using Demucs"),
        (str(script_dir / "2-divide-song-into-quarter-note-samples.py"), force_flag, "Step 2: Divide audio into quarter note samples"),
        (str(script_dir / "3-align-song-lyrics.py"), force_flag, "Step 3: Align lyrics with audio using Gentle"),
        (str(script_dir / "4-label-quarter-note-samples-with-lyrics.py"), force_flag, "Step 4: Label samples with aligned lyrics"),
        (str(script_dir / "5-filter-out-samples-with-no-lyrics.py"), [], "Step 5: Filter samples with no lyrics"),
    ]
    
    for script_path, flags, description in steps:
        cmd = [sys.executable, script_path] + flags
        if not run_command(cmd, description, song_name, output_dir):
            all_succeeded = False
            print(f"\n⚠️  Continuing to next song despite error in {song_name}...")
            break  # Skip remaining steps for this song
    
    return all_succeeded


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(
        description='Run the complete song processing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch lyrics and process a new song:
  python main.py "50 Cent" "If I Can't"
  
  # Process a specific existing song in ./input:
  python main.py if-i-cant
  
  # Process all songs in ./input:
  python main.py

Options:
  -f, --force    Force reprocessing even if steps are already completed
        """
    )
    
    # Optional positional arguments
    parser.add_argument('artist_or_song', nargs='?', default=None,
                        help='Artist name (with song arg) or song directory name')
    parser.add_argument('song', nargs='?', default=None,
                        help='Song name (requires artist arg)')
    
    # Optional flags
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force reprocessing even if steps are already completed (ignores config.json checks)')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    input_dir = script_dir / "input"
    output_dir = script_dir / "output"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build force flag for scripts that support it
    force_flag = ['-f'] if args.force else []
    
    # Determine mode: fetch new song, process specific song, or process all
    fetch_lyrics = False
    specific_song = None
    
    if args.song:
        # Both artist and song provided - fetch lyrics mode
        fetch_lyrics = True
        artist = args.artist_or_song
        song = args.song
        
        print("="*80)
        print("SONG PROCESSING PIPELINE - NEW SONG MODE")
        print("="*80)
        print(f"Artist: {artist}")
        print(f"Song: {song}")
        print(f"Force mode: {'ON' if args.force else 'OFF'}")
        print("="*80)
        
        # Step 0: Fetch lyrics
        cmd = [sys.executable, str(script_dir / "0-fetch-lyrics.py"), artist, song]
        if not run_command(cmd, "Step 0: Fetch and clean lyrics", song, output_dir):
            return 1
        
        # Get song slug for processing
        import re
        specific_song = re.sub(r'[^a-z0-9]+', '-', song.lower()).strip('-')
        
    elif args.artist_or_song:
        # Only one arg provided - specific song mode
        specific_song = args.artist_or_song
        
        print("="*80)
        print("SONG PROCESSING PIPELINE - SPECIFIC SONG MODE")
        print("="*80)
        print(f"Song: {specific_song}")
        print(f"Force mode: {'ON' if args.force else 'OFF'}")
        print("="*80)
    else:
        # No args provided - process all songs
        print("="*80)
        print("SONG PROCESSING PIPELINE - PROCESS ALL SONGS")
        print("="*80)
        print(f"Force mode: {'ON' if args.force else 'OFF'}")
        print("="*80)
    
    # Get song directories to process
    try:
        song_dirs = get_song_directories(input_dir, specific_song)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    if not song_dirs:
        print(f"No song directories found in {input_dir}")
        return 0
    
    print(f"\nFound {len(song_dirs)} song(s) to process\n")
    
    # Process each song
    results = {}
    for song_dir in song_dirs:
        success = process_song(song_dir, script_dir, output_dir, force_flag, skip_lyrics=not fetch_lyrics)
        results[song_dir.name] = success
    
    # Generate summary file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summaries_dir = output_dir / "1-run-summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summaries_dir / f"run_summary_{timestamp}.txt"
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SONG PROCESSING PIPELINE SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Force mode: {'ON' if args.force else 'OFF'}\n")
        f.write(f"Total songs processed: {len(results)}\n")
        f.write("\n")
        
        successful = [name for name, success in results.items() if success]
        failed = [name for name, success in results.items() if not success]
        
        f.write(f"✓ Successful: {len(successful)}/{len(results)}\n")
        for name in successful:
            f.write(f"  - {name}\n")
        
        if failed:
            f.write(f"\n✗ Failed: {len(failed)}/{len(results)}\n")
            for name in failed:
                f.write(f"  - {name} (see ./output/{name}/error-log.txt)\n")
        
        f.write("\n")
        f.write("Output directory: ./output/\n")
        f.write("="*80 + "\n")
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    
    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]
    
    print(f"\n✓ Successful: {len(successful)}/{len(results)}")
    for name in successful:
        print(f"  - {name}")
    
    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(results)}")
        for name in failed:
            print(f"  - {name} (see ./output/{name}/error-log.txt)")
    
    print("\nOutput directory: ./output/")
    print(f"Summary saved to: {summary_file}")
    print("\nTo curate lyrics (optional), run:")
    print("  python 6-curate-lyrics-via-chatgpt.py")
    print("\nTo overlay samples (optional), run:")
    print("  python 7-overlay-two-quarter-note-samples.py")
    print("="*80)
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
