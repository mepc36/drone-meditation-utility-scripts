#!/usr/bin/env python3
"""
Main pipeline script to process songs from lyrics fetching to filtered samples.

This script runs the following steps in order:
1. 0-fetch-lyrics.py - Fetch and clean lyrics from Genius
2. 1-separate-song-stems.py - Separate vocals using Demucs
3. 2-divide-song-into-quarter-note-samples.py - Divide audio into quarter note samples
4. 3-align-song-lyrics.py - Align lyrics with audio using Gentle
5. 4-label-quarter-note-samples-with-lyrics.py - Label samples with aligned lyrics
6. 5-filter-out-samples-with-no-lyrics.py - Filter samples with no lyrics

Note: Does NOT run 6-curate-lyrics-via-chatgpt.py or 7-overlay-two-quarter-note-samples.py
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status.
    
    Args:
        cmd: Command to run as list of strings
        description: Description of the step for logging
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"\n✗ {description} failed with error: {e}", file=sys.stderr)
        return False


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(
        description='Run the complete song processing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs the following steps in order:
  1. Fetch lyrics from Genius
  2. Separate vocals using Demucs
  3. Divide audio into quarter note samples
  4. Align lyrics with audio using Gentle
  5. Label samples with aligned lyrics
  6. Filter samples with no lyrics

Required arguments for step 1 (fetch lyrics):
  artist: Artist name (e.g., "50 Cent")
  song: Song name (e.g., "If I Can't")

Note: Steps 2-6 process all songs in ./input/ directory automatically.
        """
    )
    
    # Required arguments for 0-fetch-lyrics.py
    parser.add_argument('artist', help='Artist name (for lyrics fetching)')
    parser.add_argument('song', help='Song name (for lyrics fetching)')
    
    # Optional flags
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force reprocessing even if steps are already completed (ignores config.json checks)')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    
    # Build force flag for scripts that support it
    force_flag = ['-f'] if args.force else []
    
    print("="*80)
    print("SONG PROCESSING PIPELINE")
    print("="*80)
    print(f"Artist: {args.artist}")
    print(f"Song: {args.song}")
    print(f"Force mode: {'ON' if args.force else 'OFF'}")
    print("="*80)
    
    # Step 1: Fetch lyrics
    cmd = [sys.executable, str(script_dir / "0-fetch-lyrics.py"), args.artist, args.song]
    if not run_command(cmd, "Step 1: Fetch and clean lyrics"):
        return 1
    
    # Step 2: Separate vocals
    cmd = [sys.executable, str(script_dir / "1-separate-song-stems.py")] + force_flag
    if not run_command(cmd, "Step 2: Separate vocals using Demucs"):
        return 1
    
    # Step 3: Divide into quarter notes
    cmd = [sys.executable, str(script_dir / "2-divide-song-into-quarter-note-samples.py")] + force_flag
    if not run_command(cmd, "Step 3: Divide audio into quarter note samples"):
        return 1
    
    # Step 4: Align lyrics
    cmd = [sys.executable, str(script_dir / "3-align-song-lyrics.py")] + force_flag
    if not run_command(cmd, "Step 4: Align lyrics with audio using Gentle"):
        return 1
    
    # Step 5: Label samples
    cmd = [sys.executable, str(script_dir / "4-label-quarter-note-samples-with-lyrics.py")] + force_flag
    if not run_command(cmd, "Step 5: Label samples with aligned lyrics"):
        return 1
    
    # Step 6: Filter no-lyrics samples
    cmd = [sys.executable, str(script_dir / "5-filter-out-samples-with-no-lyrics.py")]
    if not run_command(cmd, "Step 6: Filter samples with no lyrics"):
        return 1
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\nAll steps completed successfully.")
    print("\nOutput directory: ./output/")
    print("\nTo curate lyrics (optional), run:")
    print("  python 6-curate-lyrics-via-chatgpt.py")
    print("\nTo overlay samples (optional), run:")
    print("  python 7-overlay-two-quarter-note-samples.py")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
