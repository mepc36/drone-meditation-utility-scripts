"""
Align song lyrics with audio using Gentle forced aligner.

This script:
1. Takes vocals.wav from ./input/{SONG_NAME}/demucs/
2. Takes lyrics from ./input/{SONG_NAME}/lyrics/ (must be exactly 1 .txt file)
3. Sends both to Gentle aligner API
4. Saves alignment JSON response to ./input/{SONG_NAME}/gentle/
"""

import os
from pathlib import Path
import json
import requests
import argparse
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()


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


def main():
    """Main function to align lyrics with audio."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Align song lyrics with audio using Gentle')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force reprocessing even if alignment already exists')
    args = parser.parse_args()
    
    # Get directories
    script_dir = Path(__file__).parent
    input_dir = script_dir / "input"
    
    # Get gentle aligner URL from environment
    gentle_url = os.getenv('GENTLE_ALIGNER_PROD_URL')
    if not gentle_url:
        raise ValueError("GENTLE_ALIGNER_PROD_URL not found in .env file")
    
    print(f"Gentle Aligner URL: {gentle_url}")
    
    # Find all song directories
    song_dirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name != '.DS_Store' and d.name != 'prompts']
    
    if not song_dirs:
        print("No song directories found in ./input")
        return
    
    print(f"\nFound {len(song_dirs)} song(s) to process")
    
    # Get output directory base
    output_base_dir = script_dir / "output"
    
    # Process each song directory
    for song_dir in song_dirs:
        song_name = song_dir.name
        
        print(f"\n{'='*80}")
        print(f"Processing: {song_name}")
        print(f"{'='*80}")
        
        # Check if output already exists (skip check if force flag is used)
        output_file = output_base_dir / song_name / "gentle" / "alignment.json"
        if not args.force and output_file.exists():
            print(f"⏭  Skipping - alignment already exists at {output_file}")
            print(f"   Use -f or --force flag to reprocess")
            continue
        
        # Find vocals.wav in output directory
        vocals_file = output_base_dir / song_name / "demucs" / "acappella.wav"
        if not vocals_file.exists():
            raise FileNotFoundError(f"acappella.wav not found at {vocals_file}\nRun 1-separate-song-stems.py first.")
        
        print(f"✓ Found acappella: {vocals_file}")
        
        # Find lyrics .txt file
        lyrics_dir = song_dir / "lyrics"
        if not lyrics_dir.exists():
            raise FileNotFoundError(f"Lyrics directory not found at {lyrics_dir}")
        
        lyrics_files = list(lyrics_dir.glob("*.txt"))
        
        if not lyrics_files:
            raise FileNotFoundError(f"No .txt files found in {lyrics_dir}")
        
        if len(lyrics_files) > 1:
            files_list = '\n    '.join([f.name for f in lyrics_files])
            raise ValueError(f"Found {len(lyrics_files)} .txt files in {lyrics_dir}, but only 1 is allowed.\n  Found files:\n    {files_list}")
        
        lyrics_file = lyrics_files[0]
        print(f"✓ Found lyrics: {lyrics_file}")
        
        # Create output directory
        output_dir = output_base_dir / song_name / "gentle"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Send request to Gentle aligner
        print(f"\nSending request to Gentle aligner...")
        print("This may take several minutes...")
        
        try:
            with open(vocals_file, 'rb') as audio_f, open(lyrics_file, 'rb') as lyrics_f:
                files = {
                    'audio': ('vocals.wav', audio_f, 'audio/wav'),
                    'transcript': ('lyrics.txt', lyrics_f, 'text/plain')
                }
                
                response = requests.post(gentle_url, files=files, timeout=600)
                response.raise_for_status()
            
            # Parse JSON response
            alignment_data = response.json()
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(alignment_data, f, indent=2)
            
            # Check for words in response
            if 'words' not in alignment_data:
                print(f"⚠ Warning: Response does not contain 'words' field")
                write_status_marker(output_dir, 'error', "Response does not contain 'words' field")
                continue
            
            num_words = len(alignment_data['words'])
            num_aligned = len([w for w in alignment_data['words'] if w.get('case') == 'success'])
            
            print(f"\n✓ Alignment complete!")
            print(f"  Total words: {num_words}")
            print(f"  Successfully aligned: {num_aligned}")
            print(f"  Failed: {num_words - num_aligned}")
            print(f"  Output saved to: {output_file}")
            
            # Write success marker
            write_status_marker(output_dir, 'success')
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error calling Gentle aligner: {e}"
            print(f"⚠ {error_msg}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"  Response status: {e.response.status_code}")
                print(f"  Response content: {e.response.text[:500]}")
            write_status_marker(output_dir, 'error', error_msg)
            continue
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON response from Gentle aligner"
            print(f"⚠ Error: {error_msg}")
            print(f"  Response content: {response.text[:500]}")
            write_status_marker(output_dir, 'error', error_msg)
            continue


if __name__ == "__main__":
    main()
