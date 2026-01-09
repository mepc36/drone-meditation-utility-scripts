"""
Label quarter note samples with lyrics from alignment data.

This script:
1. Reads the alignment JSON from ./input/{SONG_NAME}/gentle/alignment.json
2. Reads quarter note samples from ./output/{SONG_NAME}/quarter-note-samples(-acappella)/
3. Determines which words occur during each quarter note based on timestamps
4. Copies files to new directories with lyrics labels added to filenames
"""

import os
from pathlib import Path
import json
import shutil
import re
import argparse


def text_to_kebab_case(text: str) -> str:
    """Convert text to kebab-case.
    
    Args:
        text: Text to convert
        
    Returns:
        Kebab-case version
    """
    # Convert to lowercase
    text = text.lower()
    # Replace spaces and underscores with hyphens
    text = re.sub(r'[\s_]+', '-', text)
    # Remove non-alphanumeric characters except hyphens
    text = re.sub(r'[^a-z0-9-]+', '', text)
    # Remove multiple consecutive hyphens
    text = re.sub(r'-+', '-', text)
    # Remove leading/trailing hyphens
    text = text.strip('-')
    return text


def load_alignment_data(alignment_file: Path) -> dict:
    """Load alignment data from JSON file.
    
    Args:
        alignment_file: Path to alignment JSON file
        
    Returns:
        Dictionary with alignment data
    """
    with open(alignment_file, 'r') as f:
        return json.load(f)


def get_words_for_quarter_note(words: list, qn_start: float, qn_end: float) -> list:
    """Get all words that occur during a quarter note time window.
    
    A word is included if it overlaps at all with the quarter note window.
    
    Args:
        words: List of word dictionaries from alignment
        qn_start: Quarter note start time in seconds
        qn_end: Quarter note end time in seconds
        
    Returns:
        List of word strings that occur during this quarter note
    """
    result = []
    
    for word_data in words:
        # Skip words that failed alignment
        if word_data.get('case') != 'success':
            continue
        
        word_start = word_data.get('start')
        word_end = word_data.get('end')
        
        # Skip if missing timing data
        if word_start is None or word_end is None:
            continue
        
        # Check if word overlaps with quarter note window
        # Word overlaps if: word_start < qn_end AND word_end > qn_start
        if word_start < qn_end and word_end > qn_start:
            result.append(word_data.get('word', ''))
    
    return result


def parse_quarter_note_filename(filename: str) -> tuple:
    """Parse a quarter note filename to extract index and timestamp.
    
    Expected format: {index:04d}_{timestamp}_{prefix}.wav
    Example: 0001_0.000000_poppin-them-thangs.wav
    
    Args:
        filename: The filename to parse
        
    Returns:
        Tuple of (index, timestamp, prefix, is_partial)
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    
    if len(parts) < 3:
        raise ValueError(f"Invalid filename format: {filename}")
    
    index = int(parts[0])
    timestamp = float(parts[1])
    
    # Check if it's a partial sample
    is_partial = 'partial' in filename.lower()
    
    # Prefix is everything after timestamp
    prefix = '_'.join(parts[2:])
    if is_partial:
        prefix = prefix.replace('-partial', '').replace('_partial', '')
    
    return index, timestamp, prefix, is_partial


def process_quarter_notes(source_dir: Path, output_dir: Path, words: list, 
                          quarter_note_duration: float, is_acappella: bool = False):
    """Process quarter note samples and copy with lyrics labels.
    
    Args:
        source_dir: Directory containing original quarter note samples
        output_dir: Directory to write labeled samples
        words: List of word dictionaries from alignment
        quarter_note_duration: Duration of one quarter note in seconds
        is_acappella: Whether processing acappella samples
    """
    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all wav files
    wav_files = sorted(source_dir.glob("*.wav"))
    
    if not wav_files:
        print(f"No WAV files found in {source_dir}")
        return
    
    track_type = "acappella" if is_acappella else "full song"
    print(f"\nProcessing {track_type} samples from {source_dir.name}")
    print(f"Found {len(wav_files)} samples")
    
    for wav_file in wav_files:
        try:
            # Parse filename
            index, timestamp, prefix, is_partial = parse_quarter_note_filename(wav_file.name)
            
            # Calculate quarter note time window
            qn_start = timestamp
            qn_end = timestamp + quarter_note_duration
            
            # Get words that occur during this quarter note
            qn_words = get_words_for_quarter_note(words, qn_start, qn_end)
            
            # Create words label
            if qn_words:
                words_label = text_to_kebab_case(' '.join(qn_words))
            else:
                words_label = "no-lyrics"
            
            # Create new filename
            partial_suffix = "_partial" if is_partial else ""
            new_filename = f"{index:04d}_{timestamp:.6f}_{prefix}_{words_label}{partial_suffix}.wav"
            
            # Copy file
            output_path = output_dir / new_filename
            shutil.copy2(wav_file, output_path)
            
            if (index) % 10 == 0:
                print(f"  Processed {index}/{len(wav_files)} samples...")
        
        except Exception as e:
            print(f"Error processing {wav_file.name}: {e}")
            continue
    
    print(f"✓ Completed {track_type} labeling")
    print(f"  Output: {output_dir}")


def main():
    """Main function to label quarter note samples with lyrics."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Label quarter note samples with lyrics')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force reprocessing even if output already exists')
    args = parser.parse_args()
    
    # Get directories
    script_dir = Path(__file__).parent
    input_dir = script_dir / "input"
    output_dir = script_dir / "output"
    
    # Find all song directories
    song_dirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name != '.DS_Store' and d.name != 'prompts']
    
    if not song_dirs:
        print("No song directories found in ./input")
        return
    
    print(f"\nFound {len(song_dirs)} song(s) to process")
    
    # Process each song directory
    for song_dir in song_dirs:
        song_name = song_dir.name
        
        print(f"\n{'='*80}")
        print(f"Processing: {song_name}")
        print(f"{'='*80}")
        
        # Check if output already exists (skip check if force flag is used)
        full_song_output = output_dir / song_name / "quarter-note-samples-labeled-with-lyrics"
        if not args.force and full_song_output.exists() and list(full_song_output.glob("*.wav")):
            print(f"⏭  Skipping - labeled samples already exist at {full_song_output}")
            print(f"   Use -f or --force flag to reprocess")
            continue
        
        # Load alignment data
        alignment_file = output_dir / song_name / "gentle" / "alignment.json"
        if not alignment_file.exists():
            raise FileNotFoundError(f"Alignment file not found at {alignment_file}\nRun 3-align-song-lyrics.py first.")
        
        print(f"✓ Found alignment data: {alignment_file}")
        alignment_data = load_alignment_data(alignment_file)
        words = alignment_data.get('words', [])
        print(f"  Total words in alignment: {len(words)}")
        
        # Load config to get BPM (for quarter note duration)
        config_file = song_dir / "config" / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found at {config_file}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        bpm = config.get('bpm')
        if not bpm:
            raise ValueError(f"BPM not found in config file: {config_file}")
        
        quarter_note_duration = 60.0 / bpm
        print(f"✓ BPM: {bpm:.2f}")
        print(f"  Quarter note duration: {quarter_note_duration:.6f} seconds")
        
        # Process full song samples
        full_song_source = output_dir / song_name / "quarter-note-samples"
        process_quarter_notes(full_song_source, full_song_output, words, quarter_note_duration, is_acappella=False)
        
        # Process acappella samples
        acappella_source = output_dir / song_name / "quarter-note-samples-acappella"
        acappella_output = output_dir / song_name / "quarter-note-samples-acappella-labeled-with-lyrics"
        process_quarter_notes(acappella_source, acappella_output, words, quarter_note_duration, is_acappella=True)
        
        print(f"\n{'='*80}")
        print(f"✓ Completed labeling for {song_name}")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
