#!/usr/bin/env python3
"""
Combine two quarter note samples into a single audio file with panning and padded silence.

Usage:
    python 6-combine-quarter-notes-into-piece.py <song_path> <sample1> <sample2> [panning1] [panning2] [padding]

Arguments:
    song_path: Relative path to the song directory (e.g., "how-we-do")
    sample1: First sample index or filename (e.g., "1", "001", or "sample_001.wav")
    sample2: Second sample index or filename (e.g., "2", "002", or "sample_002.wav")
    panning1: Optional panning value for sample1 (-1.0 to 1.0, default: -1.0 hard left)
    panning2: Optional panning value for sample2 (-1.0 to 1.0, default: 1.0 hard right)
    padding: Optional padding silence in seconds (default: 2.0)

Examples:
    # Using index numbers (all defaults: hard left, hard right, 2 seconds padding)
    python script.py how-we-do 1 2
    python script.py how-we-do 001 002
    
    # Using full filenames
    python script.py how-we-do sample_001.wav sample_002.wav
    
    # With panning values
    python script.py how-we-do 1 2 -0.5 0.5
    
    # With panning and padding
    python script.py how-we-do 1 2 -0.5 0.5 3.0
"""

import sys
import os
from pathlib import Path
from pydub import AudioSegment
from pydub.generators import Sine
import datetime
import glob


def is_numeric(value):
    """Check if a string can be converted to a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def parse_arguments(args):
    """
    Parse command line arguments positionally.
    Returns: (song_path, sample1, panning1, sample2, panning2, padding)
    
    Format: song_path sample1 sample2 [panning1] [panning2] [padding]
    """
    if len(args) < 3:
        print("Error: At least 3 arguments required (song_path, sample1, sample2)")
        sys.exit(1)
    
    song_path = args[0]
    remaining = args[1:]
    
    if len(remaining) < 2:
        print("Error: Need at least 2 sample identifiers")
        sys.exit(1)
    
    # First two arguments are always sample identifiers
    sample1 = remaining[0]
    sample2 = remaining[1]
    
    # Default values
    panning1 = -1.0  # Hard left
    panning2 = 1.0   # Hard right
    padding = 2.0    # 2 seconds
    
    # Parse remaining optional numeric arguments
    if len(remaining) >= 3:
        try:
            panning1 = float(remaining[2])
        except ValueError:
            print(f"Error: Expected numeric panning value for sample1, got '{remaining[2]}'")
            sys.exit(1)
    
    if len(remaining) >= 4:
        try:
            panning2 = float(remaining[3])
        except ValueError:
            print(f"Error: Expected numeric panning value for sample2, got '{remaining[3]}'")
            sys.exit(1)
    
    if len(remaining) >= 5:
        try:
            padding = float(remaining[4])
        except ValueError:
            print(f"Error: Expected numeric padding value, got '{remaining[4]}'")
            sys.exit(1)
    
    return song_path, sample1, panning1, sample2, panning2, padding
    
    return song_path, sample1, panning1, sample2, panning2, padding


def resolve_sample_path(sample_identifier, samples_dir):
    """
    Resolve a sample identifier (index or filename) to the actual file path.
    
    Args:
        sample_identifier: Either an index number (e.g., "1", "001", "0001") or a filename
        samples_dir: Directory containing the sample files
    
    Returns:
        Path object to the sample file
    
    Raises:
        FileNotFoundError: If no matching sample is found
        ValueError: If multiple samples match
    """
    # If it's already a file path and exists, return it
    full_path = samples_dir / sample_identifier
    if full_path.exists() and full_path.is_file():
        return full_path
    
    # Try to interpret as an index number
    # Check if it's numeric (could have leading zeros like "0001")
    if sample_identifier.isdigit():
        # Look for files starting with this index
        # The new format is: INDEX_LYRICS_TIMESTAMP_SONGNAME.wav (e.g., 0001_yeah_1.914894_if-i-cant.wav)
        # Also support old format: sample_INDEX_... (e.g., sample_001_...)
        
        # Normalize to 4-digit format for new files
        index = int(sample_identifier)
        
        patterns = [
            f"{index:04d}_*",        # New format: 0001_*
            f"sample_{index:03d}*",  # Old format: sample_001*
            f"sample_{index:02d}*",  # Old format: sample_01*
            f"sample_{index}*",      # Old format: sample_1*
        ]
        
        matches = []
        for pattern in patterns:
            matches.extend(samples_dir.glob(pattern))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_matches = []
        for match in matches:
            if match not in seen:
                seen.add(match)
                unique_matches.append(match)
        
        if len(unique_matches) == 0:
            raise FileNotFoundError(f"No sample found matching index {index} in {samples_dir}")
        elif len(unique_matches) > 1:
            files_list = '\n  '.join([m.name for m in unique_matches])
            raise ValueError(f"Multiple samples found matching index {index}:\n  {files_list}")
        
        return unique_matches[0]
    
    # If we get here, it's not a valid index or existing file
    raise FileNotFoundError(f"No sample found matching '{sample_identifier}' in {samples_dir}")


def apply_panning(audio, pan_value):
    """
    Apply panning to an audio segment.
    
    Args:
        audio: AudioSegment to pan
        pan_value: -1.0 (hard left) to 1.0 (hard right), 0.0 is center
    
    Returns:
        Panned AudioSegment
    """
    # Ensure stereo
    if audio.channels == 1:
        audio = audio.set_channels(2)
    
    # Clamp pan value
    pan_value = max(-1.0, min(1.0, pan_value))
    
    # Split into left and right channels
    samples = audio.split_to_mono()
    left = samples[0]
    right = samples[1] if len(samples) > 1 else samples[0]
    
    # Apply panning
    # pan_value of -1.0: full left (left at 100%, right at 0%)
    # pan_value of 0.0: center (left at 100%, right at 100%)
    # pan_value of 1.0: full right (left at 0%, right at 100%)
    
    # Calculate gain multipliers (0.0 to 1.0)
    left_gain = (1.0 - pan_value) / 2.0
    right_gain = (1.0 + pan_value) / 2.0
    
    # Apply gain directly as multiplier
    # pydub uses dB, so convert linear gain to dB
    # dB = 20 * log10(gain), but we need to be careful with 0
    import math
    left_db = 20 * math.log10(left_gain) if left_gain > 0 else -120
    right_db = 20 * math.log10(right_gain) if right_gain > 0 else -120
    
    left = left + left_db
    right = right + right_db
    
    # Combine back to stereo
    return AudioSegment.from_mono_audiosegments(left, right)


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    
    # Parse arguments
    song_path, sample1, panning1, sample2, panning2, padding = parse_arguments(sys.argv[1:])
    
    print(f"Song path: {song_path}")
    print(f"Sample 1: {sample1} (panning: {panning1})")
    print(f"Sample 2: {sample2} (panning: {panning2})")
    print(f"Padding: {padding} seconds")
    
    # Get song name from path
    song_name = Path(song_path).name
    
    # Construct input paths
    script_dir = Path(__file__).parent
    input_base = script_dir / "output" / song_name / "quarter-note-samples-labeled-with-lyrics"
    
    # Resolve sample paths (supports index numbers or filenames)
    try:
        sample1_path = resolve_sample_path(sample1, input_base)
        print(f"Resolved sample 1: {sample1_path.name}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error resolving sample 1 '{sample1}': {e}")
        sys.exit(1)
    
    try:
        sample2_path = resolve_sample_path(sample2, input_base)
        print(f"Resolved sample 2: {sample2_path.name}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error resolving sample 2 '{sample2}': {e}")
        sys.exit(1)
    
    # Load audio files
    print(f"\nLoading {sample1}...")
    audio1 = AudioSegment.from_file(sample1_path)
    
    print(f"Loading {sample2}...")
    audio2 = AudioSegment.from_file(sample2_path)
    
    # Ensure both have same sample rate and channels
    target_sample_rate = max(audio1.frame_rate, audio2.frame_rate)
    audio1 = audio1.set_frame_rate(target_sample_rate).set_channels(2)
    audio2 = audio2.set_frame_rate(target_sample_rate).set_channels(2)
    
    # Apply panning
    print(f"\nApplying panning to {sample1} ({panning1})...")
    audio1_panned = apply_panning(audio1, panning1)
    
    print(f"Applying panning to {sample2} ({panning2})...")
    audio2_panned = apply_panning(audio2, panning2)
    
    # Overlay samples (play at the same time)
    print("\nOverlaying samples...")
    combined = audio1_panned.overlay(audio2_panned)
    
    # Add padded silence
    if padding > 0:
        print(f"Adding {padding} seconds of silence...")
        silence = AudioSegment.silent(duration=int(padding * 1000), frame_rate=combined.frame_rate)
        silence = silence.set_channels(2)  # Ensure stereo
        combined = combined + silence
    
    # Create output directory
    output_dir = script_dir / "output" / song_name / "quarter-note-samples-labeled-with-lyrics-curated-by-gpt-overlaid"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sample1_base = Path(sample1).stem
    sample2_base = Path(sample2).stem
    output_filename = f"{sample1_base}_{sample2_base}_{timestamp}.wav"
    output_path = output_dir / output_filename
    
    # Export combined audio
    print(f"\nExporting to {output_path}...")
    combined.export(output_path, format="wav")
    
    print(f"\n✓ Successfully created: {output_path}")
    print(f"  Duration: {len(combined) / 1000:.2f} seconds")
    print(f"  Sample rate: {combined.frame_rate} Hz")
    print(f"  Channels: {combined.channels}")


if __name__ == "__main__":
    main()
