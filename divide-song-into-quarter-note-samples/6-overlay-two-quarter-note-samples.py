#!/usr/bin/env python3
"""
Combine two quarter note samples into a single audio file with panning and padded silence.

Usage:
    python 6-combine-quarter-notes-into-piece.py <song_path> <sample1> [panning1] <sample2> [panning2] [padding]

Arguments:
    song_path: Relative path to the song directory (e.g., "how-we-do")
    sample1: First sample filename (e.g., "sample_001.wav")
    panning1: Optional panning value for sample1 (-1.0 to 1.0, default: -1.0 hard left)
    sample2: Second sample filename (e.g., "sample_002.wav")
    panning2: Optional panning value for sample2 (-1.0 to 1.0, default: 1.0 hard right)
    padding: Optional padding silence in seconds (default: 2.0)

Examples:
    # All defaults (hard left, hard right, 2 seconds padding)
    python script.py how-we-do sample_001.wav sample_002.wav
    
    # With panning values
    python script.py how-we-do sample_001.wav -0.5 sample_002.wav 0.5
    
    # With panning and padding
    python script.py how-we-do sample_001.wav -0.5 sample_002.wav 0.5 3.0
"""

import sys
import os
from pathlib import Path
from pydub import AudioSegment
from pydub.generators import Sine
import datetime


def is_numeric(value):
    """Check if a string can be converted to a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def parse_arguments(args):
    """
    Parse command line arguments intelligently.
    Returns: (song_path, sample1, panning1, sample2, panning2, padding)
    """
    if len(args) < 3:
        print("Error: At least 3 arguments required (song_path, sample1, sample2)")
        sys.exit(1)
    
    song_path = args[0]
    
    # Collect remaining arguments
    remaining = args[1:]
    
    # Separate filenames from numeric values
    filenames = []
    numeric_values = []
    
    for arg in remaining:
        if is_numeric(arg):
            numeric_values.append(float(arg))
        else:
            filenames.append(arg)
    
    # Must have exactly 2 filenames
    if len(filenames) != 2:
        print(f"Error: Expected 2 sample filenames, got {len(filenames)}")
        sys.exit(1)
    
    sample1, sample2 = filenames
    
    # Parse numeric values
    # They can be: [panning1, panning2, padding] or [panning1, panning2] or [padding] or []
    # Panning values are typically -1.0 to 1.0
    # Padding is typically >= 0
    
    # Default values
    panning1 = -1.0  # Hard left
    panning2 = 1.0   # Hard right
    padding = 2.0    # 2 seconds
    
    if len(numeric_values) == 0:
        # All defaults
        pass
    elif len(numeric_values) == 1:
        # Could be padding only
        # Assume it's padding (typically > 1)
        padding = numeric_values[0]
    elif len(numeric_values) == 2:
        # Two panning values
        panning1 = numeric_values[0]
        panning2 = numeric_values[1]
    elif len(numeric_values) >= 3:
        # All three values
        panning1 = numeric_values[0]
        panning2 = numeric_values[1]
        padding = numeric_values[2]
    
    return song_path, sample1, panning1, sample2, panning2, padding


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
    
    sample1_path = input_base / sample1
    sample2_path = input_base / sample2
    
    # Check if files exist
    if not sample1_path.exists():
        print(f"Error: Sample 1 not found: {sample1_path}")
        sys.exit(1)
    
    if not sample2_path.exists():
        print(f"Error: Sample 2 not found: {sample2_path}")
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
