"""
Divide an audio file into quarter note samples based on BPM.

This script:
1. Takes a single audio file from ./input (any format: mp3, wav, flac, etc.)
2. Automatically detects the song's BPM using librosa
3. Calculates quarter note duration based on BPM
4. Divides the song into quarter note segments
5. Exports each segment to ./output/audio as WAV files
"""

import os
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import json
import shutil


def load_bpm_cache(cache_file: Path) -> dict:
    """Load BPM cache from JSON file.
    
    Args:
        cache_file: Path to the BPM cache JSON file
        
    Returns:
        Dictionary with filename -> bpm mapping
    """
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_bpm_cache(cache_file: Path, cache_data: dict) -> None:
    """Save BPM cache to JSON file.
    
    Args:
        cache_file: Path to the BPM cache JSON file
        cache_data: Dictionary with filename -> bpm mapping
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)


def get_bpm(audio_file: Path, cache_file: Path) -> float:
    """Get BPM for an audio file, using cache if available.
    
    Args:
        audio_file: Path to the audio file
        cache_file: Path to the BPM cache JSON file
        
    Returns:
        BPM as a float
    """
    filename = audio_file.name
    
    # Load cache
    cache_data = load_bpm_cache(cache_file)
    
    # Check if BPM is cached
    if filename in cache_data and "bpm" in cache_data[filename]:
        bpm = cache_data[filename]["bpm"]
        print(f"Using cached BPM: {bpm:.2f}")
        return bpm
    
    # Detect BPM
    print(f"Analyzing audio to detect BPM...")
    y, sr = librosa.load(str(audio_file))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # librosa returns tempo as numpy array, extract scalar
    if hasattr(tempo, '__iter__'):
        bpm = float(tempo[0])
    else:
        bpm = float(tempo)
    
    # Save to cache
    cache_data[filename] = {"bpm": bpm}
    save_bpm_cache(cache_file, cache_data)
    print(f"Detected and cached BPM: {bpm:.2f}")
    
    return bpm


def calculate_quarter_note_duration(bpm: float) -> float:
    """
    Calculate the duration of a quarter note in seconds.
    
    BPM = beats per minute (where beat = quarter note)
    Duration of one quarter note = 60 seconds / BPM
    """
    duration_seconds = 60.0 / bpm
    return duration_seconds


def divide_song_into_quarter_notes(audio_file: Path, bpm: float, output_dir: Path) -> None:
    """
    Divide an audio file into quarter note segments.
    
    Args:
        audio_file: Path to the input audio file (any format)
        bpm: Beats per minute of the song
        output_dir: Directory to save the output segments
    """
    print(f"\nProcessing: {audio_file.name}")
    print(f"BPM: {bpm}")
    
    # Load the audio file using librosa (handles any format)
    audio_data, sample_rate = librosa.load(str(audio_file), sr=None, mono=False)
    
    # Handle stereo vs mono
    if len(audio_data.shape) == 1:
        # Mono
        total_samples = len(audio_data)
    else:
        # Stereo - transpose so shape is (samples, channels)
        audio_data = audio_data.T
        total_samples = len(audio_data)
    
    total_duration_sec = total_samples / sample_rate
    
    # Calculate quarter note duration in seconds
    quarter_note_sec = calculate_quarter_note_duration(bpm)
    quarter_note_samples = int(quarter_note_sec * sample_rate)
    
    print(f"Quarter note duration: {quarter_note_sec:.3f} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Total song duration: {total_duration_sec:.2f} seconds")
    
    # Calculate number of segments
    num_segments = int(total_samples / quarter_note_samples)
    print(f"Number of quarter note segments: {num_segments}")
    
    # Create output directory for this song
    song_name = audio_file.stem
    song_output_dir = output_dir / song_name
    song_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Divide and export segments
    print("\nExporting segments...")
    for i in range(num_segments):
        start_sample = i * quarter_note_samples
        end_sample = (i + 1) * quarter_note_samples
        
        # Extract segment
        segment = audio_data[start_sample:end_sample]
        
        # Create output filename with zero-padded index
        output_filename = f"{song_name}_quarter_note_{i+1:04d}.wav"
        output_path = song_output_dir / output_filename
        
        # Export segment
        sf.write(output_path, segment, sample_rate)
        
        if (i + 1) % 10 == 0:
            print(f"  Exported {i + 1}/{num_segments} segments...")
    
    # Handle remaining audio (partial quarter note at the end)
    remaining_start = num_segments * quarter_note_samples
    if remaining_start < total_samples:
        remaining_segment = audio_data[remaining_start:]
        output_filename = f"{song_name}_quarter_note_{num_segments+1:04d}_partial.wav"
        output_path = song_output_dir / output_filename
        sf.write(output_path, remaining_segment, sample_rate)
        print(f"  Exported 1 partial segment (remainder)")
    
    print(f"\n✓ Complete! Exported {num_segments + (1 if remaining_start < total_samples else 0)} segments")
    print(f"  Output directory: {song_output_dir}")


def main():
    """Main function to process audio files."""
    # Get directories
    script_dir = Path(__file__).parent
    input_dir = script_dir / "input"
    output_dir = script_dir / "output" / "audio"
    bpm_cache_file = input_dir / "bpm" / "bpm.json"
    
    # Clear output directory before running
    output_base = script_dir / "output"
    if output_base.exists():
        print(f"Clearing output directory...")
        shutil.rmtree(output_base)
    
    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all audio files from input directory (common formats)
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.aac', '*.ogg', '*.wma']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(ext))
    
    if not audio_files:
        print("No audio files found in ./input directory")
        print(f"Supported formats: {', '.join([e.replace('*', '') for e in audio_extensions])}")
        return
    
    # Use only the first file
    audio_file = audio_files[0]
    
    if len(audio_files) > 1:
        print(f"Warning: Found {len(audio_files)} audio files. Processing only: {audio_file.name}")
        print("Remove other files to process a different one.\n")
    
    # Get BPM (from cache or detect)
    bpm = get_bpm(audio_file, bpm_cache_file)
    
    # Process the file
    divide_song_into_quarter_notes(audio_file, bpm, output_dir)


if __name__ == "__main__":
    main()
