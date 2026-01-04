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
import re


def filename_to_kebab_case(filename: str) -> str:
    """Convert filename to kebab-case (remove extension, lowercase, replace spaces/special chars with hyphens).
    
    Args:
        filename: Original filename
        
    Returns:
        Kebab-case version without extension
    """
    # Remove extension
    name = Path(filename).stem
    # Convert to lowercase
    name = name.lower()
    # Replace spaces and underscores with hyphens
    name = re.sub(r'[\s_]+', '-', name)
    # Remove non-alphanumeric characters except hyphens
    name = re.sub(r'[^a-z0-9-]+', '', name)
    # Remove multiple consecutive hyphens
    name = re.sub(r'-+', '-', name)
    # Remove leading/trailing hyphens
    name = name.strip('-')
    return name


def load_song_config(config_file: Path) -> dict:
    """Load song configuration from JSON file.
    
    Args:
        config_file: Path to the song config JSON file
        
    Returns:
        Dictionary with song configuration (bpm, downbeat_offset, etc.)
    """
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_song_config(config_file: Path, config_data: dict) -> None:
    """Save song configuration to JSON file.
    
    Args:
        config_file: Path to the song config JSON file
        config_data: Dictionary with song configuration
    """
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)


def get_bpm(audio_file: Path) -> tuple[float, float]:
    """Get BPM and downbeat offset for an audio file, using config if available.
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        Tuple of (bpm, downbeat_offset) where downbeat_offset is in seconds
    """
    # Get config file path (same directory as audio file, same name with .json extension)
    config_file = audio_file.with_suffix('.json')
    
    # Load config
    config_data = load_song_config(config_file)
    
    # Check if BPM is in config
    if config_data and "bpm" in config_data:
        bpm = config_data["bpm"]
        downbeat_offset = config_data.get("downbeat_offset", 0.0)
        print(f"Using config from: {config_file.name}")
        print(f"BPM: {bpm:.2f}")
        if downbeat_offset > 0:
            print(f"Downbeat offset: {downbeat_offset:.3f} seconds")
        return bpm, downbeat_offset
    
    # Detect BPM
    print(f"Analyzing audio to detect BPM...")
    y, sr = librosa.load(str(audio_file))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # librosa returns tempo as numpy array, extract scalar
    if hasattr(tempo, '__iter__'):
        bpm = float(tempo[0])
    else:
        bpm = float(tempo)
    
    # Save to config (with default downbeat_offset of 0)
    config_data = {"bpm": bpm, "downbeat_offset": 0.0}
    save_song_config(config_file, config_data)
    print(f"Detected and saved BPM: {bpm:.2f}")
    print(f"Config saved to: {config_file}")
    print(f"Note: downbeat_offset set to 0.0. Edit the config file to adjust if needed.")
    
    return bpm, 0.0


def calculate_quarter_note_duration(bpm: float) -> float:
    """
    Calculate the duration of a quarter note in seconds.
    
    BPM = beats per minute (where beat = quarter note)
    Duration of one quarter note = 60 seconds / BPM
    """
    duration_seconds = 60.0 / bpm
    return duration_seconds


def divide_song_into_quarter_notes(audio_file: Path, bpm: float, downbeat_offset: float, output_dir: Path) -> None:
    """
    Divide an audio file into quarter note segments.
    
    Args:
        audio_file: Path to the input audio file (any format)
        bpm: Beats per minute of the song
        downbeat_offset: Number of seconds to skip at the start of the file
        output_dir: Directory to save the output segments
    """
    print(f"\nProcessing: {audio_file.name}")
    print(f"BPM: {bpm}")
    if downbeat_offset > 0:
        print(f"Downbeat offset: {downbeat_offset:.3f} seconds")
    
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
    
    # Apply downbeat offset - skip the specified number of seconds
    offset_samples = int(downbeat_offset * sample_rate)
    if offset_samples > 0:
        audio_data = audio_data[offset_samples:]
        total_samples = len(audio_data)
        print(f"Skipped {downbeat_offset:.3f} seconds ({offset_samples} samples)")
    
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
    
    # Clear output directory before running
    output_base = script_dir / "output"
    if output_base.exists():
        print(f"Clearing output directory...")
        shutil.rmtree(output_base)
    
    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all audio files from input directory (common formats) - search recursively
    audio_extensions = ['**/*.mp3', '**/*.wav', '**/*.flac', '**/*.m4a', '**/*.aac', '**/*.ogg', '**/*.wma']
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
    
    # Get BPM and downbeat offset (from config or detect)
    bpm, downbeat_offset = get_bpm(audio_file)
    
    # Process the file
    divide_song_into_quarter_notes(audio_file, bpm, downbeat_offset, output_dir)


if __name__ == "__main__":
    main()
