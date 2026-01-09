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
import argparse


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


def get_bpm(audio_file: Path, config_file: Path) -> tuple[float, float]:
    """Get BPM and downbeat offset for an audio file, using config if available.
    
    Args:
        audio_file: Path to the audio file
        config_file: Path to the config file
        
    Returns:
        Tuple of (bpm, downbeat_offset) where downbeat_offset is in seconds
    """
    
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


def apply_slow_then_sudden_fade_out(
    audio: np.ndarray,
    sr: int,
    fade_ms: float = 100.0,
    power: float = 4.5
) -> np.ndarray:
    """
    Applies a slow-then-sudden fade-out that ends at absolute silence.
    - audio: mono (samples,) or stereo (samples, channels)
    - fade_ms: fade duration in milliseconds
    - power: curve steepness (>1 = slower start, sharper drop)
    """
    fade_samples = int(sr * fade_ms / 1000.0)
    if fade_samples <= 1:
        return audio

    n = audio.shape[0]
    fade_samples = min(fade_samples, n)

    # Linear time 0 → 1
    t = np.linspace(0.0, 1.0, fade_samples, endpoint=True)

    # Curve: stays loud, then drops fast
    gain = 1.0 - np.power(t, power)

    # Enforce absolute silence at the final sample
    gain[-1] = 0.0

    out = audio.copy()
    if out.ndim == 1:
        out[-fade_samples:] *= gain
    else:
        out[-fade_samples:, :] *= gain[:, None]

    return out


def divide_song_into_quarter_notes(audio_file: Path, bpm: float, downbeat_offset: float, output_base_dir: Path, is_vocals: bool = False) -> None:
    """
    Divide an audio file into quarter note segments.
    
    Args:
        audio_file: Path to the input audio file (any format)
        bpm: Beats per minute of the song
        downbeat_offset: Number of seconds to skip at the start of the file
        output_base_dir: Base output directory
        is_vocals: Whether this is the vocals track
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
    
    # Get song name from directory structure (not from audio filename)
    # For main audio: ./input/{SONG_NAME}/audio/{filename}.mp3 -> use SONG_NAME
    # For acappella: ./input/{SONG_NAME}/demucs/acappella.wav -> use SONG_NAME
    if is_vocals:
        song_name = audio_file.parent.parent.name  # demucs parent's parent is SONG_NAME
    else:
        song_name = audio_file.parent.parent.name  # audio parent's parent is SONG_NAME
    
    # Determine output directory based on track type
    if is_vocals:
        song_output_dir = output_base_dir / song_name / "quarter-note-samples-acappella"
    else:
        song_output_dir = output_base_dir / song_name / "quarter-note-samples"
    song_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine filename prefix
    file_prefix = "acappella" if is_vocals else song_name
    
    # Divide and export segments
    print("\nExporting segments...")
    for i in range(num_segments):
        start_sample = i * quarter_note_samples
        end_sample = (i + 1) * quarter_note_samples
        
        # Extract segment
        segment = audio_data[start_sample:end_sample]
        
        # Apply fade-out
        segment = apply_slow_then_sudden_fade_out(
            segment,
            sample_rate,
            fade_ms=100.0,   # ~16% of a quarter note at 94 BPM
            power=4.5
        )
        
        # Calculate timestamp for this segment (in seconds)
        timestamp = i * quarter_note_sec
        
        # Create output filename with zero-padded index at start, then timestamp
        output_filename = f"{i+1:04d}_{timestamp:.6f}_{file_prefix}.wav"
        output_path = song_output_dir / output_filename
        
        # Export segment
        sf.write(output_path, segment, sample_rate)
        
        if (i + 1) % 10 == 0:
            print(f"  Exported {i + 1}/{num_segments} segments...")
    
    # Handle remaining audio (partial quarter note at the end)
    remaining_start = num_segments * quarter_note_samples
    if remaining_start < total_samples:
        remaining_segment = audio_data[remaining_start:]
        
        # Apply fade-out
        remaining_segment = apply_slow_then_sudden_fade_out(
            remaining_segment,
            sample_rate,
            fade_ms=100.0,
            power=4.5
        )
        
        # Calculate timestamp for the partial segment
        timestamp = num_segments * quarter_note_sec
        
        output_filename = f"{num_segments+1:04d}_{timestamp:.6f}_{file_prefix}_partial.wav"
        output_path = song_output_dir / output_filename
        sf.write(output_path, remaining_segment, sample_rate)
        print(f"  Exported 1 partial segment (remainder)")
    
    print(f"\n✓ Complete! Exported {num_segments + (1 if remaining_start < total_samples else 0)} segments")
    print(f"  Output directory: {song_output_dir}")


def main():
    """Main function to process audio files."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Divide audio files into quarter note samples')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force reprocessing even if output already exists')
    args = parser.parse_args()
    
    # Get directories
    script_dir = Path(__file__).parent
    input_dir = script_dir / "input"
    output_dir = script_dir / "output"
    
    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all song directories
    song_dirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name != '.DS_Store' and d.name != 'prompts']
    
    if not song_dirs:
        print("No song directories found in ./input")
        return
    
    print(f"\nFound {len(song_dirs)} song(s) to process")
    
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.aac', '*.ogg', '*.wma']
    
    # Process each song directory
    for song_dir in song_dirs:
        song_name = song_dir.name
        
        print(f"\n{'='*80}")
        print(f"Processing: {song_name}")
        print(f"{'='*80}")
        
        # Check if output already exists (skip check if force flag is used)
        main_samples_dir = output_dir / song_name / "quarter-note-samples"
        if not args.force and main_samples_dir.exists() and list(main_samples_dir.glob("*.wav")):
            print(f"⏭  Skipping - quarter note samples already exist at {main_samples_dir}")
            print(f"   Use -f or --force flag to reprocess")
            continue
        
        # Find audio file in this song's audio directory
        audio_subdir = song_dir / "audio"
        if not audio_subdir.exists():
            raise FileNotFoundError(f"Audio directory not found at {audio_subdir}")
        
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(audio_subdir.glob(ext))
        
        if not audio_files:
            raise FileNotFoundError(f"No audio files found in {audio_subdir}\nSupported formats: {', '.join(audio_extensions)}")
        
        if len(audio_files) > 1:
            files_list = '\n    '.join([f.name for f in audio_files])
            raise ValueError(f"Found {len(audio_files)} audio files in {audio_subdir}, but only 1 is allowed.\n  Found files:\n    {files_list}")
        
        audio_file = audio_files[0]
        print(f"✓ Found audio file: {audio_file.name}")
        
        # Get config file path
        config_file = song_dir / "config" / "config.json"
        
        # Get BPM and downbeat offset (from config or detect)
        bpm, downbeat_offset = get_bpm(audio_file, config_file)
        
        # Process the main audio file
        print(f"\nProcessing main audio file...")
        divide_song_into_quarter_notes(audio_file, bpm, downbeat_offset, output_dir, is_vocals=False)
        
        # Check if acappella.wav exists in the output demucs directory
        vocals_file = output_dir / song_name / "demucs" / "acappella.wav"
        
        if vocals_file.exists():
            print(f"\n\nProcessing acappella track...")
            divide_song_into_quarter_notes(vocals_file, bpm, downbeat_offset, output_dir, is_vocals=True)
        else:
            print(f"\n\nNote: acappella.wav not found at {vocals_file}")
            print(f"Run 1-separate-song-stems.py first to generate acappella track.")


if __name__ == "__main__":
    main()
