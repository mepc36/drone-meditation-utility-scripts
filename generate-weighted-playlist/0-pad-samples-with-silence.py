#!/usr/bin/env python3
"""
1-pad-samples-with-silence.py

Pads audio samples with silence to make them all the same length.
Reads source files from ./input/audio/ and writes padded files to ./output/audio/padded-audio-samples/
"""

import json
from pathlib import Path
import numpy as np
import soundfile as sf


# -------------------------------------------------------------------
# CONFIG: Load from input/config/config.json
# -------------------------------------------------------------------
CONFIG_PATH = Path("./input/config/config.json")

# Ensure input directory structure exists
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Extract config sections
pad_config = config["pad_samples_config"]
shared_config = config["shared_config"]

# Input/output locations
INPUT_AUDIO_DIR = Path("./input/audio")
OUTPUT_DIR = Path("./output/audio/padded-audio-samples")

# Desired length from config (in seconds)
# Calculate beat length from BPM: 60 seconds / BPM = seconds per beat
DESIRED_LENGTH_SECONDS = 60.0 / shared_config["bpm"]
LIVING_LENGTH_SECONDS = pad_config["living_sample_length_seconds"]

# Canonical file names
CANONICAL_FILES = [f"{name}.wav" for name in pad_config["canonical_files"]]
LIVING_FILE = f"{pad_config['living_file']}.wav"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def ensure_input_files_exist() -> None:
    """Verify input audio directory and files exist."""
    if not INPUT_AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Input audio directory not found: {INPUT_AUDIO_DIR}\n"
            "Please place your -ing audio files in ./input/audio/"
        )
    
    missing = []
    for filename in CANONICAL_FILES + [LIVING_FILE]:
        if not (INPUT_AUDIO_DIR / filename).exists():
            missing.append(filename)
    
    if missing:
        raise FileNotFoundError(
            f"Missing audio file(s) in {INPUT_AUDIO_DIR}:\n" + 
            "\n".join(f"  - {f}" for f in missing)
        )


def reset_output_dir() -> None:
    """Clear and recreate the output directory."""
    import shutil
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def pad_audio_with_silence(audio_data: np.ndarray, sample_rate: int, target_length_seconds: float) -> np.ndarray:
    """
    Adjust audio to EXACTLY match target length.
    Truncates if too long, pads with silence if too short.
    """
    current_length_seconds = len(audio_data) / sample_rate
    target_length_samples = int(target_length_seconds * sample_rate)
    current_length_samples = len(audio_data)
    
    if current_length_samples == target_length_samples:
        print(f"    Already exactly {current_length_seconds:.2f}s")
        return audio_data
    
    elif current_length_samples > target_length_samples:
        # Truncate
        print(f"    Current: {current_length_seconds:.2f}s, truncating to {target_length_seconds:.2f}s")
        return audio_data[:target_length_samples]
    
    else:
        # Pad with silence
        padding_samples = target_length_samples - current_length_samples
        
        # Create silence padding
        if audio_data.ndim == 1:
            # Mono
            silence = np.zeros(padding_samples)
        else:
            # Stereo or multi-channel
            silence = np.zeros((padding_samples, audio_data.shape[1]))
        
        padding_seconds = padding_samples / sample_rate
        print(f"    Current: {current_length_seconds:.2f}s, padding with {padding_seconds:.2f}s silence")
        
        # Concatenate audio with silence
        return np.concatenate([audio_data, silence])


def process_audio_file(input_path: Path, output_path: Path, target_length_seconds: float) -> None:
    """Load audio, pad with silence, and save to output."""
    print(f"  Processing {input_path.name}...")
    
    # Load audio
    audio_data, sample_rate = sf.read(input_path)
    
    # Pad with silence
    padded_audio = pad_audio_with_silence(audio_data, sample_rate, target_length_seconds)
    
    # Save
    sf.write(output_path, padded_audio, sample_rate)
    print(f"    Saved to {output_path.name}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    print("\nPad Audio Samples with Silence\n")
    print(f"Desired sample length: {DESIRED_LENGTH_SECONDS} seconds\n")
    
    ensure_input_files_exist()
    reset_output_dir()
    
    print("Processing audio files...\n")
    
    # Process all canonical files
    for filename in CANONICAL_FILES:
        input_path = INPUT_AUDIO_DIR / filename
        output_path = OUTPUT_DIR / filename
        process_audio_file(input_path, output_path, DESIRED_LENGTH_SECONDS)
    
    # Process Living file (with unique duration)
    input_path = INPUT_AUDIO_DIR / LIVING_FILE
    output_path = OUTPUT_DIR / LIVING_FILE
    process_audio_file(input_path, output_path, LIVING_LENGTH_SECONDS)
    
    # Create Silence file
    print(f"  Creating Silence.wav...")
    silence_path = OUTPUT_DIR / "Silence.wav"
    sample_rate = 44100  # Standard CD quality
    silence_samples = int(DESIRED_LENGTH_SECONDS * sample_rate)
    silence_audio = np.zeros(silence_samples)
    sf.write(silence_path, silence_audio, sample_rate)
    print(f"    Created {DESIRED_LENGTH_SECONDS:.2f}s of silence")
    print(f"    Saved to {silence_path.name}")
    
    print(f"\nPadding complete!")
    print(f"  Input: {INPUT_AUDIO_DIR.resolve()}")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print(f"  Canonical samples: {DESIRED_LENGTH_SECONDS} seconds")
    print(f"  Living sample: {LIVING_LENGTH_SECONDS} seconds\n")


if __name__ == "__main__":
    main()
