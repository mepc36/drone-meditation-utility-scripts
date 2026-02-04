#!/usr/bin/env python3

""" 
TODO:

CD:

cd /Users/martinconnor/Music/Music/Media.localized/Music/Unknown\ Artist/Unknown\ Album

DEFINITELY:
- make all samples have the exact same length
- make volume of panned samples equal to volume of centered samples
- find a way make a small subset of samples be a longer pause
- make centered samples more common
- add silences (just a few)
- make some samples show up only once.
- use a great number of samples, but have 4 or 5 endings rather than just 1.

BUGS:
- When a sample with the same name already exists in the unknown album on disk, the import done in script 3- will not overwrite it

MAYBE:
- add subset of rhythmically different samples
-- e.g., samples that repeat 1 sample multiple times
- add two longer samples that both can end the song
-- make them diametrical opposites to each other (e.g., make 1 "living" & the other "dying")
-- play with numerology (make one of them last for 3:33, make the other last for 6:66)
- add tinnitus to one (or many) samples?
- add rule that we must have at least 1 combination of any 2 samples that share the same word
-- e.g. -- "pain" by 50 Cent and "pain" by Eve
- create a throughline throughout the piece by using a recurring word, like "yo"

PROBABLY NOT:
- add silences??? (or is this unnecessary now??)
"""


"""
1-combine-samples-with-panning.py

Creates unique stereo combinations of audio samples with random panning.
Reads source files from ./input/audio/ and writes combined files to ./output/audio/final-sample-versions/
"""

import json
from pathlib import Path
import random
import subprocess
import numpy as np
import soundfile as sf


# -------------------------------------------------------------------
# CONFIG: Load from input/config/config.json
# -------------------------------------------------------------------
CONFIG_PATH = Path("./input/config/config.json")

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Extract config sections
combine_config = config["combine_samples_config"]
shared_config = config["shared_config"]

# Input/output locations
INPUT_AUDIO_DIR = Path("./input/audio")
OUTPUT_DIR = Path("./output/audio/final-sample-versions")

# Calculate beat length from BPM
BEAT_LENGTH_SECONDS = 60.0 / shared_config["bpm"]
SILENCE_LENGTH_SECONDS = shared_config["silent_samples_length_millisec"] / 1000.0
NUM_UNIQUE_SAMPLES = combine_config["num_unique_samples"]
MIN_SAMPLES_PER_COMBINATION = combine_config.get("min_samples_per_combination", 1)
MAX_SAMPLES_PER_COMBINATION = combine_config.get("max_samples_per_combination", 3)

# Parse panning_pattern_ratio (e.g., "2:1:1" means 2 center-only : 1 non-center-only : 1 stereo-pair)
panning_pattern_parts = [int(x) for x in combine_config.get("panning_pattern_ratio", "1:1:1").split(":")]
CENTER_ONLY_WEIGHT = panning_pattern_parts[0]  # 1 sample, center
NON_CENTER_ONLY_WEIGHT = panning_pattern_parts[1]  # 1 sample, hard left or right
STEREO_PAIR_WEIGHT = panning_pattern_parts[2]  # 2 samples, left + right

# Parse samples_to_silence_ratio (e.g., "4:1" means 4 samples : 1 silence)
silence_ratio_parts = [int(x) for x in combine_config.get("samples_to_silence_ratio", "1:0").split(":")]
SAMPLES_COUNT = silence_ratio_parts[0]
SILENCE_COUNT = silence_ratio_parts[1] if len(silence_ratio_parts) > 1 else 0


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def get_available_samples() -> dict[str, list[str]]:
    """Get all .wav files from input directory grouped by sound type.
    Expects filenames like: name_soundtype.N.wav (e.g., thinking_kick.1.wav)
    Returns dict mapping sound_type -> list of sample names (without extension).
    """
    if not INPUT_AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Input audio directory not found: {INPUT_AUDIO_DIR}\n"
            "Please create ./input/audio/ and place your audio files there."
        )
    
    wav_files = list(INPUT_AUDIO_DIR.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(
            f"No .wav files found in {INPUT_AUDIO_DIR}\n"
            "Please place audio files in ./input/audio/"
        )
    
    # Group samples by sound type
    samples_by_type = {}
    for f in wav_files:
        stem = f.stem  # filename without extension
        # Split on underscore and get [1] as sound type with suffix
        parts = stem.split('_')
        if len(parts) >= 2:
            # Extract sound type by removing numeric suffix after dot
            # e.g., "kick.1" -> "kick", "snare.6" -> "snare"
            sound_type_with_suffix = parts[1]
            sound_type = sound_type_with_suffix.split('.')[0]
            
            if sound_type not in samples_by_type:
                samples_by_type[sound_type] = []
            samples_by_type[sound_type].append(stem)
        else:
            # Fallback: if no underscore, use the whole name as sound type
            fallback_type = stem.split('.')[0]
            if fallback_type not in samples_by_type:
                samples_by_type[fallback_type] = []
            samples_by_type[fallback_type].append(stem)
    
    return samples_by_type


def ensure_input_files_exist() -> None:
    """Verify input audio directory and files exist."""
    # Just check that we have at least some samples
    samples_by_type = get_available_samples()
    if not samples_by_type:
        raise FileNotFoundError("No valid samples found")


def reset_output_dir() -> None:
    """Clear and recreate the output directory."""
    import shutil
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_audio(name: str) -> tuple[np.ndarray, int]:
    """Load audio file and return (audio_data, sample_rate)."""
    filepath = INPUT_AUDIO_DIR / f"{name}.wav"
    audio_data, sample_rate = sf.read(filepath)
    return audio_data, sample_rate


def apply_pan(audio: np.ndarray, pan_position: str) -> np.ndarray:
    """
    Apply equal-power panning to mono or stereo audio.
    pan_position: 'left', 'center', or 'right'
    Returns stereo audio with consistent perceived loudness.
    
    Uses equal-power panning curve to maintain constant power (L² + R² = 1):
    - Left: L=√2, R=0.0 (boosted by +3dB)
    - Center: L=1.0, R=1.0 (full volume)
    - Right: L=0.0, R=√2 (boosted by +3dB)
    """
    # Convert to mono if stereo
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    
    # Equal-power panning gains
    # Left/right are boosted by √2 (~1.414) to match power of center signals
    HARD_PAN_GAIN = np.sqrt(2)  # ≈ 1.4142
    
    # Create stereo output
    if pan_position == 'left':
        left = audio * HARD_PAN_GAIN
        right = np.zeros_like(audio)
    elif pan_position == 'right':
        left = np.zeros_like(audio)
        right = audio * HARD_PAN_GAIN
    else:  # center
        left = audio
        right = audio
    
    return np.column_stack([left, right])


def pad_to_length(audio: np.ndarray, sample_rate: int, target_length_seconds: float) -> np.ndarray:
    """
    Pad audio to exact target length with silence.
    If audio is longer, truncate it.
    """
    target_samples = int(target_length_seconds * sample_rate)
    current_samples = len(audio)
    
    if current_samples == target_samples:
        return audio
    elif current_samples > target_samples:
        return audio[:target_samples]
    else:
        padding_samples = target_samples - current_samples
        if audio.ndim == 1:
            silence = np.zeros(padding_samples)
        else:
            silence = np.zeros((padding_samples, audio.shape[1]))
        return np.concatenate([audio, silence])


def resample_audio(audio: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    """
    Resample audio from original_rate to target_rate using linear interpolation.
    Works with both mono and stereo audio.
    """
    if original_rate == target_rate:
        return audio
    
    # Calculate new length
    duration = len(audio) / original_rate
    new_length = int(duration * target_rate)
    
    # Create index arrays for interpolation
    old_indices = np.linspace(0, len(audio) - 1, len(audio))
    new_indices = np.linspace(0, len(audio) - 1, new_length)
    
    if audio.ndim == 1:
        # Mono
        return np.interp(new_indices, old_indices, audio)
    else:
        # Stereo - resample each channel
        resampled = np.zeros((new_length, audio.shape[1]))
        for ch in range(audio.shape[1]):
            resampled[:, ch] = np.interp(new_indices, old_indices, audio[:, ch])
        return resampled


def normalize_to_rms(audio: np.ndarray, target_rms: float = 0.15) -> np.ndarray:
    """
    Normalize audio to a target RMS (Root Mean Square) level.
    This ensures consistent perceived loudness across all samples.
    """
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms > 0:
        gain = target_rms / current_rms
        # Apply safety limiter to prevent clipping
        max_val = np.abs(audio * gain).max()
        if max_val > 0.95:
            gain = 0.95 / max_val * gain
        return audio * gain
    return audio


def create_combination(sample_names: list[str], pan_assignments: dict[str, str], 
                       sample_rate: int) -> np.ndarray:
    """
    Create a stereo mix of samples with their pan positions.
    Returns padded stereo audio.
    """
    # Load and pan each sample
    mixed = None
    
    for name in sample_names:
        audio, sr = load_audio(name)
        
        # Resample if needed
        if sr != sample_rate:
            audio = resample_audio(audio, sr, sample_rate)
        
        # Normalize individual sample to consistent loudness before mixing
        audio = normalize_to_rms(audio, target_rms=0.15)
        
        # Apply panning
        stereo = apply_pan(audio, pan_assignments[name])
        
        # Pad to beat length before mixing
        stereo = pad_to_length(stereo, sample_rate, BEAT_LENGTH_SECONDS)
        
        # Mix (sum)
        if mixed is None:
            mixed = stereo
        else:
            mixed = mixed + stereo
    
    # Normalize final mix to consistent RMS level
    mixed = normalize_to_rms(mixed, target_rms=0.15)
    
    return mixed


def generate_unique_combination(samples_by_type: dict[str, list[str]]) -> tuple[list[str], dict[str, str]]:
    """Generate a random unique combination using weighted panning patterns.
    Only combines samples with the same sound type.
    Only allows 3 specific patterns:
    1. 1 sample, center only
    2. 1 sample, hard left or hard right
    3. 2 samples, stereo pair (1 left + 1 right)
    Returns (sample_names, pan_assignments)
    """
    # First, randomly select a sound type
    sound_type = random.choice(list(samples_by_type.keys()))
    available_samples = samples_by_type[sound_type]
    
    # Build weighted pool of pattern types
    pattern_pool = (
        ['center_only'] * CENTER_ONLY_WEIGHT + 
        ['non_center_only'] * NON_CENTER_ONLY_WEIGHT + 
        ['stereo_pair'] * STEREO_PAIR_WEIGHT
    )
    
    # Select a pattern type
    pattern_type = random.choice(pattern_pool)
    
    if pattern_type == 'center_only':
        # Pattern 1: 1 sample, center
        sample_names = random.sample(available_samples, 1)
        pan_assignments = {sample_names[0]: 'center'}
    
    elif pattern_type == 'non_center_only':
        # Pattern 2: 1 sample, hard left OR hard right
        sample_names = random.sample(available_samples, 1)
        pan_position = random.choice(['left', 'right'])
        pan_assignments = {sample_names[0]: pan_position}
    
    else:  # stereo_pair
        # Pattern 3: 2 samples, one left and one right
        if len(available_samples) < 2:
            # Fallback to center if we don't have enough samples of this type
            sample_names = random.sample(available_samples, 1)
            pan_assignments = {sample_names[0]: 'center'}
        else:
            sample_names = random.sample(available_samples, 2)
            pan_assignments = {
                sample_names[0]: 'left',
                sample_names[1]: 'right'
            }
    
    return sample_names, pan_assignments


def format_filename(sample_names: list[str], pan_assignments: dict[str, str], index: int) -> str:
    """
    Format filename as: left_center_right_NNN.wav
    Only includes samples that are present, in pan order.
    """
    # Create list of (pan_position, sample_name) tuples
    pan_order = {'left': 0, 'center': 1, 'right': 2}
    samples_by_pan = [(pan_assignments[name], name) for name in sample_names]
    
    # Sort by pan position (left -> center -> right)
    samples_by_pan.sort(key=lambda x: pan_order[x[0]])
    
    # Extract sorted sample names
    sorted_names = [name.lower() for _, name in samples_by_pan]
    
    # Build filename
    name_part = "_".join(sorted_names)
    return f"{name_part}_{index:03d}.wav"


def create_silence_file(sample_rate: int, index: int) -> None:
    """Create a complete silence file using configured silence length."""
    silence_samples = int(SILENCE_LENGTH_SECONDS * sample_rate)
    # Create stereo silence
    silence_audio = np.zeros((silence_samples, 2))
    
    # Generate filename: silence_NNN.wav
    filename = f"silence_{index:03d}.wav"
    output_path = OUTPUT_DIR / filename
    
    # Save
    sf.write(output_path, silence_audio, sample_rate)


def import_folder_to_music(folder: Path) -> str:
    """Import entire folder to iTunes/Music in one operation."""
    script = f'''
tell application "Music"
    add (POSIX file "{folder.resolve()}/")
end tell
'''
    try:
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    print("\nCombine Samples with Random Panning\n")
    print(f"BPM: {shared_config['bpm']}")
    
    # Get all available samples from input directory, grouped by sound type
    samples_by_type = get_available_samples()
    total_samples = sum(len(samples) for samples in samples_by_type.values())
    print(f"Found {total_samples} sample(s) in {len(samples_by_type)} sound type(s):")
    for sound_type, samples in sorted(samples_by_type.items()):
        print(f"  {sound_type}: {len(samples)} samples")
        for sample in sorted(samples):
            print(f"    - {sample}")
    print()
    
    reset_output_dir()
    
    print("Generating unique combinations...\n")
    
    # Track combinations to ensure uniqueness
    seen_combinations = set()
    created_count = 0
    attempts = 0
    max_attempts = NUM_UNIQUE_SAMPLES * 100  # Prevent infinite loop
    
    # Get sample rate from first file
    first_sample_name = list(samples_by_type.values())[0][0]
    first_audio, sample_rate = load_audio(first_sample_name)
    
    while created_count < NUM_UNIQUE_SAMPLES and attempts < max_attempts:
        attempts += 1
        
        # Generate combination
        sample_names, pan_assignments = generate_unique_combination(samples_by_type)
        
        # Create unique key for this combination
        combo_key = tuple(sorted([f"{name}:{pan_assignments[name]}" for name in sample_names]))
        
        # Check if we've seen this exact combination before
        if combo_key in seen_combinations:
            continue
        
        seen_combinations.add(combo_key)
        created_count += 1
        
        # Create the audio
        combined_audio = create_combination(sample_names, pan_assignments, sample_rate)
        
        # Generate filename
        filename = format_filename(sample_names, pan_assignments, created_count)
        output_path = OUTPUT_DIR / filename
        
        # Save
        sf.write(output_path, combined_audio, sample_rate)
        
        if created_count % 10 == 0 or created_count == NUM_UNIQUE_SAMPLES:
            print(f"  Created {created_count}/{NUM_UNIQUE_SAMPLES} samples...")
    
    if created_count < NUM_UNIQUE_SAMPLES:
        print(f"\nWarning: Only created {created_count} unique combinations.")
        print("Consider reducing num_unique_samples in config.json")
    else:
        print(f"\nComplete! Created {created_count} unique samples.")
    
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    
    # Generate silence files based on samples_to_silence_ratio
    if SILENCE_COUNT > 0 and SAMPLES_COUNT > 0:
        # Calculate number of silence files needed
        num_silence_files = int((created_count / SAMPLES_COUNT) * SILENCE_COUNT)
        
        print(f"\nGenerating silence files...")
        print(f"  Ratio: {SAMPLES_COUNT}:{SILENCE_COUNT} (samples:silence)")
        print(f"  Creating {num_silence_files} silence files...")
        
        for i in range(1, num_silence_files + 1):
            create_silence_file(sample_rate, i)
            if i % 10 == 0 or i == num_silence_files:
                print(f"    Created {i}/{num_silence_files} silence files...")
        
        total_files = created_count + num_silence_files
        print(f"\nTotal files created: {total_files} ({created_count} samples + {num_silence_files} silence)")
    else:
        total_files = created_count
    
    # Import to iTunes
    print("\n" + "="*60)
    print("Importing entire folder to iTunes/Music...")
    print("This may take a moment...\n")
    
    result = import_folder_to_music(OUTPUT_DIR)
    
    print(f"Import complete!")
    print(f"  Imported folder: {OUTPUT_DIR.resolve()}")
    print(f"  Total files: {total_files}")
    print(f"\nNext: Run 3-import-duplicate-padded-samples-into-itunes-playlist.py\n")


if __name__ == "__main__":
    main()
