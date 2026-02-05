#!/usr/bin/env python3

r""" 
TODO:

CD:

cd /Users/martinconnor/Music/Music/Media.localized/Music/Unknown\ Artist/Unknown\ Album

DEFINITELY:
- set rules to prevent using 2 of the same (or similar) panning positions??
- always make noncentered samples fully loud?
- add a third volume tier???
- make samples more similar lengths (e.g., politics is too long)
- make volume of panned samples equal to volume of centered samples
- use a large number of samples, but have 4 or 5 endings rather than just 1.
-- IDEA: make them diametrical opposites to each other (e.g., make 1 "living" & the other "dying")
-- IDEA: play with numerology (make one of them last for 3:33, make the other last for 6:66)
- Add solo piano/hi hat/bass kick/string/etc. sounds
- add endings

MAYBE:
- add rule that we must have at least 1 combination of any 2 samples that share the same word
-- e.g. -- "pain" by 50 Cent and "pain" by Eve
- add subset of rhythmically different samples
-- e.g., samples that repeat 1 sample multiple times
- add tinnitus to one (or many) samples?

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

# Input/output locations
INPUT_AUDIO_DIR = Path("./input/audio")
OUTPUT_DIR = Path("./output/audio/final-sample-versions")

# Calculate beat length from BPM
BEAT_LENGTH_SECONDS = 60.0 / config["bpm"]
NUM_UNIQUE_SAMPLES = config["num_unique_samples"]

# Parse center_to_noncenter_to_dualpan_ratio (e.g., "2:1:1" means 2 center-only : 1 non-center-only : 1 dualpan)
panning_pattern_parts = [int(x) for x in config.get("center_to_noncenter_to_dualpan_ratio", "1:1:1").split(":")]
CENTER_ONLY_WEIGHT = panning_pattern_parts[0]  # 1 sample, center
NON_CENTER_ONLY_WEIGHT = panning_pattern_parts[1]  # 1 sample, hard left or right
DUALPAN_WEIGHT = panning_pattern_parts[2]  # 2 samples, left + right

# Parse samples_to_silence_ratio (e.g., "4:1" means 4 samples : 1 silence)
silence_ratio_parts = [int(x) for x in config.get("samples_to_silence_ratio", "1:0").split(":")]
SAMPLES_COUNT = silence_ratio_parts[0]
SILENCE_COUNT = silence_ratio_parts[1] if len(silence_ratio_parts) > 1 else 0

# Parse silence lengths and their weights (e.g., "2000:10000" with "8:1")
silence_lengths_ms = [int(x) for x in config.get("silence_lengths_millisec", "2000").split(":")]
SILENCE_LENGTHS_SECONDS = [ms / 1000.0 for ms in silence_lengths_ms]
silence_weights = [int(x) for x in config.get("silence_lengths_ratio", "1").split(":")]
SILENCE_LENGTH_WEIGHTS = silence_weights

# Parse padded centered samples config
PADDED_CENTERED_LENGTH_MS = config.get("padded_centered_samples_length_millisec", 2000)
PADDED_CENTERED_LENGTH_SECONDS = PADDED_CENTERED_LENGTH_MS / 1000.0
PADDED_CENTERED_PERCENT = config.get("padded_centered_samples_percent", 0.0) / 100.0

# Parse double-timed samples config
DOUBLE_TIMED_PERCENT = config.get("double_timed_samples_percent", 0.0) / 100.0
DOUBLE_TIMED_BEAT_LENGTH_SECONDS = BEAT_LENGTH_SECONDS / 2.0  # Half the beat length

# Panning range constants for non-center samples
# Left side uses negative values, right side uses positive values
NON_CENTER_PAN_MIN = 0.35  # Minimum distance from center (applies to both sides)
NON_CENTER_PAN_MAX = 1.0  # Maximum distance from center (applies to both sides)

# Number of segments for statistical balancing of panning distribution
PAN_DISTRIBUTION_SEGMENTS = 10  # Divide panning range into this many segments for tracking

# Validate that lengths and weights match
if len(SILENCE_LENGTHS_SECONDS) != len(SILENCE_LENGTH_WEIGHTS):
    raise ValueError(
        f"Error: silence_lengths_millisec and silence_lengths_ratio must have the same number of values.\n"
        f"Got {len(SILENCE_LENGTHS_SECONDS)} lengths but {len(SILENCE_LENGTH_WEIGHTS)} weights."
    )

# Parse volume levels and their weights (e.g., "0:-5:-10" with "1:4:1")
volume_levels_db = [float(x) for x in config.get("loud_medium_soft_values", "0").split(":")]
volume_weights = [int(x) for x in config.get("loud_medium_soft_ratio", "1").split(":")]

# Validate that volume levels and weights match
if len(volume_levels_db) != len(volume_weights):
    raise ValueError(
        f"Error: loud_medium_soft_values and loud_medium_soft_ratio must have the same number of values.\n"
        f"Got {len(volume_levels_db)} volume levels but {len(volume_weights)} weights."
    )


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


def apply_pan(audio: np.ndarray, pan_position) -> np.ndarray:
    """
    Apply equal-power panning to mono or stereo audio.
    pan_position: 'center', or numeric value from -1.0 (hard left) to 1.0 (hard right)
    Returns stereo audio with consistent perceived loudness.
    
    Uses equal-power panning curve to maintain constant power (L² + R² = 1):
    - -1.0 (hard left): L=√2, R=0.0 (boosted by +3dB)
    - 0.0 (center): L=1.0, R=1.0 (full volume)
    - 1.0 (hard right): L=0.0, R=√2 (boosted by +3dB)
    - Intermediate values use cosine/sine panning curve
    """
    # Convert to mono if stereo
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    
    # Equal-power panning gains
    # Left/right are boosted by √2 (~1.414) to match power of center signals
    HARD_PAN_GAIN = np.sqrt(2)  # ≈ 1.4142
    
    # Create stereo output
    if pan_position == 'center':
        left = audio
        right = audio
    else:
        # Numeric pan position: use equal-power panning curve
        # Convert pan (-1 to 1) to angle (0 to π/2)
        # -1 = hard left (0°), 0 = center (45°), 1 = hard right (90°)
        pan_value = float(pan_position)
        angle = (pan_value + 1.0) * np.pi / 4.0  # Maps [-1,1] to [0, π/2]
        
        # Equal-power panning using sin/cos
        left_gain = np.cos(angle) * HARD_PAN_GAIN
        right_gain = np.sin(angle) * HARD_PAN_GAIN
        
        left = audio * left_gain
        right = audio * right_gain
    
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


def apply_volume_db(audio: np.ndarray, db_reduction: float) -> np.ndarray:
    """
    Apply a dB reduction to audio.
    db_reduction: negative number for reduction, 0 for no change
    Formula: gain = 10^(dB/20)
    """
    if db_reduction == 0:
        return audio
    gain = 10 ** (db_reduction / 20)
    return audio * gain


def select_volume_level_from_pool(volume_pool: list[int]) -> tuple[float, int]:
    """
    Select a volume level from the remaining quota pool.
    Returns (db_reduction, level_index)
    """
    if not volume_pool:
        # Fallback: if pool is empty, use the first volume level
        return volume_levels_db[0], 0
    
    # Select random index from pool
    selected_idx = random.choice(volume_pool)
    return volume_levels_db[selected_idx], selected_idx


def create_combination(sample_names: list[str], pan_assignments: dict[str, str], 
                       sample_rate: int, volume_pool: list[int], use_padded_length: bool = False, use_double_time: bool = False, is_centered: bool = False) -> tuple[np.ndarray, int]:
    """
    Create a stereo mix of samples with their pan positions.
    Returns (padded stereo audio, volume_level_index).
    
    Args:
        sample_names: List of sample names to combine
        pan_assignments: Dictionary mapping sample names to pan positions
        sample_rate: Sample rate for the output
        volume_pool: Pool of remaining volume level indices to choose from (quota-based)
        use_padded_length: If True, pad to PADDED_CENTERED_LENGTH_SECONDS instead of BEAT_LENGTH_SECONDS
        use_double_time: If True, pad to DOUBLE_TIMED_BEAT_LENGTH_SECONDS (half beat length)
        is_centered: If True, sample is centered and should use centered volume pool
    """
    # Load and pan each sample
    mixed = None
    
    # Determine target length based on flags
    if use_double_time:
        target_length = DOUBLE_TIMED_BEAT_LENGTH_SECONDS
    elif use_padded_length:
        target_length = PADDED_CENTERED_LENGTH_SECONDS
    else:
        target_length = BEAT_LENGTH_SECONDS
    
    for name in sample_names:
        audio, sr = load_audio(name)
        
        # Resample if needed
        if sr != sample_rate:
            audio = resample_audio(audio, sr, sample_rate)
        
        # Normalize individual sample to consistent loudness before mixing
        audio = normalize_to_rms(audio, target_rms=0.15)
        
        # Apply panning
        stereo = apply_pan(audio, pan_assignments[name])
        
        # Pad to target length
        stereo = pad_to_length(stereo, sample_rate, target_length)
        
        # Mix (sum)
        if mixed is None:
            mixed = stereo
        else:
            mixed = mixed + stereo
    
    # Normalize final mix to consistent RMS level
    mixed = normalize_to_rms(mixed, target_rms=0.15)
    
    # Select and apply volume level from quota pool
    db_reduction, volume_idx = select_volume_level_from_pool(volume_pool)
    mixed = apply_volume_db(mixed, db_reduction)
    
    return mixed, volume_idx


def select_balanced_pan_position(side: str, pan_history: list[float]) -> float:
    """
    Select a panning position using statistical balancing to ensure even distribution.
    
    Args:
        side: Either 'left' or 'right'
        pan_history: List of previously used panning values (absolute values, 0.35 to 1.0)
    
    Returns:
        Pan position (negative for left, positive for right)
    """
    # Define the range for this side (always work with positive values)
    range_min = NON_CENTER_PAN_MIN
    range_max = NON_CENTER_PAN_MAX
    
    if not pan_history:
        # No history yet, pick random position
        position = random.uniform(range_min, range_max)
    else:
        # Divide range into segments
        segment_size = (range_max - range_min) / PAN_DISTRIBUTION_SEGMENTS
        
        # Count how many samples fall into each segment
        segment_counts = [0] * PAN_DISTRIBUTION_SEGMENTS
        for pan_val in pan_history:
            # Convert to segment index
            segment_idx = int((pan_val - range_min) / segment_size)
            # Clamp to valid range (handle edge case where pan_val == range_max)
            segment_idx = min(segment_idx, PAN_DISTRIBUTION_SEGMENTS - 1)
            segment_counts[segment_idx] += 1
        
        # Find minimum count (most sparse segments)
        min_count = min(segment_counts)
        
        # Get indices of all segments with minimum count
        sparse_segments = [i for i, count in enumerate(segment_counts) if count == min_count]
        
        # Randomly select one of the sparse segments
        selected_segment = random.choice(sparse_segments)
        
        # Generate random position within that segment
        segment_start = range_min + (selected_segment * segment_size)
        segment_end = segment_start + segment_size
        position = random.uniform(segment_start, segment_end)
    
    # Apply sign based on side
    return -position if side == 'left' else position


def generate_unique_combination(samples_by_type: dict[str, list[str]], used_once_samples: set[str], 
                               center_quota: int, left_quota: int, right_quota: int, dualpan_quota: int,
                               left_pan_history: list[float], right_pan_history: list[float]) -> tuple[list[str], dict[str, str]]:
    """Generate a random unique combination using weighted panning patterns.
    Only combines samples with the same sound type.
    Only allows 3 specific patterns:
    1. 1 sample, center only
    2. 1 sample, left or right (quota-based for 50/50 distribution)
    3. 2 samples, stereo pair (1 left + 1 right)
    
    ONCE samples (subset of SOLO): isolated, centered, appear once
    SOLO samples: isolated, can pan left/center/right, can repeat
    Regular samples: can be combined, use all patterns
    
    Returns (sample_names, pan_assignments)
    """
    # First, randomly select a sound type
    sound_type = random.choice(list(samples_by_type.keys()))
    available_samples = samples_by_type[sound_type]
    
    # For ONCE samples, filter out already-used ones
    if sound_type.lower() == 'once':
        available_samples = [s for s in available_samples if s not in used_once_samples]
        # If no ONCE samples left, return None to signal retry
        if not available_samples:
            return None, None
    
    # Build pattern pool based on sound type and remaining quotas
    if sound_type.lower() == 'once':
        # ONCE: always isolated and centered
        pattern_pool = ['center_only']
    elif sound_type.lower() == 'solo':
        # SOLO: isolated but can pan left/center/right (exclude stereo_pair)
        # Use quotas to ensure exact ratio and left/right balance
        pattern_pool = (
            ['center_only'] * center_quota + 
            ['left_only'] * left_quota +
            ['right_only'] * right_quota
        )
    else:
        # Regular samples: use all patterns with quotas
        pattern_pool = (
            ['center_only'] * center_quota + 
            ['left_only'] * left_quota +
            ['right_only'] * right_quota +
            ['stereo_pair'] * dualpan_quota
        )
    
    # If pool is empty (all quotas exhausted), return None
    if not pattern_pool:
        return None, None
    
    # Select a pattern type
    pattern_type = random.choice(pattern_pool)
    
    if pattern_type == 'center_only':
        # Pattern 1: 1 sample, center
        sample_names = random.sample(available_samples, 1)
        pan_assignments = {sample_names[0]: 'center'}
    
    elif pattern_type == 'left_only':
        # Pattern 2a: 1 sample, panned left
        # Use statistical balancing to ensure even distribution
        sample_names = random.sample(available_samples, 1)
        pan_position = select_balanced_pan_position('left', left_pan_history)
        pan_assignments = {sample_names[0]: pan_position}
    
    elif pattern_type == 'right_only':
        # Pattern 2b: 1 sample, panned right
        # Use statistical balancing to ensure even distribution
        sample_names = random.sample(available_samples, 1)
        pan_position = select_balanced_pan_position('right', right_pan_history)
        pan_assignments = {sample_names[0]: pan_position}
    
    else:  # stereo_pair
        # Pattern 3: 2 samples, one left and one right
        if len(available_samples) < 2:
            # Fallback to center if we don't have enough samples of this type
            sample_names = random.sample(available_samples, 1)
            pan_assignments = {sample_names[0]: 'center'}
        else:
            sample_names = random.sample(available_samples, 2)
            # For stereo pairs, use hard panning for clear separation
            pan_assignments = {
                sample_names[0]: -1.0,
                sample_names[1]: 1.0
            }
    
    return sample_names, pan_assignments


def format_filename(sample_names: list[str], pan_assignments: dict[str, str], index: int) -> str:
    """
    Format filename as: left_center_right_NNN.wav
    Only includes samples that are present, in pan order.
    """
    # Create list of (pan_position, sample_name) tuples
    samples_by_pan = []
    for name in sample_names:
        pan = pan_assignments[name]
        # Convert pan to numeric value for sorting
        if pan == 'center':
            pan_value = 0.0
        elif pan == 'left':
            pan_value = -1.0
        elif pan == 'right':
            pan_value = 1.0
        else:
            # Already numeric
            pan_value = float(pan)
        samples_by_pan.append((pan_value, name))
    
    # Sort by pan position (left -> center -> right)
    samples_by_pan.sort(key=lambda x: x[0])
    
    # Extract sorted sample names
    sorted_names = [name.lower() for _, name in samples_by_pan]
    
    # Build filename
    name_part = "_".join(sorted_names)
    return f"{name_part}_{index:03d}.wav"


def create_silence_file(sample_rate: int, length_seconds: float, index: int) -> None:
    """Create a complete silence file with specified length."""
    silence_samples = int(length_seconds * sample_rate)
    # Create stereo silence
    silence_audio = np.zeros((silence_samples, 2))
    
    # Generate filename: silence_LENGTHms_NNN.wav (e.g., silence_2000ms_001.wav)
    length_ms = int(length_seconds * 1000)
    filename = f"silence_{length_ms}ms_{index:03d}.wav"
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
    print(f"BPM: {config['bpm']}")
    
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
    # Count ONCE samples and calculate adjusted ratio
    once_samples_count = len(samples_by_type.get('once', []))
    print(f"Found {once_samples_count} ONCE samples (always centered)\n")
    
    # Calculate how many samples will use each pattern based on ratio
    total_ratio_parts = CENTER_ONLY_WEIGHT + NON_CENTER_ONLY_WEIGHT + DUALPAN_WEIGHT
    center_quota = int(NUM_UNIQUE_SAMPLES * CENTER_ONLY_WEIGHT / total_ratio_parts)
    noncenter_quota = int(NUM_UNIQUE_SAMPLES * NON_CENTER_ONLY_WEIGHT / total_ratio_parts)
    dualpan_quota = int(NUM_UNIQUE_SAMPLES * DUALPAN_WEIGHT / total_ratio_parts)
    
    # Split noncenter quota evenly between left and right for 50/50 distribution
    left_quota = noncenter_quota // 2
    right_quota = noncenter_quota - left_quota  # Gives right any remainder for odd numbers
    
    # Validate: ONCE samples cannot exceed center quota
    if once_samples_count > center_quota:
        raise ValueError(
            f"Error: Too many ONCE samples ({once_samples_count}) for center quota ({center_quota}).\n"
            f"Config ratio '{config.get('center_to_noncenter_to_dualpan_ratio')}' allocates {center_quota} center slots "
            f"out of {NUM_UNIQUE_SAMPLES} total samples.\n"
            f"Either:\n"
            f"  1. Reduce number of ONCE samples to {center_quota} or fewer, or\n"
            f"  2. Increase center ratio in center_to_noncenter_to_dualpan_ratio, or\n"
            f"  3. Increase num_unique_samples to {int(once_samples_count * total_ratio_parts / CENTER_ONLY_WEIGHT)} or more"
        )
    
    # Adjust quotas: ONCE samples consume center quota
    remaining_center_quota = max(0, center_quota - once_samples_count)
    
    # Calculate adjusted weights for non-ONCE samples
    # If we've used up all center quota, set center weight to 0
    adjusted_center_weight = remaining_center_quota
    adjusted_left_weight = left_quota
    adjusted_right_weight = right_quota
    adjusted_dualpan_weight = dualpan_quota
    
    print("Generating unique combinations...\n")
    
    # Track combinations to ensure uniqueness
    seen_combinations = set()
    used_once_samples = set()  # Track ONCE samples that have been used (can only appear once)
    created_count = 0
    attempts = 0
    max_attempts = NUM_UNIQUE_SAMPLES * 100  # Prevent infinite loop
    
    # Quota-based generation: track how many of each pattern we still need
    center_quota_remaining = adjusted_center_weight
    left_quota_remaining = adjusted_left_weight
    right_quota_remaining = adjusted_right_weight
    dualpan_quota_remaining = adjusted_dualpan_weight
    
    # Track panning distribution
    center_count = 0
    left_count = 0
    right_count = 0
    dualpan_count = 0
    
    # Track pan position histories for statistical balancing (store absolute values)
    left_pan_history = []  # Stores positive values (0.35 to 1.0)
    right_pan_history = []  # Stores positive values (0.35 to 1.0)
    
    # Track volume level distribution
    volume_counts = [0] * len(volume_levels_db)
    
    # Initialize volume quota pool (quota-based selection)
    # Calculate how many samples should use each volume level
    total_volume_weight = sum(volume_weights)
    volume_quotas = []
    for weight in volume_weights:
        quota = int(NUM_UNIQUE_SAMPLES * weight / total_volume_weight)
        volume_quotas.append(quota)
    
    # Calculate volume quotas for different panning types
    centered_volume_quota = center_quota  # All centered samples use first volume level
    non_centered_volume_quota = noncenter_quota  # All non-centered samples use last volume level
    last_volume_idx = len(volume_levels_db) - 1
    
    # Adjust volume quotas to account for centered and non-centered samples
    adjusted_volume_quotas = volume_quotas.copy()
    adjusted_volume_quotas[0] = max(0, volume_quotas[0] - centered_volume_quota)  # Remove centered from first
    adjusted_volume_quotas[last_volume_idx] = max(0, volume_quotas[last_volume_idx] - non_centered_volume_quota)  # Remove non-centered from last
    
    # Build separate volume pools for centered, non-centered, and dualpan samples
    centered_volume_pool = [0] * centered_volume_quota  # All indices point to first volume level
    non_centered_volume_pool = [last_volume_idx] * non_centered_volume_quota  # All indices point to last volume level
    
    dualpan_volume_pool = []
    for idx, quota in enumerate(adjusted_volume_quotas):
        dualpan_volume_pool.extend([idx] * quota)
    
    # Get sample rate from first file
    first_sample_name = list(samples_by_type.values())[0][0]
    first_audio, sample_rate = load_audio(first_sample_name)
    
    # Calculate how many centered samples should be padded
    num_padded_centered = int(center_quota * PADDED_CENTERED_PERCENT)
    padded_centered_count = 0
    centered_created_count = 0
    
    # Calculate how many samples should be double-timed
    num_double_timed = int(NUM_UNIQUE_SAMPLES * DOUBLE_TIMED_PERCENT)
    double_timed_count = 0
    
    while created_count < NUM_UNIQUE_SAMPLES and attempts < max_attempts:
        attempts += 1
        
        # Generate combination with remaining quotas
        sample_names, pan_assignments = generate_unique_combination(
            samples_by_type, used_once_samples,
            center_quota_remaining, left_quota_remaining, right_quota_remaining, dualpan_quota_remaining,
            left_pan_history, right_pan_history
        )
        
        # Check if combination generation failed (e.g., no more ONCE samples available)
        if sample_names is None:
            continue
        
        # Create unique key for this combination
        combo_key = tuple(sorted([f"{name}:{pan_assignments[name]}" for name in sample_names]))
        
        # Check if we've seen this exact combination before
        if combo_key in seen_combinations:
            continue
        
        seen_combinations.add(combo_key)
        
        # Mark ONCE samples as used (they can only appear once)
        for name in sample_names:
            # Check if this sample is from the ONCE sound type
            parts = name.split('_')
            if len(parts) >= 2:
                sound_type_with_suffix = parts[1]
                sound_type = sound_type_with_suffix.split('.')[0]
                if sound_type.lower() == 'once':
                    used_once_samples.add(name)
        
        created_count += 1
        
        # Track panning pattern and update quotas
        pan_positions = list(pan_assignments.values())
        is_centered = False
        is_non_centered = False
        is_dualpan = False
        use_padded_length = False
        use_double_time = False
        
        # Decide if this sample should be double-timed (applies to all panning types)
        if double_timed_count < num_double_timed:
            use_double_time = True
            double_timed_count += 1
        
        if len(pan_positions) == 1:
            if pan_positions[0] == 'center':
                is_centered = True
                center_count += 1
                centered_created_count += 1
                center_quota_remaining = max(0, center_quota_remaining - 1)
                
                # Decide if this centered sample should be padded (only if NOT double-timed)
                if not use_double_time and padded_centered_count < num_padded_centered:
                    use_padded_length = True
                    padded_centered_count += 1
            else:  # left or right (numeric pan value)
                is_non_centered = True
                pan_value = float(pan_positions[0])
                if pan_value < 0:  # left side
                    left_count += 1
                    left_quota_remaining = max(0, left_quota_remaining - 1)
                    # Track absolute value in history
                    left_pan_history.append(abs(pan_value))
                else:  # right side
                    right_count += 1
                    right_quota_remaining = max(0, right_quota_remaining - 1)
                    # Track absolute value in history
                    right_pan_history.append(abs(pan_value))
        else:  # 2 samples (stereo pair)
            is_dualpan = True
            dualpan_count += 1
            dualpan_quota_remaining = max(0, dualpan_quota_remaining - 1)
        
        # Create the audio with appropriate volume pool
        if is_centered:
            combined_audio, volume_idx = create_combination(sample_names, pan_assignments, sample_rate, centered_volume_pool, use_padded_length, use_double_time, is_centered=True)
            # Remove used volume index from centered pool
            if volume_idx in centered_volume_pool:
                centered_volume_pool.remove(volume_idx)
        elif is_non_centered:
            combined_audio, volume_idx = create_combination(sample_names, pan_assignments, sample_rate, non_centered_volume_pool, use_padded_length, use_double_time, is_centered=False)
            # Remove used volume index from non-centered pool
            if volume_idx in non_centered_volume_pool:
                non_centered_volume_pool.remove(volume_idx)
        else:  # is_dualpan
            combined_audio, volume_idx = create_combination(sample_names, pan_assignments, sample_rate, dualpan_volume_pool, use_padded_length, use_double_time, is_centered=False)
            # Remove used volume index from dualpan pool
            if volume_idx in dualpan_volume_pool:
                dualpan_volume_pool.remove(volume_idx)
            if volume_idx in non_centered_volume_pool:
                non_centered_volume_pool.remove(volume_idx)
        
        # Track volume level
        volume_counts[volume_idx] += 1
        
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
    
    # Display panning distribution
    from math import gcd
    from functools import reduce
    
    noncenter_count = left_count + right_count
    
    def gcd_multiple(*args):
        """Calculate GCD of multiple numbers."""
        return reduce(gcd, args)
    
    # Calculate GCD for ratio display
    counts = [c for c in [center_count, noncenter_count, dualpan_count] if c > 0]
    if len(counts) > 1:
        ratio_gcd = gcd_multiple(*counts)
    else:
        ratio_gcd = 1
    
    realized_ratio = f"{center_count//ratio_gcd}:{noncenter_count//ratio_gcd}:{dualpan_count//ratio_gcd}"
    
    # Calculate percentage differences from target
    target_center_pct = (center_quota / created_count) * 100
    target_noncenter_pct = (noncenter_quota / created_count) * 100
    target_dualpan_pct = (dualpan_quota / created_count) * 100
    
    actual_center_pct = (center_count / created_count) * 100
    actual_noncenter_pct = (noncenter_count / created_count) * 100
    actual_dualpan_pct = (dualpan_count / created_count) * 100
    
    center_diff = actual_center_pct - target_center_pct
    noncenter_diff = actual_noncenter_pct - target_noncenter_pct
    dualpan_diff = actual_dualpan_pct - target_dualpan_pct
    
    deviation_ratio = f"{center_diff:+.3f}%:{noncenter_diff:+.3f}%:{dualpan_diff:+.3f}%"
    
    print(f"\nPanning Distribution:")
    print(f"  Config ratio: {config.get('center_to_noncenter_to_dualpan_ratio', '1:1:1')}")
    print(f"  Realized ratio: {realized_ratio}")
    
    # Calculate perfect ratio scaled to match realized first value
    config_ratio_parts = [int(x) for x in config.get('center_to_noncenter_to_dualpan_ratio', '1:1:1').split(':')]
    realized_parts = [int(x) for x in realized_ratio.split(':')]
    scale_factor = realized_parts[0] / config_ratio_parts[0] if config_ratio_parts[0] != 0 else 1
    perfect_ratio_parts = [int(x * scale_factor) for x in config_ratio_parts]
    perfect_ratio = ':'.join([str(x) for x in perfect_ratio_parts])
    print(f"  Perfect ratio: {perfect_ratio}")
    
    # Calculate differential (perfect - realized): shows what's needed to reach perfect
    differential = [perfect_ratio_parts[i] - realized_parts[i] for i in range(len(realized_parts))]
    differential_str = ':'.join([f"{'+' if d > 0 else ''}{d}" for d in differential])
    print(f"  Differential: {differential_str}")
    
    # Display left/right distribution for non-center samples
    if noncenter_count > 0:
        left_pct = (left_count / noncenter_count) * 100
        right_pct = (right_count / noncenter_count) * 100
        print(f"\nLeft/Right Distribution (non-center samples only):")
        print(f"  Target: 50.0% left : 50.0% right")
        print(f"  Realized: {left_count}:{right_count} = {left_pct:.1f}% : {right_pct:.1f}%")
        diff = left_count - right_count
        print(f"  Differential: {diff:+d} (left - right)")
    
    # Display padded centered samples info
    if padded_centered_count > 0:
        padded_pct = (padded_centered_count / center_count) * 100 if center_count > 0 else 0
        target_padded_pct = PADDED_CENTERED_PERCENT * 100
        print(f"\nPadded Centered Samples:")
        print(f"  Target: {target_padded_pct:.1f}% of centered samples")
        print(f"  Realized: {padded_centered_count}/{center_count} = {padded_pct:.1f}%")
        print(f"  Length: {PADDED_CENTERED_LENGTH_MS}ms ({PADDED_CENTERED_LENGTH_SECONDS:.2f}s)")
    
    # Display double-timed samples info
    if double_timed_count > 0:
        double_timed_pct = (double_timed_count / created_count) * 100 if created_count > 0 else 0
        target_double_timed_pct = DOUBLE_TIMED_PERCENT * 100
        print(f"\nDouble-Timed Samples:")
        print(f"  Target: {target_double_timed_pct:.1f}% of all samples")
        print(f"  Realized: {double_timed_count}/{created_count} = {double_timed_pct:.1f}%")
        print(f"  BPM: {config['bpm'] * 2} (double-time)")
        print(f"  Length: {DOUBLE_TIMED_BEAT_LENGTH_SECONDS:.3f}s (half of {BEAT_LENGTH_SECONDS:.3f}s)")
    
    # Display volume distribution
    print(f"\nVolume Distribution:")
    print(f"  Config ratio: {config.get('loud_medium_soft_ratio', '1')}")
    print(f"  Config values: {config.get('loud_medium_soft_values', '0')} dB")
    
    # Calculate realized ratio
    from math import gcd
    from functools import reduce
    
    def gcd_multiple(*args):
        """Calculate GCD of multiple numbers."""
        return reduce(gcd, args)
    
    non_zero_counts = [c for c in volume_counts if c > 0]
    if len(non_zero_counts) > 1:
        vol_ratio_gcd = gcd_multiple(*non_zero_counts)
    elif len(non_zero_counts) == 1:
        vol_ratio_gcd = non_zero_counts[0]
    else:
        vol_ratio_gcd = 1
    
    realized_vol_ratio = ':'.join([str(c // vol_ratio_gcd) for c in volume_counts])
    print(f"  Realized ratio: {realized_vol_ratio}")
    
    # Display counts for each level
    for idx, (db_val, count) in enumerate(zip(volume_levels_db, volume_counts)):
        pct = (count / created_count) * 100 if created_count > 0 else 0
        print(f"    {db_val:+.1f} dB: {count} samples ({pct:.1f}%)")
    
    # Generate silence files based on samples_to_silence_ratio
    if SILENCE_COUNT > 0 and SAMPLES_COUNT > 0:
        # Calculate total number of silence files needed
        num_silence_files = int((created_count / SAMPLES_COUNT) * SILENCE_COUNT)
        
        print(f"\nGenerating silence files...")
        print(f"  Ratio: {SAMPLES_COUNT}:{SILENCE_COUNT} (samples:silence)")
        print(f"  Total silence files to create: {num_silence_files}")
        
        # Calculate distribution of silence files across different lengths based on weights
        total_weight = sum(SILENCE_LENGTH_WEIGHTS)
        silence_counts_by_length = []
        remaining_files = num_silence_files
        
        for i, weight in enumerate(SILENCE_LENGTH_WEIGHTS):
            if i == len(SILENCE_LENGTH_WEIGHTS) - 1:
                # Last length gets remaining files to ensure we hit exact total
                count = remaining_files
            else:
                count = int((num_silence_files * weight) / total_weight)
                remaining_files -= count
            silence_counts_by_length.append(count)
        
        # Display distribution
        print(f"  Silence length distribution:")
        for length_sec, count in zip(SILENCE_LENGTHS_SECONDS, silence_counts_by_length):
            length_ms = int(length_sec * 1000)
            print(f"    {length_ms}ms: {count} files")
        
        # Create silence files with appropriate lengths
        file_counter = 1
        for length_idx, (length_sec, count) in enumerate(zip(SILENCE_LENGTHS_SECONDS, silence_counts_by_length)):
            length_ms = int(length_sec * 1000)
            for i in range(count):
                create_silence_file(sample_rate, length_sec, file_counter)
                file_counter += 1
                if file_counter % 10 == 1 or file_counter == num_silence_files + 1:
                    print(f"    Created {file_counter - 1}/{num_silence_files} silence files...", end="\r")
        
        print(f"    Created {num_silence_files}/{num_silence_files} silence files...")
        
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
    print(f"\nNext: Run 2-import-duplicate-padded-samples-into-itunes-playlist.py\n")


if __name__ == "__main__":
    main()
