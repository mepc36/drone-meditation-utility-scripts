#!/usr/bin/env python3

r""" 
TODO:

CD:

cd /Users/martinconnor/Music/Music/Media.localized/Music/Unknown\ Artist/Unknown\ Album

DEFINITELY:
2. Add code to use every sample at least once; throw error if num_unique_samples < num_samples
3. Even out volume difference between panned samples and centered samples
4. Answer this question: what do we do with samples whose length is greater than an 8th note (thus causing them to be cut off prematurely)?
5. Remove volume fades added at python layer (since we already add them at Logic Pro layer)???
6. Add "short" sample group
7. Add "beautiful" sample group
8. Remove "beautiful" samples from "funny" sample group.
9. Resample for short samples
10. Slow the BPM down to 25?

MAYBE:
1. add samples that represent endings?
2 add rule that we must have at least 1 combination of any 2 samples that share the same word
-- e.g. -- "pain" by 50 Cent and "pain" by Eve
3. add subset of rhythmically different samples
-- e.g., samples that repeat 1 sample multiple times
4. Add solo piano/hi hat/bass kick/string/etc. sounds
5. get rid of the super long silent sample? (or shorten it to 30 or 40 seconds)?
6. make the samples have similar lengths to each other 
-- e.g., "politics" is so long that it gets cut off by the next sample at fast BPMs
7. Add multiple instances of the same string quartet snippets?
8. use a large number of samples, but have 4 or 5 endings rather than just 1.
-- IDEA: make them diametrical opposites to each other (e.g., make 1 "living" & the other "dying")
-- IDEA: play with numerology (make one of them last for 3:33, make the other last for 6:66)
9. Use bass guitar strings from "Buck 'Em" as 8th notes?
10. Add code so that samples from same song appear together???

"""


"""
1-combine-samples-with-panning.py

Creates unique stereo combinations of audio samples with random panning.
Reads source files from ./input/audio/ and writes combined files to ./output/audio/final-sample-versions/
"""

import json
from collections import deque
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
NUM_UNIQUE_SAMPLES = config["num_unique_samples"]  # Total files (audio + silence)

# Parse center_to_noncenter_to_dualpan_percents (e.g., "16:58:26" means 16% center-only : 58% non-center-only : 26% dualpan)
panning_pattern_parts = [int(x) for x in config.get("center_to_noncenter_to_dualpan_percents", "33:33:34").split(":")]
if sum(panning_pattern_parts) != 100:
    raise ValueError(
        f"Error: center_to_noncenter_to_dualpan_percents percentages must sum to exactly 100.\n"
        f"Got: {':'.join(map(str, panning_pattern_parts))} = {sum(panning_pattern_parts)}"
    )
CENTER_ONLY_WEIGHT = panning_pattern_parts[0]  # Percentage, center
NON_CENTER_ONLY_WEIGHT = panning_pattern_parts[1]  # Percentage, hard left or right
DUALPAN_WEIGHT = panning_pattern_parts[2]  # Percentage, left + right

# Parse samples_to_silence_percents (e.g., "87:13" means 87% samples : 13% silence)
silence_ratio_parts = [int(x) for x in config.get("samples_to_silence_percents", "100:0").split(":")]
if sum(silence_ratio_parts) != 100:
    raise ValueError(
        f"Error: samples_to_silence_percents percentages must sum to exactly 100.\n"
        f"Got: {':'.join(map(str, silence_ratio_parts))} = {sum(silence_ratio_parts)}"
    )
SAMPLES_PERCENT = silence_ratio_parts[0]  # Percentage of samples
SILENCE_PERCENT = silence_ratio_parts[1] if len(silence_ratio_parts) > 1 else 0  # Percentage of silence

# Calculate how many audio samples vs silence files based on ratio
NUM_AUDIO_SAMPLES = int(NUM_UNIQUE_SAMPLES * SAMPLES_PERCENT / 100) if SAMPLES_PERCENT > 0 else NUM_UNIQUE_SAMPLES
NUM_SILENCE_FILES = NUM_UNIQUE_SAMPLES - NUM_AUDIO_SAMPLES

# Parse silence lengths and their percentages (e.g., "2000:10000" with "86:14" means 86% are 2000ms, 14% are 10000ms)
silence_lengths_ms = [int(x) for x in config.get("silence_lengths_millisec", "2000").split(":")]
SILENCE_LENGTHS_SECONDS = [ms / 1000.0 for ms in silence_lengths_ms]
silence_percentages = [int(x) for x in config.get("silence_lengths_percents", "100").split(":")]
if sum(silence_percentages) != 100:
    raise ValueError(
        f"Error: silence_lengths_percents percentages must sum to exactly 100.\n"
        f"Got: {':'.join(map(str, silence_percentages))} = {sum(silence_percentages)}"
    )
SILENCE_LENGTH_PERCENTAGES = silence_percentages

# Parse padded centered samples config
PADDED_CENTERED_LENGTH_MS = config.get("padded_centered_samples_length_millisec", 2000)
PADDED_CENTERED_LENGTH_SECONDS = PADDED_CENTERED_LENGTH_MS / 1000.0
PADDED_CENTERED_PERCENT = config.get("padded_centered_samples_percent", 0.0) / 100.0

# Parse double-timed samples config
EIGHTH_NOTE_SAMPLES_PERCENT = config.get("eighth_note_samples_percent", 0.0) / 100.0
DOUBLE_TIMED_BEAT_LENGTH_SECONDS = BEAT_LENGTH_SECONDS / 2.0  # Half the beat length

# Parse multi-beat duration samples config
FOUR_BEAT_DURATION_PERCENT = config.get("four_beat_duration_percent", 0.0) / 100.0
TWO_BEAT_DURATION_PERCENT = config.get("two_beat_duration_percent", 0.0) / 100.0
FOUR_BEAT_LENGTH_SECONDS = BEAT_LENGTH_SECONDS * 4.0
TWO_BEAT_LENGTH_SECONDS = BEAT_LENGTH_SECONDS * 2.0

# Panning range constants for non-center samples
# Left side uses negative values, right side uses positive values
NON_CENTER_PAN_MIN = 0.35  # Minimum distance from center (applies to both sides)
NON_CENTER_PAN_MAX = 1.0  # Maximum distance from center (applies to both sides)

# Number of segments for statistical balancing of panning distribution
PAN_DISTRIBUTION_SEGMENTS = 10  # Divide panning range into this many segments for tracking

# Validate that lengths and percentages match
if len(SILENCE_LENGTHS_SECONDS) != len(SILENCE_LENGTH_PERCENTAGES):
    raise ValueError(
        f"Error: silence_lengths_millisec and silence_lengths_percents must have the same number of values.\n"
        f"Got {len(SILENCE_LENGTHS_SECONDS)} lengths but {len(SILENCE_LENGTH_PERCENTAGES)} percentages."
    )

# Parse volume levels and their percentages (e.g., "0:-5:-10" with "20:52:28" means 20% loud, 52% medium, 28% soft)
volume_levels_db = [float(x) for x in config.get("loud_medium_soft_values", "0").split(":")]
volume_percentages = [int(x) for x in config.get("loud_medium_soft_percents", "100").split(":")]

if sum(volume_percentages) != 100:
    raise ValueError(
        f"Error: loud_medium_soft_percents percentages must sum to exactly 100.\n"
        f"Got: {':'.join(map(str, volume_percentages))} = {sum(volume_percentages)}"
    )

# Validate that volume levels and percentages match
if len(volume_levels_db) != len(volume_percentages):
    raise ValueError(
        f"Error: loud_medium_soft_values and loud_medium_soft_percents must have the same number of values.\n"
        f"Got {len(volume_levels_db)} volume levels but {len(volume_percentages)} percentages."
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
        # Split on underscore and get [2] as sound type with suffix
        # e.g., "let-me-blow-ya-mind_oov_stab.1" -> parts[2] = "stab.1" -> "stab"
        parts = stem.split('_')
        if len(parts) >= 3:
            # Extract sound type by removing numeric suffix after dot
            # e.g., "stab.1" -> "stab", "kick.1" -> "kick"
            sound_type_with_suffix = parts[2]
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


def get_sound_type(sample_name: str) -> str:
    """Extract the sound type from a sample filename.
    e.g., 'let-me-blow-ya-mind_oov_stab.1' -> 'stab'
    """
    parts = sample_name.split('_')
    if len(parts) >= 3:
        return parts[2].split('.')[0].lower()
    return sample_name.split('.')[0].lower()


# -------------------------------------------------------------------
# Round-robin queue helpers — ensures each sample is used as evenly
# as possible across all output files.
# -------------------------------------------------------------------

def build_sample_round_robin(samples_by_type: dict[str, list[str]]) -> tuple[deque, list[str]]:
    """Build an initial shuffled round-robin queue from all samples.
    Returns (queue, all_sample_names).
    """
    all_samples = [s for samples in samples_by_type.values() for s in samples]
    shuffled = random.sample(all_samples, len(all_samples))
    return deque(shuffled), all_samples


def refill_round_robin_queue(sample_queue: deque, all_sample_names: list[str],
                              used_once_samples: set[str]) -> None:
    """Append a new shuffled round to the queue, excluding exhausted ONCE samples."""
    eligible = [s for s in all_sample_names
                if not (get_sound_type(s) == 'once' and s in used_once_samples)]
    random.shuffle(eligible)
    sample_queue.extend(eligible)


def dequeue_next_sample(sample_queue: deque, all_sample_names: list[str],
                         used_once_samples: set[str]) -> str | None:
    """Pop the next usable sample from the round-robin queue.
    Skips already-used ONCE samples; refills when the queue is empty.
    """
    total = len(all_sample_names)
    for _ in range(total * 3):  # upper-bound guard
        if not sample_queue:
            refill_round_robin_queue(sample_queue, all_sample_names, used_once_samples)
        if not sample_queue:
            return None  # nothing left at all
        sample = sample_queue.popleft()
        # Skip ONCE samples that have already been used
        if get_sound_type(sample) == 'once' and sample in used_once_samples:
            continue
        return sample
    return None


def dequeue_partner_sample(sample_queue: deque, all_sample_names: list[str],
                            sound_type: str, exclude_name: str,
                            used_once_samples: set[str]) -> str | None:
    """Find and remove the next sample of *sound_type* from the queue for a stereo pair.
    Scans the existing queue first; falls back to any eligible sample of that type
    (choosing least-recently used via a secondary search) when not found in queue.
    """
    # Search current queue
    for i in range(len(sample_queue)):
        s = sample_queue[i]
        if s == exclude_name:
            continue
        if get_sound_type(s) != sound_type:
            continue
        if get_sound_type(s) == 'once' and s in used_once_samples:
            continue
        del sample_queue[i]
        return s

    # Not found in queue — pick any eligible sample of that type
    eligible = [s for s in all_sample_names
                if s != exclude_name
                and get_sound_type(s) == sound_type
                and not (get_sound_type(s) == 'once' and s in used_once_samples)]
    if eligible:
        return random.choice(eligible)
    return None


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
                       sample_rate: int, volume_pool: list[int], use_padded_length: bool = False, use_double_time: bool = False, 
                       use_four_beat: bool = False, use_two_beat: bool = False, is_centered: bool = False) -> tuple[np.ndarray, int]:
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
        use_four_beat: If True, pad to FOUR_BEAT_LENGTH_SECONDS (4 beats)
        use_two_beat: If True, pad to TWO_BEAT_LENGTH_SECONDS (2 beats)
        is_centered: If True, sample is centered and should use centered volume pool
    """
    # Load and pan each sample
    mixed = None
    
    # Determine target length based on flags (priority order)
    if use_four_beat:
        target_length = FOUR_BEAT_LENGTH_SECONDS
    elif use_two_beat:
        target_length = TWO_BEAT_LENGTH_SECONDS
    elif use_double_time:
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
                               left_pan_history: list[float], right_pan_history: list[float],
                               sample_round_robin: deque, all_sample_names: list[str]) -> tuple[list[str], dict[str, str]]:
    """Generate a combination using round-robin sample selection + quota-based panning.

    Sample selection: samples are drawn from a shuffled round-robin queue so that
    every input file is used roughly the same number of times.  Once all samples
    have been used once the queue is refilled in a new random order, giving the
    next "round".

    Panning patterns (unchanged):
    1. 1 sample, center only
    2. 1 sample, left or right (quota-based for 50/50 distribution)
    3. 2 samples, stereo pair (1 left + 1 right)

    ONCE samples: isolated, centered, appear exactly once.
    SOLO samples: isolated, can pan left/center/right, no stereo pairs.
    Regular samples: can be combined, use all patterns.

    Returns (sample_names, pan_assignments)
    """
    # ----------------------------------------------------------------
    # Step 1: dequeue the primary sample from the round-robin queue
    # ----------------------------------------------------------------
    primary = dequeue_next_sample(sample_round_robin, all_sample_names, used_once_samples)
    if primary is None:
        return None, None

    sound_type = get_sound_type(primary)

    # ----------------------------------------------------------------
    # Step 2: choose panning pattern based on type + remaining quotas
    # ----------------------------------------------------------------
    if sound_type == 'once':
        # ONCE: always isolated and centered
        pattern_pool = ['center_only']
    elif sound_type == 'solo':
        # SOLO: isolated but can pan left/center/right (no stereo pairs)
        pattern_pool = (
            ['center_only'] * center_quota +
            ['left_only'] * left_quota +
            ['right_only'] * right_quota
        )
    else:
        # Regular: all patterns allowed
        pattern_pool = (
            ['center_only'] * center_quota +
            ['left_only'] * left_quota +
            ['right_only'] * right_quota +
            ['stereo_pair'] * dualpan_quota
        )

    if not pattern_pool:
        return None, None

    pattern_type = random.choice(pattern_pool)

    # ----------------------------------------------------------------
    # Step 3: build sample list and pan assignments
    # ----------------------------------------------------------------
    if pattern_type == 'center_only':
        sample_names = [primary]
        pan_assignments = {primary: 'center'}

    elif pattern_type == 'left_only':
        sample_names = [primary]
        pan_position = select_balanced_pan_position('left', left_pan_history)
        pan_assignments = {primary: pan_position}

    elif pattern_type == 'right_only':
        sample_names = [primary]
        pan_position = select_balanced_pan_position('right', right_pan_history)
        pan_assignments = {primary: pan_position}

    else:  # stereo_pair
        partner = dequeue_partner_sample(
            sample_round_robin, all_sample_names,
            sound_type, primary, used_once_samples
        )
        if partner is None:
            # Can't form a pair — fall back to center
            sample_names = [primary]
            pan_assignments = {primary: 'center'}
        else:
            sample_names = [primary, partner]
            pan_assignments = {primary: -1.0, partner: 1.0}

    return sample_names, pan_assignments


def format_filename(sample_names: list[str], pan_assignments: dict[str, str], volume_db: float, index: int) -> str:
    """
    Format filename as: left_center_right_vol_X_pan_Y_NNN.wav
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
    
    # Calculate average pan position for filename
    avg_pan = sum(p for p, _ in samples_by_pan) / len(samples_by_pan)
    
    # Format volume (remove decimal if it's a whole number, use abs value)
    vol_str = f"{abs(volume_db):.0f}" if volume_db == int(volume_db) else f"{abs(volume_db):.1f}"
    
    # Format panning (use absolute value, rounded to 1 decimal)
    pan_str = f"{abs(avg_pan):.1f}"
    
    # Build filename
    name_part = "_".join(sorted_names)
    return f"{name_part}_vol_{vol_str}_pan_{pan_str}_{index:03d}.wav"


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
    
    # Calculate how many samples will use each pattern based on percentages
    center_quota = int(NUM_AUDIO_SAMPLES * CENTER_ONLY_WEIGHT / 100)
    noncenter_quota = int(NUM_AUDIO_SAMPLES * NON_CENTER_ONLY_WEIGHT / 100)
    dualpan_quota = int(NUM_AUDIO_SAMPLES * DUALPAN_WEIGHT / 100)

    # Cap center_quota to the number of unique audio files (each center combo is 1 file = 1 unique slot)
    if center_quota > total_samples:
        center_overflow = center_quota - total_samples
        center_quota = total_samples
        # Redistribute overflow proportionally to noncenter and dualpan
        non_center_total_weight = NON_CENTER_ONLY_WEIGHT + DUALPAN_WEIGHT
        if non_center_total_weight > 0:
            noncenter_quota += int(center_overflow * NON_CENTER_ONLY_WEIGHT / non_center_total_weight)
            dualpan_quota += center_overflow - int(center_overflow * NON_CENTER_ONLY_WEIGHT / non_center_total_weight)
        else:
            dualpan_quota += center_overflow
        print(f"⚠️  Center quota capped at {total_samples} (number of unique files). Overflow redistributed to noncenter/dualpan.\n")

    # Handle case where num_audio_samples is very small and all quotas round to 0
    # Distribute remaining samples proportionally
    total_allocated = center_quota + noncenter_quota + dualpan_quota
    remaining_samples = NUM_AUDIO_SAMPLES - total_allocated
    if remaining_samples > 0:
        # Allocate remaining samples based on weights
        weights = [CENTER_ONLY_WEIGHT, NON_CENTER_ONLY_WEIGHT, DUALPAN_WEIGHT]
        max_weight_idx = weights.index(max(weights))
        if max_weight_idx == 0:
            center_quota += remaining_samples
        elif max_weight_idx == 1:
            noncenter_quota += remaining_samples
        else:
            dualpan_quota += remaining_samples
    
    # Split noncenter quota evenly between left and right for 50/50 distribution
    left_quota = noncenter_quota // 2
    right_quota = noncenter_quota - left_quota  # Gives right any remainder for odd numbers
    
    # Validate: ONCE samples cannot exceed center quota
    if once_samples_count > center_quota:
        raise ValueError(
            f"Error: Too many ONCE samples ({once_samples_count}) for center quota ({center_quota}).\n"
            f"Config ratio '{config.get('center_to_noncenter_to_dualpan_percents')}' allocates {center_quota} center slots "
            f"out of {NUM_AUDIO_SAMPLES} audio samples.\n"
            f"Either:\n"
            f"  1. Reduce number of ONCE samples to {center_quota} or fewer, or\n"
            f"  2. Increase center ratio in center_to_noncenter_to_dualpan_percents, or\n"
            f"  3. Increase num_unique_samples to {int(once_samples_count * 100 / (SAMPLES_PERCENT * CENTER_ONLY_WEIGHT / 100))} or more"
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
    
    # Build round-robin queue so every sample is used as evenly as possible.
    # Samples are drawn in a shuffled order; when all have been used once the
    # queue is automatically refilled for the next round (in a new random order).
    sample_round_robin, all_sample_names = build_sample_round_robin(samples_by_type)
    sample_usage_count: dict[str, int] = {s: 0 for s in all_sample_names}
    
    # Track combinations to ensure uniqueness
    seen_combinations = set()
    used_once_samples = set()  # Track ONCE samples that have been used (can only appear once)
    created_count = 0
    attempts = 0
    max_attempts = NUM_AUDIO_SAMPLES * 100  # Prevent infinite loop
    
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
    # Calculate how many samples should use each volume level based on percentages
    volume_quotas = []
    for pct in volume_percentages:
        quota = int(NUM_AUDIO_SAMPLES * pct / 100)
        volume_quotas.append(quota)
    
    # Build shared volume pool for all samples (all panning types can have any volume)
    volume_pool = []
    for idx, quota in enumerate(volume_quotas):
        volume_pool.extend([idx] * quota)
    
    # Get sample rate from first file
    first_sample_name = list(samples_by_type.values())[0][0]
    first_audio, sample_rate = load_audio(first_sample_name)
    
    # Calculate how many centered samples should be padded
    num_padded_centered = int(center_quota * PADDED_CENTERED_PERCENT)
    padded_centered_count = 0
    centered_created_count = 0
    
    # Calculate how many samples should be double-timed
    num_double_timed = int(NUM_AUDIO_SAMPLES * EIGHTH_NOTE_SAMPLES_PERCENT)
    double_timed_count = 0
    
    # Calculate how many samples should have 4-beat and 2-beat durations
    num_four_beat = int(NUM_AUDIO_SAMPLES * FOUR_BEAT_DURATION_PERCENT)
    num_two_beat = int(NUM_AUDIO_SAMPLES * TWO_BEAT_DURATION_PERCENT)
    four_beat_count = 0
    two_beat_count = 0
    
    while created_count < NUM_AUDIO_SAMPLES and attempts < max_attempts:
        attempts += 1
        
        # Generate combination with remaining quotas
        sample_names, pan_assignments = generate_unique_combination(
            samples_by_type, used_once_samples,
            center_quota_remaining, left_quota_remaining, right_quota_remaining, dualpan_quota_remaining,
            left_pan_history, right_pan_history,
            sample_round_robin, all_sample_names
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
        
        # Track per-sample usage for the round-robin fairness report
        for name in sample_names:
            sample_usage_count[name] = sample_usage_count.get(name, 0) + 1
        
        # Mark ONCE samples as used (they can only appear once)
        for name in sample_names:
            # Check if this sample is from the ONCE sound type
            parts = name.split('_')
            if len(parts) >= 3:
                sound_type_with_suffix = parts[2]
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
        use_four_beat = False
        use_two_beat = False
        
        # Decide beat duration (priority: 4-beat > 2-beat > double-time)
        # These are mutually exclusive
        if four_beat_count < num_four_beat:
            use_four_beat = True
            four_beat_count += 1
        elif two_beat_count < num_two_beat:
            use_two_beat = True
            two_beat_count += 1
        elif double_timed_count < num_double_timed:
            use_double_time = True
            double_timed_count += 1
        
        if len(pan_positions) == 1:
            if pan_positions[0] == 'center':
                is_centered = True
                center_count += 1
                centered_created_count += 1
                center_quota_remaining = max(0, center_quota_remaining - 1)
                
                # Decide if this centered sample should be padded (only if NOT using any other duration)
                if not use_four_beat and not use_two_beat and not use_double_time and padded_centered_count < num_padded_centered:
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
        
        # Create the audio using shared volume pool (all panning types can have any volume)
        if is_centered:
            combined_audio, volume_idx = create_combination(sample_names, pan_assignments, sample_rate, volume_pool, use_padded_length, use_double_time, use_four_beat, use_two_beat, is_centered=True)
        elif is_non_centered:
            combined_audio, volume_idx = create_combination(sample_names, pan_assignments, sample_rate, volume_pool, use_padded_length, use_double_time, use_four_beat, use_two_beat, is_centered=False)
        else:  # is_dualpan
            combined_audio, volume_idx = create_combination(sample_names, pan_assignments, sample_rate, volume_pool, use_padded_length, use_double_time, use_four_beat, use_two_beat, is_centered=False)
        
        # Remove used volume index from shared pool
        if volume_idx in volume_pool:
            volume_pool.remove(volume_idx)
        
        # Track volume level
        volume_counts[volume_idx] += 1
        volume_db = volume_levels_db[volume_idx]
        
        # Generate filename
        filename = format_filename(sample_names, pan_assignments, volume_db, created_count)
        output_path = OUTPUT_DIR / filename
        
        # Save
        sf.write(output_path, combined_audio, sample_rate)
        
        if created_count % 10 == 0 or created_count == NUM_AUDIO_SAMPLES:
            print(f"  Created {created_count}/{NUM_AUDIO_SAMPLES} samples...")
    
    if created_count < NUM_AUDIO_SAMPLES:
        print(f"\nWarning: Only created {created_count} audio samples (expected {NUM_AUDIO_SAMPLES}).")
        print("Consider reducing num_unique_samples or samples_to_silence_percents in config.json")
    else:
        print(f"\nComplete! Created {created_count} audio samples.")
    
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    
    # Display sample usage distribution (round-robin fairness report)
    usage_values = list(sample_usage_count.values())
    if usage_values:
        min_uses = min(usage_values)
        max_uses = max(usage_values)
        avg_uses = sum(usage_values) / len(usage_values)
        print(f"\nSample Usage Distribution (round-robin):")
        print(f"  Total input samples: {len(all_sample_names)}")
        print(f"  Min uses: {min_uses}  Max uses: {max_uses}  Avg: {avg_uses:.2f}")
        if max_uses - min_uses <= 1:
            print(f"  ✓ Perfectly even — all samples used {min_uses}–{max_uses} times")
        else:
            outliers = [s for s, c in sample_usage_count.items() if c == max_uses]
            print(f"  ⚠  Spread of {max_uses - min_uses} ({len(outliers)} sample(s) at max)")
    
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
    if created_count > 0:
        target_center_pct = (center_quota / created_count) * 100
        target_noncenter_pct = (noncenter_quota / created_count) * 100
        target_dualpan_pct = (dualpan_quota / created_count) * 100
        
        actual_center_pct = (center_count / created_count) * 100
        actual_noncenter_pct = (noncenter_count / created_count) * 100
        actual_dualpan_pct = (dualpan_count / created_count) * 100
        
        center_diff = actual_center_pct - target_center_pct
        noncenter_diff = actual_noncenter_pct - target_noncenter_pct
        dualpan_diff = actual_dualpan_pct - target_dualpan_pct
    else:
        target_center_pct = target_noncenter_pct = target_dualpan_pct = 0
        actual_center_pct = actual_noncenter_pct = actual_dualpan_pct = 0
        center_diff = noncenter_diff = dualpan_diff = 0
    
    deviation_ratio = f"{center_diff:+.3f}%:{noncenter_diff:+.3f}%:{dualpan_diff:+.3f}%"
    
    print(f"\nPanning Distribution:")
    print(f"  Config ratio: {config.get('center_to_noncenter_to_dualpan_percents', '33:33:34')}")
    print(f"  Realized ratio: {realized_ratio}")
    
    # Calculate perfect ratio scaled to match realized first value
    config_ratio_parts = [int(x) for x in config.get('center_to_noncenter_to_dualpan_percents', '33:33:34').split(':')]
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
        target_double_timed_pct = EIGHTH_NOTE_SAMPLES_PERCENT * 100
        print(f"\nDouble-Timed Samples:")
        print(f"  Target: {target_double_timed_pct:.1f}% of all samples")
        print(f"  Realized: {double_timed_count}/{created_count} = {double_timed_pct:.1f}%")
        print(f"  BPM: {config['bpm'] * 2} (double-time)")
        print(f"  Length: {DOUBLE_TIMED_BEAT_LENGTH_SECONDS:.3f}s (half of {BEAT_LENGTH_SECONDS:.3f}s)")
    
    # Display 4-beat duration samples info
    if four_beat_count > 0:
        four_beat_pct = (four_beat_count / created_count) * 100 if created_count > 0 else 0
        target_four_beat_pct = FOUR_BEAT_DURATION_PERCENT * 100
        print(f"\n4-Beat Duration Samples:")
        print(f"  Target: {target_four_beat_pct:.1f}% of all samples")
        print(f"  Realized: {four_beat_count}/{created_count} = {four_beat_pct:.1f}%")
        print(f"  Length: {FOUR_BEAT_LENGTH_SECONDS:.3f}s (4 beats at {config['bpm']} BPM)")
    
    # Display 2-beat duration samples info
    if two_beat_count > 0:
        two_beat_pct = (two_beat_count / created_count) * 100 if created_count > 0 else 0
        target_two_beat_pct = TWO_BEAT_DURATION_PERCENT * 100
        print(f"\n2-Beat Duration Samples:")
        print(f"  Target: {target_two_beat_pct:.1f}% of all samples")
        print(f"  Realized: {two_beat_count}/{created_count} = {two_beat_pct:.1f}%")
        print(f"  Length: {TWO_BEAT_LENGTH_SECONDS:.3f}s (2 beats at {config['bpm']} BPM)")
    
    # Display volume distribution
    print(f"\nVolume Distribution:")
    print(f"  Config ratio: {config.get('loud_medium_soft_percents', '100')}")
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
    
    # Generate silence files based on samples_to_silence_percents
    if SILENCE_PERCENT > 0 and NUM_SILENCE_FILES > 0:
        print(f"\nGenerating silence files...")
        print(f"  Ratio: {SAMPLES_PERCENT}:{SILENCE_PERCENT} (samples:silence percentages)")
        print(f"  Total silence files to create: {NUM_SILENCE_FILES}")
        
        # Calculate distribution of silence files across different lengths based on percentages
        silence_counts_by_length = []
        remaining_files = NUM_SILENCE_FILES
        
        for i, pct in enumerate(SILENCE_LENGTH_PERCENTAGES):
            if i == len(SILENCE_LENGTH_PERCENTAGES) - 1:
                # Last length gets remaining files to ensure we hit exact total
                count = remaining_files
            else:
                count = int(NUM_SILENCE_FILES * pct / 100)
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
                if file_counter % 10 == 1 or file_counter == NUM_SILENCE_FILES + 1:
                    print(f"    Created {file_counter - 1}/{NUM_SILENCE_FILES} silence files...", end="\r")
        
        print(f"    Created {NUM_SILENCE_FILES}/{NUM_SILENCE_FILES} silence files...")
        
        total_files_created = created_count + NUM_SILENCE_FILES
        print(f"\nTotal files created: {total_files_created} ({created_count} samples + {NUM_SILENCE_FILES} silence)")
    else:
        total_files_created = created_count
    
    # Import to iTunes
    print("\n" + "="*60)
    print("Importing entire folder to iTunes/Music...")
    print("This may take a moment...\n")
    
    result = import_folder_to_music(OUTPUT_DIR)
    
    print(f"Import complete!")
    print(f"  Imported folder: {OUTPUT_DIR.resolve()}")
    print(f"  Total files: {total_files_created}")
    print(f"  Expected: {NUM_UNIQUE_SAMPLES} (from config)")
    if total_files_created != NUM_UNIQUE_SAMPLES:
        print(f"  ⚠️  Mismatch: Created {total_files_created} but config specifies {NUM_UNIQUE_SAMPLES}")
    print(f"\nNext: Run 2-import-duplicate-padded-samples-into-itunes-playlist.py\n")


if __name__ == "__main__":
    main()
