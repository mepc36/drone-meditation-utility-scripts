#!/usr/bin/env python3

"""
1-combine-samples-with-panning.py

Creates unique stereo combinations of audio samples with random panning.
Reads source files from ./input/audio/ and writes combined files to ./output/audio/final-sample-versions/
"""

import json
from collections import deque
from pathlib import Path
import random
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

# Parse bpms (required)
if "bpms" not in config:
    raise ValueError("Error: 'bpms' is required in config.json")
BPM_VALUES = [int(x) for x in str(config["bpms"]).split(":")]
BEAT_LENGTHS_SECONDS = [60.0 / bpm for bpm in BPM_VALUES]

# Parse slow_to_fast_bpm_percents
bpm_percent_parts = [int(x) for x in str(config.get("slow_to_fast_bpm_percents", "100")).split(":")]
if sum(bpm_percent_parts) != 100:
    raise ValueError(
        f"Error: slow_to_fast_bpm_percents percentages must sum to exactly 100.\n"
        f"Got: {':'.join(map(str, bpm_percent_parts))} = {sum(bpm_percent_parts)}"
    )
if len(bpm_percent_parts) != len(BPM_VALUES):
    raise ValueError(
        f"Error: slow_to_fast_bpm_percents must have the same number of values as bpms.\n"
        f"Got {len(BPM_VALUES)} bpm(s) but {len(bpm_percent_parts)} percent(s)."
    )
BPM_PERCENTS = bpm_percent_parts

NUM_UNIQUE_SAMPLES = config["num_unique_samples"]  # Total files (audio + silence)

# Parse center_diagonal_dualpan_leftorright_percents
# e.g., "25:25:25:25" means 25% center : 25% diagonal : 25% dualpan : 25% leftorright
panning_pattern_parts = [int(x) for x in config.get("center_diagonal_dualpan_leftorright_percents", "25:25:25:25").split(":")]
if len(panning_pattern_parts) != 4:
    raise ValueError(
        f"Error: center_diagonal_dualpan_leftorright_percents must have exactly 4 colon-separated values (center:diagonal:dualpan:leftorright).\n"
        f"Got: {':'.join(map(str, panning_pattern_parts))}"
    )
if sum(panning_pattern_parts) != 100:
    raise ValueError(
        f"Error: center_diagonal_dualpan_leftorright_percents percentages must sum to exactly 100.\n"
        f"Got: {':'.join(map(str, panning_pattern_parts))} = {sum(panning_pattern_parts)}"
    )
CENTER_ONLY_WEIGHT = panning_pattern_parts[0]  # Percentage, center
DIAGONAL_WEIGHT = panning_pattern_parts[1]     # Percentage, diagonal left or right
DUALPAN_WEIGHT = panning_pattern_parts[2]      # Percentage, left + right stereo pair
LEFTORRIGHT_WEIGHT = panning_pattern_parts[3]  # Percentage, hard left or hard right

# Parse samples_to_silence_percents (e.g., "87:13" means 87% non-strings : 13% silence)
silence_ratio_parts = [int(x) for x in config.get("samples_to_silence_percents", "100:0").split(":")]
if sum(silence_ratio_parts) != 100:
    raise ValueError(
        f"Error: samples_to_silence_percents percentages must sum to exactly 100.\n"
        f"Got: {':'.join(map(str, silence_ratio_parts))} = {sum(silence_ratio_parts)}"
    )
SAMPLES_PERCENT = silence_ratio_parts[0]  # Percentage of non-strings (A) in the A:B ratio
SILENCE_PERCENT = silence_ratio_parts[1] if len(silence_ratio_parts) > 1 else 0  # Percentage of silence (B) in the A:B ratio

# Parse samples_to_strings_percents (e.g., "96:4" means 96% non-strings : 4% strings)
# A strings sample is any file whose sound-type field (3rd underscore-delimited token) is "strings",
# e.g. flowing-string-quartet_oov_strings.wav
strings_ratio_parts = [int(x) for x in config.get("samples_to_strings_percents", "100:0").split(":")]
if sum(strings_ratio_parts) != 100:
    raise ValueError(
        f"Error: samples_to_strings_percents percentages must sum to exactly 100.\n"
        f"Got: {':'.join(map(str, strings_ratio_parts))} = {sum(strings_ratio_parts)}"
    )
NON_STRINGS_PERCENT = strings_ratio_parts[0]
STRINGS_PERCENT = strings_ratio_parts[1] if len(strings_ratio_parts) > 1 else 0

# Calculate counts using two independent ratios sharing non-strings (A) as the common baseline:
#   samples_to_silence_percents defines  A (non-strings) : B (silence)
#   samples_to_strings_percents defines  A (non-strings) : C (strings)
# Derive a three-way proportional split from these two ratios.
if SILENCE_PERCENT == 0:
    NUM_SILENCE_FILES = 0
    NUM_AUDIO_SAMPLES = NUM_UNIQUE_SAMPLES
else:
    ratio_silence = SILENCE_PERCENT / SAMPLES_PERCENT  # silence per unit of non-strings
    ratio_strings = (STRINGS_PERCENT / NON_STRINGS_PERCENT) if (NON_STRINGS_PERCENT > 0 and STRINGS_PERCENT > 0) else 0.0
    _denom = 1.0 + ratio_silence + ratio_strings
    NUM_SILENCE_FILES = round(NUM_UNIQUE_SAMPLES * ratio_silence / _denom)
    NUM_AUDIO_SAMPLES = NUM_UNIQUE_SAMPLES - NUM_SILENCE_FILES

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

# Parse multi-beat duration samples config
FOUR_BEAT_DURATION_PERCENT = config.get("four_beat_duration_percent", 0.0) / 100.0
TWO_BEAT_DURATION_PERCENT = config.get("two_beat_duration_percent", 0.0) / 100.0

# Panning range constants for non-center samples
# Left side uses negative values, right side uses positive values
NON_CENTER_PAN_MIN = 0.50  # Minimum distance from center (applies to both sides)
NON_CENTER_PAN_MAX = 0.50  # Maximum distance from center (applies to both sides)

# Number of segments for statistical balancing of panning distribution
PAN_DISTRIBUTION_SEGMENTS = 10  # Divide panning range into this many segments for tracking

# Sound types that must appear alone — they cannot be paired in dualpan.
# Add or remove sound-type strings here to control which musical groupings are unpaired.
# Note: 'strings' is also listed here but has additional special handling elsewhere
# (always center-only, with no pan or volume adjustments).
# Note: 'acappella' is intentionally NOT listed here — acappella CAN appear in dualpan.
NOT_PAIRED_SAMPLES = ['strings', 'kickstab', 'snarestab']

# Sound types whose per-sample panning, normalisation, and padding are left untouched.
NOT_PANNED_SAMPLES = ['strings']

# Sound types whose final mix normalisation and volume level are left untouched.
NOT_VOLUME_ADJUSTED_SAMPLES = ['strings']

# Sound types where each individual sample file may only appear once in the output.
# If the quota calculated from config requests more than the number of unique files
# of these types, it is silently capped and a warning is printed.
NO_DUPLICATES_SAMPLES = ['strings']

# Sound types that are always generated at the first BPM listed in the config,
# regardless of slow_to_fast_bpm_percents. Add a type here to lock it to BPM_VALUES[0].
# Note: 'acappella' is intentionally NOT listed here — its BPM is governed by SOUND_TYPE_RULES.
NO_BPM_VARIATION_SAMPLES = []

# Per-type rules coupling panning, volume, and BPM.
# 'allowed_pannings': which panning groups this type may use (whitelisted).
# 'volume_by_panning': for each panning group, which volume is allowed:
#   'loud_only'  => always volume_levels_db[0]  (first / highest value)
#   'quiet_only' => always volume_levels_db[-1] (last / lowest value)
#   'any'        => draw freely from pool (no constraint)
# 'bpm_if_loud':     BPM constraint when volume == volume_levels_db[0]
# 'bpm_if_not_loud': BPM constraint otherwise
#   'any'       => draw freely from pool
#   'slow_only' => always BPM_VALUES[0] (first / slowest BPM)
#   'fast_only' => always BPM_VALUES[-1] (last / fastest BPM)
# 'bpm_by_panning' (optional): per-panning BPM override checked before bpm_if_loud/bpm_if_not_loud.
SOUND_TYPE_RULES: list[dict] = [
        {
        'musical_groupings': ['kick', 'snare'],
        'pannings': {
            'center': {
                'volumes': {
                    'loud': {'bpms': ['fast', 'slow']},
                },
            },
            'leftorright': {
                'volumes': {
                    'loud': {'bpms': ['fast', 'slow']},
                },
            },
            'dualpans': {
                'volumes': {
                    'loud': {'bpms': ['fast', 'slow']},
                },
            },
        },
    },
    {
        'musical_groupings': ['kickstab', 'snarestab'],
        'pannings': {
            'diagonal': {
                'volumes': {
                    'quiet': {'bpms': ['slow']},
                },
            },
            'leftorright': {
                'volumes': {
                    'loud': {'bpms': ['fast', 'slow']},
                },
            },
        },
    },
        {
        'musical_groupings': ['acappella'],
        'pannings': {
            'center': {
                'volumes': {
                    'quiet': {'bpms': ['slow']},
                },
            },
            'diagonal': {
                'volumes': {
                    'loud':  {'bpms': ['fast', 'slow']},
                    'quiet': {'bpms': ['slow']},
                },
            },
            'leftorright': {
                'volumes': {
                    'loud':  {'bpms': ['fast', 'slow']},
                },
            },
        },
    },
    {
        'musical_groupings': ['strings'],
        'pannings': {
            'untouched': {
                'volumes': {
                    'untouched': {
                        'bpms': ['untouched'],
                    },
                },
            },
        },
    },
]

# Flat lookup map built from SOUND_TYPE_RULES — use this for runtime lookups.
_SOUND_TYPE_RULE_MAP: dict[str, dict] = {
    sound_type: rule
    for rule in SOUND_TYPE_RULES
    for sound_type in rule['musical_groupings']
}


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
    Normalizes common misspellings of known sound types so rules apply consistently.
    """
    parts = sample_name.split('_')
    if len(parts) >= 3:
        raw = parts[2].split('.')[0].lower()
    else:
        raw = sample_name.split('.')[0].lower()

    # Normalize acappella misspellings (acapela, acappela, etc.) to the canonical form.
    if raw.startswith('acap'):
        return 'acappella'

    return raw


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


def refill_round_robin_queue(sample_queue: deque, all_sample_names: list[str]) -> None:
    """Append a new shuffled round to the queue."""
    eligible = list(all_sample_names)
    random.shuffle(eligible)
    sample_queue.extend(eligible)


def dequeue_next_sample(sample_queue: deque, all_sample_names: list[str],
                         require_strings: bool | None = None) -> str | None:
    """Pop the next usable sample from the round-robin queue.
    Refills when the queue is empty.
    require_strings: True = only strings samples, False = only non-strings samples, None = any
    """
    # Fast check: are there any eligible samples of the required type at all?
    # If not, return None immediately to avoid unbounded queue growth.
    if require_strings is not None:
        any_eligible = any(
            (get_sound_type(s) == 'strings') == require_strings
            for s in all_sample_names
        )
        if not any_eligible:
            return None

    # Try the current queue first; if nothing matches, do one refill and try once more.
    for pass_num in range(2):
        if not sample_queue:
            refill_round_robin_queue(sample_queue, all_sample_names)
        if not sample_queue:
            return None
        if pass_num == 1:
            # Second pass: refill before scanning to ensure fresh samples are present.
            refill_round_robin_queue(sample_queue, all_sample_names)
        for i in range(len(sample_queue)):
            s = sample_queue[i]
            is_strings = get_sound_type(s) == 'strings'
            if require_strings is True and not is_strings:
                continue
            if require_strings is False and is_strings:
                continue
            del sample_queue[i]
            return s
    return None


def dequeue_next_sample_of_types(sample_queue: deque, all_sample_names: list[str],
                                  allowed_types: set[str],
                                  exclude_name: str | None = None) -> str | None:
    """Pop the next usable non-strings sample from the round-robin queue whose
    sound_type is in *allowed_types*. Returns None if no eligible sample exists.
    exclude_name: if provided, skip that specific sample (used to avoid pairing a sample with itself).
    """
    any_eligible = any(
        get_sound_type(s) in allowed_types
        and s != exclude_name
        for s in all_sample_names
    )
    if not any_eligible:
        return None

    for pass_num in range(2):
        if not sample_queue:
            refill_round_robin_queue(sample_queue, all_sample_names)
        if not sample_queue:
            return None
        if pass_num == 1:
            refill_round_robin_queue(sample_queue, all_sample_names)
        for i in range(len(sample_queue)):
            s = sample_queue[i]
            if s == exclude_name:
                continue
            st = get_sound_type(s)
            if st not in allowed_types:
                continue
            del sample_queue[i]
            return s
    return None


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
    elif pan_position == 'hard_left':
        left = audio * HARD_PAN_GAIN
        right = np.zeros_like(audio)
    elif pan_position == 'hard_right':
        left = np.zeros_like(audio)
        right = audio * HARD_PAN_GAIN
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
                       sample_rate: int, volume_db: float, beat_length_seconds: float,
                       use_padded_length: bool = False, use_double_time: bool = False,
                       use_four_beat: bool = False, use_two_beat: bool = False, is_centered: bool = False) -> np.ndarray:
    """
    Create a stereo mix of samples with their pan positions.
    Returns padded stereo audio.

    Args:
        sample_names: List of sample names to combine
        pan_assignments: Dictionary mapping sample names to pan positions
        sample_rate: Sample rate for the output
        volume_db: dB reduction to apply (selected in main() before this call)
        use_padded_length: If True, pad to PADDED_CENTERED_LENGTH_SECONDS instead of BEAT_LENGTH_SECONDS
        use_double_time: If True, pad to DOUBLE_TIMED_BEAT_LENGTH_SECONDS (half beat length)
        use_four_beat: If True, pad to FOUR_BEAT_LENGTH_SECONDS (4 beats)
        use_two_beat: If True, pad to TWO_BEAT_LENGTH_SECONDS (2 beats)
        is_centered: If True, sample is centered
    """
    # Load and pan each sample
    mixed = None
    
    # Determine target length based on flags (priority order)
    if use_four_beat:
        target_length = beat_length_seconds * 4.0
    elif use_two_beat:
        target_length = beat_length_seconds * 2.0
    elif use_double_time:
        target_length = beat_length_seconds / 2.0
    elif use_padded_length:
        target_length = PADDED_CENTERED_LENGTH_SECONDS
    else:
        target_length = beat_length_seconds
    
    # Load and resample all samples first
    loaded_audio: dict[str, np.ndarray] = {}
    for name in sample_names:
        audio, sr = load_audio(name)
        if sr != sample_rate:
            audio = resample_audio(audio, sr, sample_rate)
        loaded_audio[name] = audio

    # For dualpan (stereo pair), truncate both samples to the length of the shorter one
    if len(sample_names) == 2:
        min_len = min(len(loaded_audio[n]) for n in sample_names)
        for name in sample_names:
            if len(loaded_audio[name]) > min_len:
                loaded_audio[name] = loaded_audio[name][:min_len]

    # Check combination-level processing flags
    has_not_panned = any(get_sound_type(name) in NOT_PANNED_SAMPLES for name in sample_names)
    has_not_volume_adjusted = any(get_sound_type(name) in NOT_VOLUME_ADJUSTED_SAMPLES for name in sample_names)

    for name in sample_names:
        audio = loaded_audio[name]

        if get_sound_type(name) in NOT_PANNED_SAMPLES:
            # NOT_PANNED_SAMPLES: skip normalisation, panning, and length truncation — preserve
            # the file at its full natural length, as-is.
            # If mono, duplicate to stereo center; if already stereo, use directly.
            if audio.ndim == 1:
                stereo = np.column_stack([audio, audio])
            else:
                stereo = audio
        else:
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

    if not has_not_panned:
        # Normalize final mix to consistent RMS level
        mixed = normalize_to_rms(mixed, target_rms=0.15)

    if not has_not_volume_adjusted:
        # Only adjust volume for types not in NOT_VOLUME_ADJUSTED_SAMPLES
        mixed = apply_volume_db(mixed, volume_db)

    return mixed


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

    # If min and max are equal, the pan position is fixed — no need for distribution logic
    if range_min == range_max:
        position = range_min
        return -position if side == 'left' else position

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


def generate_unique_combination(samples_by_type: dict[str, list[str]],
                               center_quota: int, left_quota: int, right_quota: int, dualpan_quota: int,
                               hard_left_quota: int, hard_right_quota: int,
                               left_pan_history: list[float], right_pan_history: list[float],
                               sample_round_robin: deque, all_sample_names: list[str],
                               require_strings: bool | None = None) -> tuple[list[str], dict[str, str]]:
    """Generate a combination using round-robin sample selection + quota-based panning.

    Sample selection: samples are drawn from a shuffled round-robin queue so that
    every input file is used roughly the same number of times.  Once all samples
    have been used once the queue is refilled in a new random order, giving the
    next "round".

    Panning patterns:
    1. 1 sample, center only
    2. 1 sample, diagonal left or right (quota-based for 50/50 distribution)
    3. 2 samples, stereo pair (1 left + 1 right)
    4. 1 sample, hard left or hard right (quota-based for 50/50 distribution)

    SOLO samples: isolated, can pan left/center/right, no stereo pairs.
    Regular samples: can be combined, use all patterns.

    Returns (sample_names, pan_assignments)
    """
    # ----------------------------------------------------------------
    # Step 1: dequeue the primary sample from the round-robin queue
    # ----------------------------------------------------------------
    primary = dequeue_next_sample(sample_round_robin, all_sample_names, require_strings)
    if primary is None:
        return None, None

    sound_type = get_sound_type(primary)

    # ----------------------------------------------------------------
    # Step 2: choose panning pattern based on type + remaining quotas
    # ----------------------------------------------------------------
    if sound_type == 'solo':
        # SOLO: isolated but can pan left/center/right (no stereo pairs)
        pattern_pool = (
            ['center_only'] * center_quota +
            ['left_only'] * left_quota +
            ['right_only'] * right_quota +
            ['hard_left_only'] * hard_left_quota +
            ['hard_right_only'] * hard_right_quota
        )
    elif sound_type in _SOUND_TYPE_RULE_MAP:
        # Types with explicit panning rules: build the pattern pool from pannings.
        _rule = _SOUND_TYPE_RULE_MAP[sound_type]
        _allowed = _rule['pannings']
        if 'untouched' in _allowed:
            pattern_pool = ['center_only']
        else:
            pattern_pool = []
            if 'center' in _allowed:
                pattern_pool += ['center_only'] * center_quota
            if 'diagonal' in _allowed:
                pattern_pool += ['left_only'] * left_quota + ['right_only'] * right_quota
            if 'dualpan' in _allowed:
                pattern_pool += ['stereo_pair'] * dualpan_quota
            if 'leftorright' in _allowed:
                pattern_pool += ['hard_left_only'] * hard_left_quota + ['hard_right_only'] * hard_right_quota
    elif sound_type in NOT_PAIRED_SAMPLES:
        # Catch-all for unpaired types not covered by SOUND_TYPE_RULES.
        pattern_pool = (
            ['left_only'] * left_quota +
            ['right_only'] * right_quota +
            ['hard_left_only'] * hard_left_quota +
            ['hard_right_only'] * hard_right_quota
        )
    else:
        # Regular: all types can appear in dualpan.
        # All types are paired same-type-only by default in those blocks.
        pattern_pool = (
            ['center_only'] * center_quota +
            ['left_only'] * left_quota +
            ['right_only'] * right_quota +
            ['stereo_pair'] * dualpan_quota +
            ['hard_left_only'] * hard_left_quota +
            ['hard_right_only'] * hard_right_quota
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

    elif pattern_type == 'hard_left_only':
        sample_names = [primary]
        pan_assignments = {primary: 'hard_left'}

    elif pattern_type == 'hard_right_only':
        sample_names = [primary]
        pan_assignments = {primary: 'hard_right'}

    else:  # stereo_pair
        # All types pair only with their own type by default.
        # To add cross-group mixing for a new type, add a branch here.
        _partner_allowed = {sound_type}
        partner = dequeue_next_sample_of_types(
            sample_round_robin, all_sample_names,
            _partner_allowed, exclude_name=primary
        )
        if partner is None:
            # Can't form a pair — fall back to an allowed single-channel panning.
            # Types that don't allow center fall back to diagonal.
            _rule = _SOUND_TYPE_RULE_MAP.get(sound_type)
            if _rule and 'center' not in _rule['pannings']:
                pan_position = select_balanced_pan_position('left', left_pan_history)
                sample_names = [primary]
                pan_assignments = {primary: pan_position}
            else:
                sample_names = [primary]
                pan_assignments = {primary: 'center'}
        else:
            sample_names = [primary, partner]
            # Dualpan is always hard left + hard right — never mixed with center.
            pan_assignments = {primary: 'hard_left', partner: 'hard_right'}

    return sample_names, pan_assignments


def infer_pan_group(sample_names: list[str], pan_assignments: dict[str, str]) -> str:
    """Derive the panning group label from the pan assignments.
    Returns one of: center, diagonal, dualpan, leftorright.
    """
    n = len(sample_names)
    if n == 2:
        return 'dualpan'
    # Single sample
    pan = pan_assignments[sample_names[0]]
    if pan == 'center':
        return 'center'
    if pan in ('hard_left', 'hard_right'):
        return 'leftorright'
    return 'diagonal'


def format_filename(sample_names: list[str], pan_assignments: dict[str, str], volume_db: float, index: int, bpm: int, beat_suffix: str = "1-beat") -> str:
    """
    Format filename as: left_center_right_vol-X_index-NNN_length-Xbeat_bpm-NNN_<pangroup>.wav
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
        elif pan == 'hard_left':
            pan_value = -1.0
        elif pan == 'hard_right':
            pan_value = 1.0
        else:
            # Already numeric
            pan_value = float(pan)
        samples_by_pan.append((pan_value, name))
    
    # Sort by pan position (left -> center -> right)
    samples_by_pan.sort(key=lambda x: x[0])
    
    # Extract sorted sample names
    sorted_names = [name.lower() for _, name in samples_by_pan]
    
    # Format volume (remove decimal if it's a whole number, use abs value)
    vol_str = f"{abs(volume_db):.0f}" if volume_db == int(volume_db) else f"{abs(volume_db):.1f}"
    
    pan_group = infer_pan_group(sample_names, pan_assignments)
    
    # Build filename
    name_part = "_".join(sorted_names)
    return f"{name_part}_vol-{vol_str}_index-{index:03d}_length-{beat_suffix}_bpm-{bpm}_{pan_group}.wav"


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




# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    print("\nCombine Samples with Random Panning\n")
    bpms_str = ':'.join(str(b) for b in BPM_VALUES)
    bpm_percents_str = ':'.join(str(p) for p in BPM_PERCENTS)
    print(f"BPMs: {bpms_str}  (slow_to_fast_bpm_percents: {bpm_percents_str})")
    
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
    # Calculate how many samples will use each pattern based on percentages
    center_quota = int(NUM_AUDIO_SAMPLES * CENTER_ONLY_WEIGHT / 100)
    diagonal_quota = int(NUM_AUDIO_SAMPLES * DIAGONAL_WEIGHT / 100)
    dualpan_quota = int(NUM_AUDIO_SAMPLES * DUALPAN_WEIGHT / 100)
    leftorright_quota = int(NUM_AUDIO_SAMPLES * LEFTORRIGHT_WEIGHT / 100)

    # Cap center_quota to the number of unique audio files (each center combo is 1 file = 1 unique slot)
    if center_quota > total_samples:
        center_overflow = center_quota - total_samples
        center_quota = total_samples
        # Redistribute overflow proportionally to diagonal, dualpan, and leftorright
        non_center_total_weight = DIAGONAL_WEIGHT + DUALPAN_WEIGHT + LEFTORRIGHT_WEIGHT
        if non_center_total_weight > 0:
            diagonal_quota += int(center_overflow * DIAGONAL_WEIGHT / non_center_total_weight)
            leftorright_quota += int(center_overflow * LEFTORRIGHT_WEIGHT / non_center_total_weight)
            dualpan_quota += center_overflow - int(center_overflow * DIAGONAL_WEIGHT / non_center_total_weight) - int(center_overflow * LEFTORRIGHT_WEIGHT / non_center_total_weight)
        else:
            dualpan_quota += center_overflow
        print(f"⚠️  Center quota capped at {total_samples} (number of unique files). Overflow redistributed to diagonal/dualpan/leftorright.\n")

    # Handle case where num_audio_samples is very small and all quotas round to 0
    # Distribute remaining samples proportionally
    total_allocated = center_quota + diagonal_quota + dualpan_quota + leftorright_quota
    remaining_samples = NUM_AUDIO_SAMPLES - total_allocated
    if remaining_samples > 0:
        # Allocate remaining samples based on weights
        weights = [CENTER_ONLY_WEIGHT, DIAGONAL_WEIGHT, DUALPAN_WEIGHT, LEFTORRIGHT_WEIGHT]
        max_weight_idx = weights.index(max(weights))
        if max_weight_idx == 0:
            center_quota += remaining_samples
        elif max_weight_idx == 1:
            diagonal_quota += remaining_samples
        elif max_weight_idx == 2:
            dualpan_quota += remaining_samples
        else:
            leftorright_quota += remaining_samples
    
    # Split diagonal quota evenly between left and right for 50/50 distribution
    left_quota = diagonal_quota // 2
    right_quota = diagonal_quota - left_quota  # Gives right any remainder for odd numbers
    
    # Split leftorright quota evenly between hard left and hard right for 50/50 distribution
    hard_left_quota = leftorright_quota // 2
    hard_right_quota = leftorright_quota - hard_left_quota  # Gives right any remainder for odd numbers
    
    adjusted_center_weight = center_quota
    adjusted_left_weight = left_quota
    adjusted_right_weight = right_quota
    adjusted_dualpan_weight = dualpan_quota
    adjusted_hard_left_weight = hard_left_quota
    adjusted_hard_right_weight = hard_right_quota
    
    # Calculate strings vs non-strings quotas using the independent A:C ratio.
    # Both samples_to_silence_percents (A:B) and samples_to_strings_percents (A:C) share
    # non-strings (A) as their common baseline, so strings are derived from NUM_AUDIO_SAMPLES
    # proportionally via the A:C ratio rather than as a flat percentage of it.
    if STRINGS_PERCENT == 0 or NON_STRINGS_PERCENT == 0:
        num_strings_samples = 0 if STRINGS_PERCENT == 0 else NUM_AUDIO_SAMPLES
        num_non_strings_samples = NUM_AUDIO_SAMPLES - num_strings_samples
    else:
        _ratio_strings = STRINGS_PERCENT / NON_STRINGS_PERCENT  # C per unit of A
        num_strings_samples = round(NUM_AUDIO_SAMPLES * _ratio_strings / (1.0 + _ratio_strings))
        num_non_strings_samples = NUM_AUDIO_SAMPLES - num_strings_samples

    # Cap NO_DUPLICATES_SAMPLES types to the number of unique files that actually exist.
    actual_no_dup_count = sum(
        len(v) for k, v in samples_by_type.items() if k in NO_DUPLICATES_SAMPLES
    )
    if num_strings_samples > actual_no_dup_count:
        print(
            f"⚠️  Warning: samples_to_strings_percents requests {num_strings_samples} strings "
            f"sample(s), but only {actual_no_dup_count} unique strings file(s) exist "
            f"(NO_DUPLICATES_SAMPLES is enforced). Capping at {actual_no_dup_count}."
        )
        num_strings_samples = actual_no_dup_count
        num_non_strings_samples = NUM_AUDIO_SAMPLES - num_strings_samples

    sample_round_robin, all_sample_names = build_sample_round_robin(samples_by_type)
    sample_usage_count: dict[str, int] = {s: 0 for s in all_sample_names}
    
    # Track combinations to ensure uniqueness
    seen_combinations = set()
    created_count = 0
    attempts = 0
    max_attempts = NUM_AUDIO_SAMPLES * 100  # Prevent infinite loop
    
    # Quota-based generation: track how many of each pattern we still need
    center_quota_remaining = adjusted_center_weight
    left_quota_remaining = adjusted_left_weight
    right_quota_remaining = adjusted_right_weight
    dualpan_quota_remaining = adjusted_dualpan_weight
    hard_left_quota_remaining = adjusted_hard_left_weight
    hard_right_quota_remaining = adjusted_hard_right_weight

    # Strings vs non-strings quota tracking
    strings_quota_remaining = num_strings_samples
    non_strings_quota_remaining = num_non_strings_samples
    strings_created_count = 0
    non_strings_created_count = 0
    
    # Track panning distribution
    center_count = 0
    left_count = 0
    right_count = 0
    dualpan_count = 0
    hard_left_count = 0
    hard_right_count = 0
    
    # Track pan position histories for statistical balancing (store absolute values)
    left_pan_history = []  # Stores positive values (0.35 to 1.0)
    right_pan_history = []  # Stores positive values (0.35 to 1.0)
    
    # Track volume level distribution
    volume_counts = [0] * len(volume_levels_db)
    
    # Initialize volume quota pool (quota-based selection).
    # Pools are sized against non-strings samples only — strings bypass both pools entirely.
    volume_quotas = []
    for pct in volume_percentages:
        quota = int(num_non_strings_samples * pct / 100)
        volume_quotas.append(quota)

    # Build shared volume pool for all non-strings samples
    volume_pool = []
    for idx, quota in enumerate(volume_quotas):
        volume_pool.extend([idx] * quota)

    # Build beat length pool based on slow_to_fast_bpm_percents quota
    beat_length_pool = []
    for idx, pct in enumerate(BPM_PERCENTS):
        quota = int(num_non_strings_samples * pct / 100)
        beat_length_pool.extend([idx] * quota)

    # Get sample rate from first file
    first_sample_name = list(samples_by_type.values())[0][0]
    _, sample_rate = load_audio(first_sample_name)
    
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

        # Determine whether the next sample should be strings or non-strings
        # based on remaining quotas to hit the configured ratio.
        if strings_quota_remaining > 0 and non_strings_quota_remaining > 0:
            total_remaining = strings_quota_remaining + non_strings_quota_remaining
            require_strings = random.random() < (strings_quota_remaining / total_remaining)
        elif strings_quota_remaining > 0:
            require_strings = True
        else:
            require_strings = False

        # Generate combination with remaining quotas
        sample_names, pan_assignments = generate_unique_combination(
            samples_by_type,
            center_quota_remaining, left_quota_remaining, right_quota_remaining, dualpan_quota_remaining,
            hard_left_quota_remaining, hard_right_quota_remaining,
            left_pan_history, right_pan_history,
            sample_round_robin, all_sample_names,
            require_strings
        )
        
        # Check if combination generation failed
        if sample_names is None:
            # If we were trying to find a strings sample and none are available, fall back
            # to non-strings so we don't spin forever on an unsatisfiable quota.
            if require_strings is True:
                sample_names, pan_assignments = generate_unique_combination(
                    samples_by_type,
                    center_quota_remaining, left_quota_remaining, right_quota_remaining, dualpan_quota_remaining,
                    hard_left_quota_remaining, hard_right_quota_remaining,
                    left_pan_history, right_pan_history,
                    sample_round_robin, all_sample_names,
                    require_strings=False
                )
            if sample_names is None:
                continue
        
        # Create unique key for this combination
        combo_key = tuple(sorted([f"{name}:{pan_assignments[name]}" for name in sample_names]))
        
        # Enforce uniqueness. Types in NO_DUPLICATES_SAMPLES may not repeat;
        # all other types also enforce uniqueness (each unique pan+sample combo once).
        is_strings_combo = any(get_sound_type(name) == 'strings' for name in sample_names)
        if combo_key in seen_combinations:
            continue
        seen_combinations.add(combo_key)
        
        # Track per-sample usage for the round-robin fairness report
        for name in sample_names:
            sample_usage_count[name] = sample_usage_count.get(name, 0) + 1
        
        created_count += 1

        # Track strings vs non-strings against quota
        if is_strings_combo:
            strings_created_count += 1
            strings_quota_remaining = max(0, strings_quota_remaining - 1)
        else:
            non_strings_created_count += 1
            non_strings_quota_remaining = max(0, non_strings_quota_remaining - 1)

        # Track panning pattern and update quotas
        pan_positions = list(pan_assignments.values())
        is_centered = False
        is_non_centered = False
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
            elif pan_positions[0] == 'hard_left':
                is_non_centered = True
                hard_left_count += 1
                hard_left_quota_remaining = max(0, hard_left_quota_remaining - 1)
            elif pan_positions[0] == 'hard_right':
                is_non_centered = True
                hard_right_count += 1
                hard_right_quota_remaining = max(0, hard_right_quota_remaining - 1)
            else:  # diagonal left or right (numeric pan value)
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
            dualpan_count += 1
            dualpan_quota_remaining = max(0, dualpan_quota_remaining - 1)
        
        # Determine sound type and pan group for SOUND_TYPE_RULES lookup.
        primary_sound_type = get_sound_type(sample_names[0])
        pan_group = infer_pan_group(sample_names, pan_assignments)
        _rule = _SOUND_TYPE_RULE_MAP.get(primary_sound_type)

        # Select volume. Types with 'untouched' panning (e.g. strings) are not volume-adjusted
        # and do not consume from the pool. For all others, forced-loud/quiet assignments
        # consume the matching slot from the pool so the overall ratio stays on target.
        if _rule is not None:
            if 'untouched' in _rule['pannings']:
                volume_db = 0.0  # no volume adjustment
                volume_idx = 0
            else:
                _pan_rule = _rule['pannings'].get(pan_group)
                if _pan_rule is not None:
                    _vols = set(_pan_rule['volumes'].keys())
                    if _vols == {'loud'}:
                        vol_constraint = 'loud_only'
                    elif _vols == {'quiet'}:
                        vol_constraint = 'quiet_only'
                    else:
                        vol_constraint = 'any'
                else:
                    vol_constraint = 'any'
                if vol_constraint == 'loud_only':
                    volume_db = volume_levels_db[0]
                    volume_idx = 0
                    if volume_idx in volume_pool:  # consume quota so pool compensates
                        volume_pool.remove(volume_idx)
                elif vol_constraint == 'quiet_only':
                    volume_db = volume_levels_db[-1]
                    volume_idx = len(volume_levels_db) - 1
                    if volume_idx in volume_pool:  # consume quota so pool compensates
                        volume_pool.remove(volume_idx)
                else:  # 'any' — draw freely from pool
                    volume_db, volume_idx = select_volume_level_from_pool(volume_pool)
                    if volume_idx in volume_pool:
                        volume_pool.remove(volume_idx)
                volume_counts[volume_idx] += 1
        else:
            volume_db, volume_idx = select_volume_level_from_pool(volume_pool)
            if volume_idx in volume_pool:
                volume_pool.remove(volume_idx)
            volume_counts[volume_idx] += 1

        # Select BPM. Types with 'untouched' panning (e.g. strings) use their natural file
        # length and do not consume from the pool. For all others, forced slow/fast assignments
        # consume their slot from the pool so the overall BPM ratio stays on target.
        is_loud = (volume_db == volume_levels_db[0])
        is_no_bpm_variation = any(get_sound_type(name) in NO_BPM_VARIATION_SAMPLES for name in sample_names)
        if _rule is not None:
            if 'untouched' in _rule['pannings']:
                selected_bpm_idx = 0  # untouched = natural file length, no BPM adjustment
            else:
                _pan_rule = _rule['pannings'].get(pan_group)
                if _pan_rule is not None:
                    _vol_key = 'loud' if is_loud else 'quiet'
                    _vol_rule = _pan_rule['volumes'].get(_vol_key)
                    _allowed_bpms = _vol_rule['bpms'] if _vol_rule is not None else ['slow', 'fast']
                    if _allowed_bpms == ['slow']:
                        bpm_constraint = 'slow_only'
                    elif _allowed_bpms == ['fast']:
                        bpm_constraint = 'fast_only'
                    else:
                        bpm_constraint = 'any'
                else:
                    bpm_constraint = 'any'
                if bpm_constraint == 'slow_only':
                    selected_bpm_idx = 0
                    if 0 in beat_length_pool:  # consume quota so pool compensates
                        beat_length_pool.remove(0)
                elif bpm_constraint == 'fast_only':
                    selected_bpm_idx = len(BPM_VALUES) - 1
                    fast_idx = len(BPM_VALUES) - 1
                    if fast_idx in beat_length_pool:  # consume quota so pool compensates
                        beat_length_pool.remove(fast_idx)
                else:  # 'any'
                    if beat_length_pool:
                        selected_bpm_idx = random.choice(beat_length_pool)
                        beat_length_pool.remove(selected_bpm_idx)
                    else:
                        selected_bpm_idx = 0
        elif is_no_bpm_variation:
            selected_bpm_idx = 0
            if 0 in beat_length_pool:  # consume quota so pool compensates
                beat_length_pool.remove(0)
        elif not is_loud:
            # Quiet samples are always slow BPM, regardless of type.
            selected_bpm_idx = 0
            if 0 in beat_length_pool:  # consume quota so pool compensates
                beat_length_pool.remove(0)
        elif beat_length_pool:
            selected_bpm_idx = random.choice(beat_length_pool)
            beat_length_pool.remove(selected_bpm_idx)
        else:
            selected_bpm_idx = 0  # Fallback to first BPM
        beat_length_sec = BEAT_LENGTHS_SECONDS[selected_bpm_idx]

        # Create the audio (volume_db is now pre-selected, enabling Phase 3 coupling)
        combined_audio = create_combination(sample_names, pan_assignments, sample_rate, volume_db, beat_length_sec, use_padded_length, use_double_time, use_four_beat, use_two_beat, is_centered)
        
        # Generate filename
        if use_four_beat:
            beat_suffix = "4-beat"
        elif use_two_beat:
            beat_suffix = "2-beat"
        elif use_double_time:
            beat_suffix = "half-beat"
        else:
            beat_suffix = "1-beat"
        filename = format_filename(sample_names, pan_assignments, volume_db, created_count, BPM_VALUES[selected_bpm_idx], beat_suffix)
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
    
    diagonal_count = left_count + right_count
    leftorright_count = hard_left_count + hard_right_count
    
    def gcd_multiple(*args):
        """Calculate GCD of multiple numbers."""
        return reduce(gcd, args)
    
    # Calculate GCD for ratio display
    counts = [c for c in [center_count, diagonal_count, dualpan_count, leftorright_count] if c > 0]
    if len(counts) > 1:
        ratio_gcd = gcd_multiple(*counts)
    else:
        ratio_gcd = 1
    
    realized_ratio = f"{center_count//ratio_gcd}:{diagonal_count//ratio_gcd}:{dualpan_count//ratio_gcd}:{leftorright_count//ratio_gcd}"
    
    print(f"\nPanning Distribution:")
    print(f"  Config ratio: {config.get('center_diagonal_dualpan_leftorright_percents', '25:25:25:25')}")
    print(f"  Realized ratio: {realized_ratio}")
    
    # Calculate perfect ratio scaled to match realized first value
    config_ratio_parts = [int(x) for x in config.get('center_diagonal_dualpan_leftorright_percents', '25:25:25:25').split(':')]
    realized_parts = [int(x) for x in realized_ratio.split(':')]
    scale_factor = realized_parts[0] / config_ratio_parts[0] if config_ratio_parts[0] != 0 else 1
    perfect_ratio_parts = [int(x * scale_factor) for x in config_ratio_parts]
    perfect_ratio = ':'.join([str(x) for x in perfect_ratio_parts])
    print(f"  Perfect ratio: {perfect_ratio}")
    
    # Calculate differential (perfect - realized): shows what's needed to reach perfect
    differential = [perfect_ratio_parts[i] - realized_parts[i] for i in range(len(realized_parts))]
    differential_str = ':'.join([f"{'+' if d > 0 else ''}{d}" for d in differential])
    print(f"  Differential: {differential_str}")
    
    # Display left/right distribution for diagonal samples
    if diagonal_count > 0:
        left_pct = (left_count / diagonal_count) * 100
        right_pct = (right_count / diagonal_count) * 100
        print(f"\nDiagonal Left/Right Distribution:")
        print(f"  Target: 50.0% left : 50.0% right")
        print(f"  Realized: {left_count}:{right_count} = {left_pct:.1f}% : {right_pct:.1f}%")
        diff = left_count - right_count
        print(f"  Differential: {diff:+d} (left - right)")
    
    # Display hard left/right distribution
    if leftorright_count > 0:
        hard_left_pct = (hard_left_count / leftorright_count) * 100
        hard_right_pct = (hard_right_count / leftorright_count) * 100
        print(f"\nHard Left/Right Distribution:")
        print(f"  Target: 50.0% hard left : 50.0% hard right")
        print(f"  Realized: {hard_left_count}:{hard_right_count} = {hard_left_pct:.1f}% : {hard_right_pct:.1f}%")
        diff = hard_left_count - hard_right_count
        print(f"  Differential: {diff:+d} (hard left - hard right)")
    
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
        half_lengths = ', '.join(f"{bl/2:.3f}s" for bl in BEAT_LENGTHS_SECONDS)
        full_lengths = ', '.join(f"{bl:.3f}s" for bl in BEAT_LENGTHS_SECONDS)
        bpms_x2 = ', '.join(str(b * 2) for b in BPM_VALUES)
        print(f"  BPMs (double-time): {bpms_x2}")
        print(f"  Half-beat lengths: {half_lengths} (halves of {full_lengths})")
    
    # Display 4-beat duration samples info
    if four_beat_count > 0:
        four_beat_pct = (four_beat_count / created_count) * 100 if created_count > 0 else 0
        target_four_beat_pct = FOUR_BEAT_DURATION_PERCENT * 100
        print(f"\n4-Beat Duration Samples:")
        print(f"  Target: {target_four_beat_pct:.1f}% of all samples")
        print(f"  Realized: {four_beat_count}/{created_count} = {four_beat_pct:.1f}%")
        four_beat_lengths = ', '.join(f"{bl*4:.3f}s" for bl in BEAT_LENGTHS_SECONDS)
        bpms_str = ':'.join(str(b) for b in BPM_VALUES)
        print(f"  4-beat lengths: {four_beat_lengths} (at BPMs: {bpms_str})")
    
    # Display 2-beat duration samples info
    if two_beat_count > 0:
        two_beat_pct = (two_beat_count / created_count) * 100 if created_count > 0 else 0
        target_two_beat_pct = TWO_BEAT_DURATION_PERCENT * 100
        print(f"\n2-Beat Duration Samples:")
        print(f"  Target: {target_two_beat_pct:.1f}% of all samples")
        print(f"  Realized: {two_beat_count}/{created_count} = {two_beat_pct:.1f}%")
        two_beat_lengths = ', '.join(f"{bl*2:.3f}s" for bl in BEAT_LENGTHS_SECONDS)
        bpms_str = ':'.join(str(b) for b in BPM_VALUES)
        print(f"  2-beat lengths: {two_beat_lengths} (at BPMs: {bpms_str})")
    
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
    
    # Display strings distribution
    if STRINGS_PERCENT > 0:
        strings_pct = (strings_created_count / created_count) * 100 if created_count > 0 else 0
        non_strings_pct = (non_strings_created_count / created_count) * 100 if created_count > 0 else 0
        print(f"\nStrings Distribution:")
        print(f"  Config ratio: {config.get('samples_to_strings_percents', '100:0')} (non-strings:strings)")
        print(f"  Target: {NON_STRINGS_PERCENT}% non-strings / {STRINGS_PERCENT}% strings")
        print(f"  Realized: {non_strings_created_count} non-strings ({non_strings_pct:.1f}%) / {strings_created_count} strings ({strings_pct:.1f}%)")

    print(f"\nNext: Run 2-import-duplicate-padded-samples-into-itunes-playlist.py\n")


if __name__ == "__main__":
    main()
