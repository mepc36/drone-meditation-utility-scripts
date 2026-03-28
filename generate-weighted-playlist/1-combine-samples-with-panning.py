#!/usr/bin/env python3

"""
1-combine-samples-with-panning.py

Creates unique stereo combinations of audio samples with random panning.
Reads source files from ./input/audio/ and writes combined files to ./output/audio/
"""

import json
import shutil
from collections import deque
from functools import reduce
from math import gcd
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
OUTPUT_DIR = Path("./output/audio")

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

# Parse samples_to_silence_percents (e.g., "87:13" means 87% audio samples : 13% silence)
silence_ratio_parts = [int(x) for x in config.get("samples_to_silence_percents", "100:0").split(":")]
if sum(silence_ratio_parts) != 100:
    raise ValueError(
        f"Error: samples_to_silence_percents percentages must sum to exactly 100.\n"
        f"Got: {':'.join(map(str, silence_ratio_parts))} = {sum(silence_ratio_parts)}"
    )
SAMPLES_PERCENT = silence_ratio_parts[0]  # Percentage of audio samples (A) in the A:B (audio:silence) ratio
SILENCE_PERCENT = silence_ratio_parts[1] if len(silence_ratio_parts) > 1 else 0  # Percentage of silence (B) in the A:B ratio

# Calculate audio vs silence split.
if SILENCE_PERCENT == 0:
    NUM_SILENCE_FILES = 0
    NUM_AUDIO_SAMPLES = NUM_UNIQUE_SAMPLES
else:
    ratio_silence = SILENCE_PERCENT / SAMPLES_PERCENT
    _denom = 1.0 + ratio_silence
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

# Fixed panning position for diagonal samples (left side uses negative, right side positive)
NON_CENTER_PAN = 0.53

# Shared panning rule objects — referenced by name in SOUND_TYPE_RULES below.
KICK_SNARE_PANNING_RULES: dict = {
    'leftorright': {
        'volumes': {
            'quiet': {'bpms': ['slow']},
        },
    },
    'dualpan': {
        'volumes': {
            'loud': {'bpms': ['slow', 'fast']},
        },
    },
}

KICKSTAB_SNARESTAB_PANNING_RULES: dict = {
    'center': {
        'volumes': {
            'loud': {'bpms': ['fast', 'slow']},
        },
    },
    'diagonal': {
        'volumes': {
            'quiet': {'bpms': ['slow']},
        },
    },
    'dualpan': {
        'volumes': {
            'loud': {'bpms': ['slow', 'fast']},
        },
    },
    'leftorright': {
        'volumes': {
            'quiet': {'bpms': ['slow']},
        },
    },
}

# Per-type rules: panning groups, volume levels, and BPM constraints.
# 'musical_grouping': the single sound type this rule applies to.
# 'dualpan_partners': (required) types allowed as the dualpan partner. Must include the type itself to allow self-pairing. Use [] if the type has no dualpan panning.
# 'pannings': dict of panning group → volume/BPM rules.
#   Each panning has 'volumes': dict of volume level ('loud'|'quiet') → BPM rules.
#   Each volume level has 'bpms': list of allowed BPMs (['slow'], ['fast'], or ['slow', 'fast']).
#   Special panning: 'untouched' → leave the file completely as-is (no panning, normalisation, or volume).
SOUND_TYPE_RULES: list[dict] = [
    {
        'musical_grouping': 'kick',
        'dualpan_partners': ['kick'],
        'pannings': KICK_SNARE_PANNING_RULES,
    },
    {
        'musical_grouping': 'snare',
        'dualpan_partners': ['snare'],
        'pannings': KICK_SNARE_PANNING_RULES,
    },
    {
        'musical_grouping': 'kickstab',
        'dualpan_partners': ['kickstab'],
        'pannings': KICKSTAB_SNARESTAB_PANNING_RULES,
    },
    {
        'musical_grouping': 'snarestab',
        'dualpan_partners': ['snarestab'],
        'pannings': KICKSTAB_SNARESTAB_PANNING_RULES,
    },
    {
        'musical_grouping': 'acappella',
        'dualpan_partners': [],
        'pannings': {
            'center': {
                'volumes': {
                    'quiet': {'bpms': ['slow']},
                },
            },
            'leftorright': {
                'volumes': {
                    'loud': {'bpms': ['slow']},
                },
            },
            'diagonal': {
                'volumes': {
                    'quiet': {
                        'bpms': ['slow']
                    },
                }
            },
        },
    },
    {
        'musical_grouping': 'strings',
        'dualpan_partners': [],
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

# Flat lookup map: musical_grouping → rule dict. Used for validation, PANNING_COMPAT init, and runtime.
_SOUND_TYPE_RULE_MAP: dict[str, dict] = {
    rule['musical_grouping']: rule
    for rule in SOUND_TYPE_RULES
}

# Validate: dualpan_partners is required on every rule; if non-empty, pannings must include 'dualpan'.
for _r in SOUND_TYPE_RULES:
    _mt = _r['musical_grouping']
    if 'dualpan_partners' not in _r:
        raise ValueError(
            f"SOUND_TYPE_RULES entry '{_mt}' is missing required key 'dualpan_partners'. "
            f"Use [] if this type has no dualpan panning."
        )
    if _r['dualpan_partners'] and 'dualpan' not in _r['pannings']:
        raise ValueError(
            f"SOUND_TYPE_RULES entry '{_mt}' declares dualpan_partners {_r['dualpan_partners']} "
            f"but has no 'dualpan' key under 'pannings'."
        )


def is_untouched_type(sound_type: str) -> bool:
    """Return True if this sound type should be left untouched (no panning, normalisation, or volume)."""
    r = _SOUND_TYPE_RULE_MAP.get(sound_type)
    return r is not None and 'untouched' in r['pannings']


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

# Derive loud/quiet indices by comparing dB values: louder = higher (less negative) dB.
LOUD_VOL_IDX: int = volume_levels_db.index(max(volume_levels_db))
QUIET_VOL_IDX: int = volume_levels_db.index(min(volume_levels_db))

# Derive slow/fast BPM indices by comparing BPM values.
SLOW_BPM_IDX: int = BPM_VALUES.index(min(BPM_VALUES))
FAST_BPM_IDX: int = BPM_VALUES.index(max(BPM_VALUES))

# Sound groupings used by kicksnare_stab_acappella_percents.
# Order matches the 3 colon-separated values in the config key.
SOUND_GROUP_NAMES: list[str] = ['kicksnare', 'stab', 'acappella']
SOUND_GROUP_TYPES: dict[str, set[str]] = {
    'kicksnare': {'kick', 'snare'},
    'stab':      {'kickstab', 'snarestab'},
    'acappella': {'acappella'},
}

# Derived from SOUND_TYPE_RULES: which panning modes each sound group supports.
# Populated immediately below from _SOUND_TYPE_RULE_MAP.
PANNING_COMPAT: dict[str, set[str]] = {}
for _gname, _gtypes in SOUND_GROUP_TYPES.items():
    _compat: set[str] = set()
    for _stype in _gtypes:
        _r = _SOUND_TYPE_RULE_MAP.get(_stype)
        if _r:
            for _p in _r['pannings']:
                if _p != 'untouched':
                    _compat.add(_p)
    PANNING_COMPAT[_gname] = _compat

# Parse kicksnare_stab_acappella_percents (optional).
# e.g. "30:20:50" → 30% kicksnare, 20% stab, 50% acappella of non-strings slots.
if "kicksnare_stab_acappella_percents" in config:
    _group_parts = [int(x) for x in str(config["kicksnare_stab_acappella_percents"]).split(":")]
    if len(_group_parts) != 3:
        raise ValueError(
            f"Error: kicksnare_stab_acappella_percents must have exactly 3 colon-separated values "
            f"(kicksnare:stab:acappella).\nGot: {':'.join(map(str, _group_parts))}"
        )
    if sum(_group_parts) != 100:
        raise ValueError(
            f"Error: kicksnare_stab_acappella_percents percentages must sum to exactly 100.\n"
            f"Got: {':'.join(map(str, _group_parts))} = {sum(_group_parts)}"
        )
    SOUND_GROUP_PERCENTS: list[int] | None = _group_parts
else:
    SOUND_GROUP_PERCENTS = None  # No group constraint — natural file proportions apply.


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
    """Pop the next sample from the round-robin queue whose sound_type is in *allowed_types*.
    Returns None if no eligible sample exists.
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


def create_combination(sample_names: list[str], pan_assignments: dict[str, str],
                       sample_rate: int, volume_db: float, beat_length_seconds: float) -> np.ndarray:
    """
    Create a stereo mix of samples with their pan positions.
    Returns padded stereo audio.

    Args:
        sample_names: List of sample names to combine
        pan_assignments: Dictionary mapping sample names to pan positions
        sample_rate: Sample rate for the output
        volume_db: dB reduction to apply (selected in main() before this call)
    """
    # Load and pan each sample
    mixed = None
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

    # Check whether any sample in this combination is untouched (e.g. strings)
    has_untouched = any(is_untouched_type(get_sound_type(name)) for name in sample_names)

    for name in sample_names:
        audio = loaded_audio[name]

        if is_untouched_type(get_sound_type(name)):
            # Untouched types: skip normalisation, panning, and length truncation — preserve
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

    if not has_untouched:
        # Normalize final mix to consistent RMS level
        mixed = normalize_to_rms(mixed, target_rms=0.15)
        # Apply volume adjustment
        mixed = apply_volume_db(mixed, volume_db)

    return mixed


def select_pan_position(side: str) -> float:
    """Return the fixed pan position for diagonal samples.
    Returns negative for left, positive for right.
    """
    return -NON_CENTER_PAN if side == 'left' else NON_CENTER_PAN


def gcd_multiple(*args: int) -> int:
    """Return the GCD of all given integers."""
    return reduce(gcd, args)


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


def format_filename(sample_names: list[str], pan_assignments: dict[str, str], volume_db: float, index: int, bpm: int) -> str:
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
    return f"{name_part}_vol-{vol_str}_index-{index:03d}_length-1-beat_bpm-{bpm}_{pan_group}.wav"


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
# Pre-computed deck builder
# -------------------------------------------------------------------
def build_file_deck(
    group_targets: dict[str, int],
    panning_quotas: dict[str, int],
    bpm_targets: dict[int, int],
    vol_targets: dict[int, int],
) -> list[tuple[str, str, str, str]]:
    """Pre-compute the non-strings file deck.

    Returns a shuffled list of (group, panning, vol_label, bpm_label) tuples —
    one per non-strings output file.  Directional pannings are already resolved:
      'diagonal' → 'diagonal_left' / 'diagonal_right' (50:50)
      'leftorright' → 'hard_left' / 'hard_right' (50:50)

    Groups are allocated most-constrained-first.  Overflow (when a group cannot
    fill its target from compatible panning slots) is distributed equally among
    the uncapped groups.  A startup summary is printed.
    """
    # Expand directional panning quotas into individual directional slots.
    _diag = panning_quotas.get('diagonal', 0)
    _lor  = panning_quotas.get('leftorright', 0)
    remaining: dict[str, int] = {
        'center':         panning_quotas.get('center', 0),
        'diagonal_left':  _diag // 2,
        'diagonal_right': _diag - _diag // 2,
        'dualpan':        panning_quotas.get('dualpan', 0),
        'hard_left':      _lor // 2,
        'hard_right':     _lor - _lor // 2,
    }

    def compat_dir(group: str) -> set[str]:
        """Directional panning types compatible with this sound group."""
        result: set[str] = set()
        for p in PANNING_COMPAT.get(group, set()):
            if p == 'diagonal':
                result.update({'diagonal_left', 'diagonal_right'})
            elif p == 'leftorright':
                result.update({'hard_left', 'hard_right'})
            else:
                result.add(p)
        return result

    actual_targets = dict(group_targets)
    alloc: dict[str, dict[str, int]] = {g: {} for g in group_targets}
    overflow = 0
    capped: set[str] = set()

    # Sort groups most-constrained first (fewest compatible panning slots available).
    ordered = sorted(group_targets, key=lambda g: sum(remaining.get(p, 0) for p in compat_dir(g)))

    def _fill(group: str, take: int) -> None:
        """Allocate `take` panning slots for `group`, consuming from `remaining`."""
        avail = {p: remaining[p] for p in compat_dir(group) if remaining.get(p, 0) > 0}
        total_a = sum(avail.values())
        left = take
        for p in sorted(avail, key=lambda x: -avail[x]):
            if left == 0 or total_a == 0:
                break
            share = round(avail[p] / total_a * take)
            share = min(share, remaining[p], left)
            alloc[group][p] = alloc[group].get(p, 0) + share
            remaining[p] -= share
            left -= share
        # Assign any rounding remainder to the panning with most remaining quota.
        while left > 0:
            candidates = [(p, remaining[p]) for p in compat_dir(group) if remaining.get(p, 0) > 0]
            if not candidates:
                break
            best_p = max(candidates, key=lambda x: x[1])[0]
            alloc[group][best_p] = alloc[group].get(best_p, 0) + 1
            remaining[best_p] -= 1
            left -= 1

    for group in ordered:
        needed = actual_targets[group]
        avail_total = sum(remaining.get(p, 0) for p in compat_dir(group))
        take = min(needed, avail_total)
        if take < needed:
            overflow += needed - take
            capped.add(group)
        actual_targets[group] = take
        _fill(group, take)

    # Distribute overflow equally among uncapped groups.
    if overflow > 0:
        non_capped = [g for g in SOUND_GROUP_NAMES if g not in capped]
        if non_capped:
            per, rem = divmod(overflow, len(non_capped))
            for i, group in enumerate(non_capped):
                bonus = per + (1 if i < rem else 0)
                avail_total = sum(remaining.get(p, 0) for p in compat_dir(group))
                bonus = min(bonus, avail_total)
                actual_targets[group] += bonus
                _fill(group, bonus)

    # Print startup allocation summary.
    total_assigned = sum(sum(v.values()) for v in alloc.values())
    print(f"  Deck: {total_assigned} non-strings slots")
    if overflow > 0:
        print(f"  ⚠  Panning overflow: {overflow} slot(s) redistributed among uncapped groups")
    for g in SOUND_GROUP_NAMES:
        if g in alloc:
            print(f"    {g}: {sum(alloc[g].values())} files — {dict(alloc[g])}")

    # --- Assign (vol_label, bpm_label) within each (group, panning) cell ---
    def pan_key(panning: str) -> str:
        """Map directional panning back to the high-level key in SOUND_TYPE_RULES."""
        if panning.startswith('diagonal'):
            return 'diagonal'
        if panning in ('hard_left', 'hard_right'):
            return 'leftorright'
        return panning

    def allowed_vb(group: str, panning: str) -> set[tuple[str, str]]:
        """Allowed (vol_label, bpm_label) combos for this group + panning."""
        result: set[tuple[str, str]] = set()
        for stype in SOUND_GROUP_TYPES[group]:
            rule = _SOUND_TYPE_RULE_MAP.get(stype)
            if rule is None:
                continue
            pan_rule = rule['pannings'].get(pan_key(panning))
            if pan_rule is None:
                continue
            for vol_lbl, bpm_info in pan_rule['volumes'].items():
                for bpm_lbl in bpm_info.get('bpms', []):
                    result.add((vol_lbl, bpm_lbl))
        return result

    forced: list[tuple[str, str, str, str]] = []
    free:   list[tuple[str, str, list]]     = []
    for group, pannings in alloc.items():
        for panning, count in pannings.items():
            opts = allowed_vb(group, panning)
            if not opts:
                forced += [(group, panning, 'loud', 'slow')] * count
            elif len(opts) == 1:
                v, b = next(iter(opts))
                forced += [(group, panning, v, b)] * count
            else:
                free += [(group, panning, list(opts))] * count

    # Greedily assign free slots to approach configured BPM/volume targets.
    rem_slow  = max(0, bpm_targets.get(SLOW_BPM_IDX, 0) - sum(1 for *_, b in forced if b == 'slow'))
    rem_fast  = max(0, bpm_targets.get(FAST_BPM_IDX, 0) - sum(1 for *_, b in forced if b == 'fast'))
    rem_loud  = max(0, vol_targets.get(LOUD_VOL_IDX, 0) - sum(1 for _, _, v, _ in forced if v == 'loud'))
    rem_quiet = max(0, vol_targets.get(QUIET_VOL_IDX, 0) - sum(1 for _, _, v, _ in forced if v == 'quiet'))

    random.shuffle(free)
    assigned: list[tuple[str, str, str, str]] = []
    for group, panning, opts in free:
        best: tuple[str, str] | None = None
        best_score = -1
        for v, b in opts:
            score = (int(b == 'slow' and rem_slow > 0) + int(b == 'fast' and rem_fast > 0)
                     + int(v == 'loud' and rem_loud > 0) + int(v == 'quiet' and rem_quiet > 0))
            if score > best_score:
                best_score = score
                best = (v, b)
        assert best is not None
        v, b = best
        assigned.append((group, panning, v, b))
        if b == 'slow':    rem_slow  -= 1
        elif b == 'fast':  rem_fast  -= 1
        if v == 'loud':    rem_loud  -= 1
        elif v == 'quiet': rem_quiet -= 1

    deck = forced + assigned
    random.shuffle(deck)
    return deck


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

    # Every _strings sample is added to the playlist exactly once — no duplicates allowed.
    actual_untouched_count = sum(
        len(v) for k, v in samples_by_type.items() if is_untouched_type(k)
    )
    if actual_untouched_count > NUM_UNIQUE_SAMPLES:
        raise ValueError(
            f"Error: {actual_untouched_count} strings sample(s) found, but num_unique_samples is "
            f"{NUM_UNIQUE_SAMPLES}. Increase num_unique_samples to at least {actual_untouched_count}."
        )
    num_strings_samples = actual_untouched_count
    num_non_strings_samples = NUM_AUDIO_SAMPLES - num_strings_samples

    # Panning quotas apply to non-strings samples only; strings always go center (untouched).
    total_non_strings_input = sum(len(v) for k, v in samples_by_type.items() if not is_untouched_type(k))
    center_quota = int(num_non_strings_samples * CENTER_ONLY_WEIGHT / 100)
    diagonal_quota = int(num_non_strings_samples * DIAGONAL_WEIGHT / 100)
    dualpan_quota = int(num_non_strings_samples * DUALPAN_WEIGHT / 100)
    leftorright_quota = int(num_non_strings_samples * LEFTORRIGHT_WEIGHT / 100)

    # Cap center_quota to the number of unique non-strings files (each center combo is 1 unique slot)
    if center_quota > total_non_strings_input:
        center_overflow = center_quota - total_non_strings_input
        center_quota = total_non_strings_input
        # Redistribute overflow proportionally to diagonal, dualpan, and leftorright
        non_center_total_weight = DIAGONAL_WEIGHT + DUALPAN_WEIGHT + LEFTORRIGHT_WEIGHT
        if non_center_total_weight > 0:
            diagonal_quota += int(center_overflow * DIAGONAL_WEIGHT / non_center_total_weight)
            leftorright_quota += int(center_overflow * LEFTORRIGHT_WEIGHT / non_center_total_weight)
            dualpan_quota += center_overflow - int(center_overflow * DIAGONAL_WEIGHT / non_center_total_weight) - int(center_overflow * LEFTORRIGHT_WEIGHT / non_center_total_weight)
        else:
            dualpan_quota += center_overflow
        print(f"⚠️  Center quota capped at {total_non_strings_input} (unique non-strings files). Overflow redistributed to diagonal/dualpan/leftorright.\n")

    # Handle case where num_non_strings_samples is very small and all quotas round to 0
    total_allocated = center_quota + diagonal_quota + dualpan_quota + leftorright_quota
    remaining_samples = num_non_strings_samples - total_allocated
    if remaining_samples > 0:
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
    
    sample_round_robin, all_sample_names = build_sample_round_robin(samples_by_type)
    sample_usage_count: dict[str, int] = {s: 0 for s in all_sample_names}

    # Track combinations to ensure uniqueness
    seen_combinations: set = set()
    created_count = 0
    strings_created_count = 0
    non_strings_created_count = 0

    # Pre-compute the full file deck: BPM targets, volume targets, panning quotas, group targets.
    _bpm_targets: dict[int, int] = {
        idx: int(num_non_strings_samples * pct / 100)
        for idx, pct in enumerate(BPM_PERCENTS)
    }
    _vol_targets: dict[int, int] = {
        idx: int(num_non_strings_samples * pct / 100)
        for idx, pct in enumerate(volume_percentages)
    }
    _panning_quotas = {
        'center':      center_quota,
        'diagonal':    diagonal_quota,
        'dualpan':     dualpan_quota,
        'leftorright': leftorright_quota,
    }
    if SOUND_GROUP_PERCENTS is not None:
        _group_targets: dict[str, int] = {}
        for _gname, _pct in zip(SOUND_GROUP_NAMES, SOUND_GROUP_PERCENTS):
            _group_targets[_gname] = int(num_non_strings_samples * _pct / 100)
        _gt_total = sum(_group_targets.values())
        if _gt_total < num_non_strings_samples:
            _largest = max(SOUND_GROUP_NAMES, key=lambda g: _group_targets[g])
            _group_targets[_largest] += num_non_strings_samples - _gt_total
        _non_strings_deck = build_file_deck(
            _group_targets, _panning_quotas,
            _bpm_targets, _vol_targets,
        )
    else:
        raise ValueError("kicksnare_stab_acappella_percents must be set in config.json")

    # Interleave strings slots with the non-strings deck and shuffle.
    _strings_slots: list[tuple] = [('strings', 'untouched', 'untouched', 'untouched')] * num_strings_samples
    _deck_combined: list[tuple] = _non_strings_deck + _strings_slots
    random.shuffle(_deck_combined)

    # Telemetry counters (for end-of-run summary only).
    group_appearances: dict[str, int] = {name: 0 for name in SOUND_GROUP_NAMES}
    center_count = 0
    left_count = 0
    right_count = 0
    dualpan_count = 0
    hard_left_count = 0
    hard_right_count = 0
    volume_counts = [0] * len(volume_levels_db)

    # Get sample rate from first file
    first_sample_name = list(samples_by_type.values())[0][0]
    _, sample_rate = load_audio(first_sample_name)

    for _slot_group, _slot_panning, _slot_vol, _slot_bpm in _deck_combined:
        _is_strings = (_slot_group == 'strings')

        if _is_strings:
            primary = dequeue_next_sample(sample_round_robin, all_sample_names, require_strings=True)
            if primary is None:
                continue
            sample_names = [primary]
            pan_assignments = {primary: 'center'}
            volume_db = 0.0
            volume_idx = 0
            selected_bpm_idx = SLOW_BPM_IDX
        else:
            # Retry loop: if the first primary drawn causes a uniqueness conflict, try up to 20
            # different primaries from the same group before giving up on this slot.
            combo_key = None
            for _retry in range(20):
                primary = dequeue_next_sample_of_types(
                    sample_round_robin, all_sample_names, SOUND_GROUP_TYPES[_slot_group]
                )
                if primary is None:
                    break
                primary_type = get_sound_type(primary)
                _rule = _SOUND_TYPE_RULE_MAP.get(primary_type)

                if _slot_panning == 'dualpan':
                    _partner_types = set(_rule['dualpan_partners']) if _rule else {primary_type}
                    partner = dequeue_next_sample_of_types(
                        sample_round_robin, all_sample_names, _partner_types, exclude_name=primary
                    )
                    if partner:
                        sample_names = [primary, partner]
                        pan_assignments = {primary: 'hard_left', partner: 'hard_right'}
                    else:
                        # No partner available — fall back to solo diagonal.
                        sample_names = [primary]
                        pan_assignments = {primary: select_pan_position('left')}
                elif _slot_panning == 'center':
                    sample_names = [primary]
                    pan_assignments = {primary: 'center'}
                elif _slot_panning == 'diagonal_left':
                    sample_names = [primary]
                    pan_assignments = {primary: select_pan_position('left')}
                elif _slot_panning == 'diagonal_right':
                    sample_names = [primary]
                    pan_assignments = {primary: select_pan_position('right')}
                elif _slot_panning == 'hard_left':
                    sample_names = [primary]
                    pan_assignments = {primary: 'hard_left'}
                elif _slot_panning == 'hard_right':
                    sample_names = [primary]
                    pan_assignments = {primary: 'hard_right'}
                else:
                    sample_names = [primary]
                    pan_assignments = {primary: 'center'}

                _ckey = tuple(sorted([f"{name}:{pan_assignments[name]}" for name in sample_names]))
                if _ckey not in seen_combinations:
                    combo_key = _ckey
                    break  # Unique combo found — proceed with this slot.

            if combo_key is None:
                continue  # All retries exhausted — skip this slot.

            volume_db = volume_levels_db[LOUD_VOL_IDX] if _slot_vol == 'loud' else volume_levels_db[QUIET_VOL_IDX]
            volume_idx = LOUD_VOL_IDX if _slot_vol == 'loud' else QUIET_VOL_IDX
            selected_bpm_idx = SLOW_BPM_IDX if _slot_bpm == 'slow' else FAST_BPM_IDX

        beat_length_sec = BEAT_LENGTHS_SECONDS[selected_bpm_idx]

        # Uniqueness check for strings slots (non-strings resolved within retry loop above).
        if _is_strings:
            combo_key = tuple(sorted([f"{name}:{pan_assignments[name]}" for name in sample_names]))
            if combo_key in seen_combinations:
                continue
        seen_combinations.add(combo_key)

        # Track per-sample usage for the round-robin fairness report.
        for name in sample_names:
            sample_usage_count[name] = sample_usage_count.get(name, 0) + 1

        created_count += 1

        if _is_strings:
            strings_created_count += 1
        else:
            non_strings_created_count += 1
            # Count by the deck slot's group (not by scanning sample types, which would
            # inflate stab/kicksnare counts for dualpan partners from other groups).
            group_appearances[_slot_group] = group_appearances.get(_slot_group, 0) + 1

        # Track panning distribution for the summary report.
        pan_positions = list(pan_assignments.values())
        if len(pan_positions) == 2:
            dualpan_count += 1
        else:
            _pan = pan_positions[0]
            if _pan == 'center':
                center_count += 1
            elif _pan == 'hard_left':
                hard_left_count += 1
            elif _pan == 'hard_right':
                hard_right_count += 1
            else:
                if float(_pan) < 0:
                    left_count += 1
                else:
                    right_count += 1

        # Track volume.
        if not _is_strings:
            volume_counts[volume_idx] += 1

        # Generate and save the file.
        combined_audio = create_combination(sample_names, pan_assignments, sample_rate, volume_db, beat_length_sec)
        filename = format_filename(sample_names, pan_assignments, volume_db, created_count, BPM_VALUES[selected_bpm_idx])
        output_path = OUTPUT_DIR / filename
        sf.write(output_path, combined_audio, sample_rate)

        if created_count % 10 == 0 or created_count == NUM_AUDIO_SAMPLES:
            print(f"  Created {created_count}/{NUM_AUDIO_SAMPLES} samples...")

    if created_count < NUM_AUDIO_SAMPLES:
        print(f"\nWarning: Only created {created_count} audio samples (expected {NUM_AUDIO_SAMPLES}).")
        print("  Some deck slots were skipped because no unique (sample + pan) combination could be")
        print("  found within the retry limit. This usually means the sample pool is small relative")
        print("  to num_unique_samples. Try adding more input samples or reducing num_unique_samples.")
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
    diagonal_count = left_count + right_count
    leftorright_count = hard_left_count + hard_right_count

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
    
    # Display volume distribution
    print(f"\nVolume Distribution:")
    print(f"  Config ratio: {config.get('loud_medium_soft_percents', '100')}")
    print(f"  Config values: {config.get('loud_medium_soft_values', '0')} dB")
    
    # Calculate realized ratio
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
    for db_val, count in zip(volume_levels_db, volume_counts):
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
        for length_sec, count in zip(SILENCE_LENGTHS_SECONDS, silence_counts_by_length):
            for i in range(count):
                create_silence_file(sample_rate, length_sec, file_counter)
                file_counter += 1
                if file_counter % 10 == 1 or file_counter == NUM_SILENCE_FILES + 1:
                    print(f"    Created {file_counter - 1}/{NUM_SILENCE_FILES} silence files...", end="\r")
        
        print(f"    Created {NUM_SILENCE_FILES}/{NUM_SILENCE_FILES} silence files...")
        
        total_files_created = created_count + NUM_SILENCE_FILES
        print(f"\nTotal files created: {total_files_created} ({created_count} samples + {NUM_SILENCE_FILES} silence)")
    # (No else needed: when SILENCE_PERCENT == 0, total files == created_count.)
    
    # Display strings distribution
    if actual_untouched_count > 0:
        strings_pct = (strings_created_count / created_count) * 100 if created_count > 0 else 0
        non_strings_pct = (non_strings_created_count / created_count) * 100 if created_count > 0 else 0
        print(f"\nStrings Distribution:")
        print(f"  All {actual_untouched_count} strings sample(s) added exactly once (no duplicates).")
        print(f"  Realized: {non_strings_created_count} non-strings ({non_strings_pct:.1f}%) / {strings_created_count} strings ({strings_pct:.1f}%)")

    # Display sound group distribution (only if kicksnare_stab_acappella_percents is configured)
    if SOUND_GROUP_PERCENTS is not None and non_strings_created_count > 0:
        print(f"\nSound Group Distribution:")
        print(f"  Config: kicksnare_stab_acappella_percents = {config['kicksnare_stab_acappella_percents']}")
        for _gname, _target_pct in zip(SOUND_GROUP_NAMES, SOUND_GROUP_PERCENTS):
            _count = group_appearances[_gname]
            _realized_pct = (_count / non_strings_created_count) * 100
            print(f"  {_gname}: {_count} files ({_realized_pct:.1f}%, target {_target_pct}%)")

    print(f"\nNext: Run 2-import-duplicate-padded-samples-into-itunes-playlist.py\n")


if __name__ == "__main__":
    main()
