import json
from pathlib import Path


CONFIG_PATH = Path("./input/config/config.json")
INPUT_AUDIO_DIR = Path("./input/audio")
OUTPUT_AUDIO_DIR = Path("./output/audio")
OUTPUT_RHYTHMICIZED_AUDIO_DIR = Path("./output/rhythmicized-audio")

# ── Config JSON key names ──────────────────────────────────────────────────────
CFG_BPMS                 = 'bpms'
CFG_BPM_PERCENTS         = 'slow_to_fast_bpm_percents'
CFG_PANNING_PERCENTS     = 'center_diagonal_dualpan_left_right_percents'
CFG_SILENCE_RATIO        = 'samples_to_silence_percents'
CFG_NUM_UNIQUE_SAMPLES   = 'num_unique_samples'
CFG_SILENCE_LENGTHS_MS   = 'silence_lengths_millisec'
CFG_SILENCE_LEN_PCTS     = 'silence_lengths_percents'
CFG_LOUD_QUIET_VALUES    = 'loud_quiet_values'
CFG_LOUD_QUIET_PERCENTS  = 'loud_quiet_percents'
CFG_SOUND_GROUP_PERCENTS = 'kicksnare_stab_acappella_percents'
CFG_RHYTHM_WEIGHTS       = 'rhythm_pattern_weights'

# ── Config validation counts ───────────────────────────────────────────────────
NUM_PANNING_PERCENTS     = 5
NUM_SOUND_GROUP_PERCENTS = 3
NUM_VOLUME_VALUES        = 2


def parse_colon_ints(raw: str) -> list[int]:
    return [int(x) for x in str(raw).split(":")]


def parse_colon_floats(raw: str) -> list[float]:
    return [float(x) for x in str(raw).split(":")]


def require_sums_to_100(values: list[int], key: str) -> None:
    if sum(values) != 100:
        raise ValueError(f"{key} must sum to 100, got {sum(values)}: {':'.join(map(str, values))}")


def require_same_length(a: list, b: list, key_a: str, key_b: str) -> None:
    if len(a) != len(b):
        raise ValueError(f"{key_a} has {len(a)} values but {key_b} has {len(b)}")


def load() -> dict:
    with open(CONFIG_PATH) as f:
        raw = json.load(f)

    if CFG_BPMS not in raw:
        raise ValueError(f"'{CFG_BPMS}' is required in config.json")

    bpm_values = parse_colon_ints(raw[CFG_BPMS])
    bpm_percents = parse_colon_ints(raw.get(CFG_BPM_PERCENTS, "100"))
    require_sums_to_100(bpm_percents, CFG_BPM_PERCENTS)
    require_same_length(bpm_percents, bpm_values, CFG_BPM_PERCENTS, CFG_BPMS)

    panning_percents = parse_colon_ints(raw.get(CFG_PANNING_PERCENTS, "25:25:25:13:12"))
    if len(panning_percents) != NUM_PANNING_PERCENTS:
        raise ValueError(f"{CFG_PANNING_PERCENTS} must have exactly {NUM_PANNING_PERCENTS} values (center:diagonal:dualpan:left:right)")
    require_sums_to_100(panning_percents, CFG_PANNING_PERCENTS)

    silence_ratio = parse_colon_ints(raw.get(CFG_SILENCE_RATIO, "100:0"))
    require_sums_to_100(silence_ratio, CFG_SILENCE_RATIO)
    samples_percent = silence_ratio[0]
    silence_percent = silence_ratio[1] if len(silence_ratio) > 1 else 0

    num_unique_samples = raw[CFG_NUM_UNIQUE_SAMPLES]
    if silence_percent == 0:
        num_silence_files = 0
        num_audio_samples = num_unique_samples
    else:
        silence_fraction = silence_percent / samples_percent
        num_silence_files = round(num_unique_samples * silence_fraction / (1.0 + silence_fraction))
        num_audio_samples = num_unique_samples - num_silence_files

    silence_lengths_ms = parse_colon_ints(raw.get(CFG_SILENCE_LENGTHS_MS, "2000"))
    silence_length_percents = parse_colon_ints(raw.get(CFG_SILENCE_LEN_PCTS, "100"))
    require_sums_to_100(silence_length_percents, CFG_SILENCE_LEN_PCTS)
    require_same_length(silence_lengths_ms, silence_length_percents, CFG_SILENCE_LENGTHS_MS, CFG_SILENCE_LEN_PCTS)

    volume_levels_db = parse_colon_floats(raw.get(CFG_LOUD_QUIET_VALUES, "0:-26"))
    volume_percents = parse_colon_ints(raw.get(CFG_LOUD_QUIET_PERCENTS, "50:50"))
    if len(volume_levels_db) != NUM_VOLUME_VALUES:
        raise ValueError(f"{CFG_LOUD_QUIET_VALUES} must have exactly {NUM_VOLUME_VALUES} values (loud:quiet)")
    require_sums_to_100(volume_percents, CFG_LOUD_QUIET_PERCENTS)
    require_same_length(volume_levels_db, volume_percents, CFG_LOUD_QUIET_VALUES, CFG_LOUD_QUIET_PERCENTS)

    if CFG_SOUND_GROUP_PERCENTS not in raw:
        raise ValueError(f"{CFG_SOUND_GROUP_PERCENTS} must be set in config.json")
    sound_group_percents = parse_colon_ints(raw[CFG_SOUND_GROUP_PERCENTS])
    if len(sound_group_percents) != NUM_SOUND_GROUP_PERCENTS:
        raise ValueError(f"{CFG_SOUND_GROUP_PERCENTS} must have exactly {NUM_SOUND_GROUP_PERCENTS} values (kicksnare:stab:acappella)")
    require_sums_to_100(sound_group_percents, CFG_SOUND_GROUP_PERCENTS)

    return {
        "bpm_values": bpm_values,
        "beat_lengths_seconds": [60.0 / bpm for bpm in bpm_values],
        "bpm_percents": bpm_percents,
        "slowest_bpm_index": bpm_values.index(min(bpm_values)),
        "fastest_bpm_index": bpm_values.index(max(bpm_values)),

        "num_unique_samples": num_unique_samples,
        "num_audio_samples": num_audio_samples,
        "num_silence_files": num_silence_files,
        "samples_percent": samples_percent,
        "silence_percent": silence_percent,

        "center_weight": panning_percents[0],
        "diagonal_weight": panning_percents[1],
        "dualpan_weight": panning_percents[2],
        "left_weight": panning_percents[3],
        "right_weight": panning_percents[4],

        "silence_lengths_seconds": [ms / 1000.0 for ms in silence_lengths_ms],
        "silence_length_percents": silence_length_percents,

        "volume_levels_db": volume_levels_db,
        "volume_percents": volume_percents,
        "loudest_volume_index": volume_levels_db.index(max(volume_levels_db)),
        "quietest_volume_index": volume_levels_db.index(min(volume_levels_db)),

        "sound_group_percents": sound_group_percents,

        "rhythm_pattern_weights": raw.get(CFG_RHYTHM_WEIGHTS, {}),

        "raw": raw,
    }
