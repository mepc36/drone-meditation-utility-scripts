import json
from pathlib import Path


CONFIG_PATH = Path("./input/config/config.json")
INPUT_AUDIO_DIR = Path("./input/audio")
OUTPUT_AUDIO_DIR = Path("./output/audio")
OUTPUT_RHYTHMICIZED_AUDIO_DIR = Path("./output/rhythmicized-audio")


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

    if "bpms" not in raw:
        raise ValueError("'bpms' is required in config.json")

    bpm_values = parse_colon_ints(raw["bpms"])
    bpm_percents = parse_colon_ints(raw.get("slow_to_fast_bpm_percents", "100"))
    require_sums_to_100(bpm_percents, "slow_to_fast_bpm_percents")
    require_same_length(bpm_percents, bpm_values, "slow_to_fast_bpm_percents", "bpms")

    panning_percents = parse_colon_ints(raw.get("center_diagonal_dualpan_leftorright_percents", "25:25:25:25"))
    if len(panning_percents) != 4:
        raise ValueError("center_diagonal_dualpan_leftorright_percents must have exactly 4 values (center:diagonal:dualpan:leftorright)")
    require_sums_to_100(panning_percents, "center_diagonal_dualpan_leftorright_percents")

    silence_ratio = parse_colon_ints(raw.get("samples_to_silence_percents", "100:0"))
    require_sums_to_100(silence_ratio, "samples_to_silence_percents")
    samples_percent = silence_ratio[0]
    silence_percent = silence_ratio[1] if len(silence_ratio) > 1 else 0

    num_unique_samples = raw["num_unique_samples"]
    if silence_percent == 0:
        num_silence_files = 0
        num_audio_samples = num_unique_samples
    else:
        silence_fraction = silence_percent / samples_percent
        num_silence_files = round(num_unique_samples * silence_fraction / (1.0 + silence_fraction))
        num_audio_samples = num_unique_samples - num_silence_files

    silence_lengths_ms = parse_colon_ints(raw.get("silence_lengths_millisec", "2000"))
    silence_length_percents = parse_colon_ints(raw.get("silence_lengths_percents", "100"))
    require_sums_to_100(silence_length_percents, "silence_lengths_percents")
    require_same_length(silence_lengths_ms, silence_length_percents, "silence_lengths_millisec", "silence_lengths_percents")

    volume_levels_db = parse_colon_floats(raw.get("loud_medium_soft_values", "0"))
    volume_percents = parse_colon_ints(raw.get("loud_medium_soft_percents", "100"))
    require_sums_to_100(volume_percents, "loud_medium_soft_percents")
    require_same_length(volume_levels_db, volume_percents, "loud_medium_soft_values", "loud_medium_soft_percents")

    if "kicksnare_stab_acappella_percents" not in raw:
        raise ValueError("kicksnare_stab_acappella_percents must be set in config.json")
    sound_group_percents = parse_colon_ints(raw["kicksnare_stab_acappella_percents"])
    if len(sound_group_percents) != 3:
        raise ValueError("kicksnare_stab_acappella_percents must have exactly 3 values (kicksnare:stab:acappella)")
    require_sums_to_100(sound_group_percents, "kicksnare_stab_acappella_percents")

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
        "leftorright_weight": panning_percents[3],

        "silence_lengths_seconds": [ms / 1000.0 for ms in silence_lengths_ms],
        "silence_length_percents": silence_length_percents,

        "volume_levels_db": volume_levels_db,
        "volume_percents": volume_percents,
        "loudest_volume_index": volume_levels_db.index(max(volume_levels_db)),
        "quietest_volume_index": volume_levels_db.index(min(volume_levels_db)),

        "sound_group_percents": sound_group_percents,

        "rhythmicize_output_samples": bool(raw.get("rhythmicize_output_samples", False)),

        "raw": raw,
    }
