import json
from pathlib import Path


CONFIG_PATH = Path("./input/config/config.json")
INPUT_AUDIO_DIR = Path("./input/audio")
OUTPUT_AUDIO_DIR = Path("./output/audio")
OUTPUT_RHYTHMICIZED_AUDIO_DIR = Path("./output/rhythmicized-audio")

# ── Config JSON key names ──────────────────────────────────────────────────────
CFG_BPMS                    = 'bpms'
CFG_SILENCE_RATIO           = 'samples_to_silence_percents'
CFG_SILENCE_LENGTHS_MS      = 'silence_lengths_millisec'
CFG_SILENCE_LEN_PCTS        = 'silence_lengths_percents'
CFG_LOUD_QUIET_VALUES       = 'loud_quiet_values'
CFG_SOUND_GROUP_PERCENTS    = 'kicksnare_stab_acappella_percents'
CFG_STRINGS_NONSTRINGS_PCTS = 'strings_nonstrings_pcts'
# ── Config validation counts ───────────────────────────────────────────────────
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

    bpm_values = sorted(parse_colon_floats(raw[CFG_BPMS]))  # slow→fast

    silence_ratio = parse_colon_ints(raw.get(CFG_SILENCE_RATIO, "100:0"))
    require_sums_to_100(silence_ratio, CFG_SILENCE_RATIO)
    samples_percent = silence_ratio[0]
    silence_percent = silence_ratio[1] if len(silence_ratio) > 1 else 0

    if CFG_SOUND_GROUP_PERCENTS not in raw:
        raise ValueError(f"{CFG_SOUND_GROUP_PERCENTS} must be set in config.json")
    sound_group_percents = parse_colon_ints(raw[CFG_SOUND_GROUP_PERCENTS])
    if len(sound_group_percents) != NUM_SOUND_GROUP_PERCENTS:
        raise ValueError(f"{CFG_SOUND_GROUP_PERCENTS} must have exactly {NUM_SOUND_GROUP_PERCENTS} values (kicksnare:stab:acappella)")
    require_sums_to_100(sound_group_percents, CFG_SOUND_GROUP_PERCENTS)

    kicksnare_pct = sound_group_percents[0]
    if kicksnare_pct == 0:
        raise ValueError("kicksnare percent cannot be 0 — cannot derive total sample count from kicksnare files")
    kicksnare_files = list(INPUT_AUDIO_DIR.glob("*_kick.wav")) + list(INPUT_AUDIO_DIR.glob("*_snare.wav"))
    kicksnare_count = len(kicksnare_files)
    if kicksnare_count == 0:
        raise ValueError(f"No kick/snare files found in {INPUT_AUDIO_DIR}. Cannot determine sample count.")
    num_audio_samples = round(kicksnare_count * 100 / kicksnare_pct)

    if silence_percent == 0:
        num_silence_files = 0
    else:
        num_silence_files = round(num_audio_samples * silence_percent / samples_percent)
    num_unique_samples = num_audio_samples + num_silence_files

    silence_lengths_ms = parse_colon_ints(raw.get(CFG_SILENCE_LENGTHS_MS, "2000"))
    silence_length_percents = parse_colon_ints(raw.get(CFG_SILENCE_LEN_PCTS, "100"))
    require_sums_to_100(silence_length_percents, CFG_SILENCE_LEN_PCTS)
    require_same_length(silence_lengths_ms, silence_length_percents, CFG_SILENCE_LENGTHS_MS, CFG_SILENCE_LEN_PCTS)

    volume_levels_db = parse_colon_floats(raw.get(CFG_LOUD_QUIET_VALUES, "0:-26"))
    if len(volume_levels_db) != NUM_VOLUME_VALUES:
        raise ValueError(f"{CFG_LOUD_QUIET_VALUES} must have exactly {NUM_VOLUME_VALUES} values (loud:quiet)")
    volume_levels_db = sorted(volume_levels_db, reverse=True)  # loud→quiet (high dB first)

    strings_nonstrings_pcts = None
    if CFG_STRINGS_NONSTRINGS_PCTS in raw:
        snp = parse_colon_floats(raw[CFG_STRINGS_NONSTRINGS_PCTS])
        if len(snp) != 2:
            raise ValueError(f"{CFG_STRINGS_NONSTRINGS_PCTS} must have exactly 2 values (strings:nonstrings)")
        if snp[1] == 0:
            raise ValueError(f"nonstrings percent in {CFG_STRINGS_NONSTRINGS_PCTS} cannot be 0")
        strings_nonstrings_pcts = snp

    return {
        "bpm_values": bpm_values,
        "beat_lengths_seconds": [60.0 / bpm for bpm in bpm_values],
        "slowest_bpm_index": bpm_values.index(min(bpm_values)),
        "fastest_bpm_index": bpm_values.index(max(bpm_values)),

        "num_unique_samples": num_unique_samples,
        "num_audio_samples": num_audio_samples,
        "num_silence_files": num_silence_files,
        "samples_percent": samples_percent,
        "silence_percent": silence_percent,

        "silence_lengths_seconds": [ms / 1000.0 for ms in silence_lengths_ms],
        "silence_length_percents": silence_length_percents,

        "volume_levels_db": volume_levels_db,
        "loudest_volume_index": volume_levels_db.index(max(volume_levels_db)),
        "quietest_volume_index": volume_levels_db.index(min(volume_levels_db)),

        "sound_group_percents": sound_group_percents,
        "strings_nonstrings_pcts": strings_nonstrings_pcts,

        "raw": raw,
    }
