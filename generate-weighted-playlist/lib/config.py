import json
from pathlib import Path

from .constants import SOUND_GROUP_NAMES


CONFIG_PATH = Path("./input/config/config.json")
INPUT_AUDIO_DIR = Path("./input/audio")
OUTPUT_AUDIO_DIR = Path("./output/audio")
OUTPUT_RHYTHMICIZED_AUDIO_DIR = Path("./output/rhythmicized-audio")

# ── Config JSON key names ──────────────────────────────────────────────────────
CFG_BPMS                 = 'bpms'
CFG_SILENCE_RATIO        = 'samples_to_silence_percents'
CFG_NUM_UNIQUE_SAMPLES   = 'num_unique_samples'
CFG_SILENCE_LENGTHS_MS   = 'silence_lengths_millisec'
CFG_SILENCE_LEN_PCTS     = 'silence_lengths_percents'
CFG_LOUD_QUIET_VALUES    = 'loud_quiet_values'
CFG_SOUND_GROUP_PERCENTS = 'kicksnare_stab_acappella_percents'
CFG_STRINGS_VOL_REDUCTION = 'strings_volume_reduction'
CFG_ACAPPELLA_VOL_REDUCTION = 'acappella_volume_reduction'
CFG_SAMPLE_BIAS             = 'sample_bias'

# ── Sample-bias helpers ─────────────────────────────────────────────────────────
_VALID_BIAS_GROUPS = set(SOUND_GROUP_NAMES)

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


def parse_sample_bias(
    raw: dict,
    sound_group_percents: list[int],
    num_audio_samples: int,
) -> dict | None:
    if CFG_SAMPLE_BIAS not in raw:
        return None
    bias_raw = raw[CFG_SAMPLE_BIAS]
    if not bias_raw.get('is_sample_bias_enabled', False):
        return None

    group_pct_map = dict(zip(SOUND_GROUP_NAMES, sound_group_percents))
    result: dict[str, list[dict]] = {}

    for key, entries in bias_raw.items():
        if key == 'is_sample_bias_enabled':
            continue
        if key not in _VALID_BIAS_GROUPS:
            raise ValueError(
                f"sample_bias contains unknown group '{key}'. "
                f"Must be one of: {sorted(_VALID_BIAS_GROUPS)}"
            )
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"sample_bias.{key} must be a non-empty list")

        total_pct = 0
        has_unbiased = False
        parsed_entries: list[dict] = []
        seen_biased_samples: set[str] = set()
        for entry in entries:
            if 'unbiased_pool_pct' in entry:
                if has_unbiased:
                    raise ValueError(f"sample_bias.{key} has more than one unbiased_pool_pct entry")
                pct_val = entry['unbiased_pool_pct']
                if not isinstance(pct_val, int) or pct_val <= 0:
                    raise ValueError(
                        f"sample_bias.{key} unbiased_pool_pct must be a positive integer, got {pct_val!r}"
                    )
                has_unbiased = True
                total_pct += pct_val
                parsed_entries.append({'unbiased_pool_pct': pct_val})
            elif entry.get('is_random') is True:
                if 'biased_pool_pct' not in entry:
                    raise ValueError(
                        f"sample_bias.{key} is_random entry is missing 'biased_pool_pct'"
                    )
                pct_val = entry['biased_pool_pct']
                if not isinstance(pct_val, int) or pct_val <= 0:
                    raise ValueError(
                        f"sample_bias.{key} is_random biased_pool_pct must be a positive integer, got {pct_val!r}"
                    )
                if 'include' in entry and 'exclude' in entry:
                    raise ValueError(
                        f"sample_bias.{key} is_random entry cannot specify both 'include' and 'exclude'"
                    )
                include_all = entry.get('include_all', False)
                if include_all and 'exclude' in entry:
                    raise ValueError(
                        f"sample_bias.{key} is_random entry cannot specify both 'include_all' and 'exclude'"
                    )
                for filter_key in ('include', 'exclude'):
                    if filter_key in entry:
                        if not isinstance(entry[filter_key], list):
                            raise ValueError(
                                f"sample_bias.{key} is_random entry '{filter_key}' must be a list"
                            )
                        if not include_all and len(entry[filter_key]) == 0:
                            raise ValueError(
                                f"sample_bias.{key} is_random entry '{filter_key}' must not be empty — "
                                f"omit the key entirely to allow the full group pool"
                            )
                total_pct += pct_val
                parsed_entry: dict = {
                    'is_random': True,
                    'biased_pool_pct': pct_val,
                }
                if include_all:
                    parsed_entry['include_all'] = True
                elif 'include' in entry:
                    parsed_entry['include'] = list(entry['include'])
                if 'exclude' in entry:
                    parsed_entry['exclude'] = list(entry['exclude'])
                parsed_entries.append(parsed_entry)
            elif 'biased_sample' in entry and 'biased_pool_pct' in entry:
                pct_val = entry['biased_pool_pct']
                if not isinstance(pct_val, int) or pct_val <= 0:
                    raise ValueError(
                        f"sample_bias.{key} biased_pool_pct must be a positive integer, got {pct_val!r}"
                    )
                sample_name = entry['biased_sample']
                if sample_name in seen_biased_samples:
                    raise ValueError(
                        f"sample_bias.{key} biased_sample '{sample_name}' appears more than once "
                        f"in the same group"
                    )
                seen_biased_samples.add(sample_name)
                total_pct += pct_val
                parsed_entries.append({
                    'biased_sample': sample_name,
                    'biased_pool_pct': pct_val,
                })
            else:
                raise ValueError(
                    f"sample_bias.{key} entry must have 'biased_sample' + 'biased_pool_pct', "
                    f"'is_random: true' + 'biased_pool_pct', "
                    f"or 'unbiased_pool_pct', got: {entry!r}"
                )

        if total_pct != 100:
            raise ValueError(
                f"sample_bias.{key} percents must sum to 100, got {total_pct}"
            )

        group_slot_count = round(num_audio_samples * group_pct_map.get(key, 0) / 100)
        if group_slot_count == 0:
            raise ValueError(
                f"sample_bias.{key} is configured but '{key}' has 0% in "
                f"{CFG_SOUND_GROUP_PERCENTS} — either remove the bias entry or increase its percent"
            )
        for entry in parsed_entries:
            if 'biased_pool_pct' in entry:
                slots = round(group_slot_count * entry['biased_pool_pct'] / 100)
                if slots < 1:
                    raise ValueError(
                        f"sample_bias.{key} biased_pool_pct {entry['biased_pool_pct']} yields 0 slots "
                        f"for group size ~{group_slot_count} — increase num_unique_samples or lower biased_pool_pct"
                    )

        result[key] = parsed_entries

    return result if result else None


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
    if len(volume_levels_db) != NUM_VOLUME_VALUES:
        raise ValueError(f"{CFG_LOUD_QUIET_VALUES} must have exactly {NUM_VOLUME_VALUES} values (loud:quiet)")
    volume_levels_db = sorted(volume_levels_db, reverse=True)  # loud→quiet (high dB first)

    if CFG_SOUND_GROUP_PERCENTS not in raw:
        raise ValueError(f"{CFG_SOUND_GROUP_PERCENTS} must be set in config.json")
    sound_group_percents = parse_colon_ints(raw[CFG_SOUND_GROUP_PERCENTS])
    if len(sound_group_percents) != NUM_SOUND_GROUP_PERCENTS:
        raise ValueError(f"{CFG_SOUND_GROUP_PERCENTS} must have exactly {NUM_SOUND_GROUP_PERCENTS} values (kicksnare:stab:acappella)")
    require_sums_to_100(sound_group_percents, CFG_SOUND_GROUP_PERCENTS)

    strings_volume_reduction = raw.get(CFG_STRINGS_VOL_REDUCTION, 0)
    if not isinstance(strings_volume_reduction, int) or strings_volume_reduction < 0:
        raise ValueError(f"{CFG_STRINGS_VOL_REDUCTION} must be a non-negative integer, got {strings_volume_reduction!r}")

    acappella_volume_reduction = raw.get(CFG_ACAPPELLA_VOL_REDUCTION, 0)
    if not isinstance(acappella_volume_reduction, int) or acappella_volume_reduction < 0:
        raise ValueError(f"{CFG_ACAPPELLA_VOL_REDUCTION} must be a non-negative integer, got {acappella_volume_reduction!r}")

    sample_bias = parse_sample_bias(raw, sound_group_percents, num_audio_samples)

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

        "strings_volume_reduction": strings_volume_reduction,
        "acappella_volume_reduction": acappella_volume_reduction,

        "sample_bias": sample_bias,

        "raw": raw,
    }
