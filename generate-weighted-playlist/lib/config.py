import json
from pathlib import Path

from .constants import SOUND_GROUP_NAMES, KICKSNARE, STAB, ACAPPELLA, PERMUTATION_COMBOS_PER_SAMPLE


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
CFG_STRINGS_VOL_ADJUSTMENT = 'strings_volume_adjustment_db'
CFG_ACAPPELLA_VOL_ADJUSTMENT = 'acappella_volume_adjustment_db'
CFG_SAMPLE_BIAS             = 'sample_bias'
CFG_PERMUTATION_MODE        = 'permutation_mode'
CFG_KICK_SNARE_PERMUTATION_MODE = 'kick_snare_permutation_mode'
CFG_PERMUTATION_TOLERANCE   = 'permutation_tolerance_pct'
CFG_PERMUTATION_MAX_FILES   = 'permutation_max_files'
CFG_KS_IGNORED_CAP          = 'kicksnare_ignored_cap'
CFG_STRINGS_DUPLICATION_SUBGROUPS = 'num_times_strings_duplication_subgroups'
CFG_DUPLICATE_KICKSNARE       = 'duplicate_kicksnare'

# ── Sample-bias helpers ─────────────────────────────────────────────────────────
_VALID_BIAS_GROUPS = set(SOUND_GROUP_NAMES)

# ── Config validation counts ───────────────────────────────────────────────────
NUM_SOUND_GROUP_PERCENTS = 3
NUM_VOLUME_VALUES        = 2


def _compute_balanced_permutation_total(
    raw_counts: dict[str, int],
    group_percents: list[int],
    tolerance: float,
    max_files: int,
    ks_ignored_cap: int = 0,
) -> int:
    """Return estimated total slot count using the beat-aware trimming planner.

    Delegates to plan_permutation_trimming for an accurate estimate; falls back
    to a simple sum if the planner raises (e.g. no KS samples on disk yet).
    """
    # Lazy import to avoid circular dependency
    try:
        from .deck_builder import plan_permutation_trimming
        from .sample_queue import load_samples_grouped_by_type
        samples_by_type = load_samples_grouped_by_type(INPUT_AUDIO_DIR)
        _, m_stab, m_acap, diag = plan_permutation_trimming(
            samples_by_type, group_percents, max_files=max_files,
            tolerance=tolerance, ks_ignored_cap=ks_ignored_cap,
        )
        return diag['total_files']
    except Exception:
        # Fallback: sum raw permutation slot counts as a rough estimate
        return sum(raw_counts.values())


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
                        f"for group size ~{group_slot_count} — add more kick/snare input samples or lower biased_pool_pct"
                    )

        result[key] = parsed_entries

    return result if result else None


def load() -> dict:
    with open(CONFIG_PATH) as f:
        raw = json.load(f)

    if CFG_BPMS not in raw:
        raise ValueError(f"'{CFG_BPMS}' is required in config.json")

    if CFG_PERMUTATION_MODE not in raw:
        raise ValueError(f"'{CFG_PERMUTATION_MODE}' is required in config.json")
    perm_cfg = raw[CFG_PERMUTATION_MODE]

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
    duplicate_kicksnare = float(raw.get(CFG_DUPLICATE_KICKSNARE, 0))
    if duplicate_kicksnare < 0.0:
        raise ValueError(f"{CFG_DUPLICATE_KICKSNARE} must be >= 0, got {duplicate_kicksnare!r}")
    if bool(perm_cfg.get(CFG_KICK_SNARE_PERMUTATION_MODE, False)) and duplicate_kicksnare > 0.0:
        print(f"  [config] {CFG_DUPLICATE_KICKSNARE}={duplicate_kicksnare} ignored in permutation mode")
        duplicate_kicksnare = 0.0
    kicksnare_count = round(kicksnare_count * (1.0 + duplicate_kicksnare))
    strings_files = list(INPUT_AUDIO_DIR.glob("*_strings.wav"))
    strings_count = len(strings_files)

    strings_duplication_subgroups = raw.get(CFG_STRINGS_DUPLICATION_SUBGROUPS, {})
    if not isinstance(strings_duplication_subgroups, dict):
        raise ValueError(
            f"{CFG_STRINGS_DUPLICATION_SUBGROUPS} must be a JSON object mapping descriptor → count, "
            f"got {strings_duplication_subgroups!r}"
        )
    for key, val in strings_duplication_subgroups.items():
        if not isinstance(val, int) or val < 0:
            raise ValueError(
                f"{CFG_STRINGS_DUPLICATION_SUBGROUPS}: value for {key!r} must be 0 or a positive integer "
                f"(0 = no duplication, 1 = duplicate once = 2× total), got {val!r}"
            )
    known_descriptors = {f.stem.split('_')[1] for f in strings_files}

    # Remove any keys not found in input/audio
    stale_descriptors = set(strings_duplication_subgroups) - known_descriptors
    if stale_descriptors:
        for desc in stale_descriptors:
            del strings_duplication_subgroups[desc]
        print(
            f"  [config] Removed {len(stale_descriptors)} stale strings descriptor(s) from config.json "
            f"not found in input/audio: {sorted(stale_descriptors)}"
        )

    # Auto-add any descriptors found in input but missing from the config (default 0)
    missing_descriptors = known_descriptors - set(strings_duplication_subgroups)
    if missing_descriptors:
        for desc in sorted(missing_descriptors):
            strings_duplication_subgroups[desc] = 0
        print(
            f"  [config] Added {len(missing_descriptors)} missing strings descriptor(s) to config.json "
            f"with default value 0: {sorted(missing_descriptors)}"
        )

    if stale_descriptors or missing_descriptors:
        raw[CFG_STRINGS_DUPLICATION_SUBGROUPS] = dict(sorted(strings_duplication_subgroups.items()))
        with open(CONFIG_PATH, 'w') as _cfg_f:
            json.dump(raw, _cfg_f, indent=2)
            _cfg_f.write('\n')
        strings_duplication_subgroups = raw[CFG_STRINGS_DUPLICATION_SUBGROUPS]

    num_strings_slots = sum(
        1 + strings_duplication_subgroups.get(f.stem.split('_')[1], 0)
        for f in strings_files
    )

    if CFG_PERMUTATION_TOLERANCE not in perm_cfg:
        raise ValueError(f"{CFG_PERMUTATION_TOLERANCE} is required in config.json under {CFG_PERMUTATION_MODE}")
    if CFG_PERMUTATION_MAX_FILES not in perm_cfg:
        raise ValueError(f"{CFG_PERMUTATION_MAX_FILES} is required in config.json under {CFG_PERMUTATION_MODE}")
    if CFG_KS_IGNORED_CAP not in perm_cfg:
        raise ValueError(f"{CFG_KS_IGNORED_CAP} is required in config.json under {CFG_PERMUTATION_MODE}")
    permutation_tolerance = float(perm_cfg[CFG_PERMUTATION_TOLERANCE])
    permutation_max_files = int(perm_cfg[CFG_PERMUTATION_MAX_FILES])
    ks_ignored_cap        = int(perm_cfg[CFG_KS_IGNORED_CAP])
    if permutation_tolerance <= 0:
        raise ValueError(f"{CFG_PERMUTATION_TOLERANCE} must be a positive number, got {permutation_tolerance}")
    if permutation_max_files <= 0:
        raise ValueError(f"{CFG_PERMUTATION_MAX_FILES} must be a positive integer, got {permutation_max_files}")
    if ks_ignored_cap < 0:
        raise ValueError(f"{CFG_KS_IGNORED_CAP} must be 0 or a positive integer, got {ks_ignored_cap}")

    kick_snare_permutation_mode = bool(perm_cfg.get(CFG_KICK_SNARE_PERMUTATION_MODE, False))
    if kick_snare_permutation_mode:
        stab_count = len(
            list(INPUT_AUDIO_DIR.glob("*_kickstab.wav")) +
            list(INPUT_AUDIO_DIR.glob("*_snarestab.wav"))
        )
        acap_count = len(list(INPUT_AUDIO_DIR.glob("*_acappella.wav")))
        raw_perm_counts = {
            KICKSNARE: kicksnare_count * PERMUTATION_COMBOS_PER_SAMPLE[KICKSNARE],
            STAB:      stab_count      * PERMUTATION_COMBOS_PER_SAMPLE[STAB],
            ACAPPELLA: acap_count      * PERMUTATION_COMBOS_PER_SAMPLE[ACAPPELLA],
        }
        permutation_non_strings_samples = _compute_balanced_permutation_total(
            raw_perm_counts, sound_group_percents,
            tolerance=permutation_tolerance, max_files=permutation_max_files,
            ks_ignored_cap=ks_ignored_cap,
        )
        num_audio_samples = permutation_non_strings_samples + num_strings_slots
    else:
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

    strings_volume_adjustment = raw.get(CFG_STRINGS_VOL_ADJUSTMENT, 0)
    if not isinstance(strings_volume_adjustment, (int, float)):
        raise ValueError(f"{CFG_STRINGS_VOL_ADJUSTMENT} must be a number (dB), got {strings_volume_adjustment!r}")

    acappella_volume_adjustment = raw.get(CFG_ACAPPELLA_VOL_ADJUSTMENT, 0)
    if not isinstance(acappella_volume_adjustment, (int, float)):
        raise ValueError(f"{CFG_ACAPPELLA_VOL_ADJUSTMENT} must be a number (dB), got {acappella_volume_adjustment!r}")

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

        "strings_volume_adjustment_db": strings_volume_adjustment,
        "acappella_volume_adjustment_db": acappella_volume_adjustment,

        "sample_bias": sample_bias,

        "kick_snare_permutation_mode": kick_snare_permutation_mode,
        "permutation_tolerance_pct": permutation_tolerance,
        "permutation_max_files": permutation_max_files,
        "kicksnare_ignored_cap": ks_ignored_cap,
        "strings_duplication_subgroups": strings_duplication_subgroups,

        "raw": raw,
    }
