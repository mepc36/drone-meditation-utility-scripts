import random
from collections import defaultdict, deque
from pathlib import Path

from .constants import KICK, SNARE, KICKSTAB, SNARESTAB, ACAPPELLA, STRINGS, SOUND_GROUP_TYPES
from .sound_rules import sound_type_of

_VALID_SOUND_TYPES = {KICK, SNARE, KICKSTAB, SNARESTAB, ACAPPELLA, STRINGS}


def load_samples_grouped_by_type(input_audio_dir: Path) -> dict[str, list[str]]:
    if not input_audio_dir.exists():
        raise FileNotFoundError(f"Input audio directory not found: {input_audio_dir}")
    wav_files = list(input_audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {input_audio_dir}")
    grouped: dict[str, list[str]] = defaultdict(list)
    for f in wav_files:
        sound_type = sound_type_of(f.stem)
        if sound_type not in _VALID_SOUND_TYPES:
            raise ValueError(
                f"Unrecognized sound type {sound_type!r} derived from file {f.name!r}. "
                f"Must be one of: {sorted(_VALID_SOUND_TYPES)}"
            )
        grouped[sound_type].append(f.stem)
    return dict(grouped)


def create_shuffled_sample_queue(samples_by_type: dict[str, list[str]]) -> tuple[deque, list[str]]:
    all_samples = [s for samples in samples_by_type.values() for s in samples]
    return deque(random.sample(all_samples, len(all_samples))), all_samples


def _refill_queue(queue: deque, all_samples: list[str]) -> None:
    shuffled = list(all_samples)
    random.shuffle(shuffled)
    queue.extend(shuffled)


def _draw_from_queue_with_filter(queue: deque, all_samples: list[str], predicate) -> str | None:
    if not any(predicate(s) for s in all_samples):
        return None
    for _ in range(2):
        if not queue:
            _refill_queue(queue, all_samples)
        for i in range(len(queue)):
            if predicate(queue[i]):
                sample = queue[i]
                del queue[i]
                return sample
        _refill_queue(queue, all_samples)
    return None


def draw_next_strings_sample(queue: deque, all_samples: list[str]) -> str | None:
    return _draw_from_queue_with_filter(
        queue, all_samples,
        lambda s: sound_type_of(s) == STRINGS,
    )


def draw_next_sample_of_types(
    queue: deque,
    all_samples: list[str],
    allowed_types: set[str],
    exclude_name: str | None = None,
) -> str | None:
    return _draw_from_queue_with_filter(
        queue, all_samples,
        lambda s: sound_type_of(s) in allowed_types and s != exclude_name,
    )


def validate_sample_bias(
    bias_config: dict,
    samples_by_type: dict[str, list[str]],
) -> None:
    all_samples_flat = {s for samples in samples_by_type.values() for s in samples}
    for group, entries in bias_config.items():
        valid_types = SOUND_GROUP_TYPES.get(group, set())
        all_samples_in_group = {s for t in valid_types for s in samples_by_type.get(t, [])}
        for entry in entries:
            if 'biased_sample' in entry:
                sample_name = entry['biased_sample']
                if sample_name not in all_samples_in_group:
                    if sample_name in all_samples_flat:
                        raise ValueError(
                            f"sample_bias.{group} biased_sample '{sample_name}' exists but does not belong "
                            f"to the '{group}' group (valid types: {sorted(valid_types)})"
                        )
                    raise ValueError(
                        f"sample_bias.{group} biased_sample '{sample_name}' not found in input/audio/"
                    )
            if entry.get('is_random'):
                for filter_key in ('include', 'exclude'):
                    for sample_name in entry.get(filter_key, []):
                        if sample_name not in all_samples_in_group:
                            if sample_name in all_samples_flat:
                                raise ValueError(
                                    f"sample_bias.{group} is_random {filter_key} '{sample_name}' exists "
                                    f"but does not belong to the '{group}' group (valid types: {sorted(valid_types)})"
                                )
                            raise ValueError(
                                f"sample_bias.{group} is_random {filter_key} '{sample_name}' not found in input/audio/"
                            )
                pool = _random_eligible_pool(entry, all_samples_in_group)
                if not pool:
                    raise ValueError(
                        f"sample_bias.{group} is_random entry has an empty eligible pool after applying "
                        f"include/exclude filters — no sample can be randomly selected"
                    )


def _random_eligible_pool(entry: dict, all_samples_in_group: set[str]) -> list[str]:
    if entry.get('include_all'):
        return list(all_samples_in_group)
    if 'include' in entry:
        return [s for s in entry['include'] if s in all_samples_in_group]
    if 'exclude' in entry:
        excluded = set(entry['exclude'])
        return [s for s in all_samples_in_group if s not in excluded]
    return list(all_samples_in_group)


def resolve_random_entries(
    bias_config: dict,
    samples_by_type: dict[str, list[str]],
) -> dict:
    """Return a copy of bias_config with all is_random entries resolved to concrete biased_sample entries."""
    resolved: dict[str, list[dict]] = {}
    for group, entries in bias_config.items():
        valid_types = SOUND_GROUP_TYPES.get(group, set())
        all_samples_in_group = {s for t in valid_types for s in samples_by_type.get(t, [])}
        already_chosen: set[str] = {e['biased_sample'] for e in entries if 'biased_sample' in e}
        new_entries: list[dict] = []
        for entry in entries:
            if not entry.get('is_random'):
                new_entries.append(entry)
                continue
            pool = [s for s in _random_eligible_pool(entry, all_samples_in_group) if s not in already_chosen]
            if not pool:
                raise ValueError(
                    f"sample_bias.{group} is_random entry has no remaining candidates after "
                    f"excluding already-chosen samples {sorted(already_chosen)}"
                )
            chosen = random.choice(pool)
            already_chosen.add(chosen)
            new_entries.append({'biased_sample': chosen, 'biased_pool_pct': entry['biased_pool_pct'], 'was_random': True})
        resolved[group] = new_entries
    return resolved


def build_biased_reservations(
    bias_config: dict,
    group_targets: dict[str, int],
) -> dict[str, deque]:
    reservations: dict[str, deque] = {}
    for group, entries in bias_config.items():
        total_slots = group_targets.get(group, 0)
        if total_slots == 0:
            continue
        reservation_list: list[str | None] = []
        biased_total = 0
        for entry in entries:
            if 'biased_sample' in entry:
                count = round(total_slots * entry['biased_pool_pct'] / 100)
                reservation_list.extend([entry['biased_sample']] * count)
                biased_total += count
        unbiased_count = total_slots - biased_total
        reservation_list.extend([None] * unbiased_count)
        random.shuffle(reservation_list)
        reservations[group] = deque(reservation_list)
    return reservations
