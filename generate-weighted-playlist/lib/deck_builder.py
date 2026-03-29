import random
import dataclasses
from dataclasses import dataclass

from .constants import (
    HARD_CENTER, HARD_LEFT, HARD_RIGHT, DIAGONAL_LEFT, DIAGONAL_RIGHT,
    DUALPAN, UNTOUCHED,
    LOUD, QUIET, SLOW, FAST,
    SOUND_GROUP_NAMES, SOUND_GROUP_TYPES,
)
from .sound_rules import derive_type, panning_compat, pattern_weights, rules_by_sound_type


@dataclass(frozen=True)
class SlotSpec:
    sound_group: str
    panning: float | str       # float position or 'dualpan' / 'untouched'
    volume_label: float | str  # actual dB float, or 'untouched' for strings slots
    bpm_label: int | str       # actual BPM int, or 'untouched' for strings slots
    rhythm: tuple[float | str, ...] = ()
    beat_pannings: tuple[float | str, ...] = ()  # per-beat chosen pannings, parallel to rhythm


def _expand_directional_quotas(panning_quotas: dict[str, int]) -> dict:
    diagonal = panning_quotas.get('diagonal', 0)
    return {
        HARD_CENTER:    panning_quotas.get('center', 0),
        DIAGONAL_LEFT:  diagonal // 2,
        DIAGONAL_RIGHT: diagonal - diagonal // 2,
        DUALPAN:        panning_quotas.get('dualpan', 0),
        HARD_LEFT:      panning_quotas.get('left', 0),
        HARD_RIGHT:     panning_quotas.get('right', 0),
    }


def _directional_pannings_for_group(group: str) -> set:
    # panning_compat already contains concrete numeric panning values
    return panning_compat.get(group, set())


def _allocate_panning_slots(group_targets: dict[str, int], available_slots: dict[str, int]) -> tuple[dict[str, dict[str, int]], int]:
    allocation: dict[str, dict[str, int]] = {g: {} for g in group_targets}
    actual_targets = dict(group_targets)
    overflow = 0
    capped: set[str] = set()

    ordered_by_most_constrained = sorted(
        group_targets,
        key=lambda g: sum(available_slots.get(p, 0) for p in _directional_pannings_for_group(g)),
    )

    def fill_group(group: str, slots_needed: int) -> None:
        compatible = {p: available_slots[p] for p in _directional_pannings_for_group(group) if available_slots.get(p, 0) > 0}
        total_available = sum(compatible.values())
        remaining = slots_needed
        for pan in sorted(compatible, key=lambda x: -compatible[x]):
            if remaining == 0 or total_available == 0:
                break
            share = min(round(compatible[pan] / total_available * slots_needed), available_slots[pan], remaining)
            allocation[group][pan] = allocation[group].get(pan, 0) + share
            available_slots[pan] -= share
            remaining -= share
        while remaining > 0:
            candidates = [(p, available_slots[p]) for p in _directional_pannings_for_group(group) if available_slots.get(p, 0) > 0]
            if not candidates:
                break
            best_pan = max(candidates, key=lambda x: x[1])[0]
            allocation[group][best_pan] = allocation[group].get(best_pan, 0) + 1
            available_slots[best_pan] -= 1
            remaining -= 1

    for group in ordered_by_most_constrained:
        needed = actual_targets[group]
        can_fill = sum(available_slots.get(p, 0) for p in _directional_pannings_for_group(group))
        take = min(needed, can_fill)
        if take < needed:
            overflow += needed - take
            capped.add(group)
        actual_targets[group] = take
        fill_group(group, take)

    if overflow > 0:
        uncapped_groups = [g for g in SOUND_GROUP_NAMES if g not in capped]
        per_group, remainder = divmod(overflow, len(uncapped_groups)) if uncapped_groups else (0, 0)
        for i, group in enumerate(uncapped_groups):
            bonus = min(per_group + (1 if i < remainder else 0),
                        sum(available_slots.get(p, 0) for p in _directional_pannings_for_group(group)))
            actual_targets[group] += bonus
            fill_group(group, bonus)

    return allocation, overflow


def _hashable_beat(b) -> tuple:
    """Convert a beat dict to a hashable (duration, (panning, ...)) tuple."""
    if not isinstance(b, dict):
        raise TypeError(f"Beat must be a dict with 'musical_duration' and 'possible_pannings', got {b!r}")
    return (float(b['musical_duration']), tuple(b['possible_pannings']))


def _extract_rhythm_and_pannings(raw_pattern: tuple) -> tuple[tuple, tuple]:
    """Split a raw (hashable) beat pattern into separate duration and panning tuples.

    Format: (type_str, (duration, (pannings,...)), ...).
    A panning is randomly chosen from the candidate list for each beat.
    """
    if raw_pattern == (UNTOUCHED,):
        return (UNTOUCHED,), ()
    durations: list = []
    pannings: list = []
    for b in raw_pattern[1:]:  # skip the type string at index 0
        if not (isinstance(b, tuple) and len(b) == 2 and isinstance(b[1], tuple)):
            raise TypeError(f"Hashed beat must be a (duration, (pannings,...)) tuple, got {b!r}")
        duration, pan_options = b
        durations.append(float(duration))
        pannings.append(random.choice(pan_options) if pan_options else '')
    return tuple(durations), tuple(pannings)



def _allowed_volume_bpm_combos(group: str, panning: float | None) -> set[tuple]:
    result: set[tuple[str, str]] = set()
    for sound_type in SOUND_GROUP_TYPES[group]:
        rule = rules_by_sound_type.get(sound_type)
        if rule is None:
            continue
        pan_rule = rule['musical_patterns'].get(panning)
        if pan_rule is None:
            continue
        for vol_label in pan_rule['volumes']:
            for bpm_label in pan_rule['bpms']:
                result.add((vol_label, bpm_label))
    return result


def _assign_volume_and_bpm(
    allocation: dict[str, dict[str, int]],
    slowest_bpm_index: int,
    fastest_bpm_index: int,
    loudest_volume_index: int,
    quietest_volume_index: int,
    bpm_targets: dict[int, int],
    vol_targets: dict[int, int],
) -> list[SlotSpec]:
    forced: list[SlotSpec] = []
    free: list[tuple[str, str, list]] = []

    for group, pannings in allocation.items():
        for panning, count in pannings.items():
            opts = _allowed_volume_bpm_combos(group, panning)
            if not opts:
                forced += [SlotSpec(group, panning, LOUD, SLOW)] * count
            elif len(opts) == 1:
                vol, bpm = next(iter(opts))
                forced += [SlotSpec(group, panning, vol, bpm)] * count
            else:
                free += [(group, panning, list(opts))] * count

    remaining_slow  = max(0, bpm_targets.get(slowest_bpm_index, 0)   - sum(1 for s in forced if s.bpm_label == SLOW))
    remaining_fast  = max(0, bpm_targets.get(fastest_bpm_index, 0)    - sum(1 for s in forced if s.bpm_label == FAST))
    remaining_loud  = max(0, vol_targets.get(loudest_volume_index, 0) - sum(1 for s in forced if s.volume_label == LOUD))
    remaining_quiet = max(0, vol_targets.get(quietest_volume_index, 0)- sum(1 for s in forced if s.volume_label == QUIET))

    random.shuffle(free)
    assigned: list[SlotSpec] = []
    for group, panning, opts in free:
        best_combo = max(
            opts,
            key=lambda vb: (
                int(vb[1] == SLOW  and remaining_slow  > 0) +
                int(vb[1] == FAST  and remaining_fast  > 0) +
                int(vb[0] == LOUD  and remaining_loud  > 0) +
                int(vb[0] == QUIET and remaining_quiet > 0)
            ),
        )
        vol, bpm = best_combo
        assigned.append(SlotSpec(group, panning, vol, bpm))
        if bpm == SLOW:    remaining_slow  -= 1
        elif bpm == FAST:  remaining_fast  -= 1
        if vol == LOUD:    remaining_loud  -= 1
        elif vol == QUIET: remaining_quiet -= 1

    return forced + assigned


def _rhythm_patterns_for_slot(slot: SlotSpec) -> list[tuple]:
    """Return unique hashable rhythm patterns for this slot from sound_rules.
    Each tuple starts with a type string (for weight lookup), followed by
    (duration, (pannings,...)) beat tuples."""
    seen: set[tuple] = set()
    found: list[tuple] = []
    for sound_type in SOUND_GROUP_TYPES.get(slot.sound_group, set()):
        rule = rules_by_sound_type.get(sound_type)
        if rule is None:
            continue
        pan_rule = rule['musical_patterns'].get(slot.panning)
        if pan_rule is None:
            continue
        if slot.volume_label not in pan_rule['volumes']:
            continue
        if slot.bpm_label not in pan_rule['bpms']:
            continue
        for entry in pan_rule.get('rhythm_patterns', []):
            if entry is UNTOUCHED:
                pat = (UNTOUCHED,)
            elif isinstance(entry, list):
                pat = (derive_type(entry),) + tuple(_hashable_beat(b) for b in entry)
            else:
                raise TypeError(f"Unexpected rhythm_pattern entry: {entry!r}")
            if pat not in seen:
                seen.add(pat)
                found.append(pat)
    return found


def plan_output_files(
    group_targets: dict[str, int],
    panning_quotas: dict[str, int],
    bpm_targets: dict[int, int],
    vol_targets: dict[int, int],
    slowest_bpm_index: int,
    fastest_bpm_index: int,
    loudest_volume_index: int,
    quietest_volume_index: int,
) -> list[SlotSpec]:
    available_slots = _expand_directional_quotas(panning_quotas)
    allocation, overflow = _allocate_panning_slots(group_targets, available_slots)

    total_slots = sum(sum(v.values()) for v in allocation.values())
    print(f"  Deck: {total_slots} non-strings slots")
    if overflow > 0:
        print(f"  ⚠  Panning overflow: {overflow} slot(s) redistributed among uncapped groups")
    for group in SOUND_GROUP_NAMES:
        if group in allocation:
            print(f"    {group}: {sum(allocation[group].values())} files — {dict(allocation[group])}")

    deck = _assign_volume_and_bpm(
        allocation,
        slowest_bpm_index, fastest_bpm_index,
        loudest_volume_index, quietest_volume_index,
        bpm_targets, vol_targets,
    )
    random.shuffle(deck)

    # Assign rhythm patterns from sound rules, distributed evenly within each group
    slot_patterns = [_rhythm_patterns_for_slot(s) for s in deck]
    groups: dict[tuple, list[int]] = {}
    for i, avail in enumerate(slot_patterns):
        groups.setdefault(tuple(avail), []).append(i)

    assigned_rhythms: list[tuple] = [()] * len(deck)
    assigned_pannings: list[tuple] = [()] * len(deck)
    for avail_tuple, indices in groups.items():
        if not avail_tuple:
            continue
        patterns = list(avail_tuple)
        n = len(indices)
        weights = [pattern_weights.get(pat[0], 1) if pat != (UNTOUCHED,) else 1 for pat in patterns]
        total_w = sum(weights)
        counts = [round(n * w / total_w) for w in weights]
        diff = n - sum(counts)
        for i in range(abs(diff)):
            counts[i % len(counts)] += 1 if diff > 0 else -1
        flat: list[tuple] = [pat for pat, cnt in zip(patterns, counts) for _ in range(cnt)]
        random.shuffle(flat)
        for idx, raw_pattern in zip(indices, flat):
            rhythm, beat_pannings = _extract_rhythm_and_pannings(raw_pattern)
            assigned_rhythms[idx] = rhythm
            assigned_pannings[idx] = beat_pannings

    deck = [dataclasses.replace(slot, rhythm=assigned_rhythms[i], beat_pannings=assigned_pannings[i]) for i, slot in enumerate(deck)]

    # Decrement quota for secondary beats (beat 2+). The primary beat is already
    # counted by _allocate_panning_slots; secondary beats consume additional quota.
    secondary_usage: dict[float | str, int] = {}
    for slot in deck:
        for bp in slot.beat_pannings[1:]:
            if bp and bp is not UNTOUCHED:
                secondary_usage[bp] = secondary_usage.get(bp, 0) + 1
    if secondary_usage:
        for pan, count in secondary_usage.items():
            available_slots[pan] = available_slots.get(pan, 0) - count
        over = {pan: -cnt for pan, cnt in available_slots.items() if cnt < 0}
        if over:
            print(f"  ⚠  Secondary beat quota exceeded: {over} extra use(s)")

    missing = [s for s in deck if not s.rhythm]
    if missing:
        details = ', '.join(
            f"{s.sound_group}/{s.panning}/{s.volume_label}" for s in missing[:5]
        )
        raise ValueError(
            f"{len(missing)} slot(s) have no rhythm_patterns defined in sound_rules "
            f"(first {min(5, len(missing))}: {details})"
        )

    return deck
