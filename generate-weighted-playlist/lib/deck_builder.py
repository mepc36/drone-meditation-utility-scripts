import random
import dataclasses
from dataclasses import dataclass

from .sound_rules import SOUND_GROUP_NAMES, SOUND_GROUP_TYPES, panning_compat, rules_by_sound_type


@dataclass(frozen=True)
class SlotSpec:
    sound_group: str
    panning: str
    volume_label: str
    bpm_label: str
    rhythm: tuple[float | str, ...] = ()


def _expand_directional_quotas(panning_quotas: dict[str, int]) -> dict[str, int]:
    diagonal = panning_quotas.get('diagonal', 0)
    leftorright = panning_quotas.get('leftorright', 0)
    return {
        'center':         panning_quotas.get('center', 0),
        'diagonal_left':  diagonal // 2,
        'diagonal_right': diagonal - diagonal // 2,
        'dualpan':        panning_quotas.get('dualpan', 0),
        'hard_left':      leftorright // 2,
        'hard_right':     leftorright - leftorright // 2,
    }


def _directional_pannings_for_group(group: str) -> set[str]:
    result: set[str] = set()
    for pan in panning_compat.get(group, set()):
        if pan == 'diagonal':
            result.update({'diagonal_left', 'diagonal_right'})
        elif pan == 'leftorright':
            result.update({'hard_left', 'hard_right'})
        else:
            result.add(pan)
    return result


def _allocate_panning_slots(group_targets: dict[str, int], available_slots: dict[str, int]) -> dict[str, dict[str, int]]:
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


def _panning_key_for_rule_lookup(panning: str) -> str:
    if panning.startswith('diagonal'):
        return 'diagonal'
    if panning in ('hard_left', 'hard_right'):
        return 'leftorright'
    return panning


def _allowed_volume_bpm_combos(group: str, panning: str) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for sound_type in SOUND_GROUP_TYPES[group]:
        rule = rules_by_sound_type.get(sound_type)
        if rule is None:
            continue
        pan_rule = rule['pannings'].get(_panning_key_for_rule_lookup(panning))
        if pan_rule is None:
            continue
        for vol_label, bpm_info in pan_rule['volumes'].items():
            for bpm_label in bpm_info.get('bpms', []):
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
                forced += [SlotSpec(group, panning, 'loud', 'slow')] * count
            elif len(opts) == 1:
                vol, bpm = next(iter(opts))
                forced += [SlotSpec(group, panning, vol, bpm)] * count
            else:
                free += [(group, panning, list(opts))] * count

    remaining_slow  = max(0, bpm_targets.get(slowest_bpm_index, 0)   - sum(1 for s in forced if s.bpm_label == 'slow'))
    remaining_fast  = max(0, bpm_targets.get(fastest_bpm_index, 0)    - sum(1 for s in forced if s.bpm_label == 'fast'))
    remaining_loud  = max(0, vol_targets.get(loudest_volume_index, 0) - sum(1 for s in forced if s.volume_label == 'loud'))
    remaining_quiet = max(0, vol_targets.get(quietest_volume_index, 0)- sum(1 for s in forced if s.volume_label == 'quiet'))

    random.shuffle(free)
    assigned: list[SlotSpec] = []
    for group, panning, opts in free:
        best_combo = max(
            opts,
            key=lambda vb: (
                int(vb[1] == 'slow' and remaining_slow > 0) +
                int(vb[1] == 'fast' and remaining_fast > 0) +
                int(vb[0] == 'loud' and remaining_loud > 0) +
                int(vb[0] == 'quiet' and remaining_quiet > 0)
            ),
        )
        vol, bpm = best_combo
        assigned.append(SlotSpec(group, panning, vol, bpm))
        if bpm == 'slow':    remaining_slow  -= 1
        elif bpm == 'fast':  remaining_fast  -= 1
        if vol == 'loud':    remaining_loud  -= 1
        elif vol == 'quiet': remaining_quiet -= 1

    return forced + assigned


def _rhythm_patterns_for_slot(slot: SlotSpec) -> list[tuple[float | str, ...]]:
    """Return rhythm patterns for this slot by looking up sound_rules.
    Duplicates are preserved — repeat a pattern to give it more weight."""
    found: list[tuple[float | str, ...]] = []
    pan_key = _panning_key_for_rule_lookup(slot.panning)
    for sound_type in SOUND_GROUP_TYPES.get(slot.sound_group, set()):
        rule = rules_by_sound_type.get(sound_type)
        if rule is None:
            continue
        pan_rule = rule['pannings'].get(pan_key)
        if pan_rule is None:
            continue
        vol_rule = pan_rule['volumes'].get(slot.volume_label)
        if vol_rule is None:
            continue
        for p in vol_rule.get('rhythm_patterns', []):
            if p == 'untouched':
                found.append(('untouched',))
            elif isinstance(p, (int, float)):
                found.append((float(p),))
            else:
                found.append(tuple(p))
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

    assigned: list[tuple[float, ...]] = [()] * len(deck)
    for avail_tuple, indices in groups.items():
        if not avail_tuple:
            continue
        patterns = list(avail_tuple)
        n, p = len(indices), len(patterns)
        counts = [n // p + (1 if i < n % p else 0) for i in range(p)]
        flat: list[tuple[float, ...]] = [pat for pat, cnt in zip(patterns, counts) for _ in range(cnt)]
        random.shuffle(flat)
        for idx, rhythm in zip(indices, flat):
            assigned[idx] = rhythm

    deck = [dataclasses.replace(slot, rhythm=assigned[i]) for i, slot in enumerate(deck)]

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
