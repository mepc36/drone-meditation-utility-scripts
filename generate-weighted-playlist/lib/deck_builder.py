import random
import dataclasses
from dataclasses import dataclass

from .constants import (
    HARD_CENTER, HARD_LEFT, HARD_RIGHT, DIAGONAL_LEFT, DIAGONAL_RIGHT,
    DUALPAN_LEFTRIGHT, DUALPAN_DIAGONAL, UNTOUCHED,
    SOUND_GROUP_NAMES, SOUND_GROUP_TYPES,
    KICK, SNARE, KICKSNARE, KICKSTAB, SNARESTAB, STAB, ACAPPELLA,
    PANNING_CENTER, PANNING_DIAGONAL, PANNING_LEFT, PANNING_RIGHT, PANNING_DUALPAN,
    VOLUMES, BPMS, RHYTHM_PATTERNS, MUSICAL_PATTERNS, POSSIBLE_PANNINGS, MUSICAL_DURATION,
    RHYTHM_PATTERN, RHYTHM_PERCENT, SAMPLE_ROLE, SampleRole, MUSIC_PATTERN_PERCENT,
    RandomPan,
    RANDOM_PAN_MIN_DIFF,
)
from .runtime_constants import LOUD, QUIET, SLOW, FAST
from .sound_rules import derive_panning_key, derive_type, panning_compat, panning_percents, rules_by_sound_type, KICK_SNARE_MUSICAL_PATTERNS, KICKSTAB_SNARESTAB_MUSICAL_PATTERNS, ACAPPELLA_MUSICAL_PATTERNS


@dataclass(frozen=True)
class SlotSpec:
    sound_group: str
    panning: float | str       # float position or 'dualpan' / 'untouched'
    volume_label: float | str  # actual dB float, or 'untouched' for strings slots
    bpm_label: float | str     # actual BPM float, or 'untouched' for strings slots
    rhythm: tuple[float | str, ...] = ()
    beat_pannings: tuple[float | str, ...] = ()  # per-beat chosen pannings, parallel to rhythm
    beat_roles: tuple[SampleRole | None, ...] = ()  # per-beat role; SAME = reuse primary, NEW = draw fresh


def _expand_directional_quotas(panning_quotas: dict[str, int]) -> dict:
    diagonal = panning_quotas.get(PANNING_DIAGONAL, 0)
    return {
        HARD_CENTER:    panning_quotas.get(PANNING_CENTER, 0),
        DIAGONAL_LEFT:  diagonal // 2,
        DIAGONAL_RIGHT: diagonal - diagonal // 2,
        DUALPAN_LEFTRIGHT: panning_quotas.get(PANNING_DUALPAN, 0),
        DUALPAN_DIAGONAL:  panning_quotas.get(PANNING_DUALPAN, 0),
        HARD_LEFT:      panning_quotas.get(PANNING_LEFT, 0),
        HARD_RIGHT:     panning_quotas.get(PANNING_RIGHT, 0),
    }


def _directional_pannings_for_group(group: str) -> set:
    # panning_compat already contains concrete numeric panning values
    return panning_compat.get(group, set())


def _allocate_panning_slots(
    group_targets: dict[str, int],
    available_slots: dict[str, int],
) -> tuple[dict[str, dict[str, int]], int]:
    allocation: dict[str, dict[str, int]] = {g: {} for g in group_targets}
    actual_targets = dict(group_targets)
    overflow = 0
    capped: set[str] = set()

    ordered_by_most_constrained = sorted(
        group_targets,
        key=lambda g: (sum(available_slots.get(p, 0) for p in _directional_pannings_for_group(g)), group_targets[g]),
    )

    def fill_group(group: str, slots_needed: int) -> None:
        group_pct = panning_percents.get(group, {})
        all_pans = _directional_pannings_for_group(group)
        # Exclude pannings with explicit 0% weight so they never receive slots.
        eligible_pans = {p for p in all_pans if group_pct.get(p, 1) > 0}
        compatible_pans = [p for p in eligible_pans if available_slots.get(p, 0) > 0]
        remaining = slots_needed
        if group_pct and compatible_pans:
            total_pct = sum(group_pct.get(p, 0) for p in compatible_pans)
            for pan in sorted(compatible_pans, key=lambda p: -group_pct.get(p, 0)):
                if remaining == 0 or total_pct == 0:
                    break
                share = min(round(group_pct.get(pan, 0) / total_pct * slots_needed), available_slots[pan], remaining)
                allocation[group][pan] = allocation[group].get(pan, 0) + share
                available_slots[pan] -= share
                remaining -= share
        else:
            total_available = sum(available_slots.get(p, 0) for p in compatible_pans)
            for pan in sorted(compatible_pans, key=lambda p: -available_slots.get(p, 0)):
                if remaining == 0 or total_available == 0:
                    break
                share = min(round(available_slots[pan] / total_available * slots_needed), available_slots[pan], remaining)
                allocation[group][pan] = allocation[group].get(pan, 0) + share
                available_slots[pan] -= share
                remaining -= share
        while remaining > 0:
            candidates = [(p, available_slots[p]) for p in eligible_pans if available_slots.get(p, 0) > 0]
            if not candidates:
                break
            best_pan = max(candidates, key=lambda x: x[1])[0]
            allocation[group][best_pan] = allocation[group].get(best_pan, 0) + 1
            available_slots[best_pan] -= 1
            remaining -= 1

    for group in ordered_by_most_constrained:
        needed = actual_targets[group]
        group_pct = panning_percents.get(group, {})
        eligible = {p for p in _directional_pannings_for_group(group) if group_pct.get(p, 1) > 0}
        can_fill = sum(available_slots.get(p, 0) for p in eligible)
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
    """Convert a beat dict to a hashable (duration, (panning, ...), role) tuple."""
    if not isinstance(b, dict):
        raise TypeError(f"Beat must be a dict with 'musical_duration' and 'possible_pannings', got {b!r}")
    return (float(b[MUSICAL_DURATION]), tuple(b[POSSIBLE_PANNINGS]), b.get(SAMPLE_ROLE))


def _resolve_rp(raw) -> list:
    """Resolve a RHYTHM_PATTERN value: call it if it's a factory callable, else return as-is."""
    return raw() if callable(raw) else raw


def _sample_random_pan_with_distance(rp: RandomPan, prev_pan: float, min_diff: float) -> float:
    """Sample uniformly from the RandomPan range, excluding values within min_diff of prev_pan.

    Builds valid sub-intervals from the intersection of:
        [-rp.max_magnitude, -rp.min_magnitude] ∪ [rp.min_magnitude, rp.max_magnitude]
    with:
        (-∞, prev_pan - min_diff] ∪ [prev_pan + min_diff, +∞)
    then samples uniformly by weight across the valid intervals.

    Falls back to the unconstrained RandomPan range if no valid interval exists
    (should not occur with standard config values).
    """
    lobes = [
        (-rp.max_magnitude, -rp.min_magnitude),
        (rp.min_magnitude, rp.max_magnitude),
    ]
    below = prev_pan - min_diff
    above = prev_pan + min_diff
    valid: list[tuple[float, float]] = []
    for lo, hi in lobes:
        s, e = lo, min(hi, below)
        if s <= e:
            valid.append((s, e))
        s, e = max(lo, above), hi
        if s <= e:
            valid.append((s, e))
    if not valid:
        side = random.choice([-1.0, 1.0])
        return side * random.uniform(rp.min_magnitude, rp.max_magnitude)
    lengths = [e - s for s, e in valid]
    total = sum(lengths)
    r = random.uniform(0.0, total)
    cumulative = 0.0
    for (s, e), length in zip(valid, lengths):
        cumulative += length
        if r <= cumulative:
            return random.uniform(s, e)
    return random.uniform(valid[-1][0], valid[-1][1])


def _extract_rhythm_and_pannings(raw_pattern: tuple) -> tuple[tuple, tuple, tuple]:
    """Split a raw (hashable) beat pattern into duration, panning, and role tuples.

    Format: (type_str, rhythm_percent, (duration, (pannings,...), role), ...).
    A panning is randomly chosen from the candidate list for each beat.

    Grouped Panning Rule (applied to RandomPan beats):
      N = total beats in pattern
      N <= 1 : single beat, no constraint
      N == 2 : both beats independently randomized (no shared anchor)
      N == 3 : first 2 beats share one random position; 3rd is independently
               randomized but must be at least RANDOM_PAN_MIN_DIFF from beat 2
      N >= 4 : first N-2 beats share one position; remaining beats are each
               independently constrained by RANDOM_PAN_MIN_DIFF from the preceding beat
    """
    if raw_pattern == (UNTOUCHED,):
        return (UNTOUCHED,), (), ()
    durations: list = []
    pannings: list = []
    roles: list = []

    beats = raw_pattern[2:]  # skip type string at index 0 and percent at index 1
    n = len(beats)

    # Grouped Panning Rule for RandomPan beats:
    #   N <= 1: no constraint needed
    #   N == 2: fully independent — each beat gets its own randomized position (anchor_size = 0)
    #   N == 3: first 2 beats share one position; 3rd is independently constrained by RANDOM_PAN_MIN_DIFF
    #   N >= 4: first N-2 beats share one position; remaining beats are each independent (anchor_size = N-2)
    if n <= 1:
        anchor_size = n
    elif n == 2:
        anchor_size = 0
    elif n == 3:
        anchor_size = 2
    else:
        anchor_size = n - 2

    _shared_random_pan: float | None = None
    for _beat_index, b in enumerate(beats):
        if not (isinstance(b, tuple) and len(b) in (2, 3) and isinstance(b[1], tuple)):
            raise TypeError(f"Hashed beat must be a (duration, (pannings,...)[, role]) tuple, got {b!r}")
        duration, pan_options = b[0], b[1]
        role = b[2] if len(b) == 3 else None
        durations.append(float(duration))
        chosen = random.choice(pan_options) if pan_options else ''
        if isinstance(chosen, RandomPan):
            if _beat_index < anchor_size:
                # Anchor group: all beats in this range resolve to the same position.
                if _shared_random_pan is None:
                    side = random.choice([-1.0, 1.0])
                    magnitude = random.uniform(chosen.min_magnitude, chosen.max_magnitude)
                    _shared_random_pan = side * magnitude
                chosen = _shared_random_pan
            elif pannings:
                # Tail beat: independently randomized, constrained to be at least
                # RANDOM_PAN_MIN_DIFF from the immediately preceding panning.
                chosen = _sample_random_pan_with_distance(chosen, pannings[-1], RANDOM_PAN_MIN_DIFF)
            else:
                # First tail beat with no preceding panning (N=2): free random draw.
                side = random.choice([-1.0, 1.0])
                magnitude = random.uniform(chosen.min_magnitude, chosen.max_magnitude)
                chosen = side * magnitude
        pannings.append(chosen)
        roles.append(role)
    return tuple(durations), tuple(pannings), tuple(roles)


def _allowed_volume_bpm_combos(group: str, panning: float | None) -> set[tuple]:
    result: set[tuple[str, str]] = set()
    for sound_type in SOUND_GROUP_TYPES[group]:
        rule = rules_by_sound_type.get(sound_type)
        if rule is None:
            continue
        for pan_rule in rule[MUSICAL_PATTERNS]:
            if derive_panning_key(pan_rule) != panning:
                continue
            for vol_label in pan_rule[VOLUMES]:
                for bpm_label in pan_rule[BPMS]:
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
                raise ValueError(
                    f"No allowed volume/bpm combinations found for group '{group}', panning {panning!r}. "
                    f"Check that sound_rules has a MUSICAL_PATTERNS entry matching this panning."
                )
            elif len(opts) == 1:
                vol, bpm = next(iter(opts))
                forced += [SlotSpec(group, panning, vol, bpm)] * count
            else:
                free += [(group, panning, list(opts))] * count

    remaining_slow  = max(0, bpm_targets.get(slowest_bpm_index, 0) - sum(1 for s in forced if s.bpm_label == SLOW))
    remaining_fast  = max(0, bpm_targets.get(fastest_bpm_index, 0) - sum(1 for s in forced if s.bpm_label == FAST))
    remaining_loud  = max(0, vol_targets.get(loudest_volume_index, 0) - sum(1 for s in forced if s.volume_label == LOUD))
    remaining_quiet = max(0, vol_targets.get(quietest_volume_index, 0) - sum(1 for s in forced if s.volume_label == QUIET))

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
        if bpm == SLOW:
            remaining_slow -= 1
        elif bpm == FAST:
            remaining_fast -= 1
        if vol == LOUD:
            remaining_loud -= 1
        elif vol == QUIET:
            remaining_quiet -= 1

    return forced + assigned


def _rhythm_patterns_for_slot(slot: SlotSpec) -> list[tuple]:
    """Return unique hashable rhythm patterns for this slot from sound_rules.
    Each tuple is (type_str, rhythm_percent, (duration, (pannings,...)), ...).
    """
    seen: set[tuple] = set()
    found: list[tuple] = []
    for sound_type in SOUND_GROUP_TYPES.get(slot.sound_group, set()):
        rule = rules_by_sound_type.get(sound_type)
        if rule is None:
            continue
        for pan_rule in rule[MUSICAL_PATTERNS]:
            if derive_panning_key(pan_rule) != slot.panning:
                continue
            if slot.volume_label not in pan_rule[VOLUMES]:
                continue
            if slot.bpm_label not in pan_rule[BPMS]:
                continue
            for entry in pan_rule.get(RHYTHM_PATTERNS, []):
                if entry is UNTOUCHED:
                    pat = (UNTOUCHED,)
                elif isinstance(entry, dict):
                    beats = _resolve_rp(entry[RHYTHM_PATTERN])
                    pat = (derive_type(beats), entry[RHYTHM_PERCENT]) + tuple(_hashable_beat(b) for b in beats)
                else:
                    raise TypeError(f"Unexpected rhythm_pattern entry: {entry!r}")
                if pat not in seen:
                    seen.add(pat)
                    found.append(pat)
    return found


def compute_group_beat_multipliers() -> dict[str, float]:
    """Return the average beats per allocated slot for each sound group.

    Beat count for a slot = number of note-events in the rhythm pattern,
    weighted by MUSIC_PATTERN_PERCENT and RHYTHM_PERCENT.

    Used to pre-correct group_targets so the configured ratio applies to
    total beats heard, not just output file counts or input draws.
    Multi-beat patterns (e.g. quarter-quarter-quarter = 3 beats) contribute
    proportionally more to the group's beat weight.
    """
    multipliers: dict[str, float] = {}
    for group in SOUND_GROUP_NAMES:
        sound_type = next(iter(SOUND_GROUP_TYPES[group]))
        rule = rules_by_sound_type.get(sound_type)
        if rule is None:
            multipliers[group] = 1.0
            continue
        total_beats = 0.0
        for mp in rule[MUSICAL_PATTERNS]:
            mp_weight = mp[MUSIC_PATTERN_PERCENT] / 100.0
            rp_list = mp[RHYTHM_PATTERNS]
            if not rp_list or rp_list[0] is UNTOUCHED:
                continue
            for rp_entry in rp_list:
                rp_weight = rp_entry[RHYTHM_PERCENT] / 100.0
                beats = _resolve_rp(rp_entry[RHYTHM_PATTERN])
                total_beats += mp_weight * rp_weight * len(beats)
        multipliers[group] = total_beats if total_beats > 0 else 1.0
    return multipliers


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
    if overflow > 0:
        print(f"  ⚠  Panning overflow: {overflow} slot(s) redistributed among uncapped groups")
    for group in SOUND_GROUP_NAMES:
        if group in allocation:
            left  = allocation[group].get(HARD_LEFT, 0) + allocation[group].get(DIAGONAL_LEFT, 0)
            right = allocation[group].get(HARD_RIGHT, 0) + allocation[group].get(DIAGONAL_RIGHT, 0)
            total_g = sum(allocation[group].values())

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
    assigned_roles: list[tuple] = [()] * len(deck)
    for avail_tuple, indices in groups.items():
        if not avail_tuple:
            continue
        patterns = list(avail_tuple)
        n = len(indices)
        weights = [pat[1] if pat != (UNTOUCHED,) else 1 for pat in patterns]
        total_w = sum(weights)
        counts = [round(n * w / total_w) for w in weights]
        diff = n - sum(counts)
        for i in range(abs(diff)):
            counts[i % len(counts)] += 1 if diff > 0 else -1
        flat: list[tuple] = [pat for pat, cnt in zip(patterns, counts) for _ in range(cnt)]
        random.shuffle(flat)
        for idx, raw_pattern in zip(indices, flat):
            rhythm, beat_pannings, beat_roles = _extract_rhythm_and_pannings(raw_pattern)
            assigned_rhythms[idx] = rhythm
            assigned_pannings[idx] = beat_pannings
            assigned_roles[idx] = beat_roles

    deck = [
        dataclasses.replace(slot, rhythm=assigned_rhythms[i], beat_pannings=assigned_pannings[i], beat_roles=assigned_roles[i])
        for i, slot in enumerate(deck)
    ]

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

