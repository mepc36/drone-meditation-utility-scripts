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
    forced_sample: str | None = None  # permutation mode: pre-assigned sample name (bypasses queue draw)
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


def _extract_rhythm_and_pannings(raw_pattern: tuple) -> tuple[tuple, tuple, tuple]:
    """Split a raw (hashable) beat pattern into duration, panning, and role tuples.

    Format: (type_str, rhythm_percent, (duration, (pannings,...), role), ...).
    A panning is randomly chosen from the candidate list for each beat.
    """
    if raw_pattern == (UNTOUCHED,):
        return (UNTOUCHED,), (), ()
    durations: list = []
    pannings: list = []
    roles: list = []
    # For RandomPan beats: beats 1 and 2 share the same resolved position so
    # they feel like a pair, but beat 3 onwards each get an independent random
    # position to add movement through the rhythm.
    _shared_random_pan: float | None = None
    _beat_index = 0
    for b in raw_pattern[2:]:  # skip type string at index 0 and percent at index 1
        if not (isinstance(b, tuple) and len(b) in (2, 3) and isinstance(b[1], tuple)):
            raise TypeError(f"Hashed beat must be a (duration, (pannings,...)[, role]) tuple, got {b!r}")
        duration, pan_options = b[0], b[1]
        role = b[2] if len(b) == 3 else None
        durations.append(float(duration))
        chosen = random.choice(pan_options) if pan_options else ''
        if isinstance(chosen, RandomPan):
            if _beat_index < 2:
                # Beats 1 & 2: resolve once and share the same position.
                if _shared_random_pan is None:
                    side = random.choice([-1.0, 1.0])
                    magnitude = random.uniform(chosen.min_magnitude, chosen.max_magnitude)
                    _shared_random_pan = side * magnitude
                chosen = _shared_random_pan
            else:
                # Beat 3+: each gets its own independent random position.
                side = random.choice([-1.0, 1.0])
                magnitude = random.uniform(chosen.min_magnitude, chosen.max_magnitude)
                chosen = side * magnitude
        pannings.append(chosen)
        roles.append(role)
        _beat_index += 1
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


def build_permutation_kick_snare_deck(
    samples_by_type: dict[str, list[str]],
) -> list[SlotSpec]:
    """Permutation mode: one SlotSpec per (kick/snare sample × rhythm) pair.

    The rhythm and forced_sample are pre-assigned so every combination appears
    exactly once. The returned list is shuffled before being returned.
    """
    kick_snare_samples = (
        list(samples_by_type.get(KICK, [])) +
        list(samples_by_type.get(SNARE, []))
    )

    # Collect entry templates; resolve RHYTHM_PATTERN per sample so that callable
    # (variable) patterns produce a fresh random rhythm for every sample.
    entry_templates: list[tuple] = []  # (entry, vol_label, bpm_label)
    for pattern_group in KICK_SNARE_MUSICAL_PATTERNS:
        vol_label = pattern_group[VOLUMES][0]
        bpm_label = pattern_group[BPMS][0]
        for entry in pattern_group[RHYTHM_PATTERNS]:
            entry_templates.append((entry, vol_label, bpm_label))

    slots: list[SlotSpec] = [
        SlotSpec(
            sound_group=KICKSNARE,
            panning=HARD_CENTER,
            volume_label=vol_label,
            bpm_label=bpm_label,
            rhythm=rhythm,
            beat_pannings=beat_pannings,
            beat_roles=beat_roles,
            forced_sample=sample,
        )
        for sample in kick_snare_samples
        for entry, vol_label, bpm_label in entry_templates
        for raw in [_resolve_rp(entry[RHYTHM_PATTERN])]
        for pat in [(derive_type(raw), entry[RHYTHM_PERCENT]) + tuple(_hashable_beat(b) for b in raw)]
        for rhythm, beat_pannings, beat_roles in [_extract_rhythm_and_pannings(pat)]
    ]
    random.shuffle(slots)
    return slots


def build_permutation_stab_deck(
    samples_by_type: dict[str, list[str]],
) -> list[SlotSpec]:
    """Permutation mode: one SlotSpec per (stab sample × musical_pattern × rhythm) combo.

    The panning, rhythm, and forced_sample are pre-assigned so every combination
    appears exactly once.  For DUALPAN slots the partner sample is drawn at
    render time by resolve_slot (same as the probabilistic path).
    """
    stab_samples = (
        list(samples_by_type.get(KICKSTAB, [])) +
        list(samples_by_type.get(SNARESTAB, []))
    )
    if not stab_samples:
        return []

    slot_templates: list[tuple] = []  # (panning, pat, vol_label, bpm_label)
    for pattern_group in KICKSTAB_SNARESTAB_MUSICAL_PATTERNS:
        vol_label = pattern_group[VOLUMES][0]
        bpm_label = pattern_group[BPMS][0]
        panning = derive_panning_key(pattern_group)
        for entry in pattern_group[RHYTHM_PATTERNS]:
            raw = _resolve_rp(entry[RHYTHM_PATTERN])
            pat = (derive_type(raw), entry[RHYTHM_PERCENT]) + tuple(_hashable_beat(b) for b in raw)
            slot_templates.append((panning, pat, vol_label, bpm_label))

    slots: list[SlotSpec] = [
        SlotSpec(
            sound_group=STAB,
            panning=panning,
            volume_label=vol_label,
            bpm_label=bpm_label,
            rhythm=rhythm,
            beat_pannings=beat_pannings,
            beat_roles=beat_roles,
            forced_sample=sample,
        )
        for sample in stab_samples
        for panning, pat, vol_label, bpm_label in slot_templates
        for rhythm, beat_pannings, beat_roles in [_extract_rhythm_and_pannings(pat)]
    ]
    random.shuffle(slots)
    return slots


def build_permutation_acappella_deck(
    samples_by_type: dict[str, list[str]],
) -> list[SlotSpec]:
    """Permutation mode: one SlotSpec per (acappella sample × musical_pattern × rhythm) combo."""
    acap_samples = list(samples_by_type.get(ACAPPELLA, []))
    if not acap_samples:
        return []

    slot_templates: list[tuple] = []  # (panning, pat, vol_label, bpm_label)
    for pattern_group in ACAPPELLA_MUSICAL_PATTERNS:
        vol_label = pattern_group[VOLUMES][0]
        bpm_label = pattern_group[BPMS][0]
        panning = derive_panning_key(pattern_group)
        for entry in pattern_group[RHYTHM_PATTERNS]:
            raw = _resolve_rp(entry[RHYTHM_PATTERN])
            pat = (derive_type(raw), entry[RHYTHM_PERCENT]) + tuple(_hashable_beat(b) for b in raw)
            slot_templates.append((panning, pat, vol_label, bpm_label))

    slots: list[SlotSpec] = [
        SlotSpec(
            sound_group=ACAPPELLA,
            panning=panning,
            volume_label=vol_label,
            bpm_label=bpm_label,
            rhythm=rhythm,
            beat_pannings=beat_pannings,
            beat_roles=beat_roles,
            forced_sample=sample,
        )
        for sample in acap_samples
        for panning, pat, vol_label, bpm_label in slot_templates
        for rhythm, beat_pannings, beat_roles in [_extract_rhythm_and_pannings(pat)]
    ]
    random.shuffle(slots)
    return slots


def _exact_beats_per_file(pattern_list: list) -> int:
    """Return the total number of beats across all rhythm entries in all musical pattern groups.

    In permutation mode every sample is paired with every rhythm entry exactly once,
    so the total beats contributed per input file = sum of len(rhythm) for every
    rhythm entry across every musical pattern group.
    """
    total = 0
    for pattern_group in pattern_list:
        for entry in pattern_group[RHYTHM_PATTERNS]:
            total += len(_resolve_rp(entry[RHYTHM_PATTERN]))
    return total


def _exact_slots_per_file(pattern_list: list) -> int:
    """Return the number of output files produced per input sample in permutation mode.

    Each (sample × rhythm_entry) pair becomes one output file, so this equals
    the total number of rhythm entries across all musical pattern groups.
    """
    total = 0
    for pattern_group in pattern_list:
        total += len(pattern_group[RHYTHM_PATTERNS])
    return total


def plan_permutation_trimming(
    samples_by_type: dict[str, list[str]],
    group_percents: list[int],
    max_files: int,
    tolerance: float,
    ks_ignored_cap: int = 0,
    seed: int = 42,
) -> tuple[dict[str, list[str]], int, int, dict]:
    """Determine how many input samples to use per group and how many times to
    replicate stab/acappella decks in order to match the target beat ratio.

    Trimming priority: stab first (typically most files), then kicksnare,
    then acappella. Replication is used when a group is under-represented;
    trimming when over-represented.

    Raises ValueError if the target cannot be reached within tolerance without
    exceeding max_files. Adjust permutation_tolerance_pct or
    permutation_max_files in config.json to loosen the constraints.

    Returns
    -------
    trimmed_samples_by_type : dict
        Input sample lists with low-priority samples removed (seeded shuffle).
    m_stab : int
        Multiplier to apply to the full stab deck.
    m_acap : int
        Multiplier to apply to the full acappella deck.
    diagnostics : dict
        Keys: ignored_by_type, k_ks, k_stab, k_acap, m_stab, m_acap,
              beat_pcts, total_files, within_tolerance.
    """
    from math import ceil
    from .constants import KICK, SNARE, KICKSTAB, SNARESTAB

    p_ks, p_stab, p_acap = [group_percents[SOUND_GROUP_NAMES.index(g)] for g in (KICKSNARE, STAB, ACAPPELLA)]

    bps_ks   = _exact_beats_per_file(KICK_SNARE_MUSICAL_PATTERNS)
    bps_stab = _exact_beats_per_file(KICKSTAB_SNARESTAB_MUSICAL_PATTERNS)
    bps_acap = _exact_beats_per_file(ACAPPELLA_MUSICAL_PATTERNS)

    # Files per input sample in permutation mode (= rhythm-entry count, not beat count)
    spf_ks   = _exact_slots_per_file(KICK_SNARE_MUSICAL_PATTERNS)
    spf_stab = _exact_slots_per_file(KICKSTAB_SNARESTAB_MUSICAL_PATTERNS)
    spf_acap = _exact_slots_per_file(ACAPPELLA_MUSICAL_PATTERNS)

    # Stable shuffle so excluded samples are deterministic
    rng = random.Random(seed)

    def shuffle_copy(lst: list[str]) -> list[str]:
        c = list(lst)
        rng.shuffle(c)
        return c

    kicks_all   = shuffle_copy(samples_by_type.get(KICK, []))
    snares_all  = shuffle_copy(samples_by_type.get(SNARE, []))
    kstabs_all  = shuffle_copy(samples_by_type.get(KICKSTAB, []))
    sstabs_all  = shuffle_copy(samples_by_type.get(SNARESTAB, []))
    acaps_all   = shuffle_copy(samples_by_type.get(ACAPPELLA, []))

    available_ks   = len(kicks_all) + len(snares_all)
    available_stab = len(kstabs_all) + len(sstabs_all)
    available_acap = len(acaps_all)

    if available_ks == 0:
        raise ValueError(
            "No kicksnare samples found in input/audio. "
            "At least one kick or snare sample is required in permutation mode."
        )

    def resolve_group(ideal_slots: float, available: int) -> tuple[int, int]:
        """Return (k_samples, m_multiplier) for a group.

        If the ideal total beat-slot count exceeds available samples, replicate
        (m > 1, use all samples). If it is below available, trim to ideal.
        m_multiplier is always ≥ 1.
        """
        if available == 0:
            return 0, 0
        if ideal_slots >= available:
            return available, max(1, round(ideal_slots / available))
        else:
            return max(1, round(ideal_slots)), 1

    def compute_beat_pcts(k_ks: int, k_stab: int, m_stab: int, k_acap: int, m_acap: int) -> tuple[float, float, float]:
        b_ks   = k_ks   * bps_ks
        b_stab = k_stab * m_stab * bps_stab
        b_acap = k_acap * m_acap * bps_acap
        total  = b_ks + b_stab + b_acap
        if total == 0:
            return 0.0, 0.0, 0.0
        return b_ks / total * 100, b_stab / total * 100, b_acap / total * 100

    def total_files(k_ks: int, k_stab: int, m_stab: int, k_acap: int, m_acap: int) -> int:
        return k_ks * spf_ks + k_stab * m_stab * spf_stab + k_acap * m_acap * spf_acap

    def within_tol(pcts: tuple[float, float, float], targets: tuple[int, int, int]) -> bool:
        return all(abs(p - t) <= tolerance for p, t in zip(pcts, targets))

    best: tuple | None = None  # (k_ks, k_stab, m_stab, k_acap, m_acap, pcts)

    # Sweep keep-count from all kicksnare files down to the cap floor.
    # ks_ignored_cap=0 means never remove any kicksnare samples (sweep stays at available_ks).
    # For each candidate, stab and acappella counts are derived analytically —
    # trimming when over-represented, replicating when under-represented.
    # This keeps the search 1-dimensional while honouring the priority:
    # stab is trimmed/replicated freely for every kicksnare count, kicksnare
    # only shrinks when stab/acappella adjustment alone is insufficient,
    # acappella is adjusted last.
    min_ks = max(1, available_ks - ks_ignored_cap)
    for k_ks in range(available_ks, min_ks - 1, -1):
        if k_ks == 0:
            continue
        if p_stab > 0 and available_stab > 0:
            ideal_stab = k_ks * bps_ks * p_stab / (p_ks * bps_stab)
            k_stab, m_stab = resolve_group(ideal_stab, available_stab)
        else:
            k_stab, m_stab = available_stab, 0
        if p_acap > 0 and available_acap > 0:
            ideal_acap = k_ks * bps_ks * p_acap / (p_ks * bps_acap)
            k_acap, m_acap = resolve_group(ideal_acap, available_acap)
        else:
            k_acap, m_acap = available_acap, 0
        tf = total_files(k_ks, k_stab, m_stab, k_acap, m_acap)
        if tf > max_files:
            continue
        pcts = compute_beat_pcts(k_ks, k_stab, m_stab, k_acap, m_acap)
        if within_tol(pcts, (p_ks, p_stab, p_acap)):
            best = (k_ks, k_stab, m_stab, k_acap, m_acap, pcts)
            break

    if best is None:
        # Find the minimum achievable tolerance given max_files and ks_ignored_cap.
        min_deviation = float('inf')
        for k_ks in range(available_ks, min_ks - 1, -1):
            if k_ks == 0:
                continue
            if p_stab > 0 and available_stab > 0:
                ideal_stab = k_ks * bps_ks * p_stab / (p_ks * bps_stab)
                k_stab_c, m_stab_c = resolve_group(ideal_stab, available_stab)
            else:
                k_stab_c, m_stab_c = available_stab, 0
            if p_acap > 0 and available_acap > 0:
                ideal_acap = k_ks * bps_ks * p_acap / (p_ks * bps_acap)
                k_acap_c, m_acap_c = resolve_group(ideal_acap, available_acap)
            else:
                k_acap_c, m_acap_c = available_acap, 0
            if total_files(k_ks, k_stab_c, m_stab_c, k_acap_c, m_acap_c) > max_files:
                continue
            pcts_c = compute_beat_pcts(k_ks, k_stab_c, m_stab_c, k_acap_c, m_acap_c)
            deviation = max(abs(p - t) for p, t in zip(pcts_c, (p_ks, p_stab, p_acap)))
            if deviation < min_deviation:
                min_deviation = deviation
        if min_deviation == float('inf'):
            achievable_str = "none (all configurations exceed max_files)"
        else:
            achievable_str = f"±{min_deviation:.1f}%"
        raise ValueError(
            f"Cannot achieve target beat ratio {':'.join(map(str, group_percents))} "
            f"within ±{tolerance}% tolerance without exceeding {max_files} total output files "
            f"(kicksnare removal capped at {ks_ignored_cap}). "
            f"Best achievable tolerance with current permutation_max_files / kicksnare_ignored_cap / "
            f"kicksnare_stab_acappella_percents:\n\n{achievable_str}.\n\n"
            f"Increase permutation_tolerance_pct, permutation_max_files, or kicksnare_ignored_cap "
            f"in config.json, or adjust kicksnare_stab_acappella_percents."
        )

    k_ks, k_stab, m_stab, k_acap, m_acap, pcts = best

    # Build trimmed sample lists, splitting drops evenly between sub-types.
    # The first sub-type (kick / kickstab) takes the ceiling when total_drop is odd;
    # the second (snare / snarestab) takes the floor.
    # Remainders are redistributed if one sub-type runs out of samples to drop.
    def _even_split_keep(a_count: int, b_count: int, total_keep: int) -> tuple[int, int]:
        total_drop = (a_count + b_count) - total_keep
        if total_drop <= 0:
            return a_count, b_count
        a_drop = min((total_drop + 1) // 2, a_count)   # ceiling → a takes extra
        b_drop = min(total_drop // 2, b_count)          # floor
        deficit = total_drop - (a_drop + b_drop)
        if deficit > 0:
            b_drop = min(b_drop + deficit, b_count)
            deficit = total_drop - (a_drop + b_drop)
        if deficit > 0:
            a_drop = min(a_drop + deficit, a_count)
        return a_count - a_drop, b_count - b_drop

    kick_keep,  snare_keep  = _even_split_keep(len(kicks_all),  len(snares_all),  k_ks)
    kstab_keep, sstab_keep  = _even_split_keep(len(kstabs_all), len(sstabs_all),  k_stab)

    trimmed: dict[str, list[str]] = dict(samples_by_type)
    trimmed[KICK]      = kicks_all[:kick_keep]
    trimmed[SNARE]     = snares_all[:snare_keep]
    trimmed[KICKSTAB]  = kstabs_all[:kstab_keep]
    trimmed[SNARESTAB] = sstabs_all[:sstab_keep]
    trimmed[ACAPPELLA] = acaps_all[:k_acap]

    ignored_by_type: dict[str, list[str]] = {}
    if kick_keep  < len(kicks_all):   ignored_by_type[KICK]      = kicks_all[kick_keep:]
    if snare_keep < len(snares_all):  ignored_by_type[SNARE]     = snares_all[snare_keep:]
    if kstab_keep < len(kstabs_all):  ignored_by_type[KICKSTAB]  = kstabs_all[kstab_keep:]
    if sstab_keep < len(sstabs_all):  ignored_by_type[SNARESTAB] = sstabs_all[sstab_keep:]
    if k_acap     < len(acaps_all):   ignored_by_type[ACAPPELLA] = acaps_all[k_acap:]

    within_tolerance = within_tol(pcts, (p_ks, p_stab, p_acap))
    total_f = total_files(k_ks, k_stab, m_stab, k_acap, m_acap)

    diagnostics = {
        'ignored_by_type': ignored_by_type,
        'k_ks':   k_ks,
        'k_stab': k_stab,
        'k_acap': k_acap,
        'm_stab': m_stab,
        'm_acap': m_acap,
        'beat_pcts': pcts,
        'total_files': total_f,
        'within_tolerance': within_tolerance,
        'bps_ks': bps_ks,
        'bps_stab': bps_stab,
        'bps_acap': bps_acap,
        # Exact beats-per-output-file for each group in permutation mode
        # (= total beats per sample / number of rhythm entries per sample)
        'perm_beat_multipliers': {
            KICKSNARE: bps_ks / spf_ks,
            STAB:      bps_stab / spf_stab,
            ACAPPELLA: bps_acap / spf_acap,
        },
    }
    return trimmed, m_stab, m_acap, diagnostics


def balance_permutation_decks(
    decks_by_group: dict[str, list[SlotSpec]],
    group_percents: list[int],
    m_stab: int = 1,
    m_acap: int = 1,
) -> list[SlotSpec]:
    """Combine pre-built permutation decks using pre-computed multipliers.

    KICKSNARE is used as-is (multiplier = 1, already trimmed by plan_permutation_trimming).
    STAB is replicated m_stab times; ACAPPELLA is replicated m_acap times.
    """
    combined: list[SlotSpec] = []
    multipliers = {KICKSNARE: 1, STAB: m_stab, ACAPPELLA: m_acap}

    for group in SOUND_GROUP_NAMES:
        deck = decks_by_group.get(group, [])
        if not deck:
            continue
        m = multipliers.get(group, 1)
        combined.extend(deck * m)
        print(f"  Permutation deck: {group} — {len(deck)} raw slots × {m} = {len(deck) * m} total")

    random.shuffle(combined)
    return combined
