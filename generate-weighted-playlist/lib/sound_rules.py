from .constants import (
    HARD_CENTER, HARD_LEFT, HARD_RIGHT,
    DUALPAN_LEFTRIGHT, DUALPAN_DIAGONAL, UNTOUCHED,
    KICK, SNARE, KICKSTAB, SNARESTAB, ACAPPELLA, STRINGS,
    KICKSNARE, STAB, SampleRole,
    SOUND_GROUP_TYPES,
    PERMUTATION_COMBOS_PER_SAMPLE,
    QUARTER_NOTE, QUARTER_NOTE_REST, EIGHTH, SIXTEENTH, DOTTED_EIGHTH,
    RHYTHM_PATTERN_SEQUENCES,
    MUSICAL_DURATION, POSSIBLE_PANNINGS, SAMPLE_ROLE, RHYTHM_PATTERNS, VOLUMES, BPMS,
    MUSICAL_GROUPING, DUALPAN_PARTNERS, MUSICAL_PATTERNS,
    RHYTHM_PATTERN, RHYTHM_PERCENT,
    MUSIC_PATTERN_PERCENT,
)
from .runtime_constants import LOUD, QUIET, SLOW, FAST


# Reverse of RHYTHM_PATTERN_SEQUENCES: duration tuple → rhythm name.
_RHYTHM_BY_SEQUENCE: dict[tuple, str] = {
    seq: name for name, seq in RHYTHM_PATTERN_SEQUENCES.items()
}


def derive_type(pattern: list) -> str:
    seq = tuple(beat[MUSICAL_DURATION] for beat in pattern)
    name = _RHYTHM_BY_SEQUENCE.get(seq)
    if name is not None:
        return name
    raise ValueError(
        f"Cannot derive rhythm type from duration sequence {seq!r}. "
        f"Known patterns: {list(_RHYTHM_BY_SEQUENCE)}"
    )


def derive_panning_key(entry: dict):
    rp = entry[RHYTHM_PATTERNS]
    if rp and rp[0] is UNTOUCHED:
        return UNTOUCHED
    return rp[0][RHYTHM_PATTERN][0][POSSIBLE_PANNINGS][0]


def with_roles(beats: list, roles: tuple) -> list:
    """Attach SampleRole labels to beats, enabling different samples per beat position.

    Example — A/B/A pattern on a triple rhythm:
        with_roles(triple_rhythm(panning), (SampleRole.SAME, SampleRole.NEW, SampleRole.SAME))

    SampleRole.SAME  — reuse the primary sample drawn for this slot
    SampleRole.NEW   — draw a fresh different sample for this beat
    None             — no role constraint; draw freely (same behavior as today)
    """
    if len(beats) < 2:
        raise ValueError(
            f"with_roles: needs at least 2 beats to be meaningful (got {len(beats)}). "
            f"Use the plain rhythm function for single-beat patterns."
        )
    if len(roles) != len(beats):
        raise ValueError(
            f"with_roles: roles length ({len(roles)}) does not match beats length ({len(beats)}). "
            f"roles={roles!r}"
        )
    for i, role in enumerate(roles):
        if role is not None and not isinstance(role, SampleRole):
            raise TypeError(
                f"with_roles: roles[{i}]={role!r} is not a SampleRole or None. "
                f"Use SampleRole.SAME, SampleRole.NEW, or None."
            )
    if roles[0] == SampleRole.NEW:
        raise ValueError(
            f"with_roles: roles[0] cannot be SampleRole.NEW — the first beat establishes "
            f"the primary sample; there is nothing to be 'new' relative to. "
            f"Use SampleRole.SAME or None for the first beat."
        )
    if not any(r == SampleRole.NEW for r in roles):
        raise ValueError(
            f"with_roles: no SampleRole.NEW found in roles={roles!r}. "
            f"If every beat reuses the primary sample, with_roles is unnecessary — "
            f"use the plain rhythm function instead."
        )
    for i in range(1, len(roles)):
        if roles[i] == SampleRole.NEW and roles[i - 1] == SampleRole.NEW:
            raise ValueError(
                f"with_roles: roles[{i - 1}] and roles[{i}] are both SampleRole.NEW. "
                f"Consecutive NEW beats would draw a third distinct sample, which is not supported. "
                f"Separate NEW beats with at least one SAME or None beat."
            )
    result = []
    for beat, role in zip(beats, roles):
        if role is not None:
            result.append({**beat, SAMPLE_ROLE: role})
        else:
            result.append(beat)
    return result


def _beat(duration: float, panning) -> dict:
    return {MUSICAL_DURATION: duration, POSSIBLE_PANNINGS: [panning]}


def quarter_rhythm(panning) -> list:
    return [_beat(QUARTER_NOTE, panning)]


def quarter_quarter_rhythm(panning) -> list:
    return [_beat(QUARTER_NOTE, panning), _beat(QUARTER_NOTE, panning)]


def eighth_eighth_rhythm(panning) -> list:
    return [_beat(EIGHTH, panning), _beat(EIGHTH, panning)]


def quarter_eighth_eighth_rhythm(panning) -> list:
    return [_beat(QUARTER_NOTE, panning), _beat(EIGHTH, panning), _beat(EIGHTH, panning)]


def sixteenth_rhythm(panning) -> list:
    return [_beat(SIXTEENTH, panning)]


def sixteenth_sixteenth_sixteenth_sixteenth_quarter_rhythm(panning) -> list:
    return [_beat(SIXTEENTH, panning) for _ in range(4)] + [_beat(QUARTER_NOTE, panning)]


def eighth_sixteenth_sixteenth_quarter_rhythm(panning) -> list:
    return [_beat(EIGHTH, panning), _beat(SIXTEENTH, panning), _beat(SIXTEENTH, panning), _beat(QUARTER_NOTE, panning)]


def sixteenth_dottedeighth_quarter_rhythm(panning) -> list:
    return [_beat(SIXTEENTH, panning), _beat(DOTTED_EIGHTH, panning), _beat(QUARTER_NOTE, panning)]


def sixteenth_eighth_sixteenth_quarter_rhythm(panning) -> list:
    return [_beat(SIXTEENTH, panning), _beat(EIGHTH, panning), _beat(SIXTEENTH, panning), _beat(QUARTER_NOTE, panning)]


def sixteenth_dottedeighth_sixteenth_dottedeight_rhythm(panning) -> list:
    return [_beat(SIXTEENTH, panning), _beat(DOTTED_EIGHTH, panning), _beat(SIXTEENTH, panning), _beat(DOTTED_EIGHTH, panning)]


def sixteenth_dottedeighth_rhythm(panning) -> list:
    return [_beat(SIXTEENTH, panning), _beat(DOTTED_EIGHTH, panning)]


def eighth_eighth_eighth_eighth_rhythm(panning) -> list:
    return [_beat(EIGHTH, panning) for _ in range(4)]


def eighth_eighth_eighth_rhythm(panning) -> list:
    return [_beat(EIGHTH, panning) for _ in range(3)]


def eighth_rhythm(panning) -> list:
    return [_beat(EIGHTH, panning)]


def eighth_eighth_quarter_rhythm(panning) -> list:
    return [_beat(EIGHTH, panning), _beat(EIGHTH, panning), _beat(QUARTER_NOTE, panning)]


def quarter_rest_quarter_rhythm(panning) -> list:
    return [_beat(QUARTER_NOTE, panning), _beat(QUARTER_NOTE_REST, panning), _beat(QUARTER_NOTE, panning)]


def quarter_quarter_quarter_rhythm(panning) -> list:
    return [_beat(QUARTER_NOTE, panning) for _ in range(3)]


def quarter_quarter_quarter_quarter_rhythm(panning) -> list:
    return [_beat(QUARTER_NOTE, panning) for _ in range(4)]


KICK_SNARE_MUSICAL_PATTERNS: list = [
    {
        VOLUMES: [QUIET],
        BPMS: [SLOW],
        MUSIC_PATTERN_PERCENT: 100,
        RHYTHM_PATTERNS: [
            {
                RHYTHM_PATTERN: quarter_rhythm(HARD_CENTER),
                RHYTHM_PERCENT: 50,
            },
            {
                RHYTHM_PATTERN: sixteenth_dottedeighth_rhythm(HARD_CENTER),
                RHYTHM_PERCENT: 9,
            },
            {
                RHYTHM_PATTERN: quarter_quarter_rhythm(HARD_CENTER),
                RHYTHM_PERCENT: 9,
            },
            {
                RHYTHM_PATTERN: quarter_eighth_eighth_rhythm(HARD_CENTER),
                RHYTHM_PERCENT: 8,
            },
            {
                RHYTHM_PATTERN: sixteenth_dottedeighth_quarter_rhythm(HARD_CENTER),
                RHYTHM_PERCENT: 8,
            },
            {
                RHYTHM_PATTERN: sixteenth_dottedeighth_sixteenth_dottedeight_rhythm(HARD_CENTER),
                RHYTHM_PERCENT: 8,
            },
            {
                RHYTHM_PATTERN: with_roles(quarter_quarter_quarter_rhythm(HARD_CENTER), (SampleRole.SAME, SampleRole.NEW, SampleRole.SAME)),
                RHYTHM_PERCENT: 8
            }
        ],
    }
]

KICKSTAB_SNARESTAB_MUSICAL_PATTERNS: list = [
        {
        VOLUMES: [LOUD],
        BPMS: [FAST],
        MUSIC_PATTERN_PERCENT: 25,
        RHYTHM_PATTERNS: [
            {
                RHYTHM_PATTERN: quarter_rhythm(HARD_LEFT),
                RHYTHM_PERCENT: 50,
            },
            {
                RHYTHM_PATTERN: quarter_eighth_eighth_rhythm(HARD_LEFT),
                RHYTHM_PERCENT: 25,
            },
            {
                RHYTHM_PATTERN: sixteenth_sixteenth_sixteenth_sixteenth_quarter_rhythm(HARD_LEFT),
                RHYTHM_PERCENT: 25,
            },
        ]

    },
    {
        VOLUMES: [LOUD],
        BPMS: [FAST],
        MUSIC_PATTERN_PERCENT: 25,
        RHYTHM_PATTERNS: [
            {
                RHYTHM_PATTERN: quarter_rhythm(HARD_RIGHT),
                RHYTHM_PERCENT: 50,
            },
            {
                RHYTHM_PATTERN: quarter_eighth_eighth_rhythm(HARD_RIGHT),
                RHYTHM_PERCENT: 25,
            },
            {
                RHYTHM_PATTERN: sixteenth_sixteenth_sixteenth_sixteenth_quarter_rhythm(HARD_RIGHT),
                RHYTHM_PERCENT: 25,
            },
        ]

    },
    {
        VOLUMES: [LOUD],
        BPMS: [FAST],
        MUSIC_PATTERN_PERCENT: 50,
        RHYTHM_PATTERNS: [
            {
                RHYTHM_PATTERN: quarter_rhythm(DUALPAN_LEFTRIGHT),
                RHYTHM_PERCENT: 100,
            },
        ],
    },
]


ACAPPELLA_MUSICAL_PATTERNS: list = [
    {
        VOLUMES: [LOUD],
        BPMS: [SLOW],
        MUSIC_PATTERN_PERCENT: 50,
        RHYTHM_PATTERNS: [
            {
                RHYTHM_PATTERN: quarter_rhythm(HARD_LEFT),
                RHYTHM_PERCENT: 100,
            },
        ],
    },
        {
        VOLUMES: [LOUD],
        BPMS: [SLOW],
        MUSIC_PATTERN_PERCENT: 50,
        RHYTHM_PATTERNS: [
            {
                RHYTHM_PATTERN: quarter_rhythm(HARD_RIGHT),
                RHYTHM_PERCENT: 100,
            },
        ],
    },
]

# Startup check: PERMUTATION_COMBOS_PER_SAMPLE in constants.py must stay in
# sync with the actual rhythm counts in each musical pattern list.
_actual_combos: dict[str, int] = {
    KICKSNARE:  sum(len(mp[RHYTHM_PATTERNS]) for mp in KICK_SNARE_MUSICAL_PATTERNS),
    STAB:       sum(len(mp[RHYTHM_PATTERNS]) for mp in KICKSTAB_SNARESTAB_MUSICAL_PATTERNS),
    ACAPPELLA:  sum(len(mp[RHYTHM_PATTERNS]) for mp in ACAPPELLA_MUSICAL_PATTERNS),
}
for _group, _expected in PERMUTATION_COMBOS_PER_SAMPLE.items():
    if _actual_combos.get(_group) != _expected:
        raise ValueError(
            f"PERMUTATION_COMBOS_PER_SAMPLE['{_group}'] is {_expected} in constants.py "
            f"but the actual rhythm count is {_actual_combos.get(_group)}. "
            f"Update PERMUTATION_COMBOS_PER_SAMPLE in constants.py to match."
        )


_SOUND_TYPE_RULES: list[dict] = [
    {
        MUSICAL_GROUPING: KICK,
        DUALPAN_PARTNERS: [],
        MUSICAL_PATTERNS: KICK_SNARE_MUSICAL_PATTERNS,
    },
    {
        MUSICAL_GROUPING: SNARE,
        DUALPAN_PARTNERS: [],
        MUSICAL_PATTERNS: KICK_SNARE_MUSICAL_PATTERNS,
    },
    {
        MUSICAL_GROUPING: KICKSTAB,
        DUALPAN_PARTNERS: [KICKSTAB],
        MUSICAL_PATTERNS: KICKSTAB_SNARESTAB_MUSICAL_PATTERNS,
    },
    {
        MUSICAL_GROUPING: SNARESTAB,
        DUALPAN_PARTNERS: [SNARESTAB],
        MUSICAL_PATTERNS: KICKSTAB_SNARESTAB_MUSICAL_PATTERNS,
    },
    {
        MUSICAL_GROUPING: ACAPPELLA,
        DUALPAN_PARTNERS: [],
        MUSICAL_PATTERNS: ACAPPELLA_MUSICAL_PATTERNS,
    },
    {
        MUSICAL_GROUPING: STRINGS,
        DUALPAN_PARTNERS: [],
        MUSICAL_PATTERNS: [
            {
                VOLUMES: [UNTOUCHED],
                BPMS: [UNTOUCHED],
                MUSIC_PATTERN_PERCENT: 100,
                RHYTHM_PATTERNS: [UNTOUCHED],
            },
        ],
    },
]

_VALID_MUSICAL_GROUPINGS = {KICK, SNARE, KICKSTAB, SNARESTAB, ACAPPELLA, STRINGS}

for _rule in _SOUND_TYPE_RULES:
    if _rule[MUSICAL_GROUPING] not in _VALID_MUSICAL_GROUPINGS:
        raise ValueError(
            f"SOUND_TYPE_RULES entry has unknown musical_grouping: {_rule[MUSICAL_GROUPING]!r}. "
            f"Must be one of {sorted(_VALID_MUSICAL_GROUPINGS)!r}."
        )

# Validate music_pattern_percent sums, pattern shapes, rhythm percent sums, and first-beat panning consistency.
for _rule in _SOUND_TYPE_RULES:
    _name = _rule[MUSICAL_GROUPING]
    _total_mp_pct = sum(e[MUSIC_PATTERN_PERCENT] for e in _rule[MUSICAL_PATTERNS])
    if _total_mp_pct != 100:
        raise ValueError(
            f"SOUND_TYPE_RULES entry '{_name}': MUSICAL_PATTERNS percents sum to "
            f"{_total_mp_pct}, must be 100."
        )
    for _group_entry in _rule[MUSICAL_PATTERNS]:
        _rp = _group_entry[RHYTHM_PATTERNS]
        if _rp and _rp[0] is UNTOUCHED:
            continue
        _total_pct = sum(_entry[RHYTHM_PERCENT] for _entry in _rp)
        if _total_pct != 100:
            raise ValueError(
                f"SOUND_TYPE_RULES entry '{_name}': RHYTHM_PATTERNS percents sum to "
                f"{_total_pct}, must be 100."
            )
        for _entry in _rp:
            derive_type(_entry[RHYTHM_PATTERN])  # raises ValueError on invalid shape
        _first_pannings = {
            _entry[RHYTHM_PATTERN][0][POSSIBLE_PANNINGS][0]
            for _entry in _rp
        }
        if len(_first_pannings) > 1:
            raise ValueError(
                f"SOUND_TYPE_RULES entry '{_name}': panning group has inconsistent "
                f"first-beat pannings across rhythm_patterns: {_first_pannings!r}"
            )

rules_by_sound_type: dict[str, dict] = {
    rule[MUSICAL_GROUPING]: rule for rule in _SOUND_TYPE_RULES
}

for _rule in _SOUND_TYPE_RULES:
    _name = _rule[MUSICAL_GROUPING]
    if DUALPAN_PARTNERS not in _rule:
        raise ValueError(
            f"SOUND_TYPE_RULES entry '{_name}' is missing '{DUALPAN_PARTNERS}'. Use [] if no dualpan."
        )
    if _rule[DUALPAN_PARTNERS] and not any(derive_panning_key(e) in (DUALPAN_LEFTRIGHT, DUALPAN_DIAGONAL) for e in _rule[MUSICAL_PATTERNS]):
        raise ValueError(
            f"SOUND_TYPE_RULES entry '{_name}' declares dualpan_partners but has no 'dualpan' panning."
        )

panning_compat: dict[str, set] = {
    group: {
        derive_panning_key(entry)
        for sound_type in types
        for entry in rules_by_sound_type.get(sound_type, {}).get(MUSICAL_PATTERNS, [])
        if derive_panning_key(entry) is not UNTOUCHED
    }
    for group, types in SOUND_GROUP_TYPES.items()
}

panning_percents: dict[str, dict] = {
    group: {
        derive_panning_key(entry): entry[MUSIC_PATTERN_PERCENT]
        for sound_type in types
        for entry in rules_by_sound_type.get(sound_type, {}).get(MUSICAL_PATTERNS, [])
        if derive_panning_key(entry) is not UNTOUCHED
    }
    for group, types in SOUND_GROUP_TYPES.items()
}



def sound_type_of(sample_name: str) -> str:
    parts = sample_name.split('_')
    raw = parts[2].split('.')[0].lower() if len(parts) >= 3 else sample_name.split('.')[0].lower()
    return ACAPPELLA if raw == ACAPPELLA else raw


def passes_through_unmodified(sound_type: str) -> bool:
    rule = rules_by_sound_type.get(sound_type)
    if rule is None:
        return False
    return any(
        entry[RHYTHM_PATTERNS] and entry[RHYTHM_PATTERNS][0] is UNTOUCHED
        for entry in rule[MUSICAL_PATTERNS]
    )


