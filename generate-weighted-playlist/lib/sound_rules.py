from .constants import (
    HARD_CENTER, HARD_LEFT, HARD_RIGHT, DIAGONAL_LEFT, DIAGONAL_RIGHT,
    DUALPAN_LEFTRIGHT, DUALPAN_DIAGONAL, UNTOUCHED,
    LOUD, QUIET, SLOW, FAST,
    KICK, SNARE, KICKSTAB, SNARESTAB, ACAPPELLA, STRINGS,
    SOUND_GROUP_TYPES,
    QUARTER_NOTE, QUARTER_NOTE_REST,
    SINGLE_RHYTHM, DOUBLE_RHYTHM, SINGLE_REST_RHYTHM, SINGLE_REST_SINGLE_RHYTHM, TRIPLE_RHYTHM, SINGLE_SINGLE_REST_RHYTHM,
    MUSICAL_DURATION, POSSIBLE_PANNINGS, RHYTHM_PATTERNS, VOLUMES, BPMS,
    MUSICAL_GROUPING, DUALPAN_PARTNERS, MUSICAL_PATTERNS,
    RHYTHM_PATTERN, RHYTHM_PERCENT,
    MUSIC_PATTERN_PERCENT,
)


def derive_type(pattern: list) -> str:
    if len(pattern) == 1 and pattern[0][MUSICAL_DURATION] == QUARTER_NOTE:
        return SINGLE_RHYTHM
    if (len(pattern) == 2
            and pattern[0][MUSICAL_DURATION] == QUARTER_NOTE
            and pattern[1][MUSICAL_DURATION] == QUARTER_NOTE):
        return DOUBLE_RHYTHM
    if (len(pattern) == 2
            and pattern[0][MUSICAL_DURATION] == QUARTER_NOTE
            and pattern[1][MUSICAL_DURATION] == QUARTER_NOTE_REST):
        return SINGLE_REST_RHYTHM
    if (len(pattern) == 3
            and pattern[0][MUSICAL_DURATION] == QUARTER_NOTE
            and pattern[1][MUSICAL_DURATION] == QUARTER_NOTE_REST
            and pattern[2][MUSICAL_DURATION] == QUARTER_NOTE):
        return SINGLE_REST_SINGLE_RHYTHM
    if (len(pattern) == 3
            and pattern[0][MUSICAL_DURATION] == QUARTER_NOTE
            and pattern[1][MUSICAL_DURATION] == QUARTER_NOTE
            and pattern[2][MUSICAL_DURATION] == QUARTER_NOTE):
        return TRIPLE_RHYTHM
    if (len(pattern) == 3
            and pattern[0][MUSICAL_DURATION] == QUARTER_NOTE
            and pattern[1][MUSICAL_DURATION] == QUARTER_NOTE
            and pattern[2][MUSICAL_DURATION] == QUARTER_NOTE_REST):
        return SINGLE_SINGLE_REST_RHYTHM
    raise ValueError(
        f"Cannot derive pattern type from: {pattern!r}. "
        f"Must be {SINGLE_RHYTHM} (len=1, QN), {DOUBLE_RHYTHM} (len=2, QN+QN), "
        f"{SINGLE_REST_RHYTHM} (len=2, QN+QNR), "
        f"{SINGLE_REST_SINGLE_RHYTHM} (len=3, QN+QNR+QN), "
        f"{TRIPLE_RHYTHM} (len=3, QN+QN+QN), "
        f"or {SINGLE_SINGLE_REST_RHYTHM} (len=3, QN+QN+QNR)."
    )


def derive_panning_key(entry: dict):
    rp = entry[RHYTHM_PATTERNS]
    if rp and rp[0] is UNTOUCHED:
        return UNTOUCHED
    return rp[0][RHYTHM_PATTERN][0][POSSIBLE_PANNINGS][0]


def single_rhythm(panning) -> list:
    return [
        {
            MUSICAL_DURATION: QUARTER_NOTE,
            POSSIBLE_PANNINGS: [panning],
        },
    ]


def rest_rhythm(panning) -> list:
    return [
        {
            MUSICAL_DURATION: QUARTER_NOTE_REST,
            POSSIBLE_PANNINGS: [panning],
        },
    ]


def double_rhythm(panning1, panning2) -> list:
    return [*single_rhythm(panning1), *single_rhythm(panning2)]


def single_rest_rhythm(panning, rest_panning) -> list:
    return [*single_rhythm(panning), *rest_rhythm(rest_panning)]


def single_rest_single_rhythm(panning, rest_panning, final_panning) -> list:
    return [*single_rhythm(panning), *rest_rhythm(rest_panning), *single_rhythm(final_panning)]


def triple_rhythm(panning1, panning2, panning3) -> list:
    return [*single_rhythm(panning1), *single_rhythm(panning2), *single_rhythm(panning3)]


def single_single_rest(panning1, panning2, rest_panning) -> list:
    return [*single_rhythm(panning1), *single_rhythm(panning2), *rest_rhythm(rest_panning)]


KICK_SNARE_MUSICAL_PATTERNS: list = [
    {
        VOLUMES: [QUIET],
        BPMS: [SLOW],
        MUSIC_PATTERN_PERCENT: 100,
        RHYTHM_PATTERNS: [
            {
                RHYTHM_PATTERN: single_rhythm(HARD_CENTER),
                RHYTHM_PERCENT: 100,
            },
        ],
    },
]


KICKSTAB_SNARESTAB_MUSICAL_PATTERNS: list = [
    {
        VOLUMES: [LOUD],
        BPMS: [FAST],
        MUSIC_PATTERN_PERCENT: 30,
        RHYTHM_PATTERNS: [
            {
                RHYTHM_PATTERN: single_rhythm(HARD_LEFT),
                RHYTHM_PERCENT: 50,
            },
            {
                RHYTHM_PATTERN: double_rhythm(HARD_LEFT, HARD_LEFT),
                RHYTHM_PERCENT: 50,
            },
        ],
    },
    {
        VOLUMES: [LOUD],
        BPMS: [FAST],
        MUSIC_PATTERN_PERCENT: 30,
        RHYTHM_PATTERNS: [
            {
                RHYTHM_PATTERN: single_rhythm(HARD_RIGHT),
                RHYTHM_PERCENT: 50,
            },
            {
                RHYTHM_PATTERN: double_rhythm(HARD_RIGHT, HARD_RIGHT),
                RHYTHM_PERCENT: 50,
            },
        ],
    },
    {
        VOLUMES: [LOUD],
        BPMS: [FAST],
        MUSIC_PATTERN_PERCENT: 40,
        RHYTHM_PATTERNS: [
            {
                RHYTHM_PATTERN: single_rhythm(DUALPAN_LEFTRIGHT),
                RHYTHM_PERCENT: 100,
            },
        ],
    },
]


ACAPPELLA_MUSICAL_PATTERNS: list = [
    {
        VOLUMES: [LOUD],
        BPMS: [SLOW],
        MUSIC_PATTERN_PERCENT: 100,
        RHYTHM_PATTERNS: [
            {
                RHYTHM_PATTERN: single_rhythm(DUALPAN_LEFTRIGHT),
                RHYTHM_PERCENT: 100,
            },
        ],
    },
]


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
        DUALPAN_PARTNERS: [ACAPPELLA],
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
    return rule is not None and UNTOUCHED in rule[MUSICAL_PATTERNS]


