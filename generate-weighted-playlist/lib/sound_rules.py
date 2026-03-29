from . import config as _cfg
from .constants import (
    HARD_CENTER, HARD_LEFT, HARD_RIGHT, DIAGONAL_LEFT, DIAGONAL_RIGHT,
    DUALPAN, UNTOUCHED,
    LOUD, QUIET, SLOW, FAST,
    KICK, SNARE, KICKSTAB, SNARESTAB, ACAPPELLA, STRINGS,
    SOUND_GROUP_TYPES,
    QUARTER_NOTE, QUARTER_NOTE_REST,
    SINGLE_RHYTHM, DOUBLE_RHYTHM, SINGLE_AND_REST_RHYTHM,
    ACAPPELLA_PREFIX,
    MUSICAL_DURATION, POSSIBLE_PANNINGS, RHYTHM_PATTERNS, VOLUMES, BPMS,
    MUSICAL_GROUPING, DUALPAN_PARTNERS, MUSICAL_PATTERNS,
)

_conf = _cfg.load()
pattern_weights = _conf.get(_cfg.CFG_RHYTHM_WEIGHTS, {})


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
        return SINGLE_AND_REST_RHYTHM
    raise ValueError(
        f"Cannot derive pattern type from: {pattern!r}. "
        f"Must be {SINGLE_RHYTHM} (len=1, QN), {DOUBLE_RHYTHM} (len=2, QN+QN), "
        f"or {SINGLE_AND_REST_RHYTHM} (len=2, QN+QNR)."
    )


def _derive_panning_key(entry: dict):
    rp = entry[RHYTHM_PATTERNS]
    if rp and rp[0] is UNTOUCHED:
        return UNTOUCHED
    return rp[0][0][POSSIBLE_PANNINGS][0]


def _pannings_to_dict(pannings_list: list) -> dict:
    result = {}
    for entry in pannings_list:
        key = _derive_panning_key(entry)
        if key in result:
            raise ValueError(f"Duplicate derived panning key: {key!r}")
        result[key] = entry
    return result


def single_rhythm(pannings: list) -> list:
    if len(pannings) > 1:
        raise ValueError(
            f"single_rhythm expects exactly 1 panning entry, got {len(pannings)}"
        )
    return [
        {
            MUSICAL_DURATION: QUARTER_NOTE,
            POSSIBLE_PANNINGS: pannings[0],
        },
    ]


def double_rhythm(pannings: list) -> list:
    if len(pannings) == 1 or len(pannings) > 2:
        raise ValueError(
            f"double_rhythm expects exactly 2 panning entries, got {len(pannings)}"
        )
    return [
        {
            MUSICAL_DURATION: QUARTER_NOTE,
            POSSIBLE_PANNINGS: pannings[0],
        },
        {
            MUSICAL_DURATION: QUARTER_NOTE,
            POSSIBLE_PANNINGS: pannings[1],
        },
    ]


def single_and_rest_rhythm(pannings: list) -> list:
    if len(pannings) > 1:
        raise ValueError(
            f"single_and_rest_rhythm expects exactly 1 panning entry, got {len(pannings)}"
        )
    return [
        {
            MUSICAL_DURATION: QUARTER_NOTE,
            POSSIBLE_PANNINGS: pannings[0],
        },
        {
            MUSICAL_DURATION: QUARTER_NOTE_REST,
            POSSIBLE_PANNINGS: [HARD_CENTER],
        },
    ]


KICK_SNARE_MUSICAL_PATTERNS: list = [
    {
        VOLUMES: [QUIET],
        BPMS: [SLOW],
        RHYTHM_PATTERNS: [
            single_rhythm([[HARD_CENTER]]),
            single_and_rest_rhythm([[HARD_CENTER]]),
        ],
    },
    {
        VOLUMES: [LOUD],
        BPMS: [FAST],
        RHYTHM_PATTERNS: [
            single_rhythm([[HARD_LEFT]]),
            double_rhythm([[HARD_LEFT], [HARD_RIGHT]]),
        ],
    },
    {
        VOLUMES: [LOUD],
        BPMS: [FAST],
        RHYTHM_PATTERNS: [
            single_rhythm([[HARD_RIGHT]]),
            double_rhythm([[HARD_RIGHT], [HARD_LEFT]]),
        ],
    },
    {
        VOLUMES: [LOUD],
        BPMS: [FAST],
        RHYTHM_PATTERNS: [
            single_rhythm([[DUALPAN]]),
            double_rhythm([[DUALPAN], [DUALPAN]]),
        ],
    },
]


KICKSTAB_SNARESTAB_MUSICAL_PATTERNS: list = [
    {
        VOLUMES: [QUIET],
        BPMS: [SLOW],
        RHYTHM_PATTERNS: [
            single_rhythm([[DIAGONAL_LEFT]]),
            single_and_rest_rhythm([[DIAGONAL_LEFT]]),
        ],
    },
    {
        VOLUMES: [QUIET],
        BPMS: [SLOW],
        RHYTHM_PATTERNS: [
            single_rhythm([[DIAGONAL_RIGHT]]),
            single_and_rest_rhythm([[DIAGONAL_RIGHT]]),
        ],
    },
    {
        VOLUMES: [LOUD],
        BPMS: [FAST],
        RHYTHM_PATTERNS: [
            single_rhythm([[HARD_LEFT]]),
            single_and_rest_rhythm([[HARD_LEFT]]),
        ],
    },
    {
        VOLUMES: [LOUD],
        BPMS: [FAST],
        RHYTHM_PATTERNS: [
            single_rhythm([[HARD_RIGHT]]),
            single_and_rest_rhythm([[HARD_RIGHT]]),
        ],
    },
]

ACAPPELLA_MUSICAL_PATTERNS = [
            {
                VOLUMES: [LOUD],
                BPMS: [FAST],
                RHYTHM_PATTERNS: [
                    single_rhythm([[HARD_CENTER]]),
                ],
            },
            {
                VOLUMES: [QUIET],
                BPMS: [SLOW],
                RHYTHM_PATTERNS: [
                    single_rhythm([[HARD_LEFT]]),
                    double_rhythm([[HARD_LEFT], [HARD_RIGHT]]),
                ],
            },
            {
                VOLUMES: [QUIET],
                BPMS: [SLOW],
                RHYTHM_PATTERNS: [
                    single_rhythm([[HARD_RIGHT]]),
                    double_rhythm([[HARD_RIGHT], [HARD_LEFT]]),
                ],
            },
            {
                VOLUMES: [QUIET],
                BPMS: [SLOW],
                RHYTHM_PATTERNS: [
                    single_rhythm([[DIAGONAL_LEFT]]),
                    double_rhythm([[DIAGONAL_LEFT], [DIAGONAL_RIGHT]]),
                ],
            },
            {
                VOLUMES: [QUIET],
                BPMS: [SLOW],
                RHYTHM_PATTERNS: [
                    single_rhythm([[DIAGONAL_RIGHT]]),
                    double_rhythm([[DIAGONAL_RIGHT], [DIAGONAL_LEFT]]),
                ],
            },
        ]


_SOUND_TYPE_RULES: list[dict] = [
    {
        MUSICAL_GROUPING: KICK,
        DUALPAN_PARTNERS: [KICK],
        MUSICAL_PATTERNS: KICK_SNARE_MUSICAL_PATTERNS,
    },
    {
        MUSICAL_GROUPING: SNARE,
        DUALPAN_PARTNERS: [SNARE],
        MUSICAL_PATTERNS: KICK_SNARE_MUSICAL_PATTERNS,
    },
    {
        MUSICAL_GROUPING: KICKSTAB,
        DUALPAN_PARTNERS: [],
        MUSICAL_PATTERNS: KICKSTAB_SNARESTAB_MUSICAL_PATTERNS,
    },
    {
        MUSICAL_GROUPING: SNARESTAB,
        DUALPAN_PARTNERS: [],
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

# Validate pattern shapes and first-beat panning consistency within each panning group,
# then derive the panning key and convert lists to lookup dicts.
for _rule in _SOUND_TYPE_RULES:
    _name = _rule[MUSICAL_GROUPING]
    for _group_entry in _rule[MUSICAL_PATTERNS]:
        _rp = _group_entry[RHYTHM_PATTERNS]
        if _rp and _rp[0] is UNTOUCHED:
            continue
        for _pat in _rp:
            if isinstance(_pat, list):
                derive_type(_pat)  # raises ValueError on invalid shape
        _first_pannings = {
            _pat[0][POSSIBLE_PANNINGS][0]
            for _pat in _rp
            if isinstance(_pat, list)
        }
        if len(_first_pannings) > 1:
            raise ValueError(
                f"SOUND_TYPE_RULES entry '{_name}': panning group has inconsistent "
                f"first-beat pannings across rhythm_patterns: {_first_pannings!r}"
            )

for _rule in _SOUND_TYPE_RULES:
    _rule[MUSICAL_PATTERNS] = _pannings_to_dict(_rule[MUSICAL_PATTERNS])

rules_by_sound_type: dict[str, dict] = {
    rule[MUSICAL_GROUPING]: rule for rule in _SOUND_TYPE_RULES
}

for _rule in _SOUND_TYPE_RULES:
    _name = _rule[MUSICAL_GROUPING]
    if DUALPAN_PARTNERS not in _rule:
        raise ValueError(
            f"SOUND_TYPE_RULES entry '{_name}' is missing '{DUALPAN_PARTNERS}'. Use [] if no dualpan."
        )
    if _rule[DUALPAN_PARTNERS] and DUALPAN not in _rule[MUSICAL_PATTERNS]:
        raise ValueError(
            f"SOUND_TYPE_RULES entry '{_name}' declares dualpan_partners but has no 'dualpan' panning."
        )

panning_compat: dict[str, set] = {
    group: {
        pan
        for sound_type in types
        for pan in rules_by_sound_type.get(sound_type, {}).get(MUSICAL_PATTERNS, {})
        if pan is not UNTOUCHED
    }
    for group, types in SOUND_GROUP_TYPES.items()
}


def sound_type_of(sample_name: str) -> str:
    parts = sample_name.split('_')
    raw = parts[2].split('.')[0].lower() if len(parts) >= 3 else sample_name.split('.')[0].lower()
    return ACAPPELLA if raw.startswith(ACAPPELLA_PREFIX) else raw


def passes_through_unmodified(sound_type: str) -> bool:
    rule = rules_by_sound_type.get(sound_type)
    return rule is not None and UNTOUCHED in rule[MUSICAL_PATTERNS]


