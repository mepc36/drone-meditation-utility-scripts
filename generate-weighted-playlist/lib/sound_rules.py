from . import config as _cfg
from .constants import (
    HARD_CENTER, HARD_LEFT, HARD_RIGHT, DIAGONAL_LEFT, DIAGONAL_RIGHT,
    DUALPAN, UNTOUCHED,
    LOUD, QUIET, SLOW, FAST,
    KICK, SNARE, KICKSTAB, SNARESTAB, ACAPPELLA, STRINGS,
    SOUND_GROUP_TYPES,
    QUARTER_NOTE, QUARTER_NOTE_REST,
)

_conf = _cfg.load()
pattern_weights = _conf.get('rhythm_pattern_weights', {})


def derive_type(pattern: list) -> str:
    if len(pattern) == 1 and pattern[0]['musical_duration'] == QUARTER_NOTE:
        return 'single'
    if (len(pattern) == 2
            and pattern[0]['musical_duration'] == QUARTER_NOTE
            and pattern[1]['musical_duration'] == QUARTER_NOTE):
        return 'double'
    if (len(pattern) == 2
            and pattern[0]['musical_duration'] == QUARTER_NOTE
            and pattern[1]['musical_duration'] == QUARTER_NOTE_REST):
        return 'single_and_rest'
    raise ValueError(
        f"Cannot derive pattern type from: {pattern!r}. "
        f"Must be single (len=1, QN), double (len=2, QN+QN), "
        f"or single_and_rest (len=2, QN+QNR)."
    )


def _derive_panning_key(entry: dict):
    rp = entry['rhythm_patterns']
    if rp and rp[0] is UNTOUCHED:
        return UNTOUCHED
    return rp[0][0]['possible_pannings'][0]


def _pannings_to_dict(pannings_list: list) -> dict:
    result = {}
    for entry in pannings_list:
        key = _derive_panning_key(entry)
        if key in result:
            raise ValueError(f"Duplicate derived panning key: {key!r}")
        result[key] = entry
    return result


# ── Per-panning rhythm pattern pools ─────────────────────────────────────────
# Named <sound_group>_<panning>_<volume>_RHYTHM_PATTERNS.
# 2nd-beat possible_pannings by parent panning position (PPP):
#   HARD_LEFT      → [HARD_CENTER, HARD_RIGHT]
#   HARD_RIGHT     → [HARD_CENTER, HARD_LEFT]
#   DIAGONAL_LEFT  → [DIAGONAL_RIGHT]
#   DIAGONAL_RIGHT → [DIAGONAL_LEFT]
#   DUALPAN        → [HARD_CENTER]


# ── kick / snare ──────────────────────────────────────────────────────────────

# ── kickstab / snarestab ──────────────────────────────────────────────────────

# ── acappella ─────────────────────────────────────────────────────────────────

_KICK_SNARE_PANNINGS: list = [
    {
        'volumes': [QUIET],
        'bpms': [SLOW],
        'rhythm_patterns': [
            [
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [HARD_CENTER],
                },
            ],
        ],
    },
    {
        'volumes': [LOUD],
        'bpms': [FAST],
        'rhythm_patterns': [
            [
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [HARD_LEFT],
                },
            ],
            [
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [HARD_LEFT],
                },
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [HARD_RIGHT],
                },
            ],
            [
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [HARD_LEFT],
                },
                {
                    'musical_duration': QUARTER_NOTE_REST,
                    'possible_pannings': [HARD_CENTER],
                },
            ],
        ],
    },
    {
        'volumes': [LOUD],
        'bpms': [FAST],
        'rhythm_patterns': [
            [
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [HARD_RIGHT],
                },
            ],
            [
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [HARD_RIGHT],
                },
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [HARD_LEFT],
                },
            ],
            [
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [HARD_RIGHT],
                },
                {
                    'musical_duration': QUARTER_NOTE_REST,
                    'possible_pannings': [HARD_CENTER],
                },
            ],
        ],
    },
    {
        'volumes': [LOUD],
        'bpms': [FAST],
        'rhythm_patterns': [
            [
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [DUALPAN],
                },
            ],
            [
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [DUALPAN],
                },
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [DUALPAN],
                },
            ],
            [
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [DUALPAN],
                },
                {
                    'musical_duration': QUARTER_NOTE_REST,
                    'possible_pannings': [HARD_CENTER],
                },
            ],
        ],
    },
]


_KICKSTAB_SNARESTAB_PANNINGS: list = [
    {
        'volumes': [QUIET],
        'bpms': [SLOW],
        'rhythm_patterns': [
            [
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [DIAGONAL_LEFT],
                },
            ],
        ],
    },
    {
        'volumes': [QUIET],
        'bpms': [SLOW],
        'rhythm_patterns': [
            [
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [DIAGONAL_RIGHT],
                },
            ],
        ],
    },
    {
        'volumes': [LOUD],
        'bpms': [FAST],
        'rhythm_patterns': [
            [
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [HARD_LEFT],
                },
            ],
            [
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [HARD_LEFT],
                },
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [HARD_RIGHT],
                },
            ],
            [
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [HARD_LEFT],
                },
                {
                    'musical_duration': QUARTER_NOTE_REST,
                    'possible_pannings': [HARD_CENTER],
                },
            ],
        ],
    },
    {
        'volumes': [LOUD],
        'bpms': [FAST],
        'rhythm_patterns': [
            [
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [HARD_RIGHT],
                },
            ],
            [
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [HARD_RIGHT],
                },
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [HARD_LEFT],
                },
            ],
            [
                {
                    'musical_duration': QUARTER_NOTE,
                    'possible_pannings': [HARD_RIGHT],
                },
                {
                    'musical_duration': QUARTER_NOTE_REST,
                    'possible_pannings': [HARD_CENTER],
                },
            ],
        ],
    },
]


_SOUND_TYPE_RULES: list[dict] = [
    {
        'musical_grouping': KICK,
        'dualpan_partners': [KICK],
        'musical_patterns': _KICK_SNARE_PANNINGS,
    },
    {
        'musical_grouping': SNARE,
        'dualpan_partners': [SNARE],
        'musical_patterns': _KICK_SNARE_PANNINGS,
    },
    {
        'musical_grouping': KICKSTAB,
        'dualpan_partners': [],
        'musical_patterns': _KICKSTAB_SNARESTAB_PANNINGS,
    },
    {
        'musical_grouping': SNARESTAB,
        'dualpan_partners': [],
        'musical_patterns': _KICKSTAB_SNARESTAB_PANNINGS,
    },
    {
        'musical_grouping': ACAPPELLA,
        'dualpan_partners': [],
        'musical_patterns': [
            {
                'volumes': [LOUD],
                'bpms': [FAST],
                'rhythm_patterns': [
                    [
                        {
                            'musical_duration': QUARTER_NOTE,
                            'possible_pannings': [HARD_CENTER],
                        },
                    ],
                ],
            },
            {
                'volumes': [QUIET],
                'bpms': [SLOW],
                'rhythm_patterns': [
                    [
                        {
                            'musical_duration': QUARTER_NOTE,
                            'possible_pannings': [HARD_LEFT],
                        },
                    ],
                    [
                        {
                            'musical_duration': QUARTER_NOTE,
                            'possible_pannings': [HARD_LEFT],
                        },
                        {
                            'musical_duration': QUARTER_NOTE,
                            'possible_pannings': [HARD_RIGHT],
                        },
                    ],
                ],
            },
            {
                'volumes': [QUIET],
                'bpms': [SLOW],
                'rhythm_patterns': [
                    [
                        {
                            'musical_duration': QUARTER_NOTE,
                            'possible_pannings': [HARD_RIGHT],
                        },
                    ],
                    [
                        {
                            'musical_duration': QUARTER_NOTE,
                            'possible_pannings': [HARD_RIGHT],
                        },
                        {
                            'musical_duration': QUARTER_NOTE,
                            'possible_pannings': [HARD_LEFT],
                        },
                    ],
                ],
            },
            {
                'volumes': [QUIET],
                'bpms': [SLOW],
                'rhythm_patterns': [
                    [
                        {
                            'musical_duration': QUARTER_NOTE,
                            'possible_pannings': [DIAGONAL_LEFT],
                        },
                    ],
                    [
                        {
                            'musical_duration': QUARTER_NOTE,
                            'possible_pannings': [DIAGONAL_LEFT],
                        },
                        {
                            'musical_duration': QUARTER_NOTE,
                            'possible_pannings': [DIAGONAL_RIGHT],
                        },
                    ],
                ],
            },
            {
                'volumes': [QUIET],
                'bpms': [SLOW],
                'rhythm_patterns': [
                    [
                        {
                            'musical_duration': QUARTER_NOTE,
                            'possible_pannings': [DIAGONAL_RIGHT],
                        },
                    ],
                    [
                        {
                            'musical_duration': QUARTER_NOTE,
                            'possible_pannings': [DIAGONAL_RIGHT],
                        },
                        {
                            'musical_duration': QUARTER_NOTE,
                            'possible_pannings': [DIAGONAL_LEFT],
                        },
                    ],
                ],
            },
        ],
    },
    {
        'musical_grouping': STRINGS,
        'dualpan_partners': [],
        'musical_patterns': [
            {
                'volumes': [UNTOUCHED],
                'bpms': [UNTOUCHED],
                'rhythm_patterns': [UNTOUCHED],
            },
        ],
    },
]

_VALID_MUSICAL_GROUPINGS = {KICK, SNARE, KICKSTAB, SNARESTAB, ACAPPELLA, STRINGS}

for _rule in _SOUND_TYPE_RULES:
    if _rule['musical_grouping'] not in _VALID_MUSICAL_GROUPINGS:
        raise ValueError(
            f"SOUND_TYPE_RULES entry has unknown musical_grouping: {_rule['musical_grouping']!r}. "
            f"Must be one of {sorted(_VALID_MUSICAL_GROUPINGS)!r}."
        )

# Validate pattern shapes and first-beat panning consistency within each panning group,
# then derive the panning key and convert lists to lookup dicts.
for _rule in _SOUND_TYPE_RULES:
    _name = _rule['musical_grouping']
    for _group_entry in _rule['musical_patterns']:
        _rp = _group_entry['rhythm_patterns']
        if _rp and _rp[0] is UNTOUCHED:
            continue
        for _pat in _rp:
            if isinstance(_pat, list):
                derive_type(_pat)  # raises ValueError on invalid shape
        _first_pannings = {
            _pat[0]['possible_pannings'][0]
            for _pat in _rp
            if isinstance(_pat, list)
        }
        if len(_first_pannings) > 1:
            raise ValueError(
                f"SOUND_TYPE_RULES entry '{_name}': panning group has inconsistent "
                f"first-beat pannings across rhythm_patterns: {_first_pannings!r}"
            )

for _rule in _SOUND_TYPE_RULES:
    _rule['musical_patterns'] = _pannings_to_dict(_rule['musical_patterns'])

rules_by_sound_type: dict[str, dict] = {
    rule['musical_grouping']: rule for rule in _SOUND_TYPE_RULES
}

for _rule in _SOUND_TYPE_RULES:
    _name = _rule['musical_grouping']
    if 'dualpan_partners' not in _rule:
        raise ValueError(
            f"SOUND_TYPE_RULES entry '{_name}' is missing 'dualpan_partners'. Use [] if no dualpan."
        )
    if _rule['dualpan_partners'] and DUALPAN not in _rule['musical_patterns']:
        raise ValueError(
            f"SOUND_TYPE_RULES entry '{_name}' declares dualpan_partners but has no 'dualpan' panning."
        )

panning_compat: dict[str, set] = {
    group: {
        pan
        for sound_type in types
        for pan in rules_by_sound_type.get(sound_type, {}).get('musical_patterns', {})
        if pan is not UNTOUCHED
    }
    for group, types in SOUND_GROUP_TYPES.items()
}


def sound_type_of(sample_name: str) -> str:
    parts = sample_name.split('_')
    raw = parts[2].split('.')[0].lower() if len(parts) >= 3 else sample_name.split('.')[0].lower()
    return ACAPPELLA if raw.startswith('acap') else raw


def passes_through_unmodified(sound_type: str) -> bool:
    rule = rules_by_sound_type.get(sound_type)
    return rule is not None and UNTOUCHED in rule['musical_patterns']


