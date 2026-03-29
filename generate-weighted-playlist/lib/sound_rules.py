from .constants import (
    QUARTER_NOTE, QUARTER_NOTE_REST, DIAGONAL_PAN_OFFSET,
    HARD_CENTER, HARD_LEFT, HARD_RIGHT, DIAGONAL_LEFT, DIAGONAL_RIGHT,
    DUALPAN, UNTOUCHED,
    LOUD, QUIET, SLOW, FAST,
    KICK, SNARE, KICKSTAB, SNARESTAB, ACAPPELLA, STRINGS,
    KICKSNARE, STAB,
    SOUND_GROUP_NAMES, SOUND_GROUP_TYPES,
)


def _expand_rhythmic_patterns(num_patterns: list[dict]) -> list[list]:
    return [entry['rhythmic_pattern'] for entry in num_patterns for _ in range(entry['number'])]


# Patterns where every beat inherits the slot-level panning (used for dualpan,
# center, diagonal, and any other single-panning-type rules).
_SIMPLE_RHYTHMIC_PATTERNS = [
    {
        'rhythmic_pattern': [QUARTER_NOTE],
        'number': 17,
    },
    {
        'rhythmic_pattern': [QUARTER_NOTE, QUARTER_NOTE],
        'number': 1,
    },
    {
        'rhythmic_pattern': [QUARTER_NOTE, QUARTER_NOTE_REST],
        'number': 1,
    },
]

SIMPLE_RHYTHMIC_PATTERNS = _expand_rhythmic_patterns(_SIMPLE_RHYTHMIC_PATTERNS)

_KICK_SNARE_PANNINGS: dict = {
    HARD_CENTER: {
        'volumes': {
            QUIET: {
                'bpms': [SLOW],
                'rhythm_patterns': [[QUARTER_NOTE]],
            },
        },
    },
    HARD_LEFT: {
        'volumes': {
            LOUD: {
                'bpms': [SLOW],
                'rhythm_patterns': SIMPLE_RHYTHMIC_PATTERNS,
            },
        },
    },
    HARD_RIGHT: {
        'volumes': {
            LOUD: {
                'bpms': [SLOW, FAST],
                'rhythm_patterns': SIMPLE_RHYTHMIC_PATTERNS,
            },
        },
    },
    DUALPAN: {
        'volumes': {
            LOUD: {
                'bpms': [SLOW, FAST],
                'rhythm_patterns': SIMPLE_RHYTHMIC_PATTERNS,
            },
        },
    },
}


_KICKSTAB_SNARESTAB_PANNINGS: dict = {
    DIAGONAL_LEFT: {
        'volumes': {
            QUIET: {
                'bpms': [SLOW],
                'rhythm_patterns': [[QUARTER_NOTE]],
            },
        },
    },
    DIAGONAL_RIGHT: {
        'volumes': {
            QUIET: {
                'bpms': [SLOW],
                'rhythm_patterns': [[QUARTER_NOTE]],
            },
        },
    },
    HARD_LEFT: {
        'volumes': {
            LOUD: {
                'bpms': [SLOW],
                'rhythm_patterns': SIMPLE_RHYTHMIC_PATTERNS,
            },
        },
    },
    HARD_RIGHT: {
        'volumes': {
            LOUD: {
                'bpms': [SLOW],
                'rhythm_patterns': SIMPLE_RHYTHMIC_PATTERNS,
            },
        },
    },
}


_SOUND_TYPE_RULES: list[dict] = [
    {
        'musical_grouping': KICK,
        'dualpan_partners': [KICK],
        'pannings': _KICK_SNARE_PANNINGS,
    },
    {
        'musical_grouping': SNARE,
        'dualpan_partners': [SNARE],
        'pannings': _KICK_SNARE_PANNINGS,
    },
    {
        'musical_grouping': KICKSTAB,
        'dualpan_partners': [],
        'pannings': _KICKSTAB_SNARESTAB_PANNINGS,
    },
    {
        'musical_grouping': SNARESTAB,
        'dualpan_partners': [],
        'pannings': _KICKSTAB_SNARESTAB_PANNINGS,
    },
    {
        'musical_grouping': ACAPPELLA,
        'dualpan_partners': [],
        'pannings': {
            HARD_CENTER: {
                'volumes': {
                    LOUD: {
                        'bpms': [SLOW, FAST],
                        'rhythm_patterns': [[QUARTER_NOTE]],
                    },
                },
            },
            HARD_LEFT: {
                'volumes': {
                    QUIET: {
                        'bpms': [SLOW, FAST],
                        'rhythm_patterns': SIMPLE_RHYTHMIC_PATTERNS,
                    },
                },
            },
            HARD_RIGHT: {
                'volumes': {
                    QUIET: {
                        'bpms': [SLOW, FAST],
                        'rhythm_patterns': SIMPLE_RHYTHMIC_PATTERNS,
                    },
                },
            },
            DIAGONAL_LEFT: {
                'volumes': {
                    QUIET: {
                        'bpms': [SLOW],
                        'rhythm_patterns': [[QUARTER_NOTE]],
                    },
                },
            },
            DIAGONAL_RIGHT: {
                'volumes': {
                    QUIET: {
                        'bpms': [SLOW, FAST],
                        'rhythm_patterns': [[QUARTER_NOTE]],
                    },
                },
            },
        },
    },
    {
        'musical_grouping': STRINGS,
        'dualpan_partners': [],
        'pannings': {
            UNTOUCHED: {
                'volumes': {
                    UNTOUCHED: {
                        'bpms': [UNTOUCHED],
                        'rhythm_patterns': [UNTOUCHED],
                    },
                },
            },
        },
    },
]

rules_by_sound_type: dict[str, dict] = {
    rule['musical_grouping']: rule for rule in _SOUND_TYPE_RULES
}

for _rule in _SOUND_TYPE_RULES:
    _name = _rule['musical_grouping']
    if 'dualpan_partners' not in _rule:
        raise ValueError(
            f"SOUND_TYPE_RULES entry '{_name}' is missing 'dualpan_partners'. Use [] if no dualpan."
        )
    if _rule['dualpan_partners'] and DUALPAN not in _rule['pannings']:
        raise ValueError(
            f"SOUND_TYPE_RULES entry '{_name}' declares dualpan_partners but has no 'dualpan' panning."
        )

panning_compat: dict[str, set] = {
    group: {
        pan
        for sound_type in types
        for pan in rules_by_sound_type.get(sound_type, {}).get('pannings', {})
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
    return rule is not None and UNTOUCHED in rule['pannings']


