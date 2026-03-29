DIAGONAL_PAN_OFFSET = 0.65

SOUND_GROUP_NAMES: list[str] = ['kicksnare', 'stab', 'acappella']

SOUND_GROUP_TYPES: dict[str, set[str]] = {
    'kicksnare': {'kick', 'snare'},
    'stab':      {'kickstab', 'snarestab'},
    'acappella': {'acappella'},
}

def expand_rhythmic_patterns(num_patterns: list[dict]) -> list[list]:
    return [entry['rhythmic_pattern'] for entry in num_patterns for _ in range(entry['number'])]


_QUARTER_NOTE_NUM_PATTERNS = [
    {'rhythmic_pattern': [1],    'number': 21},
    {'rhythmic_pattern': [1, 1], 'number': 1},
    {'rhythmic_pattern': [1, 1, 0], 'number': 1},
    {'rhythmic_pattern': [1, 0], 'number': 1},
]

QUARTER_NOTE_RHYTHMIC_PATTERNS = expand_rhythmic_patterns(_QUARTER_NOTE_NUM_PATTERNS)
print('$$$ QUARTER_NOTE_RHYTHMIC_PATTERNS:', QUARTER_NOTE_RHYTHMIC_PATTERNS)


_KICK_SNARE_PANNINGS: dict = {
    'center': {
        'volumes': {
        'quiet': {
            'bpms': ['slow'],
            'rhythm_patterns': [[1]]
        }
        }
    },
    'leftorright': {
        'volumes': {
            'loud': {
                'bpms': ['slow'],
                'rhythm_patterns': QUARTER_NOTE_RHYTHMIC_PATTERNS
            }
        }
    },
    'dualpan': {
        'volumes': {
            'loud': {
                'bpms': ['slow', 'fast'],
                'rhythm_patterns': QUARTER_NOTE_RHYTHMIC_PATTERNS
            }
        }
    },
}

_KICKSTAB_SNARESTAB_PANNINGS: dict = {
    'diagonal': {
        'volumes': {
            'quiet': {
                'bpms': ['slow'],
                'rhythm_patterns': QUARTER_NOTE_RHYTHMIC_PATTERNS
            }
        }
    },
    'leftorright': {
        'volumes': {
            'loud': {
                'bpms': ['slow'],
                'rhythm_patterns': QUARTER_NOTE_RHYTHMIC_PATTERNS
            }
        }
    },
}

SOUND_TYPE_RULES: list[dict] = [
    {
        'musical_grouping': 'kick',
        'dualpan_partners': ['kick'],
        'pannings': _KICK_SNARE_PANNINGS
    },
    {
        'musical_grouping': 'snare',
        'dualpan_partners': ['snare'],
        'pannings': _KICK_SNARE_PANNINGS
    },
    {
        'musical_grouping': 'kickstab',
        'dualpan_partners': [],
        'pannings': _KICKSTAB_SNARESTAB_PANNINGS
    },
    {
        'musical_grouping': 'snarestab',
        'dualpan_partners': [],
        'pannings': _KICKSTAB_SNARESTAB_PANNINGS
    },
    {
        'musical_grouping': 'acappella',
        'dualpan_partners': [],
        'pannings': {
            'center': {
                'volumes': {
                    'loud': {
                        'bpms': ['slow'],
                        'rhythm_patterns': [[1], [1, 1]]
                    }
                }
            },
            'leftorright': {
                'volumes': {
                    'quiet': {
                        'bpms': ['slow'],
                        'rhythm_patterns': QUARTER_NOTE_RHYTHMIC_PATTERNS
                    },
                }
            },
        },
    },
    {
        'musical_grouping': 'strings',
        'dualpan_partners': [],
        'pannings': {
            'untouched': {
                'volumes': {
                    'untouched': {
                        'bpms': ['untouched'],
                        'rhythm_patterns': ['untouched']
                    }
                }
            },
        },
    },
]

rules_by_sound_type: dict[str, dict] = {
    rule['musical_grouping']: rule for rule in SOUND_TYPE_RULES
}

for _rule in SOUND_TYPE_RULES:
    _name = _rule['musical_grouping']
    if 'dualpan_partners' not in _rule:
        raise ValueError(
            f"SOUND_TYPE_RULES entry '{_name}' is missing 'dualpan_partners'. Use [] if no dualpan."
        )
    if _rule['dualpan_partners'] and 'dualpan' not in _rule['pannings']:
        raise ValueError(
            f"SOUND_TYPE_RULES entry '{_name}' declares dualpan_partners but has no 'dualpan' panning."
        )

panning_compat: dict[str, set[str]] = {
    group:
    {
        pan
        for sound_type in types
        for pan in rules_by_sound_type.get(sound_type, {}).get('pannings', {})
        if pan != 'untouched'
    }
    for group, types in SOUND_GROUP_TYPES.items()
}


def sound_type_of(sample_name: str) -> str:
    parts = sample_name.split('_')
    raw = parts[2].split('.')[0].lower() if len(parts) >= 3 else sample_name.split('.')[0].lower()
    return 'acappella' if raw.startswith('acap') else raw


def passes_through_unmodified(sound_type: str) -> bool:
    rule = rules_by_sound_type.get(sound_type)
    return rule is not None and 'untouched' in rule['pannings']


def diagonal_pan_value_for_side(side: str) -> float:
    return -DIAGONAL_PAN_OFFSET if side == 'left' else DIAGONAL_PAN_OFFSET