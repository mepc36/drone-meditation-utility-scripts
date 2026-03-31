from . import config as _cfg
from .constants import (
    HARD_CENTER, HARD_LEFT, HARD_RIGHT, DIAGONAL_LEFT, DIAGONAL_RIGHT,
    DUALPAN, UNTOUCHED,
    LOUD, QUIET, SLOW, FAST,
    KICK, SNARE, KICKSTAB, SNARESTAB, ACAPPELLA, STRINGS,
    SOUND_GROUP_TYPES,
    QUARTER_NOTE, QUARTER_NOTE_REST,
    SINGLE_RHYTHM, DOUBLE_RHYTHM, SINGLE_REST_RHYTHM, SINGLE_REST_SINGLE_RHYTHM, TRIPLE_RHYTHM, SINGLE_SINGLE_REST_RHYTHM,
    VALID_RHYTHM_PATTERN_NAMES,
    MUSICAL_DURATION, POSSIBLE_PANNINGS, RHYTHM_PATTERNS, VOLUMES, BPMS,
    MUSICAL_GROUPING, DUALPAN_PARTNERS, MUSICAL_PATTERNS,
)

_conf = _cfg.load()
pattern_weights = _conf.get(_cfg.CFG_RHYTHM_WEIGHTS, {})

_invalid_rhythm_keys = set(pattern_weights.keys()) - VALID_RHYTHM_PATTERN_NAMES
if _invalid_rhythm_keys:
    raise ValueError(
        f"Invalid rhythm_pattern_weights keys: {sorted(_invalid_rhythm_keys)}. "
        f"Valid names are: {sorted(VALID_RHYTHM_PATTERN_NAMES)}"
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
    return rp[0][0][POSSIBLE_PANNINGS][0]


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
        RHYTHM_PATTERNS: [
            single_rhythm(HARD_CENTER),
        ],
    }
]


KICKSTAB_SNARESTAB_MUSICAL_PATTERNS: list = [
    {
        VOLUMES: [LOUD],
        BPMS: [FAST],
        RHYTHM_PATTERNS: [
            single_rhythm(HARD_LEFT),
            double_rhythm(HARD_LEFT, HARD_LEFT),
        ],
    },
    {
        VOLUMES: [LOUD],
        BPMS: [FAST],
        RHYTHM_PATTERNS: [
            single_rhythm(HARD_RIGHT),
            double_rhythm(HARD_RIGHT, HARD_RIGHT),
        ],
    },
                {
        VOLUMES: [LOUD],
        BPMS: [FAST],
        RHYTHM_PATTERNS: [
            single_rhythm(DUALPAN),
        ],
    },
]

ACAPPELLA_MUSICAL_PATTERNS = [
    #     {
    #     VOLUMES: [QUIET],
    #     BPMS: [SLOW],
    #     RHYTHM_PATTERNS: [
    #         single_rhythm(HARD_CENTER),
    #     ],
    # },
    {
        VOLUMES: [LOUD],
        BPMS: [SLOW],
        RHYTHM_PATTERNS: [
            single_rhythm(DIAGONAL_LEFT),
        ],
    },
        {
        VOLUMES: [LOUD],
        BPMS: [SLOW],
        RHYTHM_PATTERNS: [
            single_rhythm(DIAGONAL_RIGHT),
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

# Validate pattern shapes and first-beat panning consistency within each panning group.
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

rules_by_sound_type: dict[str, dict] = {
    rule[MUSICAL_GROUPING]: rule for rule in _SOUND_TYPE_RULES
}

for _rule in _SOUND_TYPE_RULES:
    _name = _rule[MUSICAL_GROUPING]
    if DUALPAN_PARTNERS not in _rule:
        raise ValueError(
            f"SOUND_TYPE_RULES entry '{_name}' is missing '{DUALPAN_PARTNERS}'. Use [] if no dualpan."
        )
    if _rule[DUALPAN_PARTNERS] and not any(derive_panning_key(e) == DUALPAN for e in _rule[MUSICAL_PATTERNS]):
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

# Validate that every panning used in sound_rules has a non-zero allocation in config.
_PANNING_WEIGHT_BY_VALUE: dict = {
    HARD_CENTER:    ("center",   _conf["center_weight"]),
    DIAGONAL_LEFT:  ("diagonal", _conf["diagonal_weight"]),
    DIAGONAL_RIGHT: ("diagonal", _conf["diagonal_weight"]),
    DUALPAN:        ("dualpan",  _conf["dualpan_weight"]),
    HARD_LEFT:      ("left",     _conf["left_weight"]),
    HARD_RIGHT:     ("right",    _conf["right_weight"]),
}

for _rule in _SOUND_TYPE_RULES:
    _seen_pans: set = set()
    for _entry in _rule[MUSICAL_PATTERNS]:
        _pan = derive_panning_key(_entry)
        if _pan is UNTOUCHED or _pan in _seen_pans:
            continue
        _seen_pans.add(_pan)
        _label, _weight = _PANNING_WEIGHT_BY_VALUE[_pan]
        if _weight == 0:
            raise ValueError(
                f"sound_rules uses '{_label}' panning for '{_rule[MUSICAL_GROUPING]}' "
                f"but '{_cfg.CFG_PANNING_PERCENTS}' allocates 0% to it."
            )

# Validate that every rhythm_pattern_weights key is actually used in sound_rules.
_used_rhythm_types: set[str] = set()
for _rule in _SOUND_TYPE_RULES:
    for _pan_entry in _rule[MUSICAL_PATTERNS]:
        _rp = _pan_entry[RHYTHM_PATTERNS]
        if _rp and _rp[0] is UNTOUCHED:
            continue
        for _pat in _rp:
            if isinstance(_pat, list):
                _used_rhythm_types.add(derive_type(_pat))

_unused_rhythm_weights = set(pattern_weights.keys()) - _used_rhythm_types
if _unused_rhythm_weights:
    raise ValueError(
        f"rhythm_pattern_weights defines unused keys: {sorted(_unused_rhythm_weights)}. "
        f"Either use them in sound_rules or remove them from the config."
    )

_unweighted_rhythm_types = _used_rhythm_types - set(pattern_weights.keys())
if _unweighted_rhythm_types:
    raise ValueError(
        f"sound_rules uses rhythm types with no weight in rhythm_pattern_weights: "
        f"{sorted(_unweighted_rhythm_types)}. Add them to the config."
    )


def sound_type_of(sample_name: str) -> str:
    parts = sample_name.split('_')
    raw = parts[2].split('.')[0].lower() if len(parts) >= 3 else sample_name.split('.')[0].lower()
    return ACAPPELLA if raw == ACAPPELLA else raw


def passes_through_unmodified(sound_type: str) -> bool:
    rule = rules_by_sound_type.get(sound_type)
    return rule is not None and UNTOUCHED in rule[MUSICAL_PATTERNS]


