# Rhythmic durations (in quarter notes)
QUARTER_NOTE = 1
QUARTER_NOTE_REST = 0

# Panning positions (numeric: -1.0 = full left, 0.0 = center, 1.0 = full right)
DIAGONAL_PAN = 0.38
HARD_CENTER  =  0.0
HARD_LEFT    = -1.0
HARD_RIGHT   =  1.0
DIAGONAL_LEFT  = DIAGONAL_PAN * -1
DIAGONAL_RIGHT = DIAGONAL_PAN * 1

# Sound types (individual sample categories)
KICK      = 'kick'
SNARE     = 'snare'
KICKSTAB  = 'kickstab'
SNARESTAB = 'snarestab'
ACAPPELLA = 'acappella'
STRINGS   = 'strings'

# Sound groups (collections of sound types)
KICKSNARE = 'kicksnare'
STAB      = 'stab'

# Ordered list of non-strings sound groups (used for quota allocation)
SOUND_GROUP_NAMES: list[str] = [KICKSNARE, STAB, ACAPPELLA]

# Maps each sound group to the set of sound types it contains
SOUND_GROUP_TYPES: dict[str, set[str]] = {
    KICKSNARE: {KICK, SNARE},
    STAB:      {KICKSTAB, SNARESTAB},
    ACAPPELLA: {ACAPPELLA},
}

# Special panning modes (not positions on the -1..1 stereo field)
DUALPAN_LEFTRIGHT = 2.0   # sentinel: two samples panned hard left + hard right simultaneously
DUALPAN_DIAGONAL  = 3.0   # sentinel: two samples panned diagonal left + diagonal right simultaneously
UNTOUCHED         = None  # sentinel: pass audio through without any panning or processing

# ── Rhythm pattern type identifiers ──────────────────────────────────────────
SINGLE_RHYTHM             = 'single'
DOUBLE_RHYTHM             = 'double'
SINGLE_REST_RHYTHM        = 'single_rest'
SINGLE_REST_SINGLE_RHYTHM = 'single_rest_single'
TRIPLE_RHYTHM             = 'triple'
SINGLE_SINGLE_REST_RHYTHM = 'single_single_rest'

# Maps each rhythm pattern name to its canonical beat sequence (tuple of duration values).
# Used by analyze_ratios to auto-build the filename suffix → pattern name mapping.
# Update this whenever a new rhythm pattern is added.
RHYTHM_PATTERN_SEQUENCES: dict[str, tuple[int, ...]] = {
    SINGLE_RHYTHM:             (QUARTER_NOTE,),
    DOUBLE_RHYTHM:             (QUARTER_NOTE, QUARTER_NOTE),
    SINGLE_REST_RHYTHM:        (QUARTER_NOTE, QUARTER_NOTE_REST),
    SINGLE_REST_SINGLE_RHYTHM: (QUARTER_NOTE, QUARTER_NOTE_REST, QUARTER_NOTE),
    TRIPLE_RHYTHM:             (QUARTER_NOTE, QUARTER_NOTE, QUARTER_NOTE),
    SINGLE_SINGLE_REST_RHYTHM: (QUARTER_NOTE, QUARTER_NOTE, QUARTER_NOTE_REST),
}

# ── Musical pattern dict keys ─────────────────────────────────────────────────
MUSICAL_DURATION  = 'musical_duration'
POSSIBLE_PANNINGS = 'possible_pannings'
RHYTHM_PATTERN    = 'rhythm_pattern'
RHYTHM_PERCENT    = 'rhythm_percent'
MUSIC_PATTERN_PERCENT = 'music_pattern_percent'
RHYTHM_PATTERNS   = 'rhythm_patterns'
VOLUMES           = 'volumes'
BPMS              = 'bpms'
MUSICAL_GROUPING  = 'musical_grouping'
DUALPAN_PARTNERS  = 'dualpan_partners'
MUSICAL_PATTERNS  = 'musical_patterns'

# ── Panning group label strings ────────────────────────────────────────────────
PANNING_CENTER        = 'center'
PANNING_DIAGONAL      = 'diagonal'
PANNING_LEFT          = 'left'
PANNING_RIGHT         = 'right'
PANNING_DUALPAN       = 'dualpan'
PANNING_LEFT_OR_RIGHT = 'leftorright'

# ── Beat name strings for output filenames ────────────────────────────────────
BEAT_NAME_QUARTER_NOTE_REST = 'quarternoterest'
BEAT_NAME_QUARTER_NOTE      = 'quarter'

# ── Operational limits ────────────────────────────────────────────────────────
MAX_DRAW_RETRIES = 20
