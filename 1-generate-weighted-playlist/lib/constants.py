# Rhythmic durations (in quarter notes)
QUARTER_NOTE   = 1
EIGHTH         = 0.5
SIXTEENTH      = 0.25
DOTTED_EIGHTH  = 0.75

# Panning positions (numeric: -1.0 = full left, 0.0 = center, 1.0 = full right)
DIAGONAL_PAN = 0.38
HARD_CENTER  =  0.0
HARD_LEFT    = -1.0
HARD_RIGHT   =  1.0
DIAGONAL_LEFT  = -DIAGONAL_PAN
DIAGONAL_RIGHT =  DIAGONAL_PAN

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

# Random panning: a namedtuple that specifies a symmetric magnitude range.
# When used in POSSIBLE_PANNINGS, a random side (left or right) and a random
# magnitude in [min_magnitude, max_magnitude] are chosen at deck-build time.
# The same two bounds apply symmetrically to both sides — no separate L/R values needed.
# Example: RandomPan(RANDOM_PAN_MIN, RANDOM_PAN_MAX)
#          → uniform draw from [-1.0, -0.1] ∪ [0.1, 1.0]
from collections import namedtuple
RandomPan = namedtuple('RandomPan', ['min_magnitude', 'max_magnitude'])
RANDOM_PAN_MIN = 0.6   # closest-to-center edge of the random pan zone
RANDOM_PAN_MAX = 1.0   # farthest-from-center edge of the random pan zone
RANDOM_PAN_MIN_DIFF = 0.25  # tail beats must be at least this far from the preceding panning

# ── Rhythm pattern type identifiers ──────────────────────────────────────────
QUARTER_RHYTHM                                          = 'quarter'
DOUBLE_RHYTHM                                          = 'double'
TRIPLE_RHYTHM                                          = 'triple'
QUARTER_EIGHTH_EIGHTH_RHYTHM                           = 'quarter_eighth_eighth'
SIXTEENTH_SIXTEENTH_SIXTEENTH_SIXTEENTH_QUARTER_RHYTHM = 'sixteenth_sixteenth_sixteenth_sixteenth_quarter'
SIXTEENTH_DOTTEDEIGHTH_RHYTHM                          = 'sixteenth_dottedeighth'
SIXTEENTH_DOTTEDEIGHTH_QUARTER_RHYTHM                 = 'sixteenth_dottedeighth_quarter'

# Maps each rhythm pattern name to its canonical beat sequence (tuple of duration values).
# Used by analyze_ratios to auto-build the filename suffix → pattern name mapping.
# Update this whenever a new rhythm pattern is added.
RHYTHM_PATTERN_SEQUENCES: dict[str, tuple] = {
    QUARTER_RHYTHM:                                      (QUARTER_NOTE,),
    DOUBLE_RHYTHM:                                       (QUARTER_NOTE, QUARTER_NOTE),
    TRIPLE_RHYTHM:                                       (QUARTER_NOTE, QUARTER_NOTE, QUARTER_NOTE),
    QUARTER_EIGHTH_EIGHTH_RHYTHM:                        (QUARTER_NOTE, EIGHTH, EIGHTH),
    SIXTEENTH_SIXTEENTH_SIXTEENTH_SIXTEENTH_QUARTER_RHYTHM: (SIXTEENTH, SIXTEENTH, SIXTEENTH, SIXTEENTH, QUARTER_NOTE),
    SIXTEENTH_DOTTEDEIGHTH_RHYTHM:                       (SIXTEENTH, DOTTED_EIGHTH),
    SIXTEENTH_DOTTEDEIGHTH_QUARTER_RHYTHM:               (SIXTEENTH, DOTTED_EIGHTH, QUARTER_NOTE),
}

# ── Sample role labels for with_roles() ─────────────────────────────────────
from enum import IntEnum

class SampleRole(IntEnum):
    """Per-beat role for multi-sample rhythm patterns.

    SAME  — this beat reuses the sample drawn for the previous SAME/FIRST beat
            that shares the same contiguous block.  Use to repeat beat 1's
            sample at beat 3 in an A/B/A pattern.
    NEW   — draw a fresh sample for this beat (excluded from the previous one).
    """
    SAME = 0   # reuse: same sample as the most recently established 'reference'
    NEW  = 1   # draw: a different sample from any beat already drawn this slot


# ── Musical pattern dict keys ─────────────────────────────────────────────────
MUSICAL_DURATION  = 'musical_duration'
POSSIBLE_PANNINGS = 'possible_pannings'
SAMPLE_ROLE       = 'sample_role'
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
BEAT_NAME_QUARTER_NOTE  = 'quarter'
BEAT_NAME_EIGHTH        = 'eighth'
BEAT_NAME_SIXTEENTH     = 'sixteenth'
BEAT_NAME_DOTTED_EIGHTH = 'dottedeighth'

BEAT_NAMES: dict[float, str] = {
    QUARTER_NOTE:    BEAT_NAME_QUARTER_NOTE,
    EIGHTH:          BEAT_NAME_EIGHTH,
    SIXTEENTH:       BEAT_NAME_SIXTEENTH,
    DOTTED_EIGHTH:   BEAT_NAME_DOTTED_EIGHTH,
}

# ── Operational limits ────────────────────────────────────────────────────────
MAX_DRAW_RETRIES = 200


