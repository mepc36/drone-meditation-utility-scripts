from . import config as _cfg

# Rhythmic durations (in quarter notes)
QUARTER_NOTE = 1
QUARTER_NOTE_REST = 0

# Diagonal stereo offset (0 = center, 1.0 = full hard pan)
DIAGONAL_PAN_OFFSET = 0.65

# Panning positions (numeric: -1.0 = full left, 0.0 = center, 1.0 = full right)
HARD_CENTER  =  0.0
HARD_LEFT    = -1.0
HARD_RIGHT   =  1.0
DIAGONAL_LEFT  = DIAGONAL_PAN_OFFSET * -1
DIAGONAL_RIGHT =  DIAGONAL_PAN_OFFSET

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
DUALPAN   = 2.0   # sentinel: two samples panned hard left + hard right simultaneously
UNTOUCHED = None  # sentinel: pass audio through without any panning or processing

# ── Config-sourced constants ───────────────────────────────────────────────────
# Loaded at import time so that sound_rules (which imports this module) builds
# its panning/volume/bpm rule dictionaries with real numeric values.

def _get_loud_db(volume_levels_db: list[float]) -> float:
    """Loudest volume = max dB value (least negative, e.g. 0 dB)."""
    return max(volume_levels_db)

def _get_quiet_db(volume_levels_db: list[float]) -> float:
    """Quietest volume = min dB value (most negative, e.g. -26 dB)."""
    return min(volume_levels_db)

_conf = _cfg.load()
LOUD:  float = _get_loud_db(_conf['volume_levels_db'])
QUIET: float = _get_quiet_db(_conf['volume_levels_db'])
SLOW:  int   = min(_conf['bpm_values'])
FAST:  int   = max(_conf['bpm_values'])

# ── Rhythm pattern type identifiers ──────────────────────────────────────────
SINGLE_RHYTHM          = 'single'
DOUBLE_RHYTHM          = 'double'
SINGLE_AND_REST_RHYTHM = 'single_and_rest'

# ── Sound type detection ───────────────────────────────────────────────────────
ACAPPELLA_PREFIX = 'acap'

# ── Musical pattern dict keys ─────────────────────────────────────────────────
MUSICAL_DURATION  = 'musical_duration'
POSSIBLE_PANNINGS = 'possible_pannings'
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

# ── Fractional note durations in beats ────────────────────────────────────────
SIXTEENTH_NOTE = 0.25
EIGHTH_NOTE    = 0.5
HALF_NOTE      = 2.0

# ── Beat name strings for output filenames ────────────────────────────────────
BEAT_NAME_QUARTER_NOTE_REST = 'quarternoterest'
BEAT_NAME_SIXTEENTH         = 'sixteenth'
BEAT_NAME_EIGHTH            = 'eighth'
BEAT_NAME_QUARTER_NOTE      = 'quarter'
BEAT_NAME_HALF_NOTE         = 'half'

# ── Operational limits ────────────────────────────────────────────────────────
MAX_DRAW_RETRIES = 20
