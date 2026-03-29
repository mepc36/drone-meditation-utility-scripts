from . import config as _cfg

# Rhythmic durations (in quarter notes)
QUARTER_NOTE = 1
QUARTER_NOTE_REST = 0

# Panning positions (numeric: -1.0 = full left, 0.0 = center, 1.0 = full right)
HARD_CENTER  =  0.0
HARD_LEFT    = -1.0
HARD_RIGHT   =  1.0
DIAGONAL_LEFT  = -0.65
DIAGONAL_RIGHT =  0.65

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

_conf = _cfg.load()
LOUD:  float = max(_conf['volume_levels_db'])
QUIET: float = min(_conf['volume_levels_db'])
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

# ── Beat name strings for output filenames ────────────────────────────────────
BEAT_NAME_QUARTER_NOTE_REST = 'quarternoterest'
BEAT_NAME_QUARTER_NOTE      = 'quarter'

# ── Operational limits ────────────────────────────────────────────────────────
MAX_DRAW_RETRIES = 20
