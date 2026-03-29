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
