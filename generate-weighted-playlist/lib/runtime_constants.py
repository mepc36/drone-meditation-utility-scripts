from . import config as _cfg

# ── Config-sourced constants ───────────────────────────────────────────────────
# Loaded at import time so that sound_rules (which imports this module) builds
# its panning/volume/bpm rule dictionaries with real numeric values.

_conf = _cfg.load()
LOUD:  float = max(_conf['volume_levels_db'])
QUIET: float = min(_conf['volume_levels_db'])
SLOW:  float = min(_conf['bpm_values'])
FAST:  float = max(_conf['bpm_values'])
