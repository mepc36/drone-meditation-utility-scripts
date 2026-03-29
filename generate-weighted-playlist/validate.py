from lib import constants, sound_rules, deck_builder, audio_processing
print('All imports OK')
print('HARD_CENTER:', constants.HARD_CENTER, type(constants.HARD_CENTER).__name__)
print('HARD_LEFT:', constants.HARD_LEFT)
print('HARD_RIGHT:', constants.HARD_RIGHT)
print('DIAGONAL_LEFT:', constants.DIAGONAL_LEFT)
print('DIAGONAL_RIGHT:', constants.DIAGONAL_RIGHT)
print('DUALPAN:', constants.DUALPAN, type(constants.DUALPAN).__name__)
print('UNTOUCHED:', constants.UNTOUCHED)
print('LOUD:', constants.LOUD, type(constants.LOUD).__name__)
print('QUIET:', constants.QUIET, type(constants.QUIET).__name__)
print('SLOW:', constants.SLOW, type(constants.SLOW).__name__)
print('FAST:', constants.FAST, type(constants.FAST).__name__)

import lib.sound_rules as sr
from lib.constants import (
    HARD_CENTER, HARD_LEFT, DIAGONAL_LEFT, DIAGONAL_RIGHT,
    MUSICAL_PATTERNS, KICKSNARE, STAB,
)
rule = sr.rules_by_sound_type['kick']
print('kick rule panning keys:', list(rule[MUSICAL_PATTERNS].keys()))
print('center vol rule (lookup by float 0.0):', rule[MUSICAL_PATTERNS].get(HARD_CENTER))
print('panning_compat kicksnare:', sr.panning_compat[KICKSNARE])
print('panning_compat stab:', sr.panning_compat[STAB])

# Verify diagonal_left and diagonal_right are in stab compat
stab_compat = sr.panning_compat[STAB]
print('DIAGONAL_LEFT in stab:', DIAGONAL_LEFT in stab_compat)
print('DIAGONAL_RIGHT in stab:', DIAGONAL_RIGHT in stab_compat)
