#!/usr/bin/env python3
"""
calculate-min-num-unique-samples.py

Calculates the minimum num_unique_samples needed to ensure every sample
in every sound group gets used at least once.

Formula:
  min = max(
    ceil(N_kicksnare / p_kicksnare),
    ceil(N_stab      / p_stab),
    ceil(N_acappella / p_acappella),
  ) + N_strings

Run from the project root:
  python utility-scripts/calculate-min-num-unique-samples.py
"""

import json
import math
from pathlib import Path

STRINGS = 'strings'

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "input" / "config" / "config.json"
AUDIO_DIR = ROOT / "input" / "audio"

with open(CONFIG_PATH) as f:
    config = json.load(f)

# --- Read percents from config ---
ks_percents_raw = config.get("kicksnare_stab_acappella_percents")
if ks_percents_raw is None:
    print("kicksnare_stab_acappella_percents not set in config — no group constraint applies.")
    print("Minimum num_unique_samples = total non-strings samples + strings samples.")
    ks_percents_raw = None

if ks_percents_raw is not None:
    parts = [int(x) for x in str(ks_percents_raw).split(":")]
    p_kicksnare, p_stab, p_acappella = parts[0] / 100, parts[1] / 100, parts[2] / 100

# --- Count samples by sound group ---
KICKSNARE_TYPES = {"kick", "snare"}
STAB_TYPES = {"kickstab", "snarestab"}

def get_sound_type(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) >= 3:
        raw = parts[2].split(".")[0].lower()
    else:
        raw = stem.split(".")[0].lower()
    if raw.startswith("acap"):
        return "acappella"
    return raw

n_kicksnare = n_stab = n_acappella = n_strings = 0

for wav in AUDIO_DIR.glob("*.wav"):
    st = get_sound_type(wav.stem)
    if st in KICKSNARE_TYPES:
        n_kicksnare += 1
    elif st in STAB_TYPES:
        n_stab += 1
    elif st == "acappella":
        n_acappella += 1
    elif st == STRINGS:
        n_strings += 1

print(f"Sample counts from {AUDIO_DIR.relative_to(ROOT)}:")
print(f"  kicksnare : {n_kicksnare}")
print(f"  stab      : {n_stab}")
print(f"  acappella : {n_acappella}")
print(f"  strings   : {n_strings}  (always used exactly once, added on top)")
print()

if ks_percents_raw is None:
    total = n_kicksnare + n_stab + n_acappella + n_strings
    print(f"Minimum num_unique_samples: {total}")
else:
    print(f"kicksnare_stab_acappella_percents: {ks_percents_raw}  ({p_kicksnare:.0%} / {p_stab:.0%} / {p_acappella:.0%})")
    print()

    constraints = []
    for label, count, pct in [
        ("kicksnare", n_kicksnare, p_kicksnare),
        ("stab",      n_stab,      p_stab),
        ("acappella", n_acappella, p_acappella),
    ]:
        if pct == 0:
            if count > 0:
                print(f"  WARNING: {count} {label} sample(s) exist but {label} percent is 0% — they can never be used.")
            needed = 0
        else:
            needed = math.ceil(count / pct)
        print(f"  ceil({count} / {pct:.0%}) = {needed}  [{label}]")
        constraints.append((needed, label))

    max_needed, binding_group = max(constraints)
    result = max_needed + n_strings

    print()
    print(f"Binding constraint: {binding_group}  (needs {max_needed} non-strings slots)")
    print(f"+ {n_strings} strings sample(s)")
    print()
    print(f"Minimum num_unique_samples: {result}")
    print()
    current = config.get("num_unique_samples")
    if current is not None:
        if current >= result:
            print(f"Current setting ({current}) is sufficient. ✓")
        else:
            print(f"Current setting ({current}) is TOO LOW — increase to at least {result}.")
