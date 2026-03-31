"""Count input files by sound type, then count output file attributes."""
import json
import os
import re
import sys
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.constants import (
    STRINGS,
    SOUND_GROUP_NAMES, SOUND_GROUP_TYPES,
    PANNING_CENTER, PANNING_DIAGONAL, PANNING_DUALPAN, PANNING_LEFT_OR_RIGHT,
    HARD_CENTER, HARD_LEFT, HARD_RIGHT, DIAGONAL_LEFT, DIAGONAL_RIGHT,
    DUALPAN_LEFTRIGHT, DUALPAN_DIAGONAL, UNTOUCHED,
    LOUD, QUIET, SLOW,
    QUARTER_NOTE, QUARTER_NOTE_REST, BEAT_NAME_QUARTER_NOTE, BEAT_NAME_QUARTER_NOTE_REST,
    RHYTHM_PATTERN_SEQUENCES,
    SINGLE_RHYTHM, DOUBLE_RHYTHM,
    MUSICAL_PATTERNS, VOLUMES, BPMS, MUSIC_PATTERN_PERCENT,
    RHYTHM_PATTERNS, RHYTHM_PATTERN, RHYTHM_PERCENT,
)
from lib.sound_rules import rules_by_sound_type, derive_type, derive_panning_key

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg_path = os.path.join(BASE, "input/config/config.json")
input_audio_dir = os.path.join(BASE, "input/audio")
rhythmicized_dir = os.path.join(BASE, "output/rhythmicized-audio")

with open(cfg_path) as f:
    cfg = json.load(f)

cfg_sound_group_pcts = [int(x) for x in cfg.get('kicksnare_stab_acappella_percents', '').split(':')]
cfg_num_unique = cfg.get('num_unique_samples', 0)
_sil_ratio = [int(x) for x in cfg.get('samples_to_silence_percents', '100:0').split(':')]
_samp_pct, _sil_pct = _sil_ratio[0], (_sil_ratio[1] if len(_sil_ratio) > 1 else 0)
if _sil_pct > 0:
    _sil_frac = _sil_pct / _samp_pct
    cfg_num_silence = round(cfg_num_unique * _sil_frac / (1.0 + _sil_frac))
else:
    cfg_num_silence = 0

# ── Input file counts ─────────────────────────────────────────────────────────
input_wav = [f for f in os.listdir(input_audio_dir) if f.endswith(".wav")]

def sound_type_of(fname):
    stem = fname[:-4] if fname.endswith(".wav") else fname
    parts = stem.split("_")
    return parts[2].split(".")[0].lower() if len(parts) >= 3 else ""

input_type_counts = Counter(sound_type_of(f) for f in input_wav)

_type_to_group = {t: grp for grp in SOUND_GROUP_NAMES for t in SOUND_GROUP_TYPES[grp]}
input_group_counts = Counter()
for t, n in input_type_counts.items():
    grp = _type_to_group.get(t, "other")
    input_group_counts[grp] += n


# ── Output file counts ────────────────────────────────────────────────────────
def _is_strings_file(fname):
    stem = fname[:-4] if fname.endswith(".wav") else fname
    parts = stem.split("_")
    return len(parts) >= 3 and parts[2].split(".")[0].lower() == STRINGS

_BEAT_NAMES = {
    QUARTER_NOTE:      BEAT_NAME_QUARTER_NOTE,
    QUARTER_NOTE_REST: BEAT_NAME_QUARTER_NOTE_REST,
}
SUFFIX_MAP = {
    '-'.join(_BEAT_NAMES[b] for b in beats): name
    for name, beats in RHYTHM_PATTERN_SEQUENCES.items()
}

all_wav = [f for f in os.listdir(rhythmicized_dir) if f.endswith(".wav")]
silence_wav = [f for f in all_wav if f.startswith("silence_")]
strings_wav = [f for f in all_wav if _is_strings_file(f) and not f.startswith("silence_")]
wav = [f for f in all_wav if not _is_strings_file(f) and not f.startswith("silence_")]
N = len(wav)  # non-strings musical files
N_strings = len(strings_wav)
N_total = len(all_wav)

if len(all_wav) == 0:
    print("No .wav files found in", rhythmicized_dir)
    raise SystemExit(1)

pan_counts = Counter()
grp_counts = Counter()
vol_counts = Counter()
bpm_counts = Counter()
rhy_counts = Counter()

for fname in wav:
    for p in (PANNING_CENTER, PANNING_DIAGONAL, PANNING_DUALPAN, PANNING_LEFT_OR_RIGHT):
        if f"_{p}_" in fname:
            pan_counts[p] += 1
            break

    for grp in SOUND_GROUP_NAMES:
        types = sorted(SOUND_GROUP_TYPES[grp], key=len, reverse=True)
        pat = r'_(' + '|'.join(re.escape(t) for t in types) + r')[._]'
        if re.search(pat, fname):
            grp_counts[grp] += 1
            break
    else:
        grp_counts["uncategorized"] += 1

    m = re.search(r'_vol(-?\d+)_', fname)
    if m:
        vol_counts["loud" if int(m.group(1)) == 0 else "quiet"] += 1

    m = re.search(r'_bpm-([\d.]+)_', fname)
    if m:
        bpm_counts[float(m.group(1))] += 1

    m = re.search(r'_bpm-[\d.]+_[\w]+_(.+?)\.wav$', fname)
    if m:
        suffix = m.group(1)
        for pat_suffix, pat_name in SUFFIX_MAP.items():
            if suffix == pat_suffix:
                rhy_counts[pat_name] += 1
                break

bpms = sorted(bpm_counts.keys())

# ── Compute expected counts from sound rules + configured percents ────────────
_non_strings_N = N  # use actual N of non-strings musical files
group_n = [round(_non_strings_N * pct / 100) for pct in cfg_sound_group_pcts]

_cfg_bpms = sorted(float(x) for x in cfg.get('bpms', '').split(':'))
_slow_bpm = min(_cfg_bpms) if _cfg_bpms else None
_fast_bpm = max(_cfg_bpms) if _cfg_bpms else None
_single_bpm = _slow_bpm == _fast_bpm

expected_loud = 0.0
expected_quiet = 0.0
expected_bpm_raw: dict = {}
expected_pan_raw: dict = {HARD_CENTER: 0.0, HARD_LEFT: 0.0, HARD_RIGHT: 0.0,
                          DIAGONAL_LEFT: 0.0, DIAGONAL_RIGHT: 0.0,
                          DUALPAN_LEFTRIGHT: 0.0, DUALPAN_DIAGONAL: 0.0}
expected_rhy: dict = {k: 0.0 for k in RHYTHM_PATTERN_SEQUENCES}

for grp, grp_count in zip(SOUND_GROUP_NAMES, group_n):
    n_types = len(SOUND_GROUP_TYPES[grp])
    for sound_type in SOUND_GROUP_TYPES[grp]:
        rule = rules_by_sound_type.get(sound_type)
        if rule is None:
            continue
        type_n = grp_count / n_types
        for entry in rule[MUSICAL_PATTERNS]:
            rp = entry[RHYTHM_PATTERNS]
            if rp and rp[0] is UNTOUCHED:
                continue
            mp_frac = entry[MUSIC_PATTERN_PERCENT] / 100
            entry_n = type_n * mp_frac
            for v in entry[VOLUMES]:
                if v is not UNTOUCHED:
                    if v == LOUD:
                        expected_loud += entry_n
                    elif v == QUIET:
                        expected_quiet += entry_n
            for b in entry[BPMS]:
                if b is not UNTOUCHED:
                    bpm_val = _slow_bpm if (b == SLOW or _single_bpm) else _fast_bpm
                    if bpm_val is not None:
                        expected_bpm_raw[bpm_val] = expected_bpm_raw.get(bpm_val, 0.0) + entry_n
            pkey = derive_panning_key(entry)
            for rp_entry in rp:
                rp_frac = rp_entry[RHYTHM_PERCENT] / 100
                rp_n = entry_n * rp_frac
                expected_pan_raw[pkey] = expected_pan_raw.get(pkey, 0.0) + rp_n
                rhy_type = derive_type(rp_entry[RHYTHM_PATTERN])
                if rhy_type in expected_rhy:
                    expected_rhy[rhy_type] += rp_n

expected_pan = {
    PANNING_CENTER:        expected_pan_raw.get(HARD_CENTER, 0),
    PANNING_DIAGONAL:      expected_pan_raw.get(DIAGONAL_LEFT, 0) + expected_pan_raw.get(DIAGONAL_RIGHT, 0),
    PANNING_DUALPAN:       expected_pan_raw.get(DUALPAN_LEFTRIGHT, 0) + expected_pan_raw.get(DUALPAN_DIAGONAL, 0),
    PANNING_LEFT_OR_RIGHT: expected_pan_raw.get(HARD_LEFT, 0) + expected_pan_raw.get(HARD_RIGHT, 0),
}

# ── Print table ───────────────────────────────────────────────────────────────
_LOG_RHYTHMS = [SINGLE_RHYTHM, DOUBLE_RHYTHM]

def _status(actual, expected):
    if expected == 0:
        return "🟢" if actual == 0 else "🔴"
    pct_off = abs(actual - expected) / expected * 100
    if actual == expected:
        return "🟢"
    if pct_off <= 20:
        return "⚠️"
    return "🔴"

def _row(label, actual, expected=None):
    delta = (actual - expected) if expected is not None else None
    delta_str = (f"{delta:+d}" if delta != 0 else "  0") if delta is not None else "   "
    exp_str   = str(expected) if expected is not None else "  -"
    status    = f"  {_status(actual, expected)}" if expected is not None else ""
    return f"  {label:<18} {actual:>6}  {exp_str:>8}  {delta_str:>5}  {status}"

def _header():
    return f"  {'':18} {'Actual':>6}  {'Expected':>8}  {'Delta':>5}  {'Status:':<6}"

print()
print("INPUT FILES:")
for grp in SOUND_GROUP_NAMES:
    types = sorted(SOUND_GROUP_TYPES[grp])
    detail = "  ".join(f"{t}={input_type_counts[t]}" for t in types if input_type_counts.get(t, 0) > 0)
    print(f"  {grp:<18} {input_group_counts[grp]}  [{detail}]")
print(f"  {'total':<18} {sum(input_group_counts.values())}")

print()
print(f"OUTPUT FILES  (total={len(all_wav)}/{cfg_num_unique}  silence={len(silence_wav)}/{cfg_num_silence}  N_non-strings_non-silence={N}):")
print()

print("  SOUND GROUP:")
print(_header())
for grp, exp in zip(SOUND_GROUP_NAMES, group_n):
    print(_row(grp, grp_counts[grp], exp))
print(_row("strings", N_strings, input_type_counts.get(STRINGS, 0)))
print(_row("silence", len(silence_wav), cfg_num_silence))
unc = grp_counts["uncategorized"]
if unc:
    print(_row("uncategorized", unc))

print()
print("  PANNING:")
for p, exp in zip(
    (PANNING_CENTER, PANNING_DIAGONAL, PANNING_DUALPAN, PANNING_LEFT_OR_RIGHT),
    [round(expected_pan[p]) for p in (PANNING_CENTER, PANNING_DIAGONAL, PANNING_DUALPAN, PANNING_LEFT_OR_RIGHT)],
):
    print(_row(p, pan_counts[p], exp))

print()
print("  VOLUME:")
print(_row("loud",  vol_counts["loud"],  round(expected_loud)))
print(_row("quiet", vol_counts["quiet"], round(expected_quiet)))

print()
print("  BPM:")
for b in bpms:
    lbl = str(int(b)) if b == int(b) else str(b)
    print(_row(lbl, bpm_counts[b], round(expected_bpm_raw.get(b, 0))))

print()
print("  RHYTHM PATTERN:")
for pat in _LOG_RHYTHMS:
    print(_row(pat, rhy_counts.get(pat, 0), round(expected_rhy.get(pat, 0))))
print()

# ── Chart ─────────────────────────────────────────────────────────────────────
DIMS = [
    {
        "title": "Sound Group",
        "labels": list(SOUND_GROUP_NAMES) + ["strings", "silence"],
        "counts": [grp_counts[g] for g in SOUND_GROUP_NAMES] + [N_strings, len(silence_wav)],
        "expected": [round(N * pct / 100) for pct in cfg_sound_group_pcts] + [input_type_counts.get(STRINGS, 0), cfg_num_silence],
    },
    {
        "title": "Panning",
        "labels": [PANNING_CENTER, PANNING_DIAGONAL, PANNING_DUALPAN, PANNING_LEFT_OR_RIGHT],
        "counts": [pan_counts[p] for p in (PANNING_CENTER, PANNING_DIAGONAL, PANNING_DUALPAN, PANNING_LEFT_OR_RIGHT)],
        "expected": [round(expected_pan[p]) for p in (PANNING_CENTER, PANNING_DIAGONAL, PANNING_DUALPAN, PANNING_LEFT_OR_RIGHT)],
    },
    {
        "title": "Volume",
        "labels": ["loud", "quiet"],
        "counts": [vol_counts["loud"], vol_counts["quiet"]],
        "expected": [round(expected_loud), round(expected_quiet)],
    },
    {
        "title": "BPM",
        "labels": [str(int(b)) if b == int(b) else str(b) for b in bpms],
        "counts": [bpm_counts[b] for b in bpms],
        "expected": [round(expected_bpm_raw.get(b, 0)) for b in bpms],
    },
    {
        "title": "Rhythm Pattern",
        "labels": list(RHYTHM_PATTERN_SEQUENCES.keys()),
        "counts": [rhy_counts.get(k, 0) for k in RHYTHM_PATTERN_SEQUENCES],
        "expected": [round(expected_rhy.get(k, 0)) for k in RHYTHM_PATTERN_SEQUENCES],
    },
]

fig, axes = plt.subplots(1, len(DIMS), figsize=(4 * len(DIMS), 6))
fig.suptitle(f"Rhythmicized Output  (total={N_total}/{cfg_num_unique}  N_non-strings_non-silence={N})", fontsize=14, fontweight="bold", y=1.01)

COLOR = "#762A83"

for ax, dim in zip(axes, DIMS):
    labels = dim["labels"]
    counts = dim["counts"]
    x = np.arange(len(labels))
    bars = ax.bar(x, counts, color=COLOR, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(dim["title"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Files")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    expected_list = dim.get("expected", [])
    for i, (bar, v) in enumerate(zip(bars, counts)):
        exp = expected_list[i] if i < len(expected_list) else None
        label = f"{v}/{exp}" if exp is not None else str(v)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                label, ha="center", va="bottom", fontsize=8)

plt.tight_layout()
out_dir = os.path.join(BASE, "output/analyze-ratios")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "rhythmicized-ratios.png")
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"Chart saved → {out_path}")
