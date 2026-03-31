"""Count input files by sound type, then count output file attributes."""
import os, re, json, sys
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
    QUARTER_NOTE, QUARTER_NOTE_REST, BEAT_NAME_QUARTER_NOTE, BEAT_NAME_QUARTER_NOTE_REST,
    RHYTHM_PATTERN_SEQUENCES,
)
import lib.config as _cfg_mod

BASE             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg_path         = os.path.join(BASE, "input/config/config.json")
input_audio_dir  = os.path.join(BASE, "input/audio")
rhythmicized_dir = os.path.join(BASE, "output/rhythmicized-audio")

with open(cfg_path) as f:
    cfg = json.load(f)

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

all_wav     = [f for f in os.listdir(rhythmicized_dir) if f.endswith(".wav")]
silence_wav = [f for f in all_wav if f.startswith("silence_")]
wav         = [f for f in all_wav if not _is_strings_file(f) and not f.startswith("silence_")]
N           = len(wav)

if len(all_wav) == 0:
    print("No .wav files found in", rhythmicized_dir)
    raise SystemExit(1)

pan_counts = Counter()
grp_counts = Counter()
vol_counts = Counter()
bpm_counts = Counter()
rhy_counts = Counter()

rp_weights = cfg.get(_cfg_mod.CFG_RHYTHM_WEIGHTS, {})

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

# ── Print table ───────────────────────────────────────────────────────────────
def _row(label, count, total=None):
    pct = f"  ({count/total*100:.1f}%)" if total else ""
    return f"  {label:<16} {count}{pct}"

print()
print("INPUT FILES:")
for grp in SOUND_GROUP_NAMES:
    types = sorted(SOUND_GROUP_TYPES[grp])
    detail = "  ".join(f"{t}={input_type_counts[t]}" for t in types if input_type_counts.get(t, 0) > 0)
    print(f"  {grp:<16} {input_group_counts[grp]}  [{detail}]")
print(f"  {'total':<16} {sum(input_group_counts.values())}")

print()
print(f"OUTPUT FILES  (total={len(all_wav)}  silence={len(silence_wav)}  musical={N}):")
print()
print("  PANNING:")
for p in (PANNING_CENTER, PANNING_DIAGONAL, PANNING_DUALPAN, PANNING_LEFT_OR_RIGHT):
    print(_row(p, pan_counts[p], N))

print()
print("  SOUND GROUP:")
for grp in SOUND_GROUP_NAMES:
    print(_row(grp, grp_counts[grp], N))
unc = grp_counts["uncategorized"]
if unc:
    print(_row("uncategorized", unc))

print()
print("  VOLUME:")
for lbl in ("loud", "quiet"):
    print(_row(lbl, vol_counts[lbl], N))

print()
print("  BPM:")
for b in bpms:
    lbl = str(int(b)) if b == int(b) else str(b)
    print(_row(lbl, bpm_counts[b], N))

print()
print("  RHYTHM PATTERN:")
for pat in rp_weights:
    print(_row(pat, rhy_counts.get(pat, 0), N))
print()

# ── Chart ─────────────────────────────────────────────────────────────────────
DIMS = [
    {
        "title": "Panning",
        "labels": [PANNING_CENTER, PANNING_DIAGONAL, PANNING_DUALPAN, PANNING_LEFT_OR_RIGHT],
        "counts": [pan_counts[p] for p in (PANNING_CENTER, PANNING_DIAGONAL, PANNING_DUALPAN, PANNING_LEFT_OR_RIGHT)],
    },
    {
        "title": "Sound Group",
        "labels": SOUND_GROUP_NAMES,
        "counts": [grp_counts[g] for g in SOUND_GROUP_NAMES],
    },
    {
        "title": "Volume",
        "labels": ["loud", "quiet"],
        "counts": [vol_counts["loud"], vol_counts["quiet"]],
    },
    {
        "title": "BPM",
        "labels": [str(int(b)) if b == int(b) else str(b) for b in bpms],
        "counts": [bpm_counts[b] for b in bpms],
    },
    {
        "title": "Rhythm Pattern",
        "labels": list(rp_weights.keys()),
        "counts": [rhy_counts.get(k, 0) for k in rp_weights],
    },
]

fig, axes = plt.subplots(1, len(DIMS), figsize=(4 * len(DIMS), 6))
fig.suptitle(f"Rhythmicized Output  (N={N})", fontsize=14, fontweight="bold", y=1.01)

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
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                str(v), ha="center", va="bottom", fontsize=8)

plt.tight_layout()
out_dir = os.path.join(BASE, "output/analyze-ratios")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "rhythmicized-ratios.png")
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"Chart saved → {out_path}")
