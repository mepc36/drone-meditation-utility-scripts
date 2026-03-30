"""Analyze rhythmicized output vs config targets and plot one chart per musical param."""
import os, re, json, sys
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.constants import (
    KICKSNARE, STAB, ACAPPELLA, STRINGS,
    PANNING_CENTER, PANNING_DIAGONAL, PANNING_DUALPAN, PANNING_LEFT_OR_RIGHT,
    SINGLE_RHYTHM, DOUBLE_RHYTHM, SINGLE_REST_RHYTHM, SINGLE_REST_SINGLE_RHYTHM,
    BEAT_NAME_QUARTER_NOTE, BEAT_NAME_QUARTER_NOTE_REST,
)
import lib.config as _cfg_mod

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg_path        = os.path.join(BASE, "input/config/config.json")
rhythmicized_dir = os.path.join(BASE, "output/rhythmicized-audio")

with open(cfg_path) as f:
    cfg = json.load(f)

def parse_pct(s):
    return [int(x) for x in str(s).split(":")]

pan_pcts   = parse_pct(cfg[_cfg_mod.CFG_PANNING_PERCENTS])
pan_target = pan_pcts[:3] + [pan_pcts[3] + pan_pcts[4]]   # combine left+right → leftorright
ks_target  = parse_pct(cfg[_cfg_mod.CFG_SOUND_GROUP_PERCENTS])
vol_target = parse_pct(cfg[_cfg_mod.CFG_LOUD_QUIET_PERCENTS])
bpms       = [float(x) for x in str(cfg[_cfg_mod.CFG_BPMS]).split(":")]
bpm_target = parse_pct(cfg.get(_cfg_mod.CFG_BPM_PERCENTS, "100"))
rp_weights = cfg.get(_cfg_mod.CFG_RHYTHM_WEIGHTS, {})
rp_total   = sum(rp_weights.values()) or 1

SUFFIX_MAP = {
    f"{BEAT_NAME_QUARTER_NOTE}-{BEAT_NAME_QUARTER_NOTE_REST}-{BEAT_NAME_QUARTER_NOTE}": SINGLE_REST_SINGLE_RHYTHM,
    f"{BEAT_NAME_QUARTER_NOTE}-{BEAT_NAME_QUARTER_NOTE_REST}": SINGLE_REST_RHYTHM,
    f"{BEAT_NAME_QUARTER_NOTE}-{BEAT_NAME_QUARTER_NOTE}": DOUBLE_RHYTHM,
    BEAT_NAME_QUARTER_NOTE: SINGLE_RHYTHM,
}

# ── Parse filenames ───────────────────────────────────────────────────────────
def _is_strings_file(fname: str) -> bool:
    """True when the output filename comes from a strings-type input sample.

    Output filenames follow the pattern:
      {song}_{section}_{sound_type}[.N]_vol-…
    We check the third underscore-separated token (same logic as sound_type_of).
    Strings pass through unmodified and must not count against panning / volume /
    sound-group / BPM quotas.
    """
    stem = fname[:-4] if fname.endswith(".wav") else fname
    parts = stem.split("_")
    if len(parts) < 3:
        return False
    return parts[2].split(".")[0].lower() == STRINGS


all_wav = [f for f in os.listdir(rhythmicized_dir) if f.endswith(".wav")]
N_all = len(all_wav)          # total including strings (for Files / samples_to_silence check)
wav   = [f for f in all_wav if not _is_strings_file(f)]  # non-strings only
N     = len(wav)              # denominator for panning / volume / sound-group / BPM
if N_all == 0:
    print("No .wav files found in", rhythmicized_dir)
    raise SystemExit(1)

pan_counts = Counter()
grp_counts = Counter()
vol_counts = Counter()
bpm_counts = Counter()
rhy_counts = Counter()
uncategorized_files = []

for fname in wav:
    # panning
    for p in (PANNING_CENTER, PANNING_DIAGONAL, PANNING_DUALPAN, PANNING_LEFT_OR_RIGHT):
        if f"_{p}_" in fname:
            pan_counts[p] += 1
            break

    # sound group
    if re.search(r'_(kickstab|snarestab)\.', fname):
        grp_counts[STAB] += 1
    elif re.search(r'_acappella_', fname):
        grp_counts[ACAPPELLA] += 1
    elif re.search(r'_(kick|snare)[._]', fname):
        grp_counts[KICKSNARE] += 1
    else:
        grp_counts['uncategorized'] += 1
        uncategorized_files.append(fname)

    # volume
    m = re.search(r'_vol(-?\d+)_', fname)
    if m:
        vol_counts['loud' if int(m.group(1)) == 0 else 'quiet'] += 1

    # bpm
    m = re.search(r'_bpm-([\d.]+)_', fname)
    if m:
        bpm_counts[float(m.group(1))] += 1

    # rhythm pattern suffix
    m = re.search(r'_bpm-[\d.]+_[\w]+_(.+?)\.wav$', fname)
    if m:
        suffix = m.group(1)
        for pat_suffix, pat_name in SUFFIX_MAP.items():
            if suffix == pat_suffix:
                rhy_counts[pat_name] += 1
                break

# ── Assemble dimensions ───────────────────────────────────────────────────────
DIMS = [
    {
        "title": "Panning",
        "labels": [PANNING_CENTER, PANNING_DIAGONAL, PANNING_DUALPAN, PANNING_LEFT_OR_RIGHT],
        "targets": pan_target,
        "actuals": [pan_counts[k] for k in (PANNING_CENTER, PANNING_DIAGONAL, PANNING_DUALPAN, PANNING_LEFT_OR_RIGHT)],
    },
    {
        "title": "Sound Group",
        "labels": [KICKSNARE, STAB, ACAPPELLA],
        "targets": ks_target,
        "actuals": [grp_counts[k] for k in (KICKSNARE, STAB, ACAPPELLA)],
    },

    {
        "title": "Volume",
        "labels": ["loud", "quiet"],
        "targets": vol_target,
        "actuals": [vol_counts["loud"], vol_counts["quiet"]],
    },
    {
        "title": "BPM",
        "labels": [str(int(b)) if b == int(b) else str(b) for b in bpms],
        "targets": bpm_target,
        "actuals": [bpm_counts[b] for b in bpms],
    },
    {
        "title": "Rhythm Pattern",
        "labels": list(rp_weights.keys()),
        "targets": [rp_weights[k] / rp_total * 100 for k in rp_weights],
        "actuals": [rhy_counts.get(k, 0) for k in rp_weights],
    },
]

# ── Text table ────────────────────────────────────────────────────────────────
# NOTE: all DIMS use only non-strings files (N) as their denominator.
def status(delta, total):
    pct = abs(delta) / total * 100 if total else 0
    if pct <= 3:   return "✅"
    if pct <= 10:  return "⚠️"
    return "❌"

# ── Collect all data rows to calculate column widths ─────────────────────────
_n_target = int(cfg.get("num_unique_samples", 0))
_file_delta = N_all - _n_target
_uncategorized = grp_counts["uncategorized"]

# Each entry is either a section header (str) or a data row (tuple of 4: lbl, exp, act, diff, status)
_sections = []

# Files block
_sections.append(("FILES", [
    ("total",       str(_n_target),  str(N_all),     f"{_file_delta:+d}", "✅" if _file_delta == 0 else "❌"),
    ("  strings",   "—",             str(N_all - N), "—",                 "ℹ️"),
]))

for dim in DIMS:
    total = N if dim["title"] != "Rhythm Pattern" else sum(dim["actuals"])
    dim_rows = []
    for lbl, tgt, act in zip(dim["labels"], dim["targets"], dim["actuals"]):
        tgt_count = round(tgt / 100 * total) if total else 0
        delta     = act - tgt_count
        dsym      = "~0" if delta == 0 else f"{delta:+d}"
        dim_rows.append((lbl, str(tgt_count), str(act), dsym, status(delta, total)))
    _sections.append((dim["title"].upper(), dim_rows))

# Uncategorized files row
_sections.append(("UNCATEGORIZED", [
    ("files", "0", str(_uncategorized), f"{_uncategorized:+d}" if _uncategorized else "~0", "✅" if _uncategorized == 0 else "❌"),
]))

# Calculate column widths across all data rows
all_rows = [r for _, rows in _sections for r in rows]
col_w  = [max(len(r[i]) for r in all_rows) for i in range(5)]
header = ("", "Expected", "Actual", "Diff", "Status")
col_w  = [max(col_w[i], len(header[i])) for i in range(5)]
sep    = "  ".join("─" * w for w in col_w)

print()
print("  ".join(h.ljust(col_w[i]) for i, h in enumerate(header)))
print(sep)
for section_title, section_rows in _sections:
    print(f"\n{section_title}:")
    for row in section_rows:
        print("  ".join(cell.ljust(col_w[i]) for i, cell in enumerate(row)))
print()

if uncategorized_files:
    print(f"Uncategorized files:")
    for f in sorted(uncategorized_files):
        print(f"  {f}")
    print()

# ── Charts  (one figure with one subplot per musical param) ───────────────────
fig, axes = plt.subplots(1, len(DIMS), figsize=(5 * len(DIMS), 6))
fig.suptitle(
    f"Rhythmicized Output vs Config Targets  (N={N})",
    fontsize=14, fontweight="bold", y=1.01,
)

COLOR_TARGET = "#1B7837"   # dark green  – Expected
COLOR_ACTUAL = "#762A83"   # deep purple – Actual

for ax, dim in zip(axes, DIMS):
    labels  = dim["labels"]
    total   = N if dim["title"] != "Rhythm Pattern" else sum(dim["actuals"])
    targets_count = [round(t / 100 * total) if total else 0 for t in dim["targets"]]
    actuals_count = dim["actuals"]

    x = np.arange(len(labels))
    w = 0.35

    bars_t = ax.bar(x - w/2, targets_count, w, label="Expected", color=COLOR_TARGET, alpha=0.85)
    bars_a = ax.bar(x + w/2, actuals_count, w, label="Actual",   color=COLOR_ACTUAL, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(dim["title"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Files")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate all bars with raw counts
    for bar, v in zip(bars_t, targets_count):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                str(v), ha="center", va="bottom", fontsize=7.5)
    for bar, v in zip(bars_a, actuals_count):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                str(v), ha="center", va="bottom", fontsize=7.5)

patch_t = mpatches.Patch(color=COLOR_TARGET, alpha=0.85, label="Expected")
patch_a = mpatches.Patch(color=COLOR_ACTUAL, alpha=0.85, label="Actual")
fig.legend(handles=[patch_t, patch_a],
           loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.06))

plt.tight_layout()
out_dir = os.path.join(BASE, "output/analyze-ratios")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "rhythmicized-ratios.png")
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"Chart saved → {out_path}")
