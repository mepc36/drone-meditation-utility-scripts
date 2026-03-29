"""Analyze rhythmicized output vs config targets and plot one chart per musical param."""
import os, re, json
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg_path        = os.path.join(BASE, "input/config/config.json")
rhythmicized_dir = os.path.join(BASE, "output/rhythmicized-audio")

with open(cfg_path) as f:
    cfg = json.load(f)

def parse_pct(s):
    return [int(x) for x in str(s).split(":")]

pan_pcts   = parse_pct(cfg["center_diagonal_dualpan_left_right_percents"])
pan_target = pan_pcts[:3] + [pan_pcts[3] + pan_pcts[4]]   # combine left+right → leftorright
ks_target  = parse_pct(cfg["kicksnare_stab_acappella_percents"])
vol_target = parse_pct(cfg["loud_quiet_percents"])
bpms       = [int(x) for x in str(cfg["bpms"]).split(":")]
bpm_target = parse_pct(cfg.get("slow_to_fast_bpm_percents", "100"))
rp_weights = cfg.get("rhythm_pattern_weights", {})
rp_total   = sum(rp_weights.values()) or 1

SUFFIX_MAP = {
    "quarter-quarternoterest": "single_and_rest",
    "quarter-quarter":          "double",
    "quarter":                  "single",
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
    return parts[2].split(".")[0].lower() == "strings"


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

for fname in wav:
    # panning
    for p in ("center", "diagonal", "dualpan", "leftorright"):
        if f"_{p}_" in fname:
            pan_counts[p] += 1
            break

    # sound group
    if re.search(r'_(kickstab|snarestab)\.\d+_', fname):
        grp_counts['stab'] += 1
    elif re.search(r'_acapp?ela?_', fname):
        grp_counts['acappella'] += 1
    else:
        grp_counts['kicksnare'] += 1

    # volume
    m = re.search(r'_vol(-?\d+)_', fname)
    if m:
        vol_counts['loud' if int(m.group(1)) == 0 else 'quiet'] += 1

    # bpm
    m = re.search(r'_bpm-(\d+)_', fname)
    if m:
        bpm_counts[int(m.group(1))] += 1

    # rhythm pattern suffix
    m = re.search(r'_bpm-\d+_[\w]+_(.+?)\.wav$', fname)
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
        "labels": ["center", "diagonal", "dualpan", "leftorright"],
        "targets": pan_target,
        "actuals": [pan_counts[k] for k in ("center", "diagonal", "dualpan", "leftorright")],
    },
    {
        "title": "Sound Group",
        "labels": ["kicksnare", "stab", "acappella"],
        "targets": ks_target,
        "actuals": [grp_counts[k] for k in ("kicksnare", "stab", "acappella")],
    },
    {
        "title": "Volume",
        "labels": ["loud", "quiet"],
        "targets": vol_target,
        "actuals": [vol_counts["loud"], vol_counts["quiet"]],
    },
    {
        "title": "BPM",
        "labels": [str(b) for b in bpms],
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
def status(delta_pct):
    a = abs(delta_pct)
    if a <= 3:   return "✅"
    if a <= 10:  return "⚠️"
    return "❌"

rows = []
_n_target = int(cfg.get("num_unique_samples", 0))
_file_delta = N_all - _n_target
rows.append(("Files (total)", str(_n_target), str(N_all),
             f"{_file_delta:+d}", "✅" if _file_delta == 0 else "❌"))
rows.append(("  of which strings", "—", str(N_all - N), "—", "ℹ️"))

for dim in DIMS:
    total = N if dim["title"] != "Rhythm Pattern" else sum(dim["actuals"])
    for lbl, tgt, act in zip(dim["labels"], dim["targets"], dim["actuals"]):
        pct   = act / total * 100 if total else 0.0
        delta = pct - tgt
        dsym  = f"~0%" if abs(delta) < 0.5 else f"{delta:+.1f}%"
        rows.append((f"{dim['title']}: {lbl}", f"{tgt:.1f}%", f"{pct:.1f}%", dsym, status(delta)))

col_w  = [max(len(r[i]) for r in rows) for i in range(5)]
header = ("Dimension", "Target", "Actual", "Delta", "Status")
col_w  = [max(col_w[i], len(header[i])) for i in range(5)]
sep    = "  ".join("─" * w for w in col_w)

print()
print("  ".join(h.ljust(col_w[i]) for i, h in enumerate(header)))
print(sep)
for row in rows:
    print("  ".join(cell.ljust(col_w[i]) for i, cell in enumerate(row)))
print()

# ── Charts  (one figure with one subplot per musical param) ───────────────────
fig, axes = plt.subplots(1, len(DIMS), figsize=(5 * len(DIMS), 6))
fig.suptitle(
    f"Rhythmicized Output vs Config Targets  (N={N})",
    fontsize=14, fontweight="bold", y=1.01,
)

COLOR_TARGET = "#4472C4"
COLOR_ACTUAL = "#ED7D31"
COLOR_POS    = "#70AD47"   # delta bar positive  (actual > target)
COLOR_NEG    = "#FF0000"   # delta bar negative  (actual < target)

for ax, dim in zip(axes, DIMS):
    labels  = dim["labels"]
    targets = dim["targets"]
    total   = N if dim["title"] != "Rhythm Pattern" else sum(dim["actuals"])
    actuals_pct = [a / total * 100 if total else 0.0 for a in dim["actuals"]]
    deltas      = [a - t for a, t in zip(actuals_pct, targets)]

    x = np.arange(len(labels))
    w = 0.25

    bars_t = ax.bar(x - w,     targets,       w, label="Target", color=COLOR_TARGET, alpha=0.85)
    bars_a = ax.bar(x,         actuals_pct,   w, label="Actual", color=COLOR_ACTUAL, alpha=0.85)
    bars_d = ax.bar(x + w,     deltas,        w, label="Delta",
                    color=[COLOR_POS if d >= 0 else COLOR_NEG for d in deltas], alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(dim["title"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("%" )
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate delta bars with the numeric value
    for bar, d in zip(bars_d, deltas):
        vert = bar.get_height()
        ax_y = vert + 0.4 if vert >= 0 else vert - 1.2
        ax.text(bar.get_x() + bar.get_width() / 2, ax_y,
                f"{d:+.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

patch_t = mpatches.Patch(color=COLOR_TARGET, alpha=0.85, label="Target")
patch_a = mpatches.Patch(color=COLOR_ACTUAL, alpha=0.85, label="Actual")
patch_pos = mpatches.Patch(color=COLOR_POS,  alpha=0.85, label="Delta (+)")
patch_neg = mpatches.Patch(color=COLOR_NEG,  alpha=0.85, label="Delta (−)")
fig.legend(handles=[patch_t, patch_a, patch_pos, patch_neg],
           loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.06))

plt.tight_layout()
out_path = os.path.join(BASE, "output/rhythmicized-ratios.png")
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"Chart saved → {out_path}")
