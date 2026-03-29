"""Analyze output/audio files vs config targets and print a delta chart."""
import os, re, json, sys
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg_path  = os.path.join(BASE, "input/config/config.json")
audio_dir = os.path.join(BASE, "output/audio")
rhythmicized_dir = os.path.join(BASE, "output/rhythmicized-audio")

with open(cfg_path) as f:
    cfg = json.load(f)

def parse_pct(s):
    return [int(x) for x in str(s).split(":")]

pan_pcts = parse_pct(cfg["center_diagonal_dualpan_left_right_percents"])   # center:diag:dual:left:right
pan_target = pan_pcts[:3] + [pan_pcts[3] + pan_pcts[4]]                    # combine left+right → leftorright
ks_target  = parse_pct(cfg["kicksnare_stab_acappella_percents"])               # kicksnare:stab:acap
vol_target = parse_pct(cfg["loud_quiet_percents"])                        # loud:quiet
bpms       = [int(x) for x in str(cfg["bpms"]).split(":")]
bpm_target = parse_pct(cfg.get("slow_to_fast_bpm_percents", "100"))
n_target   = int(cfg["num_unique_samples"])

wav = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
N = len(wav)

# ── Panning ──────────────────────────────────────────────────────────
pan_counts = Counter()
for fname in wav:
    for p in ("center", "diagonal", "dualpan", "leftorright"):
        if f"_{p}_" in fname:
            pan_counts[p] += 1
            break

# ── Sound group ──────────────────────────────────────────────────────
grp_counts = Counter()
for fname in wav:
    if re.search(r'_(kickstab|snarestab)\.\d+_', fname):
        grp_counts['stab'] += 1
    elif re.search(r'_acapp?ella_', fname):
        grp_counts['acappella'] += 1
    else:
        grp_counts['kicksnare'] += 1

# ── Volume ────────────────────────────────────────────────────────────
vol_counts = Counter()
for fname in wav:
    m = re.search(r'_vol(-?\d+)_', fname)
    if m:
        vol_counts['loud' if int(m.group(1)) == 0 else 'quiet'] += 1
    else:
        vol_counts['unknown'] += 1

# ── BPM ───────────────────────────────────────────────────────────────
bpm_counts = Counter()
for fname in wav:
    m = re.search(r'_bpm-(\d+)_', fname)
    if m:
        bpm_counts[int(m.group(1))] += 1

# ══════════════════════════════════════════════════════════════════════
# Rendering
# ══════════════════════════════════════════════════════════════════════

def status(delta_pct):
    a = abs(delta_pct)
    if a <= 3:   return "✅"
    if a <= 10:  return "⚠️"
    return "❌"

rows = []

# Files (absolute, not percentage)
file_delta = N - n_target
rows.append(("Files", str(n_target), str(N), str(file_delta), "✅" if file_delta == 0 else "❌"))

# Panning
pan_labels = ["Panning: center", "Panning: diagonal", "Panning: dualpan", "Panning: leftorright"]
pan_actuals = [pan_counts[k] for k in ("center", "diagonal", "dualpan", "leftorright")]
for lbl, act, tgt in zip(pan_labels, pan_actuals, pan_target):
    pct = act / N * 100 if N else 0.0
    delta = pct - tgt
    dsym = f"~0" if abs(delta) < 0.5 else f"{delta:+.1f}%"
    rows.append((lbl, f"{tgt}%", f"{pct:.1f}%", dsym, status(delta)))

# Sound group
snd_labels = ["Sound: kicksnare", "Sound: stab", "Sound: acappella"]
snd_actuals = [grp_counts[k] for k in ("kicksnare", "stab", "acappella")]
for lbl, act, tgt in zip(snd_labels, snd_actuals, ks_target):
    pct = act / N * 100 if N else 0.0
    delta = pct - tgt
    dsym = f"~0" if abs(delta) < 0.5 else f"{delta:+.1f}%"
    rows.append((lbl, f"{tgt}%", f"{pct:.1f}%", dsym, status(delta)))

# Volume
vol_labels = ["Volume: loud", "Volume: quiet"]
vol_actuals = [vol_counts["loud"], vol_counts["quiet"]]
for lbl, act, tgt in zip(vol_labels, vol_actuals, vol_target):
    pct = act / N * 100 if N else 0.0
    delta = pct - tgt
    dsym = f"~0" if abs(delta) < 0.5 else f"{delta:+.1f}%"
    rows.append((lbl, f"{tgt}%", f"{pct:.1f}%", dsym, status(delta)))

# BPM
bpm_speed = ["slow", "fast"]
for i, (b, tgt) in enumerate(zip(bpms, bpm_target)):
    act = bpm_counts[b]
    pct = act / N * 100 if N else 0.0
    delta = pct - tgt
    dsym = f"~0" if abs(delta) < 0.5 else f"{delta:+.1f}%"
    rows.append((f"BPM: {b} ({bpm_speed[i]})", f"{tgt}%", f"{pct:.1f}%", dsym, status(delta)))

# Print table
col_w = [max(len(r[i]) for r in rows) for i in range(5)]
col_w[0] = max(col_w[0], len("Dimension"))
header = ("Dimension", "Target", "Actual", "Delta", "Status")
sep = "  ".join("─" * w for w in col_w)
print()
print("  ".join(h.ljust(col_w[i]) for i, h in enumerate(header)))
print(sep)
for row in rows:
    print("  ".join(cell.ljust(col_w[i]) for i, cell in enumerate(row)))

# ── Rhythmic patterns ─────────────────────────────────────────────────
# Load QUARTER_NOTE_RHYTHMIC_PATTERNS from sound_rules without side effects
sys.path.insert(0, BASE)
import importlib, unittest.mock
with unittest.mock.patch("builtins.print"):  # suppress the $$$ debug print
    sr = importlib.import_module("lib.sound_rules")

pool = sr.SIMPLE_RHYTHMIC_PATTERNS
pool_total = len(pool)

# Map pattern → filename suffix
def pattern_to_suffix(p):
    parts = []
    for beat in p:
        if beat == 1:
            parts.append("quarter")
        elif beat == 0:
            parts.append("quarternoterest")
        else:
            parts.append(str(beat))
    return "-".join(parts)

pool_counts = Counter(pattern_to_suffix(p) for p in pool)

# Count suffixes in rhythmicized output
rhy_wav = [f for f in os.listdir(rhythmicized_dir) if f.endswith(".wav")]
rhy_total = len(rhy_wav)

rhy_counts = Counter()
for fname in rhy_wav:
    m = re.search(r'_bpm-\d+_\w+_(.+)\.wav$', fname)
    if m:
        rhy_counts[m.group(1)] += 1

all_suffixes = sorted(set(list(pool_counts.keys()) + list(rhy_counts.keys())))

print(f"\n{'═'*70}")
print(f"  RHYTHMIC PATTERNS  (pool size: {pool_total}  |  rhythmicized files: {rhy_total})")
print(f"{'═'*70}")
print(f"  {'Pattern':<22}  {'Pool':>6}  {'Pool%':>6}  {'Output':>7}  {'Out%':>6}  {'Delta':>7}")
print(f"  {'-'*66}")
for suffix in all_suffixes:
    pool_n   = pool_counts.get(suffix, 0)
    pool_pct = pool_n / pool_total * 100 if pool_total else 0
    out_n    = rhy_counts.get(suffix, 0)
    out_pct  = out_n / rhy_total * 100 if rhy_total else 0
    delta    = out_pct - pool_pct
    dsym     = f"{delta:+.1f}%" if abs(delta) >= 0.5 else "   ~0%"
    print(f"  {suffix:<22}  {pool_n:>6}  {pool_pct:>5.1f}%  {out_n:>7}  {out_pct:>5.1f}%  {dsym:>7}")
print()
