"""Analyze output/audio files vs config targets and print a delta chart."""
import os, re, json, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.constants import (
    KICKSNARE, STAB, ACAPPELLA,
    PANNING_CENTER, PANNING_DIAGONAL, PANNING_DUALPAN, PANNING_LEFT_OR_RIGHT,
    SINGLE_RHYTHM, DOUBLE_RHYTHM, SINGLE_AND_REST_RHYTHM,
    BEAT_NAME_QUARTER_NOTE, BEAT_NAME_QUARTER_NOTE_REST,
)
import lib.config as _cfg_mod

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg_path  = os.path.join(BASE, "input/config/config.json")
audio_dir = os.path.join(BASE, "output/audio")
rhythmicized_dir = os.path.join(BASE, "output/rhythmicized-audio")

with open(cfg_path) as f:
    cfg = json.load(f)

def parse_pct(s):
    return [int(x) for x in str(s).split(":")]

pan_pcts = parse_pct(cfg[_cfg_mod.CFG_PANNING_PERCENTS])   # center:diag:dual:left:right
pan_target = pan_pcts[:3] + [pan_pcts[3] + pan_pcts[4]]    # combine left+right → leftorright
ks_target  = parse_pct(cfg[_cfg_mod.CFG_SOUND_GROUP_PERCENTS])  # kicksnare:stab:acap
vol_target = parse_pct(cfg[_cfg_mod.CFG_LOUD_QUIET_PERCENTS])  # loud:quiet
bpms       = [int(x) for x in str(cfg[_cfg_mod.CFG_BPMS]).split(":")]
bpm_target = parse_pct(cfg.get(_cfg_mod.CFG_BPM_PERCENTS, "100"))
n_target   = int(cfg[_cfg_mod.CFG_NUM_UNIQUE_SAMPLES])

wav = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
N = len(wav)

# ── Panning ──────────────────────────────────────────────────────────
pan_counts = Counter()
for fname in wav:
    for p in (PANNING_CENTER, PANNING_DIAGONAL, PANNING_DUALPAN, PANNING_LEFT_OR_RIGHT):
        if f"_{p}_" in fname:
            pan_counts[p] += 1
            break

# ── Sound group ──────────────────────────────────────────────────────
grp_counts = Counter()
for fname in wav:
    if re.search(r'_(kickstab|snarestab)\.',  fname):
        grp_counts[STAB] += 1
    elif re.search(r'_acapp?ella_', fname):
        grp_counts[ACAPPELLA] += 1
    else:
        grp_counts[KICKSNARE] += 1

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
pan_actuals = [pan_counts[k] for k in (PANNING_CENTER, PANNING_DIAGONAL, PANNING_DUALPAN, PANNING_LEFT_OR_RIGHT)]
for lbl, act, tgt in zip(pan_labels, pan_actuals, pan_target):
    pct = act / N * 100 if N else 0.0
    delta = pct - tgt
    dsym = f"~0" if abs(delta) < 0.5 else f"{delta:+.1f}%"
    rows.append((lbl, f"{tgt}%", f"{pct:.1f}%", dsym, status(delta)))

# Sound group
snd_labels = ["Sound: kicksnare", "Sound: stab", "Sound: acappella"]
snd_actuals = [grp_counts[k] for k in (KICKSNARE, STAB, ACAPPELLA)]
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
# Targets come from rhythm_pattern_weights in config
rp_weights = cfg.get("rhythm_pattern_weights", {})
rp_total_weight = sum(rp_weights.values()) if rp_weights else 0

# Suffix patterns in filenames:
#   single           → ends with _<panning>_quarter.wav
#   double           → ends with _<panning>_quarter-quarter.wav
#   single_and_rest  → ends with _<panning>_quarter-quarternoterest.wav
rhy_wav = [f for f in os.listdir(rhythmicized_dir) if f.endswith(".wav")]
rhy_total = len(rhy_wav)

SUFFIX_MAP = {
    f"{BEAT_NAME_QUARTER_NOTE}-{BEAT_NAME_QUARTER_NOTE_REST}": SINGLE_AND_REST_RHYTHM,
    f"{BEAT_NAME_QUARTER_NOTE}-{BEAT_NAME_QUARTER_NOTE}": DOUBLE_RHYTHM,
    BEAT_NAME_QUARTER_NOTE: SINGLE_RHYTHM,
}

rhy_counts = Counter()
for fname in rhy_wav:
    m = re.search(r'_bpm-\d+_[\w]+_(.+?)\.wav$', fname)
    if m:
        suffix = m.group(1)
        for pat_suffix, pat_name in SUFFIX_MAP.items():
            if suffix == pat_suffix:
                rhy_counts[pat_name] += 1
                break

pattern_names = list(rp_weights.keys()) if rp_weights else sorted(rhy_counts.keys())

print(f"\n{'═'*72}")
print(f"  RHYTHMIC PATTERNS  (rhythmicized files: {rhy_total})")
print(f"{'═'*72}")
print(f"  {'Pattern':<20}  {'Weight':>7}  {'Target%':>8}  {'Actual':>7}  {'Actual%':>8}  {'Delta':>7}")
print(f"  {'-'*67}")
for name in pattern_names:
    weight   = rp_weights.get(name, 0)
    tgt_pct  = weight / rp_total_weight * 100 if rp_total_weight else 0
    act_n    = rhy_counts.get(name, 0)
    act_pct  = act_n / rhy_total * 100 if rhy_total else 0
    delta    = act_pct - tgt_pct
    dsym     = f"{delta:+.1f}%" if abs(delta) >= 0.5 else "   ~0%"
    print(f"  {name:<20}  {weight:>7}  {tgt_pct:>7.1f}%  {act_n:>7}  {act_pct:>7.1f}%  {dsym:>7}")
print()
