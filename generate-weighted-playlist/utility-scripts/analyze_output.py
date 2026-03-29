"""Analyze output/audio files vs config targets and print a delta chart."""
import os, re, json
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg_path  = os.path.join(BASE, "input/config/config.json")
audio_dir = os.path.join(BASE, "output/audio")

with open(cfg_path) as f:
    cfg = json.load(f)

def parse_pct(s):
    return [int(x) for x in str(s).split(":")]

pan_target = parse_pct(cfg["center_diagonal_dualpan_leftorright_percents"])   # center:diag:dual:lor
ks_target  = parse_pct(cfg["kicksnare_stab_acappella_percents"])               # kicksnare:stab:acap
vol_target = parse_pct(cfg["loud_medium_soft_percents"])                        # loud:soft
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
        vol_counts['loud' if int(m.group(1)) == 0 else 'soft'] += 1
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
W = 25  # bar width

def bar(pct):
    filled = round(pct / 100 * W)
    return "█" * filled + "░" * (W - filled)

def section(title, labels, actuals, targets, total):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")
    print(f"  {'Label':<16}  {'Target':>6}  {'Actual':>6}  {'Delta':>7}  Actual / Target")
    print(f"  {'-'*66}")
    for lbl, act, tgt in zip(labels, actuals, targets):
        pct = act / total * 100 if total else 0.0
        delta = pct - tgt
        dsym = f"{delta:+.1f}%" if abs(delta) >= 0.5 else "   ~0%"
        print(f"  {lbl:<16}  {tgt:>5}%   {pct:>5.1f}%  {dsym:>7}  A:{bar(pct)}")
        print(f"  {'':16}                          T:{bar(tgt)}")

print("=" * 70)
print("  OUTPUT vs CONFIG — DELTA ANALYSIS")
print("=" * 70)
print(f"\n  Files in output/audio : {N}")
print(f"  config num_unique_samples : {n_target}")
print(f"  Count delta : {N - n_target:+d}")

section(
    f"PANNING  (config: center:diagonal:dualpan:leftorright = {cfg['center_diagonal_dualpan_leftorright_percents']})",
    ["center", "diagonal", "dualpan", "leftorright"],
    [pan_counts["center"], pan_counts["diagonal"], pan_counts["dualpan"], pan_counts["leftorright"]],
    pan_target, N,
)

section(
    f"SOUND GROUP  (config: kicksnare:stab:acappella = {cfg['kicksnare_stab_acappella_percents']})",
    ["kicksnare", "stab", "acappella"],
    [grp_counts["kicksnare"], grp_counts["stab"], grp_counts["acappella"]],
    ks_target, N,
)

section(
    f"VOLUME  (config: loud:soft = {cfg['loud_medium_soft_percents']})",
    ["loud  (vol=0)", "soft  (vol=-22)"],
    [vol_counts["loud"], vol_counts["soft"]],
    vol_target, N,
)

section(
    f"BPM  (config bpms={bpms}  percents={bpm_target})",
    [f"bpm-{b}" for b in bpms],
    [bpm_counts[b] for b in bpms],
    bpm_target, N,
)

print(f"\n{'═'*70}")
print("  RAW COUNTS")
print(f"{'═'*70}")
print(f"  panning     : {dict(pan_counts)}")
print(f"  sound group : {dict(grp_counts)}")
print(f"  volume      : {dict(vol_counts)}")
print(f"  bpm         : {dict(bpm_counts)}")
print()
