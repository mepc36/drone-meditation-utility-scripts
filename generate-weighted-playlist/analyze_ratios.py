"""Count input files by sound type, then count output file attributes."""
import json
import re
import sys
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lib.constants import (
    STRINGS,
    SOUND_GROUP_NAMES, SOUND_GROUP_TYPES,
    PANNING_CENTER, PANNING_DIAGONAL, PANNING_DUALPAN, PANNING_LEFT_OR_RIGHT,
    HARD_CENTER, HARD_LEFT, HARD_RIGHT, DIAGONAL_LEFT, DIAGONAL_RIGHT,
    DUALPAN_LEFTRIGHT, DUALPAN_DIAGONAL, UNTOUCHED,
    QUARTER_NOTE, QUARTER_NOTE_REST, BEAT_NAME_QUARTER_NOTE, BEAT_NAME_QUARTER_NOTE_REST,
    EIGHTH, SIXTEENTH, DOTTED_EIGHTH,
    BEAT_NAME_EIGHTH, BEAT_NAME_SIXTEENTH, BEAT_NAME_DOTTED_EIGHTH,
    RHYTHM_PATTERN_SEQUENCES,
    QUARTER_RHYTHM, DOUBLE_RHYTHM,
    MUSICAL_PATTERNS, VOLUMES, BPMS, MUSIC_PATTERN_PERCENT,
    RHYTHM_PATTERNS, RHYTHM_PATTERN, RHYTHM_PERCENT,
)
from lib.runtime_constants import LOUD, QUIET, SLOW
from lib.sound_rules import rules_by_sound_type, derive_type, derive_panning_key, sound_type_of
from lib.config import CFG_SOUND_GROUP_PERCENTS, CFG_SILENCE_RATIO, CFG_BPMS

BASE = Path(__file__).parent
cfg_path = BASE / "input/config/config.json"
input_audio_dir = BASE / "input/audio"
rhythmicized_dir = BASE / "output/rhythmicized-audio"

with open(cfg_path) as f:
    cfg = json.load(f)

cfg_sound_group_pcts = [int(x) for x in cfg.get(CFG_SOUND_GROUP_PERCENTS, '').split(':')]
_sil_ratio = [int(x) for x in cfg.get(CFG_SILENCE_RATIO, '100:0').split(':')]
_samp_pct, _sil_pct = _sil_ratio[0], (_sil_ratio[1] if len(_sil_ratio) > 1 else 0)

# Derive total counts from actual kicksnare input files (mirrors lib/config.py logic)
_kicksnare_count = sum(
    1 for f in input_audio_dir.iterdir()
    if f.name.endswith('_kick.wav') or f.name.endswith('_snare.wav')
)
_kicksnare_pct = cfg_sound_group_pcts[0] if cfg_sound_group_pcts else 50
cfg_num_audio = round(_kicksnare_count * 100 / _kicksnare_pct) if _kicksnare_pct else 0
if _sil_pct > 0:
    cfg_num_silence = round(cfg_num_audio * _sil_pct / _samp_pct)
else:
    cfg_num_silence = 0
cfg_num_unique = cfg_num_audio + cfg_num_silence

# ── Input file counts ─────────────────────────────────────────────────────────
input_wav = [f.name for f in input_audio_dir.glob("*.wav")]

input_type_counts = Counter(sound_type_of(f) for f in input_wav)

_type_to_group = {t: grp for grp in SOUND_GROUP_NAMES for t in SOUND_GROUP_TYPES[grp]}
input_group_counts = Counter()
for t, n in input_type_counts.items():
    grp = _type_to_group.get(t, "other")
    input_group_counts[grp] += n


# ── Output file counts ────────────────────────────────────────────────────────
_BEAT_NAMES = {
    QUARTER_NOTE:      BEAT_NAME_QUARTER_NOTE,
    QUARTER_NOTE_REST: BEAT_NAME_QUARTER_NOTE_REST,
    EIGHTH:            BEAT_NAME_EIGHTH,
    SIXTEENTH:         BEAT_NAME_SIXTEENTH,
    DOTTED_EIGHTH:     BEAT_NAME_DOTTED_EIGHTH,
}
SUFFIX_MAP = {
    '-'.join(_BEAT_NAMES[b] for b in beats): name
    for name, beats in RHYTHM_PATTERN_SEQUENCES.items()
}

all_wav = [f.name for f in rhythmicized_dir.glob("*.wav")]
silence_wav = [f for f in all_wav if f.startswith("silence_")]
strings_wav = [f for f in all_wav if sound_type_of(f) == STRINGS and not f.startswith("silence_")]
wav = [f for f in all_wav if sound_type_of(f) != STRINGS and not f.startswith("silence_")]
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

_cfg_bpms = sorted(float(x) for x in cfg.get(CFG_BPMS, '').split(':'))
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
_LOG_RHYTHMS = [QUARTER_RHYTHM, DOUBLE_RHYTHM]

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
out_dir = BASE / "output/analyze-ratios"
out_dir.mkdir(parents=True, exist_ok=True)

_sidecar_path = out_dir / "last-run-resolved-bias.json"
_resolved_bias = None
if _sidecar_path.exists():
    with open(_sidecar_path) as _f:
        _resolved_bias = json.load(_f)

# Print any randomly-resolved sample bias entries from the last run
if _resolved_bias:
    _random_entries = [
        (grp, entry)
        for grp, entries in _resolved_bias.items()
        for entry in entries
        if entry.get('was_random')
    ]
    if _random_entries:
        print("  SAMPLE BIAS  (randomly resolved):")
        for _grp, _entry in _random_entries:
            print(f"    {_grp:<14}  {_entry['biased_sample']}  ({_entry['biased_pool_pct']}%)")
        print()


def _file_contains_sample(fname: str, sample_name: str) -> bool:
    """True if an output filename was built from the given sample."""
    name_part = fname.split('_vol-')[0] if '_vol-' in fname else fname
    return (
        name_part == sample_name
        or name_part.startswith(sample_name + '_')
        or name_part.endswith('_' + sample_name)
    )


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
        "labels": [k for k in RHYTHM_PATTERN_SEQUENCES if rhy_counts.get(k, 0) > 0 or expected_rhy.get(k, 0) > 0],
        "counts": [rhy_counts.get(k, 0) for k in RHYTHM_PATTERN_SEQUENCES if rhy_counts.get(k, 0) > 0 or expected_rhy.get(k, 0) > 0],
        "expected": [round(expected_rhy.get(k, 0)) for k in RHYTHM_PATTERN_SEQUENCES if rhy_counts.get(k, 0) > 0 or expected_rhy.get(k, 0) > 0],
    },
]

if _resolved_bias:
    _all_biased_samples: list[str] = [
        _e['biased_sample']
        for _grp_entries in _resolved_bias.values()
        for _e in _grp_entries
        if 'biased_sample' in _e
    ]
    _total_biased_expected: int = sum(
        round(round(N * (cfg_sound_group_pcts[SOUND_GROUP_NAMES.index(_g)] if _g in SOUND_GROUP_NAMES else 0) / 100) * _e['biased_pool_pct'] / 100)
        for _g, _g_entries in _resolved_bias.items()
        for _e in _g_entries
        if 'biased_sample' in _e
    )
    _unbiased_added = False

    _bias_labels: list[str] = []
    _bias_counts: list[int] = []
    _bias_expected: list[int] = []
    _bias_was_random: list[bool] = []
    for _grp, _entries in _resolved_bias.items():
        _grp_idx = SOUND_GROUP_NAMES.index(_grp) if _grp in SOUND_GROUP_NAMES else -1
        _grp_pct = cfg_sound_group_pcts[_grp_idx] if _grp_idx >= 0 else 0
        _grp_n = round(N * _grp_pct / 100)
        for _entry in _entries:
            if 'biased_sample' in _entry:
                _sample = _entry['biased_sample']
                _was_random = _entry.get('was_random', False)
                _count = sum(1 for f in wav if _file_contains_sample(f, _sample))
                _expected = round(_grp_n * _entry['biased_pool_pct'] / 100)
                _word = _sample.split('_')[1] if '_' in _sample else _sample
                _bias_labels.append(_word)
                _bias_counts.append(_count)
                _bias_expected.append(_expected)
                _bias_was_random.append(_was_random)
            elif 'unbiased_pool_pct' in _entry and not _unbiased_added:
                _unbiased_added = True
                _count = sum(1 for f in wav if not any(_file_contains_sample(f, s) for s in _all_biased_samples))
                _expected = N - _total_biased_expected
                _bias_labels.append('unbiased')
                _bias_counts.append(_count)
                _bias_expected.append(_expected)
                _bias_was_random.append(False)
    if _bias_labels:
        DIMS.append({
            "title": "Biased Samples",
            "labels": _bias_labels,
            "counts": _bias_counts,
            "expected": _bias_expected,
            "bar_colors": ["#C0392B" if r else "#2166AC" for r in _bias_was_random],
        })

COLOR = "#2166AC"

fig, axes = plt.subplots(1, len(DIMS), figsize=(4 * len(DIMS), 6))
fig.suptitle(f"Rhythmicized Output  (total={N_total}/{cfg_num_unique}  N_non-strings_non-silence={N})", fontsize=14, fontweight="bold", y=1.01)

for ax, dim in zip(axes, DIMS):
    labels = dim["labels"]
    counts = dim["counts"]
    x = np.arange(len(labels))
    bar_colors = dim.get("bar_colors") or [COLOR] * len(labels)
    bars = ax.bar(x, counts, color=bar_colors, alpha=0.85)
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
out_path = out_dir / "rhythmicized-ratios.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"Chart saved → {out_path}")
