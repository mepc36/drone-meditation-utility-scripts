"""Terminal reporting functions: panning, volume, and beat-ratio summaries."""
from math import gcd
from functools import reduce

import lib.config as cfg
from .constants import SOUND_GROUP_NAMES, STRINGS


def _gcd_of(*values: int) -> int:
    return reduce(gcd, values)


def print_panning_report(center: int, left: int, right: int, dualpan: int, hard_left: int, hard_right: int, conf: dict) -> None:
    diagonal = left + right
    leftorright = hard_left + hard_right
    total = center + diagonal + dualpan + leftorright
    if total == 0:
        return
    SEP = '=' * 60
    col_w = 10
    entries = [("center", center), ("dualpan", dualpan), ("L/R", leftorright)]
    if diagonal > 0:
        entries.append(("diag", diagonal))
    header = "".join(f"{name} #".ljust(col_w) + f"{name} %".ljust(col_w) for name, _ in entries)
    data   = "".join(str(count).ljust(col_w) + f"{count/total*100:.0f}%".ljust(col_w) for _, count in entries)
    print(f"{SEP}\nPANNING")
    print(f"  {header}")
    print(f"  {data}")


def print_volume_report(volume_counts: list[int], total_created: int, conf: dict) -> None:
    volume_levels_db = conf['volume_levels_db']
    active = [(db, count) for db, count in zip(volume_levels_db, volume_counts) if count > 0]
    total_v = sum(c for _, c in active)
    non_zero = [c for _, c in active]
    vol_gcd = _gcd_of(*non_zero) if len(non_zero) > 1 else (non_zero[0] if non_zero else 1)
    realized_ratio = ':'.join(str(c // vol_gcd) for _, c in active)
    SEP = '=' * 60
    col_w = 10
    header = "".join(f"{db:+.0f}dB #".ljust(col_w) + f"{db:+.0f}dB %".ljust(col_w) for db, _ in active) + "ratio"
    data   = "".join(str(count).ljust(col_w) + f"{count/total_v*100:.0f}%".ljust(col_w) for _, count in active) + realized_ratio
    print(f"{SEP}\nVOLUME")
    print(f"  {header}")
    print(f"  {data}")


def print_sound_group_report(group_appearances: dict[str, int], non_strings_created: int, strings_created: int, conf: dict, beat_multipliers: dict[str, float], input_counts: dict[str, int] | None = None) -> None:
    group_beats = {g: group_appearances[g] * beat_multipliers.get(g, 1.0) for g in SOUND_GROUP_NAMES}
    total_beats = sum(group_beats.values())
    target_percents = conf['sound_group_percents']
    TOLERANCE = 3.0
    W_INP, W_DUP, W_OUT, W_BEAT, W_IGN, W_PCT, W_TGT, W_DEL = 15, 14, 16, 9, 11, 17, 17, 16

    def _row(group_name, inp, dup, out, beat, ign, pct_s, tgt_s, del_s):
        return (
            f"  {group_name:<10}"
            f"  {str(inp):>{W_INP}}"
            f"  {str(dup):>{W_DUP}}"
            f"  {str(out):>{W_OUT}}"
            f"  {str(beat):>{W_BEAT}}"
            f"  {str(ign):>{W_IGN}}"
            f"  {str(pct_s):>{W_PCT}}"
            f"  {str(tgt_s):>{W_TGT}}"
            f"  {str(del_s):>{W_DEL}}"
        )

    header_row = _row('', 'num_input_files', 'num_duplicates', 'num_output_files', 'num_beats', 'num_ignored', 'actual_output_pct', 'target_output_pct', 'output_delta_pct')
    SEP = '=' * len(header_row)
    lines = []
    any_bad = False
    for group, target_pct in zip(SOUND_GROUP_NAMES, target_percents):
        count = group_appearances[group]
        beats = group_beats[group]
        realized_pct = (beats / total_beats * 100) if total_beats > 0 else 0
        off = abs(realized_pct - target_pct)
        if off > TOLERANCE:
            any_bad = True
        delta = abs(realized_pct - target_pct)
        inp = input_counts.get(group, 'N/A') if input_counts is not None else 'N/A'
        inp_int = inp if isinstance(inp, int) else 0
        dup = round(count / inp_int) if inp_int > 0 else 'N/A'
        lines.append(_row(group, inp, dup, count, int(beats), 'N/A', f"{realized_pct:.1f}%", f"{target_pct}%", f"Δ{delta:.1f}%"))
    target_str = conf['raw'][cfg.CFG_SOUND_GROUP_PERCENTS]
    print(SEP)
    print(f"BEAT RATIO CHECK  (target {target_str}, tolerance ±{TOLERANCE:.0f}%)")
    print(header_row)
    for line in lines:
        print(line)
    strings_pct = (strings_created / (total_beats + strings_created) * 100) if (total_beats + strings_created) > 0 else 0
    strings_inp = input_counts.get(STRINGS, 'N/A') if input_counts is not None else 'N/A'
    print(_row('strings', strings_inp, 'N/A', strings_created, strings_created, 'N/A', f"{strings_pct:.1f}%", 'N/A%', 'N/A'))
    num_silence = conf.get('num_silence_files', 0)
    if num_silence > 0:
        sil_pct = conf.get('silence_percent', 0)
        print(_row('silence', 'N/A', 'N/A', num_silence, num_silence, 'N/A', f"{sil_pct}%", f"{sil_pct}%", 'N/A'))
    if any_bad:
        print(f"  ⚠  RATIO OUTSIDE TOLERANCE — check beat_multipliers or input sample counts")
    print(SEP)
