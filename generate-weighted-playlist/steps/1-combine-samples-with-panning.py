#!/usr/bin/env python3

import os
import sys
import json
import random
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from math import gcd
from pathlib import Path
from threading import Semaphore

sys.path.insert(0, str(Path(__file__).parent.parent))

import soundfile as sf
import numpy as np

import lib.config as cfg
from lib.audio_processing import apply_rhythm_pattern, load_audio, load_and_prepare_sample, mix_samples_into_stereo_clip, reduce_volume_by_db, write_silence_file
from lib.deck_builder import SlotSpec, plan_output_files, build_permutation_kick_snare_deck, build_permutation_stab_deck, build_permutation_acappella_deck, balance_permutation_decks, compute_group_beat_multipliers, plan_permutation_trimming
from lib.sample_queue import (
    build_biased_reservations,
    create_shuffled_sample_queue,
    draw_next_sample_of_types,
    draw_next_strings_sample,
    load_samples_grouped_by_type,
    resolve_random_entries,
    validate_sample_bias,
)
from lib.sound_rules import (
    passes_through_unmodified,
    rules_by_sound_type,
    sound_type_of,
    derive_panning_key,
)
from lib.constants import (
    HARD_CENTER, HARD_LEFT, HARD_RIGHT, DIAGONAL_LEFT, DIAGONAL_RIGHT,
    SampleRole,
    DUALPAN_LEFTRIGHT, DUALPAN_DIAGONAL, UNTOUCHED,
    SOUND_GROUP_NAMES, SOUND_GROUP_TYPES,
    STRINGS, ACAPPELLA, KICKSNARE, STAB,
    PANNING_CENTER, PANNING_DIAGONAL, PANNING_LEFT, PANNING_RIGHT,
    PANNING_DUALPAN, PANNING_LEFT_OR_RIGHT,
    MUSICAL_PATTERNS, VOLUMES, BPMS,
    MUSIC_PATTERN_PERCENT,
    QUARTER_NOTE, QUARTER_NOTE_REST,
    BEAT_NAME_QUARTER_NOTE_REST, BEAT_NAME_QUARTER_NOTE,
    EIGHTH, SIXTEENTH, DOTTED_EIGHTH,
    BEAT_NAME_EIGHTH, BEAT_NAME_SIXTEENTH, BEAT_NAME_DOTTED_EIGHTH,
    MAX_DRAW_RETRIES,
)
from lib.runtime_constants import LOUD, QUIET, SLOW, FAST


_RENDER_WORKERS = min(4, os.cpu_count() or 2)
_MAX_PENDING_RENDERS = _RENDER_WORKERS * 8
_WRITE_WORKERS = min(16, (os.cpu_count() or 4) * 2)
_MAX_PENDING_WRITES = _WRITE_WORKERS * 8


def _bounded_write(path: Path, audio, sample_rate: int, sem: Semaphore) -> None:
    try:
        sf.write(path, audio, sample_rate)
    finally:
        sem.release()


def _render_and_write(
    audio,
    sample_rate: int,
    beat_length: float,
    rhythm: tuple,
    beat_pannings: tuple,
    output_path: Path,
    rhythmicized_path,
    write_sem: Semaphore,
    write_executor: ThreadPoolExecutor,
    render_sem: Semaphore,
    per_beat_audio: list | None = None,
) -> None:
    try:
        write_sem.acquire()
        write_executor.submit(_bounded_write, output_path, audio, sample_rate, write_sem)
        if rhythmicized_path is not None:
            rhythmicized_audio = audio if rhythm == (UNTOUCHED,) else apply_rhythm_pattern(
                audio, sample_rate, beat_length, rhythm, beat_pannings, per_beat_audio
            )
            write_sem.acquire()
            write_executor.submit(_bounded_write, rhythmicized_path, rhythmicized_audio, sample_rate, write_sem)
    finally:
        render_sem.release()


def clear_output_directory(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def panning_group_from_assignments(sample_names: list[str], pan_assignments: dict) -> str:
    pannings = {pan_assignments[n] for n in sample_names}
    if len(pannings) > 1:
        # Two samples with distinct positions (e.g. HARD_LEFT+HARD_RIGHT or DIAGONAL pair) → dualpan
        return PANNING_DUALPAN
    pan = next(iter(pannings))
    if pan == HARD_CENTER:
        return PANNING_CENTER
    if pan in (HARD_LEFT, HARD_RIGHT):
        return PANNING_LEFT_OR_RIGHT
    return PANNING_DIAGONAL


BEAT_NAMES: dict[float, str] = {
    QUARTER_NOTE_REST: BEAT_NAME_QUARTER_NOTE_REST,
    QUARTER_NOTE:      BEAT_NAME_QUARTER_NOTE,
    EIGHTH:            BEAT_NAME_EIGHTH,
    SIXTEENTH:         BEAT_NAME_SIXTEENTH,
    DOTTED_EIGHTH:     BEAT_NAME_DOTTED_EIGHTH,
}


def rhythm_to_name(rhythm: tuple[float | str, ...]) -> str:
    if rhythm == (UNTOUCHED,):
        return ''
    return '-'.join(BEAT_NAMES.get(v, str(v)) for v in rhythm)


def build_output_filename(
    sample_names: list[str],
    pan_assignments: dict[str, str],
    volume_db: float,
    index: int,
    bpm: int,
    rhythm: tuple[float, ...] = (),
) -> str:
    ordered_by_pan = sorted(sample_names, key=lambda n: pan_assignments[n])
    name_part = "_".join(n.lower() for n in ordered_by_pan)
    vol_str = f"{abs(volume_db):.0f}" if volume_db == int(volume_db) else f"{abs(volume_db):.1f}"
    pan_group = panning_group_from_assignments(sample_names, pan_assignments)
    rhythm_name = rhythm_to_name(rhythm) if rhythm else ""
    rhythm_part = f"_{rhythm_name}" if rhythm_name else ""
    return f"{name_part}_vol-{vol_str}_index-{index:03d}_bpm-{bpm}_{pan_group}{rhythm_part}.wav"


def gcd_of(*values: int) -> int:
    return reduce(gcd, values)


def resolve_slot(
    slot: SlotSpec,
    sample_queue: deque,
    all_samples: list[str],
    seen_combinations: set,
    conf: dict,
    forced_primary: str | None = None,
) -> tuple[list[str], dict[str, str], float, int, int] | None:
    loudest_idx = conf['loudest_volume_index']
    quietest_idx = conf['quietest_volume_index']
    slowest_idx = conf['slowest_bpm_index']
    fastest_idx = conf['fastest_bpm_index']
    volume_levels_db = conf['volume_levels_db']

    if slot.sound_group == STRINGS:
        primary = draw_next_strings_sample(sample_queue, all_samples)
        if primary is None:
            return None
        sample_names = [primary]
        pan_assignments = {primary: HARD_CENTER}
        combo_key = tuple(sorted(f"{n}:{pan_assignments[n]}" for n in sample_names))
        if combo_key in seen_combinations:
            return None
        return sample_names, pan_assignments, 0.0, 0, slowest_idx

    is_biased = forced_primary is not None

    for _ in range(1 if is_biased else MAX_DRAW_RETRIES):
        primary = forced_primary if is_biased else draw_next_sample_of_types(
            sample_queue, all_samples, SOUND_GROUP_TYPES[slot.sound_group]
        )
        if primary is None:
            return None

        rule = rules_by_sound_type.get(sound_type_of(primary))

        if slot.panning == DUALPAN_LEFTRIGHT:
            partner_types = set(rule['dualpan_partners']) if rule else {sound_type_of(primary)}
            partner = draw_next_sample_of_types(sample_queue, all_samples, partner_types, exclude_name=primary)
            if partner:
                sample_names = [primary, partner]
                pan_assignments = {primary: HARD_LEFT, partner: HARD_RIGHT}
            else:
                sample_names = [primary]
                pan_assignments = {primary: DIAGONAL_LEFT}
        elif slot.panning == DUALPAN_DIAGONAL:
            partner_types = set(rule['dualpan_partners']) if rule else {sound_type_of(primary)}
            partner = draw_next_sample_of_types(sample_queue, all_samples, partner_types, exclude_name=primary)
            if partner:
                sample_names = [primary, partner]
                pan_assignments = {primary: DIAGONAL_LEFT, partner: DIAGONAL_RIGHT}
            else:
                sample_names = [primary]
                pan_assignments = {primary: DIAGONAL_LEFT}
        elif slot.panning == HARD_CENTER:
            sample_names, pan_assignments = [primary], {primary: HARD_CENTER}
        elif slot.panning == DIAGONAL_LEFT:
            sample_names, pan_assignments = [primary], {primary: DIAGONAL_LEFT}
        elif slot.panning == DIAGONAL_RIGHT:
            sample_names, pan_assignments = [primary], {primary: DIAGONAL_RIGHT}
        elif slot.panning == HARD_LEFT:
            sample_names, pan_assignments = [primary], {primary: HARD_LEFT}
        elif slot.panning == HARD_RIGHT:
            sample_names, pan_assignments = [primary], {primary: HARD_RIGHT}
        else:
            raise ValueError(f"Unhandled panning value: {slot.panning!r}")

        combo_key = tuple(sorted(f"{n}:{pan_assignments[n]}" for n in sample_names))
        if is_biased or slot.sound_group != KICKSNARE or combo_key not in seen_combinations:
            volume_db = volume_levels_db[loudest_idx if slot.volume_label == LOUD else quietest_idx]
            volume_idx = loudest_idx if slot.volume_label == LOUD else quietest_idx
            bpm_idx = slowest_idx if slot.bpm_label == SLOW else fastest_idx
            return sample_names, pan_assignments, volume_db, volume_idx, bpm_idx

    return None


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
    vol_gcd = gcd_of(*non_zero) if len(non_zero) > 1 else (non_zero[0] if non_zero else 1)
    realized_ratio = ':'.join(str(c // vol_gcd) for _, c in active)
    SEP = '=' * 60
    col_w = 10
    header = "".join(f"{db:+.0f}dB #".ljust(col_w) + f"{db:+.0f}dB %".ljust(col_w) for db, _ in active) + "ratio"
    data   = "".join(str(count).ljust(col_w) + f"{count/total_v*100:.0f}%".ljust(col_w) for _, count in active) + realized_ratio
    print(f"{SEP}\nVOLUME")
    print(f"  {header}")
    print(f"  {data}")


def print_sound_group_report(group_appearances: dict[str, int], non_strings_created: int, strings_created: int, conf: dict, beat_multipliers: dict[str, float], perm_multipliers: dict[str, int] | None = None, removed_counts: dict[str, int] | None = None, input_counts: dict[str, int] | None = None) -> None:
    group_beats = {g: group_appearances[g] * beat_multipliers.get(g, 1.0) for g in SOUND_GROUP_NAMES}
    total_beats = sum(group_beats.values())
    target_percents = conf['sound_group_percents']
    TOLERANCE = conf.get('permutation_tolerance_pct', 3.0) if conf.get('kick_snare_permutation_mode') else 3.0
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
        rm = removed_counts.get(group, 0) if removed_counts is not None else 'N/A'
        inp = input_counts.get(group, 'N/A') if input_counts is not None else 'N/A'
        rm_int = rm if isinstance(rm, int) else 0
        inp_int = inp if isinstance(inp, int) else 0
        effective_inp = inp_int - rm_int
        dup = round(count / effective_inp) if effective_inp > 0 else 'N/A'
        lines.append(_row(group, inp, dup, count, int(beats), rm, f"{realized_pct:.1f}%", f"{target_pct}%", f"Δ{delta:.1f}%"))
    target_str = conf['raw'][cfg.CFG_SOUND_GROUP_PERCENTS]
    print(SEP)
    print(f"BEAT RATIO CHECK  (target {target_str}, tolerance \u00b1{TOLERANCE:.0f}%)")
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
        print(f"  \u26a0  RATIO OUTSIDE TOLERANCE — check beat_multipliers or input sample counts")
    print(SEP)


def print_biased_sample_report(
    bias_config: dict,
    sample_usage_count: dict[str, int],
    group_targets: dict[str, int],
) -> None:
    print(f"\nBiased Sample Distribution:")
    for group, entries in bias_config.items():
        group_total = group_targets.get(group, 0)
        for entry in entries:
            if 'biased_sample' not in entry:
                continue
            sample = entry['biased_sample']
            target_pct = entry['biased_pool_pct']
            target_count = round(group_total * target_pct / 100)
            actual_count = sample_usage_count.get(sample, 0)
            actual_pct = (actual_count / group_total * 100) if group_total > 0 else 0
            print(f"  {group}/{sample}: {actual_count} uses ({actual_pct:.1f}%, target {target_pct}%/{target_count} slots)")


def generate_silence_files(
    output_dir: Path,
    rhythmicized_output_dir: Path,
    sample_rate: int,
    conf: dict,
    starting_index: int,
) -> None:
    num_silence = conf['num_silence_files']
    lengths = conf['silence_lengths_seconds']
    percents = conf['silence_length_percents']
    print(f"\nGenerating silence files...")
    print(f"  Ratio: {conf['samples_percent']}:{conf['silence_percent']} (samples:silence percentages)")
    print(f"  Total silence files to create: {num_silence}")

    counts_per_length: list[int] = []
    remaining = num_silence
    for i, pct in enumerate(percents):
        count = remaining if i == len(percents) - 1 else int(num_silence * pct / 100)
        counts_per_length.append(count)
        remaining -= count

    print(f"  Silence length distribution:")
    for length_sec, count in zip(lengths, counts_per_length):
        print(f"    {int(length_sec * 1000)}ms: {count} files")

    file_index = starting_index
    for length_sec, count in zip(lengths, counts_per_length):
        for _ in range(count):
            write_silence_file(output_dir, sample_rate, length_sec, file_index)
            write_silence_file(rhythmicized_output_dir, sample_rate, length_sec, file_index)
            file_index += 1
            if (file_index - starting_index) % 10 == 1:
                print(f"    Created {file_index - starting_index}/{num_silence} silence files...", end="\r")
    print(f"    Created {num_silence}/{num_silence} silence files...")


def main() -> None:
    conf = cfg.load()
    output_dir = cfg.OUTPUT_AUDIO_DIR
    input_audio_dir = cfg.INPUT_AUDIO_DIR
    rhythmicized_output_dir = cfg.OUTPUT_RHYTHMICIZED_AUDIO_DIR

    bpms_str = ':'.join(str(b) for b in conf['bpm_values'])

    samples_by_type = load_samples_grouped_by_type(input_audio_dir)
    _raw_input_counts: dict[str, int] = {
        g: sum(len(samples_by_type.get(t, [])) for t in SOUND_GROUP_TYPES[g])
        for g in SOUND_GROUP_NAMES
    }
    _raw_input_counts[STRINGS] = len(samples_by_type.get(STRINGS, []))
    total_sample_count = sum(len(s) for s in samples_by_type.values())
    print(f"Found {total_sample_count} samples in ./{input_audio_dir}:")
    for sound_type, samples in sorted(samples_by_type.items()):
        print(f"  - {sound_type}: {len(samples)}")
    print()

    clear_output_directory(output_dir)
    clear_output_directory(rhythmicized_output_dir)

    # Expand strings sample list according to the global duplication count.
    _duplicate_strings = conf['duplicate_strings']
    if _duplicate_strings > 0:
        _expanded_strings: list[str] = []
        for name in samples_by_type.get(STRINGS, []):
            _expanded_strings.extend([name] * (1 + round(_duplicate_strings)))
        samples_by_type = {**samples_by_type, STRINGS: _expanded_strings}

    num_pass_through = sum(len(v) for k, v in samples_by_type.items() if passes_through_unmodified(k))
    num_strings_samples = num_pass_through
    if num_strings_samples > conf['num_audio_samples']:
        raise ValueError(
            f"{num_strings_samples} strings slots exceeds total audio slots {conf['num_audio_samples']}. "
            f"Reduce num_times_duplicate_strings or the number of strings input files."
        )
    # Anchor to kicksnare: every KS file appears exactly once.
    # Other groups scaled so total beats heard match the configured ratio.
    beat_multipliers = compute_group_beat_multipliers()
    ks_slot_count = sum(len(samples_by_type.get(t, [])) for t in SOUND_GROUP_TYPES[KICKSNARE])
    ks_pct = conf['sound_group_percents'][SOUND_GROUP_NAMES.index(KICKSNARE)]
    ks_beats = ks_slot_count * beat_multipliers[KICKSNARE]
    group_targets: dict[str, int] = {}
    for group, pct in zip(SOUND_GROUP_NAMES, conf['sound_group_percents']):
        if group == KICKSNARE:
            group_targets[group] = ks_slot_count
        else:
            group_targets[group] = round(ks_beats * pct / ks_pct / beat_multipliers[group])
    print(f"  Beat-anchor multipliers: "
          + ", ".join(f"{g}={beat_multipliers[g]:.2f}" for g in SOUND_GROUP_NAMES))
    print(f"  \nFile-count targets: "
          + "".join(f"\n   - {g}={group_targets[g]}" for g in SOUND_GROUP_NAMES))
    # Recompute silence count based on beat-anchored total (may differ from config estimate).
    # In permutation mode, cfg.load() already computed the correct value from the actual
    # permutation deck size; group_targets here reflects raw input counts, not output files.
    if not conf['kick_snare_permutation_mode']:
        _audio_total = sum(group_targets.values()) + num_strings_samples
        conf['num_silence_files'] = (
            round(_audio_total * conf['silence_percent'] / conf['samples_percent'])
            if conf['samples_percent'] > 0 and conf['silence_percent'] > 0
            else 0
        )

    # Derive panning quotas, bpm targets, and volume targets from sound_rules.
    # In permutation mode, kick/snare slots are handled separately and excluded here.
    panning_quotas = {PANNING_CENTER: 0, PANNING_DIAGONAL: 0, PANNING_DUALPAN: 0, PANNING_LEFT: 0, PANNING_RIGHT: 0}
    bpm_targets    = {conf['slowest_bpm_index']: 0, conf['fastest_bpm_index']: 0}
    vol_targets    = {conf['loudest_volume_index']: 0, conf['quietest_volume_index']: 0}
    single_bpm     = conf['slowest_bpm_index'] == conf['fastest_bpm_index']
    for group, count in group_targets.items():
        if conf['kick_snare_permutation_mode']:
            continue
        bpm_labels, vol_weights, pan_weights = set(), {}, {}
        for sound_type in SOUND_GROUP_TYPES[group]:
            rule = rules_by_sound_type.get(sound_type)
            if rule is None:
                continue
            for entry in rule[MUSICAL_PATTERNS]:
                bpm_labels.update(b for b in entry[BPMS] if b is not UNTOUCHED)
                for v in entry[VOLUMES]:
                    if v is not UNTOUCHED:
                        vol_weights[v] = vol_weights.get(v, 0) + entry[MUSIC_PATTERN_PERCENT]
                pkey = derive_panning_key(entry)
                if pkey is not UNTOUCHED:
                    pan_weights[pkey] = pan_weights.get(pkey, 0) + entry[MUSIC_PATTERN_PERCENT]
        if single_bpm:
            if bpm_labels:
                bpm_targets[conf['slowest_bpm_index']] += count
        elif SLOW in bpm_labels and FAST not in bpm_labels:
            bpm_targets[conf['slowest_bpm_index']] += count
        elif FAST in bpm_labels and SLOW not in bpm_labels:
            bpm_targets[conf['fastest_bpm_index']] += count
        total_vol_w = sum(vol_weights.values()) or 1
        vol_remaining = count
        for i, (vlabel, vw) in enumerate(sorted(vol_weights.items(), key=lambda x: -x[1])):
            share = vol_remaining if i == len(vol_weights) - 1 else round(count * vw / total_vol_w)
            share = min(share, vol_remaining)
            if vlabel == LOUD:
                vol_targets[conf['loudest_volume_index']] += share
            elif vlabel == QUIET:
                vol_targets[conf['quietest_volume_index']] += share
            vol_remaining -= share
        total_w = sum(pan_weights.values()) or 1
        remaining = count
        for i, (pkey, w) in enumerate(sorted(pan_weights.items(), key=lambda x: -x[1])):
            share = remaining if i == len(pan_weights) - 1 else round(count * w / total_w)
            share = min(share, remaining)
            if pkey == HARD_CENTER:
                panning_quotas[PANNING_CENTER] += share
            elif pkey in (DIAGONAL_LEFT, DIAGONAL_RIGHT):
                panning_quotas[PANNING_DIAGONAL] += share
            elif pkey in (DUALPAN_LEFTRIGHT, DUALPAN_DIAGONAL):
                panning_quotas[PANNING_DUALPAN] += share
            elif pkey == HARD_LEFT:
                panning_quotas[PANNING_LEFT] += share
            elif pkey == HARD_RIGHT:
                panning_quotas[PANNING_RIGHT] += share
            remaining -= share

    perm_multipliers: dict[str, int] = {}
    removed_counts: dict[str, int] | None = None

    if conf['kick_snare_permutation_mode']:
        while True:
            try:
                trimmed_samples, m_stab, m_acap, diag = plan_permutation_trimming(
                    samples_by_type,
                    conf['sound_group_percents'],
                    max_files=conf['permutation_max_files'],
                    tolerance=conf['permutation_tolerance_pct'],
                    ks_ignored_cap=conf['kicksnare_ignored_cap'],
                )
                break
            except ValueError as _trim_err:
                if not sys.stdin.isatty():
                    raise
                print(f"\n  {'=' * 60}")
                print(f"\n  ✗  {_trim_err}")
                print(f"\n  Current values:")
                print(f"    permutation_tolerance_pct : {conf['permutation_tolerance_pct']}")
                print(f"    permutation_max_files     : {conf['permutation_max_files']}")
                print(f"\n  Enter new values (or press Enter to keep current, 'n' to abort):")
                _raw_tol = input(f"    permutation_tolerance_pct [{conf['permutation_tolerance_pct']}]: ").strip()
                _raw_max = input(f"    permutation_max_files [{conf['permutation_max_files']}]: ").strip()
                if _raw_tol.lower() in ('n', 'no') or _raw_max.lower() in ('n', 'no'):
                    print("  Aborted.")
                    sys.exit(0)
                try:
                    if _raw_tol:
                        _new_tol = float(_raw_tol)
                        if _new_tol <= 0:
                            raise ValueError("must be positive")
                        conf['permutation_tolerance_pct'] = _new_tol
                    if _raw_max:
                        _new_max = int(_raw_max)
                        if _new_max <= 0:
                            raise ValueError("must be positive")
                        conf['permutation_max_files'] = _new_max
                except ValueError as _ve:
                    print(f"  Invalid input ({_ve}), try again.")
                    continue
                # Persist updated values back to config.json
                _raw_cfg = conf['raw']
                _raw_cfg[cfg.CFG_PERMUTATION_MODE][cfg.CFG_PERMUTATION_TOLERANCE] = conf['permutation_tolerance_pct']
                _raw_cfg[cfg.CFG_PERMUTATION_MODE][cfg.CFG_PERMUTATION_MAX_FILES] = conf['permutation_max_files']
                with open(cfg.CONFIG_PATH, 'w') as _cfg_f:
                    json.dump(_raw_cfg, _cfg_f, indent=2)
                    _cfg_f.write('\n')
                print(f"  ✓  config.json updated. Retrying...")

        # Print ignored-samples report and allow interactive swaps
        ignored = diag['ignored_by_type']
        p_ks, p_stab, p_acap = diag['beat_pcts']
        target_str_perm = conf['raw'][cfg.CFG_SOUND_GROUP_PERCENTS]
        SEP60 = '=' * 60

        # Build a stable flat list of ignored slots: (number, sound_type, current_name, original_name)
        # Each slot is identified by a 1-based index that never changes.
        ignored_slots: list[list] = []  # [idx, sound_type, current_name, original_name]
        for sound_type, names in ignored.items():
            for name in sorted(names):
                ignored_slots.append([len(ignored_slots) + 1, sound_type, name, name])

        # Build a per-sound-type pool of available (kept) samples for swapping
        # kept = all samples of that type that are NOT currently ignored
        def _kept_pool(sound_type: str) -> list[str]:
            currently_ignored = {s[2] for s in ignored_slots if s[1] == sound_type}
            return [s for s in samples_by_type.get(sound_type, []) if s not in currently_ignored]

        def _print_trim_report() -> None:
            print(f"\n{SEP60}")
            print(f"PERMUTATION TRIM  (target {target_str_perm})")
            if ignored_slots:
                total_ignored = len(ignored_slots)
                print(f"  {total_ignored} sample(s) will be ignored to improve beat ratio:")
                for idx, sound_type, current, original in ignored_slots:
                    if current != original:
                        print(f"    [{idx}] [{sound_type}]  {current}  <--- CHANGED FROM {original}")
                    else:
                        print(f"    [{idx}] [{sound_type}]  {current}")
            else:
                print("  No samples ignored.")
            print(f"\n  Multipliers: stab ×{m_stab}, acappella ×{m_acap}")
            print(f"  Projected beat ratio: {p_ks:.1f}% KS / {p_stab:.1f}% stab / {p_acap:.1f}% acap"
                  f"  (target {target_str_perm})")
            print(f"  Projected total files: {diag['total_files']}")
            if not diag['within_tolerance']:
                print(f"  ⚠  Beat ratio is outside ±3% tolerance — best achievable given constraints.")

        _print_trim_report()

        if sys.stdin.isatty() and ignored_slots:
            while True:
                raw = input("\n  Enter a slot number to swap it out, 'n' to abort, or Enter to proceed: ").strip()
                if not raw:
                    break
                if raw.lower() in ('n', 'no'):
                    print("  Aborted.")
                    sys.exit(0)
                if not raw.isdigit():
                    print("  Please enter a number.")
                    continue
                slot_num = int(raw)
                matching = [s for s in ignored_slots if s[0] == slot_num]
                if not matching:
                    print(f"  No slot #{slot_num}. Valid numbers: {[s[0] for s in ignored_slots]}")
                    continue
                slot = matching[0]
                sound_type = slot[1]
                pool = _kept_pool(sound_type)
                if not pool:
                    print(f"  No kept {sound_type} samples available to swap in.")
                    continue
                replacement = random.choice(pool)
                slot[2] = replacement  # update current name; original stays fixed
                _print_trim_report()

            # Apply any swaps back into trimmed_samples
            for idx, sound_type, current, original in ignored_slots:
                if current != original:
                    # Remove original from trimmed set, add original back, remove current
                    for typ in (sound_type,):
                        lst = list(trimmed_samples.get(typ, []))
                        if original in lst:
                            lst.remove(original)  # original was kept (it's the replacement now)
                        if current not in lst:
                            lst.append(current)   # current replaces it in the ignored set
                        trimmed_samples = {**trimmed_samples, typ: lst}

            # Rebuild ignored_by_type from final slot state so the render loop is consistent
            from collections import defaultdict as _dd
            final_ignored: dict[str, list[str]] = _dd(list)
            for idx, sound_type, current, original in ignored_slots:
                final_ignored[sound_type].append(current)
            # trimmed_samples must exclude all currently-ignored names
            for sound_type, ignored_names in final_ignored.items():
                ignored_set = set(ignored_names)
                trimmed_samples = {
                    **trimmed_samples,
                    sound_type: [s for s in samples_by_type.get(sound_type, []) if s not in ignored_set],
                }

        if sys.stdin.isatty() and not ignored_slots:
            answer = input("\n  Proceed? [Y/n] ").strip().lower()
            if answer and answer not in ('y', 'yes'):
                print("  Aborted.")
                sys.exit(0)
        elif not sys.stdin.isatty():
            print("  (Non-interactive mode — proceeding automatically)")

        removed_counts = {
            g: sum(1 for _, st, _, _ in ignored_slots if st in SOUND_GROUP_TYPES[g])
            for g in SOUND_GROUP_NAMES
        }

        # Rebuild deck using trimmed samples and pre-computed multipliers
        ks_deck   = build_permutation_kick_snare_deck(trimmed_samples)
        stab_deck = build_permutation_stab_deck(trimmed_samples)
        acap_deck = build_permutation_acappella_deck(trimmed_samples)
        non_strings_deck = balance_permutation_decks(
            {KICKSNARE: ks_deck, STAB: stab_deck, ACAPPELLA: acap_deck},
            conf['sound_group_percents'],
            m_stab=m_stab,
            m_acap=m_acap,
        )
        # Update samples_by_type so the render loop and queue use trimmed sets
        samples_by_type = trimmed_samples
        # Use exact beats-per-output-file for the post-run sound group report
        beat_multipliers = diag['perm_beat_multipliers']
        perm_multipliers = {KICKSNARE: 1, STAB: m_stab, ACAPPELLA: m_acap}
    else:
        non_strings_deck = plan_output_files(
            group_targets, panning_quotas, bpm_targets, vol_targets,
            conf['slowest_bpm_index'], conf['fastest_bpm_index'],
            conf['loudest_volume_index'], conf['quietest_volume_index'],
        )
    strings_slots = [SlotSpec(STRINGS, UNTOUCHED, UNTOUCHED, UNTOUCHED, rhythm=(UNTOUCHED,))] * num_strings_samples
    full_deck = non_strings_deck + strings_slots
    random.shuffle(full_deck)

    total_slots = len(full_deck)
    _print_interval = max(100, total_slots // 200)
    print(f"\n  Total samples to generate: {total_slots}\n")
    if conf['kick_snare_permutation_mode'] and not sys.stdin.isatty():
        pass  # prompt already handled above

    sample_queue, all_samples = create_shuffled_sample_queue(samples_by_type)
    sample_usage_count: dict[str, int] = {s: 0 for s in all_samples}
    seen_combinations: set = set()
    # In permutation mode each (sample × panning × bpm) combo repeats exactly
    # M times (the LCM multiplier). Cache only the base stereo audio (pre-rhythm)
    # so each mix is computed once and rhythm is re-applied cheaply per slot.
    render_cache: dict[tuple, np.ndarray] | None = (
        {} if conf['kick_snare_permutation_mode'] else None
    )

    if conf.get('sample_bias'):
        validate_sample_bias(conf['sample_bias'], samples_by_type)
    resolved_bias = resolve_random_entries(conf.get('sample_bias') or {}, samples_by_type)
    biased_reservations = build_biased_reservations(resolved_bias, group_targets)

    if resolved_bias:
        _sidecar_dir = cfg.OUTPUT_AUDIO_DIR.parent / "analyze-ratios"
        _sidecar_dir.mkdir(parents=True, exist_ok=True)
        with open(_sidecar_dir / "last-run-resolved-bias.json", "w") as _f:
            json.dump(resolved_bias, _f, indent=2)

    first_sample = list(samples_by_type.values())[0][0]
    _, sample_rate = load_audio(input_audio_dir, first_sample)

    # Pre-load every input sample once: load + resample + normalize.
    # The render loop re-uses these arrays instead of hitting disk per slot.
    all_unique_samples = sorted(s for samples in samples_by_type.values() for s in samples)
    print(f"  Pre-loading {len(all_unique_samples)} input sample(s)...")
    prepared_cache: dict[str, np.ndarray] = {
        name: load_and_prepare_sample(name, input_audio_dir, sample_rate)
        for name in all_unique_samples
    }

    created_count = strings_created = non_strings_created = 0
    group_appearances: dict[str, int] = {g: 0 for g in SOUND_GROUP_NAMES}
    center_count = left_count = right_count = dualpan_count = hard_left_count = hard_right_count = 0
    volume_counts = [0] * len(conf['volume_levels_db'])

    _render_sem = Semaphore(_MAX_PENDING_RENDERS)
    render_executor = ThreadPoolExecutor(max_workers=_RENDER_WORKERS)
    _write_sem = Semaphore(_MAX_PENDING_WRITES)
    write_executor = ThreadPoolExecutor(max_workers=_WRITE_WORKERS)

    for slot in full_deck:
        # In permutation mode, kick/snare slots carry a forced_sample assigned at
        # deck-build time. That takes priority over any sample_bias reservation.
        forced_primary = slot.forced_sample
        if forced_primary is None:
            reservation = biased_reservations.get(slot.sound_group)
            if reservation:
                forced_primary = reservation.popleft()

        resolved = resolve_slot(slot, sample_queue, all_samples, seen_combinations, conf, forced_primary=forced_primary)
        if resolved is None:
            continue
        sample_names, pan_assignments, volume_db, volume_idx, bpm_idx = resolved

        combo_key = tuple(sorted(f"{n}:{pan_assignments[n]}" for n in sample_names))
        if forced_primary is None and slot.sound_group == KICKSNARE:
            seen_combinations.add(combo_key)
        for name in sample_names:
            sample_usage_count[name] = sample_usage_count.get(name, 0) + 1

        created_count += 1
        if slot.sound_group == STRINGS:
            strings_created += 1
        else:
            non_strings_created += 1
            group_appearances[slot.sound_group] = group_appearances.get(slot.sound_group, 0) + 1

        pan_values = list(pan_assignments.values())
        if len(pan_values) == 2:
            dualpan_count += 1
        else:
            pan = pan_values[0]
            if pan == HARD_CENTER:
                center_count += 1
            elif pan == HARD_LEFT:
                hard_left_count += 1
            elif pan == HARD_RIGHT:
                hard_right_count += 1
            elif float(pan) < 0:
                left_count += 1
            elif float(pan) > 0:
                right_count += 1
            else:
                raise ValueError(f"Unrecognised single-sample pan value in reporting: {pan!r}")

        if slot.sound_group != STRINGS:
            volume_counts[volume_idx] += 1

        beat_length = conf['beat_lengths_seconds'][bpm_idx]

        extra_vol_db = 0.0
        if slot.sound_group == STRINGS and conf['strings_volume_adjustment_db']:
            extra_vol_db = float(conf['strings_volume_adjustment_db'])
        elif slot.sound_group == ACAPPELLA and conf['acappella_volume_adjustment_db']:
            extra_vol_db = float(conf['acappella_volume_adjustment_db'])

        render_key: tuple | None = None
        if render_cache is not None:
            render_key = (
                tuple(sorted(f"{n}:{pan_assignments[n]}" for n in sample_names)),
                bpm_idx, volume_db, extra_vol_db,
            )

        if render_key is not None and render_key in render_cache:
            audio = render_cache[render_key]
        else:
            audio = mix_samples_into_stereo_clip(
                sample_names, pan_assignments, input_audio_dir, sample_rate, volume_db, beat_length,
                prepared_cache=prepared_cache,
            )
            if extra_vol_db != 0.0:
                audio = reduce_volume_by_db(audio, extra_vol_db)
            if render_key is not None:
                render_cache[render_key] = audio

        # Role-based per-beat audio (e.g. A/B/A patterns).
        # SampleRole.SAME  → reuse the primary sample's audio
        # SampleRole.NEW   → draw a fresh different sample for this beat
        per_beat_audio_list: list | None = None
        if slot.beat_roles and any(r == SampleRole.NEW for r in slot.beat_roles):
            allowed_types = SOUND_GROUP_TYPES[slot.sound_group]
            new_sample: str | None = None
            new_audio = None
            prev_drawn = sample_names[0]
            other = draw_next_sample_of_types(sample_queue, all_samples, allowed_types, exclude_name=prev_drawn)
            if other is not None:
                new_sample = other
                new_audio = mix_samples_into_stereo_clip(
                    [other], {other: slot.panning}, input_audio_dir, sample_rate,
                    volume_db, beat_length, prepared_cache=prepared_cache,
                )
                if extra_vol_db != 0.0:
                    new_audio = reduce_volume_by_db(new_audio, extra_vol_db)
                sample_usage_count[other] = sample_usage_count.get(other, 0) + 1
            per_beat_audio_list = [
                (new_audio if new_audio is not None else audio) if r == SampleRole.NEW else audio
                for r in slot.beat_roles
            ]
            # Expand sample_names / pan_assignments for the output filename.
            if new_sample is not None:
                pan_assignments = {**pan_assignments, new_sample: slot.panning}
                sample_names = [sample_names[0], new_sample]

        filename = build_output_filename(sample_names, pan_assignments, volume_db, created_count, conf['bpm_values'][bpm_idx], slot.rhythm)
        rhythmicized_path = (rhythmicized_output_dir / filename) if slot.rhythm else None
        _render_sem.acquire()
        render_executor.submit(
            _render_and_write,
            audio, sample_rate, beat_length, slot.rhythm, slot.beat_pannings,
            output_dir / filename, rhythmicized_path,
            _write_sem, write_executor, _render_sem,
            per_beat_audio_list,
        )

        if created_count % _print_interval == 0 or created_count == total_slots:
            print(f"  - Created {created_count}/{total_slots} samples...")

    render_executor.shutdown(wait=True)
    write_executor.shutdown(wait=True)

    SEP = '=' * 60
    num_silence_out = conf.get('num_silence_files', 0)
    if conf['silence_percent'] > 0 and num_silence_out > 0:
        generate_silence_files(output_dir, rhythmicized_output_dir, sample_rate, conf, starting_index=1)

    warn = f"\n⚠  Only created {created_count}/{conf['num_audio_samples']} audio samples — some slots skipped.\n" if created_count < conf['num_audio_samples'] else ""
    output_lines = [f"  - {created_count} samples"]
    if num_silence_out > 0:
        output_lines.append(f"  - {num_silence_out} silence")
        output_lines.append(f"  - = {created_count + num_silence_out} total output files")
    print(f"\n{warn}{SEP}\n  Output: → ./{output_dir}")
    for line in output_lines:
        print(line)
    print_panning_report(center_count, left_count, right_count, dualpan_count, hard_left_count, hard_right_count, conf)
    print_volume_report(volume_counts, created_count, conf)
    if resolved_bias:
        print_biased_sample_report(resolved_bias, sample_usage_count, group_targets)

    print_sound_group_report(group_appearances, non_strings_created, strings_created, conf, beat_multipliers, perm_multipliers, removed_counts, _raw_input_counts)


if __name__ == "__main__":
    main()
