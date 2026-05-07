#!/usr/bin/env python3

import os
import sys
import random
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Semaphore

sys.path.insert(0, str(Path(__file__).parent.parent))

import soundfile as sf
import numpy as np

import lib.config as cfg
from lib.audio_processing import apply_rhythm_pattern, load_audio, load_and_prepare_sample, mix_samples_into_stereo_clip, write_silence_file
from lib.deck_builder import SlotSpec, plan_output_files, compute_group_beat_multipliers
from lib.sample_queue import (
    create_shuffled_sample_queue,
    draw_next_sample_of_types,
    draw_next_strings_sample,
    load_samples_grouped_by_type,
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
    BEAT_NAMES,
    MAX_DRAW_RETRIES,
)
from lib.reporting import print_panning_report, print_volume_report, print_sound_group_report
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


def resolve_slot(
    slot: SlotSpec,
    sample_queue: deque,
    all_samples: list[str],
    seen_combinations: set,
    conf: dict,
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

    for _ in range(MAX_DRAW_RETRIES):
        primary = draw_next_sample_of_types(
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
        if slot.sound_group != KICKSNARE or combo_key not in seen_combinations:
            volume_db = volume_levels_db[loudest_idx if slot.volume_label == LOUD else quietest_idx]
            volume_idx = loudest_idx if slot.volume_label == LOUD else quietest_idx
            bpm_idx = slowest_idx if slot.bpm_label == SLOW else fastest_idx
            return sample_names, pan_assignments, volume_db, volume_idx, bpm_idx

    return None




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

    num_pass_through = sum(len(v) for k, v in samples_by_type.items() if passes_through_unmodified(k))
    num_strings_samples = num_pass_through
    if num_strings_samples > conf['num_audio_samples']:
        raise ValueError(
            f"{num_strings_samples} strings slots exceeds total audio slots {conf['num_audio_samples']}. "
            f"Reduce the number of strings input files."
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
    # Recompute silence count based on beat-anchored total.
    _audio_total = sum(group_targets.values()) + num_strings_samples
    conf['num_silence_files'] = (
        round(_audio_total * conf['silence_percent'] / conf['samples_percent'])
        if conf['samples_percent'] > 0 and conf['silence_percent'] > 0
        else 0
    )

    # Derive panning quotas, bpm targets, and volume targets from sound_rules.
    panning_quotas = {PANNING_CENTER: 0, PANNING_DIAGONAL: 0, PANNING_DUALPAN: 0, PANNING_LEFT: 0, PANNING_RIGHT: 0}
    bpm_targets    = {conf['slowest_bpm_index']: 0, conf['fastest_bpm_index']: 0}
    vol_targets    = {conf['loudest_volume_index']: 0, conf['quietest_volume_index']: 0}
    single_bpm     = conf['slowest_bpm_index'] == conf['fastest_bpm_index']
    for group, count in group_targets.items():
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

    sample_queue, all_samples = create_shuffled_sample_queue(samples_by_type)
    seen_combinations: set = set()

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
        resolved = resolve_slot(slot, sample_queue, all_samples, seen_combinations, conf)
        if resolved is None:
            continue
        sample_names, pan_assignments, volume_db, volume_idx, bpm_idx = resolved

        combo_key = tuple(sorted(f"{n}:{pan_assignments[n]}" for n in sample_names))
        if slot.sound_group == KICKSNARE:
            seen_combinations.add(combo_key)

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

        audio = mix_samples_into_stereo_clip(
            sample_names, pan_assignments, input_audio_dir, sample_rate, volume_db, beat_length,
            prepared_cache=prepared_cache,
        )

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
    print_sound_group_report(group_appearances, non_strings_created, strings_created, conf, beat_multipliers, _raw_input_counts)


if __name__ == "__main__":
    main()
