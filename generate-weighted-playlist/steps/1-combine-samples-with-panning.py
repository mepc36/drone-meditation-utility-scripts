#!/usr/bin/env python3

import sys
import json
import random
import shutil
from collections import deque
from functools import reduce
from math import gcd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import soundfile as sf

import lib.config as cfg
from lib.audio_processing import apply_rhythm_pattern, load_audio, mix_samples_into_stereo_clip, reduce_volume_by_db, write_silence_file
from lib.deck_builder import SlotSpec, plan_output_files, build_permutation_kick_snare_deck
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
    DUALPAN_LEFTRIGHT, DUALPAN_DIAGONAL, UNTOUCHED,
    SOUND_GROUP_NAMES, SOUND_GROUP_TYPES,
    STRINGS, ACAPPELLA, KICKSNARE,
    PANNING_CENTER, PANNING_DIAGONAL, PANNING_LEFT, PANNING_RIGHT,
    PANNING_DUALPAN, PANNING_LEFT_OR_RIGHT,
    MUSICAL_PATTERNS, VOLUMES, BPMS,
    MUSIC_PATTERN_PERCENT,
    QUARTER_NOTE, QUARTER_NOTE_REST,
    BEAT_NAME_QUARTER_NOTE_REST, BEAT_NAME_QUARTER_NOTE,
    EIGHTH, SIXTEENTH, DOTTED_EIGHTH,
    BEAT_NAME_EIGHTH, BEAT_NAME_SIXTEENTH, BEAT_NAME_DOTTED_EIGHTH,
    MAX_DRAW_RETRIES,
    KICK_SNARE_PERMUTATION_MODE,
)
from lib.runtime_constants import LOUD, QUIET, SLOW, FAST


def clear_output_directory(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def panning_group_from_assignments(sample_names: list[str], pan_assignments: dict) -> str:
    if len(sample_names) == 2:
        return PANNING_DUALPAN
    pan = pan_assignments[sample_names[0]]
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
        if is_biased or combo_key not in seen_combinations:
            volume_db = volume_levels_db[loudest_idx if slot.volume_label == LOUD else quietest_idx]
            volume_idx = loudest_idx if slot.volume_label == LOUD else quietest_idx
            bpm_idx = slowest_idx if slot.bpm_label == SLOW else fastest_idx
            return sample_names, pan_assignments, volume_db, volume_idx, bpm_idx

    return None


def print_sample_usage_report(sample_usage_count: dict[str, int], all_samples: list[str]) -> None:
    usage_values = list(sample_usage_count.values())
    min_uses, max_uses = min(usage_values), max(usage_values)
    avg_uses = sum(usage_values) / len(usage_values)
    print(f"\nSample Usage Distribution (round-robin):")
    print(f"  Total input samples: {len(all_samples)}")
    print(f"  Min uses: {min_uses}  Max uses: {max_uses}  Avg: {avg_uses:.2f}")
    if max_uses - min_uses <= 1:
        print(f"  ✓ Perfectly even — all samples used {min_uses}–{max_uses} times")
    else:
        print(f"  ⚠  Spread of {max_uses - min_uses} ({sum(1 for c in usage_values if c == max_uses)} sample(s) at max)")


def print_panning_report(center: int, left: int, right: int, dualpan: int, hard_left: int, hard_right: int, conf: dict) -> None:
    diagonal = left + right
    leftorright = hard_left + hard_right
    total = center + diagonal + dualpan + leftorright
    print(f"\nPanning Distribution:")
    if total > 0:
        print(f"  center:    {center}  ({center/total*100:.1f}%)")
        print(f"  diagonal:  {diagonal}  ({diagonal/total*100:.1f}%)")
        print(f"  dualpan:   {dualpan}  ({dualpan/total*100:.1f}%)")
        print(f"  leftright: {leftorright}  ({leftorright/total*100:.1f}%)")
    if diagonal > 0:
        print(f"\nDiagonal Left/Right Distribution:")
        print(f"  Realized: {left}:{right} = {left/diagonal*100:.1f}% : {right/diagonal*100:.1f}%")
    if leftorright > 0:
        print(f"\nHard Left/Right Distribution:")
        print(f"  Realized: {hard_left}:{hard_right} = {hard_left/leftorright*100:.1f}% : {hard_right/leftorright*100:.1f}%")


def print_volume_report(volume_counts: list[int], total_created: int, conf: dict) -> None:
    volume_levels_db = conf['volume_levels_db']
    non_zero = [c for c in volume_counts if c > 0]
    vol_gcd = gcd_of(*non_zero) if len(non_zero) > 1 else (non_zero[0] if non_zero else 1)
    realized_ratio = ':'.join(str(c // vol_gcd) for c in volume_counts)
    print(f"\nVolume Distribution:")
    print(f"  Realized ratio: {realized_ratio}")
    for db_val, count in zip(volume_levels_db, volume_counts):
        pct = (count / total_created * 100) if total_created > 0 else 0
        print(f"    {db_val:+.1f} dB: {count} samples ({pct:.1f}%)")


def print_sound_group_report(group_appearances: dict[str, int], non_strings_created: int, conf: dict) -> None:
    print(f"\nSound Group Distribution:")
    print(f"  Config: kicksnare_stab_acappella_percents = {conf['raw'][cfg.CFG_SOUND_GROUP_PERCENTS]}")
    for group, target_pct in zip(SOUND_GROUP_NAMES, conf['sound_group_percents']):
        count = group_appearances[group]
        realized_pct = (count / non_strings_created * 100) if non_strings_created > 0 else 0
        print(f"  {group}: {count} files ({realized_pct:.1f}%, target {target_pct}%)")


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
    print(f"\nCombine Samples with Random Panning\n")
    print(f"BPMs: {bpms_str}")

    samples_by_type = load_samples_grouped_by_type(input_audio_dir)
    total_sample_count = sum(len(s) for s in samples_by_type.values())
    print(f"Found {total_sample_count} sample(s) in {len(samples_by_type)} sound type(s):")
    for sound_type, samples in sorted(samples_by_type.items()):
        print(f"  {sound_type}: {len(samples)} samples")
        for sample in sorted(samples):
            print(f"    - {sample}")
    print()

    clear_output_directory(output_dir)
    clear_output_directory(rhythmicized_output_dir)

    num_pass_through = sum(len(v) for k, v in samples_by_type.items() if passes_through_unmodified(k))
    if num_pass_through > conf['num_audio_samples']:
        raise ValueError(
            f"{num_pass_through} strings samples found but total audio slots is {conf['num_audio_samples']}. "
            f"Reduce the number of strings input files to at most {conf['num_audio_samples']}."
        )
    num_strings_samples = num_pass_through
    num_non_strings_samples = conf['num_audio_samples'] - num_strings_samples

    group_targets = {
        group: int(num_non_strings_samples * pct / 100)
        for group, pct in zip(SOUND_GROUP_NAMES, conf['sound_group_percents'])
    }
    group_total = sum(group_targets.values())
    if group_total < num_non_strings_samples:
        largest_group = max(group_targets, key=group_targets.get)
        group_targets[largest_group] += num_non_strings_samples - group_total

    # Derive panning quotas, bpm targets, and volume targets from sound_rules.
    # In permutation mode, kick/snare slots are handled separately and excluded here.
    panning_quotas = {PANNING_CENTER: 0, PANNING_DIAGONAL: 0, PANNING_DUALPAN: 0, PANNING_LEFT: 0, PANNING_RIGHT: 0}
    bpm_targets    = {conf['slowest_bpm_index']: 0, conf['fastest_bpm_index']: 0}
    vol_targets    = {conf['loudest_volume_index']: 0, conf['quietest_volume_index']: 0}
    single_bpm     = conf['slowest_bpm_index'] == conf['fastest_bpm_index']
    for group, count in group_targets.items():
        if KICK_SNARE_PERMUTATION_MODE and group == KICKSNARE:
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

    if KICK_SNARE_PERMUTATION_MODE:
        ks_deck = build_permutation_kick_snare_deck(samples_by_type)
        non_ks_targets = {g: t for g, t in group_targets.items() if g != KICKSNARE}
        non_ks_deck = plan_output_files(
            non_ks_targets, panning_quotas, bpm_targets, vol_targets,
            conf['slowest_bpm_index'], conf['fastest_bpm_index'],
            conf['loudest_volume_index'], conf['quietest_volume_index'],
        )
        non_strings_deck = ks_deck + non_ks_deck
    else:
        non_strings_deck = plan_output_files(
            group_targets, panning_quotas, bpm_targets, vol_targets,
            conf['slowest_bpm_index'], conf['fastest_bpm_index'],
            conf['loudest_volume_index'], conf['quietest_volume_index'],
        )
    strings_slots = [SlotSpec(STRINGS, UNTOUCHED, UNTOUCHED, UNTOUCHED, rhythm=(UNTOUCHED,))] * num_strings_samples
    full_deck = non_strings_deck + strings_slots
    random.shuffle(full_deck)

    sample_queue, all_samples = create_shuffled_sample_queue(samples_by_type)
    sample_usage_count: dict[str, int] = {s: 0 for s in all_samples}
    seen_combinations: set = set()

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

    created_count = strings_created = non_strings_created = 0
    group_appearances: dict[str, int] = {g: 0 for g in SOUND_GROUP_NAMES}
    center_count = left_count = right_count = dualpan_count = hard_left_count = hard_right_count = 0
    volume_counts = [0] * len(conf['volume_levels_db'])

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
        if forced_primary is None:
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
        audio = mix_samples_into_stereo_clip(sample_names, pan_assignments, input_audio_dir, sample_rate, volume_db, beat_length)
        if slot.sound_group == STRINGS and conf['strings_volume_reduction']:
            audio = reduce_volume_by_db(audio, -conf['strings_volume_reduction'])
        if slot.sound_group == ACAPPELLA and conf['acappella_volume_reduction']:
            audio = reduce_volume_by_db(audio, -conf['acappella_volume_reduction'])
        filename = build_output_filename(sample_names, pan_assignments, volume_db, created_count, conf['bpm_values'][bpm_idx], slot.rhythm)
        sf.write(output_dir / filename, audio, sample_rate)

        if slot.rhythm:
            if slot.rhythm == (UNTOUCHED,):
                sf.write(rhythmicized_output_dir / filename, audio, sample_rate)
            else:
                rhythmicized_audio = apply_rhythm_pattern(audio, sample_rate, beat_length, slot.rhythm, slot.beat_pannings)
                sf.write(rhythmicized_output_dir / filename, rhythmicized_audio, sample_rate)

        if created_count % 10 == 0 or created_count == conf['num_audio_samples']:
            print(f"  Created {created_count}/{conf['num_audio_samples']} samples...")

    if created_count < conf['num_audio_samples']:
        print(f"\nWarning: Only created {created_count} audio samples (expected {conf['num_audio_samples']}).")
        print("  Some deck slots were skipped because no unique (sample + pan) combination could be")
        print("  found within the retry limit. Try adding more input samples or adjusting the kicksnare percents.")
    else:
        print(f"\nComplete! Created {created_count} audio samples.")
    print(f"  Output: {output_dir.resolve()}")

    print_sample_usage_report(sample_usage_count, all_samples)
    print_panning_report(center_count, left_count, right_count, dualpan_count, hard_left_count, hard_right_count, conf)
    print_volume_report(volume_counts, created_count, conf)

    if conf['silence_percent'] > 0 and conf['num_silence_files'] > 0:
        generate_silence_files(output_dir, rhythmicized_output_dir, sample_rate, conf, starting_index=1)
        total = created_count + conf['num_silence_files']
        print(f"\nTotal files created: {total} ({created_count} samples + {conf['num_silence_files']} silence)")

    if num_strings_samples > 0:
        strings_pct = strings_created / created_count * 100 if created_count else 0
        non_strings_pct = non_strings_created / created_count * 100 if created_count else 0
        print(f"\nStrings Distribution:")
        print(f"  All {num_strings_samples} strings sample(s) added exactly once (no duplicates).")
        print(f"  Realized: {non_strings_created} non-strings ({non_strings_pct:.1f}%) / {strings_created} strings ({strings_pct:.1f}%)")

    print_sound_group_report(group_appearances, non_strings_created, conf)
    if resolved_bias:
        print_biased_sample_report(resolved_bias, sample_usage_count, group_targets)
    print(f"\nNext: Run 2-import-duplicate-padded-samples-into-itunes-playlist.py (builds playlist and plays via mpv)\n")


if __name__ == "__main__":
    main()
