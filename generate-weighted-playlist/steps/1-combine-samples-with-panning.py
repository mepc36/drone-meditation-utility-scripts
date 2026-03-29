#!/usr/bin/env python3

import sys
import random
import shutil
from collections import deque
from functools import reduce
from math import gcd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import soundfile as sf

import lib.config as cfg
from lib.audio_processing import apply_rhythm_pattern, load_audio, mix_samples_into_stereo_clip, write_silence_file
from lib.deck_builder import SlotSpec, plan_output_files
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
)
from lib.constants import (
    HARD_CENTER, HARD_LEFT, HARD_RIGHT, DIAGONAL_LEFT, DIAGONAL_RIGHT,
    DUALPAN, UNTOUCHED,
    LOUD, SLOW,
    SOUND_GROUP_NAMES, SOUND_GROUP_TYPES,
    STRINGS,
)


def clear_output_directory(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def panning_group_from_assignments(sample_names: list[str], pan_assignments: dict) -> str:
    if len(sample_names) == 2:
        return 'dualpan'
    pan = pan_assignments[sample_names[0]]
    if pan == HARD_CENTER:
        return 'center'
    if pan in (HARD_LEFT, HARD_RIGHT):
        return 'leftorright'
    return 'diagonal'


def pan_numeric_value(pan: float) -> float:
    return float(pan)


BEAT_NAMES: dict[float, str] = {
    0.0:  'quarternoterest',
    0.25: 'sixteenth',
    0.5:  'eighth',
    1.0:  'quarter',
    2.0:  'half',
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
    ordered_by_pan = sorted(sample_names, key=lambda n: pan_numeric_value(pan_assignments[n]))
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

    for _ in range(20):
        primary = draw_next_sample_of_types(sample_queue, all_samples, SOUND_GROUP_TYPES[slot.sound_group])
        if primary is None:
            return None

        rule = rules_by_sound_type.get(sound_type_of(primary))

        if slot.panning == DUALPAN:
            partner_types = set(rule['dualpan_partners']) if rule else {sound_type_of(primary)}
            partner = draw_next_sample_of_types(sample_queue, all_samples, partner_types, exclude_name=primary)
            if partner:
                sample_names = [primary, partner]
                pan_assignments = {primary: HARD_LEFT, partner: HARD_RIGHT}
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
            sample_names, pan_assignments = [primary], {primary: HARD_CENTER}

        combo_key = tuple(sorted(f"{n}:{pan_assignments[n]}" for n in sample_names))
        if combo_key not in seen_combinations:
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
    counts = [c for c in [center, diagonal, dualpan, leftorright] if c > 0]
    ratio_gcd = gcd_of(*counts) if len(counts) > 1 else 1
    realized_parts = [center // ratio_gcd, diagonal // ratio_gcd, dualpan // ratio_gcd, leftorright // ratio_gcd]
    realized_ratio = ':'.join(map(str, realized_parts))
    config_ratio_str = conf['raw'].get('center_diagonal_dualpan_left_right_percents', '25:25:25:13:12')
    config_parts = [int(x) for x in config_ratio_str.split(':')]
    # Combine left+right into a single leftorright bucket to compare against realized totals
    config_parts_combined = config_parts[:3] + [config_parts[3] + config_parts[4]]
    scale = realized_parts[0] / config_parts_combined[0] if config_parts_combined[0] != 0 else 1
    perfect_parts = [int(x * scale) for x in config_parts_combined]
    differential = ':'.join(f"{'+' if d > 0 else ''}{d}" for d in [perfect_parts[i] - realized_parts[i] for i in range(4)])
    print(f"\nPanning Distribution:")
    print(f"  Config ratio: {config_ratio_str}")
    print(f"  Realized ratio: {realized_ratio}")
    print(f"  Perfect ratio: {':'.join(map(str, perfect_parts))}")
    print(f"  Differential: {differential}")
    if diagonal > 0:
        print(f"\nDiagonal Left/Right Distribution:")
        print(f"  Target: 50.0% left : 50.0% right")
        print(f"  Realized: {left}:{right} = {left/diagonal*100:.1f}% : {right/diagonal*100:.1f}%")
        print(f"  Differential: {left - right:+d} (left - right)")
    if leftorright > 0:
        left_w = conf['left_weight']
        right_w = conf['right_weight']
        lr_total = left_w + right_w
        left_pct_target = left_w / lr_total * 100 if lr_total > 0 else 50.0
        right_pct_target = right_w / lr_total * 100 if lr_total > 0 else 50.0
        print(f"\nHard Left/Right Distribution:")
        print(f"  Target: {left_pct_target:.1f}% hard left : {right_pct_target:.1f}% hard right")
        print(f"  Realized: {hard_left}:{hard_right} = {hard_left/leftorright*100:.1f}% : {hard_right/leftorright*100:.1f}%")
        print(f"  Differential: {hard_left - hard_right:+d} (hard left - hard right)")


def print_volume_report(volume_counts: list[int], total_created: int, conf: dict) -> None:
    volume_levels_db = conf['volume_levels_db']
    non_zero = [c for c in volume_counts if c > 0]
    vol_gcd = gcd_of(*non_zero) if len(non_zero) > 1 else (non_zero[0] if non_zero else 1)
    realized_ratio = ':'.join(str(c // vol_gcd) for c in volume_counts)
    print(f"\nVolume Distribution:")
    print(f"  Config ratio: {conf['raw'].get('loud_quiet_percents', '50:50')}")
    print(f"  Config values: {conf['raw'].get('loud_quiet_values', '0:-26')} dB")
    print(f"  Realized ratio: {realized_ratio}")
    for db_val, count in zip(volume_levels_db, volume_counts):
        pct = (count / total_created * 100) if total_created > 0 else 0
        print(f"    {db_val:+.1f} dB: {count} samples ({pct:.1f}%)")


def print_sound_group_report(group_appearances: dict[str, int], non_strings_created: int, conf: dict) -> None:
    print(f"\nSound Group Distribution:")
    print(f"  Config: kicksnare_stab_acappella_percents = {conf['raw']['kicksnare_stab_acappella_percents']}")
    for group, target_pct in zip(SOUND_GROUP_NAMES, conf['sound_group_percents']):
        count = group_appearances[group]
        realized_pct = (count / non_strings_created * 100) if non_strings_created > 0 else 0
        print(f"  {group}: {count} files ({realized_pct:.1f}%, target {target_pct}%)")


def generate_silence_files(
    output_dir: Path,
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
            file_index += 1
            if (file_index - starting_index) % 10 == 1:
                print(f"    Created {file_index - starting_index}/{num_silence} silence files...", end="\r")
    print(f"    Created {num_silence}/{num_silence} silence files...")


def main() -> None:
    conf = cfg.load()
    output_dir = cfg.OUTPUT_AUDIO_DIR
    input_audio_dir = cfg.INPUT_AUDIO_DIR
    rhythmicize = conf['rhythmicize_output_samples']
    rhythmicized_output_dir = cfg.OUTPUT_RHYTHMICIZED_AUDIO_DIR

    bpms_str = ':'.join(str(b) for b in conf['bpm_values'])
    percents_str = ':'.join(str(p) for p in conf['bpm_percents'])
    print(f"\nCombine Samples with Random Panning\n")
    print(f"BPMs: {bpms_str}  (slow_to_fast_bpm_percents: {percents_str})")

    samples_by_type = load_samples_grouped_by_type(input_audio_dir)
    total_sample_count = sum(len(s) for s in samples_by_type.values())
    print(f"Found {total_sample_count} sample(s) in {len(samples_by_type)} sound type(s):")
    for sound_type, samples in sorted(samples_by_type.items()):
        print(f"  {sound_type}: {len(samples)} samples")
        for sample in sorted(samples):
            print(f"    - {sample}")
    print()

    clear_output_directory(output_dir)
    if rhythmicize:
        clear_output_directory(rhythmicized_output_dir)

    num_pass_through = sum(len(v) for k, v in samples_by_type.items() if passes_through_unmodified(k))
    if num_pass_through > conf['num_unique_samples']:
        raise ValueError(
            f"{num_pass_through} strings samples found but num_unique_samples is {conf['num_unique_samples']}. "
            f"Increase num_unique_samples to at least {num_pass_through}."
        )
    num_strings_samples = num_pass_through
    num_non_strings_samples = conf['num_audio_samples'] - num_strings_samples

    total_non_strings_input = sum(len(v) for k, v in samples_by_type.items() if not passes_through_unmodified(k))
    center_quota  = int(num_non_strings_samples * conf['center_weight']     / 100)
    diagonal_quota = int(num_non_strings_samples * conf['diagonal_weight']   / 100)
    dualpan_quota  = int(num_non_strings_samples * conf['dualpan_weight']    / 100)
    left_quota     = int(num_non_strings_samples * conf['left_weight']       / 100)
    right_quota    = int(num_non_strings_samples * conf['right_weight']      / 100)

    if center_quota > total_non_strings_input:
        center_overflow = center_quota - total_non_strings_input
        center_quota = total_non_strings_input
        non_center_weight = conf['diagonal_weight'] + conf['dualpan_weight'] + conf['left_weight'] + conf['right_weight']
        if non_center_weight > 0:
            diagonal_quota += int(center_overflow * conf['diagonal_weight'] / non_center_weight)
            left_quota     += int(center_overflow * conf['left_weight']     / non_center_weight)
            right_quota    += int(center_overflow * conf['right_weight']    / non_center_weight)
            dualpan_quota  += center_overflow - int(center_overflow * conf['diagonal_weight'] / non_center_weight) - int(center_overflow * conf['left_weight'] / non_center_weight) - int(center_overflow * conf['right_weight'] / non_center_weight)
        else:
            dualpan_quota += center_overflow
        print(f"⚠️  Center quota capped at {total_non_strings_input}. Overflow redistributed to diagonal/dualpan/left/right.\n")

    rounding_remainder = num_non_strings_samples - (center_quota + diagonal_quota + dualpan_quota + left_quota + right_quota)
    if rounding_remainder > 0:
        heaviest_panning = max(
            [('center', conf['center_weight']), ('diagonal', conf['diagonal_weight']),
             ('dualpan', conf['dualpan_weight']), ('left', conf['left_weight']), ('right', conf['right_weight'])],
            key=lambda x: x[1],
        )[0]
        if heaviest_panning == 'center':       center_quota   += rounding_remainder
        elif heaviest_panning == 'diagonal':   diagonal_quota += rounding_remainder
        elif heaviest_panning == 'dualpan':    dualpan_quota  += rounding_remainder
        elif heaviest_panning == 'left':       left_quota     += rounding_remainder
        else:                                  right_quota    += rounding_remainder

    group_targets = {
        group: int(num_non_strings_samples * pct / 100)
        for group, pct in zip(SOUND_GROUP_NAMES, conf['sound_group_percents'])
    }
    group_total = sum(group_targets.values())
    if group_total < num_non_strings_samples:
        largest_group = max(group_targets, key=group_targets.get)
        group_targets[largest_group] += num_non_strings_samples - group_total

    bpm_targets = {idx: int(num_non_strings_samples * pct / 100) for idx, pct in enumerate(conf['bpm_percents'])}
    vol_targets = {idx: int(num_non_strings_samples * pct / 100) for idx, pct in enumerate(conf['volume_percents'])}
    panning_quotas = {'center': center_quota, 'diagonal': diagonal_quota, 'dualpan': dualpan_quota, 'left': left_quota, 'right': right_quota}

    non_strings_deck = plan_output_files(
        group_targets, panning_quotas, bpm_targets, vol_targets,
        conf['slowest_bpm_index'], conf['fastest_bpm_index'],
        conf['loudest_volume_index'], conf['quietest_volume_index'],
    )
    strings_slots = [SlotSpec(STRINGS, UNTOUCHED, UNTOUCHED, UNTOUCHED)] * num_strings_samples
    full_deck = non_strings_deck + strings_slots
    random.shuffle(full_deck)

    sample_queue, all_samples = create_shuffled_sample_queue(samples_by_type)
    sample_usage_count: dict[str, int] = {s: 0 for s in all_samples}
    seen_combinations: set = set()

    first_sample = list(samples_by_type.values())[0][0]
    _, sample_rate = load_audio(input_audio_dir, first_sample)

    created_count = strings_created = non_strings_created = 0
    group_appearances: dict[str, int] = {g: 0 for g in SOUND_GROUP_NAMES}
    center_count = left_count = right_count = dualpan_count = hard_left_count = hard_right_count = 0
    volume_counts = [0] * len(conf['volume_levels_db'])

    for slot in full_deck:
        resolved = resolve_slot(slot, sample_queue, all_samples, seen_combinations, conf)
        if resolved is None:
            continue
        sample_names, pan_assignments, volume_db, volume_idx, bpm_idx = resolved

        combo_key = tuple(sorted(f"{n}:{pan_assignments[n]}" for n in sample_names))
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
            if pan == HARD_CENTER:           center_count    += 1
            elif pan == HARD_LEFT:      hard_left_count += 1
            elif pan == HARD_RIGHT:     hard_right_count += 1
            elif float(pan) < 0:        left_count      += 1
            else:                       right_count     += 1

        if slot.sound_group != STRINGS:
            volume_counts[volume_idx] += 1

        beat_length = conf['beat_lengths_seconds'][bpm_idx]
        audio = mix_samples_into_stereo_clip(sample_names, pan_assignments, input_audio_dir, sample_rate, volume_db, beat_length)
        filename = build_output_filename(sample_names, pan_assignments, volume_db, created_count, conf['bpm_values'][bpm_idx], slot.rhythm)
        sf.write(output_dir / filename, audio, sample_rate)

        if rhythmicize and slot.rhythm:
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
        print("  found within the retry limit. Try adding more input samples or reducing num_unique_samples.")
    else:
        print(f"\nComplete! Created {created_count} audio samples.")
    print(f"  Output: {output_dir.resolve()}")

    print_sample_usage_report(sample_usage_count, all_samples)
    print_panning_report(center_count, left_count, right_count, dualpan_count, hard_left_count, hard_right_count, conf)
    print_volume_report(volume_counts, created_count, conf)

    if conf['silence_percent'] > 0 and conf['num_silence_files'] > 0:
        generate_silence_files(output_dir, sample_rate, conf, starting_index=1)
        total = created_count + conf['num_silence_files']
        print(f"\nTotal files created: {total} ({created_count} samples + {conf['num_silence_files']} silence)")

    if num_strings_samples > 0:
        strings_pct = strings_created / created_count * 100 if created_count else 0
        non_strings_pct = non_strings_created / created_count * 100 if created_count else 0
        print(f"\nStrings Distribution:")
        print(f"  All {num_strings_samples} strings sample(s) added exactly once (no duplicates).")
        print(f"  Realized: {non_strings_created} non-strings ({non_strings_pct:.1f}%) / {strings_created} strings ({strings_pct:.1f}%)")

    print_sound_group_report(group_appearances, non_strings_created, conf)
    print(f"\nNext: Run 2-import-duplicate-padded-samples-into-itunes-playlist.py\n")


if __name__ == "__main__":
    main()
