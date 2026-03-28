import random
from collections import defaultdict, deque
from pathlib import Path

from .sound_rules import sound_type_of, passes_through_unmodified


def load_samples_grouped_by_type(input_audio_dir: Path) -> dict[str, list[str]]:
    if not input_audio_dir.exists():
        raise FileNotFoundError(f"Input audio directory not found: {input_audio_dir}")
    wav_files = list(input_audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {input_audio_dir}")
    grouped: dict[str, list[str]] = defaultdict(list)
    for f in wav_files:
        grouped[sound_type_of(f.stem)].append(f.stem)
    return dict(grouped)


def create_shuffled_sample_queue(samples_by_type: dict[str, list[str]]) -> tuple[deque, list[str]]:
    all_samples = [s for samples in samples_by_type.values() for s in samples]
    return deque(random.sample(all_samples, len(all_samples))), all_samples


def _refill_queue(queue: deque, all_samples: list[str]) -> None:
    shuffled = list(all_samples)
    random.shuffle(shuffled)
    queue.extend(shuffled)


def _draw_from_queue_with_filter(queue: deque, all_samples: list[str], predicate) -> str | None:
    if not any(predicate(s) for s in all_samples):
        return None
    for _ in range(2):
        if not queue:
            _refill_queue(queue, all_samples)
        _refill_queue(queue, all_samples)
        for i in range(len(queue)):
            if predicate(queue[i]):
                sample = queue[i]
                del queue[i]
                return sample
    return None


def draw_next_strings_sample(queue: deque, all_samples: list[str]) -> str | None:
    return _draw_from_queue_with_filter(
        queue, all_samples,
        lambda s: sound_type_of(s) == 'strings',
    )


def draw_next_non_strings_sample(queue: deque, all_samples: list[str]) -> str | None:
    return _draw_from_queue_with_filter(
        queue, all_samples,
        lambda s: not passes_through_unmodified(sound_type_of(s)),
    )


def draw_next_sample_of_types(
    queue: deque,
    all_samples: list[str],
    allowed_types: set[str],
    exclude_name: str | None = None,
) -> str | None:
    return _draw_from_queue_with_filter(
        queue, all_samples,
        lambda s: sound_type_of(s) in allowed_types and s != exclude_name,
    )
