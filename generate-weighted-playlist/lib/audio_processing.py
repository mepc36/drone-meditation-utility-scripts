from pathlib import Path

import numpy as np
import soundfile as sf

from .constants import ACAPPELLA, HARD_CENTER, HARD_LEFT, HARD_RIGHT, DUALPAN_LEFTRIGHT, DUALPAN_DIAGONAL
from .sound_rules import passes_through_unmodified, sound_type_of


_HARD_PAN_GAIN = np.sqrt(2)  # boosts hard-panned signals by +3 dB to match center power
_TARGET_RMS = 0.15
_CLIP_CEILING = 0.95


def load_audio(input_audio_dir: Path, sample_name: str) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(input_audio_dir / f"{sample_name}.wav", dtype='float32')
    return audio, sample_rate


def resample_to_rate(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate:
        return audio
    new_length = int(len(audio) / from_rate * to_rate)
    old_indices = np.linspace(0, len(audio) - 1, len(audio))
    new_indices = np.linspace(0, len(audio) - 1, new_length)
    if audio.ndim == 1:
        return np.interp(new_indices, old_indices, audio)
    resampled = np.zeros((new_length, audio.shape[1]))
    for ch in range(audio.shape[1]):
        resampled[:, ch] = np.interp(new_indices, old_indices, audio[:, ch])
    return resampled


def pan_to_stereo(audio: np.ndarray, pan_position) -> np.ndarray:
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if pan_position == HARD_CENTER:
        return np.column_stack([audio, audio])
    if pan_position == HARD_LEFT:
        return np.column_stack([audio * _HARD_PAN_GAIN, np.zeros_like(audio)])
    if pan_position == HARD_RIGHT:
        return np.column_stack([np.zeros_like(audio), audio * _HARD_PAN_GAIN])
    angle = (float(pan_position) + 1.0) * np.pi / 4.0
    return np.column_stack([audio * np.cos(angle) * _HARD_PAN_GAIN,
                             audio * np.sin(angle) * _HARD_PAN_GAIN])


def pad_or_trim_to_duration(audio: np.ndarray, sample_rate: int, target_seconds: float) -> np.ndarray:
    target_samples = int(target_seconds * sample_rate)
    if len(audio) >= target_samples:
        return audio[:target_samples]
    pad_shape = (target_samples - len(audio),) if audio.ndim == 1 else (target_samples - len(audio), audio.shape[1])
    return np.concatenate([audio, np.zeros(pad_shape)])


def normalize_loudness(audio: np.ndarray) -> np.ndarray:
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms == 0:
        return audio
    gain = _TARGET_RMS / current_rms
    peak = np.abs(audio * gain).max()
    if peak > _CLIP_CEILING:
        gain = _CLIP_CEILING / peak * gain
    return audio * gain


def reduce_volume_by_db(audio: np.ndarray, db: float) -> np.ndarray:
    return audio if db == 0 else audio * (10 ** (db / 20))


def mono_to_stereo_center(audio: np.ndarray) -> np.ndarray:
    return np.column_stack([audio, audio]) if audio.ndim == 1 else audio


def load_and_prepare_sample(
    sample_name: str,
    input_audio_dir: Path,
    target_sample_rate: int,
) -> np.ndarray:
    """Load, resample, and (where applicable) normalize one input sample.

    Returns the array ready to be panned/trimmed by mix_samples_into_stereo_clip.
    Safe to cache: the result is independent of panning, BPM, and rhythm.
    """
    audio, sr = load_audio(input_audio_dir, sample_name)
    audio = resample_to_rate(audio, sr, target_sample_rate)
    if passes_through_unmodified(sound_type_of(sample_name)):
        return mono_to_stereo_center(audio)
    if sound_type_of(sample_name) == ACAPPELLA:
        return audio  # acappella skips normalize_loudness
    return normalize_loudness(audio)


def mix_samples_into_stereo_clip(
    sample_names: list[str],
    pan_assignments: dict[str, str],
    input_audio_dir: Path,
    sample_rate: int,
    volume_db: float,
    beat_length_seconds: float,
    prepared_cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    prepared: dict[str, np.ndarray] = {}
    for name in sample_names:
        if prepared_cache is not None and name in prepared_cache:
            prepared[name] = prepared_cache[name]
        else:
            audio, sr = load_audio(input_audio_dir, name)
            audio = resample_to_rate(audio, sr, sample_rate)
            if passes_through_unmodified(sound_type_of(name)):
                prepared[name] = mono_to_stereo_center(audio)
            elif sound_type_of(name) == ACAPPELLA:
                prepared[name] = audio
            else:
                prepared[name] = normalize_loudness(audio)

    has_pass_through = any(passes_through_unmodified(sound_type_of(n)) for n in sample_names)

    mixed = None
    for name in sample_names:
        audio = prepared[name]
        if passes_through_unmodified(sound_type_of(name)):
            stereo = audio  # already mono_to_stereo'd by load_and_prepare_sample
        else:
            stereo = pad_or_trim_to_duration(
                pan_to_stereo(audio, pan_assignments[name]),
                sample_rate,
                beat_length_seconds,
            )
        mixed = stereo if mixed is None else mixed + stereo

    if not has_pass_through:
        mixed = reduce_volume_by_db(normalize_loudness(mixed), volume_db)

    return mixed


def apply_rhythm_pattern(
    audio: np.ndarray,
    sample_rate: int,
    beat_length_seconds: float,
    pattern: tuple[float, ...],
    beat_pannings: tuple[str, ...] = (),
) -> np.ndarray:
    """Chop audio into rhythmic segments and concatenate into a new clip.

    Each value in `pattern` is the total slot duration in beats. Audio fills
    min(value, 1.0) beats from the start of the clip; any remaining duration
    is silence. Examples:
        0    → 1 beat of pure silence (no audio)
        0.5  → 0.5 beats of audio (eighth note), no silence
        1.0  → 1 beat of audio (quarter note), no silence
        2.0  → 1 beat of audio + 1 beat of silence

    When `beat_pannings` is provided (parallel to `pattern`), beats with a
    non-empty panning string are re-panned from mono for that beat chunk so
    that successive beats can occupy different stereo positions.  Beats with
    an empty-string panning inherit the slot-level panning already baked into
    `audio`.  All output chunks are stereo when any beat has an explicit panning.
    """
    n_channels = audio.shape[1] if audio.ndim == 2 else 1
    output_stereo = bool(beat_pannings) and any(p for p in beat_pannings)

    # Pre-compute a mono version once for beats that need re-panning.
    mono_audio = np.mean(audio, axis=1) if (output_stereo and audio.ndim == 2) else audio

    def _silence(n_samples: int) -> np.ndarray:
        if output_stereo:
            return np.zeros((n_samples, 2))
        return np.zeros((n_samples, n_channels) if n_channels > 1 else (n_samples,))

    chunks = []
    for i, duration_beats in enumerate(pattern):
        beat_pan = beat_pannings[i] if i < len(beat_pannings) else ''

        if duration_beats == 0:
            chunks.append(_silence(int(beat_length_seconds * sample_rate)))
            continue

        sound_beats = min(duration_beats, 1.0)
        silence_beats = duration_beats - sound_beats
        sound_samples = int(sound_beats * beat_length_seconds * sample_rate)

        if beat_pan and beat_pan not in (DUALPAN_LEFTRIGHT, DUALPAN_DIAGONAL):
            # Re-pan this beat from mono → stereo using the specified position.
            src = mono_audio
            chunk = src[:sound_samples]
            if len(chunk) < sound_samples:
                chunk = np.concatenate([chunk, np.zeros(sound_samples - len(chunk))])
            chunk = pan_to_stereo(chunk, beat_pan)
        else:
            # Use the audio as-is (slot-level panning already applied, or dualpan
            # which is already mixed as hard-left + hard-right stereo).
            chunk = audio[:sound_samples]
            if len(chunk) < sound_samples:
                chunk = np.concatenate([chunk, _silence(sound_samples - len(chunk))])
            if output_stereo and chunk.ndim == 1:
                chunk = np.column_stack([chunk, chunk])

        chunks.append(chunk)

        if silence_beats > 0:
            chunks.append(_silence(int(silence_beats * beat_length_seconds * sample_rate)))

    return np.concatenate(chunks)


def write_silence_file(output_dir: Path, sample_rate: int, length_seconds: float, index: int) -> None:
    length_ms = int(length_seconds * 1000)
    filename = f"silence_{length_ms}ms_{index:03d}.wav"
    sf.write(output_dir / filename, np.zeros((int(length_seconds * sample_rate), 2)), sample_rate)
