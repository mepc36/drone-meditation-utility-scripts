#!/usr/bin/env python3
"""
Tinnitus Pitch Finder (macOS, afplay)
-------------------------------------
Interactive binary-search tool to match your tinnitus pitch.

Features:
- You choose LEFT or RIGHT ear (tone is hard-panned there).
- Adjust HIGHER/LOWER until the tone matches your tinnitus.
- Press 'm' when it matches to save a 3-second tone file.

Output:
  ./output/matched_tinnitus_<frequency>Hz.wav

Controls:
  h = higher (tone is lower than tinnitus)
  l = lower  (tone is higher than tinnitus)
  r = replay
  m = match now
  s = swap ear
  q = quit
"""

import os, sys, math, subprocess, tempfile
from dataclasses import dataclass
import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass
class Config:
    sample_rate_hz: int = 96_000
    tone_duration_seconds: float = 1.0
    fade_time_seconds: float = 0.01
    amplitude: float = 0.25
    min_frequency_hz: float = 2000.0
    max_frequency_hz: float = 16000.0
    start_frequency_hz: float = 8000.0
    stop_resolution_hz: float = 5.0
    randomize_phase: bool = False


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def generate_sine_wave(frequency_hz: float, cfg: Config, duration_seconds=None) -> np.ndarray:
    """Generate a mono sine wave with fades."""
    duration = duration_seconds if duration_seconds else cfg.tone_duration_seconds
    num_samples = int(cfg.sample_rate_hz * duration)
    time_axis = np.linspace(0, duration, num_samples, endpoint=False)
    waveform = np.sin(2 * np.pi * frequency_hz * time_axis)

    fade_sample_count = int(cfg.sample_rate_hz * cfg.fade_time_seconds)
    if fade_sample_count > 0:
        fade_in = np.linspace(0, 1, fade_sample_count)
        fade_out = np.linspace(1, 0, fade_sample_count)
        waveform[:fade_sample_count] *= fade_in
        waveform[-fade_sample_count:] *= fade_out

    waveform = (cfg.amplitude * waveform / max(1e-9, np.max(np.abs(waveform)))).astype(np.float32)
    return waveform


def apply_hard_pan(mono_waveform: np.ndarray, ear_side: str) -> np.ndarray:
    """Return stereo waveform panned hard left or right."""
    left_channel = mono_waveform if ear_side == 'l' else np.zeros_like(mono_waveform)
    right_channel = mono_waveform if ear_side == 'r' else np.zeros_like(mono_waveform)
    return np.column_stack([left_channel, right_channel])


def play_audio(stereo_waveform: np.ndarray, sample_rate: int):
    """Play a stereo waveform via macOS 'afplay'."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        sf.write(temp_file.name, stereo_waveform, sample_rate)
        temp_path = temp_file.name
    try:
        subprocess.run(["afplay", temp_path], check=True)
    except Exception as e:
        print(f"Playback error: {e}", file=sys.stderr)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def prompt_for_ear() -> str:
    """Ask user which ear to test first."""
    while True:
        answer = input("Which ear first? [L/R]: ").strip().lower()
        if answer in ("l", "r"):
            return answer
        print("Please type L or R.")


def prompt_for_action() -> str:
    """Ask user what to do next."""
    while True:
        answer = input("Action: h=higher, l=lower, r=replay, m=match, s=swap ear, q=quit → ").strip().lower()
        if answer in {"h", "l", "r", "m", "s", "q"}:
            return answer
        print("Please type h/l/r/m/s/q.")


def describe_pitch(frequency_hz: float, concert_a_hz: float = 440.0):
    """Return closest pitch name and deviation in cents."""
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    semitones_from_a4 = 12 * math.log2(frequency_hz / concert_a_hz)
    midi_exact = 69 + semitones_from_a4
    midi_nearest = round(midi_exact)
    cents_deviation = (midi_exact - midi_nearest) * 100
    note_name = note_names[midi_nearest % 12]
    octave = (midi_nearest // 12) - 1
    exact_pitch_hz = concert_a_hz * (2 ** ((midi_nearest - 69) / 12))
    return note_name, octave, cents_deviation, exact_pitch_hz


# ---------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------
def main():
    cfg = Config()

    print("\n=== Tinnitus Pitch Finder ===")
    print("Start with LOW volume. Use headphones.\n")
    print(f"Range: {cfg.min_frequency_hz:.0f}–{cfg.max_frequency_hz:.0f} Hz | Start: {cfg.start_frequency_hz:.0f} Hz | Stop @ ±{cfg.stop_resolution_hz:.0f} Hz")

    ear_side = prompt_for_ear()

    lower_bound_hz = cfg.min_frequency_hz
    upper_bound_hz = cfg.max_frequency_hz
    current_frequency_hz = max(lower_bound_hz, min(cfg.start_frequency_hz, upper_bound_hz))

    print("\nInstructions:")
    print("  If the tone is LOWER than your tinnitus → press 'h' (go higher).")
    print("  If the tone is HIGHER than your tinnitus → press 'l' (go lower).")
    print("  Press 'm' when it MATCHES. 's' swaps ear. 'r' replays. 'q' quits.\n")

    while True:
        mono_tone = generate_sine_wave(current_frequency_hz, cfg)
        stereo_tone = apply_hard_pan(mono_tone, ear_side)
        print(f"\nPlaying {current_frequency_hz:.1f} Hz ({'Left' if ear_side == 'l' else 'Right'} ear)")
        play_audio(stereo_tone, cfg.sample_rate_hz)

        # Stop automatically if frequency range is narrow enough
        if (upper_bound_hz - lower_bound_hz) <= cfg.stop_resolution_hz:
            print(f"Search window ≤ {cfg.stop_resolution_hz} Hz; auto-stopping.")
            user_action = 'm'
        else:
            user_action = prompt_for_action()

        if user_action == 'q':
            break
        elif user_action == 'r':
            continue
        elif user_action == 's':
            ear_side = 'r' if ear_side == 'l' else 'l'
            continue
        elif user_action == 'm':
            matched_frequency_hz = current_frequency_hz
            note, octave, cents, exact_hz = describe_pitch(matched_frequency_hz)

            print("\n≈≈≈ Result ≈≈≈")
            print(f"Matched frequency: {matched_frequency_hz:.1f} Hz")
            print(f"Nearest pitch: {note}{octave} ({exact_hz:.1f} Hz), deviation {cents:+.0f} cents")

            os.makedirs("./output", exist_ok=True)
            output_path = f"./output/matched_tinnitus_{matched_frequency_hz:.1f}Hz.wav"
            matched_tone = generate_sine_wave(matched_frequency_hz, cfg, duration_seconds=3.0)
            sf.write(output_path, matched_tone, cfg.sample_rate_hz)
            print(f"\n✅ Saved 3-second tone to: {output_path}\n")
            return
        elif user_action == 'h':
            lower_bound_hz = max(lower_bound_hz, current_frequency_hz)
        elif user_action == 'l':
            upper_bound_hz = min(upper_bound_hz, current_frequency_hz)

        current_frequency_hz = (lower_bound_hz + upper_bound_hz) / 2.0
        current_frequency_hz = max(lower_bound_hz, min(current_frequency_hz, upper_bound_hz))


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
