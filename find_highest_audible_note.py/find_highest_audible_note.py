#!/usr/bin/env python3
"""
High-Frequency Hearing Test (macOS)
-----------------------------------
Finds your highest audible pure tone starting at 16 kHz using an adaptive (binary search) method.
Uses only numpy + soundfile (same as your other scripts) and macOS `afplay` for playback.

UPDATES
- Adds a random pre-play delay between 0.00 and 2.00 seconds before each tone.
- Tone duration increased to 1.25 seconds.

USAGE
  python hearing_threshold_high_freq.py
  (Headphones recommended. Start with LOW volume.)

Controls during the test:
  y  = I heard it
  n  = I did NOT hear it
  r  = replay the same tone
  q  = quit

Notes
- Default sample rate: 96 kHz (Nyquist 48 kHz) to minimize aliasing.
- Search range: 8 kHz – 20 kHz (you can adjust below).
- Keep volume modest to protect your hearing.
"""

# VOLUME WARNING - This script generates audio that can damage hearing
print("⚠️  WARNING: This script generates high-frequency tones that can cause hearing damage.")
print("   Loud volumes may cause permanent hearing loss, tinnitus, or ear pain.")
print("   Turn your volume DOWN to the lowest audible level before continuing.")
input("   Press Enter when volume is set safely low...")
print()

import os
import sys
import math
import time
import random
import subprocess
from dataclasses import dataclass

import numpy as np
import soundfile as sf


@dataclass
class Config:
    sample_rate_hz: int = 96_000
    duration_seconds: float = 1.25     # updated to 1.25 s
    fade_time_seconds: float = 0.01
    amplitude: float = 0.3             # 0..1; keep modest
    lower_bound_hz: float = 8_000.0    # min test frequency
    upper_bound_hz: float = 20_000.0   # max test frequency
    start_frequency_hz: float = 16_000.0
    resolution_hz: float = 25.0        # stop when range width <= this
    pre_play_delay_max_s: float = 2.0  # random delay in [0, max] seconds


def generate_tone(frequency_hz: float, cfg: Config) -> np.ndarray:
    """Create a single-channel sine wave with fades."""
    sample_count = int(cfg.sample_rate_hz * cfg.duration_seconds)
    t = np.linspace(0.0, cfg.duration_seconds, sample_count, endpoint=False, dtype=np.float64)
    # Randomize initial phase slightly to avoid consistent artifacts
    phase0 = 2.0 * np.pi * np.random.rand()
    wave = np.sin(2.0 * np.pi * frequency_hz * t + phase0)
    # Apply fade in/out
    fade_samples = int(cfg.sample_rate_hz * cfg.fade_time_seconds)
    if fade_samples > 0:
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float64)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float64)
        wave[:fade_samples] *= fade_in
        wave[-fade_samples:] *= fade_out
    # Normalize to amplitude
    wave = (cfg.amplitude * wave / max(1e-9, np.max(np.abs(wave)))).astype(np.float32)
    return wave


def play_tone_mac(tmp_wav_path: str, pre_delay_s: float):
    """Play WAV via macOS 'afplay' after a randomized delay. Blocks until done."""
    if pre_delay_s > 0:
        time.sleep(pre_delay_s)
    try:
        subprocess.run(["afplay", tmp_wav_path], check=True)
    except FileNotFoundError:
        print("Error: 'afplay' not found. On macOS it should be available by default.", file=sys.stderr)
    except subprocess.CalledProcessError:
        print("Playback error with afplay.", file=sys.stderr)


def prompt_user(prompt: str) -> str:
    """Get a sanitized single-letter response."""
    while True:
        ans = input(prompt).strip().lower()
        if ans in {"y", "n", "r", "q"}:
            return ans
        print("Please type y (heard), n (not heard), r (replay), or q (quit).")


def frequency_to_pitch(frequency_hz: float, concert_a_hz: float = 440.0) -> tuple[str, int, float]:
    """
    Convert frequency to 12-tone equal temperament pitch.
    Returns (note_name, octave, cents_deviation)
    """
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
    # Calculate semitones from A4 (440 Hz)
    semitones_from_a4 = 12 * math.log2(frequency_hz / concert_a_hz)
    
    # A4 is MIDI note 69, which is octave 4, note index 9 (A)
    midi_note_exact = 69 + semitones_from_a4
    midi_note_rounded = round(midi_note_exact)
    cents_deviation = (midi_note_exact - midi_note_rounded) * 100
    
    # Convert MIDI note to note name and octave
    note_index = midi_note_rounded % 12
    octave = (midi_note_rounded // 12) - 1
    note_name = note_names[note_index]
    
    return note_name, octave, cents_deviation


def main():
    cfg = Config()

    print("\n=== High-Frequency Hearing Test (pure sine) ===")
    print("Headphones recommended. Start with LOW volume to protect your ears.")
    print(f"Sample rate: {cfg.sample_rate_hz} Hz | Duration: {cfg.duration_seconds:.2f}s | Start: {cfg.start_frequency_hz:.0f} Hz")
    print(f"Search range: {cfg.lower_bound_hz:.0f}–{cfg.upper_bound_hz:.0f} Hz | Resolution: ±{cfg.resolution_hz:.0f} Hz")
    print(f"Random pre-play delay: 0.00–{cfg.pre_play_delay_max_s:.2f} s\n")
    print("Controls: y=heard, n=not heard, r=replay, q=quit\n")

    # Bounds
    lo = cfg.lower_bound_hz
    hi = cfg.upper_bound_hz
    f = max(lo, min(hi, cfg.start_frequency_hz))

    tmp_path = "./input/tmp_hearing_tone.wav"

    while (hi - lo) > cfg.resolution_hz:
        # Synthesize and write tone
        tone = generate_tone(f, cfg)
        sf.write(tmp_path, tone, cfg.sample_rate_hz)

        # Random pre-delay before playback to reduce anticipation bias
        pre_delay = random.uniform(0.0, cfg.pre_play_delay_max_s)
        print(f"\nPreparing {f:.0f} Hz & inserting random delay")
        play_tone_mac(tmp_path, pre_delay)

        ans = prompt_user("Heard it? [y/n] (r=replay, q=quit): ")
        if ans == "q":
            break
        if ans == "r":
            continue
        if ans == "y":
            lo = f   # heard it → raise lower bound
        else:
            hi = f   # not heard → lower upper bound

        # Next frequency: midpoint
        f = (lo + hi) / 2.0
        f = max(lo, min(hi, f))

        if hi - lo <= cfg.resolution_hz:
            break

    # Final report
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    midpoint_hz = 0.5 * (lo + hi)
    
    if hi - lo <= cfg.resolution_hz:
        print(f"\nEstimated highest audible frequency ≈ {lo:.1f}–{hi:.1f} Hz (midpoint {midpoint_hz:.1f} Hz).")
    else:
        print(f"\nStopped early. Current bounds: {lo:.1f}–{hi:.1f} Hz (midpoint {midpoint_hz:.1f} Hz).")

    # Pitch analysis
    print("\n🎵 Musical Pitch Analysis:")
    note_name, octave, cents = frequency_to_pitch(midpoint_hz)
    
    cents_str = ""
    if abs(cents) >= 1.0:
        cents_sign = "+" if cents >= 0 else ""
        cents_str = f" ({cents_sign}{cents:.0f} cents)"
    
    print(f"   Closest pitch: {note_name}{octave}{cents_str}")
    print(f"   ({midpoint_hz:.1f} Hz ≈ {note_name}{octave} in 12-tone equal temperament)")
    
    # Calculate the exact frequency of the closest pitch for comparison
    # A4 = 440 Hz = MIDI 69
    midi_note_exact = (octave + 1) * 12 + ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"].index(note_name)
    exact_pitch_freq = 440.0 * (2.0 ** ((midi_note_exact - 69) / 12.0))
    print(f"   Exact {note_name}{octave} frequency: {exact_pitch_freq:.1f} Hz")

    print("\nTip: test each ear separately; you can increase duration_seconds if you need more time to detect the tone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
