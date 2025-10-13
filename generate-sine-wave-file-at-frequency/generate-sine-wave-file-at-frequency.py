#!/usr/bin/env python3
"""
Generate and play sine waves interactively.

Usage:
  python generate_sine_wave_loop.py
  Enter a frequency in Hz (e.g. 8000) to play a tone.
  Type 'q' or 'quit' to exit.

Requirements:
  pip install numpy soundfile
Platform:
  macOS (uses 'afplay' for playback)
"""

# VOLUME WARNING - This script generates audio that can damage hearing
print("⚠️  WARNING: This script generates high-frequency tones that can cause hearing damage.")
print("   Loud volumes may cause permanent hearing loss, tinnitus, or ear pain.")
print("   Turn your volume DOWN to the lowest audible level before continuing.")
input("   Press Enter when volume is set safely low...")
print()

import os, math, subprocess
import numpy as np
import soundfile as sf

# ---------------- Config ----------------
SAMPLE_RATE_HZ = 48000
DURATION_S = 2.0
AMPLITUDE = 0.2      # 0–1 (keep safe)
FADE_TIME_S = 0.05
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Helpers ----------------
def generate_sine(freq_hz: float, sr: int, dur: float, amp: float, fade_s: float) -> np.ndarray:
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    y = np.sin(2 * np.pi * freq_hz * t)
    fade = int(sr * fade_s)
    if fade > 0:
        y[:fade] *= np.linspace(0, 1, fade)
        y[-fade:] *= np.linspace(1, 0, fade)
    y /= max(1e-9, np.max(np.abs(y)))
    return (amp * y).astype(np.float32)

def play_audio(path: str):
    try:
        subprocess.run(["afplay", path], check=True)
    except Exception as e:
        print(f"Playback failed: {e}")

# ---------------- Main ----------------
def main():
    print("\n=== Interactive Sine Wave Generator ===")
    print("Enter a frequency in Hz (e.g. 8000) and press Enter.")
    print("Type 'q' or 'quit' to exit.\n")

    while True:
        user_input = input("Frequency (Hz): ").strip().lower()
        if user_input in {"q", "quit", "exit"}:
            print("Goodbye!")
            break

        try:
            freq = float(user_input)
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        if freq <= 0:
            print("Frequency must be positive.")
            continue

        print(f"\nGenerating {DURATION_S:.1f}s sine wave at {freq:.1f} Hz...")
        y = generate_sine(freq, SAMPLE_RATE_HZ, DURATION_S, AMPLITUDE, FADE_TIME_S)
        stereo = np.column_stack([y, y])
        filename = f"sine_{freq:.1f}Hz.wav"
        path = os.path.join(OUTPUT_DIR, filename)
        sf.write(path, stereo, SAMPLE_RATE_HZ)
        print(f"💾 Saved: {path}")

        print("▶️ Playing...")
        play_audio(path)
        print("Done.\n")

if __name__ == "__main__":
    main()
