#!/usr/bin/env python3
"""
Tinnitus A/B Comparer (Random frequencies between A8 and C9)
------------------------------------------------------------
- Uses random frequencies in the continuous range A8..C9 (≈7040–8372 Hz).
- Trial sequence:  Noise → Tone A → Noise → Tone B
- Noise and tones always pan to the same side (L/R/C).
- Fixed sample rate and fade time (not configurable).
- Output folder fixed to ./output.

Controls:
  1 = first tone is closer
  2 = second tone is closer
  r = replay the same pair
  s = skip (reshuffle challenger)
  m = mark current winner as FINAL (prints nearest note, saves summary + 3s tone)
  q = quit

Configurable at runtime:
  Tone duration, tone amplitude, noise duration, noise level (dBFS), pan side.

Requirements:
  pip install numpy soundfile
Platform:
  macOS (uses 'afplay' to play audio)
"""

import os, math, time, random, tempfile, subprocess
import numpy as np
import soundfile as sf

# ---------------- Fixed settings ----------------
SAMPLE_RATE_HZ = 48000     # fixed
FADE_TIME_S    = 0.03      # fixed
OUTPUT_DIR     = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Frequency limits: A8 .. C9
A4 = 440.0
def midi_to_freq(midi: int, A4hz: float = A4) -> float:
    return A4hz * (2.0 ** ((midi - 69) / 12.0))

A8_FREQ = midi_to_freq(117)   # ≈ 7040.0 Hz
C9_FREQ = midi_to_freq(120)   # ≈ 8372.0 Hz
LOWER_HZ, UPPER_HZ = A8_FREQ, C9_FREQ

# ---------------- Helpers ----------------
NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
def freq_to_note(f_hz: float, A4hz: float = A4):
    if f_hz <= 0: return ("?", 0, 0.0, 0.0)
    semis = 12.0 * math.log2(f_hz / A4hz)
    midi_exact = 69.0 + semis
    midi_round = int(round(midi_exact))
    cents = (midi_exact - midi_round) * 100.0
    name = NOTE_NAMES[midi_round % 12]
    octave = (midi_round // 12) - 1
    exact = midi_to_freq(midi_round, A4hz)
    return name, octave, cents, exact

def db_to_amp(db: float) -> float:
    return 10.0 ** (db / 20.0)

def generate_sine(freq_hz: float, dur_s: float, amp: float) -> np.ndarray:
    n = int(SAMPLE_RATE_HZ * dur_s)
    t = np.linspace(0.0, dur_s, n, endpoint=False)
    y = np.sin(2 * np.pi * freq_hz * t)
    fade = int(SAMPLE_RATE_HZ * FADE_TIME_S)
    if fade > 0:
        y[:fade] *= np.linspace(0.0, 1.0, fade)
        y[-fade:] *= np.linspace(1.0, 0.0, fade)
    y /= max(1e-9, np.max(np.abs(y)))
    return (amp * y).astype(np.float32)

def generate_white_noise(dur_s: float, amp_db: float) -> np.ndarray:
    n = int(SAMPLE_RATE_HZ * dur_s)
    x = np.random.normal(0.0, 1.0, n)
    x /= max(1e-9, np.max(np.abs(x)))
    return (db_to_amp(amp_db) * x).astype(np.float32)

def pan_stereo(mono: np.ndarray, pan: str) -> np.ndarray:
    if pan == 'l':
        return np.column_stack([mono, np.zeros_like(mono)])
    elif pan == 'r':
        return np.column_stack([np.zeros_like(mono), mono])
    else:
        return np.column_stack([mono, mono])

def play_np_audio(stereo: np.ndarray):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    path = tmp.name
    tmp.close()
    sf.write(path, stereo, SAMPLE_RATE_HZ)
    try:
        subprocess.run(["afplay", path], check=True)
    finally:
        try: os.remove(path)
        except OSError: pass

def prompt_float(prompt_text: str, default: float) -> float:
    raw = input(f"{prompt_text} [default {default}]: ").strip()
    if raw == "": return default
    try: return float(raw)
    except ValueError:
        print("Invalid number; using default.")
        return default

def prompt_choice(prompt_text: str, choices: set, default: str) -> str:
    raw = input(f"{prompt_text} {choices} [default {default}]: ").strip().lower()
    if raw == "": return default
    return raw if raw in choices else default

# ---------------- Core compare loop ----------------
def main():
    print("\n=== Tinnitus Frequency Matcher (A8 → C9) ===")
    print("🎯 Uses adaptive narrowing: each choice focuses the search range")
    print("🔊 Sequence per trial: Noise → Tone A → Noise → Tone B")  
    print("🎧 Noise & tones share the SAME pan side. Sample rate and fade are fixed.\n")

    # Runtime config (kept minimal but useful)
    tone_duration = prompt_float("Tone duration (seconds)", 1.5)
    tone_amp      = prompt_float("Tone amplitude (0–1, keep very low)", 0.05)
    noise_duration= prompt_float("Noise duration (seconds)", 2.5)
    noise_db      = prompt_float("Noise level in dBFS (e.g., -40 very quiet)", -40.0)
    pan_side      = prompt_choice("Pan side for BOTH noise and tones? (l/r/c)", {"l","r","c"}, "c")

    # Zooming strategy - progressively narrow the search range
    current_low  = LOWER_HZ
    current_high = UPPER_HZ
    zoom_factor  = 0.7    # shrink band around winner each round (0.7 = retain 70% of range)
    min_band_hz  = 2.0    # auto-stop threshold - very precise (±1 Hz)

    # Initial random pair
    best_freq = random.uniform(current_low, current_high)
    def random_challenger():
        return random.uniform(current_low, current_high)

    challenger_freq = random_challenger()

    print("\nControls: 1 = first closer, 2 = second closer, r = replay, s = skip, m = mark final, q = quit.")
    print("Tip: keep system volume LOW; these are very high frequencies.\n")

    def play_pair(fA: float, fB: float):
        # Noise → A → Noise → B
        if noise_duration > 0:
            play_np_audio(pan_stereo(generate_white_noise(noise_duration, noise_db), pan_side))
        play_np_audio(pan_stereo(generate_sine(fA, tone_duration, tone_amp), pan_side))
        if noise_duration > 0:
            play_np_audio(pan_stereo(generate_white_noise(noise_duration, noise_db), pan_side))
        play_np_audio(pan_stereo(generate_sine(fB, tone_duration, tone_amp), pan_side))

    # Main interactive loop
    while True:
        fA, fB = best_freq, challenger_freq
        range_width = current_high - current_low
        print(f"\n🎯 Search range: {current_low:.1f} - {current_high:.1f} Hz (±{range_width/2:.1f} Hz)")
        print(f"Pair: 1) {fA:.1f} Hz   vs   2) {fB:.1f} Hz")
        play_pair(fA, fB)

        ans = input("Closer to your tinnitus? [1/2, r=replay, s=skip, m=mark final, q=quit]: ").strip().lower()
        if ans == "q":
            print("\nExiting.")
            return
        if ans == "r":
            continue
        if ans == "s":
            # reshuffle challenger, keep current band
            challenger_freq = random_challenger()
            continue
        if ans == "m":
            final_freq = best_freq
            note, octv, cents, exact = freq_to_note(final_freq)
            safe_note = note.replace("#", "sharp")
            # Save final 3s tone & summary
            final_tone = pan_stereo(generate_sine(final_freq, 3.0, tone_amp), pan_side)
            final_path = os.path.join(OUTPUT_DIR, f"final_match_{safe_note}{octv}_{final_freq:.1f}Hz.wav")
            sf.write(final_path, final_tone, SAMPLE_RATE_HZ)
            summary_path = os.path.join(OUTPUT_DIR, "final_match_summary.txt")
            with open(summary_path, "w") as f:
                f.write(f"Final frequency: {final_freq:.2f} Hz\n")
                f.write(f"Nearest note: {note}{octv} (exact {exact:.2f} Hz, {cents:+.0f} cents)\n")
            print("\n=== FINAL MATCH ===")
            print(f"Frequency: {final_freq:.2f} Hz")
            print(f"Nearest note: {note}{octv} (exact {exact:.2f} Hz, {cents:+.0f} cents)")
            print(f"💾 Saved 3s tone: {final_path}")
            print(f"📝 Summary: {summary_path}\n")
            return
        if ans not in {"1","2"}:
            print("Please type 1, 2, r, s, m, or q.")
            continue

        # Winner update + save 2s breadcrumb of chosen
        winner_freq = fA if ans == "1" else fB
        note, octv, _, _ = freq_to_note(winner_freq)
        breadcrumb = pan_stereo(generate_sine(winner_freq, 2.0, tone_amp), pan_side)
        safe_note = note.replace("#", "sharp")
        outpath = os.path.join(OUTPUT_DIR, f"choice_{safe_note}{octv}_{winner_freq:.1f}Hz.wav")
        sf.write(outpath, breadcrumb, SAMPLE_RATE_HZ)
        print(f"💾 Saved 2s snippet of your choice to: {outpath}")

        # Narrow the band around the winner and pick a new challenger
        best_freq = winner_freq
        old_range = current_high - current_low
        band_width = max(old_range, 1.0)
        new_half_band = max(band_width * zoom_factor * 0.5, min_band_hz * 0.5)
        current_low  = max(LOWER_HZ, best_freq - new_half_band)
        current_high = min(UPPER_HZ, best_freq + new_half_band)
        new_range = current_high - current_low
        
        print(f"🔍 Narrowing search around {winner_freq:.1f} Hz (range: {old_range:.1f} → {new_range:.1f} Hz)")

        if (current_high - current_low) <= min_band_hz:
            # Auto mark final if band is already very tight
            final_freq = best_freq
            note, octv, cents, exact = freq_to_note(final_freq)
            safe_note = note.replace("#", "sharp")
            final_tone = pan_stereo(generate_sine(final_freq, 3.0, tone_amp), pan_side)
            final_path = os.path.join(OUTPUT_DIR, f"final_match_{safe_note}{octv}_{final_freq:.1f}Hz.wav")
            sf.write(final_path, final_tone, SAMPLE_RATE_HZ)
            summary_path = os.path.join(OUTPUT_DIR, "final_match_summary.txt")
            with open(summary_path, "w") as f:
                f.write(f"Final frequency: {final_freq:.2f} Hz\n")
                f.write(f"Nearest note: {note}{octv} (exact {exact:.2f} Hz, {cents:+.0f} cents)\n")
            print("\n=== FINAL MATCH (auto-stop: tight band) ===")
            print(f"Frequency: {final_freq:.2f} Hz")
            print(f"Nearest note: {note}{octv} (exact {exact:.2f} Hz, {cents:+.0f} cents)")
            print(f"💾 Saved 3s tone: {final_path}")
            print(f"📝 Summary: {summary_path}\n")
            return

        challenger_freq = random.uniform(current_low, current_high)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
