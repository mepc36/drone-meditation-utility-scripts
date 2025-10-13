#!/usr/bin/env python3
"""
Tinnitus A/B Comparer (Random A8..C9, JND-aware with Placebo Rounds)
- Random frequencies anywhere between A8..C9 (≈7040–8372 Hz).
- Enforces JND: any compared pair differs by ≥ 80 Hz.
- Trial: Noise → Tone A → Noise → Tone B, same pan for both noise and tones.
- Placebo rounds (probabilistic): neither tone equals the current best; outcome ignored.
- Sample rate & fade fixed; output folder fixed to ./output.

Controls:
  1 = first tone is closer
  2 = second tone is closer
  r = replay the same pair
  s = skip (reshuffle)
  m = mark current winner as FINAL (prints nearest note, saves summary + 3s tone)
  q = quit

Runtime prompts:
  Tone duration, tone amplitude, noise duration, noise level (dBFS), pan side, placebo probability.

Requirements: pip install numpy soundfile
Platform: macOS (uses 'afplay')
"""

# VOLUME WARNING
print("⚠️ WARNING: This script generates high-frequency tones that can cause hearing damage.")
print("   Turn your volume DOWN to the lowest audible level before continuing.")
input("   Press Enter when volume is set safely low...")
print()

import os, math, random, tempfile, subprocess
import numpy as np
import soundfile as sf

# ---------------- Fixed settings ----------------
SAMPLE_RATE_HZ = 48000
FADE_TIME_S    = 0.03
OUTPUT_DIR     = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

JND_HZ = 80.0  # compared tones must be ≥ 80 Hz apart

# Frequency limits: A8 .. C9
A4 = 440.0
def midi_to_freq(midi: int, A4hz: float = A4) -> float:
    return A4hz * (2.0 ** ((midi - 69) / 12.0))

A8_FREQ = midi_to_freq(117)   # ≈ 7040 Hz
C9_FREQ = midi_to_freq(120)   # ≈ 8372 Hz
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

# ---- JND-aware random selection ----
def can_place_jnd_away(best: float, low: float, high: float, jnd: float) -> bool:
    return (best - low) >= jnd or (high - best) >= jnd

def random_jnd_challenger(best: float, low: float, high: float, jnd: float, max_tries: int = 100) -> float | None:
    if not can_place_jnd_away(best, low, high, jnd):
        return None
    for _ in range(max_tries):
        x = random.uniform(low, high)
        if abs(x - best) >= jnd:
            return x
    if (high - best) >= jnd:
        return min(high, best + jnd)
    if (best - low) >= jnd:
        return max(low, best - jnd)
    return None

def random_jnd_placebo_pair(best: float, low: float, high: float, jnd: float, max_tries: int = 200):
    """
    Return (f1, f2) such that:
      - both in [low, high]
      - |f1 - best| >= jnd and |f2 - best| >= jnd
      - |f1 - f2|  >= jnd
    """
    if (high - low) < (2 * jnd):
        return None
    for _ in range(max_tries):
        f1 = random.uniform(low, high)
        if abs(f1 - best) < jnd:
            continue
        f2 = random.uniform(low, high)
        if abs(f2 - best) < jnd or abs(f2 - f1) < jnd:
            continue
        return (f1, f2)
    candidates = []
    if (high - best) >= jnd:
        candidates.append(min(high, best + jnd))
    if (best - low) >= jnd:
        candidates.append(max(low, best - jnd))
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            if abs(candidates[i] - candidates[j]) >= jnd:
                return (candidates[i], candidates[j])
    return None

# ---------------- Core compare loop ----------------
def main():
    print("\n=== Tinnitus Frequency Matcher (A8 → C9, JND-aware, with Placebo) ===")
    print("Sequence: Noise → Tone A → Noise → Tone B  |  compared tones always differ by ≥ 80 Hz.\n")

    # Runtime config
    tone_duration = prompt_float("Tone duration (seconds)", 1.5)
    tone_amp      = prompt_float("Tone amplitude (0–1, keep very low)", 0.05)
    noise_duration= prompt_float("Noise duration (seconds)", 2.5)
    noise_db      = prompt_float("Noise level in dBFS (e.g., -40 very quiet)", -40.0)
    pan_side      = prompt_choice("Pan side for BOTH noise and tones? (l/r/c)", {"l","r","c"}, "c")
    placebo_prob  = prompt_float("Placebo round probability (0.0–1.0)", 0.20)

    # Adaptive band
    current_low  = LOWER_HZ
    current_high = UPPER_HZ
    zoom_factor  = 0.7   # retain 70% band width each round around winner

    # Initialize best and challenger with JND separation
    if (current_high - current_low) >= 2 * JND_HZ:
        best_freq = random.uniform(current_low + JND_HZ, current_high - JND_HZ)
    else:
        best_freq = (current_low + current_high) / 2.0

    challenger_freq = random_jnd_challenger(best_freq, current_low, current_high, JND_HZ)
    if challenger_freq is None:
        note, octv, cents, exact = freq_to_note(best_freq)
        safe_note = note.replace("#", "sharp")
        final_tone = pan_stereo(generate_sine(best_freq, 3.0, tone_amp), pan_side)
        final_path = os.path.join(OUTPUT_DIR, f"final_match_{safe_note}{octv}_{best_freq:.1f}Hz.wav")
        sf.write(final_path, final_tone, SAMPLE_RATE_HZ)
        summary_path = os.path.join(OUTPUT_DIR, "final_match_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Final frequency: {best_freq:.2f} Hz\nNearest note: {note}{octv} (exact {exact:.2f} Hz, {cents:+.0f} cents)\n")
        print("\n=== FINAL MATCH (band < JND) ===")
        print(f"Frequency: {best_freq:.2f} Hz  — Nearest note: {note}{octv} (exact {exact:.2f} Hz, {cents:+.0f} cents)")
        print(f"Saved 3s tone: {final_path}\nSummary: {summary_path}")
        return

    def play_pair(fA: float, fB: float):
        # Noise → A → Noise → B
        if noise_duration > 0:
            play_np_audio(pan_stereo(generate_white_noise(noise_duration, noise_db), pan_side))
        play_np_audio(pan_stereo(generate_sine(fA, tone_duration, tone_amp), pan_side))
        if noise_duration > 0:
            play_np_audio(pan_stereo(generate_white_noise(noise_duration, noise_db), pan_side))
        play_np_audio(pan_stereo(generate_sine(fB, tone_duration, tone_amp), pan_side))

    round_idx = 0
    last_pair = None
    print("\n" + "="*60)  # initial separator

    while True:
        round_idx += 1

        # Decide placebo vs normal
        is_placebo = (random.random() < max(0.0, min(1.0, placebo_prob)))
        pair = None
        if is_placebo:
            pair = random_jnd_placebo_pair(best_freq, current_low, current_high, JND_HZ)
            if pair is None:
                is_placebo = False

        if is_placebo:
            f1, f2 = pair
            fA, fB = (f1, f2) if random.choice([True, False]) else (f2, f1)
        else:
            if challenger_freq is None or abs(challenger_freq - best_freq) < JND_HZ:
                challenger_freq = random_jnd_challenger(best_freq, current_low, current_high, JND_HZ)
                if challenger_freq is None:
                    # finalize when we can't place a valid challenger
                    final_freq = best_freq
                    note, octv, cents, exact = freq_to_note(final_freq)
                    safe_note = note.replace("#", "sharp")
                    final_tone = pan_stereo(generate_sine(final_freq, 3.0, tone_amp), pan_side)
                    final_path = os.path.join(OUTPUT_DIR, f"final_match_{safe_note}{octv}_{final_freq:.1f}Hz.wav")
                    sf.write(final_path, final_tone, SAMPLE_RATE_HZ)
                    summary_path = os.path.join(OUTPUT_DIR, "final_match_summary.txt")
                    with open(summary_path, "w") as f:
                        f.write(f"Final frequency: {final_freq:.2f} Hz\nNearest note: {note}{octv} (exact {exact:.2f} Hz, {cents:+.0f} cents)\n")
                    print("\n=== FINAL MATCH (no JND-separated challenger possible) ===")
                    print(f"Frequency: {final_freq:.2f} Hz")
                    print(f"Nearest note: {note}{octv} (exact {exact:.2f} Hz, {cents:+.0f} cents)")
                    print(f"Saved 3s tone: {final_path}\nSummary: {summary_path}")
                    return
            # Randomize order to reduce bias
            if random.choice([True, False]):
                fA, fB = best_freq, challenger_freq
                best_is_first = True
            else:
                fA, fB = challenger_freq, best_freq
                best_is_first = False

        # Play without revealing frequencies yet
        play_pair(fA, fB)
        last_pair = (fA, fB, is_placebo) if is_placebo else (fA, fB, best_is_first)

        ans = input("Closer to your tinnitus? [1/2, r=replay, s=skip, m=mark final, q=quit]: ").strip().lower()

        if ans == "q":
            print("\nExiting.")
            return

        if ans == "r":
            # Replay same pair; do not reveal frequencies yet
            fA, fB, _ = last_pair
            play_pair(fA, fB)
            # Ask again
            ans = input("Closer to your tinnitus? [1/2, r=replay, s=skip, m=mark final, q=quit]: ").strip().lower()

        if ans == "m":
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
            # Reveal final pick
            print(f"\nTone 1 -- {fA:.1f} Hz")
            print(f"Tone 2 -- {fB:.1f} Hz")
            print(f"range = {abs(fA - fB):.1f} Hz (JND {JND_HZ:.0f} Hz)")
            print("\n=== FINAL MATCH ===")
            print(f"Frequency: {final_freq:.2f} Hz")
            print(f"Nearest note: {note}{octv} (exact {exact:.2f} Hz, {cents:+.0f} cents)")
            print(f"Saved 3s tone: {final_path}\nSummary: {summary_path}")
            print("\n" + "="*60)
            return

        if ans == "s":
            # Show the hidden freqs after your choice
            print(f"\nTone 1 -- {fA:.1f} Hz")
            print(f"Tone 2 -- {fB:.1f} Hz")
            print(f"range = {abs(fA - fB):.1f} Hz (JND {JND_HZ:.0f} Hz)")
            # Skip: draw a new pair of the same type
            if is_placebo:
                # new placebo pair if possible; otherwise normal next round
                pair = random_jnd_placebo_pair(best_freq, current_low, current_high, JND_HZ)
                if pair is None:
                    challenger_freq = random_jnd_challenger(best_freq, current_low, current_high, JND_HZ)
            else:
                challenger_freq = random_jnd_challenger(best_freq, current_low, current_high, JND_HZ)
            print("\n" + "="*60)
            continue

        if ans not in {"1","2"}:
            print("Please type 1, 2, r, s, m, or q.")
            print("\n" + "="*60)
            continue

        # Reveal the two frequencies only AFTER you made a selection
        print(f"\nTone 1 -- {fA:.1f} Hz")
        print(f"Tone 2 -- {fB:.1f} Hz")
        print(f"range = {abs(fA - fB):.1f} Hz (JND {JND_HZ:.0f} Hz)")

        # Handle outcome
        if is_placebo:
            # Ignore outcome; do not update best or band
            print("(Placebo round ignored.)")
            # Ensure next normal challenger exists
            challenger_freq = random_jnd_challenger(best_freq, current_low, current_high, JND_HZ)
            print("\n" + "="*60)
            continue

        # Normal round: update winner + narrow band
        chosen_is_first = (ans == "1")
        chosen_freq = fA if chosen_is_first else fB
        # Determine which is current best
        best_is_first = last_pair[2]  # True if best was in position A for that pair
        winner_freq = best_freq if ((best_is_first and chosen_is_first) or ((not best_is_first) and (not chosen_is_first))) else chosen_freq

        # Save breadcrumb of the chosen tone (2s)
        note, octv, _, _ = freq_to_note(winner_freq)
        breadcrumb = pan_stereo(generate_sine(winner_freq, 2.0, tone_amp), pan_side)
        safe_note = note.replace("#", "sharp")
        outpath = os.path.join(OUTPUT_DIR, f"choice_{safe_note}{octv}_{winner_freq:.1f}Hz.wav")
        sf.write(outpath, breadcrumb, SAMPLE_RATE_HZ)
        print(f"💾 Saved 2s snippet: {outpath}")

        # Narrow the band around the winner (keep room for ≥ JND challenger)
        best_freq = winner_freq
        band_half = max((current_high - current_low) * zoom_factor / 2.0, JND_HZ)
        current_low  = max(LOWER_HZ, best_freq - band_half)
        current_high = min(UPPER_HZ, best_freq + band_half)

        # Prepare next challenger with JND separation
        challenger_freq = random_jnd_challenger(best_freq, current_low, current_high, JND_HZ)
        if challenger_freq is None:
            # Finalize if challenger cannot be placed
            final_freq = best_freq
            note, octv, cents, exact = freq_to_note(final_freq)
            safe_note = note.replace("#", "sharp")
            final_tone = pan_stereo(generate_sine(final_freq, 3.0, tone_amp), pan_side)
            final_path = os.path.join(OUTPUT_DIR, f"final_match_{safe_note}{octv}_{final_freq:.1f}Hz.wav")
            sf.write(final_path, final_tone, SAMPLE_RATE_HZ)
            summary_path = os.path.join(OUTPUT_DIR, "final_match_summary.txt")
            with open(summary_path, "w") as f:
                f.write(f"Final frequency: {final_freq:.2f} Hz\nNearest note: {note}{octv} (exact {exact:.2f} Hz, {cents:+.0f} cents)\n")
            print("\n=== FINAL MATCH (no JND-separated challenger possible) ===")
            print(f"Frequency: {final_freq:.2f} Hz")
            print(f"Nearest note: {note}{octv} (exact {exact:.2f} Hz, {cents:+.0f} cents)")
            print(f"Saved 3s tone: {final_path}\nSummary: {summary_path}")
            print("\n" + "="*60)
            return

        print("\n" + "="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
