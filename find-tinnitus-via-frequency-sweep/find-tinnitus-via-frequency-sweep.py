#!/usr/bin/env python3
"""
Tinnitus Frequency Sweep with Configurable Noise, Silence, and Default -40 dBFS Level (macOS)
---------------------------------------------------------------------------------------------
1) Ask which ear (L/R), noise duration, silence duration.
2) Play quiet broadband noise (hard-panned to that ear) at −40 dBFS.
3) Wait in silence.
4) Play 60 s logarithmic sine sweep (2 → 12 kHz, centered).

Press Space or Enter during the sweep to mark the instantaneous frequency
and musical note.

Requirements:  Python 3, numpy, soundfile  (pip install numpy soundfile)
Platform:  macOS (uses “afplay” for playback)
"""

import os, math, time, tty, sys, termios, select, tempfile, subprocess
import numpy as np
import soundfile as sf

# ---------------- Base configuration ----------------
START_FREQ_HZ    = 2000.0
END_FREQ_HZ      = 12000.0
SWEEP_DURATION_S = 60.0
SAMPLE_RATE_HZ   = 48000
SWEEP_AMPLITUDE  = 0.20
FADE_TIME_S      = 0.05
DEFAULT_NOISE_DB = -40.0         # default noise level in dBFS
MARK_KEYS        = {b' ', b'\r', b'\n'}

# ---------------- User prompts ----------------
def prompt_user_settings():
    """Ask which ear, durations; noise level defaults to −40 dBFS."""
    while True:
        ear = input("Which ear for the noise contrast? [L/R]: ").strip().lower()
        if ear in ("l", "r"):
            break
        print("Please type L or R.")
    try:
        noise_duration = float(input("Noise duration (seconds) [default 5]: ") or 5)
    except ValueError:
        noise_duration = 5.0
    try:
        silence_duration = float(input("Silence duration (seconds) [default 4]: ") or 4)
    except ValueError:
        silence_duration = 4.0
    try:
        noise_db = float(input(f"Noise level in dBFS [default {DEFAULT_NOISE_DB:.0f}]: ") or DEFAULT_NOISE_DB)
    except ValueError:
        noise_db = DEFAULT_NOISE_DB
    return ear, noise_duration, silence_duration, noise_db

# ---------------- Audio generation ----------------
def db_to_amplitude(db_value: float) -> float:
    """Convert decibels (dBFS) to linear amplitude (0–1)."""
    return 10.0 ** (db_value / 20.0)

def generate_white_noise(duration_s, sr_hz, amplitude):
    n = int(sr_hz * duration_s)
    noise = np.random.normal(0.0, 1.0, n)
    noise /= max(1e-9, np.max(np.abs(noise)))
    return (amplitude * noise).astype(np.float32)

def hard_pan_mono_to_stereo(mono_wave, ear):
    left = mono_wave if ear == 'l' else np.zeros_like(mono_wave)
    right = mono_wave if ear == 'r' else np.zeros_like(mono_wave)
    return np.column_stack([left, right])

def generate_log_sweep(start_hz, end_hz, duration_s, sr_hz, amplitude):
    t = np.linspace(0.0, duration_s, int(sr_hz * duration_s), endpoint=False)
    k = math.log(end_hz / start_hz) / duration_s
    phase = 2.0 * math.pi * (start_hz * (np.exp(k * t) - 1.0) / k)
    sweep = np.sin(phase)
    fade_samples = int(sr_hz * FADE_TIME_S)
    if fade_samples > 0:
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        sweep[:fade_samples] *= fade_in
        sweep[-fade_samples:] *= fade_out
    sweep /= max(1e-9, np.max(np.abs(sweep)))
    mono = (amplitude * sweep).astype(np.float32)
    return np.column_stack([mono, mono])

def instantaneous_frequency_at_time(t_sec, start_hz, end_hz, duration_s):
    k = math.log(end_hz / start_hz) / duration_s
    return start_hz * math.exp(k * max(0.0, min(t_sec, duration_s)))

# ---------------- Pitch utils ----------------
NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def frequency_to_pitch(f_hz, A4=440.0):
    semis = 12 * math.log2(f_hz / A4)
    midi_exact = 69 + semis
    midi_rounded = int(round(midi_exact))
    cents = (midi_exact - midi_rounded) * 100
    name = NOTE_NAMES[midi_rounded % 12]
    octave = (midi_rounded // 12) - 1
    exact_freq = A4 * (2 ** ((midi_rounded - 69) / 12))
    return name, octave, cents, exact_freq

# ---------------- Terminal key handling ----------------
class RawTerminal:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self
    def __exit__(self, *args):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
    def key_pressed(self, timeout=0.02):
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if r:
            return os.read(self.fd, 1)
        return None

# ---------------- Playback helpers ----------------
def play_with_afplay(wave, sr):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()
    sf.write(tmp_path, wave, sr)
    proc = subprocess.Popen(["afplay", tmp_path])
    return proc, tmp_path

def cleanup(path):
    try: os.remove(path)
    except OSError: pass

# ---------------- Main flow ----------------
def main():
    print("\n=== Tinnitus Sweep with Default −40 dBFS Noise ===")
    print("Headphones recommended. Keep system volume LOW.\n")
    ear, noise_duration, silence_duration, noise_db = prompt_user_settings()
    noise_amplitude = db_to_amplitude(noise_db)

    # 1) Noise
    print(f"\nNoise: {noise_duration:.1f}s at {noise_db:.0f} dBFS → {'Left' if ear=='l' else 'Right'} ear")
    noise = generate_white_noise(noise_duration, SAMPLE_RATE_HZ, noise_amplitude)
    stereo_noise = hard_pan_mono_to_stereo(noise, ear)
    proc, path = play_with_afplay(stereo_noise, SAMPLE_RATE_HZ)
    proc.wait(); cleanup(path)

    # 2) Silence
    print(f"Silence ({silence_duration:.1f}s)…")
    time.sleep(silence_duration)

    # 3) Sweep
    print(f"\nStarting {SWEEP_DURATION_S:.0f}s sweep (2→12 kHz, centered). Press Space/Enter to mark frequency & note.")
    sweep = generate_log_sweep(START_FREQ_HZ, END_FREQ_HZ, SWEEP_DURATION_S, SAMPLE_RATE_HZ, SWEEP_AMPLITUDE)
    proc, path = play_with_afplay(sweep, SAMPLE_RATE_HZ)

    start_time = time.time()
    marks = []

    with RawTerminal() as rt:
        while proc.poll() is None:
            ch = rt.key_pressed()
            if ch in MARK_KEYS:
                elapsed = time.time() - start_time
                freq = instantaneous_frequency_at_time(elapsed, START_FREQ_HZ, END_FREQ_HZ, SWEEP_DURATION_S)
                note, octv, cents, exact = frequency_to_pitch(freq)
                marks.append((elapsed, freq, note, octv, cents, exact))
                print(f"\n⏱ t={elapsed:5.2f}s | f≈ {freq:7.1f} Hz | {note}{octv} ({exact:.1f} Hz, {cents:+.0f} cents)")

    proc.wait(); cleanup(path)

    print("\n✅ Sweep complete.")
    if not marks:
        print("No marks recorded. Try again and tap Space/Enter when the tone blends with your tinnitus.")
    else:
        print("Marks recorded:")
        for t, f, n, o, c, ex in marks:
            print(f"  t={t:5.2f}s  f≈{f:7.1f} Hz → {n}{o} ({ex:.1f} Hz, {c:+.0f} cents)")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
