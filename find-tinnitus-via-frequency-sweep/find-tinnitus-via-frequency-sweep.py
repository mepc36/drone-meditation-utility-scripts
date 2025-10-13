#!/usr/bin/env python3
"""
Tinnitus Frequency Sweep (Random 1–4 kHz Start → Age-Based Upper Limit)
Now saves a 2-second WAV in ./output for EVERY mark (Space/Enter).

Flow:
  1) Ask: noise ear (L/R), noise duration, silence duration, noise dBFS, sweep pan (L/R/C), upper limit (Hz).
  2) Play quiet broadband noise hard-panned to chosen ear.
  3) Silence.
  4) Sweep starts at a RANDOM frequency in [1 kHz, 4 kHz] and goes UP to upper limit.
  5) During the sweep, press Space or Enter to:
       - print instantaneous Hz + nearest note
       - SAVE a 2s tone at that frequency to ./output

Requirements: pip install numpy soundfile
Platform: macOS (uses 'afplay' to play audio)
"""

import os, math, time, tty, sys, termios, select, tempfile, subprocess, random
import numpy as np
import soundfile as sf

# ---------------- Core configuration ----------------
START_MIN_HZ      = 1000.0       # random lower bound
START_MAX_HZ      = 4000.0       # random upper bound
DEFAULT_UPPER_HZ  = 16000.0      # typical limit for ~35-year-old; user can override
SWEEP_DURATION_S  = 60.0
SAMPLE_RATE_HZ    = 48000
SWEEP_AMPLITUDE   = 0.20
SAVE_TONE_AMPL    = 0.20         # amplitude for saved 2s tone files
FADE_TIME_S       = 0.05
DEFAULT_NOISE_DB  = -40.0
MARK_KEYS         = {b' ', b'\r', b'\n'}

OUTPUT_DIR        = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Prompts ----------------
def prompt_user_settings():
    while True:
        ear = input("Which ear for the broadband noise? [L/R]: ").strip().lower()
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
    while True:
        sweep_pan = input("Pan sweep to which side? [L/R/C=Center]: ").strip().lower()
        if sweep_pan in ("l", "r", "c"):
            break
        print("Please type L, R, or C.")
    try:
        upper_limit_hz = float(input(f"Upper frequency limit (Hz) [default {DEFAULT_UPPER_HZ:.0f}]: ") or DEFAULT_UPPER_HZ)
    except ValueError:
        upper_limit_hz = DEFAULT_UPPER_HZ

    # guardrails: at least 8k; under Nyquist; keep headroom
    upper_limit_hz = min(max(upper_limit_hz, 8000.0), SAMPLE_RATE_HZ/2 - 500.0)
    return ear, noise_duration, silence_duration, noise_db, sweep_pan, upper_limit_hz

# ---------------- Audio helpers ----------------
def db_to_amplitude(db): 
    return 10.0 ** (db / 20.0)

def generate_white_noise(duration_s, sr, amp):
    n = int(sr * duration_s)
    x = np.random.normal(0.0, 1.0, n)
    x /= max(1e-9, np.max(np.abs(x)))
    return (amp * x).astype(np.float32)

def hard_pan_mono_to_stereo(x, pan):
    if pan == "l":
        return np.column_stack([x, np.zeros_like(x)])
    elif pan == "r":
        return np.column_stack([np.zeros_like(x), x])
    else:
        return np.column_stack([x, x])  # center

def generate_log_sweep(start_hz, end_hz, dur_s, sr, amp):
    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
    k = math.log(end_hz / start_hz) / dur_s
    phase = 2 * math.pi * (start_hz * (np.exp(k * t) - 1) / k)
    y = np.sin(phase)
    fade = int(sr * FADE_TIME_S)
    if fade > 0:
        y[:fade] *= np.linspace(0, 1, fade)
        y[-fade:] *= np.linspace(1, 0, fade)
    y /= max(1e-9, np.max(np.abs(y)))
    return (amp * y).astype(np.float32)

def generate_sine_tone(freq_hz: float, duration_s: float, sr: int, amp: float, fade_s: float) -> np.ndarray:
    """Mono sine with gentle fades."""
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    y = np.sin(2 * np.pi * freq_hz * t)
    fade = int(sr * fade_s)
    if fade > 0:
        y[:fade] *= np.linspace(0, 1, fade)
        y[-fade:] *= np.linspace(1, 0, fade)
    y /= max(1e-9, np.max(np.abs(y)))
    return (amp * y).astype(np.float32)

def instantaneous_frequency_at_time(t, f0, f1, dur):
    k = math.log(f1 / f0) / dur
    return f0 * math.exp(k * max(0.0, min(t, dur)))

# ---------------- Pitch mapping ----------------
NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def frequency_to_pitch(f, A4=440.0):
    if f <= 0: return ("?", 0, 0.0, 0.0)
    semis = 12 * math.log2(f / A4)
    midi = 69 + semis
    m = int(round(midi))
    cents = (midi - m) * 100
    name = NOTE_NAMES[m % 12]
    octv = (m // 12) - 1
    exact = A4 * (2 ** ((m - 69)/12))
    return name, octv, cents, exact

# ---------------- Terminal key input ----------------
class RawTerminal:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self
    def __exit__(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
    def key_pressed(self, timeout=0.02):
        r,_,_ = select.select([sys.stdin], [], [], timeout)
        if r: return os.read(self.fd,1)
        return None

# ---------------- Playback ----------------
def play_with_afplay(wave, sr):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    path = tmp.name; tmp.close()
    sf.write(path, wave, sr)
    p = subprocess.Popen(["afplay", path])
    return p, path

def cleanup(path): 
    try: os.remove(path)
    except OSError: pass

# ---------------- Main ----------------
def main():
    print("\n=== Tinnitus Sweep (Random 1–4 kHz Start → Age-Based Limit) ===")
    print("Headphones recommended. Keep volume LOW.\n")
    ear, noise_dur, silence_dur, noise_db, sweep_pan, upper_hz = prompt_user_settings()
    noise_amp = db_to_amplitude(noise_db)

    start_hz = random.uniform(START_MIN_HZ, START_MAX_HZ)
    print(f"\nSweep will start at random {start_hz:.1f} Hz → {upper_hz:.0f} Hz.")

    # 1) Noise (hard-panned to chosen ear)
    print(f"Playing {noise_dur:.1f}s broadband noise ({noise_db:.0f} dBFS) → {('Left' if ear=='l' else 'Right')} ear.")
    noise = generate_white_noise(noise_dur, SAMPLE_RATE_HZ, noise_amp)
    stereo_noise = hard_pan_mono_to_stereo(noise, ear)
    proc, path = play_with_afplay(stereo_noise, SAMPLE_RATE_HZ)
    proc.wait(); cleanup(path)

    # 2) Silence
    print(f"Silence {silence_dur:.1f}s…")
    time.sleep(silence_dur)

    # 3) Sweep (configurable pan)
    print(f"\nStarting sweep ({start_hz:.0f} → {upper_hz:.0f} Hz, {('Left' if sweep_pan=='l' else 'Right' if sweep_pan=='r' else 'Center')}).")
    print("Press Space/Enter to mark frequency & note; a 2s tone WAV will be saved for each mark.\n")
    sweep_mono = generate_log_sweep(start_hz, upper_hz, SWEEP_DURATION_S, SAMPLE_RATE_HZ, SWEEP_AMPLITUDE)
    stereo = hard_pan_mono_to_stereo(sweep_mono, sweep_pan)
    proc, path = play_with_afplay(stereo, SAMPLE_RATE_HZ)

    start_time = time.time()
    marks = []
    mark_index = 1

    with RawTerminal() as rt:
        while proc.poll() is None:
            ch = rt.key_pressed()
            if ch in MARK_KEYS:
                elapsed = time.time() - start_time
                freq = instantaneous_frequency_at_time(elapsed, start_hz, upper_hz, SWEEP_DURATION_S)
                note, octv, cents, exact = frequency_to_pitch(freq)
                marks.append((elapsed, freq, note, octv, cents, exact))
                print(f"\n⏱ t={elapsed:5.2f}s | f≈{freq:7.1f} Hz | {note}{octv} ({exact:.1f} Hz, {cents:+.0f} cents)")

                # --- NEW: save a 2-second tone for this mark ---
                tone_mono = generate_sine_tone(freq, duration_s=2.0, sr=SAMPLE_RATE_HZ, amp=SAVE_TONE_AMPL, fade_s=FADE_TIME_S)
                tone_stereo = np.column_stack([tone_mono, tone_mono])  # save centered stereo for universal playback
                safe_note = note.replace("#", "sharp")
                filename = f"mark{mark_index:02d}_{freq:.1f}Hz_{safe_note}{octv}_{elapsed:.2f}s.wav"
                outpath = os.path.join(OUTPUT_DIR, filename)
                sf.write(outpath, tone_stereo, SAMPLE_RATE_HZ)
                print(f"💾 Saved 2s tone: {outpath}")
                mark_index += 1

    proc.wait(); cleanup(path)

    print("\n✅ Sweep complete.")
    if not marks:
        print("No marks recorded. Try again and tap Space/Enter when the tone blends with your tinnitus.")
    else:
        print("\nMarks recorded:")
        for t,f,n,o,c,ex in marks:
            print(f"  t={t:5.2f}s  f≈{f:7.1f} Hz → {n}{o} ({ex:.1f} Hz, {c:+.0f} cents)")
        print(f"\nAll 2-second tones saved to: {os.path.abspath(OUTPUT_DIR)}")
    print()

if __name__ == "__main__":
    try: 
        main()
    except KeyboardInterrupt: 
        print("\nAborted.")
