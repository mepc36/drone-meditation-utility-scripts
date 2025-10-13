#!/usr/bin/env python3
"""
Configurable Pink-Noise + Tone Looper
(single pan for both layers + numbered tweak menu + JSON persistence)

Persistence:
- Reads ./input/last_params.json at startup (if available) to restore your last settings.
- Writes ./input/last_params.json after each render/tweak so you can resume later.

Controls:
- Renders and plays combined audio (also saves stems) to ./output
- After each render/play:
    - Press Enter to repeat with the same settings
    - Type a number (1..11) to change that single parameter
    - Type 'q' to quit

Requirements:
  pip install numpy soundfile
Platform:
  macOS (uses 'afplay' for playback)
"""

import os
import json
import math
import tempfile
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf

# ---------------- Fixed engine settings & paths ----------------
SAMPLE_RATE_HZ = 48000
FADE_TIME_S = 0.02

OUTPUT_DIR = "./output"
INPUT_DIR  = "./input"
PARAMS_JSON_PATH = os.path.join(INPUT_DIR, "last_params.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR,  exist_ok=True)

# ---------------- Utilities ----------------
def db_to_linear(db_value: float) -> float:
    return 10.0 ** (db_value / 20.0)

def apply_fades_inplace(signal: np.ndarray, fade_time_s: float, sr: int) -> None:
    fade_samples = int(sr * fade_time_s)
    if fade_samples <= 0 or fade_samples * 2 >= signal.size:
        return
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float64)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float64)
    signal[:fade_samples] *= fade_in
    signal[-fade_samples:] *= fade_out

def normalize_peak_inplace(signal: np.ndarray, peak_target: float = 1.0) -> None:
    peak = float(np.max(np.abs(signal))) if signal.size else 1.0
    if peak > 0:
        signal *= (peak_target / peak)

def hard_pan_mono_to_stereo(mono: np.ndarray, side: str) -> np.ndarray:
    """Hard-pan mono signal left or right; any other value defaults to center."""
    side = side.lower()
    if side.startswith('l'):
        return np.column_stack([mono, np.zeros_like(mono)])
    elif side.startswith('r'):
        return np.column_stack([np.zeros_like(mono), mono])
    else:
        return np.column_stack([mono, mono])

def play_with_afplay(path: str) -> None:
    try:
        subprocess.run(["afplay", path], check=True)
    except Exception as e:
        print(f"(Playback warning) {e}")

# ---------------- Pink noise (band-limited) ----------------
def generate_band_limited_pink_noise(
    duration_s: float,
    sample_rate_hz: int,
    center_frequency_hz: float,
    bandwidth_hz: float,
) -> np.ndarray:
    """
    Pink-ish noise via frequency-domain shaping (|X(f)| ∝ 1/sqrt(f)),
    then rectangular band-pass around center ± bandwidth/2.
    """
    n_samples = int(duration_s * sample_rate_hz)
    white = np.random.normal(0.0, 1.0, n_samples).astype(np.float64)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate_hz)

    eps = 1.0  # avoid f=0 blowup
    pink_shaper = 1.0 / np.sqrt(np.maximum(freqs, eps))

    half_bw = max(1.0, bandwidth_hz * 0.5)
    lower_cut = max(0.0, center_frequency_hz - half_bw)
    upper_cut = min(sample_rate_hz / 2.0, center_frequency_hz + half_bw)
    band_mask = (freqs >= lower_cut) & (freqs <= upper_cut)

    shaped = spectrum * pink_shaper
    shaped *= band_mask.astype(np.float64)

    pink = np.fft.irfft(shaped, n=n_samples)
    normalize_peak_inplace(pink, 1.0)
    apply_fades_inplace(pink, FADE_TIME_S, sample_rate_hz)
    return pink.astype(np.float32)

# ---------------- Oscillators ----------------
def osc_sine(phase: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * np.pi * phase)

def osc_saw(phase: np.ndarray) -> np.ndarray:
    return 2.0 * (phase - np.floor(phase + 0.5))

def osc_square(phase: np.ndarray) -> np.ndarray:
    return np.where(phase % 1.0 < 0.5, 1.0, -1.0)

def osc_triangle(phase: np.ndarray) -> np.ndarray:
    saw = osc_saw(phase)
    return 2.0 * (np.abs(saw) - 0.5)

WAVEFORMS = {
    1: ("sine",     osc_sine),
    2: ("sawtooth", osc_saw),
    3: ("square",   osc_square),
    4: ("triangle", osc_triangle),
}

def generate_waveform(
    frequency_hz: float,
    duration_s: float,
    sample_rate_hz: int,
    waveform_id: int,
) -> np.ndarray:
    n = int(duration_s * sample_rate_hz)
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    phase = (frequency_hz * t)  # cycles
    _, osc = WAVEFORMS.get(waveform_id, WAVEFORMS[1])
    wave = osc(phase)
    normalize_peak_inplace(wave, 1.0)
    apply_fades_inplace(wave, FADE_TIME_S, sample_rate_hz)
    return wave.astype(np.float32)

# ---------------- Amplitude modulation (pulse) ----------------
def lfo_signal(shape_id: int, n_samples: int, sample_rate_hz: int, period_s: float) -> np.ndarray:
    """LFO in [0,1]. shape: 1=sine, 2=square, 3=triangle."""
    period_s = max(1e-3, period_s)
    t = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    phase = (t / period_s) % 1.0
    if shape_id == 2:   # square
        lfo = np.where(phase < 0.5, 1.0, 0.0)
    elif shape_id == 3: # triangle
        tri = np.where(phase < 0.5, phase * 2.0, 2.0 - 2.0 * phase)
        lfo = tri
    else:               # sine (default)
        lfo = 0.5 * (1.0 + np.sin(2.0 * np.pi * phase - np.pi/2.0))
    return lfo.astype(np.float64)

def apply_amplitude_modulation(
    signal: np.ndarray,
    sample_rate_hz: int,
    modulation_period_s: float,
    modulation_depth_0_to_1: float,
    modulation_shape_id: int,
) -> np.ndarray:
    depth = float(np.clip(modulation_depth_0_to_1, 0.0, 1.0))
    if depth <= 0.0:
        return signal
    lfo = lfo_signal(modulation_shape_id, signal.size, sample_rate_hz, modulation_period_s)
    mod_gain = (1.0 - depth) + depth * lfo
    return (signal.astype(np.float64) * mod_gain).astype(np.float32)

# ---------------- Render / Save / Play ----------------
def render_layers(params: Dict[str, Any]):
    """
    Returns (stereo_combined, stereo_pink, stereo_tone)
    """
    duration_s = float(params["duration_seconds"])

    # Pink noise (mono -> pan)
    pink_mono = generate_band_limited_pink_noise(
        duration_s=duration_s,
        sample_rate_hz=SAMPLE_RATE_HZ,
        center_frequency_hz=float(params["pink_center_frequency_hz"]),
        bandwidth_hz=float(params["pink_bandwidth_hz"]),
    )
    pink_mono = (db_to_linear(float(params["pink_level_db"])) * pink_mono).astype(np.float32)
    pink_stereo = hard_pan_mono_to_stereo(pink_mono, str(params["pan_side"]))

    # Tone (mono -> modulation -> pan SAME as noise)
    tone_mono = generate_waveform(
        frequency_hz=float(params["tone_frequency_hz"]),
        duration_s=duration_s,
        sample_rate_hz=SAMPLE_RATE_HZ,
        waveform_id=int(params["tone_waveform_id"]),
    )
    tone_mono = apply_amplitude_modulation(
        signal=tone_mono,
        sample_rate_hz=SAMPLE_RATE_HZ,
        modulation_period_s=float(params["modulation_period_s"]),
        modulation_depth_0_to_1=float(params["modulation_depth_0_to_1"]),
        modulation_shape_id=int(params["modulation_shape_id"]),
    )
    tone_mono = (db_to_linear(float(params["tone_level_db"])) * tone_mono).astype(np.float32)
    tone_stereo = hard_pan_mono_to_stereo(tone_mono, str(params["pan_side"]))

    # Mix safely
    mix = (pink_stereo.astype(np.float64) + tone_stereo.astype(np.float64))
    peak = float(np.max(np.abs(mix))) if mix.size else 1.0
    if peak > 0.99:
        mix *= (0.99 / peak)
    return mix.astype(np.float32), pink_stereo.astype(np.float32), tone_stereo.astype(np.float32)

def save_wavs(stereo_combined: np.ndarray, stereo_pink: np.ndarray, stereo_tone: np.ndarray, base_label: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{base_label}_{timestamp}"
    combined_path = os.path.join(OUTPUT_DIR, f"{base}_combined.wav")
    pink_path     = os.path.join(OUTPUT_DIR, f"{base}_pink.wav")
    tone_path     = os.path.join(OUTPUT_DIR, f"{base}_tone.wav")
    sf.write(combined_path, stereo_combined, SAMPLE_RATE_HZ)
    sf.write(pink_path,     stereo_pink,     SAMPLE_RATE_HZ)
    sf.write(tone_path,     stereo_tone,     SAMPLE_RATE_HZ)
    return combined_path, pink_path, tone_path

def play_wav(path: str):
    print("▶️ Playing combined render...")
    play_with_afplay(path)
    print("Done.\n")

# ---------------- Numbered-menu + validation schema ----------------
def param_schema() -> Dict[str, Dict[str, Any]]:
    """Validation schema: type, min/max/choices, human hint."""
    nyq = SAMPLE_RATE_HZ / 2.0
    return {
        "pan_side":                 {"type": str,   "choices": {"l","r"},             "hint": "'l' or 'r'"},
        "pink_center_frequency_hz": {"type": float, "min": 20.0, "max": nyq - 100.0,  "hint": "Hz"},
        "pink_bandwidth_hz":       {"type": float, "min": 10.0,  "max": nyq - 10.0,   "hint": "Hz"},
        "pink_level_db":           {"type": float, "min": -120.0,"max": 0.0,          "hint": "dBFS"},
        "tone_waveform_id":        {"type": int,   "choices": {1,2,3,4},              "hint": "1=sine,2=saw,3=square,4=triangle"},
        "tone_frequency_hz":       {"type": float, "min": 20.0, "max": nyq - 100.0,   "hint": "Hz"},
        "tone_level_db":           {"type": float, "min": -120.0,"max": 0.0,          "hint": "dBFS"},
        "modulation_period_s":     {"type": float, "min": 0.05, "max": 30.0,          "hint": "seconds"},
        "modulation_depth_0_to_1": {"type": float, "min": 0.0,  "max": 1.0,           "hint": "0..1"},
        "modulation_shape_id":     {"type": int,   "choices": {1,2,3},                "hint": "1=sine,2=square,3=triangle"},
        "duration_seconds":        {"type": float, "min": 0.1,  "max": 120.0,         "hint": "seconds"},
    }

def defaults_params() -> Dict[str, Any]:
    return {
        # Shared pan (applies to BOTH layers)
        "pan_side": "l",                 # 'l' or 'r'

        # Pink noise
        "pink_center_frequency_hz": 8000.0,
        "pink_bandwidth_hz":        1000.0,
        "pink_level_db":            -40.0,

        # Tone
        "tone_waveform_id":         1,        # 1=sine, 2=saw, 3=square, 4=triangle
        "tone_frequency_hz":        8000.0,   # defaults to pink center; tweak as needed
        "tone_level_db":            -30.0,

        # Pulse (AM)
        "modulation_period_s":      0.8,
        "modulation_depth_0_to_1":  0.5,
        "modulation_shape_id":      1,        # 1=sine, 2=square, 3=triangle

        # Shared duration
        "duration_seconds":         3.0,
    }

def ordered_param_list() -> List[Tuple[int, str]]:
    """
    Returns a numbered list of (menu_number, param_name) in the order we present.
    """
    names = [
        "pan_side",                   # 1
        "pink_center_frequency_hz",   # 2
        "pink_bandwidth_hz",          # 3
        "pink_level_db",              # 4
        "tone_waveform_id",           # 5
        "tone_frequency_hz",          # 6
        "tone_level_db",              # 7
        "modulation_period_s",        # 8
        "modulation_depth_0_to_1",    # 9
        "modulation_shape_id",        # 10
        "duration_seconds",           # 11
    ]
    return list(enumerate(names, start=1))

def print_numbered_menu(params: Dict[str, Any]) -> None:
    print("Change one parameter by typing its number, or press Enter to repeat, or 'q' to quit.")
    for idx, name in ordered_param_list():
        print(f"  {idx:>2}) {name} = {params[name]}")
    print()

def coerce_and_validate(name: str, value_str: str, schema: Dict[str, Any]):
    spec = schema[name]
    typ = spec["type"]
    try:
        val = typ(value_str)
    except Exception:
        raise ValueError(f"Expected {typ.__name__}")
    if "min" in spec and val < spec["min"]:
        raise ValueError(f"min {spec['min']}")
    if "max" in spec and val > spec["max"]:
        raise ValueError(f"max {spec['max']}")
    if "choices" in spec and val not in spec["choices"]:
        raise ValueError(f"choices {sorted(list(spec['choices']))}")
    return val

# ---------------- Persistence helpers ----------------
def load_params_from_json(path: str, schema: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load parameters from JSON if it exists; validate each field against the schema.
    Any missing or invalid values fall back to defaults.
    """
    params = defaults.copy()
    if not os.path.exists(path):
        return params

    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print("Warning: params JSON is not a dict; using defaults.")
            return params
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}; using defaults.")
        return params

    # Validate and merge
    for key, spec in schema.items():
        if key in data:
            try:
                # Coerce then check ranges/choices
                val = coerce_and_validate(key, str(data[key]), schema)
                params[key] = val
            except Exception:
                print(f"Warning: invalid value for '{key}' in JSON; using default {defaults[key]}")
        else:
            # Missing key -> keep default
            pass

    return params

def save_params_to_json(path: str, params: Dict[str, Any]) -> None:
    try:
        with open(path, "w") as f:
            json.dump(params, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to write params JSON: {e}")

# ---------------- Interactive tweak ----------------
def tweak_one_parameter_interactively(params: Dict[str, Any]) -> bool:
    """
    Shows the numbered menu and lets the user tweak exactly one parameter.
    Returns False to quit, True to continue rendering.
    """
    print_numbered_menu(params)
    choice = input("Change #> ").strip().lower()

    if choice == "q":
        return False
    if choice == "":
        return True

    # Accept either a number OR a parameter name
    schema = param_schema()
    mapping = dict(ordered_param_list())

    if choice.isdigit():
        idx = int(choice)
        if idx not in mapping:
            print("Invalid number. Try again.")
            return tweak_one_parameter_interactively(params)
        name = mapping[idx]
    else:
        name = choice
        if name not in schema:
            print("Unknown parameter name. Try again.")
            return tweak_one_parameter_interactively(params)

    current_val = params[name]
    hint = schema[name].get("hint", "")
    new_val_str = input(f"New value for {name} (current {current_val}, {hint}): ").strip()

    try:
        new_val = coerce_and_validate(name, new_val_str, schema)
    except ValueError as e:
        print(f"Invalid value: {e}")
        return tweak_one_parameter_interactively(params)

    params[name] = new_val
    print(f"{name} updated to {new_val}\n")
    return True

# ---------------- Main ----------------
def main():
    print("\n=== Configurable Pink Noise + Tone Looper (with resumeable JSON settings) ===")
    print("Headphones recommended. Keep volume LOW.\n")

    schema   = param_schema()
    defaults = defaults_params()
    params   = load_params_from_json(PARAMS_JSON_PATH, schema, defaults)

    render_index = 1
    while True:
        print("\nRendering with current parameters...")
        for _, key in ordered_param_list():
            print(f"  {key:>26} = {params[key]}")
        print()

        stereo_combined, stereo_pink, stereo_tone = render_layers(params)

        label = f"take{render_index:02d}"
        combined_path, pink_path, tone_path = save_wavs(stereo_combined, stereo_pink, stereo_tone, label)
        print(f"💾 Saved:\n  - {combined_path}\n  - {pink_path}\n  - {tone_path}\n")

        # Persist params after each render (so even if you quit right now, it's saved)
        save_params_to_json(PARAMS_JSON_PATH, params)

        play_wav(combined_path)

        # Numbered one-parameter tweak / continue / quit
        keep_going = tweak_one_parameter_interactively(params)
        # Persist immediately after a tweak decision
        save_params_to_json(PARAMS_JSON_PATH, params)
        if not keep_going:
            break

        render_index += 1

    print("\nGoodbye!")

if __name__ == "__main__":
    # Safety reminder
    print("⚠️ WARNING: Audio playback can be loud. Keep volume low.")
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
