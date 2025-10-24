
#!/usr/bin/env python3
"""
Configurable Colored-Noise + Tone Looper
(single pan for both layers + numbered tweak menu + JSON persistence)
Saves three files per render: combined, noise-only, and tinnitus-only.

Each render session creates a uniquely named directory in ./output containing:
- Combined audio (colored noise + tinnitus tone)
- Noise only 
- Tinnitus tone only
Files are named with descriptive parameters for easy identification.

Persistence:
- Reads ./input/most_recent_config.json at startup (if available) to restore your last settings.
  (Backwards-compatible: will also read legacy ./input/last_params.json and map old keys to new.)
- Writes ./input/most_recent_config.json after each render/tweak so you can resume later.

Controls:
- Renders and plays combined audio, saves all three variants to ./output
- After each render/play:
    - Press Enter to repeat with the same settings
    - Type a number to change that single parameter
    - Type 'q' to quit

Requirements:
  pip install numpy soundfile
Platform:
  macOS (uses 'afplay' for playback)
"""

import os
import json
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
CONFIG_JSON_PATH = os.path.join(INPUT_DIR, "most_recent_config.json")  # renamed
LEGACY_JSON_PATH = os.path.join(INPUT_DIR, "last_params.json")         # legacy (read-only)

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
    """Hard-pan mono signal left, right, or center."""
    side = side.lower()
    if side.startswith('l'):
        return np.column_stack([mono, np.zeros_like(mono)])
    elif side.startswith('r'):
        return np.column_stack([np.zeros_like(mono), mono])
    else:  # center or any other value
        return np.column_stack([mono, mono])

def play_with_afplay(path: str, duration_seconds: float = None) -> None:
    try:
        if duration_seconds is not None and duration_seconds > 0:
            subprocess.run(["afplay", path], check=True, timeout=duration_seconds)
        else:
            subprocess.run(["afplay", path], check=True)
    except subprocess.TimeoutExpired:
        print(f"(Playback stopped after {duration_seconds} seconds)")
    except Exception as e:
        print(f"(Playback warning) {e}")

# ---------------- Colored noise (band-limited by center/bandwidth) ----------------
def generate_band_limited_colored_noise(
    duration_s: float,
    sample_rate_hz: int,
    center_frequency_hz: float,
    bandwidth_hz: float,
    alpha: float,
) -> np.ndarray:
    """
    General colored noise via frequency-domain shaping:
      amplitude |X(f)| ∝ 1 / f^alpha
        alpha = 0.0  -> white (flat amplitude; flat power per Hz)
        alpha = 0.5  -> pink  (≈ -3 dB/oct power)
        alpha = 1.0  -> brown (≈ -6 dB/oct power)

    Then apply a rectangular band-pass around center ± bandwidth/2.
    """
    n_samples = int(duration_s * sample_rate_hz)
    white = np.random.normal(0.0, 1.0, n_samples).astype(np.float64)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate_hz)

    eps = 1.0  # avoid f=0 blowup and overweighting ultra-low bins in short renders
    amplitude_shaper = 1.0 / np.power(np.maximum(freqs, eps), float(alpha))

    half_bw = max(1.0, bandwidth_hz * 0.5)
    lower_cut = max(0.0, center_frequency_hz - half_bw)
    upper_cut = min(sample_rate_hz / 2.0, center_frequency_hz + half_bw)
    band_mask = (freqs >= lower_cut) & (freqs <= upper_cut)

    shaped = spectrum * amplitude_shaper
    shaped *= band_mask.astype(np.float64)

    noise = np.fft.irfft(shaped, n=n_samples)
    normalize_peak_inplace(noise, 1.0)
    apply_fades_inplace(noise, FADE_TIME_S, sample_rate_hz)
    return noise.astype(np.float32)

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
def create_tinnitus_sounds(params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build colored noise + tone (both panned by the same pan_side) and return:
    (stereo_combined, stereo_noise_only, stereo_tone_only)
    """
    duration_s = float(params["duration_seconds"])

    # Noise (mono -> pan)
    noise_mono = generate_band_limited_colored_noise(
        duration_s=duration_s,
        sample_rate_hz=SAMPLE_RATE_HZ,
        center_frequency_hz=float(params["noise_center_frequency_hz"]),
        bandwidth_hz=float(params["noise_bandwidth_hz"]),
        alpha=float(params["noise_spectral_slope"]),
    )
    noise_mono = (db_to_linear(float(params["noise_level_db"])) * noise_mono).astype(np.float32)
    noise_stereo = hard_pan_mono_to_stereo(noise_mono, str(params["pan_side"]))

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

    # Mix safely for combined
    mix = (noise_stereo.astype(np.float64) + tone_stereo.astype(np.float64))
    peak = float(np.max(np.abs(mix))) if mix.size else 1.0
    if peak > 0.99:
        mix *= (0.99 / peak)
    combined = mix.astype(np.float32)

    return combined, noise_stereo, tone_stereo

def _color_name_from_alpha(alpha: float) -> str:
    # helper for filenames
    if abs(alpha - 0.0) < 1e-6:
        return "white"
    if abs(alpha - 0.5) < 1e-6:
        return "pink"
    if abs(alpha - 1.0) < 1e-6:
        return "brown"
    return f"a{alpha:.2f}"

def generate_filename_from_params(params: Dict[str, Any], file_type: str) -> str:
    """Generate descriptive filename based on parameters."""
    waveform_names = {1: "sine", 2: "saw", 3: "square", 4: "triangle"}
    modulation_names = {1: "sine", 2: "square", 3: "triangle"}

    waveform = waveform_names.get(int(params["tone_waveform_id"]), "unknown")
    mod_shape = modulation_names.get(int(params["modulation_shape_id"]), "unknown")

    noise_freq = params["noise_center_frequency_hz"]
    tone_freq = params["tone_frequency_hz"]
    noise_freq_str = f"{int(noise_freq)}" if noise_freq == int(noise_freq) else f"{noise_freq:.1f}"
    tone_freq_str = f"{int(tone_freq)}" if tone_freq == int(tone_freq) else f"{tone_freq:.1f}"

    color = _color_name_from_alpha(float(params["noise_spectral_slope"]))

    filename_parts = [
        f"noise{noise_freq_str}hz",
        f"bw{int(params['noise_bandwidth_hz'])}hz",
        f"{color}",
        f"alpha{float(params['noise_spectral_slope']):.2f}",
        f"{int(params['noise_level_db'])}db",
        f"tone{tone_freq_str}hz",
        waveform,
        f"{int(params['tone_level_db'])}db",
        f"mod{params['modulation_period_s']:.1f}s",
        f"depth{params['modulation_depth_0_to_1']:.1f}",
        mod_shape,
        f"pan{params['pan_side']}",
        f"{params['duration_seconds']:.1f}s",
        file_type
    ]

    return "_".join(filename_parts) + ".wav"

def save_all_wavs(combined: np.ndarray, noise: np.ndarray, tone: np.ndarray, 
                  params: Dict[str, Any], base_label: str) -> Tuple[str, str, str, str]:
    """
    Save all three audio files in a uniquely named directory.
    Returns (session_dir, combined_path, noise_path, tone_path)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(OUTPUT_DIR, f"{base_label}_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)

    # Generate descriptive filenames with numbered prefixes
    combined_filename = "1-" + generate_filename_from_params(params, "combined")
    tone_filename = "2-" + generate_filename_from_params(params, "tone_only")
    noise_filename = "3-" + generate_filename_from_params(params, "noise_only")

    combined_path = os.path.join(session_dir, combined_filename)
    noise_path = os.path.join(session_dir, noise_filename)
    tone_path = os.path.join(session_dir, tone_filename)

    # Save all files
    sf.write(combined_path, combined, SAMPLE_RATE_HZ)
    sf.write(noise_path, noise, SAMPLE_RATE_HZ)
    sf.write(tone_path, tone, SAMPLE_RATE_HZ)
    
    # Save configuration file in the same directory
    config_path = os.path.join(session_dir, "config.json")
    try:
        with open(config_path, "w") as f:
            json.dump(params, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to save config.json in session directory: {e}")

    return session_dir, combined_path, noise_path, tone_path

def play_wav(path: str):
    # Ask user how many seconds they want to play
    while True:
        try:
            duration_input = input("How many seconds to play? (Enter for full length, 0 to skip): ").strip()
            if duration_input == "":
                duration = None
                break
            elif duration_input == "0":
                print("Skipping playback.\n")
                return
            else:
                duration = float(duration_input)
                if duration < 0:
                    print("Please enter a positive number or 0 to skip.")
                    continue
                break
        except ValueError:
            print("Please enter a valid number.")
            continue

    if duration is None:
        print("▶️ Playing full combined render...")
    else:
        print(f"▶️ Playing {duration} seconds of combined render...")

    play_with_afplay(path, duration)
    print("Done.\n")

# ---------------- Numbered-menu + validation schema ----------------
def param_schema() -> Dict[str, Dict[str, Any]]:
    """Validation schema: type, min/max/choices, human hint."""
    nyq = SAMPLE_RATE_HZ / 2.0
    return {
        "pan_side":                 {"type": str,   "choices": {"l","r","c"},         "hint": "'l', 'r', or 'c' (center)"},
        "noise_center_frequency_hz":{"type": float, "min": 20.0, "max": nyq - 100.0,  "hint": "Hz"},
        "noise_bandwidth_hz":      {"type": float, "min": 10.0,  "max": nyq - 10.0,   "hint": "Hz"},
        "noise_spectral_slope":             {"type": float, "min": 0.0,  "max": 1.5,           "hint": "0.0=white, 0.5=pink, 1.0=brown"},
        "noise_level_db":          {"type": float, "min": -120.0,"max": 0.0,          "hint": "dBFS"},
        "tone_waveform_id":        {"type": int,   "choices": {1,2,3,4},              "hint": "1=sine,2=saw,3=square,4=triangle"},
        "tone_frequency_hz":       {"type": float, "min": 20.0, "max": nyq - 100.0,   "hint": "Hz"},
        "tone_level_db":           {"type": float, "min": -120.0,"max": 0.0,          "hint": "dBFS"},
        "modulation_period_s":     {"type": float, "min": 0.05, "max": 30.0,          "hint": "seconds"},
        "modulation_depth_0_to_1": {"type": float, "min": 0.0,  "max": 1.0,           "hint": "0..1"},
        "modulation_shape_id":     {"type": int,   "choices": {1,2,3},                "hint": "1=sine, 2=square, 3=triangle"},
        "duration_seconds":        {"type": float, "min": 0.1,  "max": 240.0,         "hint": "seconds"},
    }

def defaults_params() -> Dict[str, Any]:
    return {
        # Shared pan (applies to BOTH layers)
        "pan_side": "l",                 # 'l' or 'r'

        # Colored noise
        "noise_center_frequency_hz": 8000.0,
        "noise_bandwidth_hz":        1000.0,
        "noise_spectral_slope":               0.5,    # pink by default
        "noise_level_db":            -40.0,

        # Tone
        "tone_waveform_id":          1,        # 1=sine, 2=saw, 3=square, 4=triangle
        "tone_frequency_hz":         8000.0,   # defaults to noise center; tweak as needed
        "tone_level_db":             -30.0,

        # Pulse (AM)
        "modulation_period_s":       0.8,
        "modulation_depth_0_to_1":   0.5,
        "modulation_shape_id":       1,        # 1=sine, 2=square, 3=triangle

        # Shared duration
        "duration_seconds":          3.0,
    }

def ordered_param_list() -> List[Tuple[int, str]]:
    names = [
        "pan_side",                    # 1
        "noise_center_frequency_hz",   # 2
        "noise_bandwidth_hz",          # 3
        "noise_spectral_slope",                 # 4  (added here, after bw_hz)
        "noise_level_db",              # 5
        "tone_waveform_id",            # 6
        "tone_frequency_hz",           # 7
        "tone_level_db",               # 8
        "modulation_period_s",         # 9
        "modulation_depth_0_to_1",     # 10
        "modulation_shape_id",         # 11
        "duration_seconds",            # 12
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
def _migrate_legacy_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """Map legacy 'pink_*' keys to new 'noise_*' keys if present."""
    key_map = {
        "pink_center_frequency_hz": "noise_center_frequency_hz",
        "pink_bandwidth_hz":       "noise_bandwidth_hz",
        "pink_level_db":           "noise_level_db",
    }
    migrated = data.copy()
    for old, new in key_map.items():
        if old in migrated and new not in migrated:
            migrated[new] = migrated[old]
    # Legacy had no alpha; assume pink default if missing
    if "noise_spectral_slope" not in migrated:
        migrated["noise_spectral_slope"] = 0.5
    return migrated

def load_params_from_json(path: str, schema: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load parameters from JSON if it exists; validate each field against the schema.
    Any missing or invalid values fall back to defaults.
    Also supports migrating from legacy last_params.json with pink_* keys.
    """
    params = defaults.copy()

    data = None
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: failed to read {path}: {e}; using defaults.")
    elif os.path.exists(LEGACY_JSON_PATH):
        try:
            with open(LEGACY_JSON_PATH, "r") as f:
                data = json.load(f)
            print("Loaded legacy last_params.json; migrating keys to noise_* and saving to most_recent_config.json.")
        except Exception as e:
            print(f"Warning: failed to read legacy {LEGACY_JSON_PATH}: {e}")

    if isinstance(data, dict):
        data = _migrate_legacy_keys(data)
        # Validate and merge
        for key in schema.keys():
            if key in data:
                try:
                    val = coerce_and_validate(key, str(data[key]), schema)
                    params[key] = val
                except Exception:
                    print(f"Warning: invalid value for '{key}' in JSON; using default {defaults[key]}")
        # Save migrated config to the new path
        try:
            with open(CONFIG_JSON_PATH, "w") as f:
                json.dump(params, f, indent=2)
        except Exception as e:
            print(f"Warning: failed to write migrated config: {e}")

    return params

def save_config_to_json(path: str, params: Dict[str, Any]) -> None:
    try:
        with open(path, "w") as f:
            json.dump(params, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to write params JSON: {e}")

# ---------------- Interactive tweak ----------------
def tweak_parameter(params: Dict[str, Any]) -> bool:
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
            return tweak_parameter(params)
        name = mapping[idx]
    else:
        name = choice
        if name not in schema:
            print("Unknown parameter name. Try again.")
            return tweak_parameter(params)

    current_val = params[name]
    hint = schema[name].get("hint", "")
    new_val_str = input(f"New value for {name} (current {current_val}, {hint}): ").strip()

    try:
        new_val = coerce_and_validate(name, new_val_str, schema)
    except ValueError as e:
        print(f"Invalid value: {e}")
        return tweak_parameter(params)

    params[name] = new_val
    print(f"{name} updated to {new_val}\n")
    return True

# ---------------- Main ----------------
def main():
    print("\n=== Configurable Colored Noise + Tone Looper (combined-only + resumable JSON settings) ===")
    print("Headphones recommended. Keep volume LOW.\n")

    schema   = param_schema()
    defaults = defaults_params()
    params   = load_params_from_json(CONFIG_JSON_PATH, schema, defaults)

    render_index = 1
    while True:
        print("\nRendering with current parameters...")
        for _, key in ordered_param_list():
            print(f"  {key:>26} = {params[key]}")
        print()

        combined, noise, tone = create_tinnitus_sounds(params)

        label = f"take{render_index:02d}"
        session_dir, combined_path, noise_path, tone_path = save_all_wavs(combined, noise, tone, params, label)

        print(f"💾 Saved session to: {session_dir}")
        print(f"   📁 Combined: {os.path.basename(combined_path)}")
        print(f"   📁 Noise only: {os.path.basename(noise_path)}")
        print(f"   📁 Tinnitus only: {os.path.basename(tone_path)}\n")

        # Persist params after each render (so even if you quit right now, it's saved)
        save_config_to_json(CONFIG_JSON_PATH, params)

        play_wav(combined_path)

        # Numbered one-parameter tweak / continue / quit
        keep_going = tweak_parameter(params)
        # Persist immediately after a tweak decision
        save_config_to_json(CONFIG_JSON_PATH, params)
        if not keep_going:
            break

        render_index += 1

    print("\nGoodbye!")

if __name__ == "__main__":
    # Safety reminder
    print("⚠️ WARNING: Audio playback can be loud.")
    print("Please turn down your volume to its lowest possible, non-muted setting.")
    input("Then press Enter to continue...")
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
