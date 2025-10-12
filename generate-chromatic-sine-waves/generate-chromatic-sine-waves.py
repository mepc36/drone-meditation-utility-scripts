import numpy as np
import soundfile as sf
import os
import math

# === CONFIGURATION ===
root_frequency_hz = 8000.0            # treat as "B" root
include_octave_below = True           # add 4 kHz → 8 kHz chromatic
include_root_octave = True            # keep 8 kHz → 16 kHz chromatic
duration_seconds = 0.5
fade_time_seconds = 0.01
sample_rate_hz = 96000
waveform_type = "sine"                # "sine", "square", "triangle", "saw"
square_duty_cycle = 0.5
output_level_db = -30.0               # Output level in decibels (0 dB = full scale, negative = quieter)

# === CONSTANTS ===
concert_a_ref_hz = 440.0
concert_a_midi_number = 69
note_names_sharp = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
pitch_ratio_per_semitone = 2 ** (1/12)

# === OUTPUT FOLDER (always in root) ===
output_directory = os.path.join(os.getcwd(), "output")
os.makedirs(output_directory, exist_ok=True)

# === UTILITIES ===
def frequency_to_midi_and_cents(frequency_hz, ref_hz=concert_a_ref_hz, ref_midi=concert_a_midi_number):
    """Return nearest MIDI note and cents deviation for a given frequency."""
    if frequency_hz <= 0:
        raise ValueError("Frequency must be positive.")
    exact_midi = ref_midi + 12.0 * math.log2(frequency_hz / ref_hz)
    nearest_midi = int(round(exact_midi))
    cents_offset = (exact_midi - nearest_midi) * 100.0
    return nearest_midi, cents_offset

def midi_to_note_name_and_octave(midi_number):
    """Convert MIDI number to note name and octave number."""
    note_index = midi_number % 12
    octave_number = (midi_number // 12) - 1
    return note_names_sharp[note_index], octave_number

def fractional_part(x):
    """Return fractional part of x (x - floor(x)) as vectorized operation."""
    return x - np.floor(x)

def create_waveform(frequency_hz,
                    duration_seconds,
                    sample_rate_hz,
                    fade_time_seconds,
                    waveform_type="sine",
                    square_duty_cycle=0.5):
    """
    Generate waveform ('sine', 'square', 'triangle', 'saw') with fade-in/out and normalization.
    """
    sample_count = int(sample_rate_hz * duration_seconds)
    time_axis = np.linspace(0, duration_seconds, sample_count, endpoint=False)

    if waveform_type.lower() == "sine":
        waveform = np.sin(2 * np.pi * frequency_hz * time_axis)
    elif waveform_type.lower() == "square":
        phase_cycles = fractional_part(frequency_hz * time_axis)
        waveform = np.where(phase_cycles < square_duty_cycle, 1.0, -1.0)
    elif waveform_type.lower() == "triangle":
        phase_cycles = fractional_part(frequency_hz * time_axis)
        waveform = 4.0 * np.abs(phase_cycles - 0.5) - 1.0
    elif waveform_type.lower() == "saw":
        phase_cycles = fractional_part(frequency_hz * time_axis)
        waveform = 2.0 * phase_cycles - 1.0
    else:
        raise ValueError("Unsupported waveform_type. Use 'sine', 'square', 'triangle', or 'saw'.")

    waveform = waveform.astype(np.float32)

    # apply fades
    fade_samples = int(sample_rate_hz * fade_time_seconds)
    if fade_samples > 0:
        fade_in_curve = np.linspace(0, 1, fade_samples, dtype=np.float32)
        fade_out_curve = np.linspace(1, 0, fade_samples, dtype=np.float32)
        waveform[:fade_samples] *= fade_in_curve
        waveform[-fade_samples:] *= fade_out_curve

    # normalize and apply amplitude scaling
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform /= peak
    # Convert dB to linear amplitude: amplitude = 10^(dB/20)
    output_amplitude = 10.0 ** (output_level_db / 20.0)
    waveform *= output_amplitude
    return waveform

# === SEMITONE OFFSETS (one octave below and root) ===
semitone_offsets = []
if include_octave_below:
    semitone_offsets.extend(range(-12, 0))   # 4k → just below 8k
if include_root_octave:
    semitone_offsets.extend(range(0, 13))    # 8k → 16k inclusive

relative_cycle_from_B = ["B","C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# === GENERATE AND WRITE ===
for semitone_offset in semitone_offsets:
    frequency_hz = root_frequency_hz * (pitch_ratio_per_semitone ** semitone_offset)
    relative_name = relative_cycle_from_B[semitone_offset % 12]

    nearest_midi, cents_offset = frequency_to_midi_and_cents(frequency_hz)
    note_name_from_midi, octave_number = midi_to_note_name_and_octave(nearest_midi)

    cents_sign = "+" if cents_offset >= 0 else "-"
    cents_abs = abs(cents_offset)
    cents_str = f"{cents_sign}{cents_abs:.0f}c" if cents_abs >= 0.5 else ""

    musical_label = f"{relative_name}{octave_number}{cents_str}"
    filename = f"{waveform_type}__{frequency_hz:.2f}Hz_{musical_label}.wav"
    file_path = os.path.join(output_directory, filename)

    tone = create_waveform(
        frequency_hz,
        duration_seconds,
        sample_rate_hz,
        fade_time_seconds,
        waveform_type=waveform_type,
        square_duty_cycle=square_duty_cycle
    )

    sf.write(file_path, tone, sample_rate_hz)
    print(f"✅ Wrote {file_path}")

print("\n🎧 Done! All tones are in:", output_directory)
