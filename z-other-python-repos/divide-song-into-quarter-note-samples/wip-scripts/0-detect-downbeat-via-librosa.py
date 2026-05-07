import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy.signal
from pathlib import Path

# Supported audio patterns
AUDIO_GLOB_PATTERNS = ["*.mp3", "*.wav", "*.flac", "*.m4a", "*.aac", "*.ogg", "*.wma"]

def find_single_audio_file(audio_dir: Path) -> Path:
    """Find a single audio file in the directory.
    
    Args:
        audio_dir: Directory to search
        
    Returns:
        Path to the audio file
        
    Raises:
        FileNotFoundError: If no audio files found
        ValueError: If multiple audio files found
    """
    audio_files = []
    for pattern in AUDIO_GLOB_PATTERNS:
        audio_files.extend(audio_dir.glob(pattern))
    
    if not audio_files:
        supported = ", ".join(AUDIO_GLOB_PATTERNS)
        raise FileNotFoundError(f"No audio files found in {audio_dir}\nSupported patterns: {supported}")
    
    if len(audio_files) > 1:
        files_list = "\n    ".join([f.name for f in audio_files])
        raise ValueError(
            f"Found {len(audio_files)} audio files in {audio_dir}, but only 1 is allowed.\n"
            f"  Found files:\n    {files_list}"
        )
    
    return audio_files[0]

# --- params ---
hop_length = 512

# Beat-stability params (tune these)
window_beats = 16              # how many consecutive beats must be stable
cv_threshold = 0.06            # coefficient of variation for beat intervals
bpm_min, bpm_max = 60, 200     # reject weird tempos

# Energy gate (optional but helpful)
rms_threshold = 0.02           # normalized-ish; tune per track

# --- find audio file ---
script_dir = Path(__file__).parent
input_dir = script_dir / "input"

if not input_dir.exists():
    print(f"Input directory not found at {input_dir}", file=sys.stderr)
    sys.exit(1)

# Find all song directories
song_dirs = [
    d for d in input_dir.iterdir()
    if d.is_dir() and d.name not in {".DS_Store", "prompts"}
]

if not song_dirs:
    print(f"No song directories found in {input_dir}", file=sys.stderr)
    sys.exit(1)

# Process first song directory found
song_dir = song_dirs[0]
song_name = song_dir.name
audio_dir = song_dir / "audio"

if not audio_dir.exists():
    print(f"Audio directory not found at {audio_dir}", file=sys.stderr)
    sys.exit(1)

input_path = find_single_audio_file(audio_dir)
print(f"Processing: {song_name}")
print(f"Audio file: {input_path}")

# --- load ---
y, sr = librosa.load(input_path, sr=None, mono=True)

# --- RMS (for plotting + optional gate) ---
rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
times_rms = librosa.times_like(rms, sr=sr, hop_length=hop_length)

# normalize RMS to [0,1] for a more stable threshold feel
rms_norm = rms / (np.max(rms) + 1e-12)

# --- beat tracking ---
# beats are frame indices; convert to seconds
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length, trim=False)
beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

# --- find first "downbeat" as start of stable beat grid ---
def find_first_stable_beat(beat_times, rms_norm, times_rms,
                           window_beats=16, cv_threshold=0.06,
                           bpm_min=60, bpm_max=200, rms_threshold=0.02):
    if len(beat_times) < window_beats + 1:
        return None, None

    # precompute beat-to-beat intervals
    ibis = np.diff(beat_times)

    for i in range(0, len(ibis) - window_beats + 1):
        w = ibis[i:i + window_beats]
        mean_ibi = np.mean(w)
        std_ibi = np.std(w)
        cv = std_ibi / (mean_ibi + 1e-12)

        bpm = 60.0 / (mean_ibi + 1e-12)

        # energy gate at/near the candidate beat (optional)
        t_candidate = beat_times[i]
        rms_at_t = np.interp(t_candidate, times_rms, rms_norm)

        if (cv <= cv_threshold) and (bpm_min <= bpm <= bpm_max) and (rms_at_t >= rms_threshold):
            return t_candidate, {"cv": cv, "bpm": bpm, "rms": rms_at_t, "idx": i}

    return None, None

downbeat_t, info = find_first_stable_beat(
    beat_times, rms_norm, times_rms,
    window_beats=window_beats,
    cv_threshold=cv_threshold,
    bpm_min=bpm_min, bpm_max=bpm_max,
    rms_threshold=rms_threshold,
)

# --- plotting ---
dur = librosa.get_duration(y=y, sr=sr)

plt.figure(figsize=(12, 4))
plt.plot(times_rms, rms_norm, label="RMS (normalized)")
plt.axhline(rms_threshold, linestyle="--", label="RMS gate")

# show detected beats as vertical ticks
for t in beat_times:
    plt.axvline(t, alpha=0.08)

# mark detected first downbeat (stable beat start)
if downbeat_t is not None:
    plt.axvline(downbeat_t, linewidth=2, label=f"First stable downbeat @ {downbeat_t:.2f}s")
    plt.title(f"Downbeat (stable beat start) for {os.path.basename(input_path)}  |  bpm≈{info['bpm']:.1f}, cv={info['cv']:.3f}")
else:
    plt.title(f"No stable downbeat found (try loosening cv_threshold / window_beats) — {os.path.basename(input_path)}")

plt.xlim(0, np.ceil(dur))
plt.xlabel("Time (s)")
plt.ylabel("RMS (norm)")
plt.legend()
plt.tight_layout()
plt.show()

print("Estimated tempo from librosa.beat.beat_track:", tempo)
print("First stable downbeat time:", downbeat_t)
print("Details:", info)
