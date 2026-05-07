import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import torch
from music_interpolation import EncodecInterpolation

interp = EncodecInterpolation()

# Load and pad input audio
audio_a, _ = librosa.load('./scripts/input/sample_a.wav', sr=interp.sampling_rate, mono=False)
audio_b, _ = librosa.load('./scripts/input/sample_b.wav', sr=interp.sampling_rate, mono=False)

max_len = max(audio_a.shape[-1], audio_b.shape[-1])
if audio_a.shape[-1] < max_len:
    audio_a = np.pad(audio_a, ((0, 0), (0, max_len - audio_a.shape[-1])))
if audio_b.shape[-1] < max_len:
    audio_b = np.pad(audio_b, ((0, 0), (0, max_len - audio_b.shape[-1])))

# Compute true latent length by encoding audio_a
audio_tensor = torch.tensor(audio_a).unsqueeze(0)  # shape: (1, channels, samples)
with torch.no_grad():
    latent = interp.model.encoder(audio_tensor.float())
latent_len = latent.shape[0]

# Interpolate across latent space
num_steps = 100
segments = []
for t in np.linspace(0.0, 1.0, num_steps):
    t_coeffs = np.full((latent_len,), t)
    segment = interp.interpolate(audio_a, audio_b, t_coeffs=t_coeffs)
    segments.append(segment)

# Concatenate and export
traversal_audio = np.concatenate(segments, axis=1)
sf.write("./scripts/output/latent_traversal.wav", traversal_audio.T.astype(np.float32), interp.sampling_rate)
AudioSegment.from_wav("./scripts/output/latent_traversal.wav").export("./scripts/output/latent_traversal.mp3", format="mp3")
Path("./scripts/output/latent_traversal.wav").unlink()
