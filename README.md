# Drone Meditation Utility Scripts

This project contains **two audio utility scripts** for high-frequency sound work and hearing analysis:

1. **Chromatic Waveform Generator** - Generates chromatic scales of high-frequency waveforms (sine, square, triangle, or saw) at precise equal-tempered pitches for digital synthesis, sound design, or testing high-frequency playback systems.

2. **Find Highest Audible Note** - An interactive hearing test that finds your highest audible pure tone using adaptive binary search, starting at 16 kHz and narrowing down to your hearing threshold.romatic Waveform Generator

This project generates **chromatic scales of high-frequency waveforms** — sine, square, triangle, or saw — at precise equal-tempered pitches.  
It’s designed for use in digital synthesis, sound design, or testing high-frequency playback systems.

By default, it produces **two chromatic octaves**:
- 4 kHz → 8 kHz  
- 8 kHz → 16 kHz  

Each tone is exported as an individual `.wav` file inside an `/output` folder.

---

## 🚀 Features

### Chromatic Waveform Generator
- Generates 12-tone equal-tempered chromatic scales
- Supports **sine**, **square**, **triangle**, and **sawtooth** waves
- Two octaves: one below and one above the root frequency (default 8 kHz)
- Configurable duration, fade-in/out, and duty cycle (for square waves)
- High sample rate (96 kHz) to minimize aliasing
- Auto-labels files with musical names (e.g. `sine__8000.00Hz_B8+21c.wav`)
- Writes all files to a clean `/output` directory

### Find Highest Audible Note
- Interactive hearing test using adaptive binary search
- Tests high-frequency range (8 kHz - 20 kHz by default)
- Randomized pre-play delays to reduce anticipation bias
- User-friendly controls (y/n/r/q for heard/not heard/replay/quit)
- Precise threshold detection (±25 Hz resolution)
- Works with macOS `afplay` for reliable audio playback

---

## 🧱 Project Structure
```
.
├── README.md
├── requirements.txt
├── find_highest_audible_note.py/
│   ├── find_highest_audible_note.py
│   └── input/
│       └── tmp_hearing_tone.wav
├── generate-chromatic-sine-waves/
│   ├── generate-chromatic-sine-waves.py
│   └── output/
│       ├── sine__4000.00Hz_B7+21c.wav
│       ├── sine__4237.85Hz_C8+21c.wav
│       ├── ...
│       └── sine__16000.00Hz_B9+21c.wav
```

---

## ⚙️ Installation

### 1. System Requirements
**macOS (Homebrew required):**
```bash
brew install libsndfile
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install libsndfile1
```

**Windows:**
- libsndfile is bundled with the Python soundfile package

### 2. Python Setup
```bash
# Clone or download this repository
cd drone-meditation-utility-scripts

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python Dependencies
pip install -r requirements.txt
```

**Note:** 
- If you get "externally-managed-environment" errors, make sure you're using a virtual environment (not system Python)
- If you get numpy dependency conflicts with existing packages (torch, scipy, etc.), the requirements.txt uses compatible versions that should work with most setups
- If issues persist, try creating a fresh virtual environment

---

## 🎚️ Usage

### Chromatic Waveform Generator
Run the script to generate waveforms:

```bash
cd generate-chromatic-sine-waves
python generate-chromatic-sine-waves.py
```

All output `.wav` files will appear in the `/output` folder.

### Find Highest Audible Note
Run the hearing test (headphones recommended, start with LOW volume):

```bash
cd find_highest_audible_note.py
python find_highest_audible_note.py
```

Follow the prompts:
- `y` = I heard it
- `n` = I did NOT hear it  
- `r` = replay the same tone
- `q` = quit

---

## 🎛️ Configuration

### Chromatic Waveform Generator
Inside `generate-chromatic-sine-waves.py`, you can adjust:

| Variable | Description | Default |
|-----------|--------------|----------|
| `root_frequency_hz` | Starting frequency (root note) | `8000.0` |
| `include_octave_below` | Include octave below (4 kHz → 8 kHz) | `True` |
| `include_root_octave` | Include octave above (8 kHz → 16 kHz) | `True` |
| `duration_seconds` | Duration of each tone | `0.5` |
| `fade_time_seconds` | Fade-in/out length | `0.01` |
| `waveform_type` | `"sine"`, `"square"`, `"triangle"`, or `"saw"` | `"sine"` |
| `square_duty_cycle` | Duty cycle for square/pulse wave | `0.5` |
| `sample_rate_hz` | Sampling rate | `96000` |

### Find Highest Audible Note
Inside `find_highest_audible_note.py`, you can adjust:

| Variable | Description | Default |
|-----------|--------------|----------|
| `lower_bound_hz` | Minimum test frequency | `8000.0` |
| `upper_bound_hz` | Maximum test frequency | `20000.0` |
| `start_frequency_hz` | Starting frequency for test | `16000.0` |
| `duration_seconds` | Duration of each tone | `1.25` |
| `resolution_hz` | Stop when range width ≤ this | `25.0` |
| `amplitude` | Tone volume (0.0-1.0) | `0.3` |
| `pre_play_delay_max_s` | Max random delay before tone | `2.0` |

---

## 📁 Output Examples

### Chromatic Waveform Generator
```
generate-chromatic-sine-waves/output/
├── sine__4000.00Hz_B7+21c.wav
├── sine__4237.85Hz_C8+21c.wav
├── sine__4489.85Hz_C#8+21c.wav
├── ...
├── sine__8000.00Hz_B8+21c.wav
├── ...
└── sine__16000.00Hz_B9+21c.wav
```

Each file name includes:
- Waveform type
- Exact frequency in Hz
- Musical note + octave
- Cent deviation (if any)

### Find Highest Audible Note
```
find_highest_audible_note.py/input/
└── tmp_hearing_tone.wav  # Temporary file used during testing
```

The script outputs your estimated hearing threshold in the terminal:
```
Estimated highest audible frequency ≈ 14250.0–14275.0 Hz (midpoint 14262.5 Hz).
```

---

## 🧠 Notes

### General
- Both scripts use a **96 kHz** sample rate to prevent aliasing
- Both require `numpy` and `soundfile` dependencies

### Chromatic Waveform Generator
- 8 kHz corresponds to **B8 (+21 cents)** — slightly sharp relative to equal-tempered B8
- For non-sine shapes, expect strong harmonics approaching the Nyquist limit (48 kHz)

### Find Highest Audible Note
- **Use headphones** and start with **low volume** to protect your hearing
- Test each ear separately for best results
- The random pre-play delay helps reduce anticipation bias
- macOS only (uses `afplay` for audio playback)

---

## 🧰 Optional Development Tools
If you plan to extend this project:
```bash
pip install black flake8 pytest
```

---

## 📄 License
MIT License © 2025 Your Name  
You are free to modify, distribute, and use this project for any purpose.

---

## 💡 Future Ideas

### Chromatic Waveform Generator
- Generate multi-octave sweeps or arpeggios
- Band-limit non-sine waves to reduce aliasing
- Export combined chromatic scales as single continuous `.wav`
- Add MIDI integration or Logic Pro sampler mapping automation

### Find Highest Audible Note
- Add support for Linux/Windows audio playback
- Test multiple frequency ranges (bass, mid, treble)
- Export hearing test results to file
- Add white/pink noise masking options
- Implement psychoacoustic testing methods
