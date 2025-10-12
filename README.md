# Chromatic Waveform Generator

This project generates **chromatic scales of high-frequency waveforms** — sine, square, triangle, or saw — at precise equal-tempered pitches.  
It’s designed for use in digital synthesis, sound design, or testing high-frequency playback systems.

By default, it produces **two chromatic octaves**:
- 4 kHz → 8 kHz  
- 8 kHz → 16 kHz  

Each tone is exported as an individual `.wav` file inside an `/output` folder.

---

## 🚀 Features
- Generates 12-tone equal-tempered chromatic scales
- Supports **sine**, **square**, **triangle**, and **sawtooth** waves
- Two octaves: one below and one above the root frequency (default 8 kHz)
- Configurable duration, fade-in/out, and duty cycle (for square waves)
- High sample rate (96 kHz) to minimize aliasing
- Auto-labels files with musical names (e.g. `sine__8000.00Hz_B8+21c.wav`)
- Writes all files to a clean `/output` directory

---

## 🧱 Project Structure
```
.
├── generate_chromatic_sines_two_octaves_output.py
├── requirements.txt
├── .gitignore
├── README.md
└── output/
    ├── sine__4000.00Hz_B7.wav
    ├── sine__4244.92Hz_C8.wav
    ├── ...
    └── sine__16000.00Hz_B9.wav
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
cd generate-chromatic-sine-waves

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Note:** If you get numpy dependency conflicts with existing packages (torch, scipy, etc.), the requirements.txt uses compatible versions that should work with most setups. If issues persist, try installing in a fresh virtual environment.

---

## 🎚️ Usage

Run the script to generate waveforms:

```bash
python generate_chromatic_sines_two_octaves_output.py
```

All output `.wav` files will appear in the `/output` folder.

---

## 🎛️ Configuration

Inside `generate_chromatic_sines_two_octaves_output.py`, you can adjust:

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

---

## 📁 Output Example

```
output/
├── sine__4000.00Hz_B7.wav
├── sine__4244.92Hz_C8.wav
├── sine__4500.83Hz_C#8.wav
├── ...
├── sine__8000.00Hz_B8.wav
├── ...
└── sine__16000.00Hz_B9.wav
```

Each file name includes:
- Waveform type
- Exact frequency in Hz
- Musical note + octave
- Cent deviation (if any)

---

## 🧠 Notes
- 8 kHz corresponds to **B8 (+21 cents)** — slightly sharp relative to equal-tempered B8.
- A sample rate of **96 kHz** is recommended to prevent aliasing.
- For non-sine shapes, expect strong harmonics approaching the Nyquist limit (48 kHz).

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
- Generate multi-octave sweeps or arpeggios
- Band-limit non-sine waves to reduce aliasing
- Export combined chromatic scales as single continuous `.wav`
- Add MIDI integration or Logic Pro sampler mapping automation
