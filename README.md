# Drone Meditation Utility Scripts

These scripts were created to aid in the composition of "XVII - Drone/Meditation", by Martin "Maestro" Connor.

1. **Chromatic Waveform Generator** - Generates chromatic scales of high-frequency waveforms (sine, square, triangle, or saw) at precise equal-tempered pitches for digital synthesis, sound design, or testing high-frequency playback systems.

2. **Find Highest Audible Note** - An interactive hearing test that finds your highest audible pure tone using adaptive binary search, starting at 16 kHz and narrowing down to your hearing threshold.romatic Waveform Generator

This project generates **chromatic scales of high-frequency waveforms** — sine, square, triangle, or saw — at precise equal-tempered pitches.  
It’s designed for use in digital synthesis, sound design, or testing high-frequency playback systems.

By default, it produces **two chromatic octaves**:
- 4 kHz → 8 kHz  
- 8 kHz → 16 kHz  

Each tone is exported as an individual `.wav` file inside an `/output` folder.

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

Follow the prompts


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
