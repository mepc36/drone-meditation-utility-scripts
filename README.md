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
