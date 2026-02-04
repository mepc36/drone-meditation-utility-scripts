# Generate Weighted Playlist

## Summary

This project creates weighted meditation playlists from audio samples using a multi-stage pipeline. It combines audio samples with stereo panning effects, imports them into Apple Music, generates M3U playlists, and provides cleanup utilities. The system creates unique stereo combinations with randomized panning patterns for an immersive listening experience.

## Installation

### Requirements

- Python 3.x
- macOS with Apple Music/iTunes

### Setup

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Place your audio files in:
./input/audio/

# Configure settings in:
./input/config/config.json
```

To deactivate the virtual environment when finished:

```bash
deactivate
```

## Configuration

Edit [input/config/config.json](input/config/config.json) to customize:

- `num_unique_samples`: Number of unique sample combinations to generate (default: 100)
- `min_samples_per_combination`: Minimum samples per combination (default: 1)
- `max_samples_per_combination`: Maximum samples per combination (default: 3)
- `center_to_noncenter_to_dualpan_ratio`: Ratio of panning patterns as "Center:NonCenter:DualPan" (e.g., "1:4:2")
- `samples_to_silence_ratio`: Ratio of samples to silence as "Samples:Silence" (e.g., "12:1")
- `bpm`: Beats per minute for rhythm calculation (default: 52)
- `silent_samples_length_millisec`: Length of silence samples in milliseconds (default: 12500)
- `source_dir`: Path to source audio files for import
- `itunes_dir`: Path where iTunes/Music imports files
- `playlist_name`: Name of the generated playlist

### Run All Scripts (Recommended)

```bash
# Run the complete pipeline
python3 run-all.py
```

This executes the full workflow:
1. Cleans up previous files (script 3)
2. Combines samples with panning (script 1)
3. Imports to Music and creates playlist (script 2)

### Run Scripts Individually

```bash
# 1. Combine audio samples with stereo panning
python3 1-combine-samples-with-panning.py

# 2. Import samples to Music and generate playlist
python3 2-import-duplicate-padded-samples-into-itunes-playlist.py

# 3. Clean up files and library entries (optional)
python3 3-clean-up-itunes-playlist-tracks-and-files.py
```

### Utility Scripts

```bash
# Calculate expected session length
python3 utility_scripts/5-calculate-length-of-playlist.py
```

## Scripts

### 1-combine-samples-with-panning.py

Creates unique stereo combinations of audio samples with randomized panning patterns. This is the main generation script that produces the final audio samples.

- **Input:** Audio files from `./input/audio/`
- **Output:** Combined stereo audio files in `./output/audio/final-sample-versions/`
- **Features:**
  - Combines 1-3 samples per output file (configurable)
  - Three panning patterns: center-only, non-center-only (hard left/right), and dualpan (2 different samples)
  - Generates silence samples based on configured ratio
  - All samples normalized to consistent beat length
  - Supports SOLO samples (isolated, can pan left/center/right, can repeat)
  - Supports ONCE samples (isolated, centered only, appear only once in entire playlist)
  - Groups samples by sound type for coherent combinations
  - Automatic RMS normalization for consistent volume across all outputs
  - Random sample selection within each sound type category

### 2-import-duplicate-padded-samples-into-itunes-playlist.py

Imports generated audio files into Apple Music and creates an M3U playlist file.

- **Input:** Final sample versions from `./output/audio/final-sample-versions/`
- **Output:** 
  - M3U playlist in `./output/playlists/`
  - Tracks imported to Music app at configured `itunes_dir`
- **Features:**
  - Automatically deletes existing playlist before creating new one (prevents duplicates)
  - Batch imports entire output folder in single operation for efficiency
  - Generates M3U playlist file with references to all imported tracks
  - Verifies files exist in iTunes directory after import
  - Auto-opens playlist in Music app when complete
  - Displays import statistics and next steps

### 3-clean-up-itunes-playlist-tracks-and-files.py

Complete cleanup utility that removes all generated files and library entries. Run this when you want to start fresh.

- **Input:** Generated audio files and Music library entries
- **Output:** 
  - Deleted physical files from output directory
  - Deleted physical files from iTunes import directory
  - Removed library entries from Music app database
  - Deleted playlist file and removed playlist from Music
- **Features:**
  - Four-step cleanup process: physical files, iTunes directory, library entries, playlist
  - Batch processing of library entries (50 tracks at a time) for efficiency
  - Comprehensive error reporting with detailed statistics
  - AppleScript integration for safe Music library manipulation
  - Handles missing files gracefully (no errors if already deleted)
  - Progress indicators for long-running operations
- **Warning:** This is destructive - only run when you want to delete everything

## Utility Scripts

### utility_scripts/5-calculate-length-of-playlist.py

Statistical utility that calculates expected playlist duration based on geometric distribution.

- **Input:** Configuration from `config.json`
- **Output:** Statistical analysis of expected session duration printed to console
- **Features:**
  - Calculates expected session length using geometric probability
  - Provides standard deviation and confidence intervals
  - Displays track count breakdown by type
  - Shows human-readable duration formats (hours, minutes, seconds)
  - Models random shuffle behavior with replacement


## Project Structure

```
generate-weighted-playlist/
├── input/
│   ├── audio/                    # Source audio files (.wav)
│   └── config/
│       └── config.json           # Configuration settings
├── output/
│   ├── audio/
│   │   └── final-sample-versions/ # Script 1 output
│   └── playlists/                # Script 2 output (M3U files)
├── utility_scripts/
│   └── 5-calculate-length-of-playlist.py
├── 1-combine-samples-with-panning.py
├── 2-import-duplicate-padded-samples-into-itunes-playlist.py
├── 3-clean-up-itunes-playlist-tracks-and-files.py
└── run-all.py                     # Runs complete pipeline
```