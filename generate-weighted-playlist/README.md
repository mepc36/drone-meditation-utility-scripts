# Generate Weighted Playlist

## Summary

This project creates weighted meditation playlists from audio samples using a multi-stage pipeline. It pads audio files to consistent lengths, duplicates them according to specified ratios, imports them into Apple Music, generates M3U playlists, and provides cleanup utilities. The playlist ends when a rare "Living" sample plays, creating variable-length meditation sessions based on probability.

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

- `samples_ratio`: Duplication ratio as "Breathing:Others:Living:Silence" (e.g., "8:2:1:10")
- `desired_sample_length_seconds`: Target length for most samples (default: 7)
- `living_sample_length_seconds`: Length of the Living sample (default: 300)
- `canonical_files`: Array of activity names (Breathing, Being, Feeling, etc.)
- `itunes_dir`: Path where iTunes/Music imports files
- `playlist_name`: Name of the generated playlist

## Usage

Activate the virtual environment before running scripts:

```bash
source .venv/bin/activate
```

Run scripts in sequence:

```bash
# 1. Pad audio files with silence
python3 1-pad-samples-with-silence.py

# 2. Create weighted duplicates
python3 2-duplicate-padded-samples.py

# 3. Generate playlist and import to Music
python3 3-import-duplicate-padded-samples-into-itunes-playlist.py

# 4. Clean up files and library entries (optional)
python3 4-clean-up-itunes-playlist-tracks-and-files.py

# 5. Calculate expected session length (utility)
python3 5-calculate-length-of-playlist.py
```

## Scripts

### 1-pad-samples-with-silence.py

Pads audio samples with silence to make them all the same length.

- **Input:** Audio files from `./input/audio/`
- **Output:** Padded audio files in `./output/audio/padded-audio-samples/`

### 2-duplicate-padded-samples.py

Creates weighted duplicates of each padded audio file based on the samples ratio.

- **Input:** Padded audio from `./output/audio/padded-audio-samples/`
- **Output:** Numbered duplicates in `./output/audio/final-sample-versions/`

### 3-import-duplicate-padded-samples-into-itunes-playlist.py

Generates an M3U playlist file and imports the entire folder into Apple Music.

- **Input:** Final sample versions from `./output/audio/final-sample-versions/`
- **Output:** M3U playlist in `./output/playlists/` and tracks imported to Music app

### 4-clean-up-itunes-playlist-tracks-and-files.py

Removes all generated files and library entries to start fresh.

- **Input:** Generated audio files and Music library entries
- **Output:** Deleted physical files, removed library entries, and deleted playlist

### 5-calculate-length-of-playlist.py

Calculates the expected duration of a listening session using geometric distribution.

- **Input:** Configuration from `config.json`
- **Output:** Statistical analysis of expected session duration printed to console

## Project Structure

```
generate-weighted-playlist/
├── input/
│   ├── audio/                    # Source audio files
│   └── config/
│       └── config.json           # Configuration settings
├── output/
│   ├── audio/
│   │   ├── padded-audio-samples/ # Step 1 output
│   │   └── final-sample-versions/ # Step 2 output
│   └── playlists/                # Step 3 output
└── *.py                          # Pipeline scripts
```

## How It Works

1. **Padding**: All audio samples are padded with silence to match `desired_sample_length_seconds` (except Living, which uses `living_sample_length_seconds`)

2. **Duplication**: Files are duplicated according to the `samples_ratio`. For example, "8:2:1:10" means:
   - Breathing: 8 copies
   - Each other activity (Being, Feeling, etc.): 2 copies each
   - Living: 1 copy
   - Silence: 10 copies

3. **Import**: All duplicated files are imported into Apple Music, and an M3U playlist is created with references to these files

4. **Playback**: When shuffled, the playlist plays random samples until the rare Living sample appears, ending the session

5. **Cleanup**: Script 4 removes all generated files and library entries when you want to regenerate or start over

6. **Statistics**: Script 5 calculates expected session duration based on geometric probability (probability of selecting Living from total tracks)

## Notes

- The Living sample represents completion and should be rare in the playlist
- Silence samples provide meditation pauses between activities
- Use shuffle mode in Apple Music for varied playback
- Session duration varies based on when Living is randomly selected
- Script 4 is destructive - only run when you want to delete everything and start fresh
