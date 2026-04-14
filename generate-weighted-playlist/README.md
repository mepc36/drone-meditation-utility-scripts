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

### Key Features

- **Variable-length silences**: Configure multiple silence durations with weighted ratios for dynamic variety
- **Padded centered samples**: Add extended silence padding to a percentage of centered samples for deeper meditation moments

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
  - Generates silence samples with variable lengths based on configured ratios
  - Supports padded centered samples with extended silence for deeper meditation moments
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

## Configuration

All settings live in `./input/config/config.json`. Required fields are marked **[required]**; all others are optional.

> **Note:** The total number of output files (`num_unique_samples`) is no longer a config field. It is auto-derived from the number of kick/snare input files in `./input/audio/` and the `kicksnare_stab_acappella_percents` ratio: `round(kicksnare_file_count × 100 / kicksnare_pct)`.

### Annotated example

```json
{
  // [required] One or two BPM values, colon-separated.
  // Single value: all samples use one beat length.
  // Two values: the pipeline draws samples at both tempos. Order is normalized automatically (slow to fast).
  "bpms": "75:118",

  // [required] Percentage split across the three sound groups, in the order kicksnare:stab:acappella.
  // Must have exactly 3 colon-separated values summing to 100.
  "kicksnare_stab_acappella_percents": "60:30:10",

  // [optional] Percentage split between audio samples and silence files, in the order samples:silence.
  // A single value of "100" means no silence files. Two values must sum to 100.
  // Default: "100:0"
  "samples_to_silence_percents": "88:12",

  // [optional] Silence durations in milliseconds, colon-separated.
  // Number of values must match silence_lengths_percents.
  // Default: "2000"
  "silence_lengths_millisec": "1330:5900",

  // [optional] Percentage weight for each silence duration, colon-separated.
  // Must sum to 100. Number of values must match silence_lengths_millisec.
  // Default: "100"
  "silence_lengths_percents": "70:30",

  // [optional] Exactly two dB levels in the order loud:quiet.
  // Ordering is normalized automatically (higher dB first).
  // Default: "0:-26"
  "loud_quiet_values": "0:-28",

  // [optional] Extra volume reduction in dB applied to all strings samples on top of
  // their assigned loud/quiet level. Must be a non-negative integer.
  // Default: 0
  "strings_volume_reduction": 3,

  // [optional] Extra volume reduction in dB applied to all acappella samples on top of
  // their assigned loud/quiet level. Must be a non-negative integer.
  // Default: 0
  "acappella_volume_reduction": 17,

  // [optional] Fine-grained control over how frequently specific samples or subsets are drawn.
  // When omitted or is_sample_bias_enabled is false, all samples in each group are drawn uniformly.
  "sample_bias": {

    // Set to false to disable the entire bias system without deleting the config.
    "is_sample_bias_enabled": true,

    // One key per sound group to bias. Valid keys: "kicksnare", "stab", "acappella".
    // Each value is a list of bucket entries whose biased_pool_pct / unbiased_pool_pct values
    // must sum to exactly 100.
    "kicksnare": [

      // Bucket type A — named sample boost:
      // This percentage of the group's slots always draws from the named file.
      { "biased_sample": "my-kick.wav", "biased_pool_pct": 20 },

      // Bucket type B — random draw from the full group pool:
      // "include_all": true allows any sample in the group.
      { "is_random": true, "biased_pool_pct": 15, "include_all": true },

      // Bucket type B variant — restrict the sub-pool to an explicit list:
      { "is_random": true, "biased_pool_pct": 15, "include": ["kick-a.wav", "kick-b.wav"] },

      // Bucket type B variant — exclude specific samples from the sub-pool:
      { "is_random": true, "biased_pool_pct": 15, "exclude": ["kick-c.wav"] },

      // Bucket type C — unbiased remainder:
      // Draws uniformly from the full group pool. At most one per group.
      { "unbiased_pool_pct": 35 }
    ]
  }
}
```

### Field reference

| Field | Required | Type | Default | Description |
|---|---|---|---|---|
| `bpms` | Yes | string | — | Colon-separated BPM values. One or two values. |
| `kicksnare_stab_acappella_percents` | Yes | string | — | Three colon-separated percents for kicksnare:stab:acappella. Must sum to 100. |
| `samples_to_silence_percents` | No | string | `"100:0"` | Colon-separated percents for audio:silence split. Must sum to 100. |
| `silence_lengths_millisec` | No | string | `"2000"` | Colon-separated millisecond values for silence durations. |
| `silence_lengths_percents` | No | string | `"100"` | Colon-separated percent weights for each silence length. Must sum to 100. |
| `loud_quiet_values` | No | string | `"0:-26"` | Exactly two colon-separated dB values (loud:quiet). |
| `strings_volume_reduction` | No | integer | `0` | Extra dB cut applied to strings samples (non-negative). |
| `acappella_volume_reduction` | No | integer | `0` | Extra dB cut applied to acappella samples (non-negative). |
| `sample_bias` | No | object | none | Sample draw-weighting config. Omit to draw all samples uniformly. |

## Assumptions

The following assumptions are made about the config and the environment. Violations will raise an error or produce undefined behavior.

- Audio sample files are placed in `./input/audio/` before running any script. The pipeline does not check for or warn about missing files.
- `bpms` contains one or two values. More than two values is not supported.
- All colon-separated percent strings (e.g. `samples_to_silence_percents`, `silence_lengths_percents`, `kicksnare_stab_acappella_percents`) sum to exactly 100. No rounding is applied.
- `silence_lengths_millisec` and `silence_lengths_percents` always contain the same number of colon-separated values.
- `loud_quiet_values` always has exactly 2 values. Fewer or more will raise an error.
- `kicksnare_stab_acappella_percents` always has exactly 3 values in declaration order: kicksnare, stab, acappella.
- `strings_volume_reduction` and `acappella_volume_reduction` are non-negative integers. Fractional or negative values will raise an error.
- All keys inside `sample_bias` (other than `is_sample_bias_enabled`) must exactly match a valid sound group name: `kicksnare`, `stab`, or `acappella`. Unknown keys raise an error.
- A sound group configured in `sample_bias` must have a non-zero percent in `kicksnare_stab_acappella_percents`. A group with 0% raises an error.
- All bucket `biased_pool_pct` and `unbiased_pool_pct` values within a single bias group sum to exactly 100.
- Each bias group list contains at most one `unbiased_pool_pct` entry.
- `include` and `exclude` are mutually exclusive within a single `is_random` bucket. Specifying both raises an error.
- `include_all` and `exclude` are mutually exclusive within a single `is_random` bucket. Specifying both raises an error.
- An `is_random` bucket with `include` (and `include_all` not set) must provide a non-empty list. An empty list raises an error.
- `strings_volume_reduction` and `acappella_volume_reduction` are applied additively on top of the sample's assigned loud/quiet dB level. They are not an absolute target level.
- Each `biased_pool_pct` value, when applied to the group's calculated slot count, yields at least one slot. Very small percentages combined with few kicksnare input files may raise an error.

## Todos

1. Add setup script to fetch .wav samples from S3 if not exist
