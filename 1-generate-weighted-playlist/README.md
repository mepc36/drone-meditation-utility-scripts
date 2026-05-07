# Generate Weighted Playlist

## Summary

This project creates weighted meditation playlists from audio samples using a multi-stage pipeline. It combines audio samples with stereo panning effects, generates M3U playlists, and plays them via mpv. The system creates unique stereo combinations with randomized panning patterns for an immersive listening experience.

## Installation

### Requirements

- Python 3.x
- macOS
- mpv (for playlist playback)

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

### Input File Naming Convention

Input `.wav` files must follow this naming scheme for the pipeline to identify their sound type:

```
<any-prefix>_<any-prefix>_<type>.wav
```

The **third underscore-separated segment** (before `.wav`) determines the sound type. Valid type suffixes:

| Suffix | Sound group | Description |
|---|---|---|
| `_kick` | kicksnare | Kick drum |
| `_snare` | kicksnare | Snare drum |
| `_kickstab` | stab | Kick-type stab |
| `_snarestab` | stab | Snare-type stab |
| `_acappella` | acappella | Vocal/acappella |
| `_strings` | strings | Strings (pass-through, no processing) |

Example: `artist_song_kick.wav`, `artist_song_snare.wav`, `artist_song_strings.wav`.

Files that do not match a valid type suffix will raise an error at startup.

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
python3 1-run-all.py
```

This executes the full workflow:
1. Cleans up previous files (script 3)
2. Combines samples with panning (script 1)
3. Imports to Music and creates playlist (script 2)

### Run Scripts Individually

```bash
# 1. Combine audio samples with stereo panning
python3 steps/1-combine-samples-with-panning.py

# 2. Build M3U playlist and play via mpv
python3 steps/2-import-duplicate-padded-samples-into-itunes-playlist.py

# 3. Clean up output files (optional)
python3 steps/3-clean-up-itunes-playlist-tracks-and-files.py
```

## Scripts

### 1-combine-samples-with-panning.py

Creates unique stereo combinations of audio samples with randomized panning patterns. This is the main generation script that produces the final audio samples.

- **Input:** Audio files from `./input/audio/`
- **Output:**
  - Processed stereo audio files in `./output/audio/`
  - Rhythm-chopped versions in `./output/rhythmicized-audio/` (same content with beat-level cuts applied; used for auditioning rhythms)
- **Features:**
  - Assigns each sample a sound group, panning position, volume, BPM, and rhythm pattern
  - Panning modes: center, hard left/right, diagonal, and dualpan (two samples simultaneously)
  - Generates silence files with variable lengths based on configured ratios
  - All non-strings samples RMS-normalized then gain-adjusted to the configured dB level
  - Groups samples by sound type for coherent dualpan pairings
  - Random sample selection with shuffled-queue refill (approx. one occurrence per cycle)

### 2-import-duplicate-padded-samples-into-itunes-playlist.py

Builds an M3U playlist from the rhythmicized audio output and plays it via mpv on shuffle.

- **Input:** Rhythm-chopped audio files from `./output/rhythmicized-audio/`
- **Output:** M3U playlist in `./output/playlists/`
- **Features:**
  - Recreates playlist folder fresh on each run (prevents stale entries)
  - Writes M3U with absolute paths to all output `.wav` files
  - Prints total listen-once duration before playback
  - Streams playback via `mpv --shuffle --loop-playlist=inf` with gapless audio
  - Displays track names as each file begins playing

### 3-clean-up-itunes-playlist-tracks-and-files.py

Cleans up all generated files. Run this when you want to start fresh.

- **Input:** Generated `.wav` files in `./output/audio/` and `./output/rhythmicized-audio/`
- **Output:** Deleted audio files and playlist file
- **Features:**
  - Kills mpv before deleting files (avoids stale file errors)
  - Deletes all `.wav` files from both output directories
  - Deletes the M3U playlist file
  - Reports counts and any errors
- **Warning:** This is destructive — only run when you want to delete everything

## Project Structure

```
generate-weighted-playlist/
├── 1-run-all.py                  # Runs complete pipeline
├── input/
│   ├── audio/                    # Source audio files (.wav) — place input samples here
│   └── config/
│       └── config.json           # Active configuration
├── configs/                      # Example / saved config presets
│   ├── 1-config-117-bpm-with-strings.json
│   ├── 2-config-with-silent-beats.json
│   ├── 3-config-with-2-bpms.json
│   ├── 4-config-for-rhythmic-patterns.json
│   ├── 5-slow-centered-kick-snare-samples.json
│   ├── 6-fast-bpm-config.json
│   ├── 7-centered-fast-samples.json
│   └── z-old-configs/
├── lib/                          # Core library modules
│   ├── audio_processing.py       # DSP: panning, normalization, rhythm application
│   ├── config.py                 # Config loading and validation
│   ├── constants.py              # All shared constants
│   ├── deck_builder.py           # Slot planning (panning/volume/BPM/rhythm allocation)
│   ├── runtime_constants.py      # Config-derived LOUD/QUIET/SLOW/FAST values
│   ├── sample_queue.py           # Shuffled sample draw queue and bias resolution
│   └── sound_rules.py            # Hardcoded per-group rhythm/panning/volume rules
├── steps/
│   ├── 1-combine-samples-with-panning.py   # Main generation step
│   ├── 2-import-duplicate-padded-samples-into-itunes-playlist.py
│   └── 3-clean-up-itunes-playlist-tracks-and-files.py
├── output/
│   ├── audio/                    # Step 1 output (processed stereo files)
│   ├── rhythmicized-audio/       # Step 1 output (rhythm-chopped versions)
│   └── playlists/                # Step 2 output (M3U files)
└── audio-samples/                # Reference sample library (not used by pipeline directly)
```

### Example configs

The `configs/` directory contains ready-to-use presets. To use one, copy it to `input/config/config.json`:

```bash
cp configs/3-config-with-2-bpms.json input/config/config.json
```

| File | Description |
|---|---|
| `1-config-117-bpm-with-strings.json` | Single fast BPM with strings |
| `2-config-with-silent-beats.json` | Includes silence file generation |
| `3-config-with-2-bpms.json` | Two-tempo (slow + fast) mix |
| `4-config-for-rhythmic-patterns.json` | Emphasises multi-beat rhythm patterns |
| `5-slow-centered-kick-snare-samples.json` | Slow, center-panned kick/snare focus |
| `6-fast-bpm-config.json` | Fast BPM only |
| `7-centered-fast-samples.json` | Center-panned, fast tempo |

## Configuration

All settings live in `./input/config/config.json`. Required fields are marked **[required]**; all others are optional.

> **Note:** The total number of output files is auto-derived, not a config field. It is `round(kicksnare_count × 100 / kicksnare_pct)`.

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

## Generation Rules

These are the rules the pipeline follows when building the output deck. Most are hardcoded in `lib/`; the config controls weights and counts but not the structure.

### 1. Deck Planning

The deck planner (`lib/deck_builder.py`) decides how many output slots each sound group gets, which panning positions they use, and what volume/BPM each slot targets.

**Beat-anchoring**
Kicksnare files each appear exactly once. All other groups are scaled so that the total *beats heard* match the configured ratio, accounting for multi-beat rhythm patterns. For example, a pattern with three quarter notes counts as 3 beats, so fewer output files are needed to hit a given percentage.

**Panning allocation**
Groups are allocated panning slots in order of most-constrained first (fewest compatible panning positions). Within each group, slots are distributed across eligible pannings according to the `MUSIC_PATTERN_PERCENT` weights defined in `lib/sound_rules.py`. Overflow is redistributed to uncapped groups.

**Volume and BPM assignment**
Slots with only one valid volume/BPM combination are forced first. The remaining slots are shuffled and greedily assigned to meet outstanding slow/fast BPM and loud/quiet volume targets.

**Rhythm pattern distribution**
Rhythm patterns within each group are distributed proportionally by `RHYTHM_PERCENT` weight, then shuffled before writing.

### 2. Sound Group Rules (hardcoded)

These rules live in `lib/sound_rules.py` and are not configurable via `config.json`.

| Group | Volume | BPM | Panning | Rhythm patterns (% weight) |
|---|---|---|---|---|
| kicksnare | Quiet | Slow | Center | quarter (50%), quarter-quarter (26%), quarter-8th-8th (6%), 16th-dotted-8th (6%), 16th-dotted-8th-quarter (6%), A-B-A triple (6%) |
| stab | Loud | Fast | Hard-left (17%), hard-right (17%), random pan (16%), dualpan L+R (50%) | quarter each |
| acappella | Loud | Slow | Hard-left (50%), hard-right (50%) | quarter each |
| strings | — | — | Center (pass-through) | — (untouched, no processing) |

### 3. Sample Drawing

Samples are drawn from a shuffled deque (`lib/sample_queue.py`). When the deque is exhausted it is refilled with a fresh shuffle, so every sample appears approximately once per cycle.

- **Kicksnare**: up to 200 draw retries to avoid repeating the same sample+panning combination in the output.
- **Dualpan slots**: a partner sample of the correct type is drawn (excluding the primary). If no partner is available, the slot falls back to a diagonal-left mono pan.
- **SampleRole.SAME**: a beat reuses the primary sample drawn for that slot. **SampleRole.NEW**: a fresh sample is drawn (excluding all samples already used in the slot).

### 4. Random Panning — the N−2 Rule

When a rhythm pattern uses `RandomPan` (a symmetric magnitude range, e.g. [0.6, 1.0] meaning anywhere from 60%–100% left or right), the beats within a multi-beat pattern are grouped as follows:

| Pattern length N | Anchor group | Tail beats |
|---|---|---|
| 1 | 1 (single, fully free) | — |
| 2 | 0 (both independent) | both free |
| 3 | 2 (beats 1–2 share one position) | beat 3 independent |
| ≥ 4 | N−2 (beats 1…N−2 share one position) | beats N−1 and N independent |

All tail beats must land at least `RANDOM_PAN_MIN_DIFF` (0.25) away from the immediately preceding beat's panning position. The left/right side is chosen uniformly at random for the anchor group.

### 5. Audio Processing

- **RMS normalization**: non-strings, non-acappella samples are normalized to a target RMS of 0.15 before the volume dB adjustment is applied.
- **Clip ceiling**: after all gain is applied, audio is capped at 0.95 to prevent clipping.
- **Hard-pan gain**: hard-panned signals receive a ×√2 (+3 dB) boost to compensate for the power lost by removing the other channel, matching the perceived loudness of a center-panned signal.
- **Panning law**: constant-power panning (cosine/sine of the pan angle).
- **Strings**: converted to stereo-center, no RMS normalization, no trim-to-beat-length. Passed through as-is after the optional `strings_volume_adjustment_db` cut.
- **Acappella**: no RMS normalization (source level is preserved). The `acappella_volume_adjustment_db` cut is applied additively on top of the loud/quiet level.

## Todos

1. Add setup script to fetch .wav samples from S3 if not exist
