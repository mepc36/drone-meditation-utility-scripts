#!/usr/bin/env python3
"""
5-calculate-average-song-length.py

Calculates the expected duration of a listening session where:
- The playlist is played on random shuffle infinitely
- Each track is randomly selected from all available tracks
- The session ends when the Living sample plays
- There is exactly 1 Living sample in the playlist

This is modeled as a geometric distribution where each random selection
has a probability of 1/N of selecting Living (where N is total tracks).

Expected number of selections until Living plays = N
This includes (N-1) non-Living tracks + 1 Living track.
"""

import json
from pathlib import Path


# -------------------------------------------------------------------
# CONFIG: Load from input/config/config.json
# -------------------------------------------------------------------
CONFIG_PATH = Path("../input/config/config.json")

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Extract config sections
pad_config = config["pad_samples_config"]
shared_config = config["shared_config"]

# Parse samples_ratio (e.g., "48:4:1:4")
ratio_parts = [int(x) for x in shared_config["samples_ratio"].split(":")]
BREATHING_COUNT = ratio_parts[0]
OTHER_COUNT = ratio_parts[1]  # Per activity
LIVING_COUNT = ratio_parts[2]  # Should always be 1
SILENCE_COUNT = ratio_parts[3] if len(ratio_parts) > 3 else 0

# Number of "other" canonical activities (excluding Breathing)
NUM_OTHER_ACTIVITIES = len(pad_config["canonical_files"]) - 1

# Sample lengths
# Calculate beat length from BPM: 60 seconds / BPM = seconds per beat
DESIRED_SAMPLE_LENGTH = 60.0 / shared_config["bpm"]
LIVING_SAMPLE_LENGTH = pad_config["living_sample_length_seconds"]


# -------------------------------------------------------------------
# Calculation
# -------------------------------------------------------------------
def calculate_expected_duration() -> dict:
    """
    Calculate expected duration using geometric distribution.
    
    Returns dict with:
        - total_tracks: Total number of tracks in playlist
        - non_living_tracks: Number of non-Living tracks
        - expected_non_living_plays: Expected non-Living tracks before Living
        - expected_duration_seconds: Expected total duration in seconds
        - expected_duration_minutes: Expected total duration in minutes
        - std_dev_seconds: Standard deviation of duration in seconds
        - lower_bound_1sd: Duration 1 standard deviation below mean
        - upper_bound_1sd: Duration 1 standard deviation above mean
    """
    # Count total tracks
    total_breathing = BREATHING_COUNT
    total_others = OTHER_COUNT * NUM_OTHER_ACTIVITIES
    total_living = LIVING_COUNT
    total_silence = SILENCE_COUNT
    
    total_tracks = total_breathing + total_others + total_living + total_silence
    non_living_tracks = total_tracks - total_living
    
    # Geometric distribution:
    # Expected number of picks until Living is selected = total_tracks / LIVING_COUNT
    expected_picks_until_living = total_tracks / LIVING_COUNT
    expected_non_living_plays = expected_picks_until_living - LIVING_COUNT
    
    # All non-Living tracks have the same duration
    # Song ends when Living sample arrives (not after it finishes)
    expected_duration_seconds = expected_non_living_plays * DESIRED_SAMPLE_LENGTH
    expected_duration_minutes = expected_duration_seconds / 60
    
    # Standard deviation calculation:
    # For geometric distribution, variance of number of trials = N(N-1) where N = total_tracks
    # Standard deviation of number of non-Living plays = sqrt(N(N-1))
    # Standard deviation of duration = sqrt(N(N-1)) * sample_length
    import math
    variance_trials = total_tracks * (total_tracks - 1)
    std_dev_trials = math.sqrt(variance_trials)
    std_dev_seconds = std_dev_trials * DESIRED_SAMPLE_LENGTH
    
    lower_bound_1sd = expected_duration_seconds - std_dev_seconds
    upper_bound_1sd = expected_duration_seconds + std_dev_seconds
    
    # Ensure lower bound doesn't go negative
    if lower_bound_1sd < 0:
        lower_bound_1sd = 0
    
    return {
        "total_tracks": total_tracks,
        "non_living_tracks": non_living_tracks,
        "expected_non_living_plays": expected_non_living_plays,
        "expected_duration_seconds": expected_duration_seconds,
        "expected_duration_minutes": expected_duration_minutes,
        "std_dev_seconds": std_dev_seconds,
        "lower_bound_1sd": lower_bound_1sd,
        "upper_bound_1sd": upper_bound_1sd,
        "breathing_count": total_breathing,
        "others_count": total_others,
        "living_count": total_living,
        "silence_count": total_silence,
    }


def format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    print("\n" + "="*60)
    print("Average Session Duration Calculator")
    print("="*60 + "\n")
    
    print("Playlist Configuration:")
    print(f"  Samples ratio: {shared_config['samples_ratio']}")
    print(f"    → {BREATHING_COUNT} Breathing samples")
    print(f"    → {OTHER_COUNT} copies × {NUM_OTHER_ACTIVITIES} other activities = {OTHER_COUNT * NUM_OTHER_ACTIVITIES} samples")
    print(f"    → {LIVING_COUNT} Living sample")
    print(f"    → {SILENCE_COUNT} Silence samples")
    print(f"  Sample duration: {DESIRED_SAMPLE_LENGTH}s (canonical samples)")
    print(f"  Living duration: {LIVING_SAMPLE_LENGTH}s\n")
    
    result = calculate_expected_duration()
    
    print("Calculation Method:")
    print("  Using geometric distribution for random selection with replacement")
    print(f"  Probability of selecting Living = {result['living_count']}/{result['total_tracks']}")
    print(f"  Expected selections until Living = {result['total_tracks']}/{result['living_count']} = {result['total_tracks']:.1f}\n")
    
    print("Expected Session Duration:")
    print(f"  Total tracks in playlist: {result['total_tracks']}")
    print(f"  Expected non-Living tracks played: {result['expected_non_living_plays']:.1f}")
    print(f"  Expected duration: {result['expected_non_living_plays']:.1f} × {DESIRED_SAMPLE_LENGTH}s = {result['expected_non_living_plays'] * DESIRED_SAMPLE_LENGTH:.1f}s")
    print(f"  (Song ends when Living arrives, not after it plays)")
    print(f"\n  Total expected duration: {result['expected_duration_seconds']:.1f} seconds")
    print(f"                         = {result['expected_duration_minutes']:.2f} minutes")
    print(f"                         = {format_duration(result['expected_duration_seconds'])}")
    
    print(f"\n  Standard deviation: {result['std_dev_seconds']:.1f} seconds")
    print(f"                    = {result['std_dev_seconds']/60:.2f} minutes")
    
    print(f"\n  68% confidence interval (±1 SD):")
    print(f"    Lower bound: {result['lower_bound_1sd']:.1f}s = {format_duration(result['lower_bound_1sd'])}")
    print(f"    Upper bound: {result['upper_bound_1sd']:.1f}s = {format_duration(result['upper_bound_1sd'])}\n")
    
    print("Interpretation:")
    print("  On average, when listening to this playlist on random shuffle")
    print("  (with each track selection being independent and random),")
    print(f"  you will listen for approximately {format_duration(result['expected_duration_seconds'])}")
    print("  before the Living sample arrives and the session ends.")
    print("  (The Living sample itself is not included in this duration.)")
    print(f"\n  About 68% of sessions will fall between:")
    print(f"    {format_duration(result['lower_bound_1sd'])} and {format_duration(result['upper_bound_1sd'])}\n")


if __name__ == "__main__":
    main()
