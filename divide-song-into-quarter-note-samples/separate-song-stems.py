"""
Separate song stems using Demucs.

This script:
1. Takes a single audio file from ./input (any format: mp3, wav, flac, etc.)
2. Checks if it's already been source separated (from config)
3. If not, runs Demucs to separate the stems
4. Outputs separated stems to ./output
5. Updates the config to mark as source separated
"""

import os
from pathlib import Path
import subprocess
import json
import re
import shutil


def filename_to_kebab_case(filename: str) -> str:
    """Convert filename to kebab-case (remove extension, lowercase, replace spaces/special chars with hyphens).
    
    Args:
        filename: Original filename
        
    Returns:
        Kebab-case version without extension
    """
    # Remove extension
    name = Path(filename).stem
    # Convert to lowercase
    name = name.lower()
    # Replace spaces and underscores with hyphens
    name = re.sub(r'[\s_]+', '-', name)
    # Remove non-alphanumeric characters except hyphens
    name = re.sub(r'[^a-z0-9-]+', '', name)
    # Remove multiple consecutive hyphens
    name = re.sub(r'-+', '-', name)
    # Remove leading/trailing hyphens
    name = name.strip('-')
    return name


def load_song_config(config_file: Path) -> dict:
    """Load song configuration from JSON file.
    
    Args:
        config_file: Path to the song config JSON file
        
    Returns:
        Dictionary with song configuration
    """
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_song_config(config_file: Path, config_data: dict) -> None:
    """Save song configuration to JSON file.
    
    Args:
        config_file: Path to the song config JSON file
        config_data: Dictionary with song configuration
    """
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)


def run_demucs(audio_file: Path, output_dir: Path) -> bool:
    """Run Demucs to separate audio stems.
    
    Args:
        audio_file: Path to the input audio file
        output_dir: Path to the output directory
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\nRunning Demucs on: {audio_file.name}")
    print(f"This may take several minutes...\n")
    
    # Build command - use python -m demucs instead of the binary
    cmd = [
        "python3",
        "-m",
        "demucs.separate",
        str(audio_file),
        "-n", "83fc094f",
        "-o", str(output_dir)
    ]
    
    try:
        # Run demucs as subprocess
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        print("Demucs output:")
        print(result.stdout)
        
        if result.stderr:
            print("Demucs stderr:")
            print(result.stderr)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running Demucs: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Error: Demucs module not found")
        print("Please ensure Demucs is installed: pip install demucs")
        return False


def main():
    """Main function to process audio files."""
    # Get directories
    script_dir = Path(__file__).parent
    input_dir = script_dir / "input"
    output_dir = script_dir / "output"
    
    # Clear output directory before running
    if output_dir.exists():
        print(f"Clearing output directory...")
        shutil.rmtree(output_dir)
    
    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all audio files from input directory (common formats) - search recursively
    audio_extensions = ['**/*.mp3', '**/*.wav', '**/*.flac', '**/*.m4a', '**/*.aac', '**/*.ogg', '**/*.wma']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(ext))
    
    if not audio_files:
        print("No audio files found in ./input directory")
        print(f"Supported formats: {', '.join([e.replace('*', '') for e in audio_extensions])}")
        return
    
    # Use only the first file
    audio_file = audio_files[0]
    
    if len(audio_files) > 1:
        print(f"Warning: Found {len(audio_files)} audio files. Processing only: {audio_file.name}")
        print("Remove other files to process a different one.\n")
    
    # Get config file path (same directory as audio file)
    config_file = audio_file.with_suffix('.json')
    
    # Load config
    config_data = load_song_config(config_file)
    
    # Check if already source separated
    if config_data.get("source_separated", False):
        print(f"✓ Song already source separated: {audio_file.name}")
        print(f"  Config file: {config_file}")
        print(f"  Skipping Demucs processing.")
        print(f"\nTo re-process, set 'source_separated' to false in the config file.")
        return
    
    # Run Demucs
    success = run_demucs(audio_file, output_dir)
    
    if success:
        # Update config to mark as source separated
        config_data["source_separated"] = True
        save_song_config(config_file, config_data)
        
        print(f"\n✓ Source separation complete!")
        print(f"  Output directory: {output_dir}")
        print(f"  Config updated: {config_file}")
        print(f"  Marked as source_separated: true")
    else:
        print(f"\n✗ Source separation failed")


if __name__ == "__main__":
    main()
