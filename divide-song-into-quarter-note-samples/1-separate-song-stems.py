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
import argparse


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


def run_demucs(audio_file: Path, song_name: str, output_base_dir: Path) -> bool:
    """Run Demucs to separate audio stems.
    
    Args:
        audio_file: Path to the input audio file
        song_name: Name of the song (directory name)
        output_base_dir: Base output directory (./output)
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\nRunning Demucs on: {audio_file.name}")
    print(f"This may take several minutes...\n")
    
    # Create output directory in ./output/{song_name}/demucs/
    demucs_output_dir = output_base_dir / song_name / "demucs"
    demucs_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary output directory for demucs
    temp_output = output_base_dir / song_name / "temp_demucs"
    
    # Build command - use python -m demucs instead of the binary
    cmd = [
        "python3",
        "-m",
        "demucs.separate",
        str(audio_file),
        "-n", "83fc094f",
        "-o", str(temp_output)
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
        
        # Move only vocals.wav to the demucs output directory and rename to acappella.wav
        # Demucs outputs to temp_output/model_name/song_name/
        demucs_temp = temp_output / "83fc094f" / audio_file.stem
        vocals_file = demucs_temp / "vocals.wav"
        
        if vocals_file.exists():
            target = demucs_output_dir / "acappella.wav"
            shutil.move(str(vocals_file), str(target))
            print(f"\n✓ Saved acappella.wav to: {target}")
        else:
            print(f"\nWarning: vocals.wav not found at {vocals_file}")
        
        # Clean up temp directory
        if temp_output.exists():
            shutil.rmtree(temp_output)
        
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Separate audio stems using Demucs')
    parser.add_argument('-f', '--force', action='store_true', 
                        help='Force separation even if already marked as source_separated')
    args = parser.parse_args()
    
    # Get directories
    script_dir = Path(__file__).parent
    input_dir = script_dir / "input"
    output_dir = script_dir / "output"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all song directories
    song_dirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name != '.DS_Store' and d.name != 'openai']
    
    if not song_dirs:
        print("No song directories found in ./input")
        return
    
    print(f"\nFound {len(song_dirs)} song(s) to process")
    
    # Get all audio files with their song directories
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.aac', '*.ogg', '*.wma']
    
    # Process each song directory
    for song_dir in song_dirs:
        song_name = song_dir.name
        
        print(f"\n{'='*80}")
        print(f"Processing: {song_name}")
        print(f"{'='*80}")
        
        # Check if acappella.wav already exists (unless force flag is set)
        acappella_file = output_dir / song_name / "demucs" / "acappella.wav"
        if acappella_file.exists() and not args.force:
            print(f"⏭  Skipping - acappella.wav already exists at {acappella_file}")
            continue
        
        if args.force and acappella_file.exists():
            print(f"⚠ Force flag detected - re-processing")
        
        # Find audio file in this song's audio directory
        audio_subdir = song_dir / "audio"
        if not audio_subdir.exists():
            raise FileNotFoundError(f"Audio directory not found at {audio_subdir}")
        
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(audio_subdir.glob(ext))
        
        if not audio_files:
            raise FileNotFoundError(f"No audio files found in {audio_subdir}\nSupported formats: {', '.join(audio_extensions)}")
        
        if len(audio_files) > 1:
            files_list = '\n    '.join([f.name for f in audio_files])
            raise ValueError(f"Found {len(audio_files)} audio files in {audio_subdir}, but only 1 is allowed.\n  Found files:\n    {files_list}")
        
        audio_file = audio_files[0]
        print(f"✓ Found audio file: {audio_file.name}")
        
        # Get config file path
        config_file = song_dir / "config" / "config.json"
        
        # Load config
        config_data = load_song_config(config_file)
        
        # Check if already source separated (unless force flag is set)
        if config_data.get("source_separated", False) and not args.force:
            print(f"⏭  Skipping - already marked as source_separated in config")
            print(f"  To re-process, use the -f flag")
            continue
        
        # Run Demucs
        success = run_demucs(audio_file, song_name, output_dir)
        
        if success:
            # Update config to mark as source separated
            config_data["source_separated"] = True
            save_song_config(config_file, config_data)
            
            print(f"\n✓ Source separation complete!")
            print(f"  Output directory: {output_dir / song_name / 'demucs'}")
            print(f"  Config updated: {config_file}")
        else:
            print(f"\n✗ Source separation failed for {audio_file.name}")


if __name__ == "__main__":
    main()
