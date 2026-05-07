"""
Detect BPM using Essentia via Docker.

This script:
1. Takes audio files from ./input/{SONG_NAME}/audio
2. Uses Essentia's music extractor via Docker to analyze audio
3. Extracts BPM from the JSON output
4. Saves BPM to ./input/{SONG_NAME}/config/config.json

Requirements:
- Docker must be installed
- Pull the essentia image: docker pull mtgupf/essentia
"""

from pathlib import Path
import json
import argparse
import subprocess
import shutil


def load_song_config(config_file: Path) -> dict:
    """Load song configuration from JSON file.
    
    Args:
        config_file: Path to the song config JSON file
        
    Returns:
        Dictionary with song configuration (bpm, downbeat_offset, etc.)
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


def detect_bpm_with_essentia(audio_file: Path, output_dir: Path) -> float:
    """Detect BPM using Essentia via Docker.
    
    Args:
        audio_file: Path to the audio file
        output_dir: Directory to write essentia output JSON
        
    Returns:
        Detected BPM as float
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "essentia_analysis.json"
    
    print(f"  Analyzing audio with Essentia (Docker)...")
    print(f"  This may take a moment...")
    
    # Get the directory containing the audio file
    audio_dir = audio_file.parent
    audio_filename = audio_file.name
    output_filename = output_json.name
    
    # Construct docker command
    # Mount the audio directory to /essentia in the container
    docker_cmd = [
        "docker", "run",
        "-ti", "--rm",
        "-v", f"{audio_dir.absolute()}:/essentia",
        "mtgupf/essentia",
        "essentia_streaming_extractor_music",
        f"/essentia/{audio_filename}",
        f"/essentia/{output_filename}"
    ]
    
    try:
        # Run docker command
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Move the output file to the correct location
        temp_output = audio_dir / output_filename
        if temp_output.exists():
            shutil.move(str(temp_output), str(output_json))
        
        # Parse the JSON output
        with open(output_json, 'r') as f:
            data = json.load(f)
        
        # Extract BPM from essentia output
        bpm = data.get('rhythm', {}).get('bpm', None)
        
        if bpm is None:
            raise ValueError("BPM not found in Essentia output")
        
        return float(bpm)
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Docker command failed. Make sure Docker is running and essentia image is pulled.\n"
            f"Pull with: docker pull mtgupf/essentia\n"
            f"Error: {e.stderr}"
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Docker not found. Make sure Docker is installed and in your PATH."
        )


def main():
    """Main function to detect BPM."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect BPM using Essentia')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force redetection even if BPM already exists in config')
    args = parser.parse_args()
    
    # Get directories
    script_dir = Path(__file__).parent
    input_dir = script_dir / "input"
    output_dir = script_dir / "output"
    
    if not input_dir.exists():
        print("Input directory not found")
        return
    
    # Find all song directories
    song_dirs = [d for d in input_dir.iterdir() 
                 if d.is_dir() and d.name != '.DS_Store' and d.name != 'prompts']
    
    if not song_dirs:
        print("No song directories found in ./input")
        return
    
    print(f"\nFound {len(song_dirs)} song(s) to process")
    
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.aac', '*.ogg', '*.wma']
    
    # Process each song directory
    for song_dir in song_dirs:
        song_name = song_dir.name
        
        print(f"\n{'='*80}")
        print(f"Processing: {song_name}")
        print(f"{'='*80}")
        
        # Check if BPM already exists in config
        config_file = song_dir / "config" / "config.json"
        config_data = load_song_config(config_file)
        
        if not args.force and config_data.get('bpm'):
            print(f"⏭  Skipping - BPM already exists in config: {config_data['bpm']:.2f}")
            print(f"   Use -f or --force flag to redetect")
            continue
        
        # Find audio file in this song's audio directory
        audio_subdir = song_dir / "audio"
        if not audio_subdir.exists():
            print(f"  ⚠️  Audio directory not found at {audio_subdir}")
            continue
        
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(audio_subdir.glob(ext))
        
        if not audio_files:
            print(f"  ⚠️  No audio files found in {audio_subdir}")
            print(f"     Supported formats: {', '.join(audio_extensions)}")
            continue
        
        if len(audio_files) > 1:
            files_list = '\n    '.join([f.name for f in audio_files])
            print(f"  ⚠️  Found {len(audio_files)} audio files, but only 1 is allowed.")
            print(f"     Found files:\n    {files_list}")
            continue
        
        audio_file = audio_files[0]
        print(f"✓ Found audio file: {audio_file.name}")
        
        # Create essentia output directory
        essentia_output_dir = output_dir / song_name / "essentia"
        
        # Detect BPM using essentia
        try:
            bpm = detect_bpm_with_essentia(audio_file, essentia_output_dir)
            print(f"\n✓ Detected BPM: {bpm:.2f}")
            print(f"✓ Essentia analysis saved to: {essentia_output_dir}")
            
            # Update config (preserve existing values)
            if not config_data:
                config_data = {}
            
            config_data['bpm'] = bpm
            if 'downbeat_offset' not in config_data:
                config_data['downbeat_offset'] = 0.0
            
            # Save config
            save_song_config(config_file, config_data)
            print(f"✓ Saved to config: {config_file}")
            
        except Exception as e:
            print(f"  ❌ Error detecting BPM: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("✓ Complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
