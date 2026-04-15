#!/usr/bin/env python3

"""
run-all.py

Runs the complete pipeline:
1. 3-clean-up-itunes-playlist-tracks-and-files.py - Clean up old files
2. 1-combine-samples-with-panning.py - Generate combined samples
3. 2-import-duplicate-padded-samples-into-itunes-playlist.py - Build playlist and play via mpv
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_name: str, description: str) -> bool:
    """Run a Python script and return success status."""
    print("\n" + "="*70)
    print(f"Running: {description}")
    print("="*70 + "\n")
    
    script_path = Path(__file__).parent / script_name
    
    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            text=True
        )
        print(f"")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Script not found: {script_path}")
        return False


def main() -> None:
    print("\n" + "="*70)
    print("GENERATE WEIGHTED PLAYLIST - FULL PIPELINE")
    print("="*70)
    
    scripts = [
        ("steps/3-clean-up-itunes-playlist-tracks-and-files.py", "Step 1: Clean up old files"),
        ("steps/1-combine-samples-with-panning.py", "Step 2: Combine samples with panning"),
        ("steps/2-import-duplicate-padded-samples-into-itunes-playlist.py", "Step 3: Build playlist and play via mpv")
    ]
    
    for script_name, description in scripts:
        success = run_script(script_name, description)
        if not success:
            print("\n" + "="*70)
            print("PIPELINE FAILED - Stopping execution")
            print("="*70)
            sys.exit(1)
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nAll steps completed. Your weighted playlist is ready!\n")


if __name__ == "__main__":
    main()
