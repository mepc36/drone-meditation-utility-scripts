"""
Filter quarter note samples to find meditation-appropriate lyrics using ChatGPT.

This script:
1. Reads labeled quarter note sample filenames
2. Extracts the lyrics portion from filenames
3. Asks ChatGPT to identify self-contained meditation-appropriate phrases
4. Returns a list of filenames containing suitable lyrics
"""

import os
from pathlib import Path
import json
from dotenv import load_dotenv
from openai import OpenAI
import shutil

# Load environment variables
load_dotenv()


def extract_lyrics_from_filenames(directory: Path) -> dict:
    """Extract lyrics from quarter note sample filenames.
    
    Filename format: {index:04d}_{timestamp}_{prefix}_{lyrics}.wav
    Example: 0001_0.000000_acappella_living.wav
    
    Args:
        directory: Directory containing labeled samples
        
    Returns:
        Dictionary mapping lyrics to list of filenames
    """
    if not directory.exists():
        return {}
    
    lyrics_to_files = {}
    
    for wav_file in directory.glob("*.wav"):
        stem = wav_file.stem
        parts = stem.split('_')
        
        # Format: index_timestamp_prefix_lyrics or index_timestamp_prefix_lyrics_partial
        if len(parts) < 4:
            continue
        
        # Last part is lyrics (or second-to-last if ends with 'partial')
        if parts[-1] == 'partial':
            lyrics = parts[-2]
        else:
            lyrics = parts[-1]
        
        # Skip no-lyrics entries
        if lyrics == 'no-lyrics':
            continue
        
        if lyrics not in lyrics_to_files:
            lyrics_to_files[lyrics] = []
        
        lyrics_to_files[lyrics].append(wav_file.name)
    
    return lyrics_to_files


def filter_lyrics_with_chatgpt(lyrics_list: list, api_key: str, prompt_file: Path) -> list:
    """Use ChatGPT to filter lyrics for meditation appropriateness.
    
    Args:
        lyrics_list: List of lyrics strings (kebab-case)
        api_key: OpenAI API key
        prompt_file: Path to the prompt template file
        
    Returns:
        List of meditation-appropriate lyrics
    """
    client = OpenAI(api_key=api_key)
    
    # Read prompt template from file
    with open(prompt_file, 'r') as f:
        prompt_template = f.read()
    
    # Replace placeholder with actual lyrics list
    prompt = prompt_template.replace('{LYRICS_LIST}', json.dumps(lyrics_list, indent=2))
    
    print("Sending request to ChatGPT...")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    result_text = response.choices[0].message.content.strip()
    
    # Parse JSON response
    # Remove markdown code blocks if present
    if result_text.startswith('```'):
        lines = result_text.split('\n')
        result_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else result_text
    
    try:
        filtered_lyrics = json.loads(result_text)
        return filtered_lyrics
    except json.JSONDecodeError as e:
        print(f"Error parsing ChatGPT response: {e}")
        print(f"Response: {result_text}")
        return []


def main():
    """Main function to filter lyrics via ChatGPT."""
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    # Get directories
    script_dir = Path(__file__).parent
    input_dir = script_dir / "input"
    output_dir = script_dir / "output"
    
    # Find song directory (should only be one)
    song_dirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name != '.DS_Store']
    
    if not song_dirs:
        print("No song directories found in ./input")
        return
    
    if len(song_dirs) > 1:
        print(f"Error: Found {len(song_dirs)} song directories, but only 1 is allowed.")
        raise ValueError(f"Expected 1 song directory, found {len(song_dirs)}")
    
    song_dir = song_dirs[0]
    song_name = song_dir.name
    
    print(f"\n{'='*80}")
    print(f"Processing: {song_name}")
    print(f"{'='*80}")
    
    # Process only full song samples
    full_song_dir = output_dir / song_name / "quarter-note-samples-labeled-with-lyrics"
    
    # Extract lyrics from full song directory
    print("\nExtracting lyrics from full song filenames...")
    full_song_lyrics = extract_lyrics_from_filenames(full_song_dir)
    
    all_lyrics = sorted(list(full_song_lyrics.keys()))
    
    print(f"Found {len(all_lyrics)} unique lyric phrases")
    
    if not all_lyrics:
        print("No lyrics found to filter")
        return
    
    # Get prompt file
    prompt_file = script_dir / "input" / "openai" / "meditation-lyrics-filter-prompt.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    # Filter with ChatGPT
    filtered_lyrics = filter_lyrics_with_chatgpt(all_lyrics, api_key, prompt_file)
    
    print(f"\n{'='*80}")
    print(f"ChatGPT Results:")
    print(f"{'='*80}")
    print(f"Input: {len(all_lyrics)} unique phrases")
    print(f"Output: {len(filtered_lyrics)} meditation-appropriate phrases")
    print(f"\nFiltered lyrics:")
    for lyric in sorted(filtered_lyrics):
        print(f"  - {lyric}")
    
    # Create output summary
    output_summary = {
        "song_name": song_name,
        "total_unique_lyrics": len(all_lyrics),
        "meditation_lyrics_count": len(filtered_lyrics),
        "meditation_lyrics": sorted(filtered_lyrics),
        "meditation_lyrics_files": []
    }
    
    # Collect all matching filenames into a single list
    all_files = []
    for lyric in filtered_lyrics:
        if lyric in full_song_lyrics:
            all_files.extend(full_song_lyrics[lyric])
    
    # Sort by index (first part of filename)
    all_files.sort(key=lambda f: int(f.split('_')[0]))
    output_summary["meditation_lyrics_files"] = all_files
    
    # Save to JSON
    output_file = output_dir / song_name / "openai" / "meditation-lyrics.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_summary, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Copy meditation lyric files to their own folder
    meditation_files_dir = output_dir / song_name / "meditation-lyrics-files"
    meditation_files_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCopying meditation lyric files...")
    for filename in output_summary["meditation_lyrics_files"]:
        source = full_song_dir / filename
        dest = meditation_files_dir / filename
        if source.exists():
            shutil.copy2(source, dest)
    
    print(f"✓ Copied {len(output_summary['meditation_lyrics_files'])} files to: {meditation_files_dir}")
    
    print(f"\n{'='*80}")
    print("Summary:")
    print(f"{'='*80}")
    print(f"Total meditation lyric files: {len(output_summary['meditation_lyrics_files'])}")
    print(f"Output directory: {meditation_files_dir}")


if __name__ == "__main__":
    main()
