import re
import requests
from bs4 import BeautifulSoup
import argparse
import sys
from pathlib import Path

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def create_genius_url(artist: str, song: str) -> str:
    """Create a Genius URL from artist and song name.
    
    Args:
        artist: Artist name
        song: Song name
        
    Returns:
        Genius URL string
    """
    # Convert to lowercase and replace spaces with hyphens
    artist_slug = re.sub(r'[^a-z0-9]+', '-', artist.lower()).strip('-')
    song_slug = re.sub(r'[^a-z0-9]+', '-', song.lower()).strip('-')
    return f"https://genius.com/{artist_slug}-{song_slug}-lyrics"

def scrape_genius_lyrics(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Convert <br> to newline so get_text preserves line structure
    for br in soup.find_all("br"):
        br.replace_with("\n")

    # Genius lyrics containers
    lyric_divs = soup.select('div[class^="Lyrics__Container"]')

    if not lyric_divs:
        raise RuntimeError("No lyrics containers found — Genius layout may have changed")

    lyrics_blocks = []
    for div in lyric_divs:
        text = div.get_text(separator="\n").strip()
        if text:
            lyrics_blocks.append(text)

    lyrics = "\n\n".join(lyrics_blocks)

    # Normalize spacing
    lyrics = re.sub(r"\n{3,}", "\n\n", lyrics)

    return lyrics

def clean_lyrics(lyrics: str) -> str:
    """Clean lyrics by removing empty lines and bracketed annotations.
    
    Args:
        lyrics: Raw lyrics text
        
    Returns:
        Cleaned lyrics text
    """
    lines = lyrics.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
        
        # Skip lines that start with [ and end with ]
        if line.strip().startswith('[') and line.strip().endswith(']'):
            continue
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def main():
    parser = argparse.ArgumentParser(description='Fetch and clean song lyrics from Genius')
    parser.add_argument('artist', help='Artist name')
    parser.add_argument('song', help='Song name')
    args = parser.parse_args()
    
    # Get script directory and create input path
    script_dir = Path(__file__).parent
    
    # Create song name slug for directory
    song_slug = re.sub(r'[^a-z0-9]+', '-', args.song.lower()).strip('-')
    
    # Create output directory
    output_dir = script_dir / "input" / song_slug / "lyrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Genius URL
    genius_url = create_genius_url(args.artist, args.song)
    print(f"Fetching lyrics from: {genius_url}")
    
    try:
        # Fetch lyrics
        lyrics = scrape_genius_lyrics(genius_url)
        print(f"✓ Fetched lyrics ({len(lyrics)} characters)")
        
        # Clean lyrics
        cleaned_lyrics = clean_lyrics(lyrics)
        print(f"✓ Cleaned lyrics ({len(cleaned_lyrics)} characters)")
        
        # Write to file
        output_file = output_dir / "lyrics.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_lyrics)
        
        print(f"✓ Saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
