import re
import requests
from bs4 import BeautifulSoup
import argparse
import sys
from pathlib import Path
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CLEAN_UP_LYRICS_VIA_GPT = False

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

def clean_lyrics_via_regex(lyrics: str) -> str:
    """Clean lyrics by removing empty lines, bracketed annotations, and commas (regex method).
    
    Args:
        lyrics: Raw lyrics text
        
    Returns:
        Cleaned lyrics text
    """
    # Remove everything before "read more" (case insensitive)
    lyrics = re.sub(r'^.*?read more', '', lyrics, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove all content between square brackets (including multi-line)
    lyrics = re.sub(r'\[.*?\]', '', lyrics, flags=re.DOTALL)
    
    # Remove all content in parentheses (ad-libs like (Ha), (Ho!), etc.)
    lyrics = re.sub(r'\([^)]*\)', '', lyrics)
    
    # Remove all commas
    lyrics = lyrics.replace(',', '')
    
    # Split into lines and clean up
    lines = lyrics.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Strip whitespace
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip lines with no alphanumeric characters
        if not re.search(r'[a-zA-Z0-9]', line):
            continue
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def clean_lyrics_via_gpt(lyrics: str, api_key: str) -> str:
    """Clean lyrics using GPT to remove annotations, headers, and metadata.
    
    Args:
        lyrics: Raw lyrics text
        api_key: OpenAI API key
        
    Returns:
        Cleaned lyrics text
    """
    client = OpenAI(api_key=api_key)
    
    # Create prompt with example
    prompt = f"""You are cleaning song lyrics fetched from Genius.com. Remove all the following:
1. All section headers like [Intro: 50 Cent], [Verse 1: 50 Cent], [Chorus: 50 Cent], [Outro: 50 Cent], etc.
2. All site metadata and advertising content like "See rap shows near Philadelphia", "Get tickets as low as $42", "You might also like", etc.
3. Any other non-lyric content
4. Any empty lines

Keep ONLY the actual song lyrics - the words that are sung/rapped in the song.

**EXAMPLE INPUT:**

[Intro: 50 Cent]
Yeah
Haha, yeah
Yeah

[Chorus: 50 Cent]
If I can't do it, homie
It can't be done
Now I'ma let the champagne bottle pop, I'ma take it to the top
For sure, I'ma make it hot, baby (Baby)

[Verse 1: 50 Cent]
I apply pressure to pussies, they stunt and I pop (Yeah)
Stand alone squeezing my pistol, I'm sure that I got 'em (Uh-huh)
Now, Peter Piper picked peppers, and Run rocked rhymes
I'm 50 Cent, I write a little bit, but I pop nines (Brrat)
Tell niggas get they money right, 'cause I got mine (Uh-huh)
And I'm around, quit playing, nigga, you can't shine (Woo)
You gon' be that next chump to end up in the trunk
After being hit by the pump, is that what you want?
Be easy, nigga, I'll lay your ass out
Believe me, nigga, that's what I'm about
Gangsta, you could find a nigga sitting on chrome
Hit the clutch, hit the gear, hit the gas and I'm gone (Yeah)

[Chorus: 50 Cent]
If I can't do it, homie
It can't be done
Now, I'ma let the champagne bottle pop, I'ma take it to the top
For sure, I'ma make it hot, baby (Baby)
See rap shows near Philadelphia
Get tickets as low as $42

You might also like
Many Men (Wish Death)
50 Cent
Heat
50 Cent
Family Matters
Drake

[Verse 2: 50 Cent & Dr. Dre]
I'm down for the action, he smart with his mouth, so smack him (Woo)
You holding a strap, he might come back, so clap him (Yeah)
React like a gangsta, or die like a gangsta for acting (C'mon)
'Cause you'll get hit and homicide'll be asking, "What happened?"
Oh no, look who crept in with the .44
Twenty-inch rims sitting on low-pros (Uh-huh)
East side, west side, niggas know, yo, I'm loco (Yeah)
Even my mama said something really wrong with my brain
Niggas don't rob me, they know I'm down to die for my chain
G-Unit (Yeah), we get it popping in the hood
G-Unit (Yeah), motherfucker, what's good?
I'm waiting on niggas to act like they don't know how to act (Uh-huh)
After sippin' too much Jack, I'll blow 'em off the map
With the MAC, thinking it's all rap
'Til that ass get clapped and Doc say, "It's a wrap" (It's a wrap, nigga)

[Chorus: 50 Cent]
If I can't do it, homie
It can't be done
Now, I'ma let the champagne bottle pop (Uh-huh), I'ma take it to the top
For sure, I'ma make it hot, baby (Baby)

[Verse 3: 50 Cent]
I invented how to teach lessons to slow learners
Go ahead, act up, get smacked in the head with the burner (Ah)
I don't fight fair, I'm dirty, dirty
I'm from Southside Jamaica, Queens, nigga, you heard me? (Yeah)
When the streetlights come on, niggas blast the nines (Uh-huh)
Get locked up, then read books to pass the time (Woo)
In the game, there's ups and downs, so I stay on the grind
Niggas on my dick more than my bitch, I stay on they mind
There ain't nothing they could do to stop my shine (Uh-uh)
This is God's plan, homie, this ain't mine
I played the music loud so Grandpa called me a nuisance
And Grandma always gotta throw in her two cents
I'm the dropout who made more money than his teachers
Roofless like the coupe, but I come with more features
I am what I am, you can like it or love it
It feels good to blow fifty grand and think nothing of it, fuck it


[Chorus: 50 Cent]
If I can't do it, homie
It can't be done
Now I'ma let the champagne bottle pop, I'ma take it to the top
For sure, I'ma make it hot, baby (Baby)
If I can't do it, homie
It can't be done (Haha)
Now I'ma let the champagne bottle pop, I'ma take it to the top (Yeah)
For sure I'ma make it hot, baby (Baby)

[Outro: 50 Cent]
Uh-huh
I'ma make it hot
Dr. Dre, Aftermath
Shady, haha

**EXAMPLE OUTPUT:**

Yeah
Haha, yeah
Yeah
If I can't do it, homie
It can't be done
Now I'ma let the champagne bottle pop, I'ma take it to the top
For sure, I'ma make it hot, baby (Baby)
I apply pressure to pussies, they stunt and I pop (Yeah)
Stand alone squeezing my pistol, I'm sure that I got 'em (Uh-huh)
Now, Peter Piper picked peppers, and Run rocked rhymes
I'm 50 Cent, I write a little bit, but I pop nines (Brrat)
Tell niggas get they money right, 'cause I got mine (Uh-huh)
And I'm around, quit playing, nigga, you can't shine (Woo)
You gon' be that next chump to end up in the trunk
After being hit by the pump, is that what you want?
Be easy, nigga, I'll lay your ass out
Believe me, nigga, that's what I'm about
Gangsta, you could find a nigga sitting on chrome
Hit the clutch, hit the gear, hit the gas and I'm gone (Yeah)
If I can't do it, homie
It can't be done
Now, I'ma let the champagne bottle pop, I'ma take it to the top
For sure, I'ma make it hot, baby (Baby)
I'm down for the action, he smart with his mouth, so smack him (Woo)
You holding a strap, he might come back, so clap him (Yeah)
React like a gangsta, or die like a gangsta for acting (C'mon)
'Cause you'll get hit and homicide'll be asking, "What happened?"
Oh no, look who crept in with the .44
Twenty-inch rims sitting on low-pros (Uh-huh)
East side, west side, niggas know, yo, I'm loco (Yeah)
Even my mama said something really wrong with my brain
Niggas don't rob me, they know I'm down to die for my chain
G-Unit (Yeah), we get it popping in the hood
G-Unit (Yeah), motherfucker, what's good?
I'm waiting on niggas to act like they don't know how to act (Uh-huh)
After sippin' too much Jack, I'll blow 'em off the map
With the MAC, thinking it's all rap
'Til that ass get clapped and Doc say, "It's a wrap" (It's a wrap, nigga)
If I can't do it, homie
It can't be done
Now, I'ma let the champagne bottle pop (Uh-huh), I'ma take it to the top
For sure, I'ma make it hot, baby (Baby)
I invented how to teach lessons to slow learners
Go ahead, act up, get smacked in the head with the burner (Ah)
I don't fight fair, I'm dirty, dirty
I'm from Southside Jamaica, Queens, nigga, you heard me? (Yeah)
When the streetlights come on, niggas blast the nines (Uh-huh)
Get locked up, then read books to pass the time (Woo)
In the game, there's ups and downs, so I stay on the grind
Niggas on my dick more than my bitch, I stay on they mind
There ain't nothing they could do to stop my shine (Uh-uh)
This is God's plan, homie, this ain't mine
I played the music loud so Grandpa called me a nuisance
And Grandma always gotta throw in her two cents
I'm the dropout who made more money than his teachers
Roofless like the coupe, but I come with more features
I am what I am, you can like it or love it
It feels good to blow fifty grand and think nothing of it, fuck it
If I can't do it, homie
It can't be done
Now I'ma let the champagne bottle pop, I'ma take it to the top
For sure, I'ma make it hot, baby (Baby)
If I can't do it, homie
It can't be done (Haha)
Now I'ma let the champagne bottle pop, I'ma take it to the top (Yeah)
For sure I'ma make it hot, baby (Baby)
Uh-huh
I'ma make it hot
Dr. Dre, Aftermath
Shady, haha

---

Now clean the following lyrics:

{lyrics}

Return ONLY the cleaned lyrics with no additional commentary."""
    
    print("Sending lyrics to GPT for cleaning...")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    cleaned_lyrics = response.choices[0].message.content.strip()
    
    return cleaned_lyrics

def is_url(text: str) -> bool:
    """Check if a string is a URL.
    
    Args:
        text: String to check
        
    Returns:
        True if the string is a URL, False otherwise
    """
    return text.startswith('http://') or text.startswith('https://')


def extract_song_name_from_url(url: str) -> str:
    """Extract song name from a Genius URL for directory naming.
    
    Args:
        url: Genius URL
        
    Returns:
        Song name slug extracted from URL
    """
    # Extract the path component from the URL
    # Example: https://genius.com/artist-song-name-lyrics -> artist-song-name-lyrics
    path = url.rstrip('/').split('/')[-1]
    
    # Remove -lyrics suffix if present
    if path.endswith('-lyrics'):
        path = path[:-7]
    
    return path


def main():
    parser = argparse.ArgumentParser(
        description='Fetch and clean song lyrics from Genius',
        epilog='Usage: python 0-fetch-lyrics.py <artist> <song> OR python 0-fetch-lyrics.py <url>'
    )
    parser.add_argument('artist_or_url', help='Artist name or Genius URL')
    parser.add_argument('song', nargs='?', help='Song name (not required if URL is provided)')
    args = parser.parse_args()
    
    # Get script directory and create input path
    script_dir = Path(__file__).parent
    
    # Check if first argument is a URL
    is_url_mode = is_url(args.artist_or_url)
    
    if is_url_mode:
        genius_url = args.artist_or_url
        # Extract song name from URL for directory naming
        song_slug = extract_song_name_from_url(genius_url)
        print(f"Detected URL input")
    else:
        # Traditional mode: artist and song name
        if not args.song:
            print("❌ Error: Song name is required when not using a URL", file=sys.stderr)
            print("  Usage: python 0-fetch-lyrics.py <artist> <song>", file=sys.stderr)
            print("     OR: python 0-fetch-lyrics.py <url>", file=sys.stderr)
            sys.exit(1)
        
        # Create song name slug for directory
        song_slug = re.sub(r'[^a-z0-9]+', '-', args.song.lower()).strip('-')
        
        # Create Genius URL
        genius_url = create_genius_url(args.artist_or_url, args.song)
    
    # Create output directory
    if is_url_mode:
        # When URL is provided, write to current directory
        output_dir = script_dir
    else:
        # When artist/song provided, write to input directory structure
        output_dir = script_dir / "input" / song_slug / "lyrics"
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching lyrics from: {genius_url}")
    
    try:
        # Fetch lyrics
        lyrics = scrape_genius_lyrics(genius_url)
        print(f"✓ Fetched lyrics ({len(lyrics)} characters)")
        
        # Clean lyrics based on environment variable
        if CLEAN_UP_LYRICS_VIA_GPT:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("❌ Error: OPENAI_API_KEY not found in .env file", file=sys.stderr)
                print("  Set CLEAN_UP_LYRICS_VIA_GPT=false to use regex cleaning instead", file=sys.stderr)
                sys.exit(1)
            
            print("Cleaning lyrics via GPT...")
            cleaned_lyrics = clean_lyrics_via_gpt(lyrics, api_key)
        else:
            print("Cleaning lyrics via regex...")
            cleaned_lyrics = clean_lyrics_via_regex(lyrics)
        
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
