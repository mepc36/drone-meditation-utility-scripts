from pathlib import Path

# ------------------------------------------------------------------
# Hard-coded base path
# ------------------------------------------------------------------

BASE = Path(
    "/Users/martinconnor/Music/Music/Media.localized/Music/Maestro/Maestro — The Playlist"
)

OUTPUT = Path("./output")
OUTPUT.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Weights (ORDER MATTERS)
# ------------------------------------------------------------------

weights = [
    ("Breathing.wav", 12),
    ("Being.wav", 8),
    ("Feeling.wav", 8),
    ("Thinking.wav", 6),
    ("Listening.wav", 4),
    ("Faking.wav", 3),
    ("Waiting.wav", 2),
    ("Living.wav", 1),
]

# ------------------------------------------------------------------
# Build ordered playlist
# ------------------------------------------------------------------

items = []

for filename, weight in weights:
    path = BASE / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    items.extend([str(path)] * weight)

# ------------------------------------------------------------------
# Write playlist (renamed)
# ------------------------------------------------------------------

playlist = OUTPUT / "Maestro — The Playlist.m3u"
playlist.write_text(
    "#EXTM3U\n" + "\n".join(items) + "\n",
    encoding="utf-8"
)

print(f"✔ Wrote playlist → {playlist.resolve()}")
print(f"✔ Total entries: {len(items)}")
