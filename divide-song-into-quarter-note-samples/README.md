# Divide Song Into Quarter Note Samples

This pipeline extracts individual quarter-note samples from songs to create source material for drone meditation pieces. The scripts process songs sequentially, each building on the previous step's output.

## Scripts

### 1. separate-song-stems.py

**Input:**
- `input/{SONG_NAME}/audio/` - Original audio file
- `input/{SONG_NAME}/config/config.json` - Song configuration

**Output:**
- `output/{SONG_NAME}/demucs/` - Separated stems (vocals, drums, bass, other)

### 2. divide-song-into-quarter-note-samples.py

**Input:**
- `output/{SONG_NAME}/demucs/` - Separated stems from script 1
- `input/{SONG_NAME}/config/config.json` - Song configuration (reads BPM)

**Output:**
- `output/{SONG_NAME}/quarter-note-samples/` - Full mix divided into quarter-note WAV files
- `output/{SONG_NAME}/quarter-note-samples-acappella/` - Vocals-only divided into quarter-note WAV files

### 3. align-song-lyrics.py

**Input:**
- `output/{SONG_NAME}/demucs/vocals.wav` - Vocals stem from script 1
- `input/{SONG_NAME}/lyrics/*.txt` - Lyrics text file

**Output:**
- `output/{SONG_NAME}/gentle/alignment.json` - Word-level timestamp alignment data

### 4. label-quarter-note-samples-with-lyrics.py

**Input:**
- `output/{SONG_NAME}/gentle/alignment.json` - Alignment data from script 3
- `output/{SONG_NAME}/quarter-note-samples/` - Quarter-note samples from script 2
- `output/{SONG_NAME}/quarter-note-samples-acappella/` - Acappella samples from script 2

**Output:**
- `output/{SONG_NAME}/quarter-note-samples-labeled-with-lyrics/` - Full mix samples with lyrics in filename
- `output/{SONG_NAME}/quarter-note-samples-acappella-labeled-with-lyrics/` - Acappella samples with lyrics in filename

### 5. filter-lyrics-via-chatgpt.py

**Input:**
- `output/{SONG_NAME}/quarter-note-samples-acappella-labeled-with-lyrics/` - Labeled samples from script 4
- `openai/curate-lyrics-prompt.md` - ChatGPT prompt template

**Output:**
- `output/{SONG_NAME}/curated-lyrics-files/` - Filtered samples containing complete lyrical phrases
- `output/{SONG_NAME}/openai/` - ChatGPT analysis and filtered filename list
