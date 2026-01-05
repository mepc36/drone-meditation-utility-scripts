# Lyric Phrase Curation Instructions

You are analyzing lyrics from a hip-hop song that have been time-aligned to audio samples. Each lyric phrase is in kebab-case format (lowercase with hyphens).

## Task

Identify which phrases are **expressive and usable**. A phrase is suitable if it sounds intentional, evocative, or could stand as a meaningful utterance - even if grammatically incomplete.

## What to INCLUDE

**Expressive phrases** - phrases that feel purposeful and evocative:
- **ALL single words**: `breathing`, `living`, `feeling`, `we`, `go`, `down`, `up`, `here`, `now`
- Action verbs: `moving`, `thinking`, `holding`, `running`, `flying`
- Simple statements: `we-go`, `ya-hear`, `i-know`, `lets-go`, `right-here`
- Commands or imperatives: `take-care`, `get-down`, `wake-up`, `hold-on`, `come-on`
- Phrases with poetic "and": `and-we-go`, `and-i-know`, `and-breathe`
- Colloquial expressions: `ya-feel-me`, `you-know-what-im-sayin`, `uh-huh`, `yeah-yeah`
- Evocative fragments: `in-the-moment`, `right-now`, `all-day`, `to-the`, `from-the`
- Repetitive/rhythmic phrases: `go-go`, `yeah-yeah`, `uh-uh`

## What to EXCLUDE

**Non-expressive fragments** - only exclude phrases that sound clearly broken or unintentional:
- Mid-sentence connectors that trail: `and-then-we-are`, `but-if-the`, `so-when-i`
- Awkward partial thoughts: `i-was-about-to`, `we-were-gonna-be`, `that-was-the-one`
- Unclear fragments: `the-uh-so`, `like-when-that`

## Key Principles

1. **Single words are always included**: Any phrase consisting of just one word should be included
2. **Expressive over complete**: A fragment can be powerful even if grammatically incomplete
3. **Colloquial speech**: Hip-hop uses stylized language - trust the artistic choice
4. **When in doubt, include it**: Err on the side of keeping phrases rather than excluding them
5. **Sound matters**: If it sounds like it could be a hook, ad-lib, or intentional phrase, include it

## Your Task

Below is the list of actual lyrics from this song. Return phrases that are expressive and usable.

**IMPORTANT**: You must ONLY return lyrics from the list below. Do not add any phrases not in this list.

### Lyrics to analyze:
{LYRICS_LIST}

## Response Format

Return ONLY a JSON array of the expressive phrases:

```json
["living", "we", "and-we-go", "ya-hear", "take-care", "breathing", "to-the"]
```

Be inclusive - when there's any doubt, include the phrase.

