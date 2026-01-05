# Lyric Phrase Curation Instructions

You are analyzing lyrics from a hip-hop song that have been time-aligned to audio samples. Each lyric phrase is in kebab-case format (lowercase with hyphens).

## Task

Identify which phrases are **fully formed and self-contained**. A phrase is complete if it expresses a whole thought or action that could stand alone, even in colloquial speech.

## What to INCLUDE

**Complete phrases** - phrases that feel finished and make sense on their own:
- Single action verbs: `breathing`, `living`, `feeling`, `moving`, `thinking`
- Simple statements: `we-go`, `ya-hear`, `i-know`, `we-here`
- Commands or imperatives: `take-care`, `get-down`, `wake-up`, `hold-on`
- Phrases with poetic "and": `and-we-go`, `and-i-know`, `and-breathe` (the "and" adds poetic effect)
- Complete colloquial expressions: `ya-feel-me`, `you-know-what-im-sayin`, `lets-go`
- Short but complete thoughts: `in-the-moment`, `right-now`, `all-day`

## What to EXCLUDE

**Incomplete phrases** - fragments that trail off or need more words:
- Hanging prepositions: `going-to-the`, `in-the`, `at-a`, `with-the`
- Incomplete thoughts: `we-are-going`, `i-want-to`, `they-be`, `she-got`
- Partial verb phrases: `have-been`, `will-be`, `gonna-get`
- Dangling articles: `the-way-we`, `a-little-bit-of`
- Unfinished comparisons: `more-than`, `better-than-the`

## Key Principles

1. **Colloquial speech matters**: Hip-hop often uses shortened or stylized phrases that are complete in context
2. **Poetic "and"**: Starting with "and" doesn't make a phrase incomplete if the rest stands alone
3. **Context-free test**: Ask "could this phrase exist as a complete utterance in conversation?"
4. **Don't judge content**: Include all complete phrases regardless of meaning, tone, or appropriateness

## Your Task

Below is the list of actual lyrics from this song. Return ONLY phrases that are fully formed and complete.

**IMPORTANT**: You must ONLY return lyrics from the list below. Do not add any phrases not in this list.

### Lyrics to analyze:
{LYRICS_LIST}

## Response Format

Return ONLY a JSON array of the complete phrases:

```json
["living", "and-we-go", "ya-hear", "take-care", "breathing"]
```

Be inclusive - if there's any doubt whether a phrase is complete, include it.
