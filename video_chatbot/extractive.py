"""Answers without a language model: quote the best-matching sentences."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .models import Segment
from .retriever import Context
from .timestamps import find_times
from .vector_store import VectorStore

# Best-chunk similarity on the sample video: answerable questions scored 0.45 or
# more and unrelated ones 0.33 or less. Below this, say nothing was found.
MIN_SIMILARITY = 0.38
_MAX_QUOTES = 3
_BEST_CHUNKS = 4
_QUOTE_WORDS = 45
_LINE_WORDS = 35  # sentence length assumed for captions without punctuation
_LONGEST_SENTENCE = 100
_SENTENCE_END = re.compile(r"[.?!][\"')\]]*$")


@dataclass(frozen=True)
class Quote:
    start: float
    text: str


def to_sentences(segments: Sequence[Segment]) -> list[Quote]:
    """Join caption lines into sentences, each with the time its first line starts."""
    every_word = [w for s in segments for w in s.text.split()]
    ends = sum(bool(_SENTENCE_END.search(w)) for w in every_word)
    punctuated = ends * _LINE_WORDS >= len(every_word)
    limit = _LONGEST_SENTENCE if punctuated else _LINE_WORDS

    sentences: list[Quote] = []
    words: list[str] = []
    start = 0.0
    for segment in segments:
        for word in segment.text.split():
            if not words:
                start = segment.start
            words.append(word)
            if _SENTENCE_END.search(word) or len(words) >= limit:
                sentences.append(Quote(start, " ".join(words)))
                words = []
    if words:
        sentences.append(Quote(start, " ".join(words)))
    return sentences


def _extend(sentences: Sequence[Quote], i: int) -> tuple[Quote, int]:
    """Sentence i plus the following ones that fit, and the index after the last."""
    text, j = sentences[i].text, i + 1
    while j < len(sentences) and len(f"{text} {sentences[j].text}".split()) <= _QUOTE_WORDS:
        text = f"{text} {sentences[j].text}"
        j += 1
    return Quote(sentences[i].start, text), j


def _all_sentences(store: VectorStore, video_id: str) -> list[Quote]:
    unique = {s for c in store.all_chunks(video_id) for s in c.segments}
    return to_sentences(sorted(unique, key=lambda s: s.start))


def _end(store: VectorStore, video_id: str) -> float:
    info = store.get(video_id)
    return info.duration if info else 0.0


def _quotes_at(sentences: Sequence[Quote], times: Sequence[int]) -> list[Quote]:
    found = set()
    for seconds in times:
        spoken = [i for i, s in enumerate(sentences) if int(s.start) <= seconds]
        if spoken:
            found.add(spoken[-1])
    return [_extend(sentences, i)[0] for i in sorted(found)]


def quote_answer(
    store: VectorStore, video_id: str, question: str, context: Context
) -> list[Quote]:
    """The most relevant quotes in video order, or [] if nothing matches well."""
    times = find_times(question)
    if times:  # a request for that moment, whatever its similarity
        sentences = _all_sentences(store, video_id)
        return _quotes_at(sentences, [t for t in times if t <= _end(store, video_id)])

    if max((r.score or 0 for r in context.results), default=0) < MIN_SIMILARITY:
        return []
    # Embedding the candidates is the slow step, so only look inside the best chunks.
    best = sorted(context.results, key=lambda r: r.score or 0, reverse=True)[:_BEST_CHUNKS]
    chunks = [r.chunk for r in best]
    sentences = _all_sentences(store, video_id)
    candidates = [
        (i, *_extend(sentences, i))
        for i, s in enumerate(sentences)
        if any(c.start <= s.start < c.end for c in chunks)
    ]
    if not candidates:
        return []
    vectors = store.embed([question] + [quote.text for _, quote, _ in candidates])
    similarity = vectors[1:] @ vectors[0]

    picked: list[tuple[int, Quote, int]] = []
    for n in np.argsort(-similarity):
        first, quote, after = candidates[n]
        if len(picked) == _MAX_QUOTES or (picked and similarity[n] < MIN_SIMILARITY):
            break
        if all(after <= f or first >= a for f, _, a in picked):
            picked.append((first, quote, after))
    return [quote for _, quote, _ in sorted(picked)]


def quote_overview(store: VectorStore, video_id: str, parts: int = 6) -> list[Quote]:
    """One typical quote from each part of the video."""
    chunks, vectors = store.chunk_vectors(video_id)
    sentences = _all_sentences(store, video_id)
    quotes: list[Quote] = []
    for group in np.array_split(np.arange(len(chunks)), min(parts, len(chunks))):
        centre = vectors[group].mean(axis=0)
        chunk = chunks[group[int(np.argmax(vectors[group] @ centre))]]
        # Start where a sentence starts, not where the chunk happens to be cut.
        i = next((i for i, s in enumerate(sentences) if s.start >= chunk.start), None)
        if i is None:
            continue
        quote, _ = _extend(sentences, i)
        if quote not in quotes:
            quotes.append(quote)
    return quotes
