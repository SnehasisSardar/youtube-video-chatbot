"""Choose which parts of a video to show the model for a question."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .models import Chunk, RetrievedChunk, Segment
from .timestamps import find_times, format_time
from .vector_store import VectorStore

_WHOLE_VIDEO = re.compile(
    r"\b(summar(y|ise|ize)\w*|overview|outline|tl;?dr|recap|"
    r"(key|main|most important)\s+(points?|ideas?|topics?|takeaways?|lessons?|arguments?)|"
    r"what('s| is| was)\s+(this|the|it)\s*(video|talk|lecture|podcast|episode)?\s*(all\s+)?about)\b",
    re.I,
)
# "summarise what she says about pricing" names a topic, so similarity search fits better
_NAMES_TOPIC = re.compile(
    r"\b(about|regarding|concerning|of|on)\s+"
    r"(?!((this|the|that)\s+)?((whole|entire|full)\s+)?"
    r"(video|talk|lecture|podcast|episode|clip|it|this)\b)\w",
    re.I,
)


def is_overview_question(question: str) -> bool:
    match = _WHOLE_VIDEO.search(question)
    return bool(match) and not _NAMES_TOPIC.search(question, match.end())


@dataclass(frozen=True)
class Context:
    results: list[RetrievedChunk]  # in video order
    mode: str  # "search" or "overview"
    complete: bool  # True when the whole video is included

    @property
    def segments(self) -> list[Segment]:
        """Every caption line in the context once, in order (chunks overlap)."""
        unique = {s for r in self.results for s in r.chunk.segments}
        return sorted(unique, key=lambda s: s.start)

    def render(self) -> str:
        """The excerpts as prompt text, one timestamped caption per line."""
        seen: set[Segment] = set()
        excerpts: list[list[str]] = []
        previous = None
        for result in self.results:
            chunk = result.chunk
            if previous is None or chunk.index != previous + 1:
                excerpts.append([])
            previous = chunk.index
            for s in chunk.segments:
                if s not in seen:
                    seen.add(s)
                    excerpts[-1].append(f"[{format_time(s.start)}] {s.text}")
        return "\n".join(
            "<excerpt>\n" + "\n".join(lines) + "\n</excerpt>" for lines in excerpts if lines
        )


def sample_evenly(chunks: Sequence[Chunk], limit: int) -> list[Chunk]:
    if len(chunks) <= limit:
        return list(chunks)
    if limit == 1:
        return [chunks[0]]
    step = (len(chunks) - 1) / (limit - 1)
    return [chunks[round(i * step)] for i in range(limit)]


def retrieve(
    store: VectorStore,
    video_id: str,
    question: str,
    search_query: str | None = None,
    k: int = 8,
    overview_max_chunks: int = 120,
) -> Context:
    """Pick context for `question`.

    - Whole-video questions ("summarise this") get chunks spread over the video,
      because the few chunks most similar to the word "summary" say little.
    - Everything else gets a similarity search, plus whatever is being said at
      any time the question mentions ("what happens at 4:30?").
    """
    chunks = store.all_chunks(video_id)
    times = find_times(question)
    if is_overview_question(question) and not times:
        picked = sample_evenly(chunks, overview_max_chunks)
        return Context(
            [RetrievedChunk(c) for c in picked], "overview", len(picked) == len(chunks)
        )

    found = {r.chunk.index: r for r in store.search(video_id, search_query or question, k)}
    for seconds in times:
        for chunk in chunks:
            if int(chunk.start) <= seconds <= chunk.end:
                found.setdefault(chunk.index, RetrievedChunk(chunk))
    results = [found[i] for i in sorted(found)]
    return Context(results, "search", len(results) == len(chunks))
