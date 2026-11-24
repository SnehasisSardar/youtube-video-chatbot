"""Split a transcript into overlapping chunks that keep their caption timings."""

from collections.abc import Sequence

from .models import Chunk, Segment


def _words(segment: Segment) -> int:
    return len(segment.text.split())


def chunk_segments(
    segments: Sequence[Segment],
    target_words: int = 50,
    max_words: int = 75,
    overlap_words: int = 15,
) -> list[Chunk]:
    """Group whole caption lines into chunks.

    A chunk closes at the first sentence end after `target_words`, or before it
    would pass `max_words` (auto-captions have no punctuation). The last lines,
    up to `overlap_words`, are repeated in the next chunk so that an idea
    crossing a boundary is still retrievable. Lines are never cut, so every
    word keeps its exact timestamp.
    """
    if not 0 <= overlap_words < target_words <= max_words:
        raise ValueError("need 0 <= overlap_words < target_words <= max_words")

    chunks: list[Chunk] = []
    current: list[Segment] = []
    count = 0
    has_new = False  # False while `current` holds only lines repeated from the last chunk

    def close() -> None:
        nonlocal current, count, has_new
        chunks.append(Chunk(len(chunks), tuple(current)))
        tail: list[Segment] = []
        tail_words = 0
        for segment in reversed(current[1:]):
            if tail_words + _words(segment) > overlap_words:
                break
            tail.insert(0, segment)
            tail_words += _words(segment)
        current, count, has_new = tail, tail_words, False

    for segment in segments:
        size = _words(segment)
        if has_new and count + size > max_words:
            close()
        if not has_new and count + size > max_words:
            current, count = [], 0  # the overlap doesn't fit beside an oversized line
        current.append(segment)
        count += size
        has_new = True
        if count >= target_words and segment.text.rstrip("\"')]").endswith((".", "?", "!")):
            close()
    if has_new:
        chunks.append(Chunk(len(chunks), tuple(current)))
    return chunks
