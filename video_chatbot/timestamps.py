"""Formatting, parsing and checking of video timestamps."""

import re
from bisect import bisect_right
from collections.abc import Sequence

from .models import Segment

_TIME = r"(?:\d{1,3}:)?\d{1,3}:\d{2}"
_TIME_RE = re.compile(rf"(?<![\d:]){_TIME}(?![\d:])")
# [2:17], [2:17-2:30], [2:17, 1:03:10] ...
_CITATION_RE = re.compile(rf"\s*\[\s*({_TIME}(?:\s*(?:[-–—,;]|to|and)\s*{_TIME})*)\s*\]")
_PARTIAL_CITATION_RE = re.compile(r"\s*\[[\d:\s,;–—-]*$")


def format_time(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def parse_time(text: str) -> int:
    seconds = 0
    for part in text.split(":"):
        seconds = seconds * 60 + int(part)
    return seconds


def find_times(text: str) -> list[int]:
    """Every timestamp mentioned in free text, in seconds."""
    return [parse_time(m) for m in _TIME_RE.findall(text)]


def watch_url(video_id: str, seconds: float) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}s"


def _match_segment(seconds: int, segments: Sequence[Segment]) -> Segment | None:
    """The segment being spoken at `seconds`, or None if none of them covers it."""
    # Labels drop the fraction of a second, so compare against whole-second starts.
    i = bisect_right([int(s.start) for s in segments], seconds) - 1
    if i < 0:
        return None
    segment = segments[i]
    return segment if seconds <= segment.end else None


def link_citations(
    answer: str, video_id: str, segments: Sequence[Segment], streaming: bool = False
) -> tuple[str, list[int]]:
    """Turn [m:ss] citations into YouTube links.

    Only timestamps inside the excerpts the model was given survive; any other
    timestamp was invented, so it is removed. Returns the linked text and the
    cited times, first mention first. With `streaming`, a half-received
    citation at the end is held back.
    """
    ordered = sorted(segments, key=lambda s: s.start)
    cited: list[int] = []

    def replace(match: re.Match) -> str:
        links = []
        for seconds in find_times(match.group(1)):
            segment = _match_segment(seconds, ordered)
            if segment is None:
                continue
            start = int(segment.start)
            link = f"[\\[{format_time(start)}\\]]({watch_url(video_id, start)})"
            if link not in links:
                links.append(link)
            if start not in cited:
                cited.append(start)
        return " " + " ".join(links) if links else ""

    if streaming:
        answer = _PARTIAL_CITATION_RE.sub("", answer)
    return _CITATION_RE.sub(replace, answer).lstrip(" "), cited
