"""Fetch a video's transcript and title from YouTube."""

import html
import re
from urllib.parse import parse_qs, urlparse

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import _errors as yt_errors

from .models import Segment, VideoInfo

_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")
_HOSTS = {"youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com"}
_PATH_PREFIXES = {"embed", "shorts", "live", "v"}
_ENGLISH = ["en", "en-US", "en-GB", "en-CA", "en-AU", "en-IN"]


class VideoError(Exception):
    """A problem with the video that the user can understand and act on."""


def extract_video_id(value: str) -> str:
    """Accept a YouTube URL in any common form, or a bare video ID."""
    value = value.strip()
    if _ID_RE.fullmatch(value):
        return value

    candidate = ""
    try:
        parsed = urlparse(value if "://" in value else f"https://{value}")
        host = (parsed.hostname or "").removeprefix("www.")
    except ValueError:  # malformed, such as "http://[x"
        raise VideoError("That doesn't look like a YouTube video link.") from None
    parts = [p for p in parsed.path.split("/") if p]
    if host == "youtu.be":
        candidate = parts[0] if parts else ""
    elif host in _HOSTS:
        if len(parts) >= 2 and parts[0] in _PATH_PREFIXES:
            candidate = parts[1]
        else:
            candidate = parse_qs(parsed.query).get("v", [""])[0]
    if not _ID_RE.fullmatch(candidate):
        raise VideoError("That doesn't look like a YouTube video link.")
    return candidate


def _fetch_title(video_id: str) -> tuple[str, str]:
    """Title and channel from the public oEmbed endpoint; cosmetic, so never fatal."""
    try:
        response = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("title") or video_id, data.get("author_name") or ""
    except (requests.RequestException, ValueError):
        return video_id, ""


def _fetch_transcript(video_id: str, api: YouTubeTranscriptApi):
    """Prefer English captions (human-made first), then a translation, then the original."""
    available = api.list(video_id)
    try:
        return available.find_transcript(_ENGLISH).fetch()
    except yt_errors.NoTranscriptFound:
        original = next(iter(available))
    if original.is_translatable:
        try:
            return original.translate("en").fetch()
        except yt_errors.CouldNotRetrieveTranscript:
            pass  # YouTube often rate-limits translation; the original still works
    return original.fetch()


def clean_segments(snippets) -> list[Segment]:
    """Normalise captions: drop blanks and stop each line where the next one starts."""
    rows = []
    for s in snippets:
        text = re.sub(r"\s+", " ", html.unescape(s.text)).strip()
        if text:
            rows.append((float(s.start), float(s.duration), text))
    rows.sort(key=lambda r: r[0])

    segments = []
    for i, (start, duration, text) in enumerate(rows):
        end = start + duration
        if i + 1 < len(rows):
            end = min(end, rows[i + 1][0])  # auto-captions overlap
        segments.append(Segment(start, end, text))
    return segments


def load_video(
    value: str, api: YouTubeTranscriptApi | None = None
) -> tuple[VideoInfo, list[Segment]]:
    """Fetch everything needed to index a video."""
    video_id = extract_video_id(value)
    try:
        transcript = _fetch_transcript(video_id, api or YouTubeTranscriptApi())
    except (yt_errors.TranscriptsDisabled, yt_errors.NoTranscriptFound, StopIteration):
        raise VideoError("This video has no captions, so there is nothing to chat with.") from None
    except (yt_errors.VideoUnavailable, yt_errors.InvalidVideoId):
        raise VideoError("This video doesn't exist or is private.") from None
    except yt_errors.AgeRestricted:
        raise VideoError("This video is age-restricted, so its captions can't be read.") from None
    except yt_errors.VideoUnplayable:
        raise VideoError("YouTube won't play this video, so its captions can't be read.") from None
    except yt_errors.RequestBlocked:  # includes IpBlocked
        raise VideoError(
            "YouTube is blocking transcript requests from this network. "
            "Wait a while, or try another network (cloud servers are often blocked)."
        ) from None
    except yt_errors.CouldNotRetrieveTranscript as exc:
        raise VideoError(f"Couldn't get the captions ({type(exc).__name__}).") from exc
    except requests.RequestException as exc:
        raise VideoError("Couldn't reach YouTube. Check your connection.") from exc

    segments = clean_segments(transcript.snippets)
    if not segments:
        raise VideoError("This video's captions are empty.")
    title, channel = _fetch_title(video_id)
    info = VideoInfo(
        video_id=video_id,
        title=title,
        channel=channel,
        language=transcript.language_code,
        duration=segments[-1].end,
    )
    return info, segments
