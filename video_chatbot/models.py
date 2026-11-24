"""Plain data types shared by the pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """One caption line."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class VideoInfo:
    video_id: str
    title: str
    channel: str = ""
    language: str = "en"
    duration: float = 0.0
    chunk_count: int = 0

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


@dataclass(frozen=True)
class Chunk:
    """A run of caption lines that is embedded and retrieved as one unit."""

    index: int
    segments: tuple[Segment, ...]

    @property
    def start(self) -> float:
        return self.segments[0].start

    @property
    def end(self) -> float:
        return self.segments[-1].end

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float | None = None  # cosine similarity; None when not found by vector search
