"""Getting from a YouTube link to timestamped chunks."""

from types import SimpleNamespace

import pytest
from youtube_transcript_api import _errors as yt_errors

from video_chatbot import youtube
from video_chatbot.chunking import chunk_segments
from video_chatbot.models import Segment
from video_chatbot.youtube import VideoError, clean_segments, extract_video_id, load_video

ID = "aircAruvnKk"


@pytest.mark.parametrize(
    "value",
    [
        ID,
        f"  {ID}\n",
        f"https://www.youtube.com/watch?v={ID}",
        f"http://youtube.com/watch?list=PL1&v={ID}&t=42s",
        f"youtube.com/watch?v={ID}",
        f"https://m.youtube.com/watch?v={ID}",
        f"https://music.youtube.com/watch?v={ID}",
        f"https://youtu.be/{ID}?si=xyz",
        f"https://www.youtube.com/shorts/{ID}",
        f"https://www.youtube.com/live/{ID}?feature=share",
        f"https://www.youtube-nocookie.com/embed/{ID}",
    ],
)
def test_extracts_id(value):
    assert extract_video_id(value) == ID


@pytest.mark.parametrize(
    "value",
    [
        "",
        "hello",
        "https://vimeo.com/12345678901",
        f"https://evil.com/watch?v={ID}",
        f"https://notyoutube.com/watch?v={ID}",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/@3blue1brown",
        "https://youtu.be/",
        "http://[x",
    ],
)
def test_rejects_other_links(value):
    with pytest.raises(VideoError):
        extract_video_id(value)


def snippet(start, duration, text):
    return SimpleNamespace(start=start, duration=duration, text=text)


def test_clean_segments():
    result = clean_segments(
        [snippet(3, 4, "second"), snippet(0, 5, "it&#39;s\nfirst "), snippet(2, 1, " \n")]
    )
    assert result == [Segment(0, 3, "it's first"), Segment(3, 7, "second")]


class FakeTranscript:
    def __init__(self, code, translatable=False, translate_error=None):
        self.language_code, self.is_translatable = code, translatable
        self._error = translate_error

    def translate(self, code):
        if self._error:
            raise self._error
        return FakeTranscript(code)

    def fetch(self):
        return SimpleNamespace(
            language_code=self.language_code, snippets=[snippet(0, 2, "hello world")]
        )


class FakeList(list):
    def find_transcript(self, codes):
        for t in self:
            if t.language_code in codes:
                return t
        raise yt_errors.NoTranscriptFound(ID, codes, self)


class FakeApi:
    def __init__(self, transcripts=(), error=None):
        self._list, self._error = FakeList(transcripts), error

    def list(self, video_id):
        if self._error:
            raise self._error
        return self._list


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(youtube, "_fetch_title", lambda video_id: ("Title", "Channel"))


@pytest.mark.parametrize(
    "transcripts, expected",
    [
        ([FakeTranscript("fr"), FakeTranscript("en")], "en"),
        ([FakeTranscript("fr", translatable=True)], "en"),
        ([FakeTranscript("fr")], "fr"),
        (
            [FakeTranscript("fr", True, yt_errors.RequestBlocked(ID))],
            "fr",
        ),
    ],
)
def test_language_preference(transcripts, expected):
    info, segments = load_video(ID, FakeApi(transcripts))
    assert (info.language, info.title, info.duration) == (expected, "Title", 2)
    assert segments[0].text == "hello world"


@pytest.mark.parametrize(
    "error, message",
    [
        (yt_errors.TranscriptsDisabled(ID), "no captions"),
        (yt_errors.VideoUnavailable(ID), "doesn't exist"),
        (yt_errors.AgeRestricted(ID), "age-restricted"),
        (yt_errors.IpBlocked(ID), "blocking"),
        (yt_errors.PoTokenRequired(ID), "PoTokenRequired"),
    ],
)
def test_youtube_errors_are_explained(error, message):
    with pytest.raises(VideoError, match=message):
        load_video(ID, FakeApi(error=error))


def test_no_transcripts_at_all():
    with pytest.raises(VideoError, match="no captions"):
        load_video(ID, FakeApi([]))


def line(i, words, end="."):
    return Segment(i, i + 1, " ".join(f"w{i}" for _ in range(words)) + end)


def test_chunks_cover_every_line_in_order_and_overlap():
    lines = [line(i, 10) for i in range(50)]
    chunks = chunk_segments(lines, 40, 60, 20)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert list(dict.fromkeys(s for c in chunks for s in c.segments)) == lines
    for a, b in zip(chunks, chunks[1:]):
        assert a.segments[-2:] == b.segments[:2]  # 20 words of overlap
    assert chunks[0].start == 0 and chunks[-1].end == 50


def test_unpunctuated_captions_respect_the_maximum():
    chunks = chunk_segments([line(i, 7, "") for i in range(200)], 110, 150, 30)
    assert all(len(c.text.split()) <= 150 for c in chunks)
    assert len(chunks) > 5


def test_oversized_line_is_kept_whole():
    lines = [line(0, 20), line(1, 500), line(2, 20)]
    chunks = chunk_segments(lines, 40, 60, 25)
    assert list(dict.fromkeys(s for c in chunks for s in c.segments)) == lines
    assert len(chunks) == 3


def test_last_chunk_is_never_only_overlap():
    chunks = chunk_segments([line(i, 10) for i in range(4)], 40, 60, 20)
    assert len(chunks) == 1


def test_chunking_edge_cases():
    assert chunk_segments([]) == []
    assert len(chunk_segments([line(0, 1)])) == 1
    with pytest.raises(ValueError):
        chunk_segments([line(0, 1)], 10, 20, 10)
