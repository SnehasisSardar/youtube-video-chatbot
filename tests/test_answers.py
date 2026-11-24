"""Citations, the Claude client and the pipeline that joins them."""

import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from tests.conftest import VIDEO_ID
from video_chatbot.config import Settings
from video_chatbot.llm import ClaudeGenerator, LLMError, MissingCredentials, Turn
from video_chatbot.models import Segment
from video_chatbot.pipeline import VideoChatbot
from video_chatbot.timestamps import find_times, format_time, link_citations, parse_time

SEGMENTS = [Segment(4.2, 9.0, "a"), Segment(9.0, 14.5, "b"), Segment(3723.9, 3730, "c")]


@pytest.mark.parametrize(
    "seconds, text", [(0, "0:00"), (65.9, "1:05"), (599, "9:59"), (3723.9, "1:02:03")]
)
def test_format_time(seconds, text):
    assert format_time(seconds) == text
    assert parse_time(text) == int(seconds)


def test_find_times():
    assert find_times("at 4:30 and 1:02:03, ratio 16:9, 3pm") == [270, 3723]


def link(text, t):
    return f"[\\[{text}\\]](https://www.youtube.com/watch?v={VIDEO_ID}&t={t}s)"


@pytest.mark.parametrize(
    "answer, expected, times",
    [
        ("Yes [0:04].", f"Yes {link('0:04', 4)}.", [4]),
        ("Yes [1:02:03]", f"Yes {link('1:02:03', 3723)}", [3723]),
        ("Mid-line [0:07]", f"Mid-line {link('0:04', 4)}", [4]),
        ("Two [0:04, 0:09]", f"Two {link('0:04', 4)} {link('0:09', 9)}", [4, 9]),
        ("Range [0:04–0:09]", f"Range {link('0:04', 4)} {link('0:09', 9)}", [4, 9]),
        ("Invented [8:00].", "Invented.", []),
        ("Mixed [0:04] [8:00]", f"Mixed {link('0:04', 4)}", [4]),
        ("Before start [0:01]", "Before start", []),
        ("Not a time [1] [abc]", "Not a time [1] [abc]", []),
    ],
)
def test_link_citations(answer, expected, times):
    assert link_citations(answer, VIDEO_ID, SEGMENTS) == (expected, times)


def test_streaming_hides_unfinished_citation():
    text, _ = link_citations("Yes [0:0", VIDEO_ID, SEGMENTS, streaming=True)
    assert text == "Yes"
    assert link_citations("Yes [0:0", VIDEO_ID, SEGMENTS)[0] == "Yes [0:0"


# --- Claude client ---------------------------------------------------------


class FakeStream:
    def __init__(self, tokens, stop_reason):
        self.text_stream, self._stop = iter(tokens), stop_reason

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get_final_message(self):
        return SimpleNamespace(stop_reason=self._stop)


class FakeClient:
    def __init__(self, tokens=("Bread ", "[0:00]"), stop="end_turn", rewrite="full question", error=None):
        self.calls = []
        self.beta = SimpleNamespace(messages=self)
        self._tokens, self._stop, self._rewrite, self._error = tokens, stop, rewrite, error

    def stream(self, **kwargs):
        self.calls.append(("stream", kwargs))
        if self._error:
            raise self._error
        return FakeStream(self._tokens, self._stop)

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        if self._error:
            raise self._error
        return SimpleNamespace(
            stop_reason=self._stop,
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=self._rewrite),
            ],
        )


def api_error(cls, status):
    request = httpx2.Request("POST", "http://test")
    return cls("bad", response=httpx2.Response(status, request=request), body=None)


def generator(client, **settings):
    return ClaudeGenerator(Settings(**{"model": "claude-opus-5-5", **settings}), client=client)


def test_prompt_contains_grounding_context_and_history(info):
    client = FakeClient()
    turns = [Turn(f"q{i}", f"a{i}") for i in range(10)]
    tokens = generator(client, history_turns=2).stream_answer(info, "why?", "EXCERPTS", turns)
    assert "".join(tokens) == "Bread [0:00]"
    _, sent = client.calls[0]
    assert [m["content"] for m in sent["messages"][:-1]] == ["q8", "a8", "q9", "a9"]
    last = sent["messages"][-1]["content"]
    assert "EXCERPTS" in last and "why?" in last and info.title in last
    assert "Use only the video details and what the excerpts say" in sent["system"]
    assert "sample" not in last


def test_sampled_overview_tells_the_model(info):
    client = FakeClient()
    list(generator(client).stream_answer(info, "q", "c", partial_coverage=True))
    assert "evenly spaced samples" in client.calls[0][1]["messages"][-1]["content"]


@pytest.mark.parametrize("model, expected", [("claude-opus-5-5", True), ("claude-haiku-5-5", False)])
def test_fallbacks_only_where_supported(info, model, expected):
    client = FakeClient()
    list(generator(client, model=model).stream_answer(info, "q", "c"))
    assert ("fallbacks" in client.calls[0][1]) is expected


def test_refusal_and_truncation(info):
    with pytest.raises(LLMError, match="declined"):
        list(generator(FakeClient(stop="refusal")).stream_answer(info, "q", "c"))
    text = "".join(generator(FakeClient(stop="max_tokens")).stream_answer(info, "q", "c"))
    assert "cut off" in text


@pytest.mark.parametrize(
    "error, kind, message",
    [
        (api_error(anthropic.AuthenticationError, 401), MissingCredentials, "rejected"),
        (api_error(anthropic.RateLimitError, 429), LLMError, "Rate limited"),
        (api_error(anthropic.InternalServerError, 500), LLMError, "trouble"),
        (api_error(anthropic.BadRequestError, 400), LLMError, "rejected the request"),
        (
            anthropic.APIConnectionError(request=httpx2.Request("POST", "http://test")),
            LLMError,
            "connection",
        ),
    ],
)
def test_api_errors_are_explained(info, error, kind, message):
    with pytest.raises(kind, match=message):
        list(generator(FakeClient(error=error)).stream_answer(info, "q", "c"))


def test_rewrite_query(info):
    turns = [Turn("what is sourdough?", "Bread.")]
    client = FakeClient(rewrite=" why sourdough rises ")
    assert generator(client).rewrite_query(info, "why?", turns) == "why sourdough rises"
    assert generator(client).rewrite_query(info, "first question", []) == "first question"
    assert len(client.calls) == 1  # a first question costs no extra call


def test_rewrite_failure_keeps_the_question(info):
    turns = [Turn("q", "a")]
    for client in (
        FakeClient(error=api_error(anthropic.RateLimitError, 429)),
        FakeClient(stop="refusal"),
        FakeClient(rewrite=""),
    ):
        assert generator(client).rewrite_query(info, "why?", turns) == "why?"


def test_real_sdk_sends_and_streams(info):
    """The real SDK over a mocked connection: checks parameter names and parsing."""
    seen = {}

    def handle(request):
        seen["body"] = json.loads(request.content)
        seen["beta"] = request.headers.get("anthropic-beta", "")
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1", "type": "message", "role": "assistant", "content": [],
                    "model": "claude-opus-5-5", "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Bread "}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "[0:00]"}},
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 5},
            },
            {"type": "message_stop"},
        ]
        body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    client = anthropic.Anthropic(
        api_key="test",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handle)),
    )
    tokens = list(generator(client, effort="low").stream_answer(info, "q", "c"))
    assert tokens == ["Bread ", "[0:00]"]
    assert seen["body"]["model"] == "claude-opus-5-5"
    assert seen["body"]["output_config"] == {"effort": "low"}
    assert seen["body"]["fallbacks"] == "default"
    assert "server-side-fallback" in seen["beta"]
    assert seen["body"]["stream"] is True


def test_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(MissingCredentials):
        ClaudeGenerator(Settings())


# --- pipeline --------------------------------------------------------------


def test_pipeline_end_to_end(store, info, segments, monkeypatch):
    calls = []

    def fake_load(url):
        calls.append(url)
        return info, segments

    monkeypatch.setattr("video_chatbot.pipeline.load_video", fake_load)
    settings = Settings(chunk_target_words=40, chunk_max_words=60, chunk_overlap_words=10, top_k=2)
    bot = VideoChatbot(settings, store)
    url = f"https://youtu.be/{VIDEO_ID}"
    video = bot.index(url)
    assert bot.index(url) == video and len(calls) == 1  # second time comes from the store

    client = FakeClient()
    answer = bot.ask(video, "sourdough bread?", generator=generator(client))
    text = "".join(answer.tokens)
    assert "sourdough" in client.calls[0][1]["messages"][-1]["content"]
    linked, times = link_citations(text, VIDEO_ID, answer.context.segments)
    assert times == [0] and "t=0s" in linked

    assert bot.ask(video, "sourdough bread?").tokens is None  # excerpts-only mode
