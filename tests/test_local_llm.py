"""The Ollama generator, without Ollama."""

from types import SimpleNamespace

import httpx
import ollama
import pytest

from tests.conftest import VIDEO_ID
from video_chatbot.config import LOCAL_PREFIX, Settings
from video_chatbot.llm import LLMError, Turn
from video_chatbot.local_llm import OllamaGenerator, list_local_models
from video_chatbot.pipeline import VideoChatbot


def part(text, reason=None):
    return SimpleNamespace(message=SimpleNamespace(content=text), done_reason=reason)


class FakeOllama:
    def __init__(self, parts=(), error=None, names=()):
        self.calls, self._parts, self._error = [], parts, error
        self._names = names

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return iter(self._parts) if kwargs["stream"] else self._parts[0]

    def list(self):
        if self._error:
            raise self._error
        return SimpleNamespace(models=[SimpleNamespace(model=n) for n in self._names])


def generator(client, **settings):
    return OllamaGenerator(Settings(model=LOCAL_PREFIX + "tiny:1b", **settings), client)


def test_streams_with_the_shared_prompt(info):
    client = FakeOllama([part("Bread "), part(""), part("[0:00]", "stop")])
    turns = [Turn("q0", "a0")]
    tokens = list(generator(client).stream_answer(info, "why?", "EXCERPTS", turns))
    assert tokens == ["Bread ", "[0:00]"]
    sent = client.calls[0]
    assert sent["model"] == "tiny:1b" and sent["think"] is False
    assert [m["role"] for m in sent["messages"]] == ["system", "user", "assistant", "user"]
    assert "EXCERPTS" in sent["messages"][-1]["content"]


def test_both_calls_share_a_context_size(info):
    """A different size would make Ollama reload the model between them."""
    client = FakeOllama([part("query", "stop")])
    g = generator(client)
    assert g.rewrite_query(info, "why?", [Turn("q", "a")]) == "query"
    list(g.stream_answer(info, "q", "c"))
    assert client.calls[0]["options"]["num_ctx"] == client.calls[1]["options"]["num_ctx"]


def test_length_limit_is_shown(info):
    text = "".join(generator(FakeOllama([part("x", "length")])).stream_answer(info, "q", "c"))
    assert "cut off" in text


def test_oversized_prompt_is_refused_not_truncated(info):
    client = FakeOllama([part("x", "stop")])
    with pytest.raises(LLMError, match="too much text"):
        list(generator(client).stream_answer(info, "q", "word " * 10000))
    assert client.calls == []


@pytest.mark.parametrize(
    "error, message",
    [
        (ConnectionError("no"), "Start the Ollama app"),
        (httpx.ReadTimeout("slow"), "too long"),
        (httpx.RemoteProtocolError("dropped"), "Couldn't reach"),
        (ollama.ResponseError("missing", 404), "ollama pull tiny:1b"),
        (ollama.ResponseError("out of memory", 500), "out of memory"),
    ],
)
def test_errors_are_explained(info, error, message):
    with pytest.raises(LLMError, match=message):
        list(generator(FakeOllama(error=error)).stream_answer(info, "q", "c"))


def test_lists_only_local_chat_models():
    names = ["qwen3.5:4b", "nomic-embed-text:latest", "big:120b-cloud", "a:1b"]
    assert list_local_models(FakeOllama(names=names)) == ["a:1b", "qwen3.5:4b"]
    assert list_local_models(FakeOllama(error=ConnectionError())) == []


def test_long_overviews_are_limited(indexed, monkeypatch):
    monkeypatch.setattr(OllamaGenerator, "overview_max_chunks", 3)
    client = FakeOllama([part("ok", "stop")])
    bot = VideoChatbot(Settings(), indexed)
    result = bot.ask(indexed.get(VIDEO_ID), "Summarize this video", (), generator(client))
    assert len(result.context.results) == 3
    list(result.tokens)
    assert "evenly spaced samples" in client.calls[0]["messages"][-1]["content"]
