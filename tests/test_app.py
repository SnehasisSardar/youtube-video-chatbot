"""The Streamlit interface, driven without a browser."""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.conftest import VIDEO_ID
from video_chatbot import llm


@pytest.fixture
def app(indexed, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr("video_chatbot.vector_store.VectorStore", lambda *a, **k: indexed)
    monkeypatch.setattr("video_chatbot.local_llm.list_local_models", lambda: [])
    monkeypatch.setattr("video_chatbot.local_llm.warm_up", lambda model: None)
    st.cache_resource.clear()
    st.cache_data.clear()
    yield AppTest.from_file("../app.py", default_timeout=30)
    st.cache_resource.clear()


def open_video(app):
    app.query_params["v"] = VIDEO_ID
    return app.run()


def text_of(app):
    return "\n".join(m.value for m in app.markdown)


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_start_page_lists_videos(app):
    app.run()
    assert not app.exception
    assert app.header[0].value == "Chat with any YouTube video"
    button(app, "Five topics").click().run()
    assert app.query_params["v"] == VIDEO_ID
    assert not app.exception


def test_bad_link_shows_an_error(app):
    app.run()
    app.text_input[0].input("https://example.com")
    button(app, "Add video").click().run()
    assert "doesn't look like" in app.error[0].value


def test_excerpts_only_without_a_key(app):
    open_video(app)
    app.chat_input[0].set_value("volcano magma").run()
    assert not app.exception
    assert "From the video:" in text_of(app)
    assert "volcano magma in detail" in text_of(app)
    assert any(b.label.startswith("▶ 2:") or b.label.startswith("▶ 1:5") for b in app.button)


def test_answer_with_citations_and_seek(app, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    def fake_stream(self, info, question, context, turns=(), partial_coverage=False):
        yield "It costs $5 "
        yield "or $6 [2:0"
        yield "0]. Invented [59:59]."

    monkeypatch.setattr(llm.AnswerGenerator, "stream_answer", fake_stream)
    open_video(app)
    app.chat_input[0].set_value("volcano magma").run()
    assert not app.exception
    body = text_of(app)
    assert r"It costs \$5 or \$6 [\[2:00\]](" in body
    assert "59:59" not in body
    button(app, "▶ 2:00").click().run()
    assert not app.exception
    assert app.session_state.seek == (120, 1, True)


def test_local_model_is_the_default_without_a_key(app, monkeypatch):
    from video_chatbot import local_llm

    monkeypatch.setattr(local_llm, "list_local_models", lambda: ["a:1b", "qwen3.5:4b"])

    def fake_stream(self, info, question, context, turns=(), partial_coverage=False):
        yield "From the local model [2:00]."

    monkeypatch.setattr(local_llm.OllamaGenerator, "stream_answer", fake_stream)
    open_video(app)
    assert app.selectbox[0].value == "ollama/qwen3.5:4b"
    assert not app.info
    app.chat_input[0].set_value("volcano magma").run()
    assert not app.exception
    assert "From the local model [" in text_of(app)


def test_suggestion_button_asks(app):
    open_video(app)
    button(app, "Summarize this video").click().run()
    assert not app.exception
    assert len(app.session_state.chats[VIDEO_ID]) == 1


def test_api_failure_falls_back_to_passages(app, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    def fail(self, *args, **kwargs):
        raise llm.MissingCredentials("The Anthropic API key was rejected.")
        yield

    monkeypatch.setattr(llm.AnswerGenerator, "stream_answer", fail)
    open_video(app)
    app.chat_input[0].set_value("volcano magma").run()
    assert not app.exception
    assert "rejected" in app.error[0].value
    assert "From the video:" in text_of(app)


@pytest.mark.parametrize("key", ["", "test"])
def test_title_question(app, monkeypatch, key):
    if key:
        monkeypatch.setenv("ANTHROPIC_API_KEY", key)  # would fail if the API were called
    open_video(app)
    app.chat_input[0].set_value("whats the title of the video?").run()
    assert not app.exception and not app.error
    assert "The video is titled “Five topics”." in text_of(app)
    assert "From the video:" not in text_of(app)


def test_unrelated_question_is_refused(app):
    open_video(app)
    app.chat_input[0].set_value("quantum entanglement?").run()
    assert not app.exception
    assert "couldn't find anything" in text_of(app)
    assert not any(b.label.startswith("▶") for b in app.button)


def test_remove_video(app, indexed):
    open_video(app)
    button(app, "✕").click().run()
    assert not app.exception
    assert indexed.list_videos() == []
    assert "v" not in app.query_params


def test_unknown_video_in_url(app):
    app.query_params["v"] = "not-a-video!"
    app.run()
    assert not app.exception
    assert app.header[0].value == "Chat with any YouTube video"
