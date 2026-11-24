"""Storing chunks and choosing context for a question."""

import pytest

from tests.conftest import VIDEO_ID
from video_chatbot.chunking import chunk_segments
from video_chatbot.models import Chunk
from video_chatbot.retriever import is_overview_question, retrieve, sample_evenly
from video_chatbot.vector_store import VectorStore


def test_store_round_trip(indexed, info, segments):
    stored = indexed.get(VIDEO_ID)
    assert (stored.title, stored.channel) == (info.title, info.channel)
    chunks = indexed.all_chunks(VIDEO_ID)
    assert chunks == chunk_segments(segments, 40, 60, 10)
    assert stored.chunk_count == len(chunks)
    assert indexed.list_videos() == [stored]


@pytest.mark.parametrize("video_id", ["-abcdefghi-", "_abcdefghi_", "AAAAAAAAAAA", "aaaaaaaaaaa"])
def test_any_video_id_is_a_valid_collection(store, info, segments, video_id):
    from dataclasses import replace

    store.add(replace(info, video_id=video_id), chunk_segments(segments))
    assert store.get(video_id).video_id == video_id


def test_videos_are_isolated(indexed, info, segments):
    from dataclasses import replace

    indexed.add(replace(info, video_id="other_video", title="Other"), chunk_segments(segments[:12]))
    results = indexed.search("other_video", "volcano magma", 50)
    assert all("sourdough" in r.chunk.text for r in results)
    indexed.delete("other_video")
    assert indexed.get("other_video") is None
    assert indexed.get(VIDEO_ID) is not None


def test_missing_video(store):
    assert store.get("missing_vid") is None
    assert store.search("missing_vid", "x", 3) == []
    assert store.all_chunks("missing_vid") == []
    store.delete("missing_vid")


def test_reindexing_replaces(indexed, info, segments):
    indexed.add(info, chunk_segments(segments[:12], 40, 60, 10))
    assert all("sourdough" in c.text for c in indexed.all_chunks(VIDEO_ID))


def test_failed_indexing_leaves_nothing(store, info, segments):
    def fail(fraction):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        store.add(info, chunk_segments(segments), fail)
    assert store.get(VIDEO_ID) is None
    assert store.list_videos() == []


def test_search_finds_the_right_moment(indexed):
    best = indexed.search(VIDEO_ID, "how does volcano magma work", 3)[0]
    assert "volcano" in best.chunk.text
    assert 115 < best.chunk.start < 180
    assert 0 < best.score <= 1.0001


@pytest.mark.parametrize(
    "question",
    [
        "Summarize this video",
        "Can you summarise it?",
        "give me a summary",
        "What is this video about?",
        "What's it about?",
        "what are the key takeaways",
        "main points of the video?",
        "TLDR",
        "Give me an overview of the whole talk",
    ],
)
def test_overview_questions(question):
    assert is_overview_question(question)


@pytest.mark.parametrize(
    "question",
    [
        "What is a neural network?",
        "Summarize what she says about pricing",
        "key points on backpropagation",
        "Give me an overview of gradient descent",
        "What is the weather about to do?",
        "Why does that happen?",
    ],
)
def test_specific_questions(question):
    assert not is_overview_question(question)


def test_overview_covers_the_whole_video(indexed):
    context = retrieve(indexed, VIDEO_ID, "Summarize this video")
    assert context.mode == "overview" and context.complete
    for topic in ("sourdough", "bicycle", "volcano", "chess", "coral"):
        assert topic in context.render()


def test_long_overview_is_sampled(indexed):
    context = retrieve(indexed, VIDEO_ID, "Summarize this video", overview_max_chunks=3)
    total = len(indexed.all_chunks(VIDEO_ID))
    assert not context.complete
    assert [r.chunk.index for r in context.results] == [0, round((total - 1) / 2), total - 1]


def test_sample_evenly():
    assert sample_evenly([1, 2], 5) == [1, 2]
    assert sample_evenly(list(range(10)), 1) == [0]
    picked = sample_evenly(list(range(100)), 60)
    assert len(set(picked)) == 60 and picked[0] == 0 and picked[-1] == 99


def test_time_in_question_adds_that_moment(indexed):
    context = retrieve(indexed, VIDEO_ID, "what is said about sourdough at 4:10?", k=1)
    assert any(r.chunk.start <= 250 <= r.chunk.end for r in context.results)
    assert "coral" in context.render()


def test_search_uses_the_rewritten_query(indexed):
    context = retrieve(indexed, VIDEO_ID, "why?", "chess openings", k=1)
    assert "chess" in context.render()


def test_render_lists_each_line_once_in_order(indexed):
    context = retrieve(indexed, VIDEO_ID, "Summarize this video")
    text = context.render()
    assert text.count("[2:00]") == 1
    assert text.count("<excerpt>") == 1  # adjacent chunks join up
    assert text.index("[0:00]") < text.index("[2:00]") < text.index("[4:55]")


def test_render_separates_distant_chunks(segments):
    from video_chatbot.models import RetrievedChunk
    from video_chatbot.retriever import Context

    parts = [RetrievedChunk(Chunk(0, (segments[0],))), RetrievedChunk(Chunk(5, (segments[40],)))]
    assert Context(parts, "search", False).render().count("<excerpt>") == 2


def test_persists_to_disk(tmp_path, info, segments):
    from tests.conftest import BagOfWords

    path = tmp_path / "db"
    VectorStore(path, BagOfWords()).add(info, chunk_segments(segments))
    assert VectorStore(path, BagOfWords()).get(VIDEO_ID).title == info.title
