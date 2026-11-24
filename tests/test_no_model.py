"""Replies that need no language model."""

import pytest

from tests.conftest import VIDEO_ID
from video_chatbot.details import answer_from_details
from video_chatbot.extractive import quote_answer, quote_overview, to_sentences
from video_chatbot.models import Segment, VideoInfo
from video_chatbot.retriever import retrieve


@pytest.mark.parametrize(
    "question, expected",
    [
        ("whats the title of the video?", "Five topics"),
        ("What is the title?", "Five topics"),
        ("what's the video's title", "Five topics"),
        ("title", "Five topics"),
        ("Tell me the name of this video please", "Five topics"),
        ("What is this video called?", "Five topics"),
        ("what is the video title", "Five topics"),
        ("Who made this video?", "Test channel"),
        ("which channel is this from?", "Test channel"),
        ("What's the channel name?", "Test channel"),
        ("How long is the video?", "5:00"),
        ("video length?", "5:00"),
        ("can you give me the link to this video", "watch?v=" + VIDEO_ID),
    ],
)
def test_details_are_answered(info, question, expected):
    assert expected in answer_from_details(question, info)


@pytest.mark.parametrize(
    "question",
    [
        "What is the title of the book she mentions?",
        "What is the name of the activation function?",
        "Who made the first neural network?",
        "How long is the training process?",
        "How long does it take?",
        "Which channel does he recommend?",
        "What is the length of the vector?",
        "What is this video about?",
        "What is a neural network called in biology?",
        "What's the title and who is the guest?",
        "link between weights and biases?",
    ],
)
def test_content_questions_are_not(info, question):
    assert answer_from_details(question, info) is None


def test_missing_channel_is_not_invented():
    assert answer_from_details("who made this video?", VideoInfo("x" * 11, "T")) is None


def test_sentences_start_at_the_right_line():
    lines = [Segment(0, 2, "One starts"), Segment(2, 4, "here. Two starts"), Segment(4, 6, "now.")]
    assert [(s.start, s.text) for s in to_sentences(lines)] == [
        (0, "One starts here."),
        (2, "Two starts now."),
    ]


def test_unpunctuated_captions_are_still_split():
    lines = [Segment(i, i + 1, "word " * 8) for i in range(20)]
    assert all(len(s.text.split()) <= 35 for s in to_sentences(lines))


def test_long_punctuated_sentence_is_not_cut():
    long = " ".join(["word"] * 60) + "."
    lines = [Segment(0, 5, long)] + [Segment(i, i + 1, "A short one.") for i in range(5, 25)]
    assert to_sentences(lines)[0].text == long


def quotes(store, question, k=8):
    return quote_answer(store, VIDEO_ID, question, retrieve(store, VIDEO_ID, question, k=k))


def test_quotes_are_relevant_whole_and_in_order(indexed):
    found = quotes(indexed, "volcano magma", k=3)
    assert 1 <= len(found) <= 3
    assert all("volcano" in q.text and q.text.startswith("Line") for q in found)
    assert [q.start for q in found] == sorted(q.start for q in found)
    assert len({q.start for q in found}) == len(found)


def test_unrelated_question_gets_no_quotes(indexed):
    assert quotes(indexed, "quantum entanglement") == []


def test_asking_for_a_time_quotes_that_moment(indexed):
    found = quotes(indexed, "what happens at 4:10?")
    assert len(found) == 1 and int(found[0].start) == 250
    assert quotes(indexed, "what happens at 59:00?") == []


def test_overview_spans_the_video(indexed):
    found = quote_overview(indexed, VIDEO_ID, parts=5)
    assert found[0].start < 60 and found[-1].start > 230
    assert all(q.text.startswith("Line") for q in found)
