import re
import zlib

import chromadb
import numpy as np
import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings

from video_chatbot.chunking import chunk_segments
from video_chatbot.models import Segment, VideoInfo
from video_chatbot.vector_store import VectorStore

VIDEO_ID = "abc_DEF-123"


class BagOfWords(EmbeddingFunction[Documents]):
    """Deterministic offline embedding: texts sharing words are close."""

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            v = np.full(512, 1e-6, dtype=np.float32)
            for word in re.findall(r"\w+", text.lower()):
                v[zlib.crc32(word.encode()) % 512] += 1
            vectors.append(v)
        return vectors

    @staticmethod
    def name() -> str:
        return "bag-of-words"

    @staticmethod
    def build_from_config(config):
        return BagOfWords()

    def get_config(self):
        return {}


TOPICS = ["sourdough bread", "bicycle gears", "volcano magma", "chess openings", "coral reefs"]


@pytest.fixture
def segments():
    """Five one-minute topics, one caption every 5 seconds."""
    rows = []
    for t, topic in enumerate(TOPICS):
        for line in range(12):
            start = t * 60 + line * 5 + 0.4
            rows.append(Segment(start, start + 5, f"Line {line} explains {topic} in detail."))
    return rows


@pytest.fixture
def info():
    return VideoInfo(VIDEO_ID, "Five topics", "Test channel", "en", 300.4)


@pytest.fixture
def store(tmp_path):
    client = chromadb.PersistentClient(
        path=str(tmp_path / "chroma"), settings=chromadb.Settings(anonymized_telemetry=False)
    )
    return VectorStore(embedding_function=BagOfWords(), client=client)


@pytest.fixture
def indexed(store, info, segments):
    store.add(info, chunk_segments(segments, 40, 60, 10))
    return store
