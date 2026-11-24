"""ChromaDB storage: one persistent collection per video."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import chromadb
import numpy as np
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

from .models import Chunk, RetrievedChunk, Segment, VideoInfo

_BATCH = 64
_PREFIX, _SUFFIX = "yt-", "-v1"  # names must start and end alphanumeric; video IDs may not


def _collection_name(video_id: str) -> str:
    return f"{_PREFIX}{video_id}{_SUFFIX}"


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-12)


def _to_chunk(metadata: dict) -> Chunk:
    segments = tuple(Segment(*row) for row in json.loads(metadata["segments"]))
    return Chunk(metadata["index"], segments)


class VectorStore:
    def __init__(self, path: Path | None = None, embedding_function=None, client=None):
        if client is None:
            path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(path), settings=chromadb.Settings(anonymized_telemetry=False)
            )
        self._client = client
        # Runs locally on CPU; the 80 MB model is downloaded on first use.
        self._embed = embedding_function or ONNXMiniLM_L6_V2()

    def _collection(self, video_id: str):
        try:
            return self._client.get_collection(
                _collection_name(video_id), embedding_function=self._embed
            )
        except NotFoundError:
            return None

    def add(
        self,
        info: VideoInfo,
        chunks: Sequence[Chunk],
        on_progress: Callable[[float], None] | None = None,
    ) -> VideoInfo:
        """Embed and store a video, replacing any earlier index of it."""
        if not chunks:
            raise ValueError("no chunks to index")
        self.delete(info.video_id)
        metadata = {
            "video_id": info.video_id,
            "title": info.title,
            "channel": info.channel,
            "language": info.language,
            "duration": info.duration,
            "chunk_count": len(chunks),
        }
        collection = self._client.create_collection(
            _collection_name(info.video_id),
            metadata=metadata,
            embedding_function=self._embed,
            configuration={"hnsw": {"space": "cosine"}},
        )
        try:
            for i in range(0, len(chunks), _BATCH):
                batch = chunks[i : i + _BATCH]
                collection.add(
                    ids=[f"{c.index:06d}" for c in batch],
                    documents=[c.text for c in batch],
                    metadatas=[
                        {
                            "index": c.index,
                            "start": c.start,
                            "end": c.end,
                            "segments": json.dumps([[s.start, s.end, s.text] for s in c.segments]),
                        }
                        for c in batch
                    ],
                )
                if on_progress:
                    on_progress((i + len(batch)) / len(chunks))
        except BaseException:
            self.delete(info.video_id)  # never leave a half-indexed video behind
            raise
        return VideoInfo(**metadata)

    def get(self, video_id: str) -> VideoInfo | None:
        """The stored video, or None if it isn't (completely) indexed."""
        collection = self._collection(video_id)
        if collection is None or collection.count() != collection.metadata["chunk_count"]:
            return None
        return VideoInfo(**collection.metadata)

    def list_videos(self) -> list[VideoInfo]:
        names = [c.name for c in self._client.list_collections()]
        ids = [
            n[len(_PREFIX) : -len(_SUFFIX)]
            for n in names
            if n.startswith(_PREFIX) and n.endswith(_SUFFIX)
        ]
        videos = [v for v in map(self.get, ids) if v]
        return sorted(videos, key=lambda v: v.title.lower())

    def delete(self, video_id: str) -> None:
        try:
            self._client.delete_collection(_collection_name(video_id))
        except NotFoundError:
            pass

    def search(self, video_id: str, query: str, k: int) -> list[RetrievedChunk]:
        """The k chunks closest in meaning to the query, best first."""
        collection = self._collection(video_id)
        if collection is None:
            return []
        result = collection.query(
            query_texts=[query],
            n_results=min(k, collection.count()),
            include=["metadatas", "distances"],
        )
        return [
            RetrievedChunk(_to_chunk(m), 1 - d)
            for m, d in zip(result["metadatas"][0], result["distances"][0])
        ]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Unit-length vectors, so a dot product is the cosine similarity."""
        return _unit(np.array(self._embed(list(texts)), dtype=np.float32))

    def chunk_vectors(self, video_id: str) -> tuple[list[Chunk], np.ndarray]:
        """Every chunk with its stored embedding, in video order."""
        collection = self._collection(video_id)
        if collection is None:
            return [], np.empty((0, 0))
        rows = collection.get(include=["metadatas", "embeddings"])
        order = sorted(range(len(rows["ids"])), key=lambda i: rows["metadatas"][i]["index"])
        chunks = [_to_chunk(rows["metadatas"][i]) for i in order]
        return chunks, _unit(np.array(rows["embeddings"], dtype=np.float32)[order])

    def all_chunks(self, video_id: str) -> list[Chunk]:
        collection = self._collection(video_id)
        if collection is None:
            return []
        rows = collection.get(include=["metadatas"])["metadatas"]
        return sorted((_to_chunk(m) for m in rows), key=lambda c: c.index)
