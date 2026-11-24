"""The RAG pipeline: index a video once, then answer questions about it."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

from .chunking import chunk_segments
from .config import Settings
from .details import answer_from_details
from .extractive import Quote, quote_answer, quote_overview
from .llm import AnswerGenerator, Turn
from .models import VideoInfo
from .retriever import Context, retrieve
from .vector_store import VectorStore
from .youtube import extract_video_id, load_video


@dataclass
class Answer:
    context: Context
    search_query: str
    tokens: Iterator[str] | None = None  # the written answer, if there is a model
    direct: str = ""  # the whole answer, when no search was needed
    quotes: list[Quote] = field(default_factory=list)  # used when there is no model


class VideoChatbot:
    def __init__(self, settings: Settings, store: VectorStore):
        self.settings = settings
        self.store = store

    def index(
        self, url: str, on_progress: Callable[[float], None] | None = None
    ) -> VideoInfo:
        """Index a video, reusing the stored embeddings if they exist."""
        existing = self.store.get(extract_video_id(url))
        if existing:
            return existing
        info, segments = load_video(url)
        chunks = chunk_segments(
            segments,
            self.settings.chunk_target_words,
            self.settings.chunk_max_words,
            self.settings.chunk_overlap_words,
        )
        return self.store.add(info, chunks, on_progress)

    def ask(
        self,
        info: VideoInfo,
        question: str,
        turns: Sequence[Turn] = (),
        generator: AnswerGenerator | None = None,
    ) -> Answer:
        """Answer a question: directly, with the model, or by quoting the video."""
        direct = answer_from_details(question, info)
        if direct:
            return Answer(Context([], "details", False), question, direct=direct)

        query = generator.rewrite_query(info, question, turns) if generator else question
        overview_limit = self.settings.overview_max_chunks
        if generator and generator.overview_max_chunks:
            overview_limit = min(overview_limit, generator.overview_max_chunks)
        context = retrieve(
            self.store, info.video_id, question, query, self.settings.top_k, overview_limit
        )
        if generator is None:
            if context.mode == "overview":
                quotes = quote_overview(self.store, info.video_id)
            else:
                quotes = quote_answer(self.store, info.video_id, question, context)
            return Answer(context, query, quotes=quotes)
        tokens = generator.stream_answer(
            info,
            question,
            context.render(),
            turns,
            partial_coverage=context.mode == "overview" and not context.complete,
        )
        return Answer(context, query, tokens)
