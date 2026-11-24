"""Streamlit interface. Run with: streamlit run app.py"""

import re
from dataclasses import dataclass, field, replace

import streamlit as st

from video_chatbot import local_llm
from video_chatbot.config import LOCAL_PREFIX, MODELS, Settings
from video_chatbot.extractive import Quote
from video_chatbot.llm import (
    AnswerGenerator,
    ClaudeGenerator,
    LLMError,
    MissingCredentials,
    Turn,
    has_credentials,
)
from video_chatbot.models import RetrievedChunk, VideoInfo
from video_chatbot.pipeline import VideoChatbot
from video_chatbot.retriever import Context
from video_chatbot.timestamps import format_time, link_citations
from video_chatbot.vector_store import VectorStore
from video_chatbot.youtube import VideoError

SUGGESTIONS = [
    "Summarize this video",
    "What are the key takeaways?",
    "What examples are used?",
]


@dataclass
class Message:
    question: str
    context: Context
    search_query: str
    answer: str = ""  # empty when there is no model to write one
    quotes: list[Quote] = field(default_factory=list)
    error: str = ""


@st.cache_resource(show_spinner="Loading the embedding model…")
def get_store() -> VectorStore:
    return VectorStore(Settings().chroma_dir)


@st.cache_data(ttl=15, show_spinner=False)
def local_models() -> list[str]:
    return [LOCAL_PREFIX + name for name in local_llm.list_local_models()]


@st.cache_resource(ttl=25 * 60, show_spinner=False)
def warm_up(model: str) -> bool:
    """Once per model, again after Ollama may have unloaded it."""
    local_llm.warm_up(model.removeprefix(LOCAL_PREFIX))
    return True


def model_label(model: str) -> str:
    if model.startswith(LOCAL_PREFIX):
        return f"{model.removeprefix(LOCAL_PREFIX)} (free, on this computer)"
    return MODELS.get(model, model)


def make_generator(settings: Settings, api_key: str | None) -> AnswerGenerator | None:
    """The chosen model, or None to quote the video instead."""
    if settings.model.startswith(LOCAL_PREFIX):
        return local_llm.OllamaGenerator(settings)
    try:
        return ClaudeGenerator(settings, api_key)
    except MissingCredentials:
        return None


def plain(text: str) -> str:
    """Show text literally: captions and titles are not Markdown."""
    return re.sub(r"([\\`*_{}\[\]<>#+|~$:])", r"\\\1", text)


def no_math(markdown: str) -> str:
    """Stop Streamlit from reading "$5 to $10" as LaTeX."""
    return markdown.replace("$", r"\$")


# --- state -----------------------------------------------------------------


def open_video(video_id: str | None) -> None:
    if video_id:
        st.query_params["v"] = video_id  # survives a page refresh
    else:
        st.query_params.pop("v", None)
    st.session_state.seek = (0, 0, False)


def seek_to(seconds: int) -> None:
    _, clicks, _ = st.session_state.get("seek", (0, 0, False))
    st.session_state.seek = (seconds, clicks + 1, True)


def queue_question(question: str) -> None:
    st.session_state.queued = question


def history(video_id: str) -> list[Message]:
    return st.session_state.setdefault("chats", {}).setdefault(video_id, [])


# --- sidebar ---------------------------------------------------------------


def sidebar(store: VectorStore) -> tuple[Settings, str | None]:
    defaults = Settings()
    with st.sidebar:
        st.title("🎬 YouTube Video Chatbot")
        with st.form("add", clear_on_submit=True, border=False):
            url = st.text_input("YouTube link", placeholder="https://www.youtube.com/watch?v=…")
            submitted = st.form_submit_button("Add video", type="primary", width="stretch")
        if submitted:
            add_video(VideoChatbot(defaults, store), url)

        videos = store.list_videos()
        if videos:
            st.subheader("Your videos")
        current = st.query_params.get("v")
        for video in videos:
            name, remove = st.columns([5, 1], vertical_alignment="center")
            name.button(
                plain(video.title),
                key=f"open-{video.video_id}",
                on_click=open_video,
                args=(video.video_id,),
                type="primary" if video.video_id == current else "secondary",
                width="stretch",
            )
            if remove.button("✕", key=f"remove-{video.video_id}", help="Remove this video"):
                store.delete(video.video_id)
                st.session_state.get("chats", {}).pop(video.video_id, None)
                if video.video_id == current:
                    open_video(None)
                st.rerun()

        st.divider()
        local = local_models()
        options = list(MODELS)
        if defaults.model not in options:
            options.insert(0, defaults.model)
        options = local + options
        # Prefer Claude when there is a key, otherwise whatever runs here for free.
        preferred = defaults.model
        if local and not has_credentials():
            wanted = LOCAL_PREFIX + defaults.local_model
            preferred = wanted if wanted in local else local[0]
        model = st.selectbox(
            "Model", options, index=options.index(preferred), format_func=model_label
        )
        is_local = model.startswith(LOCAL_PREFIX)
        if is_local:
            warm_up(model)
        api_key = None
        if not is_local and not has_credentials():
            api_key = st.text_input(
                "Anthropic API key",
                type="password",
                help="Kept only for this browser session. Or set ANTHROPIC_API_KEY in .env.",
            ).strip() or None
        top_k = st.slider("Passages per question", 2, 15, min(15, max(2, defaults.top_k)))
        if not is_local and not has_credentials(api_key):
            hint = "choose a local model" if local else "start Ollama for a free local model"
            st.info(
                "No API key, so replies are quotes from the video. "
                f"For written answers, add a key or {hint}."
            )
    return replace(defaults, model=model, top_k=top_k), api_key


def add_video(bot: VideoChatbot, url: str) -> None:
    try:
        with st.spinner("Fetching captions and building the index…"):
            info = bot.index(url)
    except VideoError as exc:
        st.error(str(exc))
    else:
        open_video(info.video_id)
        st.rerun()


# --- chat ------------------------------------------------------------------


def show_passages(results: list[RetrievedChunk]) -> None:
    for result in results:
        chunk = result.chunk
        score = "" if result.score is None else f" · similarity {result.score:.2f}"
        st.markdown(f"**{format_time(chunk.start)}–{format_time(chunk.end)}**{score}")
        st.caption(plain(chunk.text))


def seek_buttons(times: list[int], key: str) -> None:
    with st.container(horizontal=True, gap="small"):
        for seconds in times:
            st.button(
                f"▶ {format_time(seconds)}",
                key=f"{key}-{seconds}",
                on_click=seek_to,
                args=(seconds,),
                help="Play from here",
            )


def show_reply(message: Message, video: VideoInfo, key: str) -> None:
    """A finished reply, with its sources."""
    if message.context.mode == "details":
        st.markdown(plain(message.answer))
        return
    if message.answer:
        text, times = link_citations(message.answer, video.video_id, message.context.segments)
        st.markdown(no_math(text))
        seek_buttons(times, key)
    if message.error:
        st.error(message.error)
    if not message.answer:
        if not message.quotes:
            st.markdown("I couldn't find anything about that in this video.")
            return
        overview = message.context.mode == "overview"
        st.markdown(
            "Moments from across the video (a written summary needs a working model):"
            if overview
            else "From the video:"
        )
        for quote in message.quotes:
            st.markdown(f"**{format_time(quote.start)}**  {plain(quote.text)}")
        seek_buttons(list(dict.fromkeys(int(q.start) for q in message.quotes)), key)
        if overview:
            return

    context = message.context
    if context.mode == "overview":
        coverage = "the whole transcript" if context.complete else "an even sample of the video"
        label = f"Based on {coverage} ({len(context.results)} passages)"
    else:
        label = f"Retrieved passages ({len(context.results)})"
    with st.expander(label):
        if message.search_query != message.question:
            st.caption(f"Searched for: {plain(message.search_query)}")
        show_passages(context.results)


def answer(
    bot: VideoChatbot, video: VideoInfo, question: str, messages: list[Message], api_key: str | None
) -> Message:
    """Run the pipeline, streaming the reply into the current container."""
    turns = [Turn(m.question, m.answer) for m in messages if m.answer and not m.error]
    error = ""
    generator = make_generator(bot.settings, api_key)
    try:
        with st.spinner("Searching the video…"):
            result = bot.ask(video, question, turns, generator)
    except LLMError as exc:
        error = str(exc)
        result = bot.ask(video, question)

    message = Message(
        question, result.context, result.search_query, result.direct, result.quotes, error
    )
    if result.tokens:
        segments = result.context.segments
        placeholder = st.empty()
        placeholder.markdown("*Thinking…*")
        try:
            for token in result.tokens:
                message.answer += token
                text, _ = link_citations(message.answer, video.video_id, segments, streaming=True)
                placeholder.markdown(no_math(text) + " ▌")
        except LLMError as exc:
            message.error = str(exc)
        placeholder.empty()
        if not message.answer:
            message.quotes = bot.ask(video, question).quotes
    return message


def chat(bot: VideoChatbot, video: VideoInfo, api_key: str | None) -> None:
    messages = history(video.video_id)
    log = st.container(height=560)
    typed = st.chat_input("Ask about this video", key=f"input-{video.video_id}")
    question = typed or st.session_state.pop("queued", None)

    with log:
        for i, message in enumerate(messages):
            st.chat_message("user").markdown(plain(message.question))
            with st.chat_message("assistant"):
                show_reply(message, video, f"{video.video_id}-{i}")
        if question:
            st.chat_message("user").markdown(plain(question))
            with st.chat_message("assistant"):
                messages.append(answer(bot, video, question, messages, api_key))
            st.rerun()
        if not messages:
            st.markdown("Ask anything about this video, or start with one of these:")
            for suggestion in SUGGESTIONS:
                st.button(suggestion, on_click=queue_question, args=(suggestion,))


def player(video: VideoInfo) -> None:
    seconds, clicks, autoplay = st.session_state.get("seek", (0, 0, False))
    # The embed reloads only when its URL changes. Moving the end point, which is
    # past the end of the video anyway, lets the same timestamp be clicked twice.
    end = int(video.duration) + 3600 + clicks if clicks else None
    st.video(video.url, start_time=seconds, end_time=end, autoplay=autoplay)
    st.markdown(f"**{plain(video.title)}**")
    details = [plain(video.channel)] if video.channel else []
    details += [
        f"{format_time(video.duration)} of captions",
        f"{video.chunk_count} chunks",
        f"language: {plain(video.language)}",
    ]
    st.caption(" · ".join(details))
    if not video.language.startswith("en"):
        st.warning(
            "The embedding model is trained on English, so search will be less "
            "accurate for this video."
        )


def main() -> None:
    st.set_page_config(page_title="YouTube Video Chatbot", page_icon="🎬", layout="wide")
    store = get_store()
    settings, api_key = sidebar(store)

    video_id = st.query_params.get("v")
    video = store.get(video_id) if video_id else None
    if video is None:
        st.header("Chat with any YouTube video")
        st.write(
            "Add a link in the sidebar. The video needs captions (automatic ones work). "
            "Then ask questions and jump to the moment each answer comes from."
        )
        return

    left, right = st.columns([2, 3], gap="large")
    with left:
        player(video)
    with right:
        chat(VideoChatbot(settings, store), video, api_key)


main()
