"""Answers from a model running on this computer, through Ollama."""

import threading
from collections.abc import Iterator

import httpx
import ollama

from .config import LOCAL_PREFIX, Settings
from .llm import CUT_OFF, AnswerGenerator, LLMError

# Ollama drops the start of a prompt that doesn't fit, without saying so. These
# limits keep the longest prompt well inside the context window.
_CONTEXT_TOKENS = 12288
_ANSWER_TOKENS = 1500
_MAX_PROMPT_CHARS = 32000
_KEEP_LOADED = "30m"
_ERRORS = (ConnectionError, ollama.ResponseError, httpx.HTTPError)


def list_local_models(client=None) -> list[str]:
    """Chat models installed in Ollama, or [] when it isn't running."""
    try:
        found = (client or ollama.Client(timeout=3)).list().models
    except _ERRORS:
        return []
    # "-cloud" models run on Ollama's servers, so they aren't local.
    return sorted(
        m.model for m in found if m.model and "embed" not in m.model and "cloud" not in m.model
    )


def warm_up(model: str) -> None:
    """Load a model ahead of the first question, without waiting for it."""

    def load() -> None:
        try:
            ollama.Client(timeout=300).chat(
                model=model, messages=[], keep_alive=_KEEP_LOADED, options={"num_ctx": _CONTEXT_TOKENS}
            )
        except _ERRORS:
            pass  # asking a question will report the problem

    threading.Thread(target=load, daemon=True).start()


def _translate_error(exc: Exception, model: str) -> LLMError:
    if isinstance(exc, ollama.ResponseError) and exc.status_code == 404:
        return LLMError(f"{model} isn't installed. Run: ollama pull {model}")
    if isinstance(exc, ollama.ResponseError):
        return LLMError(f"Ollama reported a problem: {exc.error}")
    if isinstance(exc, httpx.TimeoutException):
        return LLMError("The local model took too long to reply.")
    return LLMError("Couldn't reach Ollama. Start the Ollama app and ask again.")


class OllamaGenerator(AnswerGenerator):
    overview_max_chunks = 60

    def __init__(self, settings: Settings, client=None):
        super().__init__(settings)
        self._model = settings.model.removeprefix(LOCAL_PREFIX)
        self._client = client or ollama.Client(timeout=300)

    def _chat(self, system: str, messages: list[dict], max_tokens: int, stream: bool):
        size = len(system) + sum(len(m["content"]) for m in messages)
        if size > _MAX_PROMPT_CHARS:
            raise LLMError(
                "That's too much text for the local model. Lower \"Passages per question\" "
                "or remove and re-add the video to clear the chat."
            )
        return self._client.chat(
            model=self._model,
            messages=[{"role": "system", "content": system}, *messages],
            stream=stream,
            think=False,  # answers are lookups, and thinking makes them much slower
            keep_alive=_KEEP_LOADED,
            # Changing num_ctx between calls would reload the model.
            options={"num_ctx": _CONTEXT_TOKENS, "num_predict": max_tokens, "temperature": 0.2},
        )

    def _complete(self, system: str, prompt: str) -> str:
        try:
            reply = self._chat(system, [{"role": "user", "content": prompt}], 100, stream=False)
        except _ERRORS as exc:
            raise _translate_error(exc, self._model) from exc
        return reply.message.content or ""

    def _stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        done_reason = None
        try:
            for part in self._chat(system, messages, _ANSWER_TOKENS, stream=True):
                if part.message.content:
                    yield part.message.content
                done_reason = part.done_reason or done_reason
        except _ERRORS as exc:
            raise _translate_error(exc, self._model) from exc
        if done_reason == "length":
            yield CUT_OFF
