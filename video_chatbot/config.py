"""Settings, read from environment variables (and an optional .env file)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODELS = {
    "claude-opus-5-5": "Claude Opus 5.5 (best answers)",
    "claude-sonnet-5-5": "Claude Sonnet 5.5 (balanced)",
    "claude-haiku-5-5": "Claude Haiku 5.5 (cheapest)",
}

LOCAL_PREFIX = "ollama/"  # marks a model that runs on this computer


def _env(name: str, default, cast=str):
    """Read a setting; an unset or blank variable means the default."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        raise ValueError(f"{name} must be a valid {cast.__name__}, got {raw!r}") from None


@dataclass(frozen=True)
class Settings:
    model: str = field(default_factory=lambda: _env("CLAUDE_MODEL", "claude-opus-5-5"))
    effort: str = field(default_factory=lambda: _env("CLAUDE_EFFORT", "low"))
    local_model: str = field(default_factory=lambda: _env("LOCAL_MODEL", "qwen3.5:4b"))
    top_k: int = field(default_factory=lambda: _env("TOP_K", 8, int))
    # Sizes chosen with evals/retrieval_eval.py. all-MiniLM-L6-v2 ignores everything
    # past 256 tokens (~190 words), so don't go above that.
    chunk_target_words: int = field(default_factory=lambda: _env("CHUNK_TARGET_WORDS", 50, int))
    chunk_max_words: int = field(default_factory=lambda: _env("CHUNK_MAX_WORDS", 75, int))
    chunk_overlap_words: int = field(default_factory=lambda: _env("CHUNK_OVERLAP_WORDS", 15, int))
    chroma_dir: Path = field(default_factory=lambda: ROOT / _env("CHROMA_DIR", "data/chroma"))
    overview_max_chunks: int = 120
    history_turns: int = 6
