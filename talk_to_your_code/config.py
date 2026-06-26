from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    """Runtime settings.

    The important design choice is `db_path`: all ingested files, chunks,
    symbols, and embeddings are stored outside the target repository. This
    avoids creating `.index` folders inside user repos.
    """

    ollama_base_url: str = _env("TYC_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    chat_model: str = _env("TYC_CHAT_MODEL", "qwen2.5-coder:7b")
    planner_model: str = _env("TYC_PLANNER_MODEL", "qwen2.5-coder:7b")
    embedding_model: str = _env("TYC_EMBEDDING_MODEL", "nomic-embed-text")
    db_path: Path = Path(_env("TYC_DB_PATH", "~/.talk_to_your_code/talk_to_your_code.sqlite3")).expanduser()
    default_max_context_chars: int = int(_env("TYC_DEFAULT_CONTEXT_CHARS", "45000"))
    default_top_k: int = int(_env("TYC_DEFAULT_TOP_K", "10"))
    max_file_bytes: int = int(_env("TYC_MAX_FILE_BYTES", str(1_000_000)))
    embedding_batch_size: int = int(_env("TYC_EMBEDDING_BATCH_SIZE", "16"))
    use_model_for_token_count: bool = _env("TYC_USE_MODEL_FOR_TOKEN_COUNT", "false").lower() == "true"
    enable_graph_summary: bool = _env("TYC_ENABLE_GRAPH_SUMMARY", "true").lower() == "true"
    graph_summary_dir: Path = Path(
        _env("TYC_GRAPH_SUMMARY_DIR", "~/.talk_to_your_code/graph_summaries")
    ).expanduser()
    joern_server: str = _env("TYC_JOERN_SERVER", "localhost:8080")
    graph_summary_max_items: int = int(_env("TYC_GRAPH_SUMMARY_MAX_ITEMS", "3000"))


DEFAULT_SETTINGS = Settings()
