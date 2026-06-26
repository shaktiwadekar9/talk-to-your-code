from __future__ import annotations

import hashlib
from pathlib import Path

from .config import Settings


def repo_summary_output_dir(settings: Settings, repo_path: str | Path) -> Path:
    root = Path(repo_path).expanduser().resolve()
    # repo_hash = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    # return settings.graph_summary_dir / f"{root.name}-{repo_hash}"
    return settings.graph_summary_dir / f"{root.name}"


def generate_graph_repo_summary(settings: Settings, repo_path: str | Path) -> str:
    """
    Generate a graph-based repo summary using code-graph-ai-summarizer.

    This runs during ingestion, not during every chat query.
    """
    from code_graph_ai_summarizer import summarize_repository

    root = Path(repo_path).expanduser().resolve()
    out_dir = repo_summary_output_dir(settings, root)

    result = summarize_repository(
        repo_path=root,
        out_dir=out_dir.parent,
        joern_server=settings.joern_server,
        max_items=settings.graph_summary_max_items,
        write_outputs=True,
    )
    # print(f"Summary generated for repo {root} at {result.out_dir}: {result.summary_path}...")

    return result.summary


def load_graph_repo_summary(settings: Settings, repo_path: str | Path) -> str:
    root = Path(repo_path).expanduser().resolve()
    out_dir = repo_summary_output_dir(settings, root)
    summary_path = out_dir / "repo_summary.md"

    if not summary_path.exists():
        return ""

    return summary_path.read_text(encoding="utf-8")