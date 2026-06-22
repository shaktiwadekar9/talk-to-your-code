from __future__ import annotations

import argparse
import json
import subprocess
import sys

from .config import DEFAULT_SETTINGS
from .indexing import RepoIndexer
from .ollama_client import OllamaClient
from .orchestrator import CodeChatOrchestrator
from .storage import SQLiteStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Talk To Your Code")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("repos", help="List ingested repos")

    ingest = sub.add_parser("ingest", help="Ingest a local repo into SQLite")
    ingest.add_argument("repo_path")
    ingest.add_argument("--reingest", action="store_true")

    ask = sub.add_parser("ask", help="Ask a question about an ingested repo")
    ask.add_argument("repo_id", type=int)
    ask.add_argument("query")
    ask.add_argument("--max-context-chars", type=int, default=DEFAULT_SETTINGS.default_max_context_chars)
    ask.add_argument("--top-k", type=int, default=DEFAULT_SETTINGS.default_top_k)
    ask.add_argument("--json", action="store_true")

    sub.add_parser("api", help="Start FastAPI backend")
    sub.add_parser("ui", help="Start Streamlit UI")

    args = parser.parse_args()
    settings = DEFAULT_SETTINGS
    store = SQLiteStore(settings)
    ollama = OllamaClient(settings)

    if args.command == "repos":
        for repo in store.list_repos():
            print(f"{repo.id}: {repo.name} | {repo.root_path} | files={repo.file_count} chunks={repo.chunk_count}")
        return

    if args.command == "ingest":
        indexer = RepoIndexer(settings, store, ollama)
        repo_id, steps = indexer.ingest(args.repo_path, reingest=args.reingest)
        repo = store.get_repo(repo_id)
        print(f"Indexed repo_id={repo_id}: {repo.root_path if repo else args.repo_path}")
        for step in steps:
            print(f"- {step.name}: {step.detail}")
        return

    if args.command == "ask":
        orchestrator = CodeChatOrchestrator(settings, store, ollama)
        result = orchestrator.chat(args.repo_id, args.query, args.max_context_chars, args.top_k)
        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            print("\nIntermediate steps:")
            for step in result.intermediate_steps:
                print(f"- {step.name}: {step.detail}")
            print("\nQuery plan:")
            print(result.plan.model_dump_json(indent=2))
            print("\nAnswer:\n")
            print(result.answer.answer)
            if result.answer.evidence:
                print("\nEvidence:")
                for ev in result.answer.evidence:
                    print(f"- {ev.file_path} {ev.line_range}: {ev.reason}")
        return

    if args.command == "api":
        subprocess.run([sys.executable, "-m", "uvicorn", "talk_to_your_code.api.main:app", "--reload"], check=False)
        return

    if args.command == "ui":
        subprocess.run([sys.executable, "-m", "streamlit", "run", "talk_to_your_code/ui/streamlit_app.py"], check=False)
        return


if __name__ == "__main__":
    main()
