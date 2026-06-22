from __future__ import annotations

from fastapi import FastAPI, HTTPException

from talk_to_your_code.config import DEFAULT_SETTINGS
from talk_to_your_code.indexing import RepoIndexer
from talk_to_your_code.ollama_client import OllamaClient
from talk_to_your_code.orchestrator import CodeChatOrchestrator
from talk_to_your_code.schemas import ChatRequest, ChatResult, IngestRequest, IngestResponse, RepoRecord
from talk_to_your_code.storage import SQLiteStore

settings = DEFAULT_SETTINGS
store = SQLiteStore(settings)
ollama = OllamaClient(settings)

app = FastAPI(title="Talk To Your Code", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "db_path": str(store.db_path)}


@app.get("/repos", response_model=list[RepoRecord])
def list_repos() -> list[RepoRecord]:
    return store.list_repos()


@app.post("/repos/ingest", response_model=IngestResponse)
def ingest_repo(request: IngestRequest) -> IngestResponse:
    try:
        indexer = RepoIndexer(settings, store, ollama)
        repo_id, steps = indexer.ingest(request.repo_path, reingest=request.reingest)
        repo = store.get_repo(repo_id)
        if repo is None:
            raise RuntimeError("Repo disappeared after ingestion")
        return IngestResponse(
            repo=repo,
            indexed_files=repo.file_count,
            indexed_chunks=repo.chunk_count,
            indexed_symbols=repo.symbol_count,
            steps=steps,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResult)
def chat(request: ChatRequest) -> ChatResult:
    try:
        orchestrator = CodeChatOrchestrator(settings, store, ollama)
        return orchestrator.chat(
            repo_id=request.repo_id,
            query=request.query,
            max_context_chars=request.max_context_chars,
            top_k=request.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
