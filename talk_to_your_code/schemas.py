from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .languages import SupportedLanguage


QueryType = Literal[
    "repo_overview",
    "lookup",
    "explanation",
    "debugging",
    "change_guidance",
    "test_question",
    "unknown",
]
RetrievalMode = Literal["hybrid", "keyword_first", "symbol_first", "vector_first", "repo_map_first"]


class CodeFile(BaseModel):
    path: str
    language: SupportedLanguage
    content: str
    sha256: str
    size_bytes: int
    modified_time: float


class Symbol(BaseModel):
    name: str
    kind: str
    file_path: str
    start_line: int
    end_line: int
    language: SupportedLanguage


class CodeChunk(BaseModel):
    """Smallest searchable unit stored in SQLite.

    A chunk is usually a function/class/method. If a file has no symbols, the
    chunker falls back to line windows. The LLM should only answer using chunks
    that are retrieved and placed into context.
    """

    chunk_id: str
    file_path: str
    language: SupportedLanguage
    start_line: int
    end_line: int
    text: str
    symbol_name: str | None = None
    symbol_kind: str | None = None

    def searchable_text(self) -> str:
        parts = [f"file: {self.file_path}", f"language: {self.language}"]
        if self.symbol_name:
            parts.append(f"symbol: {self.symbol_name}")
        if self.symbol_kind:
            parts.append(f"kind: {self.symbol_kind}")
        parts.append(self.text)
        return "\n".join(parts)


class QueryPlan(BaseModel):
    """Structured LLM output used before retrieval.

    Ollama is asked to generate this using JSON-schema structured output. The
    retriever then follows this plan instead of guessing with hardcoded if/else
    rules.
    """

    query_type: QueryType = Field(default="unknown")
    intent: str = Field(default="", description="One-sentence description of what the user wants.")
    search_terms: list[str] = Field(default_factory=list, description="Exact or semantic search terms.")
    symbols: list[str] = Field(default_factory=list, description="Function/class/method names to prioritize.")
    files: list[str] = Field(default_factory=list, description="File names or paths to prioritize.")
    needs_tests: bool = False
    needs_callers: bool = False
    retrieval_mode: RetrievalMode = "hybrid"
    keyword_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    symbol_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    top_k: int = Field(default=10, ge=1, le=30)

    @classmethod
    def fallback(cls, query: str, top_k: int = 10) -> "QueryPlan":
        lower = query.lower()
        query_type: QueryType = "explanation"
        if any(w in lower for w in ["error", "bug", "fail", "failing", "exception", "traceback", "debug"]):
            query_type = "debugging"
        elif any(w in lower for w in ["where", "defined", "find", "located"]):
            query_type = "lookup"
        elif any(w in lower for w in ["test", "tests", "coverage"]):
            query_type = "test_question"
        elif any(w in lower for w in ["change", "modify", "add", "implement", "feature"]):
            query_type = "change_guidance"
        elif any(w in lower for w in ["overview", "architecture", "structure", "repo"]):
            query_type = "repo_overview"

        return cls(
            query_type=query_type,
            intent=query,
            search_terms=[query],
            symbols=[],
            files=[],
            needs_tests=query_type in {"debugging", "test_question", "change_guidance"},
            needs_callers=query_type in {"debugging", "change_guidance"},
            retrieval_mode="hybrid",
            top_k=top_k,
        )


class RetrievalHit(BaseModel):
    chunk: CodeChunk
    score: float
    reasons: list[str] = Field(default_factory=list)
    keyword_score: float = 0.0
    symbol_score: float = 0.0
    vector_score: float = 0.0
    relation_score: float = 0.0


class IncludedSnippet(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None = None
    mode: Literal["full", "trimmed", "metadata_only"]
    reason: str


class BuiltContext(BaseModel):
    context_text: str
    included: list[IncludedSnippet]
    omitted: list[str]
    used_chars: int
    max_chars: int


class AnswerEvidence(BaseModel):
    file_path: str
    line_range: str
    reason: str


class StructuredAnswer(BaseModel):
    """Structured final LLM output.

    The UI renders this object. This keeps the answer grounded and makes the
    model separate direct answer, evidence, limitations, and next steps.
    """

    answer: str
    evidence: list[AnswerEvidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    suggested_next_steps: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"


class IntermediateStep(BaseModel):
    name: str
    status: Literal["ok", "warning", "error"] = "ok"
    detail: str


class ChatResult(BaseModel):
    repo_id: int
    query: str
    plan: QueryPlan
    planner_context: str = ""
    answer: StructuredAnswer
    intermediate_steps: list[IntermediateStep]
    context: BuiltContext
    hits: list[RetrievalHit]


class RepoRecord(BaseModel):
    id: int
    name: str
    root_path: str
    created_at: str
    updated_at: str
    file_count: int = 0
    chunk_count: int = 0
    symbol_count: int = 0


class IngestRequest(BaseModel):
    repo_path: str
    reingest: bool = False


class IngestResponse(BaseModel):
    repo: RepoRecord
    indexed_files: int
    indexed_chunks: int
    indexed_symbols: int
    steps: list[IntermediateStep]


class ChatRequest(BaseModel):
    repo_id: int
    query: str
    max_context_chars: int = Field(default=45000, ge=4000, le=200000)
    top_k: int = Field(default=10, ge=1, le=30)
