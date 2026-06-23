from __future__ import annotations

from .config import Settings
from .context_builder import ContextBuilder
from .indexing import InMemoryRepoIndex
from .ollama_client import OllamaClient
from .retriever import HybridRetriever
from .schemas import ChatResult, IntermediateStep
from .storage import SQLiteStore


class CodeChatOrchestrator:
    """End-to-end query flow.

    Flow:
    1. Load selected repo index from SQLite.
    2. Ask Ollama for a structured QueryPlan.
    3. Retrieve snippets using the plan.
    4. Build a context under the UI-selected context-length budget.
    5. Ask Ollama for a structured final answer.

    Attributes:
        settings: Global settings and config.
        store: SQLite-backed storage for repo metadata and chunks.
        ollama: Client for interacting with the local Ollama server.
    """

    def __init__(self, settings: Settings, store: SQLiteStore, ollama: OllamaClient):
        self.settings = settings
        self.store = store
        self.ollama = ollama

    def chat(self, repo_id: int, query: str, max_context_chars: int, top_k: int) -> ChatResult:
        """Executes the end-to-end code chat flow for a given query and repo.
        
        Args:
            repo_id: ID of the repo to query against.
            query: The user's natural language question or request.
            max_context_chars: Maximum number of characters to include in the LLM context.
            top_k: Number of top retrieval hits to consider for context building.
            
        Returns:
            A ChatResult containing the final answer, the structured plan, retrieved hits, and intermediate steps for debugging.
        """
        steps: list[IntermediateStep] = []
        repo = self.store.get_repo(repo_id)
        if repo is None:
            raise ValueError(f"Unknown repo_id: {repo_id}")
        if repo.chunk_count == 0:
            raise ValueError("Repo is known but not indexed yet. Reingest it first.")

        index = InMemoryRepoIndex(self.store, repo_id)
        steps.append(IntermediateStep(name="Load index", detail=f"Loaded {len(index.chunks)} chunks from SQLite for {repo.name}."))

        planner_context = index.planner_context()
        plan = self.ollama.plan_query(query, top_k=top_k, repo_context=planner_context)
        plan.top_k = top_k
        steps.append(
            IntermediateStep(
                name="LLM query plan",
                detail=(
                    f"Planned as {plan.query_type} with mode {plan.retrieval_mode}. "
                    f"Planner saw repo map, file paths, and parsed symbols."
                ),
            )
        )

        retriever = HybridRetriever(index, self.ollama)
        hits = retriever.retrieve(query, plan)
        steps.append(IntermediateStep(name="Hybrid retrieval", detail=f"Retrieved {len(hits)} ranked snippets."))

        builder = ContextBuilder(index, max_context_chars=max_context_chars)
        context = builder.build(query, plan, hits)
        steps.append(
            IntermediateStep(
                name="Context build",
                detail=f"Used {context.used_chars}/{context.max_chars} chars. Included {len(context.included)} snippets, omitted {len(context.omitted)}.",
            )
        )

        answer = self.ollama.answer_with_context(query, plan, context.context_text)
        steps.append(IntermediateStep(name="Structured answer", detail="Generated schema-valid answer with evidence fields."))

        return ChatResult(
            repo_id=repo_id,
            query=query,
            plan=plan,
            planner_context=planner_context,
            answer=answer,
            intermediate_steps=steps,
            context=context,
            hits=hits,
        )
