from __future__ import annotations
from time import perf_counter

from .config import Settings
from .context_builder import ContextBuilder
from .indexing import InMemoryRepoIndex
from .ollama_client import OllamaClient
from .retriever import HybridRetriever
from .schemas import ChatResult, IntermediateStep, TimingStep
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

        # Timing and debugging helpers
        total_start = perf_counter()
        last_time = total_start
        timings: list[TimingStep] = []

        def mark_timing(name: str) -> None:
            nonlocal last_time
            now = perf_counter()
            timings.append(
                TimingStep(
                    name=name,
                    ms=round((now - last_time) * 1000, 2),
                )
            )
            last_time = now

        steps: list[IntermediateStep] = []
        repo = self.store.get_repo(repo_id)
        mark_timing("Load repo metadata")
        if repo is None:
            raise ValueError(f"Unknown repo_id: {repo_id}")
        if repo.chunk_count == 0:
            raise ValueError("Repo is known but not indexed yet. Reingest it first.")

        index = InMemoryRepoIndex(self.store, repo_id)
        mark_timing("Load index (InMemoryRepoIndex)")
        steps.append(IntermediateStep(name="Load index", detail=f"Loaded {len(index.chunks)} chunks from SQLite for {repo.name}."))

        planner_context = index.planner_context()
        mark_timing("Build planner context")
        if self.settings.use_model_for_token_count:
            planner_context_tokens = self.ollama.count_tokens(
                self.settings.planner_model,
                planner_context,
            )
            mark_timing("Count planner context tokens")
        else:
            planner_context_tokens = len(planner_context) // 4
        plan = self.ollama.plan_query(query, top_k=top_k, repo_context=planner_context)
        mark_timing("Plan query with Ollama")
        plan.top_k = top_k
        steps.append(
            IntermediateStep(
                name="LLM query plan",
                detail=(
                    f"Planned as {plan.query_type} with mode {plan.retrieval_mode}. "
                    f"Planner saw repo map, file paths, and parsed symbols."
                    f"Planner context chars: {len(planner_context)} and tokens: ~{planner_context_tokens}."
                ),
            )
        )

        retriever = HybridRetriever(index, self.ollama)
        hits = retriever.retrieve(query, plan)
        mark_timing("Hybrid retrieval")
        steps.append(IntermediateStep(name="Hybrid retrieval", detail=f"Retrieved {len(hits)} ranked snippets."))

        builder = ContextBuilder(index, max_context_chars=max_context_chars)
        context = builder.build(query, plan, hits)
        mark_timing("Context build")
        if self.settings.use_model_for_token_count:
            context.used_tokens = self.ollama.count_tokens(
                self.settings.chat_model,
                context.context_text,
            )
            mark_timing("Count context tokens")
        else:
            context.used_tokens = context.used_chars // 4
        steps.append(
            IntermediateStep(
                name="Context build",
                detail=(
                    f"Used {context.used_chars}/{context.max_chars} chars "
                    f"(~{context.used_tokens} tokens). "
                    f"Included {len(context.included)} snippets, omitted {len(context.omitted)}."
                ),
            )
        )

        answer = self.ollama.answer_with_context(query, plan, context.context_text)
        mark_timing("Answer with context")
        steps.append(IntermediateStep(name="Structured answer", detail="Generated schema-valid answer with evidence fields."))

        return ChatResult(
            repo_id=repo_id,
            query=query,
            plan=plan,
            planner_context=planner_context,
            planner_context_tokens=planner_context_tokens,
            answer=answer,
            intermediate_steps=steps,
            context=context,
            hits=hits,
            timings=timings,
            total_ms=round((perf_counter() - total_start) * 1000, 2)
        )
