from __future__ import annotations

from collections import defaultdict

from .indexing import InMemoryRepoIndex
from .ollama_client import OllamaClient
from .schemas import QueryPlan, RetrievalHit


class HybridRetriever:
    """Plan-aware hybrid retriever.

    It combines keyword, symbol, file-path, semantic-vector, reference, and test
    search. The LLM planner controls the weights and flags.

    Attributes:
        index: In-memory repo index with search capabilities.
        ollama: Ollama client for query embedding.
    """

    def __init__(self, index: InMemoryRepoIndex, ollama: OllamaClient):
        self.index = index
        self.ollama = ollama

    def retrieve(self, query: str, plan: QueryPlan) -> list[RetrievalHit]:
        """Retrieves and ranks code chunks based on the query and structured plan.
        
        Args:
            query: The original user query.
            plan: Structured retrieval plan from the LLM, containing search terms, symbols, files, and weights.

        Returns:
            A ranked list of RetrievalHit objects, sorted by combined relevance score.
        """
        top_k = plan.top_k
        merged: dict[str, RetrievalHit] = {}

        search_text = "\n".join([query] + plan.search_terms + plan.symbols + plan.files)

        keyword_scores = self.index.keyword_search(search_text, top_k=top_k * 3)
        self._merge(merged, keyword_scores, "keyword", plan.keyword_weight)

        symbol_scores = self.index.symbol_search(plan.symbols + plan.search_terms, top_k=top_k * 3)
        self._merge(merged, symbol_scores, "symbol", plan.symbol_weight)

        file_scores = self.index.file_search(plan.files, top_k=top_k * 3)
        self._merge(merged, file_scores, "file_path", 0.6)

        query_embedding = self.ollama.embed_texts([search_text])
        vector_scores = self.index.vector_search(query_embedding, top_k=top_k * 3)
        self._merge(merged, vector_scores, "vector", plan.vector_weight)

        if plan.needs_callers:
            reference_scores = self.index.reference_search(plan.symbols + plan.search_terms, top_k=top_k * 2)
            self._merge(merged, reference_scores, "caller_or_reference", 0.25)

        if plan.needs_tests:
            test_scores = self.index.test_search(search_text, top_k=top_k * 2)
            self._merge(merged, test_scores, "related_test", 0.3)

        hits = sorted(merged.values(), key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _merge(self, merged: dict[str, RetrievalHit], scores: dict[str, float], reason: str, weight: float) -> None:
        """Merges new scores into the combined retrieval hits with normalization and weighting.

        Args:
            merged: The current combined retrieval hits being built.
            scores: New scores to merge, keyed by chunk_id.
            reason: The reason/type of this score (e.g., "keyword", "symbol", "vector").
            weight: The weight to apply to this type of score when merging.

        Returns:
            None. The merged dict is modified in place.
        """
        if not scores:
            return
        max_score = max(scores.values()) or 1.0
        for chunk_id, raw_score in scores.items():
            chunk = self.index.chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            normalized = raw_score / max_score
            weighted = normalized * weight
            if chunk_id not in merged:
                merged[chunk_id] = RetrievalHit(chunk=chunk, score=0.0)
            hit = merged[chunk_id]
            hit.score += weighted
            hit.reasons.append(reason)
            if reason == "keyword":
                hit.keyword_score += normalized
            elif reason == "symbol":
                hit.symbol_score += normalized
            elif reason == "vector":
                hit.vector_score += normalized
            else:
                hit.relation_score += normalized
