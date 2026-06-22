from __future__ import annotations

from .indexing import InMemoryRepoIndex
from .schemas import BuiltContext, IncludedSnippet, QueryPlan, RetrievalHit


class ContextBuilder:
    """Builds a budget-aware prompt context.

    It does not abort when context is too large. For each hit it tries:
    full snippet -> trimmed snippet -> metadata-only fallback -> omitted list.
    This is safer for code because the model still sees file paths and line
    ranges even when full code cannot fit.

    Attributes:
        index: In-memory repo index for access to chunk metadata.
        max_context_chars: Maximum number of characters to include in the built context.
    """

    def __init__(self, index: InMemoryRepoIndex, max_context_chars: int):
        self.index = index
        self.max_context_chars = max_context_chars

    def build(self, query: str, plan: QueryPlan, hits: list[RetrievalHit]) -> BuiltContext:
        """Builds a context string that includes retrieved snippets and metadata, while respecting the character limit.
        
        Args:
            query: The original user query.
            plan: The structured retrieval plan that guided the retrieval of hits.
            hits: A ranked list of RetrievalHit objects that are candidates for inclusion in the context.

        Returns:
            A BuiltContext object containing the assembled context text, 
            lists of included and omitted snippets, and character usage statistics.
        """
        sections: list[str] = []
        included: list[IncludedSnippet] = []
        omitted: list[str] = []
        used = 0

        header = self._header(query, plan)
        repo_map = self._repo_map_block(plan)
        for block in [header, repo_map]:
            if not block:
                continue
            if used + len(block) <= self.max_context_chars:
                sections.append(block)
                used += len(block)

        for rank, hit in enumerate(hits, start=1):
            full = self._full_block(rank, hit)
            if used + len(full) <= self.max_context_chars:
                sections.append(full)
                used += len(full)
                included.append(self._included(hit, "full", ", ".join(hit.reasons)))
                continue

            trimmed = self._trimmed_block(rank, hit, remaining=self.max_context_chars - used)
            if trimmed and used + len(trimmed) <= self.max_context_chars:
                sections.append(trimmed)
                used += len(trimmed)
                included.append(self._included(hit, "trimmed", "trimmed due to context budget"))
                continue

            metadata = self._metadata_block(rank, hit)
            if used + len(metadata) <= self.max_context_chars:
                sections.append(metadata)
                used += len(metadata)
                included.append(self._included(hit, "metadata_only", "metadata only due to context budget"))
            else:
                omitted.append(f"{hit.chunk.file_path}:{hit.chunk.start_line}-{hit.chunk.end_line}")

        return BuiltContext(
            context_text="\n\n".join(sections),
            included=included,
            omitted=omitted,
            used_chars=used,
            max_chars=self.max_context_chars,
        )

    def _header(self, query: str, plan: QueryPlan) -> str:
        return f"""
Repository question:
{query}

Retrieval intent:
{plan.intent}

Rules for answer:
- Use only the snippets below.
- Cite file paths and line ranges.
- If evidence is missing, say what is missing.
""".strip()

    def _repo_map_block(self, plan: QueryPlan) -> str:
        if plan.query_type != "repo_overview" and plan.retrieval_mode != "repo_map_first":
            return ""
        return "Repository file map:\n" + self.index.repo_map(max_files=180)

    def _full_block(self, rank: int, hit: RetrievalHit) -> str:
        c = hit.chunk
        symbol = c.symbol_name or "<no symbol>"
        reasons = ", ".join(hit.reasons)
        return f"""
[Snippet {rank}]
File: {c.file_path}
Lines: {c.start_line}-{c.end_line}
Language: {c.language}
Symbol: {symbol}
Retrieval reasons: {reasons}
Score: {hit.score:.3f}
Code:
```{c.language}
{c.text}
```
""".strip()

    def _trimmed_block(self, rank: int, hit: RetrievalHit, remaining: int) -> str:
        if remaining < 800:
            return ""
        c = hit.chunk
        budget = max(400, min(2500, remaining - 400))
        text = c.text
        if len(text) <= budget:
            trimmed = text
        else:
            head_budget = budget // 2
            tail_budget = budget - head_budget
            trimmed = text[:head_budget] + "\n# ... trimmed for context budget ...\n" + text[-tail_budget:]
        return f"""
[Snippet {rank} - trimmed]
File: {c.file_path}
Lines: {c.start_line}-{c.end_line}
Language: {c.language}
Symbol: {c.symbol_name or '<no symbol>'}
Code excerpt:
```{c.language}
{trimmed}
```
""".strip()

    def _metadata_block(self, rank: int, hit: RetrievalHit) -> str:
        c = hit.chunk
        return f"""
[Snippet {rank} - metadata only]
File: {c.file_path}
Lines: {c.start_line}-{c.end_line}
Language: {c.language}
Symbol: {c.symbol_name or '<no symbol>'}
Note: code omitted because context budget was too small.
""".strip()

    @staticmethod
    def _included(hit: RetrievalHit, mode: str, reason: str) -> IncludedSnippet:
        return IncludedSnippet(
            file_path=hit.chunk.file_path,
            start_line=hit.chunk.start_line,
            end_line=hit.chunk.end_line,
            symbol_name=hit.chunk.symbol_name,
            mode=mode,  # type: ignore[arg-type]
            reason=reason,
        )
