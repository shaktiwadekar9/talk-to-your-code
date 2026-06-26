from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .config import Settings
from .ollama_client import OllamaClient
from .parser import MultiLanguageParser
from .scanner import RepoScanner
from .schemas import CodeChunk, IntermediateStep
from .storage import SQLiteStore
from .graph_summary import generate_graph_repo_summary, load_graph_repo_summary

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text):
        lower = raw.lower()
        tokens.append(lower)
        if "_" in lower:
            tokens.extend(part for part in lower.split("_") if part)
        split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", raw).lower().split()
        if len(split) > 1:
            tokens.extend(split)
    return tokens


class RepoIndexer:
    """Builds the SQLite-backed index for a repo."""

    def __init__(self, settings: Settings, store: SQLiteStore, ollama: OllamaClient):
        self.settings = settings
        self.store = store
        self.ollama = ollama
        self.scanner = RepoScanner(settings)
        self.parser = MultiLanguageParser()

    def ingest(self, repo_path: str | Path, reingest: bool = False) -> tuple[int, list[IntermediateStep]]:
        """Ingests a repository by scanning files, parsing code, generating embeddings, and saving everything to SQLite.
        
        Args:
            repo_path: The file system path (local) to the repository to ingest.
            reingest: If True, forces reingestion and refreshes the index even if the repo is already indexed.

        Returns:
            A tuple containing the repo ID and a list of IntermediateStep objects describing the steps taken during ingestion.
        """
        steps: list[IntermediateStep] = []
        repo = self.store.get_or_create_repo(repo_path)
        if repo.chunk_count > 0 and not reingest:
            steps.append(IntermediateStep(name="Repo already indexed", detail="Use reingest=True to refresh this repo."))
            return repo.id, steps

        self.ollama.health_check()
        steps.append(IntermediateStep(name="Ollama", detail="Connected to local Ollama server."))

        files = self.scanner.scan(repo_path)
        steps.append(IntermediateStep(name="Scan", detail=f"Found {len(files)} supported source/text files."))

        chunks: list[CodeChunk] = []
        symbols = []
        for file in files:
            symbols.extend(self.parser.symbols_for_file(file))
            chunks.extend(self.parser.chunks_for_file(file))
        steps.append(IntermediateStep(name="Parse/chunk", detail=f"Extracted {len(symbols)} symbols and {len(chunks)} chunks."))

        embedding_inputs = [chunk.searchable_text() for chunk in chunks]
        embeddings = self.ollama.embed_texts(embedding_inputs)
        steps.append(IntermediateStep(name="Embeddings", detail=f"Created {len(embeddings)} local embeddings with {self.settings.embedding_model}."))

        self.store.save_index(repo.id, files, chunks, symbols, embeddings)
        steps.append(
            IntermediateStep(
                name="SQLite",
                detail=f"Saved repo snapshot, chunks, symbols, and embeddings to {self.store.db_path}.",
            )
        )

        print(f"Value of self.settings.enable_graph_summary: {self.settings.enable_graph_summary}")
        if self.settings.enable_graph_summary:
            try:
                graph_summary = generate_graph_repo_summary(self.settings, repo_path)
                steps.append(
                    IntermediateStep(
                        name="Graph repo summary",
                        detail=f"Generated graph-based planner summary with {len(graph_summary)} characters.",
                    )
                )
            except Exception as exc:
                steps.append(
                    IntermediateStep(
                        name="Graph repo summary",
                        status="warning",
                        detail=f"Skipped graph summary generation: {exc}",
                    )
                )

        return repo.id, steps


class InMemoryRepoIndex:
    """In-memory retrieval structures rebuilt from SQLite.

    SQLite is the source of truth. For fast query-time retrieval, we rebuild BM25
    term counts, symbol maps, and embedding matrix in memory for the selected repo.
    """

    def __init__(self, store: SQLiteStore, repo_id: int):
        self.store = store
        self.repo_id = repo_id
        self.repo = store.get_repo(repo_id)
        self.chunks, self.embeddings = store.load_chunks_and_embeddings(repo_id)
        self.symbols = store.load_symbols(repo_id)
        self.file_paths = store.load_repo_file_paths(repo_id)
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}

        if self.repo is not None:
            self.graph_repo_summary = load_graph_repo_summary(
                store.settings,
                self.repo.root_path,
            )
        else:
            self.graph_repo_summary = ""

        self._doc_tokens: list[list[str]] = []
        self._doc_counts: list[Counter[str]] = []
        self._idf: dict[str, float] = {}
        self._avg_len = 1.0
        self._symbol_to_ids: dict[str, list[str]] = defaultdict(list)
        self._build_keyword_and_symbol_indexes()

    def _build_keyword_and_symbol_indexes(self) -> None:
        """Precomputes token counts for BM25 and builds symbol-to-chunk_id maps for fast retrieval.   
        This is done once at initialization to optimize query-time performance for keyword and symbol searches.
        """
        document_frequency: Counter[str] = Counter()
        total_len = 0
        for chunk in self.chunks:
            tokens = tokenize(chunk.searchable_text())
            counts = Counter(tokens)
            self._doc_tokens.append(tokens)
            self._doc_counts.append(counts)
            document_frequency.update(set(tokens))
            total_len += len(tokens)

            if chunk.symbol_name:
                self._symbol_to_ids[chunk.symbol_name.lower()].append(chunk.chunk_id)
                for part in tokenize(chunk.symbol_name):
                    self._symbol_to_ids[part].append(chunk.chunk_id)

        n_docs = max(1, len(self.chunks))
        self._avg_len = max(1.0, total_len / n_docs)
        self._idf = {
            term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def keyword_search(self, query: str, top_k: int) -> dict[str, float]:
        """Implements a BM25-like keyword search over the code chunks. 
        It tokenizes the query, computes term frequencies, and scores each chunk based on term overlap and inverse document frequency.
        
        Args:
            query: The natural language query string to search for.
            top_k: The number of top results to return based on relevance scores.
        Returns:
            A dictionary mapping chunk_id to relevance score for the top_k most relevant chunks.
        """
        query_terms = tokenize(query)
        if not query_terms:
            return {}
        scores: dict[str, float] = {}
        k1 = 1.5
        b = 0.75
        for i, chunk in enumerate(self.chunks):
            score = 0.0
            doc_len = max(1, len(self._doc_tokens[i]))
            counts = self._doc_counts[i]
            for term in query_terms:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = tf + k1 * (1 - b + b * doc_len / self._avg_len)
                score += idf * (tf * (k1 + 1)) / denom
            if score > 0:
                scores[chunk.chunk_id] = score
        return self._top(scores, top_k)

    def symbol_search(self, symbols: list[str], top_k: int) -> dict[str, float]:
        """Finds chunks that match the given symbol names, prioritizing exact matches and then token matches.
        
        Args:
            symbols: A list of symbol names (e.g., function or class names) to search for in the code chunks.
            top_k: The number of top results to return based on relevance scores.
        
        Returns:
            A dictionary mapping chunk_id to relevance score for the top_k most relevant chunks based on symbol matches.
        """
        scores: dict[str, float] = {}
        for symbol in symbols:
            for term in tokenize(symbol):
                for chunk_id in self._symbol_to_ids.get(term, []):
                    scores[chunk_id] = max(scores.get(chunk_id, 0.0), 1.0)
            exact = symbol.lower().strip()
            for chunk_id in self._symbol_to_ids.get(exact, []):
                scores[chunk_id] = max(scores.get(chunk_id, 0.0), 2.0)
        return self._top(scores, top_k)

    def file_search(self, files: list[str], top_k: int) -> dict[str, float]:
        """Finds chunks that are located in files matching the given file names or paths, prioritizing exact matches.

        Args:
            files: A list of file names or paths to search for in the code chunks' file paths.
            top_k: The number of top results to return based on relevance scores.
        Returns:
            A dictionary mapping chunk_id to relevance score for the top_k most relevant chunks based on file path matches.
        """
        scores: dict[str, float] = {}
        wanted = [f.lower() for f in files if f.strip()]
        if not wanted:
            return scores
        for chunk in self.chunks:
            path = chunk.file_path.lower()
            for item in wanted:
                if item in path:
                    scores[chunk.chunk_id] = max(scores.get(chunk.chunk_id, 0.0), 1.5)
        return self._top(scores, top_k)

    def reference_search(self, names: list[str], top_k: int) -> dict[str, float]:
        """Find chunks that reference symbols, useful for callers/usages.

        This is not a full call graph. It is a lightweight reference search that
        works across all supported languages and keeps the implementation small.

        Args:
            names: A list of symbol names to find references to in the code chunks.
            top_k: The number of top results to return based on relevance scores.

        Returns:
            A dictionary mapping chunk_id to relevance score for the top_k most relevant chunks based on referencing the given symbol names.
        """

        terms = [t for name in names for t in tokenize(name)]
        if not terms:
            return {}
        scores: dict[str, float] = {}
        for chunk in self.chunks:
            text = chunk.text.lower()
            score = 0.0
            for term in terms:
                if term in text:
                    score += 0.5
            if score > 0:
                scores[chunk.chunk_id] = score
        return self._top(scores, top_k)

    def test_search(self, query: str, top_k: int) -> dict[str, float]:
        scores: dict[str, float] = {}
        query_terms = set(tokenize(query))
        for chunk in self.chunks:
            path = chunk.file_path.lower()
            is_test = "test" in path or "spec" in path
            if not is_test:
                continue
            overlap = len(query_terms.intersection(tokenize(chunk.searchable_text())))
            scores[chunk.chunk_id] = 1.0 + overlap * 0.1
        return self._top(scores, top_k)

    def vector_search(self, query_embedding: np.ndarray, top_k: int) -> dict[str, float]:
        """Finds chunks with embeddings most similar to the query embedding using cosine similarity (dot product for normalized vectors).
        
        Args:
            query_embedding: A 1D numpy array representing the embedding of the query text.
            top_k: The number of top results to return based on cosine similarity scores.

        Returns:
            A dictionary mapping chunk_id to cosine similarity score for the top_k most similar chunks based on embedding similarity.
        """
        if self.embeddings.size == 0 or query_embedding.size == 0:
            return {}
        vec = query_embedding.astype(np.float32)
        if vec.ndim == 2:
            vec = vec[0]
        scores_arr = self.embeddings @ vec
        order = np.argsort(-scores_arr)[:top_k]
        return {
            self.chunks[int(i)].chunk_id: float(scores_arr[int(i)])
            for i in order
            if float(scores_arr[int(i)]) > 0
        }

    def repo_map(self, max_files: int = 200) -> str:
        shown = self.file_paths[:max_files]
        suffix = "" if len(self.file_paths) <= max_files else f"\n... {len(self.file_paths) - max_files} more files"
        return "\n".join(shown) + suffix
    
    def planner_context(self, max_files: int = 220, max_symbols: int = 350) -> str:
        """
        Compact repo context for the query-planner LLM call.

        Prefer graph-based repo summary when available.
        Fall back to repo map + symbols if graph summary was not generated.
        """
        if self.graph_repo_summary.strip():
            return "\n".join(
                [
                    "Graph-based repository summary for query planning:",
                    self.graph_repo_summary.strip(),
                    "",
                    "Use this summary to choose likely files, symbols, search terms, and retrieval mode.",
                    "Do not answer the user from this summary alone. Full code snippets will be retrieved later.",
                ]
            )

        lines: list[str] = [
            "Repository file map:",
            self.repo_map(max_files=max_files),
            "",
            "Known symbols:",
        ]

        shown_symbols = self.symbols[:max_symbols]
        if not shown_symbols:
            lines.append("")
        else:
            for symbol in shown_symbols:
                lines.append(
                    f"- {symbol.kind} {symbol.name} "
                    f"in {symbol.file_path}:{symbol.start_line}-{symbol.end_line}"
                )

        remaining = len(self.symbols) - len(shown_symbols)
        if remaining > 0:
            lines.append(f"... {remaining} more symbols")

        return "\n".join(lines)

    @staticmethod
    def _top(scores: dict[str, float], top_k: int) -> dict[str, float]:
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k])
