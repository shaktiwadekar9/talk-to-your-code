from __future__ import annotations

import json
from typing import TypeVar

import numpy as np
import requests
from pydantic import BaseModel, ValidationError

from .config import Settings
from .schemas import QueryPlan, StructuredAnswer

T = TypeVar("T", bound=BaseModel)


class OllamaClient:
    """Small Ollama REST client.

    All LLM outputs in this app use Ollama structured generation by passing a
    JSON schema to the `format` field. This is used for both query planning and
    final answers so the UI receives predictable objects instead of free text.

    Attributes:
        settings: Global settings and config, including model names and Ollama URL.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.ollama_base_url.rstrip("/")

    def health_check(self) -> None:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Could not connect to Ollama. Start it with `ollama serve` or open the Ollama app."
            ) from exc

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Create normalized embeddings for retrieval.

        Uses `/api/embed` and stores vectors in SQLite. Normalization makes
        vector search a fast dot-product cosine similarity calculation.

        Args:
            texts: List of input strings to embed.

        Returns:
            2D numpy array of shape (len(texts), embedding_dim) with normalized float32 vectors.
        """

        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        vectors: list[list[float]] = []
        batch_size = max(1, self.settings.embedding_batch_size)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            payload = {"model": self.settings.embedding_model, "input": batch}
            response = requests.post(f"{self.base_url}/api/embed", json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            if "embeddings" in data:
                vectors.extend(data["embeddings"])
            elif "embedding" in data:
                vectors.append(data["embedding"])
            else:
                raise RuntimeError(f"Unexpected Ollama embedding response: {data.keys()}")

        arr = np.asarray(vectors, dtype=np.float32)
        return self._normalize(arr)

    def plan_query(self, query: str, top_k: int, repo_context: str = "") -> QueryPlan:
        """Ask Ollama to create a structured retrieval plan based on the user query and repo context.
        
        Args:
            query: The user's natural language question or request.
            top_k: Number of top retrieval hits to consider, which can influence how many search terms Ollama includes in the plan.
            repo_context: A compact textual representation of the repository's structure, files, and symbols to guide Ollama's planning.

        Returns:
            A QueryPlan object containing the structured retrieval plan, including search terms, symbols, files, and retrieval weights.
        """

        system = (
            "You are a codebase retrieval planner. Return only JSON matching the schema. "
            "Do not answer the user. Create a retrieval plan for finding code snippets. "
            "Use the provided repository map and symbol list to avoid inventing files or symbols."
        )
        user = f"""
Repository context available to the planner:
{repo_context or "<repo context unavailable>"}

User question:
{query}

Choose one query_type:
- repo_overview: repo structure or architecture overview
- lookup: find exact files, functions, classes, routes, variables
- explanation: explain how code works
- debugging: bugs, errors, stack traces, failing behavior
- change_guidance: which files to change for a feature
- test_question: tests and coverage questions
- unknown

Return useful search_terms, possible symbols, possible files, and whether callers/tests are needed.
Prefer file paths and symbols that appear in the repository context.
If the user asks for repo structure or architecture, use retrieval_mode=repo_map_first.
Set retrieval weights between 0 and 1. Use top_k={top_k}.
"""
        try:
            return self.chat_structured(
                model=self.settings.planner_model,
                system=system,
                user=user,
                schema_model=QueryPlan,
                temperature=0.0,
            )
        except Exception:
            # Local models can occasionally fail even with structured outputs.
            # Fallback keeps the app usable while still marking the fallback in UI.
            return QueryPlan.fallback(query, top_k=top_k)

    def answer_with_context(self, query: str, plan: QueryPlan, context: str) -> StructuredAnswer:
        system = (
            "You are a local codebase assistant. Return only JSON matching the schema. "
            "Answer strictly from the provided repository context. If the context is insufficient, say so. "
            "Always cite evidence using file_path and line_range fields. Do not invent files."
        )
        user = f"""
User question:
{query}

Query plan:
{plan.model_dump_json(indent=2)}

Repository context:
{context}

Return a concise grounded answer. Include evidence for every important claim.
"""
        return self.chat_structured(
            model=self.settings.chat_model,
            system=system,
            user=user,
            schema_model=StructuredAnswer,
            temperature=0.1,
        )

    def chat_structured(
        self,
        model: str,
        system: str,
        user: str,
        schema_model: type[T],
        temperature: float = 0.0,
    ) -> T:
        """Core structured generation method used for both planning and answering.
        
        Args:
            model: Ollama model name to use for this chat.
            system: System prompt describing the assistant's role and instructions.
            user: User prompt containing the question and any relevant context.
            schema_model: A Pydantic model class that defines the expected JSON schema of the response.
            temperature: Sampling temperature for response variability.
            
        Returns:
            An instance of schema_model populated with the data from Ollama's response.
        """

        schema = schema_model.model_json_schema()
        payload = {
            "model": model,
            "stream": False,
            "format": schema,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        user
                        + "\n\nYou must respond with valid JSON that matches this JSON schema:\n"
                        + json.dumps(schema, indent=2)
                    ),
                },
            ],
        }
        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
        if not response.ok:
            print(response.text)
            response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "")
        try:
            return schema_model.model_validate_json(content)
        except ValidationError:
            parsed = json.loads(content)
            return schema_model.model_validate(parsed)
        
    def count_tokens(self, model: str, text: str) -> int:
        """Count real input tokens using Ollama's model tokenizer/eval path."""
        if not text.strip():
            return 0

        payload = {
            "model": model,
            "prompt": text,
            "stream": False,
            "options": {
                "num_predict": 0,
                "temperature": 0.0,
            },
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=300,
        )

        # Some Ollama versions may not like num_predict=0.
        # Fallback still gives real prompt_eval_count, but generates 1 token.
        if not response.ok:
            payload["options"]["num_predict"] = 1
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300,
            )

        response.raise_for_status()
        data = response.json()
        return int(data.get("prompt_eval_count", 0))

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr.astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).astype(np.float32)
