"""Gemini embedding client for Atlas vector memory."""

from __future__ import annotations

import os
from typing import List, Optional, Union


class GeminiEmbeddingClient:
    """Wraps the Google Gen AI SDK to produce text embeddings.

    This client is intentionally lightweight. It reads ``GEMINI_API_KEY`` or
    ``GOOGLE_API_KEY`` from the environment when no explicit key is supplied,
    matching the behaviour of ``google.genai.Client``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-embedding-001",
        task_type: str = "RETRIEVAL_DOCUMENT",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = model
        self.task_type = task_type
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Return one embedding vector per input string."""
        client = self._get_client()
        from google.genai import types

        if isinstance(texts, str):
            texts = [texts]
        result = client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=self.task_type),
        )
        vectors: List[List[float]] = []
        embeddings = getattr(result, "embeddings", None) or []
        for emb in embeddings:
            values = getattr(emb, "values", None)
            if values:
                vectors.append(values)
        return vectors

    def __call__(self, text: str) -> List[float]:
        """Single text convenience wrapper for ``QdrantMemory`` compatibility."""
        vectors = self.embed(text)
        if not vectors:
            raise RuntimeError(f"No embedding returned from {self.model}")
        return vectors[0]
