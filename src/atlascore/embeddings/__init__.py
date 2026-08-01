"""Embedding clients for vector memory."""

try:
    from ._gemini import GeminiEmbeddingClient
except ImportError:
    GeminiEmbeddingClient = None  # type: ignore[assignment, misc]

__all__ = ["GeminiEmbeddingClient"]
