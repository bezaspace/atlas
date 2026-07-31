"""Qdrant-backed vector memory for atlascore."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ._base import BaseMemory, MemoryContent, MemoryQueryResult


def _has_qdrant() -> bool:
    try:
        import qdrant_client  # noqa: F401

        return True
    except ImportError:
        return False


def _has_sentence_transformers() -> bool:
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401

        return True
    except ImportError:
        return False


class QdrantMemory(BaseMemory):
    """Vector memory backed by Qdrant with sentence-transformer embeddings."""

    def __init__(
        self,
        collection_name: str = "atlas_memory",
        max_memories: int = 1000,
        path: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_model: Optional[str] = "all-MiniLM-L6-v2",
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        score_threshold: float = 0.7,
    ):
        super().__init__(max_memories)

        if not _has_qdrant():
            raise ImportError(
                "qdrant-client is required for QdrantMemory. "
                'Install with: pip install "atlascore[rag]"'
            )

        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.collection_name = collection_name
        self.score_threshold = score_threshold

        if path and path == ":memory:":
            path = None

        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(path=path)

        self._embedding_fn = embedding_fn
        self._embedding_model_name = embedding_model
        self._embedding_model = None

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self._vector_size(), distance=Distance.COSINE),
        )

    def _vector_size(self) -> int:
        if self._embedding_fn is not None:
            return len(self._embedding_fn("test"))

        if not _has_sentence_transformers():
            raise ImportError(
                "sentence-transformers is required for default QdrantMemory embeddings. "
                'Install with: pip install "atlascore[rag]" or provide embedding_fn.'
            )

        model = self._get_embedding_model()
        return model.get_sentence_embedding_dimension()

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    def _embed(self, text: str) -> List[float]:
        if self._embedding_fn is not None:
            return self._embedding_fn(text)

        model = self._get_embedding_model()
        return model.encode(text).tolist()

    async def add(self, content: MemoryContent) -> None:
        from qdrant_client.models import PointStruct

        vector = self._embed(content.content)
        point_id = str(uuid.uuid4())
        payload = {
            "content": content.content,
            "mime_type": content.mime_type,
            "metadata": content.metadata,
            "timestamp": content.timestamp.isoformat(),
        }
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        await self._enforce_limit()

    async def _enforce_limit(self) -> None:
        count = self.client.count(collection_name=self.collection_name).count
        if count <= self.max_memories:
            return

        all_points = self.client.scroll(collection_name=self.collection_name, limit=count)
        points = all_points[0] if all_points else []
        points.sort(key=lambda p: p.payload.get("timestamp", ""))
        to_remove = [p.id for p in points[: count - self.max_memories]]
        if to_remove:
            self.client.delete(collection_name=self.collection_name, points_selector=to_remove)

    async def query(self, query: str, limit: int = 10) -> MemoryQueryResult:
        vector = self._embed(query)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=limit,
            score_threshold=self.score_threshold,
        )

        memories: List[MemoryContent] = []
        for point in results:
            payload = point.payload or {}
            memories.append(
                MemoryContent(
                    content=payload.get("content", ""),
                    mime_type=payload.get("mime_type", "text/plain"),
                    metadata=payload.get("metadata", {}),
                    timestamp=datetime.fromisoformat(payload.get("timestamp", datetime.now().isoformat())),
                )
            )
        return MemoryQueryResult(results=memories)

    async def get_context(self, max_items: int = 10) -> MemoryQueryResult:
        count = self.client.count(collection_name=self.collection_name).count
        if count == 0:
            return MemoryQueryResult(results=[])

        all_points = self.client.scroll(collection_name=self.collection_name, limit=count)
        points = all_points[0] if all_points else []
        points.sort(key=lambda p: p.payload.get("timestamp", ""), reverse=True)

        memories: List[MemoryContent] = []
        for point in points[:max_items]:
            payload = point.payload or {}
            memories.append(
                MemoryContent(
                    content=payload.get("content", ""),
                    mime_type=payload.get("mime_type", "text/plain"),
                    metadata=payload.get("metadata", {}),
                    timestamp=datetime.fromisoformat(payload.get("timestamp", datetime.now().isoformat())),
                )
            )
        return MemoryQueryResult(results=memories)

    async def clear(self) -> None:
        self.client.delete_collection(collection_name=self.collection_name)
        from qdrant_client.models import Distance, VectorParams

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self._vector_size(), distance=Distance.COSINE),
        )

    async def get_stats(self) -> Dict[str, Any]:
        base = await super().get_stats()
        count = self.client.count(collection_name=self.collection_name).count
        return {
            **base,
            "current_memories": count,
            "collection_name": self.collection_name,
            "is_persistent": self.client._client is not None,
        }
