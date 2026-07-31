"""Tests for Qdrant-backed vector memory."""

import os
import uuid

import pytest

from atlascore import MemoryContent, MemoryQueryResult

VECTOR_SIZE = 8


def _embedding_fn(text: str) -> list[float]:
    """Deterministic, fixed-size embedding for fast tests."""
    vec = [0.0] * VECTOR_SIZE
    if not text:
        return vec
    for i, char in enumerate(text[:VECTOR_SIZE]):
        vec[i] = (ord(char) % 256) / 255.0
    return vec


def _content(text: str) -> MemoryContent:
    return MemoryContent(content=text, metadata={"source": "test"})


@pytest.fixture
def in_memory_memory():
    pytest.importorskip("qdrant_client")
    from atlascore.memory import QdrantMemory

    return QdrantMemory(
        collection_name=f"test_{uuid.uuid4().hex}",
        path=":memory:",
        max_memories=10,
        embedding_fn=_embedding_fn,
        score_threshold=0.0,
    )


@pytest.mark.asyncio
async def test_qdrant_memory_add_and_query(in_memory_memory):
    await in_memory_memory.add(_content("Atlas uses Qdrant for vector memory."))
    await in_memory_memory.add(_content("Python is great for async agents."))

    result = await in_memory_memory.query("Atlas")
    assert isinstance(result, MemoryQueryResult)
    assert len(result.results) == 2
    contents = {r.content for r in result.results}
    assert "Atlas uses Qdrant for vector memory." in contents
    assert "Python is great for async agents." in contents


@pytest.mark.asyncio
async def test_qdrant_memory_query_respects_limit(in_memory_memory):
    await in_memory_memory.add(_content("first"))
    await in_memory_memory.add(_content("second"))

    result = await in_memory_memory.query("first", limit=1)
    assert len(result.results) == 1


@pytest.mark.asyncio
async def test_qdrant_memory_get_context(in_memory_memory):
    await in_memory_memory.add(_content("oldest"))
    await in_memory_memory.add(_content("newest"))

    context = await in_memory_memory.get_context(max_items=1)
    assert len(context.results) == 1
    assert context.results[0].content == "newest"


@pytest.mark.asyncio
async def test_qdrant_memory_enforces_max_memories(tmp_path):
    pytest.importorskip("qdrant_client")
    from atlascore.memory import QdrantMemory

    memory = QdrantMemory(
        collection_name=f"test_{uuid.uuid4().hex}",
        path=str(tmp_path / "qdrant"),
        max_memories=2,
        embedding_fn=_embedding_fn,
        score_threshold=0.0,
    )

    await memory.add(_content("one"))
    await memory.add(_content("two"))
    await memory.add(_content("three"))

    stats = await memory.get_stats()
    assert stats["current_memories"] == 2
    assert stats["is_persistent"] is True

    context = await memory.get_context(max_items=10)
    assert len(context.results) == 2
    assert context.results[0].content == "three"
    assert context.results[1].content == "two"


@pytest.mark.asyncio
async def test_qdrant_memory_clear(in_memory_memory):
    await in_memory_memory.add(_content("data"))
    assert (await in_memory_memory.get_stats())["current_memories"] == 1

    await in_memory_memory.clear()
    assert (await in_memory_memory.get_stats())["current_memories"] == 0
    result = await in_memory_memory.query("data")
    assert len(result.results) == 0


@pytest.mark.asyncio
async def test_qdrant_memory_stats_for_in_memory(in_memory_memory):
    stats = await in_memory_memory.get_stats()
    assert stats["implementation"] == "QdrantMemory"
    assert stats["is_persistent"] is False
    assert "collection_name" in stats


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("ATLAS_TEST_QDRANT_CLOUD") != "1",
    reason="Set ATLAS_TEST_QDRANT_CLOUD=1 and provide QDRANT_ENDPOINT/QDRANT_API_KEY",
)
async def test_qdrant_memory_cloud():
    from atlascore.memory import QdrantMemory

    url = (
        os.getenv("Qdrant_ENDPOINT")
        or os.getenv("QDRANT_ENDPOINT")
        or os.getenv("QDRANT_URL")
    )
    api_key = os.getenv("Qdrant_API_KEY") or os.getenv("QDRANT_API_KEY")
    assert url and api_key, "Qdrant cloud credentials not available"

    memory = QdrantMemory(
        collection_name=f"atlas_test_{uuid.uuid4().hex}",
        url=url,
        api_key=api_key,
        max_memories=10,
        embedding_fn=_embedding_fn,
        score_threshold=0.0,
    )

    await memory.add(_content("cloud qdrant memory"))
    result = await memory.query("cloud")
    assert len(result.results) == 1
    assert result.results[0].content == "cloud qdrant memory"

    await memory.clear()
