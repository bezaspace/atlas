import pytest

from atlascore import FileMemory, ListMemory, MemoryContent, MemoryQueryResult


@pytest.fixture
def content_items():
    return [
        MemoryContent(content="The user likes Python."),
        MemoryContent(content="The user prefers async code."),
        MemoryContent(content="The user is located in UTC."),
    ]


@pytest.mark.asyncio
async def test_list_memory_add_and_query(content_items, tmp_path):
    memory = ListMemory(max_memories=10)
    for item in content_items:
        await memory.add(item)

    result = await memory.query("Python")
    assert len(result.results) == 1
    assert "Python" in result.results[0].content

    context = await memory.get_context(max_items=2)
    assert len(context.results) == 2
    assert context.results[-1].content == content_items[-1].content


@pytest.mark.asyncio
async def test_list_memory_max(content_items):
    memory = ListMemory(max_memories=2)
    for item in content_items:
        await memory.add(item)

    result = await memory.get_context(max_items=10)
    assert len(result.results) == 2
    assert result.results[0].content == content_items[1].content


@pytest.mark.asyncio
async def test_file_memory_persistence(content_items, tmp_path):
    file_path = str(tmp_path / "memory.json")
    memory = FileMemory(file_path=file_path, max_memories=10)
    for item in content_items:
        await memory.add(item)

    memory2 = FileMemory(file_path=file_path, max_memories=10)
    result = await memory2.query("async")
    assert len(result.results) == 1
    assert "async" in result.results[0].content


@pytest.mark.asyncio
async def test_memory_query_result_empty():
    result = MemoryQueryResult()
    assert result.results == []
