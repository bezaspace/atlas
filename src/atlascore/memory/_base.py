"""Memory implementations for atlascore agents."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class MemoryContent(BaseModel):
    """A memory content item."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(..., description="The memory content")
    mime_type: str = Field(default="text/plain", description="MIME type of the content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional memory metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="When memory was stored")


class MemoryQueryResult(BaseModel):
    """Result of a memory query operation."""

    model_config = ConfigDict(frozen=True)

    results: List[MemoryContent] = Field(default_factory=list, description="Retrieved memories")


class BaseMemory(ABC):
    """Abstract base class for agent memory."""

    def __init__(self, max_memories: int = 1000):
        self.max_memories = max_memories

    @abstractmethod
    async def add(self, content: MemoryContent) -> None:
        """Store new content in memory."""
        pass

    @abstractmethod
    async def query(self, query: str, limit: int = 10) -> MemoryQueryResult:
        """Retrieve relevant memories based on query."""
        pass

    @abstractmethod
    async def get_context(self, max_items: int = 10) -> MemoryQueryResult:
        """Get recent/relevant context for LLM prompt."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Remove all stored memories."""
        pass

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "max_memories": self.max_memories,
            "implementation": self.__class__.__name__,
        }


class ListMemory(BaseMemory):
    """In-memory list storage with simple text search."""

    def __init__(self, max_memories: int = 1000):
        super().__init__(max_memories)
        self._memories: List[MemoryContent] = []

    async def add(self, content: MemoryContent) -> None:
        self._memories.append(content)
        if len(self._memories) > self.max_memories:
            self._memories = self._memories[-self.max_memories :]

    async def query(self, query: str, limit: int = 10) -> MemoryQueryResult:
        query_lower = query.lower()
        matching: List[MemoryContent] = []
        for memory in reversed(self._memories):
            if query_lower in memory.content.lower():
                matching.append(memory)
                if len(matching) >= limit:
                    break
        return MemoryQueryResult(results=matching)

    async def get_context(self, max_items: int = 10) -> MemoryQueryResult:
        recent = self._memories[-max_items:] if self._memories else []
        return MemoryQueryResult(results=recent)

    async def clear(self) -> None:
        self._memories.clear()

    async def get_stats(self) -> Dict[str, Any]:
        base = await super().get_stats()
        return {**base, "current_memories": len(self._memories), "is_persistent": False}


class FileMemory(BaseMemory):
    """File-based persistent storage with text search."""

    def __init__(self, file_path: str, max_memories: int = 1000):
        super().__init__(max_memories)
        self.file_path = file_path
        self._memories: List[MemoryContent] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.file_path):
            return
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._memories = [MemoryContent(**item) for item in data]
        except Exception:
            self._memories = []

    def _save(self) -> None:
        try:
            directory = os.path.dirname(self.file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump([m.model_dump() for m in self._memories], f, indent=2, default=str)
        except Exception:
            pass

    async def add(self, content: MemoryContent) -> None:
        self._memories.append(content)
        if len(self._memories) > self.max_memories:
            self._memories = self._memories[-self.max_memories :]
        self._save()

    async def query(self, query: str, limit: int = 10) -> MemoryQueryResult:
        query_lower = query.lower()
        matching: List[MemoryContent] = []
        for memory in reversed(self._memories):
            if query_lower in memory.content.lower():
                matching.append(memory)
                if len(matching) >= limit:
                    break
        return MemoryQueryResult(results=matching)

    async def get_context(self, max_items: int = 10) -> MemoryQueryResult:
        recent = self._memories[-max_items:] if self._memories else []
        return MemoryQueryResult(results=recent)

    async def clear(self) -> None:
        self._memories.clear()
        if os.path.exists(self.file_path):
            try:
                os.remove(self.file_path)
            except Exception:
                pass

    async def get_stats(self) -> Dict[str, Any]:
        base = await super().get_stats()
        return {
            **base,
            "current_memories": len(self._memories),
            "file_path": self.file_path,
            "is_persistent": True,
        }
